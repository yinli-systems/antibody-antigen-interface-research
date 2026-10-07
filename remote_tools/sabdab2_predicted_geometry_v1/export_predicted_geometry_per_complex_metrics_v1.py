#!/usr/bin/env python3
"""Re-evaluate one frozen development checkpoint and export per-complex AP."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score


PLAN_SHA256 = "86ef7dff66b4f636f929f28728cf300f360041577741c891059ff601f9c5f272"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--geometry-source", choices=("predicted", "bound"), required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--structure-cache-root", type=Path, required=True)
    parser.add_argument("--contact-cache-root", type=Path, required=True)
    parser.add_argument("--embedding-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--split-manifest-sha256", required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def _finite_ap(labels: np.ndarray, logits: np.ndarray, sigmoid: Any) -> float:
    if labels.ndim != 1 or logits.ndim != 1 or labels.shape != logits.shape:
        raise ValueError("AP labels/logits shape differs")
    if not np.isfinite(logits).all() or not np.all(np.isin(np.unique(labels), (0, 1))):
        raise ValueError("AP labels/logits content differs")
    probabilities = sigmoid(logits)
    if not np.isfinite(probabilities).all():
        raise ValueError("non-finite sigmoid probability")
    value = float(average_precision_score(labels, probabilities))
    if not math.isfinite(value):
        raise ValueError("non-finite AP")
    return value


def main() -> None:
    args = parse_args()
    if os.environ.get("SLURM_JOB_PARTITION") != "gpu_4090":
        raise RuntimeError("gpu_4090 is required")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if sha256_file(args.plan) != PLAN_SHA256:
        raise ValueError("frozen association plan SHA-256 differs")
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if (
        plan.get("status") != "FROZEN_BEFORE_PER_COMPLEX_REEVALUATION"
        or plan.get("old_sabdab2_test_loaded") is not False
        or plan.get("temporal_holdout_loaded") is not False
        or plan.get("selection_or_tuning") is not False
    ):
        raise ValueError("frozen association plan content differs")

    repository_root = args.repository_root.resolve()
    sys.path.insert(0, str(repository_root / "src"))
    sys.path.insert(0, str(repository_root / "scripts"))
    import train_framecontact as training  # noqa: PLC0415
    from interaction_first.sabdab2_contacts import (  # noqa: PLC0415
        Sabdab2ContactDataset,
    )
    from interaction_first.sabdab2_structure import (  # noqa: PLC0415
        Sabdab2StructureEmbeddingDataset,
    )

    run_root = args.run_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(output_root)
    result_path = run_root / "FRAMECONTACT_DEVELOPMENT_RESULT.json"
    checkpoint_path = run_root / "FRAMECONTACT_SELECTED.pt"
    receipt_path = run_root / "RUN_RECEIPT.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    authority = result["authority"]
    if (
        receipt.get("result_sha256") != sha256_file(result_path)
        or receipt.get("selected_checkpoint_sha256") != sha256_file(checkpoint_path)
        or authority.get("split_manifest_sha256") != args.split_manifest_sha256
        or authority.get("test_loaded") is not False
        or authority.get("external_test_loaded") is not False
        or receipt.get("test_loaded") is not False
        or receipt.get("external_test_loaded") is not False
        or sha256_file(args.split_manifest) != args.split_manifest_sha256
    ):
        raise ValueError("source run authority differs")

    configuration = result["configuration"]
    seed = int(configuration["seed"])
    if seed not in plan["seeds"]:
        raise ValueError("source seed differs from frozen plan")
    training._set_seed(seed)
    base = Sabdab2StructureEmbeddingDataset(
        args.structure_cache_root,
        args.embedding_root,
        None,
        split_role="validation",
        split_path=args.split_manifest,
        split_sha256=args.split_manifest_sha256,
    )
    dataset = Sabdab2ContactDataset(base, args.contact_cache_root)
    loader_arguments = {
        "batch_size": int(configuration["evaluation_batch_size"]),
        "workers": args.workers,
        "shuffle": False,
        "seed": seed,
        "pin_memory": True,
    }
    if "persistent_workers" in inspect.signature(training._loader).parameters:
        loader_arguments["persistent_workers"] = args.workers > 0
    loader = training._loader(dataset, **loader_arguments)
    model = training._model_from_configuration(
        dataset[0], configuration, str(result["experiment"])
    )
    device = torch.device("cuda")
    model.to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    per_complex: dict[str, dict[str, Any]] = {}
    with torch.inference_mode():
        for raw_batch in loader:
            batch = training._move_batch(raw_batch, device)
            with training._autocast(device, True):
                output = model(batch)
            for batch_index, instance_id_value in enumerate(raw_batch["instance_ids"]):
                instance_id = str(instance_id_value)
                if instance_id in per_complex:
                    raise ValueError(f"duplicate validation instance: {instance_id}")
                pair_mask = output["pair_candidate_mask"][batch_index]
                antibody_mask = output["antibody_mask"][batch_index]
                antigen_mask = output["antigen_mask"][batch_index]
                task_rows: dict[str, Any] = {}
                for task, labels, logits in (
                    (
                        "contact",
                        batch["pair_labels"][batch_index][pair_mask],
                        output["contact_logits"][batch_index][pair_mask],
                    ),
                    (
                        "paratope",
                        batch["antibody_labels"][batch_index][antibody_mask],
                        output["paratope_logits"][batch_index][antibody_mask],
                    ),
                    (
                        "epitope",
                        batch["antigen_labels"][batch_index][antigen_mask],
                        output["epitope_logits"][batch_index][antigen_mask],
                    ),
                ):
                    local_labels = labels.cpu().numpy().astype(np.uint8, copy=False)
                    local_logits = logits.float().cpu().numpy()
                    task_rows[task] = {
                        "average_precision": _finite_ap(
                            local_labels, local_logits, training._sigmoid_numpy
                        ),
                        "candidate_count": int(len(local_labels)),
                        "positive_count": int(local_labels.sum()),
                    }
                per_complex[instance_id] = task_rows

    if len(per_complex) != 549:
        raise ValueError("validation complex count differs")
    means = {
        task: float(
            np.mean([row[task]["average_precision"] for row in per_complex.values()])
        )
        for task in ("contact", "epitope", "paratope")
    }
    expected = result["selected_checkpoint_validation"]
    reproduction_error = {
        task: abs(means[task] - float(expected[task]["mean_per_complex_auprc"]))
        for task in means
    }
    if any(error > 1e-7 for error in reproduction_error.values()):
        raise ValueError(f"selected checkpoint metric did not reproduce: {reproduction_error}")

    payload = {
        "format_version": "sabdab2_predicted_geometry_per_complex_metrics_v1",
        "status": "PASS_FROZEN_DEVELOPMENT_CHECKPOINT_REEVALUATION",
        "geometry_source": args.geometry_source,
        "arm": args.arm,
        "seed": seed,
        "plan_sha256": PLAN_SHA256,
        "source_run": {
            "run_root": str(run_root),
            "result_sha256": sha256_file(result_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "receipt_sha256": sha256_file(receipt_path),
        },
        "authority": {
            "structure_manifest_sha256": authority["structure_manifest_sha256"],
            "contact_manifest_sha256": authority["contact_manifest_sha256"],
            "split_manifest_sha256": authority["split_manifest_sha256"],
            "test_loaded": False,
            "external_test_loaded": False,
        },
        "mean_per_complex_average_precision": means,
        "maximum_reproduction_error": max(reproduction_error.values()),
        "per_complex": per_complex,
        "selection_or_tuning": False,
    }
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent)
    )
    try:
        result_output = temporary / "PER_COMPLEX_METRICS.json"
        result_output.write_bytes(canonical_bytes(payload))
        output_receipt = {
            "format_version": "sabdab2_predicted_geometry_per_complex_metrics_receipt_v1",
            "status": payload["status"],
            "result_sha256": sha256_file(result_output),
            "arm": args.arm,
            "seed": seed,
            "validation_complexes": len(per_complex),
            "test_loaded": False,
            "temporal_holdout_loaded": False,
        }
        (temporary / "RECEIPT.json").write_bytes(canonical_bytes(output_receipt))
        os.replace(temporary, output_root)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(
        json.dumps(
            {
                **output_receipt,
                "receipt_sha256": sha256_file(output_root / "RECEIPT.json"),
                "output_root": str(output_root),
                "mean_per_complex_average_precision": means,
                "maximum_reproduction_error": max(reproduction_error.values()),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
