#!/usr/bin/env python3
"""Re-evaluate one selected FrameContact checkpoint and freeze pair logits."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import platform
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def close(left: float, right: float, tolerance: float = 1e-7) -> bool:
    return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)


def flat_pair_rank(scores: np.ndarray, labels: np.ndarray) -> dict[str, float | int]:
    local_scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    local_labels = np.asarray(labels, dtype=np.uint8).reshape(-1)
    positives = int(local_labels.sum())
    if positives <= 0:
        return {
            "eligible": False,
            "pairs": int(local_labels.size),
            "positives": 0,
        }
    order = np.argsort(-local_scores, kind="stable")
    ranked = local_labels[order]
    hit = int(ranked[0])
    first_rank = int(np.flatnonzero(ranked)[0] + 1)
    return {
        "eligible": True,
        "pairs": int(local_labels.size),
        "positives": positives,
        "hit_at_1": hit,
        "recall_at_1": hit / positives,
        "first_positive_rank": first_rank,
        "reciprocal_rank": 1.0 / first_rank,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--arm", choices=("candidate", "control"), required=True)
    parser.add_argument("--structure-cache-root", type=Path, required=True)
    parser.add_argument("--contact-cache-root", type=Path, required=True)
    parser.add_argument("--embedding-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--split-manifest-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if os.environ.get("SLURM_JOB_PARTITION") != "gpu_4090":
        raise RuntimeError("This evaluation is authorized on gpu_4090 only")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    repository_root = args.repository_root.resolve()
    sys.path.insert(0, str(repository_root / "src"))
    sys.path.insert(0, str(repository_root / "scripts"))

    import train_framecontact as training  # noqa: PLC0415
    from interaction_first import framecontact_metrics  # noqa: PLC0415
    ranked_pair_metrics = framecontact_metrics.ranked_pair_metrics
    if hasattr(framecontact_metrics, "positive_required_average_precision"):
        positive_required_average_precision = (
            framecontact_metrics.positive_required_average_precision
        )
    else:
        from sklearn.metrics import average_precision_score  # noqa: PLC0415

        def positive_required_average_precision(
            labels: np.ndarray, probabilities: np.ndarray
        ) -> dict[str, Any]:
            positives = int(np.asarray(labels).sum())
            if positives <= 0:
                return {"eligible": False, "value": None}
            return {
                "eligible": True,
                "value": float(average_precision_score(labels, probabilities)),
            }
    from interaction_first.sabdab2_contacts import (  # noqa: PLC0415
        Sabdab2ContactDataset,
    )
    from interaction_first.sabdab2_structure import (  # noqa: PLC0415
        Sabdab2StructureEmbeddingDataset,
    )

    run_root = args.run_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"Refusing to overwrite: {output_root}")
    result_path = run_root / "FRAMECONTACT_DEVELOPMENT_RESULT.json"
    checkpoint_path = run_root / "FRAMECONTACT_SELECTED.pt"
    receipt_path = run_root / "RUN_RECEIPT.json"
    result = read_json(result_path)
    receipt = read_json(receipt_path)
    if sha256_file(result_path) != receipt["result_sha256"]:
        raise ValueError("Development result SHA-256 differs from its receipt")
    if sha256_file(checkpoint_path) != receipt["selected_checkpoint_sha256"]:
        raise ValueError("Selected checkpoint SHA-256 differs from its receipt")
    authority = result["authority"]
    if authority.get("test_loaded") is not False or authority.get("external_test_loaded") is not False:
        raise ValueError("Source run did not keep test and external roles sealed")
    if authority.get("split_manifest_sha256") != args.split_manifest_sha256:
        raise ValueError("Source run split SHA-256 differs from requested split")
    if sha256_file(args.split_manifest) != args.split_manifest_sha256:
        raise ValueError("Split manifest file SHA-256 mismatch")

    configuration = result["configuration"]
    seed = int(result["configuration"]["seed"])
    evaluation_batch_size = int(configuration["evaluation_batch_size"])
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
        "batch_size": evaluation_batch_size,
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

    instance_ids: list[str] = []
    antigen_lengths: list[int] = []
    antibody_lengths: list[int] = []
    logits_parts: list[np.ndarray] = []
    labels_parts: list[np.ndarray] = []
    offsets = [0]
    per_instance: dict[str, dict[str, Any]] = {}
    existing = result["selected_checkpoint_validation"]["contact"]
    existing_ranked = existing["ranked_pair_by_instance"]
    existing_ap_by_instance = existing.get("per_complex_auprc_by_instance", {})
    maximum_cutoff_error = 0.0
    maximum_ap_error = 0.0
    positive_eligible = 0
    reproduced_ap_values: list[float] = []

    with torch.inference_mode():
        for raw_batch in loader:
            batch = training._move_batch(raw_batch, device)
            with training._autocast(device, True):
                output = model(batch)
            for batch_index, instance_id_value in enumerate(raw_batch["instance_ids"]):
                instance_id = str(instance_id_value)
                if instance_id in per_instance:
                    raise ValueError(f"Duplicate validation instance: {instance_id}")
                pair_mask = output["pair_candidate_mask"][batch_index]
                logits = (
                    output["contact_logits"][batch_index][pair_mask]
                    .float()
                    .cpu()
                    .numpy()
                    .astype(np.float32, copy=False)
                )
                labels = (
                    batch["pair_labels"][batch_index][pair_mask]
                    .cpu()
                    .numpy()
                    .astype(np.uint8, copy=False)
                )
                antibody_length = int(output["antibody_mask"][batch_index].sum())
                antigen_length = int(output["antigen_mask"][batch_index].sum())
                if logits.size != antibody_length * antigen_length:
                    raise ValueError(
                        f"{instance_id}: candidate matrix is not a complete rectangle"
                    )
                if not np.isfinite(logits).all():
                    raise ValueError(f"{instance_id}: non-finite contact logits")
                if not np.all(np.isin(np.unique(labels), (0, 1))):
                    raise ValueError(f"{instance_id}: non-binary contact labels")

                probabilities = training._sigmoid_numpy(logits)
                rank = flat_pair_rank(probabilities, labels)
                legacy = ranked_pair_metrics(
                    labels,
                    probabilities,
                    antibody_length=antibody_length,
                    antigen_length=antigen_length,
                )
                expected = existing_ranked[instance_id]
                for field in (
                    "antibody_length",
                    "antigen_length",
                    "positive_count",
                    "candidate_count",
                ):
                    if int(legacy[field]) != int(expected[field]):
                        raise ValueError(f"{instance_id}: {field} does not reproduce")
                for cutoff, values in legacy["cutoffs"].items():
                    for metric in ("precision", "recall", "effective_k"):
                        error = abs(float(values[metric]) - float(expected["cutoffs"][cutoff][metric]))
                        maximum_cutoff_error = max(maximum_cutoff_error, error)
                        if error > 1e-7:
                            raise ValueError(
                                f"{instance_id}: {cutoff}/{metric} reproduction error {error}"
                            )
                local_ap = positive_required_average_precision(labels, probabilities)
                expected_ap = existing_ap_by_instance.get(instance_id)
                if local_ap["eligible"]:
                    positive_eligible += 1
                    reproduced_ap_values.append(float(local_ap["value"]))
                    if existing_ap_by_instance:
                        if expected_ap is None:
                            raise ValueError(f"{instance_id}: expected AP is missing")
                        ap_error = abs(float(local_ap["value"]) - float(expected_ap))
                        maximum_ap_error = max(maximum_ap_error, ap_error)
                        if ap_error > 1e-7:
                            raise ValueError(
                                f"{instance_id}: AP reproduction error {ap_error}"
                            )

                instance_ids.append(instance_id)
                antigen_lengths.append(antigen_length)
                antibody_lengths.append(antibody_length)
                logits_parts.append(logits)
                labels_parts.append(labels)
                offsets.append(offsets[-1] + int(logits.size))
                per_instance[instance_id] = {
                    "antibody_length": antibody_length,
                    "antigen_length": antigen_length,
                    **rank,
                }

    if set(instance_ids) != set(existing_ranked):
        raise ValueError("Re-evaluated validation identifiers differ from the source result")
    aggregate_ap_error = abs(
        float(np.mean(reproduced_ap_values)) - float(existing["mean_per_complex_auprc"])
    )
    maximum_ap_error = max(maximum_ap_error, aggregate_ap_error)
    if aggregate_ap_error > 1e-7:
        raise ValueError(f"Mean per-complex AP reproduction error {aggregate_ap_error}")
    logits_all = np.concatenate(logits_parts).astype(np.float32, copy=False)
    labels_all = np.concatenate(labels_parts).astype(np.uint8, copy=False)
    eligible_records = [record for record in per_instance.values() if record["eligible"]]
    rank_summary = {
        "complexes": len(per_instance),
        "positive_eligible_complexes": len(eligible_records),
        "pairs": int(logits_all.size),
        "positives": int(labels_all.sum()),
        "hit_at_1_complexes": sum(int(record["hit_at_1"]) for record in eligible_records),
        "hit_at_1_rate_macro": float(
            np.mean([record["hit_at_1"] for record in eligible_records])
        ),
        "recall_at_1_macro": float(
            np.mean([record["recall_at_1"] for record in eligible_records])
        ),
        "recall_at_1_micro": (
            sum(int(record["hit_at_1"]) for record in eligible_records)
            / sum(int(record["positives"]) for record in eligible_records)
        ),
        "mrr_first_positive_macro": float(
            np.mean([record["reciprocal_rank"] for record in eligible_records])
        ),
        "mean_first_positive_rank": float(
            np.mean([record["first_positive_rank"] for record in eligible_records])
        ),
    }

    created_at = datetime.now(timezone.utc).isoformat()
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent))
    try:
        prediction_path = temporary / "FROZEN_PAIR_PREDICTIONS.npz"
        np.savez(
            prediction_path,
            instance_ids=np.asarray(instance_ids, dtype=np.str_),
            offsets=np.asarray(offsets, dtype=np.int64),
            antibody_lengths=np.asarray(antibody_lengths, dtype=np.int32),
            antigen_lengths=np.asarray(antigen_lengths, dtype=np.int32),
            contact_logits=logits_all,
            contact_labels=labels_all,
        )
        prediction_sha256 = sha256_file(prediction_path)
        manifest = {
            "format_version": "framecontact_frozen_pair_predictions_v1",
            "status": "complete",
            "created_at_utc": created_at,
            "arm": args.arm,
            "seed": seed,
            "scope": "SAbDab2 double-cold development validation only",
            "claim_boundary": (
                "Test and external roles remain sealed; no benchmark-improvement claim is authorized."
            ),
            "source": {
                "run_root": str(run_root),
                "result_sha256": receipt["result_sha256"],
                "selected_checkpoint_sha256": receipt["selected_checkpoint_sha256"],
                "split_manifest": str(args.split_manifest.resolve()),
                "split_manifest_sha256": args.split_manifest_sha256,
            },
            "prediction": {
                "path": prediction_path.name,
                "sha256": prediction_sha256,
                "bytes": prediction_path.stat().st_size,
                "arrays": {
                    "instance_ids": "Unicode complex identifiers in dataset order",
                    "offsets": "start/end offsets into flattened pair arrays",
                    "antibody_lengths": "antibody row count per complex",
                    "antigen_lengths": "antigen column count per complex",
                    "contact_logits": "raw float32 model logits, antibody-major row order",
                    "contact_labels": "uint8 true contact matrix, same order as logits",
                },
            },
            "calibration": {
                "validation_temperature": result["selected_checkpoint_validation"][
                    "validation_temperature"
                ],
                "contact_threshold": existing["threshold_selected_on_validation"],
                "source": "source selected-checkpoint validation result",
            },
            "reproduction": {
                "legacy_ranked_cutoffs": "PASS",
                "per_complex_auprc": "PASS",
                "maximum_absolute_ranked_cutoff_error": maximum_cutoff_error,
                "maximum_absolute_per_complex_auprc_error": maximum_ap_error,
                "positive_eligible_complexes": positive_eligible,
                "evaluation_batch_size": evaluation_batch_size,
            },
            "metric_definition": {
                "ranking_unit": "all antibody-antigen residue pairs within one complex",
                "tie_break": "stable descending score, then antibody-major row order",
                "recall_at_1": "top-one true contacts divided by all true contacts in the complex",
                "mrr_first_positive": "reciprocal rank of the first true contact",
                "aggregation": "macro over positive-eligible validation complexes",
            },
            "flat_pair_rank_summary": rank_summary,
            "per_instance": per_instance,
        }
        manifest_path = temporary / "FROZEN_PAIR_MANIFEST.json"
        write_json(manifest_path, manifest)
        receipt_out = {
            "format_version": "framecontact_frozen_pair_predictions_receipt_v1",
            "status": "complete",
            "created_at_utc": created_at,
            "output_root": str(output_root),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "artifacts": {
                prediction_path.name: prediction_sha256,
                manifest_path.name: sha256_file(manifest_path),
            },
        }
        write_json(temporary / "RUN_RECEIPT.json", receipt_out)
        os.replace(temporary, output_root)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps({"output_root": str(output_root), "summary": rank_summary}, sort_keys=True))


if __name__ == "__main__":
    main()
