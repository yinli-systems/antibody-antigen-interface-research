#!/usr/bin/env python3
"""Measure bilateral Top-K readouts on one frozen checkpoint and seed."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-receipt", type=Path, required=True)
    parser.add_argument("--task", choices=("epitope", "paratope"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def load_training_module(source_root: Path) -> Any:
    sys.path.insert(0, str(source_root / "src"))
    path = source_root / "scripts/train_framecontact.py"
    spec = importlib.util.spec_from_file_location("frozen_framecontact_train", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen training module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def masked_topk_mean_probability_logit(
    logits: torch.Tensor, mask: torch.Tensor, *, dim: int, k: int
) -> torch.Tensor:
    """Compute the frozen probability-domain Top-K mean readout locally.

    The baseline source tree predates the exported helper, so the evaluator owns
    this small, source-independent readout definition.  It does not modify the
    frozen model or checkpoint.
    """

    if tuple(logits.shape) != tuple(mask.shape) or mask.dtype != torch.bool:
        raise ValueError("logits and boolean mask must have identical shapes")
    if k < 1:
        raise ValueError("k must be positive")
    observed_count = mask.sum(dim=dim)
    effective_k = min(k, logits.shape[dim])
    probabilities = torch.sigmoid(logits.to(torch.float32))
    top_values = torch.topk(
        probabilities.masked_fill(~mask, -1.0), k=effective_k, dim=dim
    ).values
    selected = top_values.clamp_min(0.0).sum(dim=dim)
    count = observed_count.clamp(min=1, max=effective_k).to(torch.float32)
    mean_probability = selected / count
    epsilon = torch.finfo(torch.float32).eps
    result = torch.logit(mean_probability.clamp(epsilon, 1.0 - epsilon))
    return result.masked_fill(observed_count == 0, 0.0)


def main() -> None:
    args = arguments()
    if args.output_root.exists() or args.output_root.is_symlink():
        raise FileExistsError(args.output_root)
    if os.environ.get("SLURM_JOB_PARTITION") != "gpu_4090":
        raise RuntimeError("evaluation must run on gpu_4090")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
        raise RuntimeError("deterministic CUBLAS configuration is required")

    protocol_raw = args.protocol.read_bytes()
    protocol_sha = hashlib.sha256(protocol_raw).hexdigest()
    protocol = json.loads(protocol_raw)
    receipt = json.loads(args.protocol_receipt.read_text(encoding="utf-8"))
    if receipt["protocol_sha256"] != protocol_sha:
        raise ValueError("protocol receipt differs")
    if protocol["status"] != "FROZEN_BEFORE_FIXED_CHECKPOINT_READOUT_MEASUREMENT":
        raise ValueError("protocol is not frozen")
    if args.seed not in (17, 29, 43):
        raise ValueError("seed is not frozen")
    if protocol["claim_boundary"]["test_loaded"] is not False:
        raise ValueError("test access is not closed")
    if protocol["claim_boundary"]["temporal_holdout_loaded"] is not False:
        raise ValueError("temporal access is not closed")

    task_record = protocol["tasks"][args.task]
    seed_record = task_record["seeds"][str(args.seed)]
    source_root = Path(task_record["source_root"])
    source_manifest = Path(task_record["source_manifest_path"])
    result_path = Path(seed_record["run_result_path"])
    checkpoint_path = Path(seed_record["checkpoint_path"])
    run_receipt_path = Path(seed_record["run_receipt_path"])
    if sha256_file(source_manifest) != task_record["source_manifest_sha256"]:
        raise ValueError("source manifest differs")
    if sha256_file(source_root / "scripts/train_framecontact.py") != task_record["trainer_sha256"]:
        raise ValueError("trainer differs")
    for path, key in (
        (result_path, "run_result_sha256"),
        (checkpoint_path, "checkpoint_sha256"),
        (run_receipt_path, "run_receipt_sha256"),
    ):
        if sha256_file(path) != seed_record[key]:
            raise ValueError(f"frozen run authority differs: {path}")
    for name, record in protocol["authorities"].items():
        if sha256_file(Path(record["path"])) != record["sha256"]:
            raise ValueError(f"global authority differs: {name}")

    run_result = json.loads(result_path.read_text(encoding="utf-8"))
    if int(run_result["configuration"]["seed"]) != args.seed:
        raise ValueError("run seed differs")
    if float(run_result["configuration"]["distance_loss_weight"]) != 0.0:
        raise ValueError("source run used distance auxiliary")
    if run_result["authority"]["test_loaded"] is not False:
        raise ValueError("source run accessed test")
    if run_result["authority"]["external_test_loaded"] is not False:
        raise ValueError("source run accessed external test")

    trainer = load_training_module(source_root)
    trainer.verify_source_manifest(
        source_root, source_manifest, task_record["source_manifest_sha256"]
    )
    from interaction_first.framecontact import masked_logmeanexp  # type: ignore
    from interaction_first.framecontact_metrics import binary_classification_metrics  # type: ignore
    from interaction_first.sabdab2_contacts import (  # type: ignore
        Sabdab2ContactBatchCollator,
        Sabdab2ContactDataset,
    )
    from interaction_first.sabdab2_structure import Sabdab2StructureEmbeddingDataset  # type: ignore
    from sklearn.metrics import average_precision_score

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")

    authorities = protocol["authorities"]
    structure_manifest = Path(authorities["structure_manifest"]["path"])
    contact_manifest = Path(authorities["contact_manifest"]["path"])
    embedding_manifest = Path(authorities["embedding_manifest"]["path"])
    split_path = Path(authorities["development_split"]["path"])
    base = Sabdab2StructureEmbeddingDataset(
        structure_manifest.parent,
        embedding_manifest.parent,
        None,
        split_role="validation",
        split_path=split_path,
        split_sha256=authorities["development_split"]["sha256"],
    )
    dataset = Sabdab2ContactDataset(base, contact_manifest.parent)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=Sabdab2ContactBatchCollator(),
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    model = trainer._model_from_configuration(
        dataset[0], run_result["configuration"], run_result["experiment"]
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if trainer._state_sha256(model) != seed_record["selected_state_sha256"]:
        raise ValueError("selected state differs")
    model.to(device).eval()

    k_values = [int(k) for k in protocol["k_values"]]
    arm_names = (
        ["native_baseline"]
        + [f"node_plus_topk{k}" for k in k_values]
        + [f"edge_only_topk{k}" for k in k_values]
        if args.task == "paratope"
        else [f"topk{k}" for k in k_values]
    )
    labels: list[np.ndarray] = []
    scores: dict[str, list[np.ndarray]] = {arm: [] for arm in arm_names}
    per_complex: dict[str, list[float]] = {arm: [] for arm in arm_names}
    instance_ids: list[str] = []
    zero_positive_complexes = 0
    max_native_topk2_abs_error = 0.0
    started = time.monotonic()

    with torch.inference_mode():
        for raw_batch in loader:
            batch = trainer._move_batch(raw_batch, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(batch)
            pair_mask = output["pair_candidate_mask"]
            contact_logits = output["contact_logits"].float()
            if args.task == "paratope":
                mask = output["antibody_mask"]
                local_labels_tensor = batch["antibody_labels"]
                baseline = output["paratope_logits"].float()
                node = baseline - masked_logmeanexp(contact_logits, pair_mask, dim=2)
                local_scores = {"native_baseline": baseline}
                for k in k_values:
                    marginal = masked_topk_mean_probability_logit(
                        contact_logits, pair_mask, dim=2, k=k
                    )
                    local_scores[f"node_plus_topk{k}"] = node + marginal
                    local_scores[f"edge_only_topk{k}"] = marginal
            else:
                mask = output["antigen_mask"]
                local_labels_tensor = batch["antigen_labels"]
                local_scores = {}
                for k in k_values:
                    local_scores[f"topk{k}"] = masked_topk_mean_probability_logit(
                        contact_logits, pair_mask, dim=1, k=k
                    )
                native = output["epitope_logits"].float()
                max_native_topk2_abs_error = max(
                    max_native_topk2_abs_error,
                    float((native - local_scores["topk2"]).abs()[mask].max().item()),
                )

            valid = mask[0]
            local_labels = (
                local_labels_tensor[0][valid].cpu().numpy().astype(np.int8)
            )
            labels.append(local_labels)
            zero_positive_complexes += int(local_labels.sum() == 0)
            instance_ids.append(str(raw_batch["instance_ids"][0]))
            for arm, tensor in local_scores.items():
                local = tensor[0][valid].cpu().numpy().astype(np.float32)
                scores[arm].append(local)
                per_complex[arm].append(
                    float(average_precision_score(local_labels, local))
                )

    if args.task == "epitope" and max_native_topk2_abs_error > 1e-6:
        raise ValueError(
            f"native expert readout is not TopK2: {max_native_topk2_abs_error}"
        )
    pooled_labels = np.concatenate(labels)
    metrics: dict[str, Any] = {}
    for arm in arm_names:
        pooled_scores = np.concatenate(scores[arm]).astype(np.float64)
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(pooled_scores, -60.0, 60.0)))
        threshold = trainer._best_f1_threshold(pooled_labels, probabilities)
        metrics[arm] = {
            **binary_classification_metrics(
                pooled_labels, probabilities, threshold=threshold
            ),
            "mean_per_complex_ap": float(np.mean(per_complex[arm])),
            "threshold_selected_on_same_development_validation": float(threshold),
        }
        for value in metrics[arm].values():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"non-finite metric: {arm}")

    result = {
        "format_version": "bilateral_topk_readout_result_v1",
        "protocol_sha256": protocol_sha,
        "task": args.task,
        "seed": args.seed,
        "checkpoint_family": task_record["checkpoint_family"],
        "source_run": {
            "selected_epoch": seed_record["selected_epoch"],
            "selected_state_sha256": seed_record["selected_state_sha256"],
            "checkpoint_sha256": seed_record["checkpoint_sha256"],
            "distance_loss_weight": 0.0,
        },
        "counts": {
            "complexes": len(instance_ids),
            "unique_instance_ids": len(set(instance_ids)),
            "zero_positive_complexes": zero_positive_complexes,
        },
        "metrics": metrics,
        "per_complex": {
            "instance_ids": instance_ids,
            "average_precision": per_complex,
        },
        "audit": {
            "native_topk2_max_abs_error": max_native_topk2_abs_error,
            "nonfinite_score_count": int(
                sum(
                    np.size(array) - np.isfinite(array).sum()
                    for arrays in scores.values()
                    for array in arrays
                )
            ),
            "test_loaded": False,
            "temporal_holdout_loaded": False,
        },
        "runtime": {
            "seconds": time.monotonic() - started,
            "gpu": torch.cuda.get_device_name(device),
        },
    }
    if result["counts"]["complexes"] != 549:
        raise ValueError("validation cohort differs")
    if result["counts"]["unique_instance_ids"] != 549:
        raise ValueError("duplicate instance id")
    if result["audit"]["nonfinite_score_count"] != 0:
        raise ValueError("non-finite score detected")

    args.output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{args.output_root.name}.", dir=args.output_root.parent)
    )
    try:
        result_path = temporary / "BILATERAL_TOPK_READOUT_RESULT_V1.json"
        result_path.write_bytes(canonical_bytes(result))
        result_sha = sha256_file(result_path)
        (temporary / "BILATERAL_TOPK_READOUT_RESULT_V1_RECEIPT.json").write_bytes(
            canonical_bytes(
                {
                    "format_version": "bilateral_topk_readout_result_receipt_v1",
                    "protocol_sha256": protocol_sha,
                    "result_sha256": result_sha,
                    "task": args.task,
                    "seed": args.seed,
                }
            )
        )
        temporary.rename(args.output_root)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(
        json.dumps(
            {
                "task": args.task,
                "seed": args.seed,
                "result_sha256": result_sha,
                "mean_per_complex_ap": {
                    arm: metrics[arm]["mean_per_complex_ap"] for arm in arm_names
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
