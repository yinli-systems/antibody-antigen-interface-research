#!/usr/bin/env python3
"""Evaluate frozen TopK2 readout transfer on development validation only."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-receipt", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--run-result", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--structure-root", type=Path, required=True)
    parser.add_argument("--contact-root", type=Path, required=True)
    parser.add_argument("--embedding-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_training_module(source_root: Path) -> Any:
    source_dir = source_root / "src"
    sys.path.insert(0, str(source_dir))
    path = source_root / "scripts" / "train_framecontact.py"
    spec = importlib.util.spec_from_file_location("frozen_framecontact_train", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen training module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bootstrap_mean_delta(values: np.ndarray, *, seed: int, draws: int = 10000) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    sampled = np.empty(draws, dtype=np.float64)
    for index in range(draws):
        sampled[index] = rng.choice(values, size=len(values), replace=True).mean()
    return {
        "point_estimate": float(values.mean()),
        "ci95_percentile": [float(x) for x in np.quantile(sampled, [0.025, 0.975])],
        "draws": draws,
        "resampling_unit": "development_validation_complex",
    }


def main() -> None:
    args = arguments()
    if args.output_root.exists() or args.output_root.is_symlink():
        raise FileExistsError("immutable output root already exists")
    if os.environ.get("SLURM_JOB_PARTITION") != "gpu_4090":
        raise RuntimeError("evaluation must run on gpu_4090")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
        raise RuntimeError("deterministic CUBLAS configuration is required")

    protocol_raw = args.protocol.read_bytes()
    protocol = json.loads(protocol_raw)
    receipt = json.loads(args.protocol_receipt.read_text())
    protocol_sha = hashlib.sha256(protocol_raw).hexdigest()
    if receipt["protocol_sha256"] != protocol_sha:
        raise ValueError("protocol receipt differs")
    if protocol["status"] != "frozen_before_measurement":
        raise ValueError("protocol is not frozen")
    if protocol["claim_boundary"] != {
        "role": "development_validation",
        "selection": False,
        "test_loaded": False,
        "temporal_holdout_loaded": False,
    }:
        raise ValueError("claim boundary differs")

    path_hashes = {
        "source_manifest_sha256": sha256_file(args.source_manifest),
        "run_result_sha256": sha256_file(args.run_result),
        "checkpoint_file_sha256": sha256_file(args.checkpoint),
        "structure_manifest_sha256": sha256_file(args.structure_root / "STRUCTURE_CACHE_MANIFEST.json"),
        "contact_manifest_sha256": sha256_file(args.contact_root / "CONTACT_CACHE_MANIFEST.json"),
        "embedding_manifest_sha256": sha256_file(args.embedding_root / "SABDAB2_ESM_EMBEDDING_MANIFEST.json"),
        "split_sha256": sha256_file(args.split),
    }
    if path_hashes != protocol["authorities"]:
        raise ValueError("one or more frozen authorities differ")

    run_result = json.loads(args.run_result.read_text())
    if run_result["authority"]["test_loaded"] is not False:
        raise ValueError("source run accessed test")
    if run_result["authority"]["external_test_loaded"] is not False:
        raise ValueError("source run accessed external test")
    if int(run_result["selection"]["best_epoch"]) != 2:
        raise ValueError("unexpected selected checkpoint epoch")

    trainer = load_training_module(args.source_root)
    trainer.verify_source_manifest(
        args.source_root, args.source_manifest, path_hashes["source_manifest_sha256"]
    )
    from interaction_first.framecontact import (  # type: ignore
        masked_logmeanexp,
        masked_topk_mean_probability_logit,
    )
    from interaction_first.framecontact_metrics import binary_classification_metrics  # type: ignore
    from interaction_first.sabdab2_contacts import (  # type: ignore
        Sabdab2ContactBatchCollator,
        Sabdab2ContactDataset,
    )
    from interaction_first.sabdab2_structure import Sabdab2StructureEmbeddingDataset  # type: ignore

    seed = int(run_result["configuration"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    base = Sabdab2StructureEmbeddingDataset(
        args.structure_root,
        args.embedding_root,
        None,
        split_role="validation",
        split_path=args.split,
        split_sha256=path_hashes["split_sha256"],
    )
    dataset = Sabdab2ContactDataset(base, args.contact_root)
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
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if trainer._state_sha256(model) != run_result["model"]["selected_state_sha256"]:
        raise ValueError("selected model state differs")
    model.to(device).eval()

    arms = {
        "contact": ("baseline_pair", "topk2_context_equal_logit"),
        "paratope": ("baseline_node_plus_logmeanexp", "topk2_edge_only", "node_plus_topk2"),
        "epitope": ("baseline_topk2",),
    }
    labels: dict[str, list[np.ndarray]] = {task: [] for task in arms}
    scores: dict[str, dict[str, list[np.ndarray]]] = {
        task: {arm: [] for arm in task_arms} for task, task_arms in arms.items()
    }
    per_complex: dict[str, dict[str, list[float]]] = {
        task: {arm: [] for arm in task_arms} for task, task_arms in arms.items()
    }
    instance_ids: list[str] = []
    zero_positive = {task: 0 for task in arms}
    from sklearn.metrics import average_precision_score, roc_auc_score

    started = time.monotonic()
    with torch.inference_mode():
        for raw_batch in loader:
            batch = trainer._move_batch(raw_batch, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(batch)
            pair_mask = output["pair_candidate_mask"]
            contact_logits = output["contact_logits"].float()
            row_topk2 = masked_topk_mean_probability_logit(
                contact_logits, pair_mask, dim=2, k=2
            )
            column_topk2 = masked_topk_mean_probability_logit(
                contact_logits, pair_mask, dim=1, k=2
            )
            contact_context = (
                contact_logits + row_topk2[:, :, None] + column_topk2[:, None, :]
            ) / 3.0
            paratope_logmeanexp = masked_logmeanexp(contact_logits, pair_mask, dim=2)
            paratope_node = output["paratope_logits"].float() - paratope_logmeanexp
            score_tensors = {
                "contact": {
                    "baseline_pair": contact_logits,
                    "topk2_context_equal_logit": contact_context,
                },
                "paratope": {
                    "baseline_node_plus_logmeanexp": output["paratope_logits"].float(),
                    "topk2_edge_only": row_topk2,
                    "node_plus_topk2": paratope_node + row_topk2,
                },
                "epitope": {"baseline_topk2": output["epitope_logits"].float()},
            }
            masks = {
                "contact": pair_mask,
                "paratope": output["antibody_mask"],
                "epitope": output["antigen_mask"],
            }
            task_labels = {
                "contact": batch["pair_labels"],
                "paratope": batch["antibody_labels"],
                "epitope": batch["antigen_labels"],
            }
            instance_ids.append(str(raw_batch["instance_ids"][0]))
            for task in arms:
                mask = masks[task][0]
                local_labels = task_labels[task][0][mask].cpu().numpy().astype(np.int8)
                labels[task].append(local_labels)
                if int(local_labels.sum()) == 0:
                    zero_positive[task] += 1
                for arm, tensor in score_tensors[task].items():
                    local_scores = tensor[0][mask].cpu().numpy().astype(np.float32)
                    scores[task][arm].append(local_scores)
                    per_complex[task][arm].append(
                        float(average_precision_score(local_labels, local_scores))
                    )

    metrics: dict[str, Any] = {}
    paired: dict[str, Any] = {}
    for task, task_arms in arms.items():
        pooled_labels = np.concatenate(labels[task])
        metrics[task] = {}
        for arm in task_arms:
            pooled_scores = np.concatenate(scores[task][arm]).astype(np.float64)
            probabilities = 1.0 / (1.0 + np.exp(-np.clip(pooled_scores, -60.0, 60.0)))
            threshold = trainer._best_f1_threshold(pooled_labels, probabilities)
            metrics[task][arm] = {
                **binary_classification_metrics(
                    pooled_labels, probabilities, threshold=threshold
                ),
                "threshold_selected_on_same_development_validation": float(threshold),
                "mean_per_complex_ap": float(np.mean(per_complex[task][arm])),
                "complex_count": len(instance_ids),
                "zero_positive_complex_count": zero_positive[task],
            }
        baseline = task_arms[0]
        for arm in task_arms[1:]:
            delta = np.asarray(per_complex[task][arm]) - np.asarray(
                per_complex[task][baseline]
            )
            paired[f"{task}:{arm}-minus-{baseline}"] = bootstrap_mean_delta(
                delta, seed=20260828
            )

    result = {
        "format_version": "multitask_topk2_readout_development_v1",
        "protocol_sha256": protocol_sha,
        "authorities": path_hashes,
        "claim_boundary": protocol["claim_boundary"],
        "definitions": protocol["readouts"],
        "source_run": {
            "seed": seed,
            "selected_epoch": 2,
            "selected_state_sha256": run_result["model"]["selected_state_sha256"],
        },
        "counts": {
            "complexes": len(instance_ids),
            "unique_instance_ids": len(set(instance_ids)),
        },
        "metrics": metrics,
        "paired_complex_bootstrap": paired,
        "runtime": {
            "seconds": time.monotonic() - started,
            "gpu": torch.cuda.get_device_name(device),
            "nonfinite_scores": int(
                sum(
                    np.size(array) - np.isfinite(array).sum()
                    for task in scores.values()
                    for arrays in task.values()
                    for array in arrays
                )
            ),
        },
    }
    if result["counts"] != {"complexes": 549, "unique_instance_ids": 549}:
        raise ValueError("validation cohort differs")
    if result["runtime"]["nonfinite_scores"] != 0:
        raise ValueError("non-finite score detected")

    args.output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".multitask-topk2-", dir=args.output_root.parent))
    try:
        result_path = temporary / "MULTITASK_TOPK2_READOUT_RESULT_V1.json"
        result_path.write_bytes(canonical_bytes(result))
        result_sha = sha256_file(result_path)
        result_receipt = {
            "format_version": "multitask_topk2_readout_receipt_v1",
            "protocol_sha256": protocol_sha,
            "result_sha256": result_sha,
        }
        (temporary / "MULTITASK_TOPK2_READOUT_RESULT_V1_RECEIPT.json").write_bytes(
            canonical_bytes(result_receipt)
        )
        temporary.rename(args.output_root)
    except BaseException:
        import shutil

        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps({"result_sha256": result_sha, "metrics": metrics}, sort_keys=True))


if __name__ == "__main__":
    main()
