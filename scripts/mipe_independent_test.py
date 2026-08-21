#!/usr/bin/env python3
"""Evaluate frozen MIPE fold checkpoints on the untouched official test set.

The primary estimate is the arithmetic mean of five independently trained fold
models.  Both epoch-800 endpoints and validation-selected checkpoints are
reported.  No choice is made using independent-test labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROLES = ("primary_epoch_800", "secondary_validation_best")
EXPECTED_TEST_SHA256 = "adc4b5fcf1ee9bc113cc1b844b33303a7741cb46ac7b3dc6b1520eefdae509ca"
PUBLISHED_JOINT = {
    "paratope": {"auroc": 0.894, "auprc": 0.738, "mcc_at_0_5": 0.531},
    "epitope": {"auroc": 0.815, "auprc": 0.422, "mcc_at_0_5": 0.235},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)


def average_precision(target: np.ndarray, score: np.ndarray) -> float:
    target = np.asarray(target, dtype=np.uint8).reshape(-1)
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    positives = int(target.sum())
    if target.size != score.size or not target.size or positives == 0:
        return float("nan")
    order = np.argsort(-score, kind="mergesort")
    target = target[order]
    score = score[order]
    distinct = np.where(np.diff(score))[0]
    threshold_indices = np.r_[distinct, target.size - 1]
    true_positives = np.cumsum(target)[threshold_indices]
    false_positives = 1 + threshold_indices - true_positives
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / positives
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def auroc(target: np.ndarray, score: np.ndarray) -> float:
    target = np.asarray(target, dtype=np.uint8).reshape(-1)
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    positives = int(target.sum())
    negatives = int(target.size - positives)
    if target.size != score.size or positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    sorted_score = score[order]
    ranks = np.empty(target.size, dtype=np.float64)
    start = 0
    while start < target.size:
        stop = start + 1
        while stop < target.size and sorted_score[stop] == sorted_score[start]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    positive_rank_sum = float(ranks[target == 1].sum())
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (
        positives * negatives
    )


def mcc(target: np.ndarray, score: np.ndarray, threshold: float = 0.5) -> float:
    target = np.asarray(target, dtype=np.uint8).reshape(-1)
    prediction = np.asarray(score).reshape(-1) >= threshold
    positive = target == 1
    tp = int(np.sum(prediction & positive))
    tn = int(np.sum(~prediction & ~positive))
    fp = int(np.sum(prediction & ~positive))
    fn = int(np.sum(~prediction & positive))
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return float((tp * tn - fp * fn) / denominator) if denominator else 0.0


def residue_metrics(targets: Iterable[np.ndarray], scores: Iterable[np.ndarray]) -> dict[str, Any]:
    target = np.concatenate([np.asarray(item).reshape(-1) for item in targets])
    score = np.concatenate([np.asarray(item).reshape(-1) for item in scores])
    return {
        "residues": int(target.size),
        "positives": int(target.sum()),
        "prevalence": float(target.mean()),
        "auroc": auroc(target, score),
        "auprc": average_precision(target, score),
        "mcc_at_0_5": mcc(target, score, 0.5),
    }


def reciprocal_rank(target: np.ndarray, score: np.ndarray) -> float:
    target = np.asarray(target, dtype=np.uint8).reshape(-1)
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    order = np.argsort(-score, kind="mergesort")
    hits = np.flatnonzero(target[order])
    return 0.0 if hits.size == 0 else float(1.0 / (int(hits[0]) + 1))


def pair_metrics(targets: list[np.ndarray], scores: list[np.ndarray]) -> dict[str, Any]:
    per_complex_ap = [average_precision(y, s) for y, s in zip(targets, scores)]
    hit_at_1 = [float(np.asarray(y).reshape(-1)[int(np.argmax(s))]) for y, s in zip(targets, scores)]
    reciprocal_ranks = [reciprocal_rank(y, s) for y, s in zip(targets, scores)]
    flat_target = np.concatenate([np.asarray(item).reshape(-1) for item in targets])
    flat_score = np.concatenate([np.asarray(item).reshape(-1) for item in scores])
    return {
        "complexes": len(targets),
        "candidate_pairs": int(flat_target.size),
        "positive_pairs": int(flat_target.sum()),
        "prevalence": float(flat_target.mean()),
        "macro_auprc": float(np.nanmean(per_complex_ap)),
        "pooled_auprc": average_precision(flat_target, flat_score),
        "recall_at_1": float(np.mean(hit_at_1)),
        "mrr": float(np.mean(reciprocal_ranks)),
    }


def to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "float"):
        value = value.float()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--evaluator-root", type=Path, required=True)
    parser.add_argument("--test-data", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    output_root = arguments.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite {output_root}")
    output_root.mkdir(parents=True)

    test_sha256 = sha256_file(arguments.test_data)
    if test_sha256 != EXPECTED_TEST_SHA256:
        raise ValueError("Official independent-test dataset digest changed")

    sys.path.insert(0, str(arguments.evaluator_root.resolve()))
    try:
        import export_predictions as exporter
    finally:
        sys.path.pop(0)

    import pandas as pd
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for MIPE inference")
    device = torch.device(arguments.device)
    torch.cuda.set_device(device)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    authority = exporter.load_selection_authority(arguments.run_root)
    model_class, create_graph = exporter._import_official_mipe(arguments.source_root)
    dataset = pd.read_pickle(arguments.test_data)
    if len(dataset) != 64:
        raise ValueError(f"Expected 64 official test records, found {len(dataset)}")
    complex_codes = [str(sample["complex_code"]) for sample in dataset]
    if len(set(code.lower() for code in complex_codes)) != len(complex_codes):
        raise ValueError("Independent-test complex codes are not unique")

    # role -> fold -> records; each record is validated by the sealed exporter.
    all_records: dict[str, list[list[Any]]] = {role: [] for role in ROLES}
    file_digests: dict[str, str] = {}
    per_fold: dict[str, list[dict[str, Any]]] = {role: [] for role in ROLES}

    for role in ROLES:
        expected_source_files = None
        expected_hardened_files = None
        for fold_authority in authority.folds:
            checkpoint = fold_authority.for_role(role)
            payload, source_files, hardened_files = exporter._load_checkpoint_payload(
                checkpoint,
                fold=fold_authority.fold,
                authority=authority,
                expected_source_files=expected_source_files,
                expected_hardened_source_files=expected_hardened_files,
                torch=torch,
            )
            expected_source_files = source_files
            expected_hardened_files = hardened_files
            model = model_class().to(device)
            model.load_state_dict(payload["model_state_dict"], strict=True)
            records = list(
                exporter.infer_fold_records(
                    model=model,
                    samples=dataset,
                    dataset_indices=tuple(range(len(dataset))),
                    device=device,
                    create_graph=create_graph,
                    torch=torch,
                    fold=fold_authority.fold,
                    role=role,
                )
            )
            all_records[role].append(records)

            fold_root = output_root / "predictions" / role / f"fold_{fold_authority.fold}"
            fold_root.mkdir(parents=True)
            for index, record in enumerate(records):
                path = fold_root / f"complex_{index:05d}.npz"
                np.savez_compressed(
                    path,
                    complex_code=np.asarray(record.complex_code),
                    dataset_index=np.asarray(record.dataset_index, dtype=np.int64),
                    epitope_score=record.epitope_score,
                    paratope_score=record.paratope_score,
                    ag_to_ab_score=record.ag_to_ab_score,
                    ab_to_ag_score=record.ab_to_ag_score,
                    mean_bidirectional_score=record.mean_bidirectional_score,
                    epitope_target=record.epitope_target,
                    paratope_target=record.paratope_target,
                    pair_target=record.pair_target,
                )
                file_digests[str(path.relative_to(output_root))] = sha256_file(path)

            per_fold[role].append(
                {
                    "fold": fold_authority.fold,
                    "epoch": checkpoint.epoch,
                    "checkpoint_sha256": checkpoint.checkpoint_sha256,
                    "epitope": residue_metrics(
                        [record.epitope_target for record in records],
                        [record.epitope_score for record in records],
                    ),
                    "paratope": residue_metrics(
                        [record.paratope_target for record in records],
                        [record.paratope_score for record in records],
                    ),
                }
            )
            print(f"completed role={role} fold={fold_authority.fold}", flush=True)
            del model, payload
            torch.cuda.empty_cache()

    role_results: dict[str, Any] = {}
    for role in ROLES:
        ensemble_records = []
        ensemble_root = output_root / "predictions" / role / "five_fold_mean_ensemble"
        ensemble_root.mkdir(parents=True)
        for index in range(len(dataset)):
            fold_records = [fold[index] for fold in all_records[role]]
            reference = fold_records[0]
            epitope_score = np.mean(
                [record.epitope_score for record in fold_records], axis=0
            ).astype(np.float32)
            paratope_score = np.mean(
                [record.paratope_score for record in fold_records], axis=0
            ).astype(np.float32)
            ag_to_ab = np.mean(
                [record.ag_to_ab_score for record in fold_records], axis=0
            ).astype(np.float32)
            ab_to_ag = np.mean(
                [record.ab_to_ag_score for record in fold_records], axis=0
            ).astype(np.float32)
            attention = ((ag_to_ab.astype(np.float64) + ab_to_ag.T) / 2.0).astype(
                np.float32
            )
            marginal = np.outer(epitope_score, paratope_score).astype(np.float32)
            ensemble_records.append(
                {
                    "complex_code": reference.complex_code,
                    "epitope_score": epitope_score,
                    "paratope_score": paratope_score,
                    "attention": attention,
                    "marginal": marginal,
                    "epitope_target": reference.epitope_target,
                    "paratope_target": reference.paratope_target,
                    "pair_target": reference.pair_target,
                }
            )
            path = ensemble_root / f"complex_{index:05d}.npz"
            np.savez_compressed(path, **ensemble_records[-1])
            file_digests[str(path.relative_to(output_root))] = sha256_file(path)

        epitope = residue_metrics(
            [record["epitope_target"] for record in ensemble_records],
            [record["epitope_score"] for record in ensemble_records],
        )
        paratope = residue_metrics(
            [record["paratope_target"] for record in ensemble_records],
            [record["paratope_score"] for record in ensemble_records],
        )
        pair_targets = [record["pair_target"] for record in ensemble_records]
        attention_pair = pair_metrics(
            pair_targets, [record["attention"] for record in ensemble_records]
        )
        marginal_pair = pair_metrics(
            pair_targets, [record["marginal"] for record in ensemble_records]
        )
        role_results[role] = {
            "aggregation": "unweighted_arithmetic_mean_of_five_fold_models_fixed_before_test_scoring",
            "epitope": epitope,
            "paratope": paratope,
            "pair_contact": {
                "native_mean_bidirectional_attention": attention_pair,
                "predicted_residue_outer_product": marginal_pair,
            },
            "published_joint_gap": {
                "epitope": {
                    key: epitope[key] - value
                    for key, value in PUBLISHED_JOINT["epitope"].items()
                },
                "paratope": {
                    key: paratope[key] - value
                    for key, value in PUBLISHED_JOINT["paratope"].items()
                },
            },
        }

    report = {
        "schema_version": "mipe-official-independent-test-evaluation-v1",
        "status": "complete",
        "evaluated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "selection_policy": {
            "checkpoint_roles": list(ROLES),
            "ensemble": "five_fold_unweighted_probability_mean",
            "test_label_driven_model_selection": False,
            "test_label_driven_threshold_selection": False,
        },
        "dataset": {
            "path": str(arguments.test_data.resolve()),
            "sha256": test_sha256,
            "records": len(dataset),
            "unique_complex_codes": len(set(code.lower() for code in complex_codes)),
            "complex_codes": complex_codes,
        },
        "run": {
            "root": str(arguments.run_root.resolve()),
            "track_id": authority.track_id,
            "global_selection_sha256": authority.global_selection_sha256,
            "cv_data_sha256": authority.data_sha256,
        },
        "published_joint_mipe_reference": PUBLISHED_JOINT,
        "per_fold": per_fold,
        "five_fold_ensemble": role_results,
        "frozen_prediction_files": file_digests,
        "implementation": {
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "python": sys.executable,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cuda_device": torch.cuda.get_device_name(device),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
    }
    json_path = output_root / "MIPE_OFFICIAL_INDEPENDENT_TEST.json"
    write_new(json_path, canonical_json(report))

    lines = [
        "# MIPE official independent-test evaluation",
        "",
        "The 64 provided `testdata.pkl` complexes were scored without test-driven model or threshold selection.",
        "",
        "| Checkpoints | Epitope AUROC | Epitope AUPRC | Epitope MCC@0.5 | Paratope AUROC | Paratope AUPRC | Paratope MCC@0.5 | Pair attention macro AUPRC | Pair marginal macro AUPRC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for role in ROLES:
        result = role_results[role]
        lines.append(
            f"| {role} | {result['epitope']['auroc']:.6f} | {result['epitope']['auprc']:.6f} | "
            f"{result['epitope']['mcc_at_0_5']:.6f} | {result['paratope']['auroc']:.6f} | "
            f"{result['paratope']['auprc']:.6f} | {result['paratope']['mcc_at_0_5']:.6f} | "
            f"{result['pair_contact']['native_mean_bidirectional_attention']['macro_auprc']:.6f} | "
            f"{result['pair_contact']['predicted_residue_outer_product']['macro_auprc']:.6f} |"
        )
    lines += [
        "",
        "Published joint MIPE reference: epitope AUROC/AUPRC/MCC 0.815/0.422/0.235; paratope 0.894/0.738/0.531.",
        "",
        "All individual fold and ensemble predictions are frozen under `predictions/` and bound by SHA-256 in the JSON report.",
    ]
    write_new(output_root / "MIPE_OFFICIAL_INDEPENDENT_TEST.md", ("\n".join(lines) + "\n").encode())
    print(json.dumps(role_results, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
