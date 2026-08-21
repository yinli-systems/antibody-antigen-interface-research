#!/usr/bin/env python3
"""Quantify label recoverability from bound-complex C-alpha coordinates."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def best_mcc_threshold(target: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    order = np.argsort(-score, kind="mergesort")
    target = target[order].astype(np.uint8)
    score = score[order].astype(np.float64)
    true_positive = np.cumsum(target)
    false_positive = np.cumsum(1 - target)
    positive = float(true_positive[-1])
    negative = float(false_positive[-1])
    indices = np.r_[np.where(np.diff(score) != 0)[0], len(score) - 1]
    tp = true_positive[indices].astype(float)
    fp = false_positive[indices].astype(float)
    fn = positive - tp
    tn = negative - fp
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    values = np.divide(
        tp * tn - fp * fn,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    best = int(np.argmax(values))
    index = int(indices[best])
    next_score = score[index + 1] if index + 1 < len(score) else -np.inf
    threshold = (
        float((score[index] + next_score) / 2)
        if np.isfinite(next_score)
        else float(np.nextafter(score[index], -np.inf))
    )
    return threshold, float(values[best])


def reciprocal_rank(target: np.ndarray, score: np.ndarray) -> float:
    order = np.argsort(-score.reshape(-1), kind="mergesort")
    hits = np.flatnonzero(target.reshape(-1)[order])
    return 0.0 if hits.size == 0 else float(1.0 / (int(hits[0]) + 1))


def pooled_metrics(records: list[dict[str, Any]], side: str, threshold: float | None = None) -> dict[str, Any]:
    target = np.concatenate([record[f"{side}_target"].reshape(-1) for record in records])
    score = np.concatenate([record[f"{side}_score"].reshape(-1) for record in records])
    result = {
        "count": int(target.size),
        "positives": int(target.sum()),
        "prevalence": float(target.mean()),
        "auroc": float(roc_auc_score(target, score)),
        "auprc": float(average_precision_score(target, score)),
    }
    if threshold is not None:
        result["mcc_at_cv_fitted_threshold"] = float(
            matthews_corrcoef(target, score >= threshold)
        )
        result["cv_fitted_score_threshold"] = float(threshold)
        result["equivalent_max_calpha_distance_angstrom"] = float(-threshold)
    return result


def pair_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    targets = [record["pair_target"] for record in records]
    scores = [record["pair_score"] for record in records]
    target = np.concatenate([item.reshape(-1) for item in targets])
    score = np.concatenate([item.reshape(-1) for item in scores])
    return {
        "complexes": len(records),
        "candidate_pairs": int(target.size),
        "positive_pairs": int(target.sum()),
        "prevalence": float(target.mean()),
        "macro_auprc": float(
            np.mean(
                [
                    average_precision_score(y.reshape(-1), s.reshape(-1))
                    for y, s in zip(targets, scores)
                ]
            )
        ),
        "pooled_auprc": float(average_precision_score(target, score)),
        "recall_at_1": float(
            np.mean([y.reshape(-1)[int(np.argmax(s))] for y, s in zip(targets, scores)])
        ),
        "mrr": float(np.mean([reciprocal_rank(y, s) for y, s in zip(targets, scores)])),
    }


def process_dataset(dataset: Any, split: str, output_root: Path) -> list[dict[str, Any]]:
    records = []
    split_root = output_root / "predictions" / split
    split_root.mkdir(parents=True)
    for index, sample in enumerate(dataset):
        antigen = np.asarray(sample["coord_AG"], dtype=np.float64)[:, -3:]
        antibody = np.asarray(sample["coord_AB"], dtype=np.float64)[:, -3:]
        distance = np.linalg.norm(antigen[:, None, :] - antibody[None, :, :], axis=2)
        pair_target = np.zeros(distance.shape, dtype=np.uint8)
        edges = sample["edge_AGAB"]
        pair_target[np.asarray(edges[0], dtype=int), np.asarray(edges[1], dtype=int)] = 1
        record = {
            "complex_code": str(sample["complex_code"]),
            "pair_score": (-distance).astype(np.float32),
            "pair_target": pair_target,
            "epitope_score": (-distance.min(axis=1)).astype(np.float32),
            "epitope_target": np.asarray(sample["label_AG"], dtype=np.uint8),
            "paratope_score": (-distance.min(axis=0)).astype(np.float32),
            "paratope_target": np.asarray(sample["label_AB"], dtype=np.uint8),
        }
        records.append(record)
        np.savez_compressed(split_root / f"complex_{index:05d}.npz", **record)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv-data", type=Path, required=True)
    parser.add_argument("--test-data", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args()
    output_root = arguments.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite {output_root}")
    output_root.mkdir(parents=True)

    torch.storage._load_from_bytes = lambda payload: torch.load(
        io.BytesIO(payload), map_location="cpu", weights_only=False
    )
    cv_dataset = pd.read_pickle(arguments.cv_data)
    test_dataset = pd.read_pickle(arguments.test_data)
    cv_codes = {str(sample["complex_code"]).lower() for sample in cv_dataset}
    test_codes = {str(sample["complex_code"]).lower() for sample in test_dataset}
    if cv_codes & test_codes:
        raise ValueError("CV and independent-test complex codes overlap")

    cv_records = process_dataset(cv_dataset, "cv", output_root)
    test_records = process_dataset(test_dataset, "independent_test", output_root)
    thresholds = {}
    cv_metrics = {}
    test_metrics = {}
    for side in ("epitope", "paratope"):
        cv_target = np.concatenate([record[f"{side}_target"] for record in cv_records])
        cv_score = np.concatenate([record[f"{side}_score"] for record in cv_records])
        threshold, cv_mcc = best_mcc_threshold(cv_target, cv_score)
        thresholds[side] = {"score_threshold": threshold, "cv_mcc": cv_mcc}
        cv_metrics[side] = pooled_metrics(cv_records, side, threshold)
        test_metrics[side] = pooled_metrics(test_records, side, threshold)
    cv_metrics["pair_contact"] = pair_metrics(cv_records)
    test_metrics["pair_contact"] = pair_metrics(test_records)

    file_digests = {}
    for path in sorted((output_root / "predictions").rglob("*.npz")):
        file_digests[str(path.relative_to(output_root))] = sha256_file(path)
    report = {
        "schema_version": "mipe-bound-geometry-diagnostic-v1",
        "status": "complete",
        "evaluated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "claim_boundary": (
            "Bound-complex coordinate oracle diagnostic only. It directly uses the common "
            "antibody-antigen coordinate frame and is not a deployable unbound or sequence-only predictor."
        ),
        "score_definition": "negative minimum inter-chain C-alpha Euclidean distance",
        "cv_threshold_selection_only": thresholds,
        "datasets": {
            "cv": {"path": str(arguments.cv_data.resolve()), "sha256": sha256_file(arguments.cv_data), "records": len(cv_records)},
            "independent_test": {"path": str(arguments.test_data.resolve()), "sha256": sha256_file(arguments.test_data), "records": len(test_records)},
            "exact_complex_code_overlap": [],
        },
        "cv": cv_metrics,
        "independent_test": test_metrics,
        "prediction_files": file_digests,
        "implementation": {"script": str(Path(__file__).resolve()), "script_sha256": sha256_file(Path(__file__).resolve())},
    }
    (output_root / "MIPE_BOUND_GEOMETRY_DIAGNOSTIC.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    lines = [
        "# MIPE bound-geometry diagnostic",
        "",
        report["claim_boundary"],
        "",
        "| Split | Epitope AUROC | Epitope AUPRC | Epitope MCC | Paratope AUROC | Paratope AUPRC | Paratope MCC | Pair macro AUPRC | Pair pooled AUPRC | Recall@1 | MRR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, metrics in (("CV", cv_metrics), ("Independent test", test_metrics)):
        lines.append(
            f"| {label} | {metrics['epitope']['auroc']:.6f} | {metrics['epitope']['auprc']:.6f} | {metrics['epitope']['mcc_at_cv_fitted_threshold']:.6f} | "
            f"{metrics['paratope']['auroc']:.6f} | {metrics['paratope']['auprc']:.6f} | {metrics['paratope']['mcc_at_cv_fitted_threshold']:.6f} | "
            f"{metrics['pair_contact']['macro_auprc']:.6f} | {metrics['pair_contact']['pooled_auprc']:.6f} | "
            f"{metrics['pair_contact']['recall_at_1']:.6f} | {metrics['pair_contact']['mrr']:.6f} |"
        )
    (output_root / "MIPE_BOUND_GEOMETRY_DIAGNOSTIC.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(report["independent_test"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
