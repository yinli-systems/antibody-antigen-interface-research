#!/usr/bin/env python3
"""Compare contact-map marginal operators without retraining a model."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from interaction_first.metrics import average_precision, roc_auc


METHODS = ("mean", "max", "top2", "top3", "top5", "top1pct", "noisy_or")


def _node_scores(frame: Any, residue_column: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Aggregate one complex into node labels and alternative marginal scores."""
    records: dict[str, list[float]] = {name: [] for name in METHODS}
    labels: list[int] = []
    for _, group in frame.groupby(residue_column, sort=False, observed=True):
        scores = group["score"].to_numpy(dtype=np.float64)
        labels.append(int(group["label"].max()))
        ordered = np.sort(scores)[::-1]
        records["mean"].append(float(scores.mean()))
        records["max"].append(float(ordered[0]))
        for count in (2, 3, 5):
            records[f"top{count}"].append(float(ordered[: min(count, len(ordered))].mean()))
        fraction_count = max(1, int(math.ceil(0.01 * len(ordered))))
        records["top1pct"].append(float(ordered[:fraction_count].mean()))
        clipped = np.clip(scores, 1e-12, 1.0 - 1e-12)
        log_survival = float(np.log1p(-clipped).sum())
        records["noisy_or"].append(float(-np.expm1(max(log_survival, -50.0))))
    truth = np.asarray(labels, dtype=np.uint8)
    return {
        name: (truth, np.asarray(values, dtype=np.float64))
        for name, values in records.items()
    }


def _summarize(frame: Any, residue_column: str) -> dict[str, dict[str, float | None]]:
    """Compute micro and per-complex macro ranking metrics for each operator."""
    pooled: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
        name: [] for name in METHODS
    }
    macro: dict[str, dict[str, list[float]]] = {
        name: {"auprc": [], "auroc": []} for name in METHODS
    }
    for _, complex_frame in frame.groupby("instance_id", sort=False, observed=True):
        scores_by_method = _node_scores(complex_frame, residue_column)
        for name, (truth, scores) in scores_by_method.items():
            pooled[name].append((truth, scores))
            ap = average_precision(truth, scores)
            auc = roc_auc(truth, scores)
            if ap is not None:
                macro[name]["auprc"].append(float(ap))
            if auc is not None:
                macro[name]["auroc"].append(float(auc))

    result = {}
    for name in METHODS:
        truth = np.concatenate([item[0] for item in pooled[name]])
        scores = np.concatenate([item[1] for item in pooled[name]])
        result[name] = {
            "micro_auprc": average_precision(truth, scores),
            "micro_auroc": roc_auc(truth, scores),
            "macro_auprc": float(np.mean(macro[name]["auprc"])),
            "macro_auroc": float(np.mean(macro[name]["auroc"])),
            "node_count": int(len(truth)),
            "positive_count": int(truth.sum()),
        }
    return result


def main() -> None:
    """Run the post-hoc development-set marginal audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.read_parquet(args.pair_predictions)
    required = {"instance_id", "ag_res_idx", "ab_res_idx", "label", "score"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Pair predictions are missing columns: {sorted(missing)}")
    if frame.empty or not np.isfinite(frame["score"].to_numpy(dtype=np.float64)).all():
        raise ValueError("Pair predictions must contain finite scores")

    result = {
        "analysis": "post_hoc_development_marginal_ablation",
        "pair_prediction_path": str(args.pair_predictions.resolve()),
        "instance_count": int(frame["instance_id"].nunique()),
        "epitope": _summarize(frame, "ag_res_idx"),
        "paratope": _summarize(frame, "ab_res_idx"),
        "limitations": [
            "This analysis reuses a previously inspected development validation set.",
            "It compares score aggregation only and is not evidence from a retrained model.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
