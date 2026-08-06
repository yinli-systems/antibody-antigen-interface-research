#!/usr/bin/env python3
"""Paired descriptive comparison of the two frozen SAbDab2 epitope receipts.

This is an analysis-only script: it uses the already frozen validation-selected
thresholds and never selects a checkpoint or threshold from the test labels.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as parquet
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[3]
EPIFORMER_SCORE = ROOT / "work/antibody-antigen-interface-research/results/epiformer/score.json"
SPPIDER_SCORE = (
    ROOT
    / "mipe-sppider-handoff/mipe-sppider-handoff-20260729-v2/SPPIDER/results"
    / "final_v3_ensemble_seed42/epitope/final/score.json"
)
SPPIDER_PREDICTIONS = (
    ROOT
    / "mipe-sppider-handoff/mipe-sppider-handoff-20260729-v2/SPPIDER/results"
    / "final_v3_ensemble_seed42/epitope/final/residue_predictions.parquet"
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    e = json.loads(EPIFORMER_SCORE.read_text())
    s = json.loads(SPPIDER_SCORE.read_text())
    threshold = s["threshold"]
    e_rows = {row["instance_id"]: row for row in e["per_instance"]}
    table = parquet.read_table(SPPIDER_PREDICTIONS).to_pydict()
    groups: dict[str, dict[str, list[float]]] = {}
    for instance_id, label, score in zip(
        table["instance_id"], table["label"], table["score"]
    ):
        groups.setdefault(instance_id, {"label": [], "score": []})["label"].append(
            int(label)
        )
        groups[instance_id]["score"].append(float(score))
    if set(groups) != set(e_rows):
        raise ValueError("frozen EpiFormer/SPPIDER test instance IDs do not match")

    paired: list[dict[str, float | str]] = []
    for instance_id, values in groups.items():
        y = np.asarray(values["label"], dtype=np.int64)
        scores = np.asarray(values["score"], dtype=np.float64)
        pred = (scores >= threshold).astype(np.int64)
        paired.append(
            {
                "instance_id": instance_id,
                "epiformer_auprc": e_rows[instance_id]["auprc"],
                "sppider_auprc": average_precision_score(y, scores),
                "epiformer_auroc": e_rows[instance_id]["auroc"],
                "sppider_auroc": roc_auc_score(y, scores),
                "epiformer_f1": e_rows[instance_id]["f1_at_threshold"],
                "sppider_f1": f1_score(y, pred, zero_division=0),
                "epiformer_mcc": e_rows[instance_id]["mcc_at_threshold"],
                "sppider_mcc": matthews_corrcoef(y, pred),
                "sppider_precision": precision_score(y, pred, zero_division=0),
                "sppider_recall": recall_score(y, pred, zero_division=0),
            }
        )

    n = len(paired)
    rng = np.random.default_rng(20260806)
    bootstrap_indices = rng.integers(0, n, size=(10_000, n))
    summary: dict[str, dict[str, float]] = {}
    for metric in ("auprc", "auroc", "f1", "mcc"):
        e_values = np.asarray([row[f"epiformer_{metric}"] for row in paired])
        s_values = np.asarray([row[f"sppider_{metric}"] for row in paired])
        deltas = e_values - s_values
        bootstrap_means = deltas[bootstrap_indices].mean(axis=1)
        summary[metric] = {
            "epiformer_macro_mean": float(e_values.mean()),
            "sppider_macro_mean": float(s_values.mean()),
            "paired_delta_mean": float(deltas.mean()),
            "paired_delta_median": float(np.median(deltas)),
            "fraction_complexes_epiformer_better": float(np.mean(deltas > 0)),
            "paired_bootstrap_ci95_low": float(np.quantile(bootstrap_means, 0.025)),
            "paired_bootstrap_ci95_high": float(np.quantile(bootstrap_means, 0.975)),
        }

    print(
        json.dumps(
            {
                "scope": "paired descriptive analysis; shared frozen SAbDab2 epitope test",
                "instance_count": n,
                "epiformer_threshold": 0.3,
                "sppider_threshold": threshold,
                "bootstrap": {"resamples": 10_000, "seed": 20260806},
                "summary": summary,
                "receipts": {
                    "epiformer_score_sha256": digest(EPIFORMER_SCORE),
                    "sppider_score_sha256": digest(SPPIDER_SCORE),
                    "sppider_predictions_sha256": digest(SPPIDER_PREDICTIONS),
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
