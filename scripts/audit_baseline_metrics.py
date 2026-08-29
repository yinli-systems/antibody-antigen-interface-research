#!/usr/bin/env python3
"""Audit the locally available baseline receipts without re-selecting a model.

The script intentionally derives thresholded MIPE metrics from the released
confusion counts.  It never reads a frozen/external test prediction to choose a
checkpoint or threshold and emits the provenance needed for a reproducible
baseline table.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


# The research repository lives at ``<workspace>/work/<repo>``.  Keep the
# workspace root explicit so the audit can be rerun from any current directory.
ROOT = Path(__file__).resolve().parents[3]
MIPE_JSON = (
    ROOT
    / "mipe-sppider-handoff/mipe-sppider-handoff-20260729-v2/MIPE/results"
    / "track_a_eval_e7d190_20260725T035346/results/MIPE_HELDOUT_EVALUATION.json"
)
EPIFORMER_SCORE = ROOT / "work/antibody-antigen-interface-research/results/epiformer/score.json"
EPIFORMER_PROVENANCE = (
    ROOT / "work/antibody-antigen-interface-research/results/epiformer/provenance.json"
)
EPIFORMER_PREFLIGHT = (
    ROOT / "work/antibody-antigen-interface-research/results/epiformer/preflight.json"
)
EPIFORMER_VERIFICATION = (
    ROOT / "work/antibody-antigen-interface-research/results/epiformer/verification.json"
)
SPPIDER_DIR = (
    ROOT
    / "mipe-sppider-handoff/mipe-sppider-handoff-20260729-v2/SPPIDER/results"
    / "final_v3_ensemble_seed42"
)
SPPIDER_EPITOPE_PREDICTIONS = SPPIDER_DIR / "epitope/final/residue_predictions.parquet"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def threshold_metrics(counts: dict[str, int]) -> dict[str, float]:
    tp = counts["true_positives"]
    fp = counts["false_positives"]
    fn = counts["false_negatives"]
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2.0 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def recover_micro_confusion(
    *, positive_count: int, residue_count: int, f1: float, mcc: float
) -> dict[str, int]:
    """Recover the unique integer confusion matrix encoded by score.json.

    EpiFormer's SAbDab2 receipt stores micro F1/MCC and the total positive and
    residue counts, but not P/R or the four counts.  The scorer computes F1 and
    MCC from integer counts, so solving the two equations gives the exact
    counts.  The residual check makes this fail closed if a future receipt is
    rounded or changes schema.
    """
    positives = positive_count
    negatives = residue_count - positive_count
    candidates: list[tuple[float, dict[str, int]]] = []
    for tp in range(positives + 1):
        if f1 == 0.0:
            continue
        fp = round(2.0 * tp / f1 - tp - positives)
        if not 0 <= fp <= negatives:
            continue
        fn = positives - tp
        tn = negatives - fp
        actual_f1 = 2.0 * tp / (tp + fp + positives)
        denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
        actual_mcc = (tp * tn - fp * fn) / denominator if denominator else 0.0
        residual = abs(actual_f1 - f1) + abs(actual_mcc - mcc)
        if residual < 1e-10:
            candidates.append(
                (
                    residual,
                    {
                        "true_positives": tp,
                        "false_positives": fp,
                        "false_negatives": fn,
                        "true_negatives": tn,
                    },
                )
            )
    if len(candidates) != 1:
        raise ValueError(
            "EpiFormer score receipt does not encode a unique integer confusion matrix"
        )
    return candidates[0][1]


def epiformer_rows() -> list[dict[str, Any]]:
    score = json.loads(EPIFORMER_SCORE.read_text())
    provenance = json.loads(EPIFORMER_PROVENANCE.read_text())
    preflight = json.loads(EPIFORMER_PREFLIGHT.read_text())
    verification = json.loads(EPIFORMER_VERIFICATION.read_text())
    micro = score["micro"]
    # The two frozen SAbDab2 epitope receipts can be compared at the exact
    # instance level.  This is deliberately an identity audit, not a model
    # selection step.
    from pyarrow.parquet import read_table

    epiformer_ids = {item["instance_id"] for item in score["per_instance"]}
    sppider_table = read_table(SPPIDER_EPITOPE_PREDICTIONS, columns=["instance_id"])
    sppider_ids = set(sppider_table["instance_id"].to_pylist())
    if epiformer_ids != sppider_ids:
        raise ValueError("EpiFormer and SPPIDER frozen test instance IDs differ")
    counts = recover_micro_confusion(
        positive_count=micro["positive_count"],
        residue_count=micro["residue_count"],
        f1=micro["f1_at_threshold"],
        mcc=micro["mcc_at_threshold"],
    )
    derived = threshold_metrics(counts)
    return [
        {
            "source": "EpiFormer",
            "scope": "frozen SAbDab2 test (455 complexes; epitope projection only)",
            "role": "frozen_test",
            "target": "epitope",
            "auroc": micro["auroc"],
            "auprc": micro["auprc"],
            "precision": derived["precision"],
            "recall": derived["recall"],
            "f1": derived["f1"],
            "mcc": micro["mcc_at_threshold"],
            "threshold": provenance["classification_threshold"],
            "threshold_scope": "frozen threshold from formal SAbDab2 receipt",
            "test_instance_count": score["instance_count"],
            "rows_verified": verification["rows_verified"],
            "mapping_failures": preflight["failure_count"],
            "native_pair_interaction_score": False,
            "derived_confusion_counts": counts,
            "shared_with_sppider_epitope_test": True,
            "shared_instance_count": len(epiformer_ids),
        }
    ]


def mipe_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for role, role_payload in payload["role_results"].items():
        for target in ("epitope", "paratope"):
            summary = role_payload["summary"]["marginal"][target]["micro"]
            counts = summary["mcc_at_fold_specific_validation_fitted_threshold"]
            derived = threshold_metrics(counts)
            rows.append(
                {
                    "source": "MIPE",
                    "scope": "official five-fold held-out SAbDab2 complexes",
                    "role": role,
                    "target": target,
                    "auroc": summary["ranking"]["auroc"],
                    "auprc": summary["ranking"]["auprc"],
                    "precision": derived["precision"],
                    "recall": derived["recall"],
                    "f1": derived["f1"],
                    "mcc": counts["mcc"],
                    "threshold_scope": "fold-specific validation-fitted threshold",
                    "tp": counts["true_positives"],
                    "fp": counts["false_positives"],
                    "fn": counts["false_negatives"],
                    "tn": counts["true_negatives"],
                }
            )
    return rows


def sppider_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target in ("epitope", "paratope"):
        path = SPPIDER_DIR / target / "final/score.json"
        payload = json.loads(path.read_text())
        metrics = payload["residue_micro"]
        rows.append(
            {
                "source": "SPPIDER-seq",
                "scope": "frozen SAbDab2 test (455 complexes)",
                "role": "frozen_test",
                "target": target,
                "auroc": metrics["auroc"],
                "auprc": metrics["auprc"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "mcc": metrics["mcc"],
                "threshold_scope": "validation-selected threshold",
                "threshold": metrics["threshold"],
                "test_instance_count": payload["test_instance_count"],
                "native_pair_interaction_score": payload[
                    "native_pair_interaction_score"
                ],
            }
        )
    return rows


def main() -> None:
    mipe_payload = json.loads(MIPE_JSON.read_text())
    output = {
        "generated_by": str(Path(__file__).relative_to(ROOT)),
        "mipe_claim_scope": mipe_payload["scope"],
        "mipe_status": mipe_payload["status"],
        "mipe_external_test_accessed": mipe_payload["execution_facts"][
            "provided_external_test_accessed"
        ],
        "rows": epiformer_rows() + mipe_rows(mipe_payload) + sppider_rows(),
        "receipts": {
            "epiformer_score": {
                "path": str(EPIFORMER_SCORE.relative_to(ROOT)),
                "sha256": sha256(EPIFORMER_SCORE),
            },
            "epiformer_provenance": {
                "path": str(EPIFORMER_PROVENANCE.relative_to(ROOT)),
                "sha256": sha256(EPIFORMER_PROVENANCE),
            },
            "epiformer_preflight": {
                "path": str(EPIFORMER_PREFLIGHT.relative_to(ROOT)),
                "sha256": sha256(EPIFORMER_PREFLIGHT),
            },
            "epiformer_verification": {
                "path": str(EPIFORMER_VERIFICATION.relative_to(ROOT)),
                "sha256": sha256(EPIFORMER_VERIFICATION),
            },
            "mipe_heldout_json": {
                "path": str(MIPE_JSON.relative_to(ROOT)),
                "sha256": sha256(MIPE_JSON),
            },
            "sppider_epitope_score": {
                "path": str(
                    (SPPIDER_DIR / "epitope/final/score.json").relative_to(ROOT)
                ),
                "sha256": sha256(SPPIDER_DIR / "epitope/final/score.json"),
            },
            "sppider_paratope_score": {
                "path": str(
                    (SPPIDER_DIR / "paratope/final/score.json").relative_to(ROOT)
                ),
                "sha256": sha256(SPPIDER_DIR / "paratope/final/score.json"),
            },
            "sppider_epitope_predictions": {
                "path": str(SPPIDER_EPITOPE_PREDICTIONS.relative_to(ROOT)),
                "sha256": sha256(SPPIDER_EPITOPE_PREDICTIONS),
            },
        },
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
