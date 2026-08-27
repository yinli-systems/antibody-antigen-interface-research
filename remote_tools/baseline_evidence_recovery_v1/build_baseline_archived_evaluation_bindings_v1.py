#!/usr/bin/env python3
"""Bind archived baseline evaluation summaries without rescoring held-out data.

The output is deliberately data-only.  It records the exact source summary,
metric aggregation, threshold policy, and split identity for each archived
baseline.  Raw predictions and labels are never opened by this script.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


EVIDENCE = Path(
    "/ssd/scxi253/model_deployment/results/development/"
    "baseline_training_evidence_v2/BASELINE_TRAINING_EVIDENCE_V1.json"
)
EVIDENCE_RECEIPT = EVIDENCE.with_name("BASELINE_TRAINING_EVIDENCE_V1_RECEIPT.json")
EPIFORMER = Path(
    "/ssd/scxi253/model_deployment/data/results/epiformer/checkpoints/"
    "formal_epi_group_seed42/summary/epiformer_test_results.csv"
)
MIPE = Path(
    "/ssd/scxi253/model_deployment/results/mipe/"
    "track_a_eval_e7d190_20260725T035346/results/MIPE_HELDOUT_EVALUATION.json"
)
SPPIDER = Path(
    "/ssd/scxi253/model_deployment/results/sabdab2/sppider_seq/"
    "final_v3_ensemble_seed42/reports/final_research_summary.json"
)
DISCOTOPE_BINDING = Path(
    "/ssd/scxi253/model_deployment/results/development/discotope3_official_binding_v1/"
    "DISCOTOPE3_OFFICIAL_BINDING_V1.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def finite_float(value: Any, location: str) -> float:
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        raise ValueError(f"{location}: non-finite value")
    return number


def mipe_role(role: dict[str, Any]) -> dict[str, Any]:
    summary = role["summary"]
    result: dict[str, Any] = {
        "complexes": int(summary["complexes"]),
        "folds": [int(value) for value in summary["folds"]],
        "bootstrap": summary["bootstrap"],
        "tasks": {},
    }
    for task in ("epitope", "paratope"):
        marginal = summary["marginal"][task]
        ranking = marginal["micro"]["ranking"]
        macro = marginal["per_complex_macro"]
        result["tasks"][task] = {
            "micro": {
                "auroc": finite_float(ranking["auroc"], f"MIPE.{task}.auroc"),
                "average_precision": finite_float(
                    ranking["auprc"], f"MIPE.{task}.average_precision"
                ),
                "items": int(ranking["items"]),
                "positives": int(ranking["positives"]),
                "prevalence": finite_float(ranking["prevalence"], f"MIPE.{task}.prevalence"),
                "mcc_at_released_default_0_5": finite_float(
                    marginal["micro"]["mcc_at_released_default_0_5"]["mcc"],
                    f"MIPE.{task}.mcc_at_0_5",
                ),
                "mcc_at_fold_specific_validation_fitted_threshold": finite_float(
                    marginal["micro"]["mcc_at_fold_specific_validation_fitted_threshold"]["mcc"],
                    f"MIPE.{task}.mcc_at_fitted",
                ),
            },
            "per_complex_macro": {
                "average_precision_mean": finite_float(
                    macro["auprc"]["mean"], f"MIPE.{task}.macro_ap"
                ),
                "average_precision_ci95": [
                    finite_float(value, f"MIPE.{task}.macro_ap_ci")
                    for value in macro["auprc"]["ci"]
                ],
                "eligible_complexes": int(macro["auprc"]["items"]),
                "auroc_mean": finite_float(macro["auroc"]["mean"], f"MIPE.{task}.macro_auroc"),
                "auroc_ci95": [
                    finite_float(value, f"MIPE.{task}.macro_auroc_ci")
                    for value in macro["auroc"]["ci"]
                ],
            },
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()

    required = [EVIDENCE, EVIDENCE_RECEIPT, EPIFORMER, MIPE, SPPIDER, DISCOTOPE_BINDING]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing immutable source summaries: {missing}")

    with EPIFORMER.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"EpiFormer summary must contain one row, got {len(rows)}")
    epi = rows[0]
    mipe = json.loads(MIPE.read_text(encoding="utf-8"))
    sppider = json.loads(SPPIDER.read_text(encoding="utf-8"))
    discotope = json.loads(DISCOTOPE_BINDING.read_text(encoding="utf-8"))

    if mipe["status"] != "complete_create_only_without_policy_refit":
        raise ValueError(f"unexpected MIPE status: {mipe['status']}")
    if sppider["protocol"]["final_test_opened_after_model_and_threshold_selection"] is not True:
        raise ValueError("SPPIDER final summary is not bound to post-selection evaluation")
    if discotope["status"] != "PASS_OFFICIAL_SOURCE_AND_100_MODEL_ENSEMBLE_BOUND":
        raise ValueError(f"unexpected DiscoTope binding status: {discotope['status']}")

    payload: dict[str, Any] = {
        "schema_version": "baseline-archived-evaluation-bindings-v1",
        "status": "PASS_ARCHIVED_SUMMARY_BINDINGS_NO_NEW_SCORING",
        "access": {
            "raw_predictions_read": False,
            "raw_heldout_labels_read": False,
            "new_scoring_performed": False,
            "old_sabdab2_test_opened": False,
            "temporal_holdout_opened": False,
            "model_selection_performed": False,
        },
        "training_curve_evidence": {
            "result_path": str(EVIDENCE),
            "result_sha256": sha256(EVIDENCE),
            "receipt_path": str(EVIDENCE_RECEIPT),
            "receipt_sha256": sha256(EVIDENCE_RECEIPT),
        },
        "methods": {
            "epiformer": {
                "source_summary": {"path": str(EPIFORMER), "sha256": sha256(EPIFORMER)},
                "split_identity": "archived_internal_epiformer_epitope_group_test",
                "same_instances_as_current_framecontact_2565_549_protocol": False,
                "same_metric_implementation_as_current_framecontact_protocol": False,
                "seed": 42,
                "aggregation": "pooled_residue",
                "threshold_policy": "archived_summary_policy_not_recomputed",
                "tasks": {
                    "epitope": {
                        "auroc": finite_float(epi["epitope_auc"], "EpiFormer.epitope.auroc"),
                        "average_precision": finite_float(
                            epi["epitope_auprc"], "EpiFormer.epitope.average_precision"
                        ),
                        "f1": finite_float(epi["epitope_f1"], "EpiFormer.epitope.f1"),
                        "mcc": finite_float(epi["epitope_mcc"], "EpiFormer.epitope.mcc"),
                        "precision": finite_float(
                            epi["epitope_precision"], "EpiFormer.epitope.precision"
                        ),
                        "recall": finite_float(epi["epitope_recall"], "EpiFormer.epitope.recall"),
                    },
                    "paratope": {
                        "auroc": finite_float(epi["paratope_auc"], "EpiFormer.paratope.auroc"),
                        "average_precision": finite_float(
                            epi["paratope_auprc"], "EpiFormer.paratope.average_precision"
                        ),
                        "f1": finite_float(epi["paratope_f1"], "EpiFormer.paratope.f1"),
                        "mcc": finite_float(epi["paratope_mcc"], "EpiFormer.paratope.mcc"),
                        "precision": finite_float(
                            epi["paratope_precision"], "EpiFormer.paratope.precision"
                        ),
                        "recall": finite_float(epi["paratope_recall"], "EpiFormer.paratope.recall"),
                    },
                },
            },
            "mipe": {
                "source_summary": {"path": str(MIPE), "sha256": sha256(MIPE)},
                "split_identity": "authenticated_mipe_track_a_heldout_560",
                "same_instances_as_current_framecontact_2565_549_protocol": False,
                "same_metric_implementation_as_current_framecontact_protocol": False,
                "evaluation_track": mipe["evaluation_track"],
                "selection_performed_during_evaluation": bool(
                    mipe["execution_facts"]["model_or_checkpoint_selection_performed"]
                ),
                "policy_refit_performed": bool(mipe["execution_facts"]["policy_refit_performed"]),
                "roles": {
                    "primary_epoch_800": mipe_role(mipe["role_results"]["primary_epoch_800"]),
                    "secondary_validation_best": mipe_role(
                        mipe["role_results"]["secondary_validation_best"]
                    ),
                },
            },
            "sppider_seq": {
                "source_summary": {"path": str(SPPIDER), "sha256": sha256(SPPIDER)},
                "split_identity": "frozen_sabdab2_2727_481_455_antigen_cluster_split",
                "same_instances_as_current_framecontact_2565_549_protocol": False,
                "same_metric_implementation_as_current_framecontact_protocol": False,
                "seed": 42,
                "split_counts": {
                    "train": int(sppider["protocol"]["train_instance_count"]),
                    "validation": int(sppider["protocol"]["validation_instance_count"]),
                    "test": int(sppider["protocol"]["canonical_test_instance_count"]),
                    "test_antigen_clusters": int(
                        sppider["protocol"]["canonical_test_cluster_count"]
                    ),
                },
                "test_opened_after_model_and_threshold_selection": True,
                "aggregation": "pooled_residue_micro",
                "tasks": {
                    task: {
                        "auroc": finite_float(values["final_micro"]["auroc"], f"SPPIDER.{task}.auroc"),
                        "average_precision": finite_float(
                            values["final_micro"]["auprc"], f"SPPIDER.{task}.average_precision"
                        ),
                        "f1_at_validation_selected_threshold": finite_float(
                            values["final_micro"]["f1"], f"SPPIDER.{task}.f1"
                        ),
                        "mcc_at_validation_selected_threshold": finite_float(
                            values["final_micro"]["mcc"], f"SPPIDER.{task}.mcc"
                        ),
                        "validation_selected_threshold": finite_float(
                            values["final_micro"]["threshold"], f"SPPIDER.{task}.threshold"
                        ),
                        "brier": finite_float(values["calibration"]["brier_score"], f"SPPIDER.{task}.brier"),
                        "equal_mass_ece": finite_float(
                            values["calibration"]["equal_mass_ece"], f"SPPIDER.{task}.ece"
                        ),
                    }
                    for task, values in sppider["roles"].items()
                },
            },
            "discotope_3": {
                "official_repository": "https://github.com/Magnushhoie/DiscoTope-3.0",
                "official_paper_doi": "10.3389/fimmu.2024.1322712",
                "official_binding": {
                    "path": str(DISCOTOPE_BINDING),
                    "sha256": sha256(DISCOTOPE_BINDING),
                    "revision": discotope["revision"],
                    "source_archive_sha256": discotope["source_archive"]["sha256"],
                    "models_zip_sha256": discotope["models_zip"]["sha256"],
                    "xgboost_member_count": discotope["models_zip"]["xgboost_member_count"],
                },
                "training_curve_applicable": False,
                "released_model_type": "fitted_xgboost_ensemble",
                "official_source_revision_bound_locally": True,
                "official_models_zip_sha256_bound_locally": True,
                "same_protocol_heldout_evaluation_available": False,
                "status": "PASS_OFFICIAL_ARTIFACTS_BOUND_INFERENCE_NOT_RUN",
            },
        },
        "comparability": {
            "single_unified_same_instance_metric_table_authorized": False,
            "reasons": {
                "split_identity_equal": False,
                "instance_identity_equal": False,
                "aggregation_equal": False,
                "threshold_policy_equal": False,
            },
            "per_method_archived_reporting_authorized": True,
        },
    }

    output = args.output_root.resolve()
    result_path = output / "BASELINE_ARCHIVED_EVALUATION_BINDINGS_V1.json"
    receipt_path = output / "BASELINE_ARCHIVED_EVALUATION_BINDINGS_V1_RECEIPT.json"
    write_json(result_path, payload)
    receipt = {
        "schema_version": "baseline-archived-evaluation-bindings-v1-receipt",
        "status": "PASS",
        "generator_script": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
            "bytes": Path(__file__).resolve().stat().st_size,
        },
        "result": {
            "path": str(result_path),
            "sha256": sha256(result_path),
            "bytes": result_path.stat().st_size,
        },
        "source_summaries": {
            str(path): {"sha256": sha256(path), "bytes": path.stat().st_size}
            for path in required
        },
        "access": payload["access"],
    }
    write_json(receipt_path, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
