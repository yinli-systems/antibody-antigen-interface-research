"""Evidence tests for the PAIRScale e400 development holdout result."""

from __future__ import annotations

import json
from pathlib import Path


RESULT_PATH = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "pairscale"
    / "holdout256_pair_c220e77_e400.result.json"
)
STATISTICS_PATH = RESULT_PATH.with_name(
    "holdout256_pair_c220e77_e400.cluster_statistics.json"
)


def _result() -> dict[str, object]:
    return json.loads(RESULT_PATH.read_text(encoding="utf-8"))


def _statistics() -> dict[str, object]:
    return json.loads(STATISTICS_PATH.read_text(encoding="utf-8"))


def test_e400_result_binds_completed_cluster_run_and_artifacts() -> None:
    result = _result()
    assert result["status"] == (
        "completed_train_l0_passed_holdout_partner_specificity_failed_no_selection"
    )
    assert result["cluster"] == {
        "allocated_cpus": 16,
        "allocated_gpus": 1,
        "allocated_memory_gib": 64,
        "elapsed": "01:17:52",
        "exit_code": "0:0",
        "job_id": 1332891,
        "max_rss_kib": 2922688,
        "state": "COMPLETED",
    }
    artifacts = result["artifacts"]
    assert artifacts["checkpoint_generation"] == "epoch-0260-step-00049920"
    assert artifacts["checkpoint_sha256"] == (
        "7208e2a1e3c0870bc20cb2e037394312957ddac0a38badd7800c053c673e47b9"
    )
    assert artifacts["final_report_sha256"] == (
        "6f68e26d927e56a25e4ffd0379ca87293ca34d9cbd907ea1c829348a858dfc28"
    )
    assert artifacts["holdout_evaluation_sha256"] == (
        "d43c46cb887f6b68387e1e9076e79cc76094cb1a567c633676d6313cc7f77b13"
    )


def test_e400_result_records_train_fit_and_endpoint_disjoint_failure() -> None:
    result = _result()
    train = result["train_evaluation"]
    holdout = result["holdout_evaluation"]
    assert train["train_l0_gate_passed"] is True
    assert train["completed_epoch"] == 260
    assert train["macro_pair_auprc"] == 0.9705578256152209
    assert train["matched_partner_topk_win_rate"] == 1.0
    assert holdout["instance_count"] == 64
    assert holdout["gate_passed"] is False
    assert holdout["macro_pair_auprc"] == 0.14165895209956433
    assert holdout["macro_label_marginal_oracle_auprc"] == 0.1397762833247823
    assert holdout["matched_partner_topk_win_rate"] == 0.546875
    assert holdout["double_centered_partner_win_rate"] == 0.515625


def test_e400_result_forbids_selection_scaling_and_broad_claims() -> None:
    result = _result()
    evidence = result["holdout_evidence"]
    decision = result["scientific_decision"]
    assert evidence["access_condition"] == "train_only_l0_gate_passed"
    assert evidence["access_count_within_runner"] == 1
    assert evidence["model_selection_authorized"] is False
    assert evidence["checkpoint_promotion_authorized"] is False
    assert evidence["confirmatory_claim_authorized"] is False
    assert decision["model_scaling_authorized"] is False
    assert decision["next_required_evidence"] == [
        "antibody_only_control",
        "antigen_only_control",
        "paired_cluster_bootstrap",
        "strongest_separable_marginal_control",
    ]
    assert result["claim_boundary"] == (
        "Development-only endpoint-disjoint evidence. No validation, test, "
        "confirmatory, checkpoint-promotion, biological-generalization, "
        "comparative, or state-of-the-art claim is authorized."
    )


def test_e400_cluster_statistics_record_the_non_authorizing_inference() -> None:
    statistics = _statistics()
    assert statistics["bootstrap"] == {
        "comparison_count": 4,
        "confidence": 0.95,
        "independent_component_count": 58,
        "instance_count": 64,
        "iterations": 10000,
        "seed": 42,
    }
    comparisons = statistics["comparisons"]
    model_marginal = comparisons[
        "pair_head_minus_model_marginal_outer_auprc"
    ]
    label_oracle = comparisons[
        "pair_head_minus_label_marginal_oracle_auprc"
    ]
    assert model_marginal["mean_difference"] == 0.08090246090844788
    assert model_marginal["lower"] == 0.021880790496643328
    assert label_oracle["mean_difference"] == 0.0018826687747820203
    assert label_oracle["lower"] == -0.08891620114828451
    assert statistics["gate_l1"]["checks"] == {
        "double_centered_partner_lower_bound_above_zero": False,
        "pair_head_exceeds_label_marginal_oracle_lower_bound_above_zero": False,
        "pair_head_exceeds_model_marginal_lower_bound_above_zero": True,
        "topk_partner_lower_bound_above_zero": False,
    }
    assert statistics["gate_l1"]["passed"] is False
    assert statistics["gate_l1"]["scaling_authorized"] is False
