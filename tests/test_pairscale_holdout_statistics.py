"""Tests for authenticated PAIRScale holdout cluster statistics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from interaction_first.development_authority import canonical_json_bytes
from interaction_first.development_holdout import (
    REQUIRED_IMPLEMENTATION_SOURCES,
    ScoreFreeSubset,
    build_component_holdout,
    build_holdout_receipt,
)
from interaction_first.development_subset import AuthorizedInstance
from interaction_first.pairscale_holdout_statistics import (
    PairscaleHoldoutStatisticsError,
    analyze_development_holdout_statistics,
)


def _write_fixture(root: Path) -> dict[str, object]:
    instances = tuple(
        AuthorizedInstance(f"i{index:02d}", f"a{index:02d}", f"g{index:02d}")
        for index in range(12)
    )
    subset = ScoreFreeSubset(
        instances=instances,
        subset_sha256="1" * 64,
        policy_sha256="2" * 64,
        permission_name="development_only_nonconfirmatory",
        source_split="train",
    )
    assignment = build_component_holdout(
        subset,
        holdout_pair_count=10,
        seed=29,
    )
    receipt = build_holdout_receipt(
        assignment,
        implementation_source_sha256={
            name: f"{index + 3:x}" * 64
            for index, name in enumerate(sorted(REQUIRED_IMPLEMENTATION_SOURCES))
        },
    )
    assignment_bytes = canonical_json_bytes(assignment)
    receipt_bytes = canonical_json_bytes(receipt)
    holdout_ids = [row["instance_id"] for row in assignment["holdout"]["instances"]]
    per_complex = [
        {
            "accuracy": 0.9,
            "auprc": 0.8,
            "auroc": 0.9,
            "contact_iou": 0.4,
            "count": 100,
            "f1": 0.5,
            "instance_id": identifier,
            "label_marginal_oracle_auprc": 0.1,
            "mcc": 0.4,
            "model_marginal_outer_auprc": 0.2,
            "positive_count": 10,
            "precision": 0.5,
            "precision_at_true_contact_count": 0.7,
            "precision_at_true_contact_count_tie_policy": (
                "fractional_expected_credit_at_score_boundary"
            ),
            "prevalence": 0.1,
            "recall": 0.5,
            "threshold": 0.5,
        }
        for identifier in holdout_ids
    ]
    evaluation = {
        "checkpoint_promotion_authorized": False,
        "complete_candidate_domain": True,
        "confirmatory_claim_authorized": False,
        "evaluation_role": "development_holdout_once_after_frozen_train_gate",
        "gate_checks": {},
        "gate_passed": False,
        "held_out_evaluation_performed": True,
        "interpretation": "fixture",
        "macro_label_marginal_oracle_auprc": 0.1,
        "macro_model_marginal_outer_auprc": 0.2,
        "macro_pair_auprc": 0.8,
        "macro_precision_at_true_contact_count": 0.7,
        "model_selection_authorized": False,
        "negative_pair_subsampling": False,
        "partner_controls": {
            "decoy_double_centered_scores": [0.1] * len(holdout_ids),
            "decoy_policy": "deterministic_cyclic_non_cognate_antigen",
            "decoy_topk_scores": [0.2] * len(holdout_ids),
            "double_centered_partner_mean_delta": 0.8,
            "double_centered_partner_win_rate": 1.0,
            "matched_double_centered_scores": [0.9] * len(holdout_ids),
            "matched_partner_topk_mean_delta": 0.6,
            "matched_partner_topk_win_rate": 1.0,
            "matched_topk_scores": [0.8] * len(holdout_ids),
        },
        "per_complex": per_complex,
        "pooled_pair_auprc": 0.8,
        "scope": "authority_train_only_development_endpoint_disjoint_holdout",
        "unknown_cross_pair_semantics": (
            "ranking_control_only_not_a_measured_nonbinder"
        ),
    }
    evaluation_bytes = (json.dumps(evaluation, sort_keys=True) + "\n").encode()
    paths = {
        "assignment": root / "assignment.json",
        "receipt": root / "receipt.json",
        "evaluation": root / "evaluation.json",
    }
    paths["assignment"].write_bytes(assignment_bytes)
    paths["receipt"].write_bytes(receipt_bytes)
    paths["evaluation"].write_bytes(evaluation_bytes)
    return {
        "assignment_content_sha256": assignment["assignment_content_sha256"],
        "assignment_file_sha256": hashlib.sha256(assignment_bytes).hexdigest(),
        "assignment_receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
        "evaluation": evaluation,
        "evaluation_sha256": hashlib.sha256(evaluation_bytes).hexdigest(),
        "paths": paths,
    }


def _analyze(fixture: dict[str, object]) -> dict[str, object]:
    paths = fixture["paths"]
    return analyze_development_holdout_statistics(
        assignment_path=paths["assignment"],
        assignment_receipt_path=paths["receipt"],
        evaluation_path=paths["evaluation"],
        expected_assignment_file_sha256=fixture["assignment_file_sha256"],
        expected_assignment_content_sha256=fixture["assignment_content_sha256"],
        expected_assignment_receipt_sha256=fixture["assignment_receipt_sha256"],
        expected_evaluation_sha256=fixture["evaluation_sha256"],
        iterations=100,
        seed=17,
        created_at_utc="2026-07-27T18:00:00Z",
    )


def test_statistics_authenticate_components_and_never_authorize_scaling(
    tmp_path: Path,
) -> None:
    result = _analyze(_write_fixture(tmp_path))
    assert result["bootstrap"]["independent_component_count"] == 10
    assert result["bootstrap"]["instance_count"] == 10
    assert result["gate_l1"]["passed"] is False
    assert result["gate_l1"]["scaling_authorized"] is False
    assert all(result["gate_l1"]["checks"].values())
    comparison = result["comparisons"][
        "pair_head_minus_label_marginal_oracle_auprc"
    ]
    assert comparison["mean_difference"] == pytest.approx(0.7)
    assert comparison["inferential_authorized"] is True


def test_statistics_reject_evaluation_hash_tampering(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path)
    fixture["paths"]["evaluation"].write_bytes(b"{}\n")
    with pytest.raises(PairscaleHoldoutStatisticsError, match="SHA-256"):
        _analyze(fixture)


def test_statistics_reject_instance_and_authorization_tampering(
    tmp_path: Path,
) -> None:
    fixture = _write_fixture(tmp_path)
    evaluation = fixture["evaluation"]
    evaluation["per_complex"][0]["instance_id"] = "unknown"
    evaluation["model_selection_authorized"] = True
    raw = (json.dumps(evaluation, sort_keys=True) + "\n").encode()
    fixture["paths"]["evaluation"].write_bytes(raw)
    fixture["evaluation_sha256"] = hashlib.sha256(raw).hexdigest()
    with pytest.raises(
        PairscaleHoldoutStatisticsError,
        match="authorization differs",
    ):
        _analyze(fixture)
