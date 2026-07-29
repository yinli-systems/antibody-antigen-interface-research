"""Tests for authenticated PAIRScale pair-versus-unary statistics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

import pytest

from interaction_first.development_authority import canonical_json_bytes
from interaction_first.development_holdout import (
    REQUIRED_IMPLEMENTATION_SOURCES,
    ScoreFreeSubset,
    build_component_holdout,
    build_holdout_receipt,
)
from interaction_first.development_subset import AuthorizedInstance
from interaction_first.pairscale_unary_statistics import (
    PairscaleUnaryStatisticsError,
    compare_pairscale_unary_controls,
)


def _fixture(root: Path) -> dict[str, object]:
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
    assignment = build_component_holdout(subset, holdout_pair_count=10, seed=29)
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
    rows = [
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
    base = {
        "checkpoint_promotion_authorized": False,
        "complete_candidate_domain": True,
        "confirmatory_claim_authorized": False,
        "evaluation_role": "single_frozen_development_holdout_evaluation",
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
        "per_complex": rows,
        "pooled_pair_auprc": 0.8,
        "scope": "authority_train_only_development_double_cold_holdout",
        "unknown_cross_pair_semantics": (
            "ranking_control_only_not_a_measured_nonbinder"
        ),
    }
    paths = {
        "assignment": root / "assignment.json",
        "receipt": root / "receipt.json",
        "pair": root / "pair.json",
        "antibody_only": root / "antibody.json",
        "antigen_only": root / "antigen.json",
    }
    paths["assignment"].write_bytes(assignment_bytes)
    paths["receipt"].write_bytes(receipt_bytes)
    paths["pair"].write_bytes(canonical_json_bytes(base))
    evaluations = {"pair": base}
    for side, value in (("antibody_only", 0.25), ("antigen_only", 0.30)):
        unary = json.loads(json.dumps(base))
        unary.update(
            {
                "architecture_reads_partner_feature_values": False,
                "control_side": side,
                "evaluation_role": "single_fixed_checkpoint_development_holdout",
                "scaling_authorized": False,
                "scope": (
                    "authority_train_only_development_double_cold_unary_control"
                ),
            }
        )
        unary["macro_pair_auprc"] = value
        for row in unary["per_complex"]:
            row["auprc"] = value
        paths[side].write_bytes(canonical_json_bytes(unary))
        evaluations[side] = unary
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}
    return {
        "assignment_content_sha256": assignment["assignment_content_sha256"],
        "evaluations": evaluations,
        "hashes": hashes,
        "paths": paths,
    }


def _compare(fixture: dict[str, object]) -> Mapping[str, object]:
    paths = fixture["paths"]
    hashes = fixture["hashes"]
    return compare_pairscale_unary_controls(
        assignment_path=paths["assignment"],
        assignment_receipt_path=paths["receipt"],
        pair_evaluation_path=paths["pair"],
        antibody_only_evaluation_path=paths["antibody_only"],
        antigen_only_evaluation_path=paths["antigen_only"],
        expected_assignment_file_sha256=hashes["assignment"],
        expected_assignment_content_sha256=fixture["assignment_content_sha256"],
        expected_assignment_receipt_sha256=hashes["receipt"],
        expected_pair_evaluation_sha256=hashes["pair"],
        expected_antibody_only_evaluation_sha256=hashes["antibody_only"],
        expected_antigen_only_evaluation_sha256=hashes["antigen_only"],
        iterations=100,
        seed=42,
        created_at_utc="2026-07-27T19:00:00Z",
    )


def test_comparator_uses_component_paired_upper_envelope(tmp_path: Path) -> None:
    result = _compare(_fixture(tmp_path))
    assert result["bootstrap"]["independent_component_count"] == 10
    assert result["gate_l1"]["passed"] is False
    assert result["gate_l1"]["status"] == "pending_no_centering_ablation"
    assert result["gate_l1"]["unresolved_required_controls"] == [
        "no_centering_ablation"
    ]
    comparison = result["comparisons"][
        "pair_minus_per_instance_separable_upper_envelope_macro_auprc"
    ]
    assert comparison["mean_difference"] == pytest.approx(0.5)
    assert comparison["lower"] == pytest.approx(0.5)


def test_comparator_rejects_control_side_tampering(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    evaluation = fixture["evaluations"]["antibody_only"]
    evaluation["control_side"] = "antigen_only"
    raw = canonical_json_bytes(evaluation)
    fixture["paths"]["antibody_only"].write_bytes(raw)
    fixture["hashes"]["antibody_only"] = hashlib.sha256(raw).hexdigest()
    with pytest.raises(PairscaleUnaryStatisticsError, match="side differs"):
        _compare(fixture)


def test_comparator_rejects_evaluation_hash_tampering(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["paths"]["antigen_only"].write_bytes(b"{}\n")
    with pytest.raises(PairscaleUnaryStatisticsError, match="SHA-256 differs"):
        _compare(fixture)
