"""Tests for authenticated PAIRScale centering-ablation statistics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from interaction_first.development_authority import canonical_json_bytes
from interaction_first.development_holdout import (
    REQUIRED_IMPLEMENTATION_SOURCES,
    ScoreFreeSubset,
    build_component_holdout,
    build_holdout_receipt,
)
from interaction_first.development_subset import AuthorizedInstance
from interaction_first.pairscale_centering_ablation_statistics import (
    PairscaleCenteringStatisticsError,
    compare_pairscale_centering_ablation,
)


def _evaluation(instance_ids: list[str], *, auprc: float, no_centering: bool) -> dict[str, Any]:
    rows = [
        {
            "accuracy": 0.9,
            "auprc": auprc,
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
            "precision_at_true_contact_count": auprc - 0.1,
            "precision_at_true_contact_count_tie_policy": (
                "fractional_expected_credit_at_score_boundary"
            ),
            "prevalence": 0.1,
            "recall": 0.5,
            "threshold": 0.5,
        }
        for identifier in instance_ids
    ]
    matched_top = [1.0] * len(rows)
    decoy_top = ([0.4] if no_centering else [0.2]) * len(rows)
    matched_centered = [1.2] * len(rows)
    decoy_centered = ([0.8] if no_centering else [0.3]) * len(rows)
    top_deltas = [matched - decoy for matched, decoy in zip(matched_top, decoy_top)]
    centered_deltas = [
        matched - decoy
        for matched, decoy in zip(matched_centered, decoy_centered)
    ]
    result: dict[str, Any] = {
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
        "macro_pair_auprc": auprc,
        "macro_precision_at_true_contact_count": auprc - 0.1,
        "model_selection_authorized": False,
        "negative_pair_subsampling": False,
        "partner_controls": {
            "decoy_double_centered_scores": decoy_centered,
            "decoy_policy": "deterministic_cyclic_non_cognate_antigen",
            "decoy_topk_scores": decoy_top,
            "double_centered_partner_mean_delta": centered_deltas[0],
            "double_centered_partner_win_rate": 1.0,
            "matched_double_centered_scores": matched_centered,
            "matched_partner_topk_mean_delta": top_deltas[0],
            "matched_partner_topk_win_rate": 1.0,
            "matched_topk_scores": matched_top,
        },
        "per_complex": rows,
        "pooled_pair_auprc": auprc,
        "scope": "authority_train_only_development_double_cold_holdout",
        "unknown_cross_pair_semantics": (
            "ranking_control_only_not_a_measured_nonbinder"
        ),
    }
    if no_centering:
        result.pop("interpretation")
        result.update(
            {
                "evaluation_role": "single_fixed_checkpoint_development_holdout",
                "scaling_authorized": False,
                "scope": (
                    "authority_train_only_development_double_cold_no_centering"
                ),
                "training_partner_score": "raw_topk_only_no_centering",
            }
        )
    return result


def _fixture(root: Path) -> dict[str, Any]:
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
    ids = [row["instance_id"] for row in assignment["holdout"]["instances"]]
    evaluations = {
        "reference": _evaluation(ids, auprc=0.8, no_centering=False),
        "no_centering": _evaluation(ids, auprc=0.5, no_centering=True),
    }
    paths = {
        "assignment": root / "assignment.json",
        "receipt": root / "receipt.json",
        "reference": root / "reference.json",
        "no_centering": root / "no_centering.json",
    }
    paths["assignment"].write_bytes(canonical_json_bytes(assignment))
    paths["receipt"].write_bytes(canonical_json_bytes(receipt))
    for name in ("reference", "no_centering"):
        paths[name].write_bytes(canonical_json_bytes(evaluations[name]))
    return {
        "assignment_content_sha256": assignment["assignment_content_sha256"],
        "evaluations": evaluations,
        "hashes": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in paths.items()
        },
        "paths": paths,
    }


def _compare(fixture: Mapping[str, Any]) -> Mapping[str, Any]:
    return compare_pairscale_centering_ablation(
        assignment_path=fixture["paths"]["assignment"],
        assignment_receipt_path=fixture["paths"]["receipt"],
        reference_evaluation_path=fixture["paths"]["reference"],
        no_centering_evaluation_path=fixture["paths"]["no_centering"],
        expected_assignment_file_sha256=fixture["hashes"]["assignment"],
        expected_assignment_content_sha256=fixture["assignment_content_sha256"],
        expected_assignment_receipt_sha256=fixture["hashes"]["receipt"],
        expected_reference_evaluation_sha256=fixture["hashes"]["reference"],
        expected_no_centering_evaluation_sha256=fixture["hashes"]["no_centering"],
        iterations=100,
        seed=42,
        created_at_utc="2026-07-28T00:00:00Z",
    )


def test_comparator_reports_paired_centering_effects(tmp_path: Path) -> None:
    result = _compare(_fixture(tmp_path))
    assert result["bootstrap"]["independent_component_count"] == 10
    assert result["centering_ablation"]["completed"] is True
    assert result["gate_l1"]["scaling_authorized"] is False
    assert result["gate_l1"]["unresolved_required_controls"] == []
    assert result["comparisons"][
        "reference_minus_no_centering_macro_pair_auprc"
    ]["mean_difference"] == pytest.approx(0.3)
    assert result["comparisons"][
        "reference_minus_no_centering_double_centered_partner_delta"
    ]["mean_difference"] == pytest.approx(0.5)


def test_comparator_rejects_no_centering_treatment_tamper(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["evaluations"]["no_centering"]["training_partner_score"] = "raw_topk_only"
    raw = canonical_json_bytes(fixture["evaluations"]["no_centering"])
    fixture["paths"]["no_centering"].write_bytes(raw)
    fixture["hashes"]["no_centering"] = hashlib.sha256(raw).hexdigest()
    with pytest.raises(PairscaleCenteringStatisticsError, match="treatment differs"):
        _compare(fixture)


def test_comparator_rejects_summary_tamper(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["evaluations"]["reference"]["macro_pair_auprc"] = 0.9
    raw = canonical_json_bytes(fixture["evaluations"]["reference"])
    fixture["paths"]["reference"].write_bytes(raw)
    fixture["hashes"]["reference"] = hashlib.sha256(raw).hexdigest()
    with pytest.raises(PairscaleCenteringStatisticsError, match="macro_pair_auprc"):
        _compare(fixture)


def test_comparator_rejects_label_inventory_tamper(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["evaluations"]["no_centering"]["per_complex"][0]["positive_count"] = 11
    raw = canonical_json_bytes(fixture["evaluations"]["no_centering"])
    fixture["paths"]["no_centering"].write_bytes(raw)
    fixture["hashes"]["no_centering"] = hashlib.sha256(raw).hexdigest()
    with pytest.raises(PairscaleCenteringStatisticsError, match="positive_count"):
        _compare(fixture)


def test_comparator_rejects_artifact_hash_tamper(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["paths"]["no_centering"].write_bytes(b"{}\n")
    with pytest.raises(PairscaleCenteringStatisticsError, match="authentication"):
        _compare(fixture)
