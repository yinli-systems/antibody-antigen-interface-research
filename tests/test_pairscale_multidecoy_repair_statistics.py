"""Tests for authenticated PAIRScale multi-decoy repair statistics."""

from __future__ import annotations

import hashlib
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
from interaction_first.pairscale_centered_multidecoy import (
    DECOY_COUNT,
    DECOY_POLICY,
    PARTNER_OBJECTIVE,
)
from interaction_first.pairscale_multidecoy_repair_statistics import (
    PairscaleMultidecoyStatisticsError,
    compare_pairscale_multidecoy_repair,
)


def _base_evaluation(
    instance_ids: list[str],
    *,
    auprc: float,
    raw_delta: float,
    centered_delta: float,
) -> dict[str, Any]:
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
    decoy_top = [1.0 - raw_delta] * len(rows)
    matched_centered = [1.0] * len(rows)
    decoy_centered = [1.0 - centered_delta] * len(rows)
    return {
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
            "double_centered_partner_mean_delta": centered_delta,
            "double_centered_partner_win_rate": 1.0,
            "matched_double_centered_scores": matched_centered,
            "matched_partner_topk_mean_delta": raw_delta,
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
        "repair": _base_evaluation(
            ids,
            auprc=0.8,
            raw_delta=0.9,
            centered_delta=0.8,
        ),
        "reference": _base_evaluation(
            ids,
            auprc=0.6,
            raw_delta=0.3,
            centered_delta=0.2,
        ),
        "no_centering": _base_evaluation(
            ids,
            auprc=0.5,
            raw_delta=0.2,
            centered_delta=0.1,
        ),
        "antibody_only": _base_evaluation(
            ids,
            auprc=0.3,
            raw_delta=0.0,
            centered_delta=0.0,
        ),
        "antigen_only": _base_evaluation(
            ids,
            auprc=0.4,
            raw_delta=0.0,
            centered_delta=0.0,
        ),
    }
    evaluations["repair"].update(
        {
            "evaluation_role": "single_fixed_checkpoint_development_holdout",
            "partner_objective": PARTNER_OBJECTIVE,
            "scaling_authorized": False,
            "scope": (
                "authority_train_only_development_double_cold_centered_multidecoy"
            ),
            "training_decoy_count": DECOY_COUNT,
            "training_decoy_policy": DECOY_POLICY,
        }
    )
    evaluations["no_centering"].pop("interpretation")
    evaluations["no_centering"].update(
        {
            "evaluation_role": "single_fixed_checkpoint_development_holdout",
            "scaling_authorized": False,
            "scope": (
                "authority_train_only_development_double_cold_no_centering"
            ),
            "training_partner_score": "raw_topk_only_no_centering",
        }
    )
    for side in ("antibody_only", "antigen_only"):
        evaluations[side].update(
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
    paths = {
        "assignment": root / "assignment.json",
        "assignment_receipt": root / "assignment_receipt.json",
        **{
            f"{name}_evaluation": root / f"{name}_evaluation.json"
            for name in evaluations
        },
    }
    paths["assignment"].write_bytes(canonical_json_bytes(assignment))
    paths["assignment_receipt"].write_bytes(canonical_json_bytes(receipt))
    for name, evaluation in evaluations.items():
        paths[f"{name}_evaluation"].write_bytes(canonical_json_bytes(evaluation))
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
    paths = fixture["paths"]
    hashes = fixture["hashes"]
    return compare_pairscale_multidecoy_repair(
        assignment_path=paths["assignment"],
        assignment_receipt_path=paths["assignment_receipt"],
        repair_evaluation_path=paths["repair_evaluation"],
        reference_evaluation_path=paths["reference_evaluation"],
        no_centering_evaluation_path=paths["no_centering_evaluation"],
        antibody_only_evaluation_path=paths["antibody_only_evaluation"],
        antigen_only_evaluation_path=paths["antigen_only_evaluation"],
        expected_assignment_file_sha256=hashes["assignment"],
        expected_assignment_content_sha256=fixture["assignment_content_sha256"],
        expected_assignment_receipt_sha256=hashes["assignment_receipt"],
        expected_repair_evaluation_sha256=hashes["repair_evaluation"],
        expected_reference_evaluation_sha256=hashes["reference_evaluation"],
        expected_no_centering_evaluation_sha256=hashes[
            "no_centering_evaluation"
        ],
        expected_antibody_only_evaluation_sha256=hashes[
            "antibody_only_evaluation"
        ],
        expected_antigen_only_evaluation_sha256=hashes[
            "antigen_only_evaluation"
        ],
        iterations=100,
        seed=42,
        created_at_utc="2026-07-28T00:00:00Z",
    )


def _rewrite(fixture: dict[str, Any], name: str) -> None:
    raw = canonical_json_bytes(fixture["evaluations"][name])
    key = f"{name}_evaluation"
    fixture["paths"][key].write_bytes(raw)
    fixture["hashes"][key] = hashlib.sha256(raw).hexdigest()


def test_comparator_evaluates_all_frozen_gate_l1_controls(tmp_path: Path) -> None:
    result = _compare(_fixture(tmp_path))
    assert result["bootstrap"]["comparison_count"] == 13
    assert result["bootstrap"]["independent_component_count"] == 10
    assert result["gate_l1"]["passed_required_statistical_checks"] is True
    assert result["gate_l1"]["scaling_authorized"] is False
    assert result["comparisons"][
        "repair_minus_reference_macro_pair_auprc"
    ]["mean_difference"] == pytest.approx(0.2)


def test_comparator_rejects_partner_objective_tamper(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["evaluations"]["repair"]["partner_objective"] = "raw_topk"
    _rewrite(fixture, "repair")
    with pytest.raises(PairscaleMultidecoyStatisticsError, match="objective"):
        _compare(fixture)


def test_comparator_rejects_scaling_authorization_tamper(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["evaluations"]["repair"]["scaling_authorized"] = True
    _rewrite(fixture, "repair")
    with pytest.raises(PairscaleMultidecoyStatisticsError, match="scaling"):
        _compare(fixture)


def test_comparator_rejects_shared_label_inventory_tamper(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["evaluations"]["antibody_only"]["per_complex"][0][
        "positive_count"
    ] = 11
    _rewrite(fixture, "antibody_only")
    with pytest.raises(PairscaleMultidecoyStatisticsError, match="positive_count"):
        _compare(fixture)


def test_comparator_rejects_artifact_hash_tamper(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["paths"]["repair_evaluation"].write_bytes(b"{}\n")
    with pytest.raises(PairscaleMultidecoyStatisticsError, match="authentication"):
        _compare(fixture)
