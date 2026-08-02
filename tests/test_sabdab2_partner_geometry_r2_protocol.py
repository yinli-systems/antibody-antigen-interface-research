"""Validate the frozen SAbDab2 partner-geometry R2 protocol."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "protocols" / "sabdab2_partner_geometry_r2_v1.json"


def _protocol() -> dict:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def test_protocol_is_blocked_at_gate_zero_and_test_is_sealed() -> None:
    protocol = _protocol()
    authorization = protocol["authorization"]
    firewall = protocol["test_firewall"]

    assert protocol["status"] == "implementation_authorized_training_blocked_at_gate_0"
    assert authorization["source_implementation"] is True
    assert authorization["scientific_training"] is False
    assert authorization["inner_model_selection"] is False
    assert authorization["outer_fold_inference"] is False
    assert authorization["external_test_mount_or_read"] is False
    assert authorization["external_test_schema_access"] is False
    assert authorization["external_test_inference"] is False
    assert firewall["current_state"] == "sealed"
    assert firewall["access_before_final_freeze"] is False
    assert firewall["schema_inspection_before_final_freeze"] is False
    assert firewall["performance_scoring_before_final_freeze"] is False


def test_protocol_freezes_monomer_and_candidate_ladder() -> None:
    protocol = _protocol()
    monomer = protocol["frozen_monomer_encoder"]
    confirmation = monomer["confirmation"]
    ladder = {candidate["id"]: candidate for candidate in protocol["candidate_ladder"]}

    assert monomer["geometry_feature_mode"] == "distance_direction"
    assert monomer["antibody_and_antigen_parameters_shared"] is False
    assert confirmation["seeds"] == [17, 42, 73, 101, 137]
    assert confirmation["epochs"] == 20
    assert confirmation["checkpoint_rule"] == "fixed_final_epoch_20"
    assert confirmation["early_stopping"] is False
    assert confirmation["validation_checkpoint_selection"] is False
    assert set(ladder) == {"D0", "R1", "R2"}
    assert ladder["D0"]["epochs"] == 58
    assert ladder["R1"]["epochs"] == 40
    assert ladder["R2"]["epochs"] == 40
    assert ladder["R2"]["num_pair_update_layers"] == 2
    assert all(candidate["early_stopping"] is False for candidate in ladder.values())


def test_protocol_requires_native_pairs_controls_and_bidirectional_retrieval() -> None:
    protocol = _protocol()

    assert (
        protocol["primary_and_secondary_metrics"]["primary"]
        == "per_complex_macro_native_pair_matrix_auprc"
    )
    assert protocol["labels"]["primary_contact_definition"].endswith(
        "le_4.5_angstrom"
    )
    assert protocol["pair_representation"][
        "complete_frozen_candidate_domain_required_for_evaluation"
    ]
    assert not protocol["pair_representation"][
        "negative_subsampling_allowed_during_evaluation"
    ]
    assert set(protocol["retrieval"]["directions"]) == {
        "antibody_queries_antigens",
        "antigen_queries_antibodies",
    }
    assert {
        "matched_partner_shuffle",
        "antibody_only",
        "antigen_only",
        "degree_aware_random",
        "topology_only",
    }.issubset(protocol["controls"])


def test_gate_one_and_freeze_order_fail_closed() -> None:
    protocol = _protocol()
    requirements = protocol["gate_1"]["requirements"]
    order = protocol["freeze_and_test_order"]

    assert protocol["gate_1"]["all_requirements_must_pass"] is True
    assert len(requirements) == 6
    assert any("strongest_separable_baseline" in item for item in requirements)
    assert any("matched_partner_shuffle" in item for item in requirements)
    assert any("both_retrieval_directions" in item for item in requirements)
    assert order.index("run_inner_only_candidate_selection") < order.index(
        "run_outer_folds_once"
    )
    assert order.index("run_outer_folds_once") < order.index(
        "mount_external_test_once_and_run_one_shot_inference"
    )
    assert protocol["claim_policy"][
        "sota_or_world_leading_without_identical_benchmark_evidence"
    ] is False
