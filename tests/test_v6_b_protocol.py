"""Tests for the frozen V6-B logic-based decomposition protocol."""

from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path

from interaction_first.score_free_split_statistics import (
    INNER_DISTRIBUTION_THRESHOLDS,
    OUTER_DISTRIBUTION_THRESHOLDS,
    POOLED_OUTER_DISTRIBUTION_THRESHOLDS,
)
from interaction_first.v6_joint_ir import separation_gate_registry


ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "protocols" / "v6_b_logic_based_decomposition_policy_v1.json"
PROTOCOL_PATH = ROOT / "protocols" / "V6_B_LOGIC_BASED_DECOMPOSITION_PROTOCOL.md"
CONTRACT_PATH = ROOT / "protocols" / "frontier_model_contract_v1.json"


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="ascii"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v6_b_preserves_v6_a_thresholds_and_forbidden_inputs() -> None:
    v6_a = _load_json(
        ROOT / "protocols" / "v6_a_score_free_solver_policy_v1.json"
    )
    v6_b = _load_json(POLICY_PATH)
    assert v6_b["distribution_thresholds"] == v6_a["distribution_thresholds"]
    assert v6_b["forbidden_split_inputs"] == v6_a["forbidden_split_inputs"]
    expected = {
        "outer": OUTER_DISTRIBUTION_THRESHOLDS,
        "pooled_outer": POOLED_OUTER_DISTRIBUTION_THRESHOLDS,
        "inner": INNER_DISTRIBUTION_THRESHOLDS,
    }
    for scope, oracle in expected.items():
        threshold = v6_b["distribution_thresholds"][scope]
        assert threshold["maximum_smd_decimal"] == str(oracle.maximum_smd)
        assert Fraction(*threshold["maximum_tv"]) == oracle.maximum_tv
        assert Fraction(*threshold["maximum_ks"]) == oracle.maximum_ks


def test_gate_partition_is_complete_and_matches_registry() -> None:
    policy = _load_json(POLICY_PATH)
    partition = policy["exact_gate_partition"]
    assert partition == {
        "inner_gate_count_per_outer_fold": 16,
        "inner_groups": 3,
        "outer_gate_count": 18,
        "total_gate_count": 66,
    }
    registry = separation_gate_registry("0" * 64)
    inner = [gate for gate in registry if "_inner" in gate.level_id]
    outer = [gate for gate in registry if "_inner" not in gate.level_id]
    assert len(registry) == 66
    assert len(outer) == 18
    assert len(inner) == 48
    for outer_fold in range(3):
        prefix = f"outer_{outer_fold}_inner"
        assert sum(gate.level_id.startswith(prefix) for gate in inner) == 16


def test_inner_cut_requires_a_replayed_sufficient_assumption_support() -> None:
    policy = _load_json(POLICY_PATH)
    cut = policy["inner_infeasibility_cut"]
    assert cut["assumption_core_claim"] == "sufficient_not_necessarily_minimum"
    assert cut["fallback_full_outer_state_cut_authorized"] is False
    assert cut["acceptance_sequence"][-1] == (
        "add_exact_signed_support_boolean_no_good"
    )
    fields = set(cut["required_receipt_fields"])
    assert {
        "signed_support",
        "first_response_sha256",
        "replay_response_sha256",
        "independent_replay_sha256",
        "cut_sha256",
    } <= fields


def test_row_shard_is_telemetry_only_and_never_samples_authority() -> None:
    policy = _load_json(POLICY_PATH)
    shard = policy["row_shard_telemetry"]
    assert shard["full_variable_dictionary_required"] is True
    assert shard["pair_or_vertex_sampling_authorized"] is False
    assert shard["solver_invocation_authorized"] is False
    assert shard["telemetry_can_authorize_feasibility"] is False
    assert shard["telemetry_can_authorize_training"] is False
    assert Fraction(*shard["minimum_projected_resource_headroom_fraction"]) == Fraction(1, 4)


def test_only_proof_terminal_states_can_authorize_a_candidate() -> None:
    policy = _load_json(POLICY_PATH)
    region = policy["complete_feasible_region"]
    assert region["highs_complete_region_authorized"] is False
    assert set(region["rejected_statuses"]) == {
        "FEASIBLE",
        "UNKNOWN",
        "MODEL_INVALID",
        "timeout",
        "nonzero_gap",
        "incomplete_separation",
        "audit_disagreement",
    }


def test_parent_failure_is_preserved_without_full_authority_claim() -> None:
    policy = _load_json(POLICY_PATH)
    parent = policy["parent_v6_a_terminal_evidence"]
    assert parent["v6_a_stopped"] is True
    assert parent["v6_a_full_authority_infeasibility_claim"] is False
    assert parent["terminal_status"] == "STATIC_NECESSARY_CONDITION_INFEASIBLE"


def test_contract_binds_exact_v6_b_protocol_bytes() -> None:
    contract = _load_json(CONTRACT_PATH)
    v6_b = contract["prospective_pairformer_engineering"]["v6_b_decomposition"]
    assert v6_b["policy_sha256"] == _sha256(POLICY_PATH)
    assert v6_b["protocol_sha256"] == _sha256(PROTOCOL_PATH)
    assert v6_b["authorization"] == {
        "implementation_engineering": True,
        "model_selection": False,
        "outer_evaluation": False,
        "split_gate0": False,
        "training": False,
    }


def test_protocol_keeps_training_and_claim_boundaries_closed() -> None:
    text = PROTOCOL_PATH.read_text(encoding="ascii")
    assert "does not authorize model training" in text
    assert "It never samples vertices or pairs." in text
    assert "V6-B does not promise feasibility" in text
