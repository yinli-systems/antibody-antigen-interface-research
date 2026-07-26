"""Tests for the frozen V6-A score-free solver amendment."""

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


ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "protocols" / "v6_a_score_free_solver_policy_v1.json"
PROTOCOL_PATH = ROOT / "protocols" / "V6_A_SCORE_FREE_EXACT_SOLVER_PROTOCOL.md"
MODEL_CONTRACT_PATH = ROOT / "protocols" / "frontier_model_contract_v1.json"


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="ascii"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_policy_thresholds_match_the_statistics_oracle() -> None:
    policy = _load_json(POLICY_PATH)
    thresholds = policy["distribution_thresholds"]
    expected = {
        "outer": OUTER_DISTRIBUTION_THRESHOLDS,
        "pooled_outer": POOLED_OUTER_DISTRIBUTION_THRESHOLDS,
        "inner": INNER_DISTRIBUTION_THRESHOLDS,
    }
    for scope, oracle in expected.items():
        declared = thresholds[scope]
        assert declared["maximum_smd_decimal"] == str(oracle.maximum_smd)
        assert Fraction(*declared["maximum_tv"]) == oracle.maximum_tv
        assert Fraction(*declared["maximum_ks"]) == oracle.maximum_ks


def test_complete_authority_rejects_nonproof_terminal_states() -> None:
    policy = _load_json(POLICY_PATH)
    region = policy["complete_feasible_region"]
    assert region["authoritative_backend"] == (
        "cp_sat_plus_exact_decimal_fraction_separation"
    )
    assert set(region["rejected_statuses"]) == {
        "FEASIBLE",
        "UNKNOWN",
        "timeout",
        "nonzero_gap",
        "incomplete_separation",
        "audit_disagreement",
    }


def test_highs_receipt_is_explicitly_non_authorizing() -> None:
    policy = _load_json(POLICY_PATH)
    highs = policy["highs_linear_projection"]
    assert highs["complete_score_free_feasible_region_authorized"] is False
    assert highs["threads"] == 1
    assert highs["parallel"] == "off"
    assert highs["mip_rel_gap"] == 0.0
    assert highs["mip_abs_gap"] == 0.0
    assert highs["forbidden_callbacks"] == ["user_solution"]


def test_policy_binds_all_forbidden_split_input_classes() -> None:
    policy = _load_json(POLICY_PATH)
    assert set(policy["forbidden_split_inputs"]) == {
        "contact_density",
        "paratope_size",
        "epitope_size",
        "affinity",
        "assay_outcome",
        "embedding",
        "prediction",
        "loss",
        "rank",
        "model_score",
        "model_result",
    }


def test_model_contract_binds_exact_protocol_bytes() -> None:
    contract = _load_json(MODEL_CONTRACT_PATH)
    v6 = contract["prospective_pairformer_engineering"][
        "v6_a_score_free_exact_solver"
    ]
    assert v6["policy_sha256"] == _sha256(POLICY_PATH)
    assert v6["protocol_sha256"] == _sha256(PROTOCOL_PATH)
    assert v6["authorization"] == {
        "implementation_engineering": True,
        "model_selection": False,
        "outer_evaluation": False,
        "split_gate0": False,
        "training": False,
    }


def test_lexicographic_stages_are_unique_and_end_in_bit_fixing() -> None:
    policy = _load_json(POLICY_PATH)
    stages = policy["lexicographic_stage_order"]
    assert len(stages) == len(set(stages))
    assert stages[0] == "maximize_outer_minimum_training_pairs"
    assert stages[-1] == "canonical_sequential_zero_first_bit_fixing"


def test_protocol_preserves_claim_and_training_boundaries() -> None:
    text = PROTOCOL_PATH.read_text(encoding="ascii")
    assert "does not authorize model training" in text
    assert "complete_score_free_feasible_region_authorized=false" in text
    assert "does not promise feasibility" in text
