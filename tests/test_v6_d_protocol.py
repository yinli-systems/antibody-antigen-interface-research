"""Tests for the frozen non-authorizing V6-D protocol."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols" / "V6_D_TWO_CP_CONSENSUS_CORE_PROTOCOL.md"
POLICY = ROOT / "protocols" / "v6_d_two_cp_consensus_core_policy_v1.json"


def test_v6_d_policy_is_non_authorizing_and_cp_consensus_only() -> None:
    payload = json.loads(POLICY.read_text("ascii"))
    assert payload["format_version"] == "v6_d_two_cp_consensus_core_policy_v1"
    assert all(value is False for value in payload["authorization"].values())
    assert payload["parent_requirements"] == {
        "independent_cp_sat": "INDEPENDENT_DIRECT_CP_INFEASIBLE_AGREEMENT",
        "primary_cp_sat": "PROVEN_INFEASIBLE",
        "same_linear_projection_required": True,
    }
    assert payload["highs_core_audit"]["full_model_timeout_is_not_a_proof"] is True
    assert payload["highs_core_audit"]["row_subset_containment_required"] is True
    assert "test_bytes" in payload["forbidden_inputs"]
    assert payload["authority_counts"] == {
        "antibody_entities": 1383,
        "antigen_entities": 1172,
        "exact_pairs": 1830,
        "source_rows": 3229,
    }


def test_v6_d_protocol_preserves_failure_and_claim_boundaries() -> None:
    payload = PROTOCOL.read_bytes()
    assert payload.isascii()
    text = payload.decode("ascii")
    normalized = " ".join(text.split())
    for statement in (
        "does not reinterpret the HiGHS timeout as",
        "does not authorize a split",
        "does not authorize Gate 0",
        "No minimum-cardinality claim is allowed",
        "The frozen outer test remains untouched",
    ):
        assert statement in normalized
    assert "SOTA" not in text
    assert "world-leading" not in text


def test_v6_d_policy_and_protocol_do_not_name_test_artifacts() -> None:
    combined = (PROTOCOL.read_text("ascii") + POLICY.read_text("ascii")).lower()
    assert "testdata.pkl" not in combined
    assert "frozen-test" not in combined
