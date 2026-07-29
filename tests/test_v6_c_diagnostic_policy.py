"""Tests for the frozen V6-C linear-infeasibility diagnostic policy."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
POLICY = (
    ROOT
    / "protocols"
    / "v6_c_linear_infeasibility_diagnostic_policy_v1.json"
)
PROTOCOL = (
    ROOT
    / "protocols"
    / "V6_C_LINEAR_INFEASIBILITY_DIAGNOSIS_PROTOCOL.md"
)


def _load_policy() -> dict[str, object]:
    payload = POLICY.read_bytes()
    assert payload.isascii()
    parsed = json.loads(payload.decode("ascii"))
    assert json.dumps(
        parsed,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return parsed


def test_v6_c_policy_is_fail_closed() -> None:
    policy = _load_policy()
    assert policy["format_version"] == (
        "v6_c_linear_infeasibility_diagnostic_policy_v1"
    )
    assert policy["authority_counts"] == {
        "antibodies": 1383,
        "antigens": 1172,
        "observed_pairs": 1830,
        "source_rows": 3229,
    }
    assert set(policy["authorization"].values()) == {False}
    assert policy["parent_requirements"] == {
        "direct_highs": "DIRECT_HIGHS_LINEAR_INFEASIBLE_AGREEMENT",
        "independent_cp_sat": (
            "INDEPENDENT_DIRECT_CP_INFEASIBLE_AGREEMENT"
        ),
        "primary_cp_sat": "PROVEN_INFEASIBLE",
        "same_linear_projection_required": True,
    }


def test_v6_c_subset_and_irreducibility_semantics_are_exact() -> None:
    policy = _load_policy()
    core = policy["core_policy"]
    assert core["excluded_families_fixed_false"] is True
    assert core["all_families_status_required"] == "INFEASIBLE"
    assert core["replay_status_required"] == "INFEASIBLE"
    assert core["canonical_deletion_order"] is True
    assert core["final_single_deletion_statuses"] == [
        "FEASIBLE",
        "OPTIMAL",
    ]
    assert core["minimum_cardinality_claim_authorized"] is False


def test_v6_c_solver_policy_is_deterministic() -> None:
    policy = _load_policy()
    assert policy["solver"] == {
        "absolute_gap_limit": 0.0,
        "accepted_infeasibility_status": "INFEASIBLE",
        "find_multiple_cores": True,
        "num_search_workers": 1,
        "random_seed": 2026072701,
        "randomize_search": False,
        "relative_gap_limit": 0.0,
    }


def test_v6_c_protocol_is_ascii_and_preserves_test_firewall() -> None:
    payload = PROTOCOL.read_bytes()
    assert payload.isascii()
    text = payload.decode("ascii")
    assert "does not authorize a split" in text
    assert "test bytes remain forbidden" in text
    assert "excluded family is explicitly fixed false" in text
    assert "No minimum-cardinality claim is allowed" in text
