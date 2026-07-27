"""Tests for formal V6-B independent direct-CP result evidence."""

from __future__ import annotations

import json
from pathlib import Path

from interaction_first.v6_b_independent_cp_audit import canonical_json_bytes


ROOT = Path(__file__).resolve().parents[1]
RESULT = (
    ROOT / "protocols" / "v6_b_independent_cp_audit_attempt_001_result.json"
)


def test_formal_independent_cp_result_is_canonical_and_agrees() -> None:
    raw = RESULT.read_bytes()
    payload = json.loads(raw)
    assert raw == canonical_json_bytes(payload)
    assert payload["job_id"] == "1331650"
    assert payload["scheduler_state"] == "COMPLETED"
    assert payload["scheduler_exit_code"] == "0:0"
    assert payload["projection_status"] == "INFEASIBLE"
    assert payload["authority_byte_agreement"] is True
    assert payload["linear_agreement_with_primary"] is True
    assert payload["receipt_self_hash_valid"] is True
    assert payload["output_is_create_only_read_only"] is True


def test_formal_independent_cp_result_binds_full_linear_identity() -> None:
    payload = json.loads(RESULT.read_bytes())
    assert payload["normalized_authority_sha256"] == (
        "7d2658f76615e581375dea42296244e0c7ec0b9c8523da21051f11b7af6310c4"
    )
    assert payload["direct_formulation_sha256"] == (
        "4b1f51a2bc900d163875836bcae3101f7e828c7c0572f93a95877e2d2315ac39"
    )
    assert payload["ir_sha256"] == (
        "3fff0ee63e0346034ba0bc9ce306212523adc786f013aaf37a73bf57848ca05d"
    )
    assert payload["dimensions"] == {
        "variables": 163464,
        "linear_constraints": 2693786,
        "exact_gate_declarations": 66,
    }
    assert payload["v6_b_partition_or_cut_consumed"] is False


def test_formal_independent_cp_result_remains_non_authorizing() -> None:
    payload = json.loads(RESULT.read_bytes())
    assert payload["status"] == (
        "INDEPENDENT_DIRECT_CP_INFEASIBLE_AGREEMENT_PENDING_DIRECT_HIGHS"
    )
    assert payload["direct_highs_audit_completed"] is False
    assert payload["full_protocol_infeasibility_claim_authorized"] is False
    assert payload["split_gate0_authorized"] is False
    assert payload["training_authorized"] is False
    assert (
        payload["exact_score_free_separation_required_after_linear_infeasibility"]
        is False
    )
