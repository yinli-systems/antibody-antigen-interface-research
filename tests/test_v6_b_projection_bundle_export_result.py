"""Tests for successful V6-B full projection bundle export evidence."""

from __future__ import annotations

import json
from pathlib import Path

from interaction_first.v6_b_projection_bundle_export import canonical_json_bytes


ROOT = Path(__file__).resolve().parents[1]
RESULT = (
    ROOT
    / "protocols"
    / "v6_b_projection_bundle_export_attempt_002_result.json"
)


def test_bundle_export_attempt_002_is_canonical_and_complete() -> None:
    raw = RESULT.read_bytes()
    payload = json.loads(raw)
    assert raw == canonical_json_bytes(payload)
    assert payload["job_id"] == "1331854"
    assert payload["scheduler_state"] == "COMPLETED"
    assert payload["scheduler_exit_code"] == "0:0"
    assert payload["bundle_self_hash_valid"] is True
    assert payload["receipt_self_hash_valid"] is True
    assert payload["output_is_create_only_read_only"] is True
    assert payload["authority_byte_agreement"] is True


def test_bundle_export_attempt_002_binds_full_sparse_identity() -> None:
    payload = json.loads(RESULT.read_bytes())
    assert payload["bundle_sha256"] == (
        "03815396ae85f8af794795d537aff0f6badbf41b22aa823fbe3813144f9f2355"
    )
    assert payload["ir_sha256"] == (
        "3fff0ee63e0346034ba0bc9ce306212523adc786f013aaf37a73bf57848ca05d"
    )
    assert payload["dimensions"] == {
        "columns": 163464,
        "rows": 2693786,
        "nonzeros": 10365708,
    }
    assert payload["artifact_count"] == 9
    assert payload["numeric_envelope"][
        "all_integers_exactly_representable_as_float64"
    ] is True
    assert payload["numeric_envelope"]["maximum_abs_coefficient"] == 100


def test_bundle_export_attempt_002_did_not_solve_or_authorize() -> None:
    payload = json.loads(RESULT.read_bytes())
    assert payload["status"] == (
        "FULL_LINEAR_PROJECTION_BUNDLE_EXPORTED_PENDING_DIRECT_HIGHS"
    )
    assert payload["cp_sat_model_built"] is True
    assert payload["cp_sat_solver_invoked"] is False
    assert payload["highspy_imported"] is False
    assert payload["highs_solver_invoked"] is False
    assert payload["full_protocol_infeasibility_claim_authorized"] is False
    assert payload["split_gate0_authorized"] is False
    assert payload["training_authorized"] is False
