"""Tests for immutable V6-B projection bundle export attempt evidence."""

from __future__ import annotations

import json
from pathlib import Path

from interaction_first.v6_b_projection_bundle_export import canonical_json_bytes


ROOT = Path(__file__).resolve().parents[1]
ATTEMPT = ROOT / "protocols" / "v6_b_projection_bundle_export_attempt_001.json"


def test_bundle_export_attempt_001_is_canonical_and_fail_closed() -> None:
    raw = ATTEMPT.read_bytes()
    payload = json.loads(raw)
    assert raw == canonical_json_bytes(payload)
    assert payload["job_id"] == "1331704"
    assert payload["scheduler_state"] == "FAILED"
    assert payload["exit_code"] == "1:0"
    assert payload["status"] == (
        "FAILED_BEFORE_MODEL_CONSTRUCTION_AND_OUTPUT_CREATION"
    )
    assert payload["stage"] == "production_registration_digest_verification"
    assert payload["error_type"] == "AttributeError"


def test_bundle_export_attempt_001_is_not_scientific_evidence() -> None:
    payload = json.loads(ATTEMPT.read_bytes())
    assert payload["authority_or_feasibility_failure"] is False
    assert payload["output_directory_created"] is False
    assert payload["cp_sat_model_built"] is False
    assert payload["cp_sat_solver_invoked"] is False
    assert payload["bundle_exported"] is False
    assert payload["highs_solver_invoked"] is False
    assert payload["full_protocol_infeasibility_claim_authorized"] is False
    assert payload["split_gate0_authorized"] is False
    assert payload["training_authorized"] is False
