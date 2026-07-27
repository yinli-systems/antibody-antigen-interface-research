"""Tests for immutable V6-B scheduler submission evidence."""

from __future__ import annotations

import json
from pathlib import Path

from interaction_first.v6_b_production import canonical_json_bytes


ROOT = Path(__file__).resolve().parents[1]
ATTEMPT = ROOT / "protocols" / "v6_b_production_submission_attempt_001.json"
ATTEMPT_002 = ROOT / "protocols" / "v6_b_production_submission_attempt_002.json"
ATTEMPT_003 = ROOT / "protocols" / "v6_b_production_attempt_003.json"


def test_attempt_001_is_canonical_and_records_no_job_or_solver() -> None:
    raw = ATTEMPT.read_bytes()
    payload = json.loads(raw)
    assert raw == canonical_json_bytes(payload)
    assert payload["status"] == "SCHEDULER_REJECTED_ZERO_GPU_REQUEST"
    assert payload["job_submitted"] is False
    assert payload["job_id"] is None
    assert payload["output_directory_created"] is False
    assert payload["model_built"] is False
    assert payload["solver_invoked"] is False
    assert payload["training_authorized"] is False


def test_attempt_001_does_not_claim_scientific_infeasibility() -> None:
    payload = json.loads(ATTEMPT.read_bytes())
    assert payload["stage"] == "slurm_test_only"
    assert payload["authority_or_split_failure"] is False
    assert payload["registered_gpus_requested"] == 0


def test_attempt_002_is_canonical_and_records_no_job_or_solver() -> None:
    raw = ATTEMPT_002.read_bytes()
    payload = json.loads(raw)
    assert raw == canonical_json_bytes(payload)
    assert payload["status"] == "SCHEDULER_REJECTED_EXPLICIT_MEMORY_REQUEST"
    assert payload["job_submitted"] is False
    assert payload["job_id"] is None
    assert payload["output_directory_created"] is False
    assert payload["model_built"] is False
    assert payload["solver_invoked"] is False
    assert payload["training_authorized"] is False


def test_attempt_002_only_records_scheduler_envelope_incompatibility() -> None:
    payload = json.loads(ATTEMPT_002.read_bytes())
    assert payload["stage"] == "slurm_test_only"
    assert payload["authority_or_split_failure"] is False
    assert payload["registered_gpus_requested"] == 1
    assert payload["registered_memory_gib"] == 96
    assert "automatically assigns 126 GB" in payload["observed_scheduler_message"]


def test_attempt_003_is_canonical_and_preserves_fail_closed_receipts() -> None:
    raw = ATTEMPT_003.read_bytes()
    payload = json.loads(raw)
    assert raw == canonical_json_bytes(payload)
    assert payload["job_id"] == "1331564"
    assert payload["scheduler_state"] == "FAILED"
    assert payload["status"] == "FAILED_BEFORE_MODEL_CONSTRUCTION"
    assert payload["terminal_status"] == "PRODUCTION_CONTROLLER_EXCEPTION"
    assert payload["final_receipt_self_hash_valid"] is True
    assert payload["journal_hash_chain_valid"] is True
    assert payload["journal_event_count"] == 2


def test_attempt_003_does_not_claim_split_failure_or_authorize_training() -> None:
    payload = json.loads(ATTEMPT_003.read_bytes())
    assert payload["stage"] == "registered_runtime_authentication"
    assert payload["authority_or_split_failure"] is False
    assert payload["model_built"] is False
    assert payload["solver_invoked"] is False
    assert payload["split_gate0_authorized"] is False
    assert payload["training_authorized"] is False
    assert payload["error_message"] == "runtime platform differs from registration"
