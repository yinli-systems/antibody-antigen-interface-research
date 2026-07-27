"""Tests for immutable V6-B scheduler submission evidence."""

from __future__ import annotations

import json
from pathlib import Path

from interaction_first.v6_b_production import canonical_json_bytes


ROOT = Path(__file__).resolve().parents[1]
ATTEMPT = ROOT / "protocols" / "v6_b_production_submission_attempt_001.json"
ATTEMPT_002 = ROOT / "protocols" / "v6_b_production_submission_attempt_002.json"
ATTEMPT_003 = ROOT / "protocols" / "v6_b_production_attempt_003.json"
ATTEMPT_004_SUBMISSION = (
    ROOT / "protocols" / "v6_b_production_attempt_004_submission.json"
)


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


def test_attempt_004_submission_snapshot_is_canonical_and_non_authorizing() -> None:
    raw = ATTEMPT_004_SUBMISSION.read_bytes()
    payload = json.loads(raw)
    assert raw == canonical_json_bytes(payload)
    assert payload["job_id"] == "1331619"
    assert payload["scheduler_state"] == "RUNNING"
    assert payload["registration_sha256"] == (
        "22de9c83928ee82cba7c90aa1ae8b2f34943cb42c21ca283fea3ac14b3fbf12d"
    )
    assert payload["journal_hash_chain_valid"] is True
    assert payload["authenticated_raw_authority_reconstructed"] is True
    assert payload["split_gate0_authorized"] is False
    assert payload["training_authorized"] is False


def test_attempt_004_snapshot_does_not_pretend_to_be_a_solver_result() -> None:
    payload = json.loads(ATTEMPT_004_SUBMISSION.read_bytes())
    assert payload["model_built_at_snapshot"] is False
    assert payload["solver_invoked_at_snapshot"] is False
    assert payload["journal_event_types"] == [
        "ATTEMPT_STARTED",
        "RUNTIME_SOURCE_AND_PARENT_AUTHENTICATED",
        "RAW_AUTHORITY_RECONSTRUCTED",
    ]
    assert payload["accelerator_visible_to_solver"] is False
    assert payload["submitted_resources"]["solver_workers"] == 1
