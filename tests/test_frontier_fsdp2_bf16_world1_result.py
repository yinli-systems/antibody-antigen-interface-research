"""Validate the immutable one-GPU BF16 execution evidence."""

from __future__ import annotations

import json
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_world1_bf16_result_is_numerically_valid_and_non_authorizing() -> None:
    """Accept only the registered tolerance result and its narrow claim scope."""
    payload = json.loads(
        (
            REPOSITORY_ROOT
            / "protocols/frontier_fsdp2_bf16_attempt_003_world1_result.json"
        ).read_text(encoding="utf-8")
    )
    assert payload["status"] == "PASSED"
    assert payload["job"]["scheduler_state"] == "COMPLETED"
    assert payload["job"]["exit_code"] == "0:0"
    assert payload["precision"] == "bfloat16"
    assert payload["checkpoint"]["continuation_max_absolute_error"] == 0.0
    assert payload["parity"]["max_scaled_error"] < 1.0
    assert payload["artifact_audit"]["artifact_manifest_verified"] is True
    assert payload["artifact_audit"]["writable_file_count"] == 0
    assert payload["test_data_accessed"] is False
    assert not any(payload["authorization"].values())
