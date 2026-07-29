"""Tests for the independent full-linear V6-B CP-SAT audit controller."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from interaction_first.v6_b_independent_cp_audit import (
    IndependentCPAuditError,
    _publish,
    canonical_json_bytes,
    validate_primary_result,
)


REGISTRATION_SHA256 = "a" * 64


def _primary() -> dict[str, object]:
    return {
        "registration_sha256": REGISTRATION_SHA256,
        "terminal_status": "PROVEN_INFEASIBLE",
        "decomposition_result": {
            "terminal_status": "PROVEN_INFEASIBLE",
            "complete": True,
            "proven_infeasible": True,
            "candidate_present": False,
            "combined_candidate_count": 0,
            "all_66_exact_gates_passed": False,
            "oracle_history": [],
        },
        "direct_highs_fixed_assignment_audit_completed": False,
        "split_gate0_authorized": False,
        "training_authorized": False,
    }


def test_primary_result_requires_exact_no_candidate_infeasible_state() -> None:
    assert validate_primary_result(
        _primary(),
        expected_registration_sha256=REGISTRATION_SHA256,
    ) == _primary()
    candidate = _primary()
    candidate["decomposition_result"]["candidate_present"] = True
    with pytest.raises(IndependentCPAuditError, match="state differs"):
        validate_primary_result(
            candidate,
            expected_registration_sha256=REGISTRATION_SHA256,
        )


def test_primary_result_rejects_wrong_registration_or_authority() -> None:
    with pytest.raises(IndependentCPAuditError, match="registration differs"):
        validate_primary_result(_primary(), expected_registration_sha256="b" * 64)
    authorizing = _primary()
    authorizing["training_authorized"] = True
    with pytest.raises(IndependentCPAuditError, match="authorizes training"):
        validate_primary_result(
            authorizing,
            expected_registration_sha256=REGISTRATION_SHA256,
        )


def test_submission_script_is_cpu_only_inside_scheduler_allocation() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "submit_v6_b_independent_cp_audit.sbatch"
    ).read_text(encoding="ascii")
    assert "#SBATCH --gpus=1" in script
    assert "#SBATCH --mem" not in script
    assert "export CUDA_VISIBLE_DEVICES=" in script
    assert "export PYTHONHASHSEED=0" in script
    assert "export OMP_NUM_THREADS=1" in script


def test_audit_publication_is_self_hashed_create_only_and_read_only(
    tmp_path: Path,
) -> None:
    output = tmp_path / "audit"
    output.mkdir()
    receipt = {
        "format_version": "v6_b_independent_full_linear_cp_audit_receipt_v1",
        "terminal_status": "INDEPENDENT_DIRECT_CP_INFEASIBLE_AGREEMENT",
        "split_gate0_authorized": False,
        "training_authorized": False,
    }
    try:
        _publish(output, receipt)
        raw = (output / "receipt.json").read_bytes()
        published = json.loads(raw)
        claimed = published.pop("receipt_sha256")
        assert hashlib.sha256(canonical_json_bytes(published)).hexdigest() == claimed
        assert os.stat(output).st_mode & 0o222 == 0
        assert all(os.stat(path).st_mode & 0o222 == 0 for path in output.iterdir())
    finally:
        if output.exists():
            os.chmod(output, 0o700)
            for path in output.iterdir():
                os.chmod(path, 0o600)
