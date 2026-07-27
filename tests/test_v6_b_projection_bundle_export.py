"""Tests for registered V6-B solver-neutral projection bundle export."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from interaction_first.v6_b_projection_bundle_export import (
    ProjectionBundleExportError,
    _publish,
    canonical_json_bytes,
    sha256_payload,
    validate_export_registration,
    validate_formal_cp_result,
    validate_production_registration_digest,
)


HASH = "a" * 64
COMMIT = "a" * 40


def _registration() -> dict[str, object]:
    return {
        "format_version": "v6_b_linear_projection_bundle_export_registration_v1",
        "authorization": {
            "full_protocol_infeasibility_claim": False,
            "split_gate0": False,
            "training": False,
        },
        "production_registration": {"path": "/a", "sha256": HASH},
        "formal_cp_result": {
            "path": "/b",
            "sha256": HASH,
            "receipt_sha256": HASH,
        },
        "expected": {
            "normalized_authority_sha256": HASH,
            "direct_formulation_sha256": HASH,
            "ir_sha256": HASH,
            "dimensions": {
                "variables": 1,
                "linear_constraints": 2,
                "exact_gate_declarations": 3,
            },
        },
        "runtime": {
            "cp_sat_runtime_receipt_path": "/runtime.json",
            "cp_sat_runtime_receipt_sha256": HASH,
            "python_executable": "/python",
            "python_version": "3.10.12",
            "ortools_version": "9.15.6755",
            "highspy_imported": False,
        },
        "source": {
            "source_root": "/source",
            "implementation_commit": COMMIT,
            "files": {
                name: {"path": f"{name}.py", "sha256": HASH}
                for name in ("controller", "implementation", "bundle", "launcher")
            },
        },
        "slurm": {
            "gpus_requested": 1,
            "solver_workers": 0,
            "accelerator_visible_to_process": False,
            "explicit_memory_request": False,
        },
    }


def _formal() -> dict[str, object]:
    payload = {
        "format_version": "v6_b_independent_full_linear_cp_audit_receipt_v1",
        "registration_sha256": HASH,
        "terminal_status": "INDEPENDENT_DIRECT_CP_INFEASIBLE_AGREEMENT",
        "primary_independent_authority_byte_agreement": True,
        "primary_independent_linear_agreement": True,
        "projection_receipt": {
            "status": "INFEASIBLE",
            "accepted_projection_terminal_status": True,
        },
        "direct_highs_audit_completed": False,
        "full_protocol_infeasibility_claim_authorized": False,
        "split_gate0_authorized": False,
        "training_authorized": False,
    }
    payload["receipt_sha256"] = sha256_payload(payload)
    return payload


def test_export_registration_is_non_authorizing_and_scheduler_compatible() -> None:
    assert validate_export_registration(_registration()) == _registration()
    successor = _registration()
    successor["format_version"] = (
        "v6_b_linear_projection_bundle_export_registration_v2"
    )
    assert validate_export_registration(successor) == successor
    changed = _registration()
    changed["authorization"]["training"] = True
    with pytest.raises(ProjectionBundleExportError, match="authorizes"):
        validate_export_registration(changed)


def test_formal_cp_result_requires_exact_infeasibility_agreement() -> None:
    assert validate_formal_cp_result(
        _formal(),
        expected_registration_sha256=HASH,
    ) == _formal()
    changed = _formal()
    changed["projection_receipt"]["status"] = "UNKNOWN"
    changed["receipt_sha256"] = sha256_payload(
        {key: value for key, value in changed.items() if key != "receipt_sha256"}
    )
    with pytest.raises(ProjectionBundleExportError, match="state differs"):
        validate_formal_cp_result(changed, expected_registration_sha256=HASH)


def test_exporter_uses_frozen_production_registration_hash_api() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "protocols" / "v6_b_production_feasibility_registration_v4.json"
    raw = path.read_bytes()
    checked = validate_production_registration_digest(
        json.loads(raw),
        expected_sha256=hashlib.sha256(raw).hexdigest(),
    )
    assert checked["format_version"] == "v6_b_production_feasibility_registration_v4"
    with pytest.raises(ProjectionBundleExportError, match="digest differs"):
        validate_production_registration_digest(
            checked,
            expected_sha256=HASH,
        )


def test_bundle_export_publication_is_self_hashed_and_read_only(
    tmp_path: Path,
) -> None:
    output = tmp_path / "export"
    output.mkdir()
    nested = output / "bundle"
    nested.mkdir()
    (nested / "artifact").write_bytes(b"evidence")
    receipt = {
        "format_version": "v6_b_linear_projection_bundle_export_receipt_v1",
        "terminal_status": "FULL_LINEAR_PROJECTION_BUNDLE_EXPORTED",
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
        assert os.stat(nested).st_mode & 0o222 == 0
        assert all(os.stat(path).st_mode & 0o222 == 0 for path in nested.iterdir())
    finally:
        os.chmod(output, 0o700)
        os.chmod(nested, 0o700)
        for path in nested.iterdir():
            os.chmod(path, 0o600)


def test_bundle_export_launcher_is_cp_only_and_scheduler_compatible() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "submit_v6_b_projection_bundle_export.sbatch"
    ).read_text("ascii")
    assert "#SBATCH --gpus=1" in script
    assert "#SBATCH --mem" not in script
    assert "export CUDA_VISIBLE_DEVICES=" in script
    assert "export PYTHONHASHSEED=0" in script
    assert "V6B_CP_PYTHON_BIN" in script
    assert "highs_venv" not in script
