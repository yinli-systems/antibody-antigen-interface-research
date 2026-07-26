"""Tests for the V6-A production solver runtime lock controller."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "lock_v6_solver_runtime.py"
SPEC = importlib.util.spec_from_file_location("lock_v6_solver_runtime", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
SCRIPT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SCRIPT
SPEC.loader.exec_module(SCRIPT)


def test_both_integer_solver_smoke_probes_are_exact_optimal() -> None:
    receipt = SCRIPT.solver_smoke_receipt()
    assert receipt["native_library_process_isolation"] is True
    assert receipt["cp_sat"]["status"] == "OPTIMAL"
    assert receipt["cp_sat"]["num_search_workers"] == 1
    assert receipt["cp_sat"]["relative_gap_limit"] == "0"
    assert receipt["highs"]["status"] == "kOptimal"
    assert receipt["highs"]["mip_gap"] == "0.0"
    assert receipt["highs"]["locked_options"] == SCRIPT.LOCKED_HIGHS_OPTIONS


def test_wheel_metadata_is_derived_from_archive_bytes(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.whl"
    invalid.write_bytes(b"not a wheel")
    with pytest.raises(SCRIPT.RuntimeLockError, match="invalid wheel archive"):
        SCRIPT._wheel_metadata(invalid.read_bytes(), invalid.name)


def test_create_only_writer_does_not_replace_bytes(tmp_path: Path) -> None:
    output = tmp_path / "receipt.json"
    SCRIPT._write_exclusive(output, b"first\n")
    assert output.read_bytes() == b"first\n"
    assert output.stat().st_mode & 0o777 == 0o400
    with pytest.raises(FileExistsError):
        SCRIPT._write_exclusive(output, b"second\n")
    assert output.read_bytes() == b"first\n"


def test_receipt_hash_is_non_self_referential() -> None:
    payload = {
        "format_version": SCRIPT.FORMAT_VERSION,
        "training_authorized": False,
    }
    payload["receipt_sha256"] = hashlib.sha256(
        SCRIPT._canonical_json_bytes(payload)
    ).hexdigest()
    claimed = payload.pop("receipt_sha256")
    assert hashlib.sha256(SCRIPT._canonical_json_bytes(payload)).hexdigest() == claimed
    assert json.loads(SCRIPT._canonical_json_bytes(payload)) == payload
