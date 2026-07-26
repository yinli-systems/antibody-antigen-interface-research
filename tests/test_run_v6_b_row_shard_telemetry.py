"""Tests for the V6-B create-only row-shard production controller."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "run_v6_b_row_shard_telemetry.py"
SPEC = importlib.util.spec_from_file_location("v6_b_row_shard", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
SCRIPT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SCRIPT
SPEC.loader.exec_module(SCRIPT)


def test_verified_read_rejects_a_byte_change(tmp_path: Path) -> None:
    path = tmp_path / "input.json"
    path.write_bytes(b"{}\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert SCRIPT.read_verified(path, digest, "input") == b"{}\n"
    path.write_bytes(b'{"changed":true}\n')
    with pytest.raises(SCRIPT.V6BRowShardError, match="SHA-256 differs"):
        SCRIPT.read_verified(path, digest, "input")


def test_verified_read_rejects_a_symbolic_path_component(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    payload = source / "input.json"
    payload.write_bytes(b"{}\n")
    link = tmp_path / "link"
    link.symlink_to(source, target_is_directory=True)
    digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    with pytest.raises(SCRIPT.V6BRowShardError, match="symbolic link"):
        SCRIPT.read_verified(link / "input.json", digest, "input")


def test_create_only_publication_is_immutable_and_non_overwriting(
    tmp_path: Path,
) -> None:
    receipt = {
        "format_version": SCRIPT.FORMAT_VERSION,
        "terminal_status": "TELEMETRY_HEADROOM_PASSED",
        "training_authorized": False,
    }
    output = tmp_path / "evidence"
    receipt_path, report_path = SCRIPT.publish_create_only(
        output,
        receipt,
        b"# report\n",
    )
    assert json.loads(receipt_path.read_text(encoding="ascii")) == receipt
    assert report_path.read_bytes() == b"# report\n"
    assert os.stat(receipt_path).st_mode & 0o222 == 0
    assert os.stat(report_path).st_mode & 0o222 == 0
    assert os.stat(output).st_mode & 0o222 == 0
    with pytest.raises(SCRIPT.V6BRowShardError, match="already exists"):
        SCRIPT.publish_create_only(output, receipt, b"# second\n")


def test_markdown_repeats_non_authorizing_boundary() -> None:
    core = {
        "terminal_status": "TELEMETRY_HEADROOM_PASSED",
        "telemetry": {
            "full_variable_count": 10,
            "full_row_count": 20,
            "full_nonzero_count": 30,
            "row_selection": {"selected_row_count": 2},
            "serialized_full_model_bytes": 40,
            "observed_peak_rss_bytes": 50,
            "projected_peak_bytes": 60,
            "available_memory_bytes": 100,
            "resource_headroom_passed": True,
        },
        "partition": {
            "master": {"row_count": 5},
            "inner_row_counts": [5, 5, 5],
        },
    }
    report = SCRIPT._markdown(core).decode("ascii")
    assert "invoked no solver" in report
    assert "does not" in report
    assert "authorize feasibility" in report
