"""Tests for frozen V6-B full projection bundle export registration."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from interaction_first.v6_b_projection_bundle_export import (
    canonical_json_bytes,
    validate_export_registration,
)


ROOT = Path(__file__).resolve().parents[1]
REGISTRATION = (
    ROOT / "protocols" / "v6_b_projection_bundle_export_registration_v1.json"
)


def test_bundle_export_registration_is_canonical_and_non_authorizing() -> None:
    raw = REGISTRATION.read_bytes()
    payload = json.loads(raw)
    assert raw == canonical_json_bytes(payload)
    assert validate_export_registration(payload) == payload
    assert all(value is False for value in payload["authorization"].values())
    assert payload["slurm"]["solver_workers"] == 0
    assert payload["slurm"]["accelerator_visible_to_process"] is False


def test_bundle_export_registration_binds_formal_cp_identity() -> None:
    payload = json.loads(REGISTRATION.read_bytes())
    assert payload["production_registration"]["sha256"] == (
        "22de9c83928ee82cba7c90aa1ae8b2f34943cb42c21ca283fea3ac14b3fbf12d"
    )
    assert payload["formal_cp_result"]["sha256"] == (
        "8a7148c1b4888b3e03dced89fd5452efbf1a058ae92ef2eb5fc827f5d773854d"
    )
    assert payload["formal_cp_result"]["receipt_sha256"] == (
        "fcdb3ee3fd776fff9f38348b5a441dd5e19553c158093c838b0c6da20e158725"
    )
    assert payload["expected"]["ir_sha256"] == (
        "3fff0ee63e0346034ba0bc9ce306212523adc786f013aaf37a73bf57848ca05d"
    )
    assert payload["expected"]["dimensions"]["linear_constraints"] == 2693786


def test_bundle_export_registration_binds_pushed_source_commit_bytes() -> None:
    payload = json.loads(REGISTRATION.read_bytes())
    source = payload["source"]
    assert source["implementation_commit"] == (
        "adcf93881afe4734574b236301683160cb4314f4"
    )
    for reference in source["files"].values():
        observed = subprocess.run(
            ["git", "show", f"{source['implementation_commit']}:{reference['path']}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
        assert hashlib.sha256(observed).hexdigest() == reference["sha256"]
