"""Tests for the frozen V6-B independent direct-CP audit registration."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from interaction_first.v6_b_independent_cp_audit import canonical_json_bytes


ROOT = Path(__file__).resolve().parents[1]
REGISTRATION = ROOT / "protocols" / "v6_b_independent_cp_audit_registration_v1.json"


def test_independent_cp_audit_registration_is_canonical_and_non_authorizing() -> None:
    raw = REGISTRATION.read_bytes()
    payload = json.loads(raw)
    assert raw == canonical_json_bytes(payload)
    assert all(value is False for value in payload["authorization"].values())
    assert payload["limits"] == {"maximum_time_seconds": 21600}


def test_independent_cp_audit_registration_binds_controller_commit_bytes() -> None:
    payload = json.loads(REGISTRATION.read_bytes())
    source = payload["source"]
    assert source["implementation_commit"] == (
        "c73630c1a485032dda223c1c24d058ff5c117a2b"
    )
    for reference in source["files"].values():
        observed = subprocess.run(
            ["git", "show", f"{source['implementation_commit']}:{reference['path']}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
        assert hashlib.sha256(observed).hexdigest() == reference["sha256"]


def test_independent_cp_audit_binds_primary_result_and_v4_registration() -> None:
    payload = json.loads(REGISTRATION.read_bytes())
    assert payload["production_registration"]["sha256"] == (
        "22de9c83928ee82cba7c90aa1ae8b2f34943cb42c21ca283fea3ac14b3fbf12d"
    )
    assert payload["primary_result"]["sha256"] == (
        "539cc1e32a12b892ae2fbb2ea9efbc985dc5c8a9ca5e7be51be198efd8c212b6"
    )
    assert payload["module_source"]["implementation_commit"] == (
        "7b77974dafd545ae2a2feac056201eb58408157e"
    )
    assert payload["slurm"]["accelerator_visible_to_solver"] is False
    assert payload["slurm"]["solver_workers"] == 1
