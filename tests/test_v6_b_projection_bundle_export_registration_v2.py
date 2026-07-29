"""Tests for corrected V6-B projection bundle export registration V2."""

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
    ROOT / "protocols" / "v6_b_projection_bundle_export_registration_v2.json"
)


def test_bundle_export_v2_is_canonical_and_non_authorizing() -> None:
    raw = REGISTRATION.read_bytes()
    payload = json.loads(raw)
    assert raw == canonical_json_bytes(payload)
    assert validate_export_registration(payload) == payload
    assert payload["format_version"].endswith("_v2")
    assert all(value is False for value in payload["authorization"].values())


def test_bundle_export_v2_preserves_v1_scientific_identity() -> None:
    payload = json.loads(REGISTRATION.read_bytes())
    historical = json.loads(
        (
            ROOT
            / "protocols"
            / "v6_b_projection_bundle_export_registration_v1.json"
        ).read_bytes()
    )
    for field in (
        "authorization",
        "expected",
        "formal_cp_result",
        "production_registration",
        "runtime",
        "slurm",
    ):
        assert payload[field] == historical[field]
    assert payload["source"]["implementation_commit"] != historical["source"][
        "implementation_commit"
    ]


def test_bundle_export_v2_binds_corrected_pushed_source_bytes() -> None:
    payload = json.loads(REGISTRATION.read_bytes())
    source = payload["source"]
    assert source["implementation_commit"] == (
        "ad1c6eabc19842b46bee6049ddcdb498f881f9b4"
    )
    for reference in source["files"].values():
        observed = subprocess.run(
            ["git", "show", f"{source['implementation_commit']}:{reference['path']}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
        assert hashlib.sha256(observed).hexdigest() == reference["sha256"]
