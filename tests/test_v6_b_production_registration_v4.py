"""Tests for stable-ABI V6-B production registration V4."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from interaction_first.v6_b_production import canonical_json_bytes, validate_registration


ROOT = Path(__file__).resolve().parents[1]
REGISTRATION = ROOT / "protocols" / "v6_b_production_feasibility_registration_v4.json"


def test_v4_registration_is_canonical_valid_and_non_authorizing() -> None:
    raw = REGISTRATION.read_bytes()
    payload = json.loads(raw)
    assert raw == canonical_json_bytes(payload)
    assert validate_registration(payload) == payload
    assert all(value is False for value in payload["authorization"].values())


def test_v4_registration_binds_implementation_commit_bytes() -> None:
    payload = json.loads(REGISTRATION.read_bytes())
    source = payload["source"]
    assert source["implementation_commit"] == (
        "7b77974dafd545ae2a2feac056201eb58408157e"
    )
    for reference in source["files"].values():
        result = subprocess.run(
            ["git", "show", f"{source['implementation_commit']}:{reference['path']}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        )
        assert hashlib.sha256(result.stdout).hexdigest() == reference["sha256"]


def test_v4_binds_stable_abi_module_bytes_and_probe() -> None:
    payload = json.loads(REGISTRATION.read_bytes())
    abi = payload["runtime"]["platform_abi"]
    assert abi["system"] == "Linux"
    assert abi["machine"] == "x86_64"
    assert [abi["libc_family"], abi["libc_version"]] == ["glibc", "2.35"]
    assert abi["kernel_release_bound"] is False
    assert abi["ortools_module_sha256"] == (
        "67121c783ab4f9bf749a4e667d49458f885d105cc42c3805a1154faf10d329bf"
    )
    probe = ROOT / "protocols" / "v6_b_runtime_compatibility_probe_v1.json"
    assert hashlib.sha256(probe.read_bytes()).hexdigest() == (
        abi["compatibility_probe_sha256"]
    )


def test_v4_preserves_v3_science_limits_runtime_package_and_slurm() -> None:
    v3 = json.loads(
        (ROOT / "protocols" / "v6_b_production_feasibility_registration_v3.json")
        .read_bytes()
    )
    v4 = json.loads(REGISTRATION.read_bytes())
    assert v4["authority"] == v3["authority"]
    assert v4["raw_inputs"] == v3["raw_inputs"]
    assert v4["limits"] == v3["limits"]
    assert v4["slurm"] == v3["slurm"]
    for field in (
        "python_executable",
        "python_version",
        "ortools_version",
        "cp_sat_runtime_receipt_path",
        "cp_sat_runtime_receipt_sha256",
    ):
        assert v4["runtime"][field] == v3["runtime"][field]
