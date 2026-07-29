"""Tests for scheduler-memory-compatible V6-B registration V3."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from interaction_first.v6_b_production import canonical_json_bytes, validate_registration


ROOT = Path(__file__).resolve().parents[1]
REGISTRATION = ROOT / "protocols" / "v6_b_production_feasibility_registration_v3.json"


def test_v3_registration_is_canonical_valid_and_non_authorizing() -> None:
    raw = REGISTRATION.read_bytes()
    payload = json.loads(raw)
    assert raw == canonical_json_bytes(payload)
    assert validate_registration(payload) == payload
    assert all(value is False for value in payload["authorization"].values())


def test_v3_registration_binds_implementation_commit_bytes() -> None:
    payload = json.loads(REGISTRATION.read_bytes())
    source = payload["source"]
    assert source["implementation_commit"] == (
        "675c95c2a0bd67e5801a07de65fca01bb0ef7362"
    )
    for reference in source["files"].values():
        result = subprocess.run(
            [
                "git",
                "show",
                f"{source['implementation_commit']}:{reference['path']}",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
        )
        assert hashlib.sha256(result.stdout).hexdigest() == reference["sha256"]


def test_v3_uses_scheduler_memory_and_hides_the_allocated_gpu() -> None:
    payload = json.loads(REGISTRATION.read_bytes())
    slurm = payload["slurm"]
    assert slurm["gpus_requested"] == 1
    assert slurm["accelerator_visible_to_solver"] is False
    assert slurm["explicit_memory_request"] is False
    assert slurm["memory_allocation_policy"] == "partition_automatic_per_allocated_gpu"
    assert slurm["scheduler_memory_gb_per_gpu"] == 126
    launcher = subprocess.run(
        [
            "git",
            "show",
            f"{payload['source']['implementation_commit']}:"
            f"{payload['source']['files']['slurm_launcher']['path']}",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "#SBATCH --gpus=1" in launcher
    assert "#SBATCH --mem" not in launcher
    assert "export CUDA_VISIBLE_DEVICES=" in launcher


def test_v3_preserves_v2_scientific_inputs_runtime_and_solver_limits() -> None:
    v2 = json.loads(
        (ROOT / "protocols" / "v6_b_production_feasibility_registration_v2.json")
        .read_bytes()
    )
    v3 = json.loads(REGISTRATION.read_bytes())
    assert v3["authority"] == v2["authority"]
    assert v3["raw_inputs"] == v2["raw_inputs"]
    assert v3["limits"] == v2["limits"]
    assert v3["runtime"] == v2["runtime"]
    assert {
        key: value["sha256"]
        for key, value in v3["parent_evidence"].items()
        if "sha256" in value
    } == {
        key: value["sha256"]
        for key, value in v2["parent_evidence"].items()
        if "sha256" in value
    }
