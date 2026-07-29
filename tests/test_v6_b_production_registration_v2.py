"""Tests for scheduler-compatible V6-B production registration V2."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from interaction_first.v6_b_production import (
    canonical_json_bytes,
    validate_registration,
)


ROOT = Path(__file__).resolve().parents[1]
REGISTRATION = ROOT / "protocols" / "v6_b_production_feasibility_registration_v2.json"


def test_v2_registration_is_canonical_valid_and_non_authorizing() -> None:
    raw = REGISTRATION.read_bytes()
    payload = json.loads(raw)
    assert raw == canonical_json_bytes(payload)
    assert validate_registration(payload) == payload
    assert all(value is False for value in payload["authorization"].values())


def test_v2_registration_binds_implementation_commit_bytes() -> None:
    payload = json.loads(REGISTRATION.read_bytes())
    source = payload["source"]
    assert source["implementation_commit"] == (
        "5df239d0efba88028a569868b584242cccc3c491"
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


def test_v2_allocates_one_gpu_but_hides_it_from_the_solver() -> None:
    payload = json.loads(REGISTRATION.read_bytes())
    slurm = payload["slurm"]
    assert slurm["gpus_requested"] == 1
    assert slurm["allocated_gpu_purpose"] == "site_scheduler_requirement_only"
    assert slurm["accelerator_visible_to_solver"] is False
    launcher = (
        ROOT / payload["source"]["files"]["slurm_launcher"]["path"]
    ).read_text(encoding="ascii")
    assert "#SBATCH --gpus=1" in launcher
    assert "export CUDA_VISIBLE_DEVICES=" in launcher
    assert "OMP_NUM_THREADS=1" in launcher


def test_v2_preserves_v1_scientific_inputs_and_solver_limits() -> None:
    v1 = json.loads(
        (ROOT / "protocols" / "v6_b_production_feasibility_registration_v1.json")
        .read_bytes()
    )
    v2 = json.loads(REGISTRATION.read_bytes())
    assert v2["authority"] == v1["authority"]
    assert v2["raw_inputs"] == v1["raw_inputs"]
    assert v2["limits"] == v1["limits"]
    assert v2["runtime"] == v1["runtime"]
    assert {
        key: value["sha256"]
        for key, value in v2["parent_evidence"].items()
        if "sha256" in value
    } == {
        key: value["sha256"]
        for key, value in v1["parent_evidence"].items()
        if "sha256" in value
    }

