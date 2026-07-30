"""Validate the frozen public-cluster scientific-parity registration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRATION = ROOT / "protocols/antibody_ddp_scientific_parity_registration_v1.json"


def test_registration_binds_the_complete_executed_source_closure() -> None:
    """Reject source drift across the launcher, runner, and imported package."""
    payload = json.loads(REGISTRATION.read_text(encoding="ascii"))
    assert payload["format_version"] == "antibody_ddp_scientific_parity_registration_v1"
    assert payload["implementation_commit"] == "03d04e57e4ad943ef85edf4084f10b7e39f65d8e"
    for relative, expected in payload["source_files"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected
    assert payload["execution_sha256"] == hashlib.sha256(
        (ROOT / "protocols/antibody_ddp_scientific_parity_execution_v2_1.json").read_bytes()
    ).hexdigest()


def test_registration_uses_public_resources_and_remains_non_authorizing() -> None:
    """Keep production and scale closed before the parity result exists."""
    payload = json.loads(REGISTRATION.read_text(encoding="ascii"))
    assert payload["slurm"] == {
        "cpus_per_task": 32,
        "explicit_memory_request": False,
        "gpus": 8,
        "nodes": 1,
        "partition": "gpu_5090",
        "tasks": 1,
        "wall_time": "03:00:00",
    }
    assert payload["data"]["record_count"] == 8112
    assert len(payload["data"]["files"]) == 5
    assert not any(payload["authorization"].values())
    assert payload["test_access"] == {
        "downstream_labels_used": False,
        "sabdab2_heldout_accessed": False,
    }
