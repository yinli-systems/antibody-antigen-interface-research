"""Tests for the single frozen V6-B production feasibility registration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from interaction_first.v6_b_production import (
    canonical_json_bytes,
    validate_registration,
)


ROOT = Path(__file__).resolve().parents[1]
REGISTRATION = ROOT / "protocols" / "v6_b_production_feasibility_registration_v1.json"


def test_registration_is_canonical_valid_and_non_authorizing() -> None:
    raw = REGISTRATION.read_bytes()
    payload = json.loads(raw)
    assert raw == canonical_json_bytes(payload)
    assert validate_registration(payload) == payload
    assert payload["authorization"] == {
        "split_gate0_authorized": False,
        "test_access_authorized": False,
        "training_authorized": False,
    }


def test_registration_binds_every_local_source_byte() -> None:
    payload = json.loads(REGISTRATION.read_bytes())
    source = payload["source"]
    assert source["implementation_commit"] == (
        "9aeeb76c602ebb428cc4348b17b9c2c59a53048d"
    )
    for reference in source["files"].values():
        path = ROOT / reference["path"]
        assert path.is_file() and not path.is_symlink()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == reference["sha256"]


def test_registration_reproduces_parent_protocol_hashes_and_frozen_limits() -> None:
    payload = json.loads(REGISTRATION.read_bytes())
    parent = payload["parent_evidence"]
    for label, relative in (
        ("v6_b_policy", "protocols/v6_b_logic_based_decomposition_policy_v1.json"),
        ("v6_b_protocol", "protocols/V6_B_LOGIC_BASED_DECOMPOSITION_PROTOCOL.md"),
    ):
        observed = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        assert observed == parent[label]["sha256"]
    assert payload["limits"] == {
        "maximum_combined_candidates": 10000,
        "maximum_master_iterations": 1000,
        "maximum_time_seconds_per_solve": 21600,
    }
    assert payload["slurm"]["solver_workers"] == 1
    assert payload["slurm"]["gpus_requested"] == 0
