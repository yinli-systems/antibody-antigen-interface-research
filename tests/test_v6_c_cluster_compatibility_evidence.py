"""Tests for bounded V6-C cluster compatibility evidence."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT
    / "protocols"
    / "v6_c_cluster_compatibility_evidence_v1.json"
)


def test_cluster_compatibility_evidence_is_bounded() -> None:
    payload = EVIDENCE.read_bytes()
    assert payload.isascii()
    evidence = json.loads(payload.decode("ascii"))
    assert evidence["format_version"] == (
        "v6_c_cluster_compatibility_evidence_v1"
    )
    assert set(evidence["authorization"].values()) == {False}
    assert evidence["smoke"] == {
        "all_families_status": "INFEASIBLE",
        "core_replay_status": "INFEASIBLE",
        "irreducibility_proven": True,
        "production_solver_work": False,
        "scientific_work": False,
        "sufficient_core": ["one", "zero"],
        "synthetic_model_only": True,
        "synthetic_solver_smoke": True,
        "test_bytes_accessed": False,
    }


def test_cluster_source_snapshot_is_content_addressed() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="ascii"))
    source = evidence["source"]
    assert source["commit"] == "144ad81a777c7efb79990cc4c50a27204375a065"
    assert source["remote_root"].endswith("144ad81a777c")
    assert source["remote_root_mode"] == "dr-x------"
    assert source["remote_file_count"] == 807
    for name in (
        "controller_sha256",
        "diagnostic_sha256",
        "launcher_sha256",
        "policy_sha256",
        "production_audit_sha256",
    ):
        assert len(source[name]) == 64


def test_cluster_runtime_matches_locked_cp_sat_environment() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="ascii"))
    runtime = evidence["runtime"]
    assert runtime["python_version"] == "3.10.12"
    assert runtime["ortools_version"] == "9.15.6755"
    assert runtime["python_executable"].endswith(
        "/cp_sat_venv/bin/python"
    )
