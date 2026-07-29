"""Tests for superseding V6-C cluster compatibility evidence."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT
    / "protocols"
    / "v6_c_cluster_compatibility_evidence_v2.json"
)


def _evidence() -> dict[str, object]:
    payload = EVIDENCE.read_bytes()
    assert payload.isascii()
    return json.loads(payload.decode("ascii"))


def test_superseding_cluster_evidence_is_non_authorizing() -> None:
    evidence = _evidence()
    assert evidence["format_version"] == (
        "v6_c_cluster_compatibility_evidence_v2"
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


def test_superseding_source_snapshot_is_exact_and_read_only() -> None:
    source = _evidence()["source"]
    assert source["commit"] == (
        "89b906090e650608ba32dbbdb1bf3e3a9ef0a908"
    )
    assert source["git_tree_sha1"] == (
        "5a47a985c71df010af4b230ae0e160a37bfbea36"
    )
    assert source["remote_root"].endswith("89b906090e65")
    assert source["remote_root_mode"] == "dr-x------"
    assert source["remote_file_count"] == 809
    assert source["git_archive_exact_commit"] is True
    assert source["local_remote_manifest_agreement"] is True
    for name in (
        "controller_sha256",
        "diagnostic_sha256",
        "direct_highs_registration_validator_sha256",
        "freezer_sha256",
        "launcher_sha256",
        "policy_sha256",
        "production_audit_sha256",
        "registration_validator_sha256",
        "source_manifest_sha256",
    ):
        assert len(source[name]) == 64


def test_superseding_runtime_and_lineage_are_explicit() -> None:
    evidence = _evidence()
    runtime = evidence["runtime"]
    assert runtime["python_version"] == "3.10.12"
    assert runtime["ortools_version"] == "9.15.6755"
    assert runtime["python_executable"].endswith(
        "/cp_sat_venv/bin/python"
    )
    assert evidence["supersedes"] == {
        "evidence_path": (
            "protocols/v6_c_cluster_compatibility_evidence_v1.json"
        ),
        "reason": (
            "V6-C now binds the exact Direct-HiGHS registration and its "
            "validator source."
        ),
        "source_commit": (
            "144ad81a777c7efb79990cc4c50a27204375a065"
        ),
    }
