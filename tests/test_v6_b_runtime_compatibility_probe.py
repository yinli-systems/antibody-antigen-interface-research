"""Tests for the V6-B cross-node runtime compatibility evidence."""

from __future__ import annotations

import json
from pathlib import Path

from interaction_first.v6_b_production import canonical_json_bytes


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "protocols" / "v6_b_runtime_compatibility_probe_v1.json"


def test_runtime_probe_is_canonical_and_non_authorizing() -> None:
    raw = RECEIPT.read_bytes()
    payload = json.loads(raw)
    assert raw == canonical_json_bytes(payload)
    assert all(value is False for value in payload["authorization"].values())
    assert payload["scientific_or_solver_work"] == {
        "model_built": False,
        "solver_invoked": False,
        "training_run": False,
    }


def test_runtime_probe_proves_stable_abi_but_not_stable_kernel() -> None:
    payload = json.loads(RECEIPT.read_bytes())
    assert payload["conclusion"]["runtime_package_and_abi_match_across_observations"]
    assert payload["conclusion"]["exact_kernel_release_is_stable"] is False
    assert payload["stable_equality"]["libc"] == ["glibc", "2.35"]
    assert payload["stable_equality"]["machine"] == "x86_64"
    assert payload["stable_equality"]["ortools_version"] == "9.15.6755"
    assert len({row["kernel_release"] for row in payload["probe_jobs"]}) == 2


def test_only_process_local_cuda_hiding_is_accepted() -> None:
    payload = json.loads(RECEIPT.read_bytes())
    rejected, accepted, attempt_node = payload["probe_jobs"]
    assert rejected["cuda_visible_devices"] == "0"
    assert rejected["outcome"] == "DIAGNOSTIC_REJECTED_VISIBLE_ACCELERATOR"
    assert accepted["cuda_visible_devices"] == ""
    assert attempt_node["cuda_visible_devices"] == ""
    assert attempt_node["node"] == "wqd10nah10g4"
