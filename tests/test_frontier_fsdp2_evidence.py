"""Validate downloaded FSDP2/DCP execution evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_ROOT = REPOSITORY_ROOT / "results/frontier_fsdp2_dcp"


def _sha256(path: Path) -> str:
    """Return a file SHA-256 digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _remote_inventory(path: Path) -> dict[str, str]:
    """Parse a remote sha256sum inventory by path suffix."""
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, remote = line.split(None, 1)
        result[remote.strip()] = digest
    return result


def test_execution_summary_preserves_claim_boundary() -> None:
    summary = json.loads(
        (EVIDENCE_ROOT / "execution_summary.json").read_text(encoding="utf-8")
    )
    assert summary["format"] == "frontier_fsdp2_dcp_execution_summary_v1"
    assert summary["status"] == "fp32_one_and_eight_gpu_correctness_passed"
    assert summary["test_access"] == {
        "biological_data": False,
        "frozen_test": False,
        "outer_fold": False,
    }
    assert [run["world_size"] for run in summary["passed_runs"]] == [1, 8]
    assert summary["passed_runs"][1]["fsdp2_reference_max_scaled_error"] < 1.0
    assert all(
        run["dcp_restart_max_scaled_error"] == 0.0
        for run in summary["passed_runs"]
    )
    assert "No biological training" in summary["claim_boundary"]


def test_final_downloads_match_remote_artifact_inventories() -> None:
    cases = (
        ("job_1339702_fp32_1gpu", "1339702"),
        ("job_1339704_fp32_8gpu", "1339704"),
    )
    mapping = {
        "environment.txt": "environment.txt",
        "fsdp2.log": "fsdp2.log",
        "gpu.csv": "gpu.csv",
        "launch_manifest.json": "launch_manifest.json",
        "launch_manifest_creation.json": "launch_manifest_creation.json",
        "runtime_verification.json": "runtime_verification.json",
        "fsdp2/result.json": "result.json",
        "fsdp2/checkpoints/step_000003/receipt.json": "checkpoint_receipt.json",
    }
    for folder, job_id in cases:
        root = EVIDENCE_ROOT / folder
        inventory = _remote_inventory(root / "artifact_sha256.txt")
        for remote_suffix, local_name in mapping.items():
            matches = [
                digest
                for remote, digest in inventory.items()
                if remote.endswith(f"/{remote_suffix}")
            ]
            assert len(matches) == 1, (job_id, remote_suffix)
            assert _sha256(root / local_name) == matches[0]


def test_failure_evidence_is_retained_and_pretraining_never_started() -> None:
    drift = json.loads(
        (
            EVIDENCE_ROOT
            / "job_1339683_runtime_drift/runtime_verification.json"
        ).read_text(encoding="utf-8")
    )
    assert drift["valid"] is False
    assert drift["live_state_checked"] is True
    scalar_failure = (
        EVIDENCE_ROOT / "job_1339691_scalar_failure/fsdp2.log"
    ).read_text(encoding="utf-8")
    assert "fully_shard doesn't support scalar parameters" in scalar_failure
    assert "optimizer.step" not in scalar_failure


def test_runtime_lock_matches_both_final_verification_receipts() -> None:
    runtime = EVIDENCE_ROOT / (
        "runtime_attempt_004/frontier_torch213_runtime_20260730_attempt_004.json"
    )
    runtime_sha256 = _sha256(runtime)
    assert runtime_sha256 == (
        "5e8c0ecf201876d0559331299187e4a8b8820124d79bdb24b9973b3b57e24202"
    )
    for folder in ("job_1339702_fp32_1gpu", "job_1339704_fp32_8gpu"):
        verification = json.loads(
            (EVIDENCE_ROOT / folder / "runtime_verification.json").read_text(
                encoding="utf-8"
            )
        )
        assert verification["valid"] is True
        assert verification["live_state_checked"] is True
        assert verification["file_sha256"] == runtime_sha256
