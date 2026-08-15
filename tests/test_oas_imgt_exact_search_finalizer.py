"""Test exhaustive fixed-IMGT task replay and global fail-closed finalization."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import sys

import pytest

from antibody_pretraining.imgt_search_output import (
    FIXED_IMGT_MATCH_RECORD,
    FixedIMGTOutputMatch,
    pack_fixed_imgt_match,
)
from antibody_pretraining.near_identity_export import publish_export_receipt


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/finalize_oas_imgt_exact_search.py"
SPEC = importlib.util.spec_from_file_location("oas_imgt_exact_search_finalizer", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


TEST_ACCESS = MODULE.TEST_ACCESS
SOURCES = {
    "controller": "a" * 64,
    "kernel_source": "b" * 64,
    "match_schema": "c" * 64,
}
KERNEL_BINARY = "d" * 64
QUERY_BINARY = "e" * 64


def _match(task_id: int) -> FixedIMGTOutputMatch:
    return FixedIMGTOutputMatch(
        identical_positions=180,
        overlapping_positions=200,
        query_ordinal=task_id,
        query_positions=200,
        target_authority_ordinal=10_000 + task_id,
        target_positions=200,
        target_split="train" if task_id % 2 == 0 else "validation",
        union_positions=200,
    )


def _freeze(root: Path) -> None:
    for path in root.rglob("*"):
        path.chmod(path.lstat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    root.chmod(root.lstat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def _task(
    search_root: Path,
    *,
    task_id: int,
    target_receipt_sha256: str,
    query_receipt_sha256: str,
    kernel_binary_sha256: str = KERNEL_BINARY,
) -> None:
    root = search_root / "fixed-imgt" / f"mapper-{task_id:04d}"
    root.mkdir(parents=True)
    match = _match(task_id)
    match_payload = pack_fixed_imgt_match(match)
    (root / "matches.bin").write_bytes(match_payload)
    split_counts = {
        "train": int(match.target_split == "train"),
        "validation": int(match.target_split == "validation"),
    }
    receipt = {
        "claim_boundary": {
            "fixed_imgt_task_domain_complete": True,
            "full_200_position_verification_complete": True,
            "gap_aware_search_complete": False,
            "global_fixed_imgt_search_complete": False,
            "large_scale_pretraining_authorized": False,
            "near_identity_decontamination_complete": False,
            "scientific_claim": False,
        },
        "format_version": MODULE.TASK_FORMAT_VERSION,
        "independent_output_verification": {
            "match_records": 1,
            "matched_query_count": 1,
            "matched_target_count": 1,
            "split_match_counts": split_counts,
        },
        "input": {
            "global_target_receipt_sha256": target_receipt_sha256,
            "query_binary_sha256": QUERY_BINARY,
            "query_receipt_sha256": query_receipt_sha256,
            "target_binary_sha256": hashlib.sha256(f"target-{task_id}".encode()).hexdigest(),
            "target_records": 1,
        },
        "kernel_stats": {
            "format_version": "oas_imgt_exact_cuda_stats_v1",
            "matches": 1,
            "query_records": 3_103,
            "records_scanned": 1,
            "seed_lookups": 25,
            "target_file_records": 1,
        },
        "output": {
            "match_record_bytes": FIXED_IMGT_MATCH_RECORD.size,
            "matches_bin_bytes": len(match_payload),
            "matches_bin_sha256": hashlib.sha256(match_payload).hexdigest(),
        },
        "resource_receipt": {
            "cuda_visible_devices": "0",
            "gpu_compute_permitted": True,
            "scheduler_reserved_gpu_count": 1,
        },
        "source_sha256": {**SOURCES, "kernel_binary": kernel_binary_sha256},
        "task_id": task_id,
        "test_access": TEST_ACCESS,
    }
    publish_export_receipt(root / "receipt.json", receipt)
    _freeze(root)


def _global_inputs(tmp_path: Path) -> tuple[Path, Path, str, str]:
    target_root = tmp_path / "targets"
    query_root = tmp_path / "queries"
    target_root.mkdir()
    query_root.mkdir()
    publish_export_receipt(
        target_root / "global_receipt.json",
        {
            "claim_boundary": {
                "accepted_oas_domain_partition_complete": True,
                "global_task_set_complete": True,
                "near_identity_search_complete": False,
            },
            "counts": {"representable_records": 32},
            "format_version": MODULE.TARGET_FORMAT_VERSION,
            "task_count": 32,
        },
    )
    publish_export_receipt(
        query_root / "EXPORT_RECEIPT.json",
        {
            "fixed_imgt_query_count": 3_103,
            "format_version": MODULE.QUERY_FORMAT_VERSION,
            "gap_aware_query_count": 3_106,
            "outside_canonical_query_count": 3,
            "test_access": TEST_ACCESS,
        },
    )
    target = json.loads((target_root / "global_receipt.json").read_text(encoding="ascii"))
    query = json.loads((query_root / "EXPORT_RECEIPT.json").read_text(encoding="ascii"))
    return target_root, query_root, target["receipt_sha256"], query["receipt_sha256"]


def test_global_finalizer_replays_32_frozen_tasks_and_preserves_claim_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    search_root = tmp_path / "search"
    (search_root / "work").mkdir(parents=True)
    target_root, query_root, target_receipt, query_receipt = _global_inputs(tmp_path)
    for task_id in range(32):
        _task(
            search_root,
            task_id=task_id,
            target_receipt_sha256=target_receipt,
            query_receipt_sha256=query_receipt,
        )
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("SLURM_JOB_GPUS", raising=False)
    monkeypatch.delenv("SLURM_STEP_GPUS", raising=False)
    monkeypatch.delenv("OAS_SCHEDULER_RESERVED_GPU_COUNT", raising=False)

    receipt = MODULE.finalize(
        search_output_root=search_root,
        target_output_root=target_root,
        query_root=query_root,
        task_count=32,
        worker_count=4,
        expected_source_hashes=SOURCES,
        finalizer_source_sha256="f" * 64,
    )

    assert receipt["match_summary"] == {
        "match_records": 32,
        "split_match_counts": {"train": 16, "validation": 16},
        "unique_pairs": 32,
    }
    assert receipt["target_records"] == 32
    assert receipt["claim_boundary"]["fixed_imgt_representable_target_search_complete"]
    assert receipt["claim_boundary"]["gap_aware_search_complete"] is False
    assert receipt["claim_boundary"]["near_identity_decontamination_complete"] is False
    assert receipt["claim_boundary"]["large_scale_pretraining_authorized"] is False
    assert not (
        (search_root / "fixed_imgt_global_receipt.json").stat().st_mode
        & stat.S_IWUSR
    )


def test_task_auditor_rejects_writable_evidence(tmp_path: Path) -> None:
    search_root = tmp_path / "search"
    target_root, query_root, target_receipt, query_receipt = _global_inputs(tmp_path)
    del target_root, query_root
    _task(
        search_root,
        task_id=0,
        target_receipt_sha256=target_receipt,
        query_receipt_sha256=query_receipt,
    )
    root = search_root / "fixed-imgt" / "mapper-0000"
    root.chmod(0o700)
    with pytest.raises(MODULE.OASIMGTSearchFinalizeError, match="writable"):
        MODULE._audit_task(
            search_root,
            task_id=0,
            expected_source_hashes=SOURCES,
            target_receipt_sha256=target_receipt,
            query_receipt_sha256=query_receipt,
        )


def test_global_finalizer_rejects_one_task_using_a_different_binary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    search_root = tmp_path / "search"
    (search_root / "work").mkdir(parents=True)
    target_root, query_root, target_receipt, query_receipt = _global_inputs(tmp_path)
    for task_id in range(32):
        _task(
            search_root,
            task_id=task_id,
            target_receipt_sha256=target_receipt,
            query_receipt_sha256=query_receipt,
            kernel_binary_sha256=("9" * 64 if task_id == 31 else KERNEL_BINARY),
        )
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("SLURM_JOB_GPUS", raising=False)
    monkeypatch.delenv("SLURM_STEP_GPUS", raising=False)
    monkeypatch.delenv("OAS_SCHEDULER_RESERVED_GPU_COUNT", raising=False)

    with pytest.raises(MODULE.OASIMGTSearchFinalizeError, match="not identical"):
        MODULE.finalize(
            search_output_root=search_root,
            target_output_root=target_root,
            query_root=query_root,
            task_count=32,
            worker_count=4,
            expected_source_hashes=SOURCES,
            finalizer_source_sha256="f" * 64,
        )


def test_finalizer_launcher_hides_scheduler_mandated_gpu() -> None:
    launcher = (
        Path(__file__).resolve().parents[1]
        / "slurm/oas_imgt_exact_search_finalize.sbatch"
    ).read_text(encoding="utf-8")
    assert "export CUDA_VISIBLE_DEVICES=" in launcher
    assert "export OAS_SCHEDULER_RESERVED_GPU_COUNT=1" in launcher
    assert "--task-count 32" in launcher
    assert "--worker-count 8" in launcher
    assert "OAS_PYTHON_SHA256" in launcher
    assert '"${OAS_PYTHON_BIN}" "${FINALIZER}"' in launcher
