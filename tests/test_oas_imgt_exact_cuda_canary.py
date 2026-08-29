"""Test the native fixed-IMGT CUDA boundary fixture and audit contract."""

from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_oas_imgt_exact_cuda_canary.py"
SPEC = importlib.util.spec_from_file_location("oas_imgt_exact_cuda_canary", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
OASIMGTCUDACanaryError = MODULE.OASIMGTCUDACanaryError
audit = MODULE.audit
prepare = MODULE.prepare


def _kernel_stats(path: Path, *, matches: int) -> None:
    path.write_text(
        json.dumps(
            {
                "batch_records": 7,
                "batches": 1,
                "bucket_member_visits": 1,
                "cuda_device": "synthetic-test-device",
                "cuda_device_index": 0,
                "elapsed_seconds": 0.1,
                "format_version": "oas_imgt_exact_cuda_stats_v1",
                "matches": matches,
                "query_records": 3,
                "quorum_candidates": matches,
                "records_scanned": 7,
                "seed_lookups": 175,
                "target_file_records": 7,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="ascii",
    )


def test_canary_freezes_boundary_membership_and_audits_exact_bytes(tmp_path: Path) -> None:
    root = tmp_path / "fixtures"
    receipt = prepare(root)
    assert receipt["expected_match_count"] == 11
    assert receipt["test_cases"] == {
        "cross_chain_label_recall": True,
        "five_seed_false_positive_rejected": True,
        "gap_missing_end_equivalence": True,
        "identity_boundary_20_disagreements": True,
        "identity_negative_21_disagreements": True,
        "reciprocal_coverage_negative": True,
    }
    actual = tmp_path / "actual.bin"
    actual.write_bytes((root / "expected_matches.bin").read_bytes())
    stats = tmp_path / "stats.json"
    _kernel_stats(stats, matches=11)

    result = audit(root, actual, stats)
    assert result["result"] == "PASS"
    assert result["actual_match_count"] == 11


def test_canary_rejects_one_native_output_byte_change(tmp_path: Path) -> None:
    root = tmp_path / "fixtures"
    prepare(root)
    payload = bytearray((root / "expected_matches.bin").read_bytes())
    payload[-1] ^= 1
    actual = tmp_path / "actual.bin"
    actual.write_bytes(payload)
    stats = tmp_path / "stats.json"
    _kernel_stats(stats, matches=11)

    with pytest.raises((OASIMGTCUDACanaryError, RuntimeError)):
        audit(root, actual, stats)


def test_cuda_sources_preserve_lossless_and_resource_contracts() -> None:
    root = Path(__file__).resolve().parents[1]
    kernel = (root / "native/oas_imgt_exact_search.cu").read_text(encoding="utf-8")
    launcher = (root / "slurm/oas_imgt_exact_search_array.sbatch").read_text(
        encoding="utf-8"
    )
    canary_launcher = (
        root / "slurm/oas_imgt_exact_search_canary.sbatch"
    ).read_text(encoding="utf-8")

    assert "kSeedGroups = 25" in kernel
    assert "counts[cell] < 5" in kernel
    assert "10 * same >= 9 * either" in kernel
    assert "group_tag(group)" in kernel
    assert "chain_tag" not in kernel
    assert "OAS_KERNEL_BUNDLE_ROOT" in launcher
    assert "OAS_KERNEL_BINARY_SHA256" in launcher
    assert "nvcc \\" not in launcher
    assert "#SBATCH --array=0-31%32" in launcher
    assert "--batch-records 32768" in launcher
    assert "source /etc/profile.d/modules.sh" in launcher
    assert "source /etc/profile >/dev/null" not in launcher
    assert "OAS_PYTHON_SHA256" in launcher
    assert '"${OAS_PYTHON_BIN}" "${CONTROLLER}"' in launcher
    assert "-gencode arch=compute_120,code=sm_120" in canary_launcher
    assert "--batch-records 7" in canary_launcher
    assert "source /etc/profile.d/modules.sh" in canary_launcher
    assert "source /etc/profile >/dev/null" not in canary_launcher
    assert "OAS_PYTHON_SHA256" in canary_launcher
    assert '"${OAS_PYTHON_BIN}" "${CANARY_SCRIPT}" prepare' in canary_launcher


def test_published_rtx5090_canary_receipt_is_self_hashed_and_scope_bounded() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "results/pretraining/oas_imgt_exact_cuda_canary_1408348.json"
    )
    receipt = json.loads(path.read_text(encoding="utf-8"))
    claimed = receipt.pop("receipt_sha256")
    canonical = (
        json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("ascii")
    assert hashlib.sha256(canonical).hexdigest() == claimed
    assert receipt["result"] == "PASS"
    assert receipt["runtime"]["job_id"] == 1408348
    assert receipt["kernel_stats"]["matches"] == 11
    assert (
        receipt["artifacts"]["actual_matches"]["sha256"]
        == receipt["artifacts"]["expected_matches_sha256"]
    )
    assert receipt["claim_boundary"] == {
        "cuda_kernel_boundary_canary_passed": True,
        "fixed_imgt_full_oas_search_complete": False,
        "frozen_official_external_test_accessed": False,
        "gap_aware_search_complete": False,
        "large_scale_pretraining_authorized": False,
        "near_identity_decontamination_complete": False,
        "scientific_claim": False,
    }
