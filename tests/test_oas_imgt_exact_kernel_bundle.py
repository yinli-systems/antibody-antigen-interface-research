"""Test one-build kernel provenance for exhaustive fixed-IMGT search."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import stat
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SEALER = ROOT / "scripts/seal_oas_imgt_exact_kernel_bundle.py"
CANARY = ROOT / "scripts/run_oas_imgt_exact_cuda_canary.py"
TRUE_BINARY = Path(shutil.which("true") or "").resolve()
SPEC = importlib.util.spec_from_file_location("oas_imgt_kernel_bundle", SEALER)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

CANARY_SPEC = importlib.util.spec_from_file_location(
    "oas_imgt_kernel_bundle_test_canary", CANARY
)
assert CANARY_SPEC is not None and CANARY_SPEC.loader is not None
CANARY_MODULE = importlib.util.module_from_spec(CANARY_SPEC)
sys.modules[CANARY_SPEC.name] = CANARY_MODULE
CANARY_SPEC.loader.exec_module(CANARY_MODULE)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bundle(tmp_path: Path) -> Path:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    binary = bundle / "oas_imgt_exact_search"
    shutil.copyfile(TRUE_BINARY, binary)
    binary.chmod(0o700)
    CANARY_MODULE.prepare(bundle / "fixtures")
    shutil.copyfile(
        bundle / "fixtures/expected_matches.bin", bundle / "actual_matches.bin"
    )
    (bundle / "kernel_stats.json").write_text(
        json.dumps(
            {
                "format_version": "oas_imgt_exact_cuda_stats_v1",
                "matches": 11,
                "query_records": 3,
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
    (bundle / "nvcc-version.txt").write_text(
        "Cuda compilation tools, release 12.8, V12.8.93\n", encoding="ascii"
    )
    return bundle


def test_sealer_replays_canary_binds_compiler_and_makes_bundle_immutable(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    kernel_source = ROOT / "native/oas_imgt_exact_search.cu"

    receipt = MODULE.seal(
        bundle_root=bundle,
        kernel_source=kernel_source,
        canary_source=CANARY,
        nvcc_binary=TRUE_BINARY,
        nvcc_version_file=bundle / "nvcc-version.txt",
        expected_kernel_source_sha256=_sha256(kernel_source),
        expected_canary_source_sha256=_sha256(CANARY),
        build_launcher_source_sha256="a" * 64,
        sealer_source_sha256=_sha256(SEALER),
        resource_receipt_override={
            "cuda_visible_devices": "synthetic",
            "gpu_compute_permitted": False,
            "scheduler_reserved_gpu_count": 0,
        },
    )

    assert receipt["claim_boundary"]["canonical_kernel_built_once"] is True
    assert receipt["claim_boundary"]["canonical_kernel_canary_passed"] is True
    assert receipt["kernel"]["binary_sha256"] == _sha256(
        bundle / "oas_imgt_exact_search"
    )
    assert receipt["compile_arguments"] == list(MODULE.COMPILE_ARGUMENTS)
    assert receipt["test_access"]["frozen_official_external_test_accessed"] is False
    for path in (bundle, *bundle.rglob("*")):
        assert not path.lstat().st_mode & (
            stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
        )


def test_sealer_rejects_one_canary_output_byte_change(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    actual = bundle / "actual_matches.bin"
    payload = bytearray(actual.read_bytes())
    payload[-1] ^= 1
    actual.write_bytes(payload)
    kernel_source = ROOT / "native/oas_imgt_exact_search.cu"

    with pytest.raises(RuntimeError):
        MODULE.seal(
            bundle_root=bundle,
            kernel_source=kernel_source,
            canary_source=CANARY,
            nvcc_binary=TRUE_BINARY,
            nvcc_version_file=bundle / "nvcc-version.txt",
            expected_kernel_source_sha256=_sha256(kernel_source),
            expected_canary_source_sha256=_sha256(CANARY),
            build_launcher_source_sha256="a" * 64,
            sealer_source_sha256=_sha256(SEALER),
            resource_receipt_override={
                "cuda_visible_devices": "synthetic",
                "gpu_compute_permitted": False,
                "scheduler_reserved_gpu_count": 0,
            },
        )


def test_array_consumes_one_sealed_binary_instead_of_recompiling_per_task() -> None:
    array = (ROOT / "slurm/oas_imgt_exact_search_array.sbatch").read_text(
        encoding="utf-8"
    )
    builder = (ROOT / "slurm/oas_imgt_exact_kernel_build.sbatch").read_text(
        encoding="utf-8"
    )

    assert "OAS_KERNEL_BUNDLE_ROOT" in array
    assert "OAS_KERNEL_BINARY_SHA256" in array
    assert "OAS_KERNEL_BUILD_RECEIPT_FILE_SHA256" in array
    assert "KERNEL_BINARY=${OAS_KERNEL_BUNDLE_ROOT}/oas_imgt_exact_search" in array
    assert "nvcc \\" not in array
    assert "nvcc \\" in builder
    assert '"${KERNEL_BINARY}" \\' in builder
    assert 'mv "${TEMPORARY}" "${OAS_KERNEL_BUNDLE_ROOT}"' in builder
    assert "canonical_kernel_built_once" in array
