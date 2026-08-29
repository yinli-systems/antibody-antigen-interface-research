#!/usr/bin/env python3
"""Authenticate and seal one canonical fixed-IMGT CUDA kernel bundle."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any, Mapping

from antibody_pretraining.near_identity_export import publish_export_receipt


FORMAT_VERSION = "oas_imgt_exact_kernel_bundle_v1"
CANARY_FORMAT_VERSION = "oas_imgt_exact_cuda_canary_v1"
COMPILE_ARGUMENTS = (
    "-std=c++17",
    "-O3",
    "-lineinfo",
    "-gencode",
    "arch=compute_120,code=sm_120",
)
TEST_ACCESS = {
    "development_sequence_fields_accessed": False,
    "frozen_official_external_test_accessed": False,
    "model_predictions_or_scores_accessed": False,
    "task_label_values_used": False,
}


class OASIMGTKernelBundleError(RuntimeError):
    """Report mutable, contradictory, or unauthenticated kernel evidence."""


def _canonical_json_bytes(value: object) -> bytes:
    try:
        payload = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise OASIMGTKernelBundleError("Payload is not finite canonical JSON") from error
    return (payload + "\n").encode("ascii")


def _regular_file(path: Path, label: str, *, executable: bool = False) -> Path:
    candidate = path.expanduser().absolute()
    try:
        metadata = candidate.lstat()
    except OSError as error:
        raise OASIMGTKernelBundleError(f"Cannot stat {label}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise OASIMGTKernelBundleError(f"{label} is absent, symbolic, or non-regular")
    if executable and not metadata.st_mode & stat.S_IXUSR:
        raise OASIMGTKernelBundleError(f"{label} is not executable")
    return candidate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with _regular_file(path, "hashed file").open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise OASIMGTKernelBundleError(f"{label} is not lowercase SHA-256")
    return value


def _load_canary_module(path: Path):
    path = _regular_file(path, "canary source")
    specification = importlib.util.spec_from_file_location(
        "oas_imgt_exact_bundle_canary", path
    )
    if specification is None or specification.loader is None:
        raise OASIMGTKernelBundleError("Cannot load canary source")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def _resource_receipt(
    override: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if override is not None:
        expected = {
            "cuda_visible_devices": "synthetic",
            "gpu_compute_permitted": False,
            "scheduler_reserved_gpu_count": 0,
        }
        if dict(override) != expected:
            raise OASIMGTKernelBundleError("Synthetic resource override differs")
        return expected
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    assigned = os.environ.get("SLURM_JOB_GPUS") or os.environ.get(
        "SLURM_STEP_GPUS", ""
    )
    assigned_count = len([item for item in assigned.split(",") if item])
    if not visible or visible == "NoDevFiles" or assigned_count != 1:
        raise OASIMGTKernelBundleError(
            "Exactly one visible scheduler GPU is required to seal the bundle"
        )
    return {
        "cuda_visible_devices": visible,
        "gpu_compute_permitted": True,
        "scheduler_reserved_gpu_count": assigned_count,
    }


def seal(
    *,
    bundle_root: Path,
    kernel_source: Path,
    canary_source: Path,
    nvcc_binary: Path,
    nvcc_version_file: Path,
    expected_kernel_source_sha256: str,
    expected_canary_source_sha256: str,
    build_launcher_source_sha256: str,
    sealer_source_sha256: str,
    resource_receipt_override: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Replay the canary, bind compiler provenance, and seal the bundle."""
    bundle_root = bundle_root.expanduser().absolute()
    if bundle_root.is_symlink() or not bundle_root.is_dir():
        raise OASIMGTKernelBundleError("Bundle root is absent or symbolic")
    receipt_path = bundle_root / "BUILD_RECEIPT.json"
    if receipt_path.exists() or receipt_path.is_symlink():
        raise OASIMGTKernelBundleError("Refusing to overwrite a build receipt")
    expected_kernel_source_sha256 = _require_sha256(
        expected_kernel_source_sha256, "Expected kernel source"
    )
    expected_canary_source_sha256 = _require_sha256(
        expected_canary_source_sha256, "Expected canary source"
    )
    build_launcher_source_sha256 = _require_sha256(
        build_launcher_source_sha256, "Build launcher source"
    )
    sealer_source_sha256 = _require_sha256(sealer_source_sha256, "Sealer source")
    if _sha256(Path(__file__)) != sealer_source_sha256:
        raise OASIMGTKernelBundleError("Sealer source hash differs")
    kernel_source = _regular_file(kernel_source, "kernel source")
    canary_source = _regular_file(canary_source, "canary source")
    nvcc_binary = _regular_file(nvcc_binary, "nvcc binary", executable=True)
    nvcc_version_file = _regular_file(nvcc_version_file, "nvcc version file")
    if _sha256(kernel_source) != expected_kernel_source_sha256:
        raise OASIMGTKernelBundleError("Kernel source hash differs")
    if _sha256(canary_source) != expected_canary_source_sha256:
        raise OASIMGTKernelBundleError("Canary source hash differs")
    nvcc_version = nvcc_version_file.read_text(encoding="ascii")
    if "release 12.8" not in nvcc_version:
        raise OASIMGTKernelBundleError("Compiler is not the frozen CUDA 12.8 toolchain")
    kernel_binary = _regular_file(
        bundle_root / "oas_imgt_exact_search", "canonical kernel binary", executable=True
    )
    fixture_root = bundle_root / "fixtures"
    actual_matches = _regular_file(bundle_root / "actual_matches.bin", "canary matches")
    kernel_stats = _regular_file(bundle_root / "kernel_stats.json", "canary stats")
    canary = _load_canary_module(canary_source)
    canary_result = canary.audit(fixture_root, actual_matches, kernel_stats)
    if (
        canary_result.get("format_version") != CANARY_FORMAT_VERSION
        or canary_result.get("result") != "PASS"
        or canary_result.get("actual_match_count") != 11
    ):
        raise OASIMGTKernelBundleError("Canonical binary canary did not pass")
    fixture_receipt = _regular_file(
        fixture_root / "fixture_receipt.json", "canary fixture receipt"
    )
    receipt: dict[str, Any] = {
        "claim_boundary": {
            "canonical_kernel_built_once": True,
            "canonical_kernel_canary_passed": True,
            "fixed_imgt_full_oas_search_complete": False,
            "gap_aware_search_complete": False,
            "large_scale_pretraining_authorized": False,
            "near_identity_decontamination_complete": False,
            "scientific_claim": False,
        },
        "compile_arguments": list(COMPILE_ARGUMENTS),
        "compiler": {
            "nvcc_binary_sha256": _sha256(nvcc_binary),
            "nvcc_version_file_sha256": _sha256(nvcc_version_file),
            "release": "12.8",
        },
        "format_version": FORMAT_VERSION,
        "kernel": {
            "binary_file": "oas_imgt_exact_search",
            "binary_sha256": _sha256(kernel_binary),
            "source_sha256": expected_kernel_source_sha256,
        },
        "canary": {
            **canary_result,
            "fixture_receipt_file_sha256": _sha256(fixture_receipt),
            "source_sha256": expected_canary_source_sha256,
        },
        "resource_receipt": _resource_receipt(resource_receipt_override),
        "source_sha256": {
            "build_launcher": build_launcher_source_sha256,
            "sealer": sealer_source_sha256,
        },
        "test_access": TEST_ACCESS,
    }
    publish_export_receipt(receipt_path, receipt)
    for path in sorted(bundle_root.rglob("*"), reverse=True):
        path.chmod(path.lstat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    bundle_root.chmod(
        bundle_root.lstat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
    )
    return json.loads(receipt_path.read_text(encoding="ascii"))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--kernel-source", type=Path, required=True)
    parser.add_argument("--canary-source", type=Path, required=True)
    parser.add_argument("--nvcc-binary", type=Path, required=True)
    parser.add_argument("--nvcc-version-file", type=Path, required=True)
    parser.add_argument("--kernel-source-sha256", required=True)
    parser.add_argument("--canary-source-sha256", required=True)
    parser.add_argument("--build-launcher-source-sha256", required=True)
    parser.add_argument("--sealer-source-sha256", required=True)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    receipt = seal(
        bundle_root=arguments.bundle_root,
        kernel_source=arguments.kernel_source,
        canary_source=arguments.canary_source,
        nvcc_binary=arguments.nvcc_binary,
        nvcc_version_file=arguments.nvcc_version_file,
        expected_kernel_source_sha256=arguments.kernel_source_sha256,
        expected_canary_source_sha256=arguments.canary_source_sha256,
        build_launcher_source_sha256=arguments.build_launcher_source_sha256,
        sealer_source_sha256=arguments.sealer_source_sha256,
    )
    sys.stdout.buffer.write(_canonical_json_bytes(receipt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
