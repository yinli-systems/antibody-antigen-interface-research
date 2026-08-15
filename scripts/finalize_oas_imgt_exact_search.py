#!/usr/bin/env python3
"""Independently authenticate every fixed-IMGT search task and seal the track."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any, Mapping

from antibody_pretraining.imgt_search_output import (
    FIXED_IMGT_MATCH_RECORD,
    unpack_fixed_imgt_match,
)
from antibody_pretraining.near_identity_export import publish_export_receipt


FORMAT_VERSION = "oas_imgt_exact_search_global_v1"
TASK_FORMAT_VERSION = "oas_imgt_exact_search_task_v1"
TARGET_FORMAT_VERSION = "oas_imgt_target_export_global_v1"
QUERY_FORMAT_VERSION = "oas_asep_development_imgt_query_export_v2"
TEST_ACCESS = {
    "development_sequence_fields_accessed": True,
    "frozen_official_external_test_accessed": False,
    "model_predictions_or_scores_accessed": False,
    "task_label_values_used": False,
}


class OASIMGTSearchFinalizeError(RuntimeError):
    """Report missing, mutable, contradictory, or unauthenticated task evidence."""


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
        raise OASIMGTSearchFinalizeError("Payload is not finite canonical JSON") from error
    return (payload + "\n").encode("ascii")


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant: {value}")


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"Duplicate JSON key: {key}")
        output[key] = value
    return output


def _regular_file(path: Path, label: str) -> Path:
    candidate = path.expanduser().absolute()
    try:
        metadata = candidate.lstat()
    except OSError as error:
        raise OASIMGTSearchFinalizeError(f"Cannot stat {label}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise OASIMGTSearchFinalizeError(f"{label} is absent, symbolic, or non-regular")
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
        raise OASIMGTSearchFinalizeError(f"{label} is not lowercase SHA-256")
    return value


def _nonnegative_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise OASIMGTSearchFinalizeError(f"{label} is not a non-negative integer")
    return value


def _signed_json(path: Path, label: str) -> tuple[dict[str, Any], str]:
    path = _regular_file(path, label)
    raw = path.read_bytes()
    try:
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_reject_duplicates,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise OASIMGTSearchFinalizeError(f"{label} is not strict ASCII JSON") from error
    if not isinstance(value, dict) or raw != _canonical_json_bytes(value):
        raise OASIMGTSearchFinalizeError(f"{label} is not canonical JSON")
    unsigned = dict(value)
    claimed = _require_sha256(unsigned.pop("receipt_sha256", None), f"{label} hash")
    if hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest() != claimed:
        raise OASIMGTSearchFinalizeError(f"{label} self hash differs")
    return value, hashlib.sha256(raw).hexdigest()


def _assert_frozen_tree(root: Path) -> None:
    if root.is_symlink() or not root.is_dir():
        raise OASIMGTSearchFinalizeError("Task root is absent or symbolic")
    for path in (root, *root.rglob("*")):
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise OASIMGTSearchFinalizeError("Task tree contains a symbolic link")
        if metadata.st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
            raise OASIMGTSearchFinalizeError("Task tree remains writable")


def _read_matches(path: Path) -> tuple[dict[str, Any], set[tuple[int, int]]]:
    path = _regular_file(path, "task match binary")
    size = path.stat().st_size
    if size % FIXED_IMGT_MATCH_RECORD.size:
        raise OASIMGTSearchFinalizeError("Task match binary ended mid-record")
    pairs: set[tuple[int, int]] = set()
    queries: set[int] = set()
    targets: set[int] = set()
    splits: Counter[str] = Counter()
    records = 0
    with path.open("rb") as handle:
        while True:
            payload = handle.read(FIXED_IMGT_MATCH_RECORD.size)
            if not payload:
                break
            match = unpack_fixed_imgt_match(payload)
            pair = (match.target_authority_ordinal, match.query_ordinal)
            if pair in pairs:
                raise OASIMGTSearchFinalizeError("Task match pair is duplicated")
            pairs.add(pair)
            queries.add(match.query_ordinal)
            targets.add(match.target_authority_ordinal)
            splits[match.target_split] += 1
            records += 1
    return (
        {
            "match_records": records,
            "matched_query_count": len(queries),
            "matched_target_count": len(targets),
            "split_match_counts": {
                "train": splits["train"],
                "validation": splits["validation"],
            },
        },
        pairs,
    )


def _audit_task(
    search_output_root: Path,
    *,
    task_id: int,
    expected_source_hashes: Mapping[str, str],
    target_receipt_sha256: str,
    query_receipt_sha256: str,
) -> dict[str, Any]:
    root = search_output_root / "fixed-imgt" / f"mapper-{task_id:04d}"
    _assert_frozen_tree(root)
    receipt, receipt_file_sha256 = _signed_json(
        root / "receipt.json", f"search task receipt {task_id}"
    )
    boundary = receipt.get("claim_boundary")
    if (
        receipt.get("format_version") != TASK_FORMAT_VERSION
        or receipt.get("task_id") != task_id
        or receipt.get("test_access") != TEST_ACCESS
        or not isinstance(boundary, Mapping)
        or boundary.get("fixed_imgt_task_domain_complete") is not True
        or boundary.get("full_200_position_verification_complete") is not True
        or boundary.get("gap_aware_search_complete") is not False
        or boundary.get("global_fixed_imgt_search_complete") is not False
        or boundary.get("large_scale_pretraining_authorized") is not False
        or boundary.get("near_identity_decontamination_complete") is not False
        or boundary.get("scientific_claim") is not False
    ):
        raise OASIMGTSearchFinalizeError("Task claim boundary differs")
    sources = receipt.get("source_sha256")
    if not isinstance(sources, Mapping) or set(sources) != {
        "controller",
        "kernel_binary",
        "kernel_source",
        "match_schema",
    }:
        raise OASIMGTSearchFinalizeError("Task source inventory differs")
    for name, expected in expected_source_hashes.items():
        if _require_sha256(sources.get(name), f"Task source {name}") != expected:
            raise OASIMGTSearchFinalizeError(f"Task source differs for {name}")
    kernel_binary_sha256 = _require_sha256(
        sources.get("kernel_binary"), "Task kernel binary"
    )
    inputs = receipt.get("input")
    output = receipt.get("output")
    independent = receipt.get("independent_output_verification")
    stats = receipt.get("kernel_stats")
    resource = receipt.get("resource_receipt")
    if not all(
        isinstance(value, Mapping)
        for value in (inputs, output, independent, stats, resource)
    ):
        raise OASIMGTSearchFinalizeError("Task input, output, or verifier is absent")
    target_records = _nonnegative_integer(inputs.get("target_records"), "Task targets")
    if (
        inputs.get("global_target_receipt_sha256") != target_receipt_sha256
        or inputs.get("query_receipt_sha256") != query_receipt_sha256
        or _require_sha256(inputs.get("query_binary_sha256"), "Task query binary")
        != inputs.get("query_binary_sha256")
        or stats.get("format_version") != "oas_imgt_exact_cuda_stats_v1"
        or stats.get("query_records") != 3_103
        or stats.get("records_scanned") != target_records
        or stats.get("target_file_records") != target_records
        or stats.get("seed_lookups") != target_records * 25
        or stats.get("matches") != independent.get("match_records")
        or output.get("match_record_bytes") != FIXED_IMGT_MATCH_RECORD.size
        or resource.get("gpu_compute_permitted") is not True
        or resource.get("scheduler_reserved_gpu_count") != 1
        or not isinstance(resource.get("cuda_visible_devices"), str)
        or not resource.get("cuda_visible_devices")
    ):
        raise OASIMGTSearchFinalizeError("Task input and verified count accounting differs")
    matches_path = root / "matches.bin"
    match_bytes = _nonnegative_integer(output.get("matches_bin_bytes"), "Task match bytes")
    if (
        match_bytes != matches_path.stat().st_size
        or match_bytes % FIXED_IMGT_MATCH_RECORD.size
        or _require_sha256(output.get("matches_bin_sha256"), "Task match hash")
        != _sha256(matches_path)
    ):
        raise OASIMGTSearchFinalizeError("Task match inventory differs")
    observed, pairs = _read_matches(matches_path)
    if observed != dict(independent):
        raise OASIMGTSearchFinalizeError("Task independent match replay differs")
    return {
        "kernel_binary_sha256": kernel_binary_sha256,
        "match_summary": observed,
        "pairs": pairs,
        "receipt_file_sha256": receipt_file_sha256,
        "receipt_sha256": receipt["receipt_sha256"],
        "target_binary_sha256": _require_sha256(
            inputs.get("target_binary_sha256"), "Task target binary"
        ),
        "target_records": target_records,
        "task_id": task_id,
    }


def _resource_receipt() -> dict[str, Any]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    assigned = os.environ.get("SLURM_JOB_GPUS") or os.environ.get("SLURM_STEP_GPUS", "")
    assigned_count = len([item for item in assigned.split(",") if item])
    declared = os.environ.get("OAS_SCHEDULER_RESERVED_GPU_COUNT")
    if visible not in {"", "NoDevFiles"}:
        raise OASIMGTSearchFinalizeError("Finalizer must not expose a GPU")
    if declared == "1" and assigned_count == 1:
        mode = "scheduler_mandated_idle_only"
    elif declared is None and assigned_count == 0:
        mode = "none"
    else:
        raise OASIMGTSearchFinalizeError("Finalizer scheduler resource evidence differs")
    return {
        "cuda_visible_devices": False,
        "gpu_compute_permitted": False,
        "scheduler_reserved_gpu_count": assigned_count,
        "scheduler_reserved_gpu_mode": mode,
    }


def finalize(
    *,
    search_output_root: Path,
    target_output_root: Path,
    query_root: Path,
    task_count: int,
    worker_count: int,
    expected_source_hashes: Mapping[str, str],
    finalizer_source_sha256: str,
) -> dict[str, Any]:
    """Re-hash all native matches and publish a fixed-track-only receipt."""
    search_output_root = search_output_root.expanduser().absolute()
    target_output_root = target_output_root.expanduser().absolute()
    query_root = query_root.expanduser().absolute()
    output_path = search_output_root / "fixed_imgt_global_receipt.json"
    if (
        search_output_root.is_symlink()
        or not search_output_root.is_dir()
        or output_path.exists()
        or output_path.is_symlink()
        or task_count != 32
        or not 1 <= worker_count <= 8
        or any((search_output_root / "work").glob("fixed-imgt-*.partial-*"))
    ):
        raise OASIMGTSearchFinalizeError("Global finalization path or task domain differs")
    finalizer_source_sha256 = _require_sha256(
        finalizer_source_sha256, "Finalizer source"
    )
    for name, digest in expected_source_hashes.items():
        _require_sha256(digest, f"Expected source {name}")
    target_receipt, target_receipt_file_sha256 = _signed_json(
        target_output_root / "global_receipt.json", "global target receipt"
    )
    query_receipt, query_receipt_file_sha256 = _signed_json(
        query_root / "EXPORT_RECEIPT.json", "query receipt"
    )
    target_boundary = target_receipt.get("claim_boundary")
    target_counts = target_receipt.get("counts")
    if (
        target_receipt.get("format_version") != TARGET_FORMAT_VERSION
        or target_receipt.get("task_count") != task_count
        or not isinstance(target_boundary, Mapping)
        or target_boundary.get("global_task_set_complete") is not True
        or target_boundary.get("accepted_oas_domain_partition_complete") is not True
        or target_boundary.get("near_identity_search_complete") is not False
        or not isinstance(target_counts, Mapping)
    ):
        raise OASIMGTSearchFinalizeError("Global target receipt is incomplete")
    if (
        query_receipt.get("format_version") != QUERY_FORMAT_VERSION
        or query_receipt.get("fixed_imgt_query_count") != 3_103
        or query_receipt.get("gap_aware_query_count") != 3_106
        or query_receipt.get("outside_canonical_query_count") != 3
        or query_receipt.get("test_access") != TEST_ACCESS
    ):
        raise OASIMGTSearchFinalizeError("Query receipt scope differs")
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        tasks = list(
            pool.map(
                lambda task_id: _audit_task(
                    search_output_root,
                    task_id=task_id,
                    expected_source_hashes=expected_source_hashes,
                    target_receipt_sha256=target_receipt["receipt_sha256"],
                    query_receipt_sha256=query_receipt["receipt_sha256"],
                ),
                range(task_count),
            )
        )
    kernel_hashes = {task["kernel_binary_sha256"] for task in tasks}
    if len(kernel_hashes) != 1:
        raise OASIMGTSearchFinalizeError("Task kernel binaries are not identical")
    expected_targets = _nonnegative_integer(
        target_counts.get("representable_records"), "Global representable targets"
    )
    if sum(task["target_records"] for task in tasks) != expected_targets:
        raise OASIMGTSearchFinalizeError("Global target record coverage differs")
    all_pairs: set[tuple[int, int]] = set()
    split_counts: Counter[str] = Counter()
    match_records = 0
    task_receipts = []
    for task in tasks:
        overlap = all_pairs.intersection(task["pairs"])
        if overlap:
            raise OASIMGTSearchFinalizeError("A match pair occurs in multiple tasks")
        all_pairs.update(task["pairs"])
        summary = task["match_summary"]
        match_records += summary["match_records"]
        split_counts.update(summary["split_match_counts"])
        task_receipts.append({key: value for key, value in task.items() if key != "pairs"})
    if match_records != len(all_pairs):
        raise OASIMGTSearchFinalizeError("Global unique-pair accounting differs")
    receipt = {
        "claim_boundary": {
            "fixed_imgt_representable_target_search_complete": True,
            "full_200_position_verification_complete": True,
            "gap_aware_search_complete": False,
            "large_scale_pretraining_authorized": False,
            "near_identity_decontamination_complete": False,
            "scientific_claim": False,
        },
        "format_version": FORMAT_VERSION,
        "inputs": {
            "query_receipt_file_sha256": query_receipt_file_sha256,
            "query_receipt_sha256": query_receipt["receipt_sha256"],
            "target_receipt_file_sha256": target_receipt_file_sha256,
            "target_receipt_sha256": target_receipt["receipt_sha256"],
        },
        "kernel_binary_sha256": next(iter(kernel_hashes)),
        "match_summary": {
            "match_records": match_records,
            "split_match_counts": {
                "train": split_counts["train"],
                "validation": split_counts["validation"],
            },
            "unique_pairs": len(all_pairs),
        },
        "query_records": 3_103,
        "resource_receipt": _resource_receipt(),
        "source_sha256": {
            "finalizer": finalizer_source_sha256,
            **dict(expected_source_hashes),
        },
        "target_records": expected_targets,
        "task_count": task_count,
        "task_receipts": task_receipts,
        "test_access": TEST_ACCESS,
        "worker_count": worker_count,
    }
    publish_export_receipt(output_path, receipt)
    output_path.chmod(0o400)
    return json.loads(output_path.read_text(encoding="ascii"))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-output-root", type=Path, required=True)
    parser.add_argument("--target-output-root", type=Path, required=True)
    parser.add_argument("--query-root", type=Path, required=True)
    parser.add_argument("--task-count", type=int, default=32)
    parser.add_argument("--worker-count", type=int, default=8)
    parser.add_argument("--finalizer-source-sha256", required=True)
    parser.add_argument("--search-controller-source-sha256", required=True)
    parser.add_argument("--kernel-source-sha256", required=True)
    parser.add_argument("--match-schema-source-sha256", required=True)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    receipt = finalize(
        search_output_root=arguments.search_output_root,
        target_output_root=arguments.target_output_root,
        query_root=arguments.query_root,
        task_count=arguments.task_count,
        worker_count=arguments.worker_count,
        expected_source_hashes={
            "controller": arguments.search_controller_source_sha256,
            "kernel_source": arguments.kernel_source_sha256,
            "match_schema": arguments.match_schema_source_sha256,
        },
        finalizer_source_sha256=arguments.finalizer_source_sha256,
    )
    sys.stdout.buffer.write(_canonical_json_bytes(receipt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
