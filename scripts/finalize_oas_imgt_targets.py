#!/usr/bin/env python3
"""Authenticate all OAS IMGT target tasks and publish one global receipt."""

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

from antibody_pretraining.oas_imgt_export import (
    IMGT_TARGET_RECORD,
    UNUSUAL_TARGET_RECORD,
)


FORMAT_VERSION = "oas_imgt_target_export_global_v1"
TASK_FORMAT_VERSION = "oas_imgt_target_export_task_v1"
TEST_ACCESS = {
    "downstream_metrics_used": False,
    "sabdab2_heldout_accessed": False,
}


class OASIMGTFinalizeError(RuntimeError):
    """Report incomplete or inconsistent IMGT target evidence."""


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant: {value}")


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _canonical_json_bytes(payload: object) -> bytes:
    try:
        encoded = json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise OASIMGTFinalizeError("Payload is not canonical JSON") from error
    return (encoded + "\n").encode("ascii")


def _strict_json_bytes(payload: bytes, label: str) -> dict[str, Any]:
    try:
        parsed = json.loads(
            payload.decode("ascii"),
            object_pairs_hook=_reject_duplicates,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise OASIMGTFinalizeError(f"{label} must be strict ASCII JSON") from error
    if not isinstance(parsed, dict):
        raise OASIMGTFinalizeError(f"{label} must contain one JSON object")
    return parsed


def _require_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise OASIMGTFinalizeError(f"{label} is not a lowercase SHA-256")
    return value


def _nonnegative_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise OASIMGTFinalizeError(f"{label} must be a non-negative integer")
    return value


def _file_sha256(path: Path) -> str:
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise OASIMGTFinalizeError(f"Evidence file is absent or symbolic: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_signed_json(path: Path, label: str) -> tuple[dict[str, Any], str, str]:
    raw = path.read_bytes()
    payload = _strict_json_bytes(raw, label)
    if raw != _canonical_json_bytes(payload):
        raise OASIMGTFinalizeError(f"{label} is not canonical JSON")
    unsigned = dict(payload)
    claimed = _require_sha256(
        unsigned.pop("receipt_sha256", None),
        f"{label} self hash",
    )
    observed = hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest()
    if claimed != observed:
        raise OASIMGTFinalizeError(f"{label} self hash differs")
    return payload, hashlib.sha256(raw).hexdigest(), claimed


def _assert_frozen_tree(root: Path) -> None:
    if not root.is_dir() or root.is_symlink():
        raise OASIMGTFinalizeError("Published task root is absent or symbolic")
    for path in (root, *root.rglob("*")):
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise OASIMGTFinalizeError("Published task tree contains a symlink")
        if metadata.st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
            raise OASIMGTFinalizeError("Published task tree remains writable")


def _validate_count_mapping(value: object, label: str) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise OASIMGTFinalizeError(f"{label} is absent")
    result: dict[str, int] = {}
    for key, count in value.items():
        if not isinstance(key, str) or not key:
            raise OASIMGTFinalizeError(f"{label} key is invalid")
        result[key] = _nonnegative_integer(count, f"{label} {key}")
    return result


def _validate_inventory(
    root: Path,
    raw: object,
    *,
    name: str,
    record_bytes: int,
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise OASIMGTFinalizeError(f"{name} inventory is absent")
    records = _nonnegative_integer(raw.get("records"), f"{name} records")
    byte_count = _nonnegative_integer(raw.get("bytes"), f"{name} bytes")
    if raw.get("record_bytes") != record_bytes or byte_count != records * record_bytes:
        raise OASIMGTFinalizeError(f"{name} byte accounting differs")
    path = root / f"{name}.bin"
    if path.stat().st_size != byte_count:
        raise OASIMGTFinalizeError(f"{name} file size differs")
    observed = _file_sha256(path)
    claimed = _require_sha256(raw.get("file_sha256"), f"{name} file hash")
    if observed != claimed:
        raise OASIMGTFinalizeError(f"{name} file hash differs")
    return {
        "bytes": byte_count,
        "file_sha256": observed,
        "record_bytes": record_bytes,
        "records": records,
    }


def _audit_task(
    output_root: Path,
    *,
    task_id: int,
    task_count: int,
    expected_sources: Mapping[str, str],
) -> dict[str, Any]:
    root = output_root / "imgt-map" / f"mapper-{task_id:04d}"
    _assert_frozen_tree(root)
    receipt, file_sha256, payload_sha256 = _load_signed_json(
        root / "receipt.json",
        f"IMGT task receipt {task_id}",
    )
    boundary = receipt.get("claim_boundary")
    if (
        receipt.get("format_version") != TASK_FORMAT_VERSION
        or receipt.get("task_id") != task_id
        or receipt.get("task_count") != task_count
        or not isinstance(boundary, Mapping)
        or boundary.get("fixed_imgt_task_domain_complete") is not True
        or boundary.get("gap_aware_handoff_preserved") is not True
        or boundary.get("global_task_set_complete") is not False
        or boundary.get("near_identity_search_complete") is not False
        or boundary.get("pilot_only") is not False
        or boundary.get("scientific_claim") is not False
        or receipt.get("test_access") != TEST_ACCESS
    ):
        raise OASIMGTFinalizeError("IMGT receipt is not a complete fixed task")
    sources = receipt.get("source_sha256")
    if sources != dict(expected_sources):
        raise OASIMGTFinalizeError("IMGT task source identity differs")
    for key, value in expected_sources.items():
        _require_sha256(value, f"Expected source {key}")
    inventory = receipt.get("output_inventory")
    if not isinstance(inventory, Mapping):
        raise OASIMGTFinalizeError("IMGT output inventory is absent")
    representable = _validate_inventory(
        root,
        inventory.get("representable"),
        name="representable",
        record_bytes=IMGT_TARGET_RECORD.size,
    )
    unusual = _validate_inventory(
        root,
        inventory.get("unusual"),
        name="unusual",
        record_bytes=UNUSUAL_TARGET_RECORD.size,
    )
    counts = _validate_count_mapping(receipt.get("counts"), "Task counts")
    replay_counts = _validate_count_mapping(
        receipt.get("replay_counts"),
        "Replay counts",
    )
    accepted = counts.get("accepted_records", -1)
    if (
        accepted != representable["records"] + unusual["records"]
        or accepted != replay_counts.get("accepted_records")
        or counts.get("representable_records") != representable["records"]
        or counts.get("unusual_records") != unusual["records"]
        or sum(value for key, value in counts.items() if key.startswith("split:"))
        != accepted
        or sum(value for key, value in counts.items() if key.startswith("chain:"))
        != accepted
        or sum(
            value
            for key, value in counts.items()
            if key.startswith("unusual_reason:")
        )
        != unusual["records"]
    ):
        raise OASIMGTFinalizeError("IMGT task count accounting differs")
    unit_count = _nonnegative_integer(receipt.get("unit_count"), "Unit count")
    units = receipt.get("replayed_unit_receipts")
    if unit_count < 1 or not isinstance(units, list) or len(units) != unit_count:
        raise OASIMGTFinalizeError("IMGT replayed unit inventory differs")
    resource = receipt.get("resource_receipt")
    if not isinstance(resource, Mapping):
        raise OASIMGTFinalizeError("IMGT task resource receipt is absent")
    resource_receipt = dict(resource)
    if resource_receipt not in (
        {
            "cuda_visible_devices": False,
            "gpu_compute_permitted": False,
            "scheduler_reserved_gpu_count": 0,
            "scheduler_reserved_gpu_mode": "none",
        },
        {
            "cuda_visible_devices": False,
            "gpu_compute_permitted": False,
            "scheduler_reserved_gpu_count": 1,
            "scheduler_reserved_gpu_mode": "scheduler_mandated_idle_only",
        },
    ):
        raise OASIMGTFinalizeError("IMGT task resource receipt differs")
    return {
        "counts": counts,
        "file_sha256": file_sha256,
        "output_inventory": {
            "representable": representable,
            "unusual": unusual,
        },
        "payload_sha256": payload_sha256,
        "resource_receipt": resource_receipt,
        "task_id": task_id,
        "unit_count": unit_count,
    }


def _resource_receipt() -> dict[str, Any]:
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") not in {"", "NoDevFiles"}:
        raise OASIMGTFinalizeError("Finalization must not expose a GPU")
    assigned_value = os.environ.get("SLURM_JOB_GPUS") or os.environ.get(
        "SLURM_STEP_GPUS", ""
    )
    assigned_count = len([item for item in assigned_value.split(",") if item])
    declared = os.environ.get("OAS_SCHEDULER_RESERVED_GPU_COUNT")
    if declared is None and assigned_count == 0:
        mode = "none"
    elif declared == "1" and assigned_count == 1:
        mode = "scheduler_mandated_idle_only"
    else:
        raise OASIMGTFinalizeError("Finalizer idle-GPU evidence differs")
    return {
        "cuda_visible_devices": False,
        "gpu_compute_permitted": False,
        "scheduler_reserved_gpu_count": assigned_count,
        "scheduler_reserved_gpu_mode": mode,
    }


def _freeze_tree(root: Path) -> None:
    for path in root.rglob("*"):
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise OASIMGTFinalizeError("Output tree contains a symlink")
        path.chmod(metadata.st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    metadata = root.lstat()
    root.chmod(metadata.st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def finalize_imgt_export(
    *,
    output_root: Path,
    output_path: Path,
    task_count: int,
    worker_count: int,
    expected_accepted_records: int,
    expected_sources: Mapping[str, str],
    controller_source_sha256: str,
) -> dict[str, Any]:
    """Verify every immutable task and publish one fail-closed receipt."""
    if (
        not output_root.is_dir()
        or output_root.is_symlink()
        or output_path != output_root / "global_receipt.json"
        or output_path.exists()
        or output_path.is_symlink()
        or task_count < 1
        or not 1 <= worker_count <= 8
    ):
        raise OASIMGTFinalizeError("Global finalization path or domain is invalid")
    if any((output_root / "work").glob("imgt-mapper-*.partial-*")):
        raise OASIMGTFinalizeError("Partial IMGT task output remains")
    _require_sha256(controller_source_sha256, "Finalizer source")
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        tasks = list(
            pool.map(
                lambda task_id: _audit_task(
                    output_root,
                    task_id=task_id,
                    task_count=task_count,
                    expected_sources=expected_sources,
                ),
                range(task_count),
            )
        )
    resources = {json.dumps(item["resource_receipt"], sort_keys=True) for item in tasks}
    if len(resources) != 1:
        raise OASIMGTFinalizeError("Task resource receipts disagree")
    totals: Counter[str] = Counter()
    for item in tasks:
        totals.update(item["counts"])
    if totals["accepted_records"] != expected_accepted_records:
        raise OASIMGTFinalizeError("Global accepted-record count differs")
    if (
        totals["accepted_records"]
        != totals["representable_records"] + totals["unusual_records"]
    ):
        raise OASIMGTFinalizeError("Global target partition is not exhaustive")
    receipt = {
        "claim_boundary": {
            "accepted_oas_domain_partition_complete": True,
            "fixed_imgt_representable_subset_enumerated": True,
            "gap_aware_handoff_enumerated": True,
            "global_task_set_complete": True,
            "near_identity_search_complete": False,
            "scientific_claim": False,
        },
        "counts": dict(sorted(totals.items())),
        "expected_accepted_records": expected_accepted_records,
        "format_version": FORMAT_VERSION,
        "resource_receipt": _resource_receipt(),
        "source_sha256": {
            "finalizer": controller_source_sha256,
            **dict(expected_sources),
        },
        "task_count": task_count,
        "task_receipts": tasks,
        "test_access": TEST_ACCESS,
        "worker_count": worker_count,
    }
    receipt["receipt_sha256"] = hashlib.sha256(
        _canonical_json_bytes(receipt)
    ).hexdigest()
    descriptor = os.open(
        output_path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o400,
    )
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(_canonical_json_bytes(receipt))
        handle.flush()
        os.fsync(handle.fileno())
    _freeze_tree(output_root)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--task-count", type=int, default=32)
    parser.add_argument("--worker-count", type=int, default=8)
    parser.add_argument("--expected-accepted-records", type=int, required=True)
    parser.add_argument("--controller-source-sha256", required=True)
    parser.add_argument("--export-controller-source-sha256", required=True)
    parser.add_argument("--imgt-position-source-sha256", required=True)
    parser.add_argument("--imgt-record-source-sha256", required=True)
    parser.add_argument("--map-record-source-sha256", required=True)
    parser.add_argument("--preprocessing-source-sha256", required=True)
    parser.add_argument("--replay-mapper-source-sha256", required=True)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    expected_sources = {
        "controller": arguments.export_controller_source_sha256,
        "imgt_position": arguments.imgt_position_source_sha256,
        "imgt_record": arguments.imgt_record_source_sha256,
        "map_record": arguments.map_record_source_sha256,
        "preprocessing": arguments.preprocessing_source_sha256,
        "replay_mapper": arguments.replay_mapper_source_sha256,
    }
    receipt = finalize_imgt_export(
        output_root=arguments.output_root,
        output_path=arguments.output_root / "global_receipt.json",
        task_count=arguments.task_count,
        worker_count=arguments.worker_count,
        expected_accepted_records=arguments.expected_accepted_records,
        expected_sources=expected_sources,
        controller_source_sha256=arguments.controller_source_sha256,
    )
    sys.stdout.buffer.write(_canonical_json_bytes(receipt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
