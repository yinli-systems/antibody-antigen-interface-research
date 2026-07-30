#!/usr/bin/env python3
"""Authenticate every OAS map/reduce artifact and publish a corpus manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

FORMAT_VERSION = "oas_production_corpus_finalize_receipt_v1"
MANIFEST_FORMAT = "oas_production_corpus_manifest_v1"
MAP_FORMAT = "oas_production_map_task_receipt_v1"
REDUCE_FORMAT = "oas_production_reduce_bucket_receipt_v1"
SHARD_FORMAT = "oas_production_dense_bucket_shard_v1"
SUPPORTED_PROTOCOL_FORMATS = {
    "oas_production_materialization_protocol_v1",
    "oas_production_materialization_protocol_v2",
}
TEST_ACCESS = {
    "downstream_metrics_used": False,
    "sabdab2_heldout_accessed": False,
}


class ProductionFinalizeError(RuntimeError):
    """Report incomplete, mutable, or inconsistent production evidence."""


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
        raise ProductionFinalizeError("Payload is not canonical JSON") from error
    return (encoded + "\n").encode("ascii")


def _strict_json_bytes(payload: bytes, label: str) -> dict[str, Any]:
    try:
        parsed = json.loads(
            payload.decode("ascii"),
            object_pairs_hook=_reject_duplicates,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ProductionFinalizeError(f"{label} must be strict ASCII JSON") from error
    if not isinstance(parsed, dict):
        raise ProductionFinalizeError(f"{label} must contain one JSON object")
    return parsed


def _require_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ProductionFinalizeError(f"{label} is not a lowercase SHA-256")
    return value


def _file_sha256(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise ProductionFinalizeError("Evidence file is absent or symbolic")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_signed_json(
    path: Path,
    *,
    digest_field: str,
    label: str,
) -> tuple[dict[str, Any], str, str]:
    if not path.is_file() or path.is_symlink():
        raise ProductionFinalizeError(f"{label} is absent or symbolic")
    raw = path.read_bytes()
    payload = _strict_json_bytes(raw, label)
    if raw != _canonical_json_bytes(payload):
        raise ProductionFinalizeError(f"{label} is not canonical JSON")
    unsigned = dict(payload)
    claimed = _require_sha256(unsigned.pop(digest_field, None), f"{label} self hash")
    observed = hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest()
    if claimed != observed:
        raise ProductionFinalizeError(f"{label} self hash differs")
    return payload, hashlib.sha256(raw).hexdigest(), claimed


def _nonnegative_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProductionFinalizeError(f"{label} must be a non-negative integer")
    return value


def _resource_receipt(protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Prove that finalization performs no work on its reserved GPU."""
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") not in {"", "NoDevFiles"}:
        raise ProductionFinalizeError("Production finalization must not expose a GPU")
    policy = protocol.get("resource_policy")
    if not isinstance(policy, Mapping):
        raise ProductionFinalizeError("Production resource policy is absent")
    assigned_value = os.environ.get("SLURM_JOB_GPUS") or os.environ.get(
        "SLURM_STEP_GPUS"
    )
    assigned_count = len(
        [value for value in (assigned_value or "").split(",") if value]
    )
    allocation = policy.get("gpu_allocation_during_materialization", False)
    if allocation is False:
        if assigned_count:
            raise ProductionFinalizeError("Slurm assigned a GPU to CPU finalization")
        return {
            "cuda_visible_devices": False,
            "gpu_compute_permitted": False,
            "scheduler_reserved_gpu_count": 0,
            "scheduler_reserved_gpu_mode": "none",
        }
    if allocation != "scheduler_mandated_idle_only":
        raise ProductionFinalizeError("Production GPU-allocation policy differs")
    expected = policy.get("scheduler_reserved_gpu_count")
    declared = os.environ.get("OAS_SCHEDULER_RESERVED_GPU_COUNT")
    if expected != 1 or declared != "1" or assigned_count != 1:
        raise ProductionFinalizeError("Scheduler-reserved idle GPU evidence differs")
    return {
        "cuda_visible_devices": False,
        "gpu_compute_permitted": False,
        "scheduler_reserved_gpu_count": 1,
        "scheduler_reserved_gpu_mode": "scheduler_mandated_idle_only",
    }


def _assert_frozen_tree(root: Path) -> None:
    if not root.is_dir() or root.is_symlink():
        raise ProductionFinalizeError("Published artifact root is absent or symbolic")
    for path in (root, *root.rglob("*")):
        if path.is_symlink():
            raise ProductionFinalizeError("Published artifact tree contains a symlink")
        if path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
            raise ProductionFinalizeError("Published artifact tree remains writable")


def _map_identity(receipt: Mapping[str, Any]) -> dict[str, Any]:
    bindings = receipt.get("audit_bindings")
    sources = receipt.get("source_sha256")
    if not isinstance(bindings, Mapping) or not bindings:
        raise ProductionFinalizeError("Map audit bindings are absent")
    if not isinstance(sources, Mapping) or not sources:
        raise ProductionFinalizeError("Map source digests are absent")
    for key, value in (*bindings.items(), *sources.items()):
        _require_sha256(value, f"Map identity digest {key}")
    study_count = _nonnegative_integer(receipt.get("study_count"), "Study count")
    if not 0 < study_count < 2**16:
        raise ProductionFinalizeError("Study count is outside uint16")
    return {
        "audit_bindings": dict(bindings),
        "source_sha256": dict(sources),
        "study_count": study_count,
        "study_vocabulary_sha256": _require_sha256(
            receipt.get("study_vocabulary_sha256"),
            "Study vocabulary digest",
        ),
        "task_count": receipt.get("task_count"),
    }


def _load_map_receipts(
    output_root: Path,
    *,
    task_count: int,
    bucket_count: int,
    protocol_format: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, int]]:
    receipts = []
    common_identity: dict[str, Any] | None = None
    totals = {"accepted_records": 0, "source_rows": 0}
    for task_id in range(task_count):
        root = output_root / "map" / f"mapper-{task_id:04d}"
        _assert_frozen_tree(root)
        receipt, file_sha256, payload_sha256 = _load_signed_json(
            root / "receipt.json",
            digest_field="receipt_sha256",
            label=f"Map receipt {task_id}",
        )
        boundary = receipt.get("claim_boundary")
        if (
            receipt.get("format_version") != MAP_FORMAT
            or receipt.get("task_id") != task_id
            or receipt.get("task_count") != task_count
            or not isinstance(boundary, Mapping)
            or boundary.get("map_task_complete") is not True
            or boundary.get("pilot_only") is not False
            or receipt.get("test_access") != TEST_ACCESS
        ):
            raise ProductionFinalizeError(
                "Map receipt is not a complete production task"
            )
        if protocol_format == "oas_production_materialization_protocol_v2" and (
            receipt.get("resource_receipt")
            != {
                "cuda_visible_devices": False,
                "gpu_compute_permitted": False,
                "scheduler_reserved_gpu_count": 1,
                "scheduler_reserved_gpu_mode": "scheduler_mandated_idle_only",
            }
        ):
            raise ProductionFinalizeError("Map idle-GPU resource evidence differs")
        identity = _map_identity(receipt)
        if common_identity is None:
            common_identity = identity
        elif identity != common_identity:
            raise ProductionFinalizeError("Map receipt identities disagree")
        counts = receipt.get("counts")
        if not isinstance(counts, Mapping):
            raise ProductionFinalizeError("Map counts are absent")
        for key in totals:
            totals[key] += _nonnegative_integer(counts.get(key), f"Map count {key}")
        raw_inventory = receipt.get("bucket_inventory")
        if not isinstance(raw_inventory, list):
            raise ProductionFinalizeError("Map bucket inventory is absent")
        inventory: dict[int, int] = {}
        for item in raw_inventory:
            if not isinstance(item, Mapping):
                raise ProductionFinalizeError("Map bucket inventory item is invalid")
            bucket = item.get("bucket")
            records = item.get("records")
            if (
                isinstance(bucket, bool)
                or not isinstance(bucket, int)
                or not 0 <= bucket < bucket_count
                or bucket in inventory
            ):
                raise ProductionFinalizeError("Map bucket identity is invalid")
            records = _nonnegative_integer(records, "Map bucket record count")
            byte_count = _nonnegative_integer(item.get("bytes"), "Map bucket bytes")
            if records < 1 or byte_count != 240 * records:
                raise ProductionFinalizeError("Map bucket byte accounting differs")
            _require_sha256(item.get("file_sha256"), "Map bucket digest")
            inventory[bucket] = records
        receipts.append(
            {
                "file_sha256": file_sha256,
                "inventory": inventory,
                "payload_sha256": payload_sha256,
                "task_id": task_id,
            }
        )
    if common_identity is None:
        raise ProductionFinalizeError("No complete map receipt was found")
    if (
        sum(sum(item["inventory"].values()) for item in receipts)
        != totals["accepted_records"]
    ):
        raise ProductionFinalizeError(
            "Map bucket inventory does not cover accepted rows"
        )
    return receipts, common_identity, totals


def _validate_shard(
    root: Path,
    *,
    bucket: int,
    split: str,
    expected_reference: Mapping[str, Any],
) -> dict[str, Any]:
    manifest, file_sha256, payload_sha256 = _load_signed_json(
        root / "manifest.json",
        digest_field="manifest_sha256",
        label=f"Reducer {bucket} {split} manifest",
    )
    if (
        manifest.get("format_version") != SHARD_FORMAT
        or manifest.get("bucket") != bucket
        or manifest.get("split") != split
        or manifest.get("test_access") != TEST_ACCESS
        or file_sha256 != expected_reference.get("file_sha256")
        or payload_sha256 != expected_reference.get("payload_sha256")
    ):
        raise ProductionFinalizeError("Reducer shard manifest binding differs")
    record_count = _nonnegative_integer(manifest.get("record_count"), "Shard records")
    if record_count != expected_reference.get("record_count"):
        raise ProductionFinalizeError("Reducer shard record count differs")
    files = manifest.get("files")
    expected_names = {
        "chain_types.npy",
        "lengths.npy",
        "sample_keys.npy",
        "sequence_lengths.npy",
        "sequence_sha256.npy",
        "status_flags.npy",
        "strata.npy",
        "study_ids.npy",
        "tokens.npy",
    }
    if not isinstance(files, Mapping) or set(files) != expected_names:
        raise ProductionFinalizeError("Reducer shard file registry differs")
    for name, reference in files.items():
        if not isinstance(reference, Mapping):
            raise ProductionFinalizeError("Reducer shard file receipt is invalid")
        path = root / name
        byte_count = _nonnegative_integer(reference.get("bytes"), "Shard file bytes")
        if path.stat().st_size != byte_count or _file_sha256(path) != _require_sha256(
            reference.get("sha256"),
            "Shard file digest",
        ):
            raise ProductionFinalizeError("Reducer shard file differs")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if str(array.dtype) != reference.get("dtype") or list(
            array.shape
        ) != reference.get("shape"):
            raise ProductionFinalizeError("Reducer shard array schema differs")
        if array.shape[0] != record_count:
            raise ProductionFinalizeError("Reducer shard leading dimension differs")
    return {
        "manifest_file_sha256": file_sha256,
        "manifest_payload_sha256": payload_sha256,
        "record_count": record_count,
        "relative_root": str(root.relative_to(root.parents[2])),
    }


def _load_reduce_receipts(
    output_root: Path,
    *,
    protocol_file_sha256: str,
    bucket_count: int,
    task_count: int,
    map_receipts: list[dict[str, Any]],
    map_identity: Mapping[str, Any],
    expected_source_sha256: Mapping[str, str],
    protocol_format: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    buckets = []
    totals = {
        "cross_split_sequence_digests": 0,
        "input_records": 0,
        "training_records_excluded_by_validation": 0,
        "unique_records": 0,
        "unique_train_records": 0,
        "unique_validation_records": 0,
        "within_split_duplicate_records": 0,
    }
    for bucket in range(bucket_count):
        root = output_root / "reduce" / f"bucket-{bucket:03d}"
        _assert_frozen_tree(root)
        receipt, file_sha256, payload_sha256 = _load_signed_json(
            root / "receipt.json",
            digest_field="receipt_sha256",
            label=f"Reducer receipt {bucket}",
        )
        boundary = receipt.get("claim_boundary")
        if (
            receipt.get("format_version") != REDUCE_FORMAT
            or receipt.get("bucket") != bucket
            or receipt.get("protocol_file_sha256") != protocol_file_sha256
            or receipt.get("map_identity") != dict(map_identity)
            or receipt.get("source_sha256") != dict(expected_source_sha256)
            or receipt.get("test_access") != TEST_ACCESS
            or not isinstance(boundary, Mapping)
            or boundary.get("reduced_bucket_complete") is not True
            or boundary.get("large_scale_pretraining_authorized") is not False
        ):
            raise ProductionFinalizeError("Reducer receipt identity differs")
        if protocol_format == "oas_production_materialization_protocol_v2" and (
            receipt.get("resource_receipt")
            != {
                "cuda_visible_devices": False,
                "gpu_compute_permitted": False,
                "scheduler_reserved_gpu_count": 1,
                "scheduler_reserved_gpu_mode": "scheduler_mandated_idle_only",
            }
        ):
            raise ProductionFinalizeError("Reducer idle-GPU resource evidence differs")
        references = receipt.get("mapper_receipts")
        if not isinstance(references, list) or len(references) != task_count:
            raise ProductionFinalizeError("Reducer mapper receipt inventory differs")
        for task_id, reference in enumerate(references):
            expected = map_receipts[task_id]
            expected_records = expected["inventory"].get(bucket, 0)
            if reference != {
                "file_sha256": expected["file_sha256"],
                "mapper_id": task_id,
                "payload_sha256": expected["payload_sha256"],
                "records_in_bucket": expected_records,
            }:
                raise ProductionFinalizeError("Reducer mapper reference differs")
        counts = receipt.get("counts")
        if not isinstance(counts, Mapping) or set(counts) != set(totals):
            raise ProductionFinalizeError("Reducer count schema differs")
        for key in totals:
            totals[key] += _nonnegative_integer(counts[key], f"Reducer count {key}")
        if counts["input_records"] != sum(
            item["inventory"].get(bucket, 0) for item in map_receipts
        ):
            raise ProductionFinalizeError(
                "Reducer input count differs from map inventory"
            )
        if counts["unique_records"] != (
            counts["unique_train_records"] + counts["unique_validation_records"]
        ):
            raise ProductionFinalizeError("Reducer unique-count accounting differs")
        shard_references = receipt.get("shard_manifests")
        if not isinstance(shard_references, Mapping) or set(shard_references) != {
            "train",
            "validation",
        }:
            raise ProductionFinalizeError("Reducer shard reference schema differs")
        shards = {
            split: _validate_shard(
                root / split,
                bucket=bucket,
                split=split,
                expected_reference=shard_references[split],
            )
            for split in ("train", "validation")
        }
        if (
            shards["train"]["record_count"] != counts["unique_train_records"]
            or shards["validation"]["record_count"]
            != counts["unique_validation_records"]
        ):
            raise ProductionFinalizeError("Reducer shard and receipt counts disagree")
        buckets.append(
            {
                "bucket": bucket,
                "receipt_file_sha256": file_sha256,
                "receipt_payload_sha256": payload_sha256,
                "shards": shards,
            }
        )
    return buckets, totals


def _write_create_only(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def finalize_production_corpus(
    *,
    output_root: Path,
    protocol_path: Path,
    protocol_file_sha256: str,
    reducer_source_path: Path,
    reducer_source_sha256: str,
    record_source_path: Path,
    record_source_sha256: str,
    tokenizer_source_path: Path,
    tokenizer_source_sha256: str,
    controller_source_sha256: str,
) -> dict[str, Any]:
    """Verify the complete map/reduce inventory and publish a non-authorizing manifest."""
    for path, expected, label in (
        (Path(__file__), controller_source_sha256, "Controller"),
        (reducer_source_path, reducer_source_sha256, "Reducer"),
        (record_source_path, record_source_sha256, "Record module"),
        (tokenizer_source_path, tokenizer_source_sha256, "Tokenizer"),
    ):
        _require_sha256(expected, f"{label} digest")
        if _file_sha256(path) != expected:
            raise ProductionFinalizeError(f"{label} source digest differs")
    protocol_raw = protocol_path.read_bytes()
    protocol = _strict_json_bytes(protocol_raw, "Production protocol")
    protocol_format = protocol.get("format_version")
    if (
        (
            protocol_format == "oas_production_materialization_protocol_v1"
            and protocol_raw != _canonical_json_bytes(protocol)
        )
        or hashlib.sha256(protocol_raw).hexdigest() != protocol_file_sha256
        or protocol_format not in SUPPORTED_PROTOCOL_FORMATS
        or protocol.get("test_access") != TEST_ACCESS
    ):
        raise ProductionFinalizeError("Production protocol identity differs")
    resource_receipt = _resource_receipt(protocol)
    execution = protocol.get("execution")
    inputs = protocol.get("inputs")
    if not isinstance(execution, Mapping) or not isinstance(inputs, Mapping):
        raise ProductionFinalizeError("Production execution policy is absent")
    task_count = _nonnegative_integer(execution.get("map_task_count"), "Map tasks")
    bucket_count = _nonnegative_integer(execution.get("bucket_count"), "Buckets")
    if (
        task_count < 1
        or bucket_count < 1
        or execution.get("reduce_task_count") != bucket_count
    ):
        raise ProductionFinalizeError("Production task topology differs")
    manifest_path = output_root / "corpus_manifest.json"
    receipt_path = output_root / "corpus_finalize_receipt.json"
    if any(
        path.exists() or path.is_symlink() for path in (manifest_path, receipt_path)
    ):
        raise ProductionFinalizeError("Production finalization output already exists")
    map_receipts, map_identity, map_totals = _load_map_receipts(
        output_root,
        task_count=task_count,
        bucket_count=bucket_count,
        protocol_format=str(protocol_format),
    )
    if map_totals["source_rows"] != inputs.get("source_record_count"):
        raise ProductionFinalizeError("Map source rows differ from the protocol")
    expected_reducer_sources = {
        "controller": reducer_source_sha256,
        "map_record": record_source_sha256,
        "tokenizer": tokenizer_source_sha256,
    }
    buckets, totals = _load_reduce_receipts(
        output_root,
        protocol_file_sha256=protocol_file_sha256,
        bucket_count=bucket_count,
        task_count=task_count,
        map_receipts=map_receipts,
        map_identity=map_identity,
        expected_source_sha256=expected_reducer_sources,
        protocol_format=str(protocol_format),
    )
    if totals["input_records"] != map_totals["accepted_records"]:
        raise ProductionFinalizeError(
            "Global reducer input differs from accepted map rows"
        )
    manifest = {
        "authorization": {
            "large_scale_pretraining": False,
            "production_corpus_use": False,
            "scientific_claim": False,
        },
        "bucket_count": bucket_count,
        "buckets": buckets,
        "counts": {
            **totals,
            "map_accepted_records": map_totals["accepted_records"],
            "map_source_rows": map_totals["source_rows"],
        },
        "format_version": MANIFEST_FORMAT,
        "global_order": "bucket_ascending_then_sequence_sha256_ascending",
        "map_identity": map_identity,
        "protocol_file_sha256": protocol_file_sha256,
        "resource_receipt": resource_receipt,
        "source_sha256": {
            "finalizer": controller_source_sha256,
            "reducer": reducer_source_sha256,
            "map_record": record_source_sha256,
            "tokenizer": tokenizer_source_sha256,
        },
        "test_access": TEST_ACCESS,
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        _canonical_json_bytes(manifest)
    ).hexdigest()
    manifest_bytes = _canonical_json_bytes(manifest)
    _write_create_only(manifest_path, manifest_bytes)
    receipt = {
        "authorization": manifest["authorization"],
        "claim_boundary": (
            "Complete primary map/reduce inventory only; independent full replay and "
            "one-GPU loader gates remain required"
        ),
        "counts": manifest["counts"],
        "format_version": FORMAT_VERSION,
        "manifest_file_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "manifest_payload_sha256": manifest["manifest_sha256"],
        "protocol_file_sha256": protocol_file_sha256,
        "resource_receipt": resource_receipt,
        "test_access": TEST_ACCESS,
    }
    receipt["receipt_sha256"] = hashlib.sha256(
        _canonical_json_bytes(receipt)
    ).hexdigest()
    _write_create_only(receipt_path, _canonical_json_bytes(receipt))
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-file-sha256", required=True)
    parser.add_argument("--reducer-source", type=Path, required=True)
    parser.add_argument("--reducer-source-sha256", required=True)
    parser.add_argument("--record-source", type=Path, required=True)
    parser.add_argument("--record-source-sha256", required=True)
    parser.add_argument("--tokenizer-source", type=Path, required=True)
    parser.add_argument("--tokenizer-source-sha256", required=True)
    parser.add_argument("--controller-source-sha256", required=True)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    receipt = finalize_production_corpus(
        output_root=arguments.output_root,
        protocol_path=arguments.protocol,
        protocol_file_sha256=arguments.protocol_file_sha256,
        reducer_source_path=arguments.reducer_source,
        reducer_source_sha256=arguments.reducer_source_sha256,
        record_source_path=arguments.record_source,
        record_source_sha256=arguments.record_source_sha256,
        tokenizer_source_path=arguments.tokenizer_source,
        tokenizer_source_sha256=arguments.tokenizer_source_sha256,
        controller_source_sha256=arguments.controller_source_sha256,
    )
    sys.stdout.buffer.write(_canonical_json_bytes(receipt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
