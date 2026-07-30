#!/usr/bin/env python3
"""Reduce one authenticated OAS digest bucket into immutable dense shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import sys
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import antibody_pretraining.production_materialization as record_module
import antibody_pretraining.tokenizer as tokenizer_module
from antibody_pretraining.production_materialization import (
    BUCKET_COUNT,
    ProductionMapRecord,
    iter_map_records,
    status_flag_mask,
)
from antibody_pretraining.tokenizer import (
    CHAIN_TYPE_TO_ID,
    PAD_ID,
    encode_unpaired,
)

FORMAT_VERSION = "oas_production_reduce_bucket_receipt_v1"
SHARD_FORMAT = "oas_production_dense_bucket_shard_v1"
SUPPORTED_PROTOCOL_FORMATS = {
    "oas_production_materialization_protocol_v1",
    "oas_production_materialization_protocol_v2",
    "oas_production_materialization_protocol_v3",
}
MAXIMUM_SEQUENCE_TOKENS = 194
TEST_ACCESS = {
    "downstream_metrics_used": False,
    "sabdab2_heldout_accessed": False,
}


class ProductionReduceError(RuntimeError):
    """Report an invalid map inventory or exact-reduction failure."""


def _resource_receipt(protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Prove that any scheduler-reserved GPU remains hidden and unused."""
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") not in {"", "NoDevFiles"}:
        raise ProductionReduceError("Production reduction must not expose a GPU")
    policy = protocol.get("resource_policy")
    if not isinstance(policy, Mapping):
        raise ProductionReduceError("Production resource policy is absent")
    assigned_value = os.environ.get("SLURM_JOB_GPUS") or os.environ.get(
        "SLURM_STEP_GPUS"
    )
    assigned_count = len(
        [value for value in (assigned_value or "").split(",") if value]
    )
    allocation = policy.get("gpu_allocation_during_materialization", False)
    if allocation is False:
        if assigned_count:
            raise ProductionReduceError("Slurm assigned a GPU to CPU-only reduction")
        return {
            "cuda_visible_devices": False,
            "gpu_compute_permitted": False,
            "scheduler_reserved_gpu_count": 0,
            "scheduler_reserved_gpu_mode": "none",
        }
    if allocation != "scheduler_mandated_idle_only":
        raise ProductionReduceError("Production GPU-allocation policy differs")
    expected = policy.get("scheduler_reserved_gpu_count")
    declared = os.environ.get("OAS_SCHEDULER_RESERVED_GPU_COUNT")
    if expected != 1 or declared != "1" or assigned_count != 1:
        raise ProductionReduceError("Scheduler-reserved idle GPU evidence differs")
    return {
        "cuda_visible_devices": False,
        "gpu_compute_permitted": False,
        "scheduler_reserved_gpu_count": 1,
        "scheduler_reserved_gpu_mode": "scheduler_mandated_idle_only",
    }


@dataclass(frozen=True)
class ReducedRecord:
    """Hold the canonical representative selected for one sequence digest."""

    source: ProductionMapRecord
    sample_key: int


def _reject_constant(value: str) -> None:
    """Reject non-finite JSON constants."""
    raise ValueError(f"Non-finite JSON constant: {value}")


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate JSON keys at every nesting level."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _strict_json_bytes(payload: bytes, label: str) -> dict[str, Any]:
    """Decode one finite duplicate-free ASCII JSON object."""
    try:
        parsed = json.loads(
            payload.decode("ascii"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ProductionReduceError(f"{label} must be strict ASCII JSON") from error
    if not isinstance(parsed, dict):
        raise ProductionReduceError(f"{label} must contain one JSON object")
    return parsed


def _canonical_json_bytes(payload: object) -> bytes:
    """Encode one finite canonical ASCII JSON value."""
    try:
        value = json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ProductionReduceError("Receipt is not canonical JSON") from error
    return (value + "\n").encode("ascii")


def _file_sha256(path: Path) -> str:
    """Hash one regular non-symlink file without loading it into memory."""
    if not path.is_file() or path.is_symlink():
        raise ProductionReduceError("Hashed path must be a regular non-symlink file")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    """Load one strict JSON file and preserve its exact bytes."""
    if not path.is_file() or path.is_symlink():
        raise ProductionReduceError(f"{label} must be a regular non-symlink file")
    payload = path.read_bytes()
    return _strict_json_bytes(payload, label), payload


def _verify_self_hash(payload: Mapping[str, Any], label: str) -> str:
    """Verify the canonical receipt hash after removing its digest field."""
    unsigned = dict(payload)
    claimed = unsigned.pop("receipt_sha256", None)
    observed = hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest()
    if claimed != observed:
        raise ProductionReduceError(f"{label} self-hash differs")
    return observed


def _sample_key(digest: bytes) -> int:
    """Return the frozen big-endian uint64 sample key from a full digest."""
    if len(digest) != 32:
        raise ProductionReduceError("Sequence digest byte count differs")
    return int.from_bytes(digest[:8], "big", signed=False)


def _require_sha256(value: object, label: str) -> str:
    """Return one lowercase SHA-256 digest or fail closed."""
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ProductionReduceError(f"{label} is not a lowercase SHA-256 digest")
    return value


def _register_sample_key(seen: dict[int, str], digest_hex: str) -> int:
    """Register one full digest and reject a truncated-key collision."""
    _require_sha256(digest_hex, "Sequence digest")
    try:
        digest = bytes.fromhex(digest_hex)
    except ValueError as error:
        raise ProductionReduceError("Sequence digest is not hexadecimal") from error
    sample_key = _sample_key(digest)
    previous = seen.setdefault(sample_key, digest_hex)
    if previous != digest_hex:
        raise ProductionReduceError(
            "Distinct sequence digests share one uint64 sample key"
        )
    return sample_key


def _deduplicate_records(
    records: Sequence[ProductionMapRecord],
) -> tuple[list[ReducedRecord], dict[str, int]]:
    """Apply exact digest sorting, validation precedence, and collision checks."""
    ordered = sorted(
        records,
        key=lambda item: (
            item.sequence_sha256,
            record_module.SPLIT_TO_ID[item.split],
            item.authority_ordinal,
        ),
    )
    selected: list[ReducedRecord] = []
    sample_keys: dict[int, str] = {}
    counts = {
        "cross_split_sequence_digests": 0,
        "input_records": len(ordered),
        "training_records_excluded_by_validation": 0,
        "within_split_duplicate_records": 0,
    }
    index = 0
    while index < len(ordered):
        end = index + 1
        digest_hex = ordered[index].sequence_sha256
        while end < len(ordered) and ordered[end].sequence_sha256 == digest_hex:
            end += 1
        group = ordered[index:end]
        sequences = {record.sequence for record in group}
        if len(sequences) != 1:
            raise ProductionReduceError(
                "Equal sequence digest has different canonical sequence bytes"
            )
        validation = [record for record in group if record.split == "validation"]
        training = [record for record in group if record.split == "train"]
        if validation and training:
            counts["cross_split_sequence_digests"] += 1
            counts["training_records_excluded_by_validation"] += len(training)
        winning_split = validation if validation else training
        counts["within_split_duplicate_records"] += max(
            0,
            len(validation) - 1,
        ) + max(0, len(training) - 1)
        representative = min(
            winning_split,
            key=lambda item: item.authority_ordinal,
        )
        digest = bytes.fromhex(digest_hex)
        if hashlib.sha256(representative.sequence.encode("ascii")).digest() != digest:
            raise ProductionReduceError("Reduced sequence digest differs")
        sample_key = _register_sample_key(sample_keys, digest_hex)
        selected.append(ReducedRecord(source=representative, sample_key=sample_key))
        index = end
    counts["unique_records"] = len(selected)
    counts["unique_train_records"] = sum(
        record.source.split == "train" for record in selected
    )
    counts["unique_validation_records"] = sum(
        record.source.split == "validation" for record in selected
    )
    return selected, counts


def _save_array(path: Path, array: np.ndarray) -> dict[str, Any]:
    """Write one NumPy array through an exclusive durable descriptor."""
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return {
        "bytes": path.stat().st_size,
        "dtype": str(array.dtype),
        "sha256": _file_sha256(path),
        "shape": list(array.shape),
    }


def _write_split_shard(
    output: Path,
    records: Sequence[ReducedRecord],
    *,
    bucket: int,
    split: str,
) -> dict[str, Any]:
    """Encode one globally ordered split into immutable fixed-width arrays."""
    output.mkdir()
    count = len(records)
    arrays = {
        "chain_types.npy": np.empty(count, dtype=np.uint8),
        "lengths.npy": np.empty(count, dtype=np.uint16),
        "sample_keys.npy": np.empty(count, dtype=np.uint64),
        "sequence_lengths.npy": np.empty(count, dtype=np.uint16),
        "sequence_sha256.npy": np.empty((count, 32), dtype=np.uint8),
        "status_flags.npy": np.empty(count, dtype=np.uint8),
        "strata.npy": np.empty(count, dtype=np.uint8),
        "study_ids.npy": np.empty(count, dtype=np.uint16),
        "tokens.npy": np.full(
            (count, MAXIMUM_SEQUENCE_TOKENS),
            PAD_ID,
            dtype=np.uint8,
        ),
    }
    for index, reduced in enumerate(records):
        source = reduced.source
        encoded = encode_unpaired(source.sequence, source.chain_type)
        if len(encoded.input_ids) > MAXIMUM_SEQUENCE_TOKENS:
            raise ProductionReduceError("Encoded sequence exceeds the frozen width")
        arrays["chain_types.npy"][index] = CHAIN_TYPE_TO_ID[source.chain_type]
        arrays["lengths.npy"][index] = len(encoded.input_ids)
        arrays["sample_keys.npy"][index] = reduced.sample_key
        arrays["sequence_lengths.npy"][index] = len(source.sequence)
        arrays["sequence_sha256.npy"][index] = np.frombuffer(
            bytes.fromhex(source.sequence_sha256),
            dtype=np.uint8,
        )
        arrays["status_flags.npy"][index] = status_flag_mask(source.status_flags)
        arrays["strata.npy"][index] = record_module.STRATUM_TO_ID[source.stratum]
        arrays["study_ids.npy"][index] = source.study_id
        arrays["tokens.npy"][index, : len(encoded.input_ids)] = encoded.input_ids
    files = {name: _save_array(output / name, value) for name, value in arrays.items()}
    manifest = {
        "bucket": bucket,
        "files": files,
        "format_version": SHARD_FORMAT,
        "global_order": "sequence_sha256_ascending",
        "maximum_sequence_tokens": MAXIMUM_SEQUENCE_TOKENS,
        "record_count": count,
        "split": split,
        "test_access": TEST_ACCESS,
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        _canonical_json_bytes(manifest)
    ).hexdigest()
    descriptor = os.open(
        output / "manifest.json",
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o444,
    )
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(_canonical_json_bytes(manifest))
        handle.flush()
        os.fsync(handle.fileno())
    return manifest


def _mapper_bucket_input(
    mapper_root: Path,
    *,
    mapper_id: int,
    task_count: int,
    bucket: int,
    common_identity: Mapping[str, Any] | None,
) -> tuple[list[ProductionMapRecord], dict[str, Any], Mapping[str, Any]]:
    """Authenticate one mapper receipt and read its selected bucket, if nonempty."""
    if not mapper_root.is_dir() or mapper_root.is_symlink():
        raise ProductionReduceError("Mapper root must be a non-symlink directory")
    receipt, receipt_bytes = _load_json(mapper_root / "receipt.json", "Map receipt")
    payload_sha256 = _verify_self_hash(receipt, "Map receipt")
    if (
        receipt.get("format_version") != "oas_production_map_task_receipt_v1"
        or receipt.get("task_id") != mapper_id
        or receipt.get("task_count") != task_count
        or receipt.get("claim_boundary", {}).get("map_task_complete") is not True
        or receipt.get("claim_boundary", {}).get("pilot_only") is not False
        or receipt.get("test_access") != TEST_ACCESS
    ):
        raise ProductionReduceError("Map receipt is not a complete production task")
    bindings = receipt.get("audit_bindings")
    if not isinstance(bindings, Mapping) or not bindings:
        raise ProductionReduceError("Map receipt audit bindings are absent")
    if any(
        _require_sha256(value, f"Map audit binding {key}") != value
        for key, value in bindings.items()
    ):
        raise AssertionError("unreachable")
    source_sha256 = receipt.get("source_sha256")
    if not isinstance(source_sha256, Mapping) or not source_sha256:
        raise ProductionReduceError("Map receipt source digests are absent")
    if any(
        _require_sha256(value, f"Map source digest {key}") != value
        for key, value in source_sha256.items()
    ):
        raise AssertionError("unreachable")
    study_count = receipt.get("study_count")
    if (
        isinstance(study_count, bool)
        or not isinstance(study_count, int)
        or not 0 < study_count < 2**16
    ):
        raise ProductionReduceError("Map receipt study count is invalid")
    study_vocabulary_sha256 = _require_sha256(
        receipt.get("study_vocabulary_sha256"),
        "Map study vocabulary digest",
    )
    identity = {
        "audit_bindings": dict(bindings),
        "source_sha256": dict(source_sha256),
        "study_count": study_count,
        "study_vocabulary_sha256": study_vocabulary_sha256,
        "task_count": receipt.get("task_count"),
    }
    if common_identity is not None and identity != dict(common_identity):
        raise ProductionReduceError("Map receipt identities disagree")
    raw_inventory = receipt.get("bucket_inventory")
    if not isinstance(raw_inventory, list):
        raise ProductionReduceError("Map bucket inventory is absent")
    inventory: dict[int, Mapping[str, Any]] = {}
    for item in raw_inventory:
        if not isinstance(item, Mapping):
            raise ProductionReduceError("Map bucket inventory item is invalid")
        item_bucket = item.get("bucket")
        records = item.get("records")
        byte_count = item.get("bytes")
        if (
            isinstance(item_bucket, bool)
            or not isinstance(item_bucket, int)
            or not 0 <= item_bucket < BUCKET_COUNT
            or isinstance(records, bool)
            or not isinstance(records, int)
            or records < 1
            or isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count < 1
        ):
            raise ProductionReduceError("Map bucket inventory item is invalid")
        _require_sha256(item.get("file_sha256"), "Map bucket file digest")
        if item_bucket in inventory:
            raise ProductionReduceError("Map bucket inventory is duplicated")
        inventory[item_bucket] = item
    path = mapper_root / "buckets" / f"bucket-{bucket:03d}.bin"
    reference = inventory.get(bucket)
    records: list[ProductionMapRecord] = []
    if reference is None:
        if path.exists() or path.is_symlink():
            raise ProductionReduceError("Unregistered map bucket file exists")
    else:
        expected_bytes = reference["records"] * record_module.MAP_RECORD.size
        if (
            reference.get("bytes") != expected_bytes
            or not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != expected_bytes
            or _file_sha256(path) != reference.get("file_sha256")
        ):
            raise ProductionReduceError("Map bucket file differs from its receipt")
        with path.open("rb") as handle:
            records.extend(iter_map_records(handle))
        if len(records) != reference.get("records"):
            raise ProductionReduceError("Map bucket record count differs")
        if any(
            bytes.fromhex(record.sequence_sha256)[0] != bucket for record in records
        ):
            raise ProductionReduceError("Map record is routed to the wrong bucket")
    receipt_reference = {
        "file_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
        "mapper_id": mapper_id,
        "payload_sha256": payload_sha256,
        "records_in_bucket": len(records),
    }
    return records, receipt_reference, identity


def _freeze_tree(root: Path) -> None:
    """Remove write bits from every published artifact and directory."""
    for path in sorted(root.rglob("*")):
        if path.is_file():
            path.chmod(0o444)
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_dir():
            path.chmod(0o555)
    root.chmod(0o555)


def _freeze_files(root: Path) -> None:
    """Freeze evidence files without preventing the atomic directory rename."""
    for path in sorted(root.rglob("*")):
        if path.is_file():
            path.chmod(0o444)


def _record_partial_failure(root: Path, error: BaseException) -> None:
    """Best-effort preserve a concise failure marker beside partial evidence."""
    if not root.exists() or root.is_symlink():
        return
    marker = root / "FAILURE.txt"
    try:
        descriptor = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
        with os.fdopen(descriptor, "wb") as handle:
            message = f"{type(error).__name__}: {error}\n".encode(
                "ascii",
                errors="backslashreplace",
            )
            handle.write(message)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError:
        pass


def reduce_production_bucket(
    *,
    bucket: int,
    map_root: Path,
    protocol_path: Path,
    protocol_file_sha256: str,
    output_root: Path,
    expected_controller_source_sha256: str,
    expected_record_source_sha256: str,
    expected_tokenizer_source_sha256: str,
) -> dict[str, Any]:
    """Authenticate every mapper and atomically publish one reduced bucket."""
    for path, expected, label in (
        (Path(__file__), expected_controller_source_sha256, "Controller"),
        (Path(record_module.__file__), expected_record_source_sha256, "Record module"),
        (
            Path(tokenizer_module.__file__),
            expected_tokenizer_source_sha256,
            "Tokenizer",
        ),
    ):
        if _file_sha256(path) != expected:
            raise ProductionReduceError(f"{label} source digest differs")
    if (
        isinstance(bucket, bool)
        or not isinstance(bucket, int)
        or not 0 <= bucket < BUCKET_COUNT
    ):
        raise ProductionReduceError("Bucket is outside the frozen domain")
    protocol, protocol_bytes = _load_json(protocol_path, "Protocol")
    _require_sha256(protocol_file_sha256, "Protocol file digest")
    if (
        hashlib.sha256(protocol_bytes).hexdigest() != protocol_file_sha256
        or protocol.get("format_version") not in SUPPORTED_PROTOCOL_FORMATS
    ):
        raise ProductionReduceError("Production protocol identity differs")
    resource_receipt = _resource_receipt(protocol)
    execution = protocol.get("execution")
    if not isinstance(execution, Mapping):
        raise ProductionReduceError("Production execution policy is absent")
    task_count = execution.get("map_task_count")
    if (
        isinstance(task_count, bool)
        or not isinstance(task_count, int)
        or task_count < 1
    ):
        raise ProductionReduceError("Map task count is invalid")
    minimum_free = protocol.get("resource_policy", {}).get(
        "shared_storage_minimum_free_bytes"
    )
    if (
        isinstance(minimum_free, bool)
        or not isinstance(minimum_free, int)
        or shutil.disk_usage(output_root.parent).free < minimum_free
    ):
        raise ProductionReduceError("Shared storage does not satisfy the reserve")

    final_root = output_root / "reduce" / f"bucket-{bucket:03d}"
    if final_root.exists() or final_root.is_symlink():
        raise ProductionReduceError("Published reducer output already exists")
    partial_root = (
        output_root / "work" / f"reducer-{bucket:03d}.partial-{uuid.uuid4().hex}"
    )
    partial_root.mkdir(parents=True, exist_ok=False)
    try:
        all_records: list[ProductionMapRecord] = []
        mapper_receipts = []
        common_identity: Mapping[str, Any] | None = None
        for mapper_id in range(task_count):
            records, reference, identity = _mapper_bucket_input(
                map_root / "map" / f"mapper-{mapper_id:04d}",
                mapper_id=mapper_id,
                task_count=task_count,
                bucket=bucket,
                common_identity=common_identity,
            )
            if common_identity is None:
                common_identity = identity
            all_records.extend(records)
            mapper_receipts.append(reference)
        reduced, counts = _deduplicate_records(all_records)
        train = [record for record in reduced if record.source.split == "train"]
        validation = [
            record for record in reduced if record.source.split == "validation"
        ]
        manifests = {
            "train": _write_split_shard(
                partial_root / "train",
                train,
                bucket=bucket,
                split="train",
            ),
            "validation": _write_split_shard(
                partial_root / "validation",
                validation,
                bucket=bucket,
                split="validation",
            ),
        }
        receipt = {
            "map_identity": dict(common_identity or {}),
            "bucket": bucket,
            "claim_boundary": {
                "global_dedup_complete": False,
                "large_scale_pretraining_authorized": False,
                "reduced_bucket_complete": True,
                "scientific_claim": False,
            },
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "counts": counts,
            "format_version": FORMAT_VERSION,
            "host": socket.gethostname(),
            "resource_receipt": resource_receipt,
            "mapper_receipts": mapper_receipts,
            "protocol_file_sha256": protocol_file_sha256,
            "shard_manifests": {
                split: {
                    "file_sha256": _file_sha256(partial_root / split / "manifest.json"),
                    "payload_sha256": manifest["manifest_sha256"],
                    "record_count": manifest["record_count"],
                }
                for split, manifest in manifests.items()
            },
            "source_sha256": {
                "controller": expected_controller_source_sha256,
                "map_record": expected_record_source_sha256,
                "tokenizer": expected_tokenizer_source_sha256,
            },
            "test_access": TEST_ACCESS,
        }
        receipt["receipt_sha256"] = hashlib.sha256(
            _canonical_json_bytes(receipt)
        ).hexdigest()
        descriptor = os.open(
            partial_root / "receipt.json",
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o444,
        )
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_json_bytes(receipt))
            handle.flush()
            os.fsync(handle.fileno())
        _freeze_files(partial_root)
        final_root.parent.mkdir(parents=True, exist_ok=True)
        partial_root.rename(final_root)
        _freeze_tree(final_root)
        return receipt
    except BaseException as error:
        _record_partial_failure(partial_root, error)
        raise


def _parser() -> argparse.ArgumentParser:
    """Build the strict reducer command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", type=int, required=True)
    parser.add_argument("--map-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-file-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--controller-source-sha256", required=True)
    parser.add_argument("--record-source-sha256", required=True)
    parser.add_argument("--tokenizer-source-sha256", required=True)
    return parser


def main() -> int:
    """Run one reducer and print its canonical receipt."""
    arguments = _parser().parse_args()
    receipt = reduce_production_bucket(
        bucket=arguments.bucket,
        map_root=arguments.map_root,
        protocol_path=arguments.protocol,
        protocol_file_sha256=arguments.protocol_file_sha256,
        output_root=arguments.output_root,
        expected_controller_source_sha256=arguments.controller_source_sha256,
        expected_record_source_sha256=arguments.record_source_sha256,
        expected_tokenizer_source_sha256=arguments.tokenizer_source_sha256,
    )
    sys.stdout.buffer.write(_canonical_json_bytes(receipt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
