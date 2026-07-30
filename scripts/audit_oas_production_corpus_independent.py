#!/usr/bin/env python3
"""Independently replay OAS map buckets and verify every reduced shard."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import struct
from typing import Any, Mapping, NamedTuple

import numpy as np


FORMAT_VERSION = "oas_production_corpus_independent_audit_v1"
MAP_FORMAT = "oas_production_map_task_receipt_v1"
REDUCE_FORMAT = "oas_production_reduce_bucket_receipt_v1"
MANIFEST_FORMAT = "oas_production_corpus_manifest_v1"
SHARD_FORMAT = "oas_production_dense_bucket_shard_v1"
MAP_RECORD = struct.Struct(">32sQBBBBHH192s")
MAXIMUM_SEQUENCE_LENGTH = 192
MAXIMUM_SEQUENCE_TOKENS = 194
PAD_ID = 0
CLS_ID = 2
SEP_ID = 3
RESIDUES = "ACDEFGHIKLMNPQRSTVWYX"
TOKEN_ID = {residue: index + 5 for index, residue in enumerate(RESIDUES)}
TEST_ACCESS = {
    "downstream_metrics_used": False,
    "sabdab2_heldout_accessed": False,
}
IDLE_GPU_RESOURCE_RECEIPT = {
    "cuda_visible_devices": False,
    "gpu_compute_permitted": False,
    "scheduler_reserved_gpu_count": 1,
    "scheduler_reserved_gpu_mode": "scheduler_mandated_idle_only",
}


class IndependentCorpusAuditError(RuntimeError):
    """Report a replay, schema, digest, or validation-precedence disagreement."""


class Record(NamedTuple):
    """Hold one independently decoded fixed-width map record."""

    digest: bytes
    ordinal: int
    split_id: int
    chain_type: int
    stratum: int
    status_flags: int
    study_id: int
    sequence: bytes


def _reject_constant(value: str) -> None:
    raise IndependentCorpusAuditError(f"Non-finite JSON constant is forbidden: {value}")


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise IndependentCorpusAuditError(f"Duplicate JSON key is forbidden: {key}")
        value[key] = item
    return value


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def _nonnegative_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise IndependentCorpusAuditError(f"{label} is not a non-negative integer")
    return value


def _sha256_value(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise IndependentCorpusAuditError(f"{label} is not a lowercase SHA-256")
    return value


def _validated_mapper_counts(
    counts: object,
    *,
    protocol_format: str,
) -> dict[str, int]:
    """Validate mapper accounting while preserving inferable legacy zeros."""
    if (
        not isinstance(counts, Mapping)
        or "accepted_records" not in counts
        or "source_rows" not in counts
        or any(
            key
            not in {
                "accepted_records",
                "plan_fraction_rejected",
                "plan_fraction_selected",
                "source_rows",
            }
            and not key.startswith("quality_rejected:")
            for key in counts
        )
    ):
        raise IndependentCorpusAuditError("Mapper count schema differs")
    validated = {
        key: _nonnegative_integer(value, f"Mapper count {key}")
        for key, value in counts.items()
    }
    selected_present = "plan_fraction_selected" in validated
    rejected_present = "plan_fraction_rejected" in validated
    if protocol_format == "oas_production_materialization_protocol_v3":
        if not selected_present:
            raise IndependentCorpusAuditError("Mapper row accounting differs")
        if (
            not rejected_present
            and validated["plan_fraction_selected"] != validated["source_rows"]
        ):
            raise IndependentCorpusAuditError("Mapper row accounting differs")
        validated.setdefault("plan_fraction_rejected", 0)
    has_plan_accounting = selected_present and "plan_fraction_rejected" in validated
    selected = validated.get("plan_fraction_selected", 0)
    rejected = validated.get("plan_fraction_rejected", 0)
    quality_rejected = sum(
        value
        for key, value in validated.items()
        if key.startswith("quality_rejected:")
    )
    if (
        has_plan_accounting
        and (
            selected + rejected != validated["source_rows"]
            or validated["accepted_records"] + quality_rejected != selected
        )
    ) or (
        not has_plan_accounting
        and validated["accepted_records"] > validated["source_rows"]
    ):
        raise IndependentCorpusAuditError("Mapper row accounting differs")
    return validated


def _resource_receipt(protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Prove that independent replay performs no work on its reserved GPU."""
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") not in {"", "NoDevFiles"}:
        raise IndependentCorpusAuditError("Independent replay must not expose a GPU")
    policy = protocol.get("resource_policy")
    if not isinstance(policy, Mapping):
        raise IndependentCorpusAuditError("Production resource policy is absent")
    assigned_value = os.environ.get("SLURM_JOB_GPUS") or os.environ.get(
        "SLURM_STEP_GPUS"
    )
    assigned_count = len(
        [value for value in (assigned_value or "").split(",") if value]
    )
    allocation = policy.get("gpu_allocation_during_materialization", False)
    if allocation is False:
        if assigned_count:
            raise IndependentCorpusAuditError("Slurm assigned a GPU to CPU replay")
        return {
            "cuda_visible_devices": False,
            "gpu_compute_permitted": False,
            "scheduler_reserved_gpu_count": 0,
            "scheduler_reserved_gpu_mode": "none",
        }
    if allocation != "scheduler_mandated_idle_only":
        raise IndependentCorpusAuditError("Production GPU-allocation policy differs")
    expected = policy.get("scheduler_reserved_gpu_count")
    declared = os.environ.get("OAS_SCHEDULER_RESERVED_GPU_COUNT")
    if expected != 1 or declared != "1" or assigned_count != 1:
        raise IndependentCorpusAuditError("Scheduler-reserved idle GPU evidence differs")
    return dict(IDLE_GPU_RESOURCE_RECEIPT)


def _file_sha256(path: Path) -> str:
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise IndependentCorpusAuditError(f"Evidence is not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, *, require_canonical: bool = True) -> tuple[dict[str, Any], bytes]:
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise IndependentCorpusAuditError(f"Input is not a regular file: {path}")
    raw = path.read_bytes()
    try:
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_reject_duplicates,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise IndependentCorpusAuditError(f"Input is not strict ASCII JSON: {path}") from error
    if not isinstance(value, dict) or (
        require_canonical and raw != _canonical_json_bytes(value)
    ):
        raise IndependentCorpusAuditError(f"Input is not canonical JSON: {path}")
    return value, raw


def _load_self_hashed(path: Path, digest_field: str) -> tuple[dict[str, Any], str, str]:
    value, raw = _load_json(path)
    unsigned = dict(value)
    claimed = unsigned.pop(digest_field, None)
    observed = hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest()
    if claimed != observed:
        raise IndependentCorpusAuditError(f"Self hash differs: {path}")
    return value, hashlib.sha256(raw).hexdigest(), observed


def _assert_frozen(root: Path) -> None:
    if not root.is_dir() or root.is_symlink():
        raise IndependentCorpusAuditError(f"Published root is invalid: {root}")
    for path in (root, *root.rglob("*")):
        if path.is_symlink() or path.stat().st_mode & 0o222:
            raise IndependentCorpusAuditError(f"Published tree is mutable: {root}")


def _decode_record(payload: bytes, *, bucket: int, study_count: int) -> Record:
    """Decode and validate the binary schema without importing production code."""
    if len(payload) != MAP_RECORD.size:
        raise IndependentCorpusAuditError("Map record byte count differs")
    digest, ordinal, split_id, chain, stratum, flags, study, length, padded = (
        MAP_RECORD.unpack(payload)
    )
    if (
        digest[0] != bucket
        or split_id not in {0, 1}
        or chain not in {1, 2, 3}
        or stratum not in {0, 1, 2, 3}
        or flags & ~15
        or not 0 <= study < study_count
        or not 80 <= length <= MAXIMUM_SEQUENCE_LENGTH
        or any(padded[length:])
    ):
        raise IndependentCorpusAuditError("Map record field is outside its domain")
    sequence = padded[:length]
    if any(chr(value) not in RESIDUES for value in sequence):
        raise IndependentCorpusAuditError("Map sequence contains an unsupported residue")
    if hashlib.sha256(sequence).digest() != digest:
        raise IndependentCorpusAuditError("Map sequence digest differs")
    if MAP_RECORD.pack(
        digest,
        ordinal,
        split_id,
        chain,
        stratum,
        flags,
        study,
        length,
        sequence.ljust(MAXIMUM_SEQUENCE_LENGTH, b"\0"),
    ) != payload:
        raise IndependentCorpusAuditError("Map record is not canonical")
    return Record(digest, ordinal, split_id, chain, stratum, flags, study, sequence)


def _deduplicate(records: list[Record]) -> tuple[dict[int, list[Record]], dict[str, int]]:
    """Replay validation precedence and minimum-ordinal representative selection."""
    records.sort(key=lambda item: (item.digest, item.split_id, item.ordinal))
    selected = {0: [], 1: []}
    sample_keys: dict[int, bytes] = {}
    counts = {
        "cross_split_sequence_digests": 0,
        "input_records": len(records),
        "training_records_excluded_by_validation": 0,
        "unique_records": 0,
        "unique_train_records": 0,
        "unique_validation_records": 0,
        "within_split_duplicate_records": 0,
    }
    index = 0
    while index < len(records):
        end = index + 1
        while end < len(records) and records[end].digest == records[index].digest:
            end += 1
        group = records[index:end]
        if len({record.sequence for record in group}) != 1:
            raise IndependentCorpusAuditError("Digest collision has different sequences")
        validation = [record for record in group if record.split_id == 0]
        training = [record for record in group if record.split_id == 1]
        if validation and training:
            counts["cross_split_sequence_digests"] += 1
            counts["training_records_excluded_by_validation"] += len(training)
        counts["within_split_duplicate_records"] += max(0, len(validation) - 1)
        counts["within_split_duplicate_records"] += max(0, len(training) - 1)
        winners = validation if validation else training
        representative = min(winners, key=lambda item: item.ordinal)
        sample_key = int.from_bytes(representative.digest[:8], "big")
        previous = sample_keys.setdefault(sample_key, representative.digest)
        if previous != representative.digest:
            raise IndependentCorpusAuditError("Distinct digests share a uint64 sample key")
        selected[representative.split_id].append(representative)
        index = end
    counts["unique_validation_records"] = len(selected[0])
    counts["unique_train_records"] = len(selected[1])
    counts["unique_records"] = len(selected[0]) + len(selected[1])
    return selected, counts


def _load_shard_arrays(
    root: Path,
    *,
    bucket: int,
    split: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    manifest, file_sha256, payload_sha256 = _load_self_hashed(
        root / "manifest.json", "manifest_sha256"
    )
    if (
        manifest.get("format_version") != SHARD_FORMAT
        or manifest.get("bucket") != bucket
        or manifest.get("split") != split
        or manifest.get("test_access") != TEST_ACCESS
    ):
        raise IndependentCorpusAuditError("Shard manifest identity differs")
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise IndependentCorpusAuditError("Shard file inventory is absent")
    arrays: dict[str, np.ndarray] = {}
    for name, reference in files.items():
        path = root / name
        if (
            not isinstance(reference, Mapping)
            or path.stat().st_size != reference.get("bytes")
            or _file_sha256(path) != reference.get("sha256")
        ):
            raise IndependentCorpusAuditError("Shard file digest or size differs")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if str(array.dtype) != reference.get("dtype") or list(array.shape) != reference.get(
            "shape"
        ):
            raise IndependentCorpusAuditError("Shard array schema differs")
        arrays[name] = array
    record_count = _nonnegative_integer(manifest.get("record_count"), "Shard records")
    if any(array.shape[0] != record_count for array in arrays.values()):
        raise IndependentCorpusAuditError("Shard leading dimensions differ")
    reducer_reference = {
        "file_sha256": file_sha256,
        "payload_sha256": payload_sha256,
        "record_count": record_count,
    }
    published_reference = {
        "manifest_file_sha256": file_sha256,
        "manifest_payload_sha256": payload_sha256,
        "record_count": record_count,
        "relative_root": str(root.relative_to(root.parents[2])),
    }
    return arrays, reducer_reference, published_reference


def _verify_selected(
    arrays: Mapping[str, np.ndarray],
    records: list[Record],
) -> str:
    required = {
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
    if set(arrays) != required or any(array.shape[0] != len(records) for array in arrays.values()):
        raise IndependentCorpusAuditError("Shard array inventory differs")
    semantic = hashlib.sha256()
    for start in range(0, len(records), 8192):
        chunk = records[start : start + 8192]
        stop = start + len(chunk)
        digests = np.frombuffer(b"".join(record.digest for record in chunk), dtype=np.uint8)
        digests = digests.reshape(len(chunk), 32)
        lengths = np.asarray([len(record.sequence) for record in chunk], dtype=np.uint16)
        sample_keys = np.asarray(
            [int.from_bytes(record.digest[:8], "big") for record in chunk],
            dtype=np.uint64,
        )
        tokens = np.zeros((len(chunk), MAXIMUM_SEQUENCE_TOKENS), dtype=np.uint8)
        for row, record in enumerate(chunk):
            tokens[row, 0] = CLS_ID
            tokens[row, 1 : 1 + len(record.sequence)] = [
                TOKEN_ID[chr(value)] for value in record.sequence
            ]
            tokens[row, 1 + len(record.sequence)] = SEP_ID
            semantic.update(record.digest)
            semantic.update(
                struct.pack(
                    ">QBBBBH",
                    record.ordinal,
                    record.split_id,
                    record.chain_type,
                    record.stratum,
                    record.status_flags,
                    record.study_id,
                )
            )
        expected = {
            "chain_types.npy": np.asarray(
                [record.chain_type for record in chunk], dtype=np.uint8
            ),
            "lengths.npy": lengths + 2,
            "sample_keys.npy": sample_keys,
            "sequence_lengths.npy": lengths,
            "sequence_sha256.npy": digests,
            "status_flags.npy": np.asarray(
                [record.status_flags for record in chunk], dtype=np.uint8
            ),
            "strata.npy": np.asarray([record.stratum for record in chunk], dtype=np.uint8),
            "study_ids.npy": np.asarray([record.study_id for record in chunk], dtype=np.uint16),
            "tokens.npy": tokens,
        }
        for name, value in expected.items():
            if not np.array_equal(np.asarray(arrays[name][start:stop]), value):
                raise IndependentCorpusAuditError(f"Reduced shard differs: {name}")
    return semantic.hexdigest()


def _map_inventory(
    output_root: Path,
    *,
    task_count: int,
    bucket_count: int,
    protocol_format: str,
) -> tuple[
    list[dict[int, Mapping[str, Any]]],
    int,
    int,
    int,
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, int],
]:
    inventories = []
    receipt_references = []
    source_rows = 0
    accepted_records = 0
    study_count = -1
    metadata_normalization_totals = {"age_nan_to_null": 0, "strict_json": 0}
    common_identity: dict[str, Any] | None = None
    for task_id in range(task_count):
        root = output_root / "map" / f"mapper-{task_id:04d}"
        _assert_frozen(root)
        receipt, file_sha256, payload_sha256 = _load_self_hashed(
            root / "receipt.json", "receipt_sha256"
        )
        if (
            receipt.get("format_version") != MAP_FORMAT
            or receipt.get("task_id") != task_id
            or receipt.get("task_count") != task_count
            or receipt.get("claim_boundary", {}).get("map_task_complete") is not True
            or receipt.get("claim_boundary", {}).get("pilot_only") is not False
            or receipt.get("test_access") != TEST_ACCESS
        ):
            raise IndependentCorpusAuditError("Mapper completion identity differs")
        if (
            protocol_format
            in {
                "oas_production_materialization_protocol_v2",
                "oas_production_materialization_protocol_v3",
            }
            and receipt.get("resource_receipt") != IDLE_GPU_RESOURCE_RECEIPT
        ):
            raise IndependentCorpusAuditError("Mapper idle-GPU evidence differs")
        bindings = receipt.get("audit_bindings")
        sources = receipt.get("source_sha256")
        if not isinstance(bindings, Mapping) or not bindings:
            raise IndependentCorpusAuditError("Mapper audit bindings are absent")
        if not isinstance(sources, Mapping) or not sources:
            raise IndependentCorpusAuditError("Mapper source bindings are absent")
        for key, value in (*bindings.items(), *sources.items()):
            _sha256_value(value, f"Mapper identity {key}")
        current_study_count = _nonnegative_integer(
            receipt.get("study_count"), "Mapper study count"
        )
        if not 0 < current_study_count < 2**16:
            raise IndependentCorpusAuditError("Mapper study count is outside uint16")
        identity = {
            "audit_bindings": dict(bindings),
            "source_sha256": dict(sources),
            "study_count": current_study_count,
            "study_vocabulary_sha256": _sha256_value(
                receipt.get("study_vocabulary_sha256"),
                "Mapper study vocabulary",
            ),
            "task_count": receipt.get("task_count"),
        }
        if common_identity is None:
            common_identity = identity
        elif identity != common_identity:
            raise IndependentCorpusAuditError("Mapper identities disagree")
        if study_count < 0:
            study_count = current_study_count
        elif current_study_count != study_count:
            raise IndependentCorpusAuditError("Mapper study vocabularies disagree")
        counts = receipt.get("counts", {})
        _validated_mapper_counts(counts, protocol_format=protocol_format)
        normalization_counts = receipt.get("metadata_normalization_counts")
        if protocol_format == "oas_production_materialization_protocol_v3":
            if not isinstance(normalization_counts, Mapping):
                raise IndependentCorpusAuditError(
                    "Mapper metadata-normalization counts differ"
                )
            validated_normalization_counts = {
                key: _nonnegative_integer(value, f"Mapper normalization {key}")
                for key, value in normalization_counts.items()
            }
            if (
                any(
                    key not in metadata_normalization_totals
                    for key in validated_normalization_counts
                )
                or sum(validated_normalization_counts.values())
                != receipt.get("unit_count")
            ):
                raise IndependentCorpusAuditError(
                    "Mapper metadata-normalization counts differ"
                )
            observed_unit_counts = {"age_nan_to_null": 0, "strict_json": 0}
            units = receipt.get("unit_receipts")
            if not isinstance(units, list) or len(units) != receipt.get("unit_count"):
                raise IndependentCorpusAuditError("Mapper unit inventory differs")
            for unit in units:
                if not isinstance(unit, Mapping):
                    raise IndependentCorpusAuditError("Mapper unit receipt is invalid")
                mode = unit.get("metadata_normalization")
                if mode not in observed_unit_counts:
                    raise IndependentCorpusAuditError(
                        "Mapper unit normalization mode differs"
                    )
                observed_unit_counts[mode] += 1
            if {
                key: value for key, value in observed_unit_counts.items() if value
            } != validated_normalization_counts:
                raise IndependentCorpusAuditError(
                    "Mapper unit normalization accounting differs"
                )
            for key, value in observed_unit_counts.items():
                metadata_normalization_totals[key] += value
        source_rows += _nonnegative_integer(counts["source_rows"], "Mapper source rows")
        accepted_records += _nonnegative_integer(
            counts["accepted_records"], "Mapper accepted records"
        )
        inventory: dict[int, Mapping[str, Any]] = {}
        for item in receipt.get("bucket_inventory", []):
            bucket = item.get("bucket")
            records = item.get("records")
            byte_count = item.get("bytes")
            if (
                isinstance(bucket, bool)
                or not isinstance(bucket, int)
                or not 0 <= bucket < bucket_count
                or bucket in inventory
                or isinstance(records, bool)
                or not isinstance(records, int)
                or records < 1
                or byte_count != records * MAP_RECORD.size
            ):
                raise IndependentCorpusAuditError("Mapper bucket inventory is invalid")
            _sha256_value(item.get("file_sha256"), "Mapper bucket digest")
            inventory[bucket] = item
        inventories.append(inventory)
        receipt_references.append(
            {
                "file_sha256": file_sha256,
                "inventory": {key: value["records"] for key, value in inventory.items()},
                "payload_sha256": payload_sha256,
                "task_id": task_id,
            }
        )
    if common_identity is None or sum(
        sum(item["inventory"].values()) for item in receipt_references
    ) != accepted_records:
        raise IndependentCorpusAuditError("Mapper inventory accounting differs")
    return (
        inventories,
        source_rows,
        accepted_records,
        study_count,
        receipt_references,
        common_identity,
        metadata_normalization_totals,
    )


def _audit_bucket(
    bucket: int,
    *,
    output_root: Path,
    inventories: list[dict[int, Mapping[str, Any]]],
    study_count: int,
    protocol_file_sha256: str,
    protocol_format: str,
    map_receipts: list[dict[str, Any]],
    map_identity: Mapping[str, Any],
    manifest_bucket: Mapping[str, Any],
    manifest_sources: Mapping[str, Any],
) -> dict[str, Any]:
    records: list[Record] = []
    for mapper_id, inventory in enumerate(inventories):
        reference = inventory.get(bucket)
        path = output_root / "map" / f"mapper-{mapper_id:04d}/buckets/bucket-{bucket:03d}.bin"
        if reference is None:
            if path.exists() or path.is_symlink():
                raise IndependentCorpusAuditError("Unregistered mapper bucket exists")
            continue
        if (
            path.stat().st_size != reference.get("bytes")
            or _file_sha256(path) != reference.get("file_sha256")
            or path.stat().st_size != int(reference.get("records", -1)) * MAP_RECORD.size
        ):
            raise IndependentCorpusAuditError("Mapper bucket file differs")
        with path.open("rb") as handle:
            while payload := handle.read(MAP_RECORD.size):
                records.append(_decode_record(payload, bucket=bucket, study_count=study_count))
    selected, counts = _deduplicate(records)
    reduce_root = output_root / "reduce" / f"bucket-{bucket:03d}"
    _assert_frozen(reduce_root)
    reducer, reducer_file_sha256, reducer_payload_sha256 = _load_self_hashed(
        reduce_root / "receipt.json", "receipt_sha256"
    )
    expected_source_sha256 = {
        "controller": manifest_sources.get("reducer"),
        "map_record": manifest_sources.get("map_record"),
        "tokenizer": manifest_sources.get("tokenizer"),
    }
    for key, value in expected_source_sha256.items():
        _sha256_value(value, f"Manifest source {key}")
    boundary = reducer.get("claim_boundary")
    if (
        reducer.get("format_version") != REDUCE_FORMAT
        or reducer.get("bucket") != bucket
        or reducer.get("protocol_file_sha256") != protocol_file_sha256
        or reducer.get("map_identity") != dict(map_identity)
        or reducer.get("source_sha256") != expected_source_sha256
        or reducer.get("test_access") != TEST_ACCESS
        or not isinstance(boundary, Mapping)
        or boundary.get("reduced_bucket_complete") is not True
        or boundary.get("large_scale_pretraining_authorized") is not False
    ):
        raise IndependentCorpusAuditError("Reducer completion identity differs")
    if (
        protocol_format
        in {
            "oas_production_materialization_protocol_v2",
            "oas_production_materialization_protocol_v3",
        }
        and reducer.get("resource_receipt") != IDLE_GPU_RESOURCE_RECEIPT
    ):
        raise IndependentCorpusAuditError("Reducer idle-GPU evidence differs")
    expected_mapper_receipts = [
        {
            "file_sha256": item["file_sha256"],
            "mapper_id": mapper_id,
            "payload_sha256": item["payload_sha256"],
            "records_in_bucket": item["inventory"].get(bucket, 0),
        }
        for mapper_id, item in enumerate(map_receipts)
    ]
    if reducer.get("mapper_receipts") != expected_mapper_receipts:
        raise IndependentCorpusAuditError("Reducer mapper references differ")
    if reducer.get("counts") != counts:
        raise IndependentCorpusAuditError("Reducer counts differ from independent replay")
    digests = {}
    reducer_shards = {}
    published_shards = {}
    for split_id, split in ((0, "validation"), (1, "train")):
        arrays, reducer_reference, published_reference = _load_shard_arrays(
            reduce_root / split,
            bucket=bucket,
            split=split,
        )
        reducer_shards[split] = reducer_reference
        published_shards[split] = published_reference
        digests[split] = _verify_selected(arrays, selected[split_id])
    if reducer.get("shard_manifests") != reducer_shards:
        raise IndependentCorpusAuditError("Reducer shard references differ")
    expected_manifest_bucket = {
        "bucket": bucket,
        "receipt_file_sha256": reducer_file_sha256,
        "receipt_payload_sha256": reducer_payload_sha256,
        "shards": published_shards,
    }
    if dict(manifest_bucket) != expected_manifest_bucket:
        raise IndependentCorpusAuditError("Corpus manifest bucket reference differs")
    return {
        "bucket": bucket,
        "counts": counts,
        "selected_semantic_sha256": digests,
    }


def audit_corpus(
    *,
    output_root: Path,
    protocol_path: Path,
    worker_count: int,
) -> dict[str, Any]:
    if not 1 <= worker_count <= 8:
        raise IndependentCorpusAuditError("Worker count is outside the frozen domain")
    protocol, protocol_raw = _load_json(protocol_path, require_canonical=False)
    manifest, manifest_file_sha256, manifest_payload_sha256 = _load_self_hashed(
        output_root / "corpus_manifest.json", "manifest_sha256"
    )
    if (
        protocol.get("format_version") not in {
            "oas_production_materialization_protocol_v1",
            "oas_production_materialization_protocol_v2",
            "oas_production_materialization_protocol_v3",
        }
        or manifest.get("format_version") != MANIFEST_FORMAT
        or manifest.get("protocol_file_sha256") != hashlib.sha256(protocol_raw).hexdigest()
        or manifest.get("test_access") != TEST_ACCESS
    ):
        raise IndependentCorpusAuditError("Protocol or corpus manifest identity differs")
    resource_receipt = _resource_receipt(protocol)
    if manifest.get("resource_receipt") != resource_receipt:
        raise IndependentCorpusAuditError("Finalization resource evidence differs")
    execution = protocol.get("execution")
    inputs = protocol.get("inputs")
    if not isinstance(execution, Mapping) or not isinstance(inputs, Mapping):
        raise IndependentCorpusAuditError("Protocol topology or inputs are absent")
    task_count = _nonnegative_integer(execution.get("map_task_count"), "Map tasks")
    bucket_count = _nonnegative_integer(execution.get("bucket_count"), "Buckets")
    if (
        task_count < 1
        or bucket_count < 1
        or execution.get("reduce_task_count") != bucket_count
        or manifest.get("bucket_count") != bucket_count
        or manifest.get("global_order")
        != "bucket_ascending_then_sequence_sha256_ascending"
        or manifest.get("authorization")
        != {
            "large_scale_pretraining": False,
            "production_corpus_use": False,
            "scientific_claim": False,
        }
    ):
        raise IndependentCorpusAuditError("Protocol or manifest topology differs")
    raw_manifest_buckets = manifest.get("buckets")
    if not isinstance(raw_manifest_buckets, list) or len(raw_manifest_buckets) != bucket_count:
        raise IndependentCorpusAuditError("Manifest bucket inventory differs")
    manifest_buckets: dict[int, Mapping[str, Any]] = {}
    for item in raw_manifest_buckets:
        if not isinstance(item, Mapping):
            raise IndependentCorpusAuditError("Manifest bucket item is invalid")
        bucket = item.get("bucket")
        if (
            isinstance(bucket, bool)
            or not isinstance(bucket, int)
            or not 0 <= bucket < bucket_count
            or bucket in manifest_buckets
        ):
            raise IndependentCorpusAuditError("Manifest bucket identity differs")
        manifest_buckets[bucket] = item
    manifest_sources = manifest.get("source_sha256")
    if not isinstance(manifest_sources, Mapping) or set(manifest_sources) != {
        "finalizer",
        "map_record",
        "reducer",
        "tokenizer",
    }:
        raise IndependentCorpusAuditError("Manifest source inventory differs")
    for key, value in manifest_sources.items():
        _sha256_value(value, f"Manifest source {key}")
    protocol_format = str(protocol["format_version"])
    (
        inventories,
        source_rows,
        accepted_records,
        study_count,
        map_receipts,
        map_identity,
        metadata_normalization_counts,
    ) = _map_inventory(
        output_root,
        task_count=task_count,
        bucket_count=bucket_count,
        protocol_format=protocol_format,
    )
    if manifest.get("map_identity") != map_identity:
        raise IndependentCorpusAuditError("Manifest map identity differs")
    if protocol_format == "oas_production_materialization_protocol_v3":
        normalization = protocol.get("metadata_normalization")
        if not isinstance(normalization, Mapping) or metadata_normalization_counts != {
            "age_nan_to_null": normalization.get("affected_unit_count"),
            "strict_json": normalization.get("strict_unit_count"),
        }:
            raise IndependentCorpusAuditError(
                "Global metadata-normalization accounting differs"
            )
    protocol_file_sha256 = hashlib.sha256(protocol_raw).hexdigest()
    buckets = []
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        futures = {
            pool.submit(
                _audit_bucket,
                bucket,
                output_root=output_root,
                inventories=inventories,
                study_count=study_count,
                protocol_file_sha256=protocol_file_sha256,
                protocol_format=protocol_format,
                map_receipts=map_receipts,
                map_identity=map_identity,
                manifest_bucket=manifest_buckets[bucket],
                manifest_sources=manifest_sources,
            ): bucket
            for bucket in range(bucket_count)
        }
        for future in as_completed(futures):
            buckets.append(future.result())
    buckets.sort(key=lambda item: int(item["bucket"]))
    total_keys = set(buckets[0]["counts"]) if buckets else set()
    totals = {key: sum(item["counts"][key] for item in buckets) for key in total_keys}
    expected_counts = {
        **totals,
        "map_accepted_records": accepted_records,
        "map_source_rows": source_rows,
    }
    if (
        source_rows
        != _nonnegative_integer(inputs.get("source_record_count"), "Protocol source rows")
        or accepted_records != totals["input_records"]
        or manifest.get("counts") != expected_counts
    ):
        raise IndependentCorpusAuditError("Global count accounting differs")
    return {
        "authorization": {
            "independent_replay": True,
            "large_scale_pretraining": False,
            "production_corpus_use": False,
            "scientific_claim": False,
        },
        "buckets": buckets,
        "claim_boundary": (
            "Independent complete map-bucket replay and reduced-shard verification; "
            "one-GPU loader and checkpoint recovery remain required"
        ),
        "counts": expected_counts,
        "format_version": FORMAT_VERSION,
        "inputs": {
            "auditor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "manifest_file_sha256": manifest_file_sha256,
            "manifest_payload_sha256": manifest_payload_sha256,
            "output_root": str(output_root.resolve()),
            "protocol_file_sha256": hashlib.sha256(protocol_raw).hexdigest(),
        },
        "resource_receipt": resource_receipt,
        "metadata_normalization_counts": metadata_normalization_counts,
        "test_access": TEST_ACCESS,
        "worker_count": worker_count,
    }


def _publish(path: Path, value: dict[str, Any]) -> None:
    payload = dict(value)
    payload["receipt_sha256"] = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(_canonical_json_bytes(payload))
        handle.flush()
        os.fsync(handle.fileno())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--worker-count", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    result = audit_corpus(
        output_root=arguments.output_root,
        protocol_path=arguments.protocol,
        worker_count=arguments.worker_count,
    )
    _publish(arguments.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
