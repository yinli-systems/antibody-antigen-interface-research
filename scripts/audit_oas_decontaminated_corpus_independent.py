#!/usr/bin/env python3
"""Independently replay digest exclusion and row alignment for every OAS shard.

This auditor deliberately does not import the filtering implementation.  It
reconstructs the exclusion mask from the frozen digest stream, authenticates
both source and candidate arrays, and compares every retained row in every
array before publishing a non-authorizing receipt.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys
from typing import Any, Mapping, Optional

import numpy as np


AUDIT_FORMAT = "oas_production_corpus_independent_audit_v1"
FINALIZE_FORMAT = "oas_decontaminated_corpus_finalize_v1"
MANIFEST_FORMAT = "oas_production_corpus_manifest_v1"
PROTOCOL_FORMAT = "oas_decontaminated_corpus_protocol_v1"
SHARD_FORMAT = "oas_production_dense_bucket_shard_v1"
EXCLUSION_FORMAT = "oas_near_identity_exclusion_compilation_v1"
SOURCE_AUDIT_FORMAT = "oas_production_corpus_independent_audit_v1"
EXPECTED_ARRAYS = {
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
SPLITS = ("train", "validation")
TEST_ACCESS = {
    "downstream_metrics_used": False,
    "sabdab2_heldout_accessed": False,
}
AUTHORIZATION = {
    "large_scale_pretraining": False,
    "production_corpus_use": False,
    "scientific_claim": False,
}
AUDIT_AUTHORIZATION = {
    "independent_replay": True,
    **AUTHORIZATION,
}


class IndependentDecontaminationAuditError(RuntimeError):
    """Report any provenance, exclusion, or retained-row disagreement."""


def _canonical_json_bytes(value: object) -> bytes:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise IndependentDecontaminationAuditError(
            "Payload is not finite canonical JSON"
        ) from error
    return (encoded + "\n").encode("ascii")


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise IndependentDecontaminationAuditError("Duplicate JSON key: " + key)
        output[key] = value
    return output


def _reject_constant(value: str) -> None:
    raise IndependentDecontaminationAuditError("Non-finite JSON constant: " + value)


def _sha256_value(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise IndependentDecontaminationAuditError(label + " is not SHA-256")
    return value


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise IndependentDecontaminationAuditError(label + " is not non-negative")
    return value


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise IndependentDecontaminationAuditError(label + " is not an object")
    return value


def _regular_file(path: Path, label: str, *, frozen: bool = True) -> Path:
    candidate = Path(path).expanduser().absolute()
    try:
        metadata = candidate.lstat()
    except OSError as error:
        raise IndependentDecontaminationAuditError("Cannot stat " + label) from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise IndependentDecontaminationAuditError(label + " is not a regular file")
    if frozen and metadata.st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise IndependentDecontaminationAuditError(label + " remains writable")
    return candidate


def _file_sha256(path: Path, *, frozen: bool = True) -> str:
    digest = hashlib.sha256()
    with _regular_file(path, "Hashed evidence", frozen=frozen).open(
        "rb", buffering=16 * 1024 * 1024
    ) as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_signed(
    path: Path,
    *,
    digest_field: str,
    label: str,
    expected_file_sha256: Optional[str] = None,
    expected_payload_sha256: Optional[str] = None,
) -> tuple[dict[str, Any], str, str]:
    candidate = _regular_file(path, label)
    raw = candidate.read_bytes()
    try:
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_reject_duplicates,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise IndependentDecontaminationAuditError(
            label + " is not strict ASCII JSON"
        ) from error
    if not isinstance(value, dict) or raw != _canonical_json_bytes(value):
        raise IndependentDecontaminationAuditError(label + " is not canonical JSON")
    unsigned = dict(value)
    claimed = _sha256_value(unsigned.pop(digest_field, None), label + " self hash")
    observed = hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest()
    file_digest = hashlib.sha256(raw).hexdigest()
    if claimed != observed:
        raise IndependentDecontaminationAuditError(label + " self hash differs")
    if expected_file_sha256 is not None and file_digest != _sha256_value(
        expected_file_sha256, label + " expected file hash"
    ):
        raise IndependentDecontaminationAuditError(label + " file hash differs")
    if expected_payload_sha256 is not None and claimed != _sha256_value(
        expected_payload_sha256, label + " expected payload hash"
    ):
        raise IndependentDecontaminationAuditError(label + " payload hash differs")
    return value, file_digest, claimed


def _safe_relative_root(base: Path, relative_value: object, label: str) -> Path:
    if not isinstance(relative_value, str) or not relative_value:
        raise IndependentDecontaminationAuditError(label + " relative root is invalid")
    relative = PurePosixPath(relative_value)
    if relative.is_absolute() or "." in relative.parts or ".." in relative.parts:
        raise IndependentDecontaminationAuditError(label + " relative root is unsafe")
    base = base.resolve(strict=True)
    current = base
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise IndependentDecontaminationAuditError(label + " path is symbolic")
    resolved = current.resolve(strict=True)
    if not resolved.is_relative_to(base) or not resolved.is_dir():
        raise IndependentDecontaminationAuditError(label + " escapes its corpus")
    return resolved


def _assert_frozen_tree(root: Path, label: str) -> None:
    if root.is_symlink() or not root.is_dir():
        raise IndependentDecontaminationAuditError(label + " is absent or symbolic")
    for path in (root, *root.rglob("*")):
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode) or metadata.st_mode & (
            stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
        ):
            raise IndependentDecontaminationAuditError(label + " remains mutable")


def _resource_receipt() -> dict[str, Any]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible not in {"", "NoDevFiles"}:
        raise IndependentDecontaminationAuditError("Auditor must not expose a GPU")
    assigned = os.environ.get("SLURM_JOB_GPUS") or os.environ.get(
        "SLURM_STEP_GPUS", ""
    )
    assigned_count = len([value for value in assigned.split(",") if value])
    declared = os.environ.get("OAS_SCHEDULER_RESERVED_GPU_COUNT")
    if assigned_count == 0 and declared is None:
        mode = "none"
    elif assigned_count == 1 and declared == "1":
        mode = "scheduler_mandated_idle_only"
    else:
        raise IndependentDecontaminationAuditError(
            "Auditor scheduler resource evidence differs"
        )
    return {
        "cuda_visible_devices": False,
        "gpu_compute_permitted": False,
        "scheduler_reserved_gpu_count": assigned_count,
        "scheduler_reserved_gpu_mode": mode,
    }


def _close_arrays(arrays: Mapping[str, np.ndarray]) -> None:
    for array in arrays.values():
        mapping = getattr(array, "_mmap", None)
        if mapping is not None:
            mapping.close()


def _open_shard(
    *,
    corpus_root: Path,
    reference: Mapping[str, Any],
    expected_relative_root: str,
    bucket: int,
    split: str,
    candidate: bool,
) -> tuple[dict[str, np.ndarray], dict[str, Any], str, str]:
    if reference.get("relative_root") != expected_relative_root:
        raise IndependentDecontaminationAuditError("Shard relative root differs")
    root = _safe_relative_root(corpus_root, expected_relative_root, "Shard")
    _assert_frozen_tree(root, "Shard")
    manifest, file_digest, payload_digest = _load_signed(
        root / "manifest.json",
        digest_field="manifest_sha256",
        label="Shard manifest",
        expected_file_sha256=str(reference.get("manifest_file_sha256")),
        expected_payload_sha256=str(reference.get("manifest_payload_sha256")),
    )
    if (
        manifest.get("format_version") != SHARD_FORMAT
        or manifest.get("bucket") != bucket
        or manifest.get("split") != split
        or manifest.get("record_count") != reference.get("record_count")
        or manifest.get("test_access") != TEST_ACCESS
    ):
        raise IndependentDecontaminationAuditError("Shard manifest identity differs")
    if candidate:
        decontamination = _mapping(
            manifest.get("decontamination"), "Candidate decontamination"
        )
        if decontamination.get("exclusion_rule") != "full_sequence_sha256_membership":
            raise IndependentDecontaminationAuditError(
                "Candidate exclusion rule differs"
            )
    files = _mapping(manifest.get("files"), "Shard files")
    if set(files) != EXPECTED_ARRAYS or {
        path.name for path in root.iterdir()
    } != EXPECTED_ARRAYS | {"manifest.json"}:
        raise IndependentDecontaminationAuditError("Shard file inventory differs")
    record_count = _integer(manifest.get("record_count"), "Shard records")
    arrays: dict[str, np.ndarray] = {}
    try:
        for name in sorted(EXPECTED_ARRAYS):
            item = _mapping(files.get(name), "Shard array reference")
            path = _regular_file(root / name, "Shard array")
            if (
                path.stat().st_size != item.get("bytes")
                or _file_sha256(path) != item.get("sha256")
            ):
                raise IndependentDecontaminationAuditError(
                    "Shard array size or hash differs"
                )
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if (
                str(array.dtype) != item.get("dtype")
                or list(array.shape) != item.get("shape")
                or array.shape[0] != record_count
                or array.flags.writeable
            ):
                raise IndependentDecontaminationAuditError(
                    "Shard array schema differs"
                )
            arrays[name] = array
        return arrays, manifest, file_digest, payload_digest
    except BaseException:
        _close_arrays(arrays)
        raise


def _bucket_bounds(path: Path, record_count: int, bucket: int) -> tuple[int, int]:
    def lower_bound(first_byte: int) -> int:
        low = 0
        high = record_count
        with path.open("rb", buffering=0) as handle:
            while low < high:
                middle = (low + high) // 2
                handle.seek(middle * 32)
                observed = handle.read(1)
                if len(observed) != 1:
                    raise IndependentDecontaminationAuditError(
                        "Exclusion stream is truncated"
                    )
                if observed[0] < first_byte:
                    low = middle + 1
                else:
                    high = middle
        return low

    start = lower_bound(bucket)
    end = record_count if bucket == 255 else lower_bound(bucket + 1)
    if not 0 <= start <= end <= record_count:
        raise IndependentDecontaminationAuditError("Exclusion bucket bounds differ")
    return start, end


def _exclusion_bucket(
    path: Path, *, record_count: int, bucket: int
) -> tuple[np.ndarray, np.ndarray]:
    start, end = _bucket_bounds(path, record_count, bucket)
    if start == end:
        raw = np.empty((0, 32), dtype=np.uint8)
    else:
        raw = np.memmap(
            path,
            dtype=np.uint8,
            mode="r",
            offset=start * 32,
            shape=(end - start, 32),
        )
    keys = raw.reshape(-1, 32).view("|S32").reshape(-1)
    if keys.size and (
        np.any(raw[:, 0] != bucket)
        or (keys.size > 1 and np.any(keys[1:] <= keys[:-1]))
    ):
        raise IndependentDecontaminationAuditError(
            "Exclusion bucket is not strictly sorted"
        )
    return raw, keys


def _digest_keys(array: np.ndarray, *, bucket: int, label: str) -> np.ndarray:
    if array.dtype != np.dtype("uint8") or array.ndim != 2 or array.shape[1] != 32:
        raise IndependentDecontaminationAuditError(label + " digest schema differs")
    keys = array.reshape(-1, 32).view("|S32").reshape(-1)
    if keys.size and (
        np.any(array[:, 0] != bucket)
        or (keys.size > 1 and np.any(keys[1:] <= keys[:-1]))
    ):
        raise IndependentDecontaminationAuditError(label + " digest order differs")
    return keys


def _audit_split(
    *,
    bucket: int,
    split: str,
    source_root: Path,
    source_reference: Mapping[str, Any],
    candidate_root: Path,
    candidate_reference: Mapping[str, Any],
    exclusion_keys: np.ndarray,
    chunk_records: int,
) -> dict[str, Any]:
    source_arrays, source_manifest, source_file, source_payload = _open_shard(
        corpus_root=source_root,
        reference=source_reference,
        expected_relative_root=str(source_reference.get("relative_root")),
        bucket=bucket,
        split=split,
        candidate=False,
    )
    candidate_arrays, candidate_manifest, candidate_file, candidate_payload = _open_shard(
        corpus_root=candidate_root,
        reference=candidate_reference,
        expected_relative_root=f"reduce/bucket-{bucket:03d}/{split}",
        bucket=bucket,
        split=split,
        candidate=True,
    )
    semantic = hashlib.sha256()
    try:
        source_keys = _digest_keys(
            source_arrays["sequence_sha256.npy"], bucket=bucket, label="Source"
        )
        candidate_keys = _digest_keys(
            candidate_arrays["sequence_sha256.npy"],
            bucket=bucket,
            label="Candidate",
        )
        if exclusion_keys.size:
            positions = np.searchsorted(exclusion_keys, source_keys)
            bounded = np.minimum(positions, exclusion_keys.size - 1)
            excluded = (positions < exclusion_keys.size) & (
                exclusion_keys[bounded] == source_keys
            )
            candidate_positions = np.searchsorted(exclusion_keys, candidate_keys)
            candidate_bounded = np.minimum(
                candidate_positions, exclusion_keys.size - 1
            )
            if np.any(
                (candidate_positions < exclusion_keys.size)
                & (exclusion_keys[candidate_bounded] == candidate_keys)
            ):
                raise IndependentDecontaminationAuditError(
                    "Candidate still overlaps the exclusion set"
                )
        else:
            excluded = np.zeros(len(source_keys), dtype=np.bool_)
        keep = ~excluded
        retained = int(np.count_nonzero(keep))
        if retained != len(candidate_keys):
            raise IndependentDecontaminationAuditError(
                "Candidate retained-row count differs"
            )
        candidate_offset = 0
        for name in sorted(EXPECTED_ARRAYS):
            source = source_arrays[name]
            candidate = candidate_arrays[name]
            semantic.update(name.encode("ascii") + b"\0")
            candidate_offset = 0
            for start in range(0, len(keep), chunk_records):
                end = min(start + chunk_records, len(keep))
                expected = np.asarray(source[start:end])[keep[start:end]]
                stop = candidate_offset + len(expected)
                observed = np.asarray(candidate[candidate_offset:stop])
                if not np.array_equal(expected, observed):
                    raise IndependentDecontaminationAuditError(
                        "Candidate row alignment differs: " + name
                    )
                semantic.update(expected.tobytes(order="C"))
                candidate_offset = stop
            if candidate_offset != len(candidate):
                raise IndependentDecontaminationAuditError(
                    "Candidate array length differs: " + name
                )
        decontamination = _mapping(
            candidate_manifest.get("decontamination"), "Candidate decontamination"
        )
        excluded_count = int(np.count_nonzero(excluded))
        if (
            decontamination.get("source_record_count") != len(source_keys)
            or decontamination.get("excluded_records") != excluded_count
            or candidate_manifest.get("source_manifest_file_sha256") != source_file
            or candidate_manifest.get("source_manifest_payload_sha256")
            != source_payload
        ):
            raise IndependentDecontaminationAuditError(
                "Candidate source binding or counts differ"
            )
        return {
            "candidate_manifest_file_sha256": candidate_file,
            "candidate_manifest_payload_sha256": candidate_payload,
            "excluded_records": excluded_count,
            "replay_semantic_sha256": semantic.hexdigest(),
            "retained_records": retained,
            "source_manifest_file_sha256": source_file,
            "source_manifest_payload_sha256": source_payload,
            "source_records": len(source_keys),
            "split": split,
        }
    finally:
        _close_arrays(source_arrays)
        _close_arrays(candidate_arrays)


def _audit_bucket(
    bucket: int,
    *,
    source_root: Path,
    source_bucket: Mapping[str, Any],
    candidate_root: Path,
    candidate_bucket: Mapping[str, Any],
    exclusion_path: Path,
    exclusion_record_count: int,
    chunk_records: int,
) -> dict[str, Any]:
    exclusion_raw, exclusion_keys = _exclusion_bucket(
        exclusion_path, record_count=exclusion_record_count, bucket=bucket
    )
    try:
        source_shards = _mapping(source_bucket.get("shards"), "Source shards")
        candidate_shards = _mapping(candidate_bucket.get("shards"), "Candidate shards")
        if set(source_shards) != set(SPLITS) or set(candidate_shards) != set(SPLITS):
            raise IndependentDecontaminationAuditError("Bucket split inventory differs")
        splits = {
            split: _audit_split(
                bucket=bucket,
                split=split,
                source_root=source_root,
                source_reference=_mapping(source_shards.get(split), "Source shard"),
                candidate_root=candidate_root,
                candidate_reference=_mapping(
                    candidate_shards.get(split), "Candidate shard"
                ),
                exclusion_keys=exclusion_keys,
                chunk_records=chunk_records,
            )
            for split in SPLITS
        }
        return {"bucket": bucket, "shards": splits}
    finally:
        mapping = getattr(exclusion_raw, "_mmap", None)
        if mapping is not None:
            mapping.close()


def audit_corpus(
    *,
    source_root: Path,
    exclusion_root: Path,
    output_root: Path,
    protocol_path: Path,
    expected_protocol_file_sha256: str,
    auditor_source_sha256: str,
    worker_count: int,
    chunk_records: int,
) -> dict[str, Any]:
    if not 1 <= worker_count <= 8 or chunk_records <= 0:
        raise IndependentDecontaminationAuditError("Auditor execution domain differs")
    actual_auditor_sha256 = _file_sha256(Path(__file__), frozen=False)
    if actual_auditor_sha256 != _sha256_value(
        auditor_source_sha256, "Auditor source"
    ):
        raise IndependentDecontaminationAuditError("Auditor source hash differs")
    protocol, protocol_file, protocol_payload = _load_signed(
        protocol_path,
        digest_field="protocol_sha256",
        label="Decontamination protocol",
        expected_file_sha256=expected_protocol_file_sha256,
    )
    execution = _mapping(protocol.get("execution"), "Protocol execution")
    inputs = _mapping(protocol.get("inputs"), "Protocol inputs")
    sources = _mapping(protocol.get("source_sha256"), "Protocol sources")
    bucket_count = _integer(execution.get("bucket_count"), "Buckets")
    if (
        protocol.get("format_version") != PROTOCOL_FORMAT
        or protocol.get("authorization") != AUTHORIZATION
        or protocol.get("test_access") != TEST_ACCESS
        or not 1 <= bucket_count <= 256
        or sources.get("independent_auditor") != actual_auditor_sha256
    ):
        raise IndependentDecontaminationAuditError("Auditor protocol scope differs")
    source_manifest, source_manifest_file, source_manifest_payload = _load_signed(
        source_root / "corpus_manifest.json",
        digest_field="manifest_sha256",
        label="Source corpus manifest",
        expected_file_sha256=str(inputs.get("source_manifest_file_sha256")),
        expected_payload_sha256=str(inputs.get("source_manifest_payload_sha256")),
    )
    source_audit, source_audit_file, source_audit_payload = _load_signed(
        source_root / "independent_audit_receipt.json",
        digest_field="receipt_sha256",
        label="Source independent audit",
        expected_file_sha256=str(inputs.get("source_audit_file_sha256")),
        expected_payload_sha256=str(inputs.get("source_audit_payload_sha256")),
    )
    exclusion, exclusion_file, exclusion_payload = _load_signed(
        exclusion_root / "receipt.json",
        digest_field="receipt_sha256",
        label="Exclusion receipt",
        expected_file_sha256=str(inputs.get("exclusion_receipt_file_sha256")),
        expected_payload_sha256=str(inputs.get("exclusion_receipt_payload_sha256")),
    )
    candidate_manifest, candidate_file, candidate_payload = _load_signed(
        output_root / "corpus_manifest.json",
        digest_field="manifest_sha256",
        label="Candidate corpus manifest",
    )
    finalize, _, _ = _load_signed(
        output_root / "corpus_finalize_receipt.json",
        digest_field="receipt_sha256",
        label="Candidate finalization receipt",
    )
    if (
        source_manifest.get("format_version") != MANIFEST_FORMAT
        or source_manifest.get("bucket_count") != bucket_count
        or source_manifest.get("authorization") != AUTHORIZATION
        or source_manifest.get("test_access") != TEST_ACCESS
        or source_audit.get("format_version") != SOURCE_AUDIT_FORMAT
        or source_audit.get("authorization") != AUDIT_AUTHORIZATION
        or source_audit.get("inputs", {}).get("manifest_file_sha256")
        != source_manifest_file
        or source_audit.get("inputs", {}).get("manifest_payload_sha256")
        != source_manifest_payload
        or source_audit_file != inputs.get("source_audit_file_sha256")
        or source_audit_payload != inputs.get("source_audit_payload_sha256")
        or exclusion.get("format_version") != EXCLUSION_FORMAT
        or exclusion.get("claim_boundary", {}).get(
            "near_identity_exclusion_list_complete"
        )
        is not True
        or exclusion_file != inputs.get("exclusion_receipt_file_sha256")
        or exclusion_payload != inputs.get("exclusion_receipt_payload_sha256")
        or candidate_manifest.get("format_version") != MANIFEST_FORMAT
        or candidate_manifest.get("bucket_count") != bucket_count
        or candidate_manifest.get("authorization") != AUTHORIZATION
        or candidate_manifest.get("test_access") != TEST_ACCESS
        or candidate_manifest.get("protocol_file_sha256") != protocol_file
        or candidate_manifest.get("source_sha256") != sources
        or candidate_manifest.get("map_identity") != source_manifest.get("map_identity")
        or finalize.get("format_version") != FINALIZE_FORMAT
        or finalize.get("manifest_file_sha256") != candidate_file
        or finalize.get("manifest_payload_sha256") != candidate_payload
        or finalize.get("protocol_file_sha256") != protocol_file
        or finalize.get("protocol_payload_sha256") != protocol_payload
    ):
        raise IndependentDecontaminationAuditError("Evidence graph differs")
    exclusion_output = _mapping(exclusion.get("output"), "Exclusion output")
    exclusion_records = _integer(
        exclusion_output.get("excluded_sequence_sha256_records"),
        "Exclusion records",
    )
    exclusion_path = _regular_file(
        exclusion_root / "excluded_sequence_sha256.bin", "Exclusion digest stream"
    )
    if (
        exclusion_output.get("excluded_sequence_sha256_record_bytes") != 32
        or exclusion_path.stat().st_size != exclusion_records * 32
        or exclusion_path.stat().st_size
        != exclusion_output.get("excluded_sequence_sha256_bytes")
        or _file_sha256(exclusion_path)
        != exclusion_output.get("excluded_sequence_sha256_file_sha256")
    ):
        raise IndependentDecontaminationAuditError("Exclusion stream differs")
    source_buckets = source_manifest.get("buckets")
    candidate_buckets = candidate_manifest.get("buckets")
    if (
        not isinstance(source_buckets, list)
        or not isinstance(candidate_buckets, list)
        or len(source_buckets) != bucket_count
        or len(candidate_buckets) != bucket_count
        or [item.get("bucket") for item in source_buckets] != list(range(bucket_count))
        or [item.get("bucket") for item in candidate_buckets]
        != list(range(bucket_count))
    ):
        raise IndependentDecontaminationAuditError("Bucket inventory differs")
    results = []
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        futures = {
            pool.submit(
                _audit_bucket,
                bucket,
                source_root=source_root,
                source_bucket=_mapping(source_buckets[bucket], "Source bucket"),
                candidate_root=output_root,
                candidate_bucket=_mapping(candidate_buckets[bucket], "Candidate bucket"),
                exclusion_path=exclusion_path,
                exclusion_record_count=exclusion_records,
                chunk_records=chunk_records,
            ): bucket
            for bucket in range(bucket_count)
        }
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda item: int(item["bucket"]))
    totals = {
        "decontamination_excluded_train_records": sum(
            int(item["shards"]["train"]["excluded_records"]) for item in results
        ),
        "decontamination_excluded_validation_records": sum(
            int(item["shards"]["validation"]["excluded_records"])
            for item in results
        ),
        "decontamination_source_train_records": sum(
            int(item["shards"]["train"]["source_records"]) for item in results
        ),
        "decontamination_source_validation_records": sum(
            int(item["shards"]["validation"]["source_records"])
            for item in results
        ),
        "unique_train_records": sum(
            int(item["shards"]["train"]["retained_records"]) for item in results
        ),
        "unique_validation_records": sum(
            int(item["shards"]["validation"]["retained_records"])
            for item in results
        ),
    }
    totals["decontamination_excluded_records"] = (
        totals["decontamination_excluded_train_records"]
        + totals["decontamination_excluded_validation_records"]
    )
    totals["decontamination_source_records"] = (
        totals["decontamination_source_train_records"]
        + totals["decontamination_source_validation_records"]
    )
    totals["unique_records"] = (
        totals["unique_train_records"] + totals["unique_validation_records"]
    )
    if candidate_manifest.get("counts") != totals or finalize.get("counts") != totals:
        raise IndependentDecontaminationAuditError("Global replay counts differ")
    return {
        "authorization": AUDIT_AUTHORIZATION,
        "buckets": results,
        "claim_boundary": {
            "all_candidate_array_rows_replayed": True,
            "all_source_rows_accounted": True,
            "independent_replay_complete": True,
            "large_scale_pretraining_authorized": False,
            "zero_excluded_digest_overlap_observed": True,
        },
        "counts": totals,
        "format_version": AUDIT_FORMAT,
        "inputs": {
            "auditor_sha256": actual_auditor_sha256,
            "exclusion_digest_file_sha256": exclusion_output[
                "excluded_sequence_sha256_file_sha256"
            ],
            "exclusion_receipt_file_sha256": exclusion_file,
            "exclusion_receipt_payload_sha256": exclusion_payload,
            "manifest_file_sha256": candidate_file,
            "manifest_payload_sha256": candidate_payload,
            "protocol_file_sha256": protocol_file,
            "protocol_payload_sha256": protocol_payload,
            "source_audit_file_sha256": source_audit_file,
            "source_audit_payload_sha256": source_audit_payload,
            "source_manifest_file_sha256": source_manifest_file,
            "source_manifest_payload_sha256": source_manifest_payload,
        },
        "resource_receipt": _resource_receipt(),
        "test_access": TEST_ACCESS,
        "worker_count": worker_count,
    }


def _publish(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise IndependentDecontaminationAuditError("Audit output already exists")
    payload = dict(value)
    payload["receipt_sha256"] = hashlib.sha256(
        _canonical_json_bytes(payload)
    ).hexdigest()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_json_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--exclusion-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--expected-protocol-file-sha256", required=True)
    parser.add_argument("--auditor-source-sha256", required=True)
    parser.add_argument("--worker-count", type=int, default=8)
    parser.add_argument("--chunk-records", type=int, default=131_072)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    receipt = audit_corpus(
        source_root=arguments.source_root,
        exclusion_root=arguments.exclusion_root,
        output_root=arguments.output_root,
        protocol_path=arguments.protocol,
        expected_protocol_file_sha256=arguments.expected_protocol_file_sha256,
        auditor_source_sha256=arguments.auditor_source_sha256,
        worker_count=arguments.worker_count,
        chunk_records=arguments.chunk_records,
    )
    output = arguments.output_root / "independent_audit_receipt.json"
    _publish(output, receipt)
    published, _, _ = _load_signed(
        output,
        digest_field="receipt_sha256",
        label="Independent decontamination audit",
    )
    sys.stdout.buffer.write(_canonical_json_bytes(published))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
