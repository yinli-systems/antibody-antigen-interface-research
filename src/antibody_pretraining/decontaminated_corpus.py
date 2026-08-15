"""Deterministically materialize digest-excluded OAS corpus shards.

The source corpus is already globally deduplicated and sorted by sequence
SHA-256 inside each first-byte bucket.  Removing a signed set of full-sequence
digests after deduplication is therefore exactly equivalent to removing every
matching raw row before deduplication, while avoiding a second interpretation
of the raw OAS CSV data.

This module only produces authenticated, immutable candidate shards.  It does
not authorize pretraining and it does not claim that the heuristic MMseqs2
track is exhaustive.  A separate, independently implemented replay remains
required before a filtered corpus can enter a training registration.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
from typing import Any, Mapping, Optional, Tuple
import uuid

import numpy as np

from .near_identity_exclusion import EXCLUSION_DIGEST_RECORD
from .production_corpus import EXPECTED_ARRAYS


PROTOCOL_FORMAT = "oas_decontaminated_corpus_protocol_v1"
SOURCE_MANIFEST_FORMAT = "oas_production_corpus_manifest_v1"
SOURCE_AUDIT_FORMAT = "oas_production_corpus_independent_audit_v1"
SOURCE_SHARD_FORMAT = "oas_production_dense_bucket_shard_v1"
EXCLUSION_FORMAT = "oas_near_identity_exclusion_compilation_v1"
TASK_FORMAT = "oas_decontaminated_corpus_task_v1"
FINALIZE_FORMAT = "oas_decontaminated_corpus_finalize_v1"
PROTOCOL_SOURCE_KEYS = {
    "filter_controller",
    "filter_core",
    "finalizer",
    "independent_auditor",
}
TEST_ACCESS = {
    "downstream_metrics_used": False,
    "sabdab2_heldout_accessed": False,
}
SPLITS = ("train", "validation")


class DecontaminatedCorpusError(RuntimeError):
    """Report an unauthenticated input or inconsistent filtered shard."""


def canonical_json_bytes(value: object) -> bytes:
    """Encode one finite canonical ASCII JSON object."""
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise DecontaminatedCorpusError("Payload is not canonical JSON") from error
    return (encoded + "\n").encode("ascii")


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output = {}  # type: dict[str, Any]
    for key, value in pairs:
        if key in output:
            raise DecontaminatedCorpusError("Duplicate JSON key: " + key)
        output[key] = value
    return output


def _reject_constant(value: str) -> None:
    raise DecontaminatedCorpusError("Non-finite JSON constant: " + value)


def require_sha256(value: object, label: str) -> str:
    """Validate one lowercase SHA-256 string."""
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise DecontaminatedCorpusError(label + " is not a lowercase SHA-256")
    return value


def _nonnegative_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DecontaminatedCorpusError(label + " is not a non-negative integer")
    return value


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DecontaminatedCorpusError(label + " is not an object")
    return value


def _protocol_source_sha256(protocol: Mapping[str, Any]) -> dict[str, str]:
    sources = _mapping(protocol.get("source_sha256"), "Protocol source inventory")
    if set(sources) != PROTOCOL_SOURCE_KEYS:
        raise DecontaminatedCorpusError("Protocol source inventory differs")
    return {
        key: require_sha256(value, "Protocol source " + key)
        for key, value in sources.items()
    }


def _regular_file(path: Path, label: str, *, frozen: bool = True) -> Path:
    candidate = Path(path).expanduser().absolute()
    try:
        metadata = candidate.lstat()
    except OSError as error:
        raise DecontaminatedCorpusError("Cannot stat " + label) from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise DecontaminatedCorpusError(label + " is absent, symbolic, or non-regular")
    if frozen and metadata.st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise DecontaminatedCorpusError(label + " remains writable")
    return candidate


def file_sha256(path: Path) -> str:
    """Hash one regular file without following a symbolic link."""
    candidate = _regular_file(path, "Evidence file", frozen=False)
    digest = hashlib.sha256()
    with candidate.open("rb", buffering=16 * 1024 * 1024) as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_signed_json(
    path: Path,
    *,
    digest_field: str,
    label: str,
    expected_file_sha256: Optional[str] = None,
    expected_payload_sha256: Optional[str] = None,
) -> Tuple[dict[str, Any], str, str]:
    """Load, canonicalize, and authenticate one frozen JSON receipt."""
    candidate = _regular_file(path, label)
    raw = candidate.read_bytes()
    try:
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_reject_duplicates,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DecontaminatedCorpusError(label + " is not strict ASCII JSON") from error
    if not isinstance(value, dict) or raw != canonical_json_bytes(value):
        raise DecontaminatedCorpusError(label + " is not canonical JSON")
    unsigned = dict(value)
    claimed = require_sha256(unsigned.pop(digest_field, None), label + " self hash")
    observed = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    file_digest = hashlib.sha256(raw).hexdigest()
    if claimed != observed:
        raise DecontaminatedCorpusError(label + " self hash differs")
    if (
        expected_file_sha256 is not None
        and file_digest != require_sha256(expected_file_sha256, label + " expected file hash")
    ):
        raise DecontaminatedCorpusError(label + " file hash differs")
    if (
        expected_payload_sha256 is not None
        and claimed
        != require_sha256(expected_payload_sha256, label + " expected payload hash")
    ):
        raise DecontaminatedCorpusError(label + " payload hash differs")
    return value, file_digest, claimed


def publish_signed_json(path: Path, value: Mapping[str, Any], *, digest_field: str) -> None:
    """Create, fsync, and freeze one self-signed canonical JSON receipt."""
    if digest_field in value:
        raise DecontaminatedCorpusError("Unsigned JSON already contains its digest")
    payload = dict(value)
    payload[digest_field] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    raw = canonical_json_bytes(payload)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _safe_source_root(source_root: Path, relative_root: object) -> Path:
    if not isinstance(relative_root, str) or not relative_root:
        raise DecontaminatedCorpusError("Source shard root is invalid")
    relative = PurePosixPath(relative_root)
    if relative.is_absolute() or ".." in relative.parts or "." in relative.parts:
        raise DecontaminatedCorpusError("Source shard root is unsafe")
    base = source_root.resolve(strict=True)
    current = base
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise DecontaminatedCorpusError("Source shard path contains a symlink")
    resolved = current.resolve(strict=True)
    if not resolved.is_relative_to(base) or not resolved.is_dir():
        raise DecontaminatedCorpusError("Source shard root escapes the corpus")
    return resolved


def _assert_frozen_shard(root: Path) -> None:
    if root.is_symlink() or not root.is_dir():
        raise DecontaminatedCorpusError("Source shard is absent or symbolic")
    for path in (root, *root.iterdir()):
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode) or metadata.st_mode & (
            stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
        ):
            raise DecontaminatedCorpusError("Source shard remains mutable")


def _bucket_bounds(path: Path, *, records: int, bucket: int) -> tuple[int, int]:
    """Locate a first-byte digest bucket using O(log n) fixed-record seeks."""
    if not 0 <= bucket < 256:
        raise DecontaminatedCorpusError("Digest bucket is outside uint8")

    def lower_bound(first_byte: int) -> int:
        low = 0
        high = records
        with path.open("rb", buffering=0) as handle:
            while low < high:
                middle = (low + high) // 2
                handle.seek(middle * EXCLUSION_DIGEST_RECORD.size)
                observed = handle.read(1)
                if len(observed) != 1:
                    raise DecontaminatedCorpusError("Exclusion digest stream is truncated")
                if observed[0] < first_byte:
                    low = middle + 1
                else:
                    high = middle
        return low

    start = lower_bound(bucket)
    end = records if bucket == 255 else lower_bound(bucket + 1)
    if not 0 <= start <= end <= records:
        raise DecontaminatedCorpusError("Exclusion bucket bounds differ")
    return start, end


def _digest_keys(array: np.ndarray, *, label: str) -> np.ndarray:
    if array.dtype != np.dtype("uint8") or array.ndim != 2 or array.shape[1] != 32:
        raise DecontaminatedCorpusError(label + " digest array schema differs")
    contiguous = np.ascontiguousarray(array)
    return contiguous.reshape(-1, 32).view("|S32").reshape(-1)


def _load_exclusion_bucket(
    path: Path, *, record_count: int, bucket: int
) -> tuple[np.ndarray, np.ndarray]:
    start, end = _bucket_bounds(path, records=record_count, bucket=bucket)
    if start == end:
        empty = np.empty((0, 32), dtype=np.uint8)
        return empty, empty.view("|S32").reshape(-1)
    raw = np.memmap(
        path,
        dtype=np.uint8,
        mode="r",
        offset=start * EXCLUSION_DIGEST_RECORD.size,
        shape=(end - start, 32),
    )
    keys = raw.reshape(-1, 32).view("|S32").reshape(-1)
    if keys.size and (
        np.any(raw[:, 0] != bucket)
        or (keys.size > 1 and np.any(keys[1:] <= keys[:-1]))
    ):
        raise DecontaminatedCorpusError("Exclusion bucket is not strictly sorted")
    return raw, keys


def _source_shard(
    *,
    source_root: Path,
    reference: Mapping[str, Any],
    bucket: int,
    split: str,
) -> tuple[Path, Mapping[str, Any]]:
    root = _safe_source_root(source_root, reference.get("relative_root"))
    _assert_frozen_shard(root)
    manifest, file_digest, payload_digest = load_signed_json(
        root / "manifest.json",
        digest_field="manifest_sha256",
        label=f"Source shard {bucket} {split}",
        expected_file_sha256=str(reference.get("manifest_file_sha256")),
        expected_payload_sha256=str(reference.get("manifest_payload_sha256")),
    )
    if (
        manifest.get("format_version") != SOURCE_SHARD_FORMAT
        or manifest.get("bucket") != bucket
        or manifest.get("split") != split
        or manifest.get("test_access") != TEST_ACCESS
        or manifest.get("record_count") != reference.get("record_count")
        or file_digest != reference.get("manifest_file_sha256")
        or payload_digest != reference.get("manifest_payload_sha256")
    ):
        raise DecontaminatedCorpusError("Source shard binding differs")
    files = _mapping(manifest.get("files"), "Source shard files")
    if set(files) != EXPECTED_ARRAYS:
        raise DecontaminatedCorpusError("Source shard file inventory differs")
    return root, manifest


def _load_source_arrays(
    root: Path, manifest: Mapping[str, Any]
) -> dict[str, np.ndarray]:
    record_count = _nonnegative_integer(manifest.get("record_count"), "Source records")
    files = _mapping(manifest.get("files"), "Source shard files")
    arrays = {}  # type: dict[str, np.ndarray]
    try:
        for name in sorted(EXPECTED_ARRAYS):
            reference = _mapping(files.get(name), "Source array reference")
            path = _regular_file(root / name, "Source array")
            if (
                path.stat().st_size != reference.get("bytes")
                or file_sha256(path) != reference.get("sha256")
            ):
                raise DecontaminatedCorpusError("Source array inventory differs: " + name)
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if (
                str(array.dtype) != reference.get("dtype")
                or list(array.shape) != reference.get("shape")
                or array.shape[0] != record_count
                or array.flags.writeable
            ):
                raise DecontaminatedCorpusError("Source array schema differs: " + name)
            arrays[name] = array
        return arrays
    except BaseException:
        for array in arrays.values():
            mapping = getattr(array, "_mmap", None)
            if mapping is not None:
                mapping.close()
        raise


def _close_arrays(arrays: Mapping[str, np.ndarray]) -> None:
    for array in arrays.values():
        mapping = getattr(array, "_mmap", None)
        if mapping is not None:
            mapping.close()


def _write_filtered_array(
    *, source: np.ndarray, keep: np.ndarray, path: Path, chunk_records: int
) -> Mapping[str, Any]:
    retained = int(np.count_nonzero(keep))
    shape = (retained, *source.shape[1:])
    output = np.lib.format.open_memmap(path, mode="w+", dtype=source.dtype, shape=shape)
    written = 0
    try:
        for start in range(0, len(keep), chunk_records):
            end = min(start + chunk_records, len(keep))
            selected = source[start:end][keep[start:end]]
            output[written : written + len(selected)] = selected
            written += len(selected)
        if written != retained:
            raise DecontaminatedCorpusError("Filtered array write count differs")
        output.flush()
    finally:
        mapping = getattr(output, "_mmap", None)
        if mapping is not None:
            mapping.close()
    with path.open("rb") as handle:
        os.fsync(handle.fileno())
    return {
        "bytes": path.stat().st_size,
        "dtype": str(source.dtype),
        "sha256": file_sha256(path),
        "shape": list(shape),
    }


def _sum_sequence_lengths(
    lengths: np.ndarray,
    *,
    selected: Optional[np.ndarray] = None,
    chunk_records: int = 131_072,
) -> int:
    """Sum authenticated residue lengths without a corpus-sized temporary copy."""
    if lengths.ndim != 1 or chunk_records <= 0:
        raise DecontaminatedCorpusError("Sequence-length accounting schema differs")
    if selected is not None and (
        selected.dtype != np.dtype("bool")
        or selected.ndim != 1
        or len(selected) != len(lengths)
    ):
        raise DecontaminatedCorpusError("Sequence-length selector differs")
    total = 0
    for start in range(0, len(lengths), chunk_records):
        end = min(start + chunk_records, len(lengths))
        block = np.asarray(lengths[start:end])
        if selected is not None:
            block = block[selected[start:end]]
        total += int(np.sum(block, dtype=np.uint64))
    return total


def filter_shard(
    *,
    source_root: Path,
    source_reference: Mapping[str, Any],
    output_root: Path,
    exclusion_path: Path,
    exclusion_record_count: int,
    bucket: int,
    split: str,
    chunk_records: int = 131_072,
) -> Mapping[str, Any]:
    """Filter one authenticated source shard and publish one shard manifest."""
    if split not in SPLITS or chunk_records <= 0:
        raise DecontaminatedCorpusError("Filtered shard parameters differ")
    if output_root.exists() or output_root.is_symlink():
        raise DecontaminatedCorpusError("Refusing to overwrite filtered shard")
    source_shard_root, source_manifest = _source_shard(
        source_root=source_root,
        reference=source_reference,
        bucket=bucket,
        split=split,
    )
    arrays = _load_source_arrays(source_shard_root, source_manifest)
    exclusion_raw, exclusion_keys = _load_exclusion_bucket(
        exclusion_path,
        record_count=exclusion_record_count,
        bucket=bucket,
    )
    partial = output_root.parent / (output_root.name + ".partial-" + uuid.uuid4().hex)
    partial.mkdir(parents=False, mode=0o700)
    try:
        digest_array = arrays["sequence_sha256.npy"]
        source_keys = _digest_keys(digest_array, label="Source")
        if source_keys.size and (
            np.any(digest_array[:, 0] != bucket)
            or (source_keys.size > 1 and np.any(source_keys[1:] <= source_keys[:-1]))
        ):
            raise DecontaminatedCorpusError("Source digest order differs")
        if exclusion_keys.size:
            positions = np.searchsorted(exclusion_keys, source_keys)
            bounded = np.minimum(positions, exclusion_keys.size - 1)
            excluded = (positions < exclusion_keys.size) & (
                exclusion_keys[bounded] == source_keys
            )
        else:
            excluded = np.zeros(len(source_keys), dtype=np.bool_)
        keep = ~excluded
        retained = int(np.count_nonzero(keep))
        source_residue_tokens = _sum_sequence_lengths(
            arrays["sequence_lengths.npy"], chunk_records=chunk_records
        )
        excluded_residue_tokens = _sum_sequence_lengths(
            arrays["sequence_lengths.npy"],
            selected=excluded,
            chunk_records=chunk_records,
        )
        retained_residue_tokens = source_residue_tokens - excluded_residue_tokens
        files = {}
        for name in sorted(EXPECTED_ARRAYS):
            files[name] = _write_filtered_array(
                source=arrays[name],
                keep=keep,
                path=partial / name,
                chunk_records=chunk_records,
            )
        manifest = {
            "bucket": bucket,
            "decontamination": {
                "excluded_records": int(np.count_nonzero(excluded)),
                "excluded_residue_tokens": excluded_residue_tokens,
                "exclusion_rule": "full_sequence_sha256_membership",
                "source_record_count": len(source_keys),
                "source_residue_tokens": source_residue_tokens,
            },
            "files": files,
            "format_version": SOURCE_SHARD_FORMAT,
            "global_order": "sequence_sha256_ascending",
            "maximum_sequence_tokens": int(arrays["tokens.npy"].shape[1]),
            "record_count": retained,
            "residue_token_count": retained_residue_tokens,
            "source_manifest_file_sha256": str(
                source_reference.get("manifest_file_sha256")
            ),
            "source_manifest_payload_sha256": str(
                source_reference.get("manifest_payload_sha256")
            ),
            "split": split,
            "test_access": TEST_ACCESS,
        }
        publish_signed_json(partial / "manifest.json", manifest, digest_field="manifest_sha256")
        for path in partial.iterdir():
            path.chmod(0o400)
        partial.chmod(0o500)
        os.replace(partial, output_root)
        published, file_digest, payload_digest = load_signed_json(
            output_root / "manifest.json",
            digest_field="manifest_sha256",
            label="Filtered shard manifest",
        )
        return {
            "excluded_records": int(np.count_nonzero(excluded)),
            "excluded_residue_tokens": excluded_residue_tokens,
            "manifest_file_sha256": file_digest,
            "manifest_payload_sha256": payload_digest,
            "record_count": retained,
            "residue_token_count": retained_residue_tokens,
            "relative_root": "filtered/" + output_root.parent.name + "/" + split,
            "source_manifest_file_sha256": str(
                source_reference.get("manifest_file_sha256")
            ),
            "source_manifest_payload_sha256": str(
                source_reference.get("manifest_payload_sha256")
            ),
            "source_record_count": len(source_keys),
            "source_residue_tokens": source_residue_tokens,
            "verified_manifest": published,
        }
    except BaseException:
        if partial.exists():
            partial.chmod(0o700)
            for path in partial.iterdir():
                path.chmod(path.lstat().st_mode | stat.S_IWUSR)
            shutil.rmtree(partial)
        raise
    finally:
        _close_arrays(arrays)
        mapping = getattr(exclusion_raw, "_mmap", None)
        if mapping is not None:
            mapping.close()


def _resource_receipt() -> Mapping[str, Any]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible not in {"", "NoDevFiles"}:
        raise DecontaminatedCorpusError("Filtering must not expose a GPU")
    assigned = os.environ.get("SLURM_JOB_GPUS") or os.environ.get("SLURM_STEP_GPUS", "")
    assigned_count = len([item for item in assigned.split(",") if item])
    declared = os.environ.get("OAS_SCHEDULER_RESERVED_GPU_COUNT")
    if declared == "1" and assigned_count == 1:
        mode = "scheduler_mandated_idle_only"
    elif declared is None and assigned_count == 0:
        mode = "none"
    else:
        raise DecontaminatedCorpusError("Filtering scheduler evidence differs")
    return {
        "cuda_visible_devices": False,
        "gpu_compute_permitted": False,
        "scheduler_reserved_gpu_count": assigned_count,
        "scheduler_reserved_gpu_mode": mode,
    }


def filter_task(
    *,
    task_id: int,
    source_root: Path,
    exclusion_root: Path,
    output_root: Path,
    protocol_path: Path,
    expected_protocol_file_sha256: str,
    controller_source_sha256: str,
    task_count: int = 32,
    bucket_count: int = 256,
) -> Mapping[str, Any]:
    """Filter the deterministic modulo bucket slice assigned to one task."""
    if (
        isinstance(task_id, bool)
        or not 0 <= task_id < task_count
        or isinstance(task_count, bool)
        or not 1 <= task_count <= 256
        or isinstance(bucket_count, bool)
        or not task_count <= bucket_count <= 256
    ):
        raise DecontaminatedCorpusError("Filter task topology differs")
    controller_source_sha256 = require_sha256(
        controller_source_sha256, "Filter controller source"
    )
    protocol_path = _regular_file(protocol_path, "Decontamination protocol")
    protocol, protocol_file_digest, protocol_payload_digest = load_signed_json(
        protocol_path,
        digest_field="protocol_sha256",
        label="Decontamination protocol",
        expected_file_sha256=expected_protocol_file_sha256,
    )
    inputs = _mapping(protocol.get("inputs"), "Protocol inputs")
    execution = _mapping(protocol.get("execution"), "Protocol execution")
    policy = _mapping(protocol.get("policy"), "Protocol policy")
    protocol_sources = _protocol_source_sha256(protocol)
    if (
        protocol.get("format_version") != PROTOCOL_FORMAT
        or protocol.get("test_access") != TEST_ACCESS
        or execution.get("task_count") != task_count
        or execution.get("bucket_count") != bucket_count
        or policy.get("excluded_splits") != list(SPLITS)
        or policy.get("exclusion_key") != "full_sequence_sha256"
        or protocol.get("authorization")
        != {
            "large_scale_pretraining": False,
            "production_corpus_use": False,
            "scientific_claim": False,
        }
    ):
        raise DecontaminatedCorpusError("Decontamination protocol scope differs")
    if (
        protocol_sources["filter_controller"] != controller_source_sha256
        or protocol_sources["filter_core"] != file_sha256(Path(__file__))
    ):
        raise DecontaminatedCorpusError("Decontamination source binding differs")
    source_root = Path(source_root).expanduser().absolute()
    exclusion_root = Path(exclusion_root).expanduser().absolute()
    output_root = Path(output_root).expanduser().absolute()
    for root, label in ((source_root, "source"), (exclusion_root, "exclusion")):
        if root.is_symlink() or not root.is_dir():
            raise DecontaminatedCorpusError(label + " root is absent or symbolic")
    source_manifest, source_manifest_file, source_manifest_payload = load_signed_json(
        source_root / "corpus_manifest.json",
        digest_field="manifest_sha256",
        label="Source corpus manifest",
        expected_file_sha256=str(inputs.get("source_manifest_file_sha256")),
        expected_payload_sha256=str(inputs.get("source_manifest_payload_sha256")),
    )
    source_audit, source_audit_file, source_audit_payload = load_signed_json(
        source_root / "independent_audit_receipt.json",
        digest_field="receipt_sha256",
        label="Source corpus independent audit",
        expected_file_sha256=str(inputs.get("source_audit_file_sha256")),
        expected_payload_sha256=str(inputs.get("source_audit_payload_sha256")),
    )
    exclusion, exclusion_file, exclusion_payload = load_signed_json(
        exclusion_root / "receipt.json",
        digest_field="receipt_sha256",
        label="Near-identity exclusion receipt",
        expected_file_sha256=str(inputs.get("exclusion_receipt_file_sha256")),
        expected_payload_sha256=str(inputs.get("exclusion_receipt_payload_sha256")),
    )
    if (
        source_manifest.get("format_version") != SOURCE_MANIFEST_FORMAT
        or source_manifest.get("bucket_count") != bucket_count
        or source_manifest.get("test_access") != TEST_ACCESS
        or protocol.get("map_identity") != source_manifest.get("map_identity")
        or source_manifest.get("authorization")
        != {
            "large_scale_pretraining": False,
            "production_corpus_use": False,
            "scientific_claim": False,
        }
        or source_audit.get("format_version") != SOURCE_AUDIT_FORMAT
        or source_audit.get("test_access") != TEST_ACCESS
        or source_audit.get("authorization")
        != {
            "independent_replay": True,
            "large_scale_pretraining": False,
            "production_corpus_use": False,
            "scientific_claim": False,
        }
        or source_audit.get("inputs", {}).get("manifest_file_sha256")
        != source_manifest_file
        or source_audit.get("inputs", {}).get("manifest_payload_sha256")
        != source_manifest_payload
        or exclusion.get("format_version") != EXCLUSION_FORMAT
        or exclusion.get("test_access") != TEST_ACCESS
        or exclusion.get("claim_boundary", {}).get(
            "near_identity_exclusion_list_complete"
        )
        is not True
        or exclusion.get("claim_boundary", {}).get("original_training_corpus_filtered")
        is not False
    ):
        raise DecontaminatedCorpusError("Source or exclusion evidence scope differs")
    exclusion_output = _mapping(exclusion.get("output"), "Exclusion output")
    exclusion_records = _nonnegative_integer(
        exclusion_output.get("excluded_sequence_sha256_records"),
        "Exclusion records",
    )
    exclusion_path = _regular_file(
        exclusion_root / "excluded_sequence_sha256.bin",
        "Exclusion digest stream",
    )
    if (
        exclusion_output.get("excluded_sequence_sha256_record_bytes")
        != EXCLUSION_DIGEST_RECORD.size
        or exclusion_path.stat().st_size != exclusion_records * EXCLUSION_DIGEST_RECORD.size
        or exclusion_path.stat().st_size
        != exclusion_output.get("excluded_sequence_sha256_bytes")
        or file_sha256(exclusion_path)
        != exclusion_output.get("excluded_sequence_sha256_file_sha256")
        or exclusion_file != inputs.get("exclusion_receipt_file_sha256")
        or exclusion_payload != inputs.get("exclusion_receipt_payload_sha256")
    ):
        raise DecontaminatedCorpusError("Exclusion digest inventory differs")
    final_root = output_root / "tasks" / f"task-{task_id:04d}"
    if final_root.exists() or final_root.is_symlink():
        raise DecontaminatedCorpusError("Refusing to overwrite filter task")
    final_root.parent.mkdir(parents=True, exist_ok=True)
    partial = final_root.parent / (final_root.name + ".partial-" + uuid.uuid4().hex)
    partial.mkdir(mode=0o700)
    buckets = source_manifest.get("buckets")
    if not isinstance(buckets, list) or len(buckets) != bucket_count:
        raise DecontaminatedCorpusError("Source bucket inventory differs")
    summaries = []
    totals = Counter()
    try:
        for bucket in range(task_id, bucket_count, task_count):
            item = _mapping(buckets[bucket], "Source bucket")
            if item.get("bucket") != bucket:
                raise DecontaminatedCorpusError("Source bucket ordering differs")
            references = _mapping(item.get("shards"), "Source bucket shards")
            bucket_root = partial / f"bucket-{bucket:03d}"
            bucket_root.mkdir(mode=0o700)
            split_summaries = {}
            for split in SPLITS:
                result = filter_shard(
                    source_root=source_root,
                    source_reference=_mapping(references.get(split), "Source shard"),
                    output_root=bucket_root / split,
                    exclusion_path=exclusion_path,
                    exclusion_record_count=exclusion_records,
                    bucket=bucket,
                    split=split,
                )
                verified = dict(result)
                verified.pop("verified_manifest", None)
                verified["relative_root"] = (
                    f"tasks/task-{task_id:04d}/bucket-{bucket:03d}/{split}"
                )
                split_summaries[split] = verified
                totals["source_" + split] += result["source_record_count"]
                totals["excluded_" + split] += result["excluded_records"]
                totals["retained_" + split] += result["record_count"]
                totals["source_" + split + "_residue_tokens"] += result[
                    "source_residue_tokens"
                ]
                totals["excluded_" + split + "_residue_tokens"] += result[
                    "excluded_residue_tokens"
                ]
                totals["retained_" + split + "_residue_tokens"] += result[
                    "residue_token_count"
                ]
            summaries.append({"bucket": bucket, "shards": split_summaries})
        receipt = {
            "authorization": {
                "large_scale_pretraining": False,
                "production_corpus_use": False,
                "scientific_claim": False,
            },
            "buckets": summaries,
            "claim_boundary": {
                "assigned_bucket_slice_complete": True,
                "independent_full_replay_complete": False,
                "large_scale_pretraining_authorized": False,
                "zero_excluded_digest_overlap_observed": True,
            },
            "counts": dict(sorted(totals.items())),
            "format_version": TASK_FORMAT,
            "input": {
                "exclusion_digest_file_sha256": exclusion_output[
                    "excluded_sequence_sha256_file_sha256"
                ],
                "exclusion_receipt_file_sha256": exclusion_file,
                "exclusion_receipt_payload_sha256": exclusion_payload,
                "source_audit_file_sha256": source_audit_file,
                "source_audit_payload_sha256": source_audit_payload,
                "source_manifest_file_sha256": source_manifest_file,
                "source_manifest_payload_sha256": source_manifest_payload,
            },
            "protocol_file_sha256": protocol_file_digest,
            "protocol_payload_sha256": protocol_payload_digest,
            "resource_receipt": _resource_receipt(),
            "source_sha256": protocol_sources,
            "task_count": task_count,
            "task_id": task_id,
            "test_access": TEST_ACCESS,
        }
        publish_signed_json(partial / "receipt.json", receipt, digest_field="receipt_sha256")
        for path in partial.rglob("*"):
            if path.is_file():
                path.chmod(0o400)
        for path in sorted(partial.rglob("*"), reverse=True):
            if path.is_dir():
                path.chmod(0o500)
        partial.chmod(0o500)
        os.replace(partial, final_root)
        return load_signed_json(
            final_root / "receipt.json",
            digest_field="receipt_sha256",
            label="Filtered task receipt",
        )[0]
    except BaseException:
        if partial.exists():
            for path in (partial, *partial.rglob("*")):
                try:
                    mode = path.lstat().st_mode
                    if stat.S_ISDIR(mode):
                        path.chmod(mode | stat.S_IWUSR | stat.S_IXUSR)
                    elif stat.S_ISREG(mode):
                        path.chmod(mode | stat.S_IWUSR)
                except FileNotFoundError:
                    continue
            shutil.rmtree(partial)
        raise


def _assert_frozen_tree(root: Path, label: str) -> None:
    if root.is_symlink() or not root.is_dir():
        raise DecontaminatedCorpusError(label + " is absent or symbolic")
    for path in (root, *root.rglob("*")):
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode) or metadata.st_mode & (
            stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
        ):
            raise DecontaminatedCorpusError(label + " remains mutable")


def _filtered_shard_reference(
    *,
    output_root: Path,
    task_id: int,
    bucket: int,
    split: str,
    summary: Mapping[str, Any],
) -> tuple[dict[str, Any], Path]:
    relative = f"tasks/task-{task_id:04d}/bucket-{bucket:03d}/{split}"
    if summary.get("relative_root") != relative:
        raise DecontaminatedCorpusError("Filtered shard relative root differs")
    root = output_root / relative
    _assert_frozen_tree(root, "Filtered shard")
    manifest, file_digest, payload_digest = load_signed_json(
        root / "manifest.json",
        digest_field="manifest_sha256",
        label="Filtered shard manifest",
        expected_file_sha256=str(summary.get("manifest_file_sha256")),
        expected_payload_sha256=str(summary.get("manifest_payload_sha256")),
    )
    if (
        manifest.get("format_version") != SOURCE_SHARD_FORMAT
        or manifest.get("bucket") != bucket
        or manifest.get("split") != split
        or manifest.get("test_access") != TEST_ACCESS
        or manifest.get("record_count") != summary.get("record_count")
        or manifest.get("source_manifest_file_sha256")
        != summary.get("source_manifest_file_sha256")
        or manifest.get("source_manifest_payload_sha256")
        != summary.get("source_manifest_payload_sha256")
    ):
        raise DecontaminatedCorpusError("Filtered shard manifest binding differs")
    decontamination = _mapping(
        manifest.get("decontamination"), "Filtered shard decontamination"
    )
    source_records = _nonnegative_integer(
        summary.get("source_record_count"), "Filtered source records"
    )
    excluded_records = _nonnegative_integer(
        summary.get("excluded_records"), "Filtered excluded records"
    )
    retained_records = _nonnegative_integer(
        summary.get("record_count"), "Filtered retained records"
    )
    source_residue_tokens = _nonnegative_integer(
        summary.get("source_residue_tokens"), "Filtered source residue tokens"
    )
    excluded_residue_tokens = _nonnegative_integer(
        summary.get("excluded_residue_tokens"), "Filtered excluded residue tokens"
    )
    retained_residue_tokens = _nonnegative_integer(
        summary.get("residue_token_count"), "Filtered retained residue tokens"
    )
    if (
        source_records != excluded_records + retained_records
        or source_residue_tokens
        != excluded_residue_tokens + retained_residue_tokens
        or decontamination.get("source_record_count") != source_records
        or decontamination.get("excluded_records") != excluded_records
        or decontamination.get("source_residue_tokens") != source_residue_tokens
        or decontamination.get("excluded_residue_tokens")
        != excluded_residue_tokens
        or manifest.get("residue_token_count") != retained_residue_tokens
        or decontamination.get("exclusion_rule")
        != "full_sequence_sha256_membership"
    ):
        raise DecontaminatedCorpusError("Filtered shard count accounting differs")
    entries = {path.name for path in root.iterdir()}
    if entries != set(EXPECTED_ARRAYS) | {"manifest.json"}:
        raise DecontaminatedCorpusError("Filtered shard file inventory differs")
    arrays = _load_source_arrays(root, manifest)
    observed_retained_residue_tokens = _sum_sequence_lengths(
        arrays["sequence_lengths.npy"]
    )
    _close_arrays(arrays)
    if observed_retained_residue_tokens != retained_residue_tokens:
        raise DecontaminatedCorpusError("Filtered residue-token total differs")
    return (
        {
            "manifest_file_sha256": file_digest,
            "manifest_payload_sha256": payload_digest,
            "record_count": retained_records,
            "residue_token_count": retained_residue_tokens,
            "relative_root": f"reduce/bucket-{bucket:03d}/{split}",
        },
        root,
    )


def _publish_hardlink_view(
    *,
    output_root: Path,
    shard_roots: Mapping[tuple[int, str], Path],
    bucket_count: int,
) -> None:
    final = output_root / "reduce"
    if final.exists() or final.is_symlink():
        raise DecontaminatedCorpusError("Canonical filtered view already exists")
    partial = output_root / (".reduce.partial-" + uuid.uuid4().hex)
    partial.mkdir(mode=0o700)
    try:
        for bucket in range(bucket_count):
            bucket_root = partial / f"bucket-{bucket:03d}"
            bucket_root.mkdir(mode=0o700)
            for split in SPLITS:
                source = shard_roots[(bucket, split)]
                destination = bucket_root / split
                destination.mkdir(mode=0o700)
                for source_path in sorted(source.iterdir()):
                    metadata = source_path.lstat()
                    if (
                        stat.S_ISLNK(metadata.st_mode)
                        or not stat.S_ISREG(metadata.st_mode)
                        or metadata.st_mode
                        & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
                    ):
                        raise DecontaminatedCorpusError(
                            "Filtered publication source is mutable"
                        )
                    destination_path = destination / source_path.name
                    os.link(source_path, destination_path, follow_symlinks=False)
                    if (
                        destination_path.stat().st_ino != source_path.stat().st_ino
                        or destination_path.stat().st_dev != source_path.stat().st_dev
                    ):
                        raise DecontaminatedCorpusError(
                            "Filtered publication hardlink differs"
                        )
                destination.chmod(0o500)
            bucket_root.chmod(0o500)
        partial.chmod(0o500)
        os.replace(partial, final)
    except BaseException:
        if partial.exists():
            for path in (partial, *partial.rglob("*")):
                try:
                    mode = path.lstat().st_mode
                    if stat.S_ISDIR(mode):
                        path.chmod(mode | stat.S_IWUSR | stat.S_IXUSR)
                    elif stat.S_ISREG(mode):
                        path.chmod(mode | stat.S_IWUSR)
                except FileNotFoundError:
                    continue
            shutil.rmtree(partial)
        raise


def finalize_corpus(
    *,
    output_root: Path,
    protocol_path: Path,
    expected_protocol_file_sha256: str,
    finalizer_source_path: Path,
    finalizer_source_sha256: str,
) -> Mapping[str, Any]:
    """Authenticate every filter task and publish a non-authorizing corpus view."""
    finalizer_source_path = _regular_file(
        finalizer_source_path, "Decontamination finalizer source", frozen=False
    )
    finalizer_source_sha256 = require_sha256(
        finalizer_source_sha256, "Decontamination finalizer source"
    )
    if file_sha256(finalizer_source_path) != finalizer_source_sha256:
        raise DecontaminatedCorpusError("Decontamination finalizer source differs")
    protocol, protocol_file, protocol_payload = load_signed_json(
        protocol_path,
        digest_field="protocol_sha256",
        label="Decontamination protocol",
        expected_file_sha256=expected_protocol_file_sha256,
    )
    execution = _mapping(protocol.get("execution"), "Protocol execution")
    policy = _mapping(protocol.get("policy"), "Protocol policy")
    sources = _protocol_source_sha256(protocol)
    task_count = _nonnegative_integer(execution.get("task_count"), "Filter tasks")
    bucket_count = _nonnegative_integer(execution.get("bucket_count"), "Buckets")
    if (
        protocol.get("format_version") != PROTOCOL_FORMAT
        or protocol.get("test_access") != TEST_ACCESS
        or protocol.get("authorization")
        != {
            "large_scale_pretraining": False,
            "production_corpus_use": False,
            "scientific_claim": False,
        }
        or policy.get("excluded_splits") != list(SPLITS)
        or policy.get("exclusion_key") != "full_sequence_sha256"
        or not 1 <= task_count <= bucket_count <= 256
        or sources["filter_core"] != file_sha256(Path(__file__))
        or sources["finalizer"] != finalizer_source_sha256
    ):
        raise DecontaminatedCorpusError("Decontamination finalization scope differs")
    output_root = Path(output_root).expanduser().absolute()
    if output_root.is_symlink() or not output_root.is_dir():
        raise DecontaminatedCorpusError("Decontaminated corpus root is invalid")
    manifest_path = output_root / "corpus_manifest.json"
    receipt_path = output_root / "corpus_finalize_receipt.json"
    if any(path.exists() or path.is_symlink() for path in (manifest_path, receipt_path)):
        raise DecontaminatedCorpusError("Decontaminated corpus is already finalized")
    task_parent = output_root / "tasks"
    expected_tasks = {f"task-{task_id:04d}" for task_id in range(task_count)}
    if (
        task_parent.is_symlink()
        or not task_parent.is_dir()
        or {path.name for path in task_parent.iterdir()} != expected_tasks
    ):
        raise DecontaminatedCorpusError("Filtered task inventory differs")
    buckets: dict[int, dict[str, Any]] = {}
    shard_roots: dict[tuple[int, str], Path] = {}
    task_receipts = []
    totals = Counter()
    common_input: Optional[dict[str, Any]] = None
    for task_id in range(task_count):
        task_root = task_parent / f"task-{task_id:04d}"
        _assert_frozen_tree(task_root, "Filtered task")
        receipt, file_digest, payload_digest = load_signed_json(
            task_root / "receipt.json",
            digest_field="receipt_sha256",
            label="Filtered task receipt",
        )
        task_buckets = receipt.get("buckets")
        task_input = receipt.get("input")
        resource = receipt.get("resource_receipt")
        if (
            receipt.get("format_version") != TASK_FORMAT
            or receipt.get("task_id") != task_id
            or receipt.get("task_count") != task_count
            or receipt.get("protocol_file_sha256") != protocol_file
            or receipt.get("protocol_payload_sha256") != protocol_payload
            or receipt.get("source_sha256") != sources
            or receipt.get("test_access") != TEST_ACCESS
            or receipt.get("authorization") != protocol.get("authorization")
            or receipt.get("claim_boundary", {}).get(
                "assigned_bucket_slice_complete"
            )
            is not True
            or receipt.get("claim_boundary", {}).get(
                "independent_full_replay_complete"
            )
            is not False
            or not isinstance(task_buckets, list)
            or [item.get("bucket") for item in task_buckets]
            != list(range(task_id, bucket_count, task_count))
            or not isinstance(task_input, Mapping)
            or not isinstance(resource, Mapping)
            or resource.get("cuda_visible_devices") is not False
            or resource.get("gpu_compute_permitted") is not False
            or resource.get("scheduler_reserved_gpu_count") not in {0, 1}
        ):
            raise DecontaminatedCorpusError("Filtered task identity differs")
        current_input = dict(task_input)
        if common_input is None:
            common_input = current_input
        elif current_input != common_input:
            raise DecontaminatedCorpusError("Filtered task inputs disagree")
        observed_counts = Counter()
        for item in task_buckets:
            bucket = _nonnegative_integer(item.get("bucket"), "Filtered bucket")
            if bucket in buckets or not 0 <= bucket < bucket_count:
                raise DecontaminatedCorpusError("Filtered bucket is duplicated")
            summaries = _mapping(item.get("shards"), "Filtered shard summaries")
            if set(summaries) != set(SPLITS):
                raise DecontaminatedCorpusError("Filtered split inventory differs")
            published = {}
            for split in SPLITS:
                summary = _mapping(summaries.get(split), "Filtered shard summary")
                reference, shard_root = _filtered_shard_reference(
                    output_root=output_root,
                    task_id=task_id,
                    bucket=bucket,
                    split=split,
                    summary=summary,
                )
                published[split] = reference
                shard_roots[(bucket, split)] = shard_root
                observed_counts["source_" + split] += int(
                    summary["source_record_count"]
                )
                observed_counts["excluded_" + split] += int(
                    summary["excluded_records"]
                )
                observed_counts["retained_" + split] += int(summary["record_count"])
                observed_counts[
                    "source_" + split + "_residue_tokens"
                ] += int(summary["source_residue_tokens"])
                observed_counts[
                    "excluded_" + split + "_residue_tokens"
                ] += int(summary["excluded_residue_tokens"])
                observed_counts[
                    "retained_" + split + "_residue_tokens"
                ] += int(summary["residue_token_count"])
            buckets[bucket] = {"bucket": bucket, "shards": published}
        if dict(sorted(observed_counts.items())) != receipt.get("counts"):
            raise DecontaminatedCorpusError("Filtered task totals differ")
        totals.update(observed_counts)
        task_receipts.append(
            {
                "file_sha256": file_digest,
                "payload_sha256": payload_digest,
                "relative_path": f"tasks/task-{task_id:04d}/receipt.json",
                "task_id": task_id,
            }
        )
    if set(buckets) != set(range(bucket_count)) or common_input is None:
        raise DecontaminatedCorpusError("Filtered bucket coverage differs")
    protocol_inputs = _mapping(protocol.get("inputs"), "Protocol inputs")
    if any(common_input.get(key) != value for key, value in protocol_inputs.items()):
        raise DecontaminatedCorpusError("Filtered task protocol inputs differ")
    _publish_hardlink_view(
        output_root=output_root,
        shard_roots=shard_roots,
        bucket_count=bucket_count,
    )
    counts = {
        "decontamination_excluded_records": totals["excluded_train"]
        + totals["excluded_validation"],
        "decontamination_excluded_train_records": totals["excluded_train"],
        "decontamination_excluded_validation_records": totals[
            "excluded_validation"
        ],
        "decontamination_excluded_residue_tokens": totals[
            "excluded_train_residue_tokens"
        ]
        + totals["excluded_validation_residue_tokens"],
        "decontamination_excluded_train_residue_tokens": totals[
            "excluded_train_residue_tokens"
        ],
        "decontamination_excluded_validation_residue_tokens": totals[
            "excluded_validation_residue_tokens"
        ],
        "decontamination_source_records": totals["source_train"]
        + totals["source_validation"],
        "decontamination_source_train_records": totals["source_train"],
        "decontamination_source_validation_records": totals["source_validation"],
        "decontamination_source_residue_tokens": totals[
            "source_train_residue_tokens"
        ]
        + totals["source_validation_residue_tokens"],
        "decontamination_source_train_residue_tokens": totals[
            "source_train_residue_tokens"
        ],
        "decontamination_source_validation_residue_tokens": totals[
            "source_validation_residue_tokens"
        ],
        "unique_records": totals["retained_train"]
        + totals["retained_validation"],
        "unique_train_records": totals["retained_train"],
        "unique_validation_records": totals["retained_validation"],
        "unique_residue_tokens": totals["retained_train_residue_tokens"]
        + totals["retained_validation_residue_tokens"],
        "unique_train_residue_tokens": totals["retained_train_residue_tokens"],
        "unique_validation_residue_tokens": totals[
            "retained_validation_residue_tokens"
        ],
    }
    resource = _resource_receipt()
    manifest = {
        "authorization": protocol["authorization"],
        "bucket_count": bucket_count,
        "buckets": [buckets[bucket] for bucket in range(bucket_count)],
        "counts": counts,
        "decontamination": {
            "inputs": common_input,
            "policy": dict(policy),
            "publication_mode": "immutable_same_filesystem_hardlink_view",
            "task_receipts": task_receipts,
        },
        "format_version": SOURCE_MANIFEST_FORMAT,
        "global_order": "bucket_ascending_then_sequence_sha256_ascending",
        "map_identity": protocol.get("map_identity"),
        "protocol_file_sha256": protocol_file,
        "resource_receipt": resource,
        "source_sha256": sources,
        "test_access": TEST_ACCESS,
    }
    publish_signed_json(manifest_path, manifest, digest_field="manifest_sha256")
    published_manifest, manifest_file, manifest_payload = load_signed_json(
        manifest_path,
        digest_field="manifest_sha256",
        label="Decontaminated corpus manifest",
    )
    receipt = {
        "authorization": protocol["authorization"],
        "claim_boundary": {
            "all_filter_tasks_authenticated": True,
            "all_filtered_shards_authenticated": True,
            "independent_full_replay_complete": False,
            "large_scale_pretraining_authorized": False,
            "near_identity_decontamination_candidate_complete": True,
        },
        "counts": counts,
        "format_version": FINALIZE_FORMAT,
        "manifest_file_sha256": manifest_file,
        "manifest_payload_sha256": manifest_payload,
        "protocol_file_sha256": protocol_file,
        "protocol_payload_sha256": protocol_payload,
        "resource_receipt": resource,
        "source_sha256": sources,
        "task_receipts": task_receipts,
        "test_access": TEST_ACCESS,
    }
    publish_signed_json(receipt_path, receipt, digest_field="receipt_sha256")
    if published_manifest.get("counts") != counts:
        raise DecontaminatedCorpusError("Published corpus counts differ")
    return load_signed_json(
        receipt_path,
        digest_field="receipt_sha256",
        label="Decontamination finalization receipt",
    )[0]


__all__ = [
    "DecontaminatedCorpusError",
    "FINALIZE_FORMAT",
    "PROTOCOL_FORMAT",
    "TASK_FORMAT",
    "canonical_json_bytes",
    "file_sha256",
    "finalize_corpus",
    "filter_shard",
    "filter_task",
    "load_signed_json",
    "publish_signed_json",
    "require_sha256",
]
