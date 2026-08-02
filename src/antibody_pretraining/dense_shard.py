"""Create-only fixed-width NumPy shards for efficient antibody MLM input."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .tokenizer import CHAIN_TYPE_TO_ID, PAD_ID, encode_unpaired


SHARD_FORMAT = "antibody_dense_sequence_shard_v1"


class DenseShardError(RuntimeError):
    """Report an invalid or mutable dense sequence shard."""


@dataclass(frozen=True)
class DenseShardRecord:
    """Represent one canonical sequence selected for a dense shard."""

    chain_type: str
    sample_key: int
    sequence: str


@dataclass(frozen=True)
class DenseShard:
    """Expose memory-mapped arrays and their immutable manifest."""

    chain_types: np.ndarray
    lengths: np.ndarray
    manifest: dict[str, Any]
    sample_keys: np.ndarray
    tokens: np.ndarray

    def __len__(self) -> int:
        return int(self.tokens.shape[0])


def _canonical_json_bytes(payload: object) -> bytes:
    """Encode finite sorted ASCII JSON with one terminal line feed."""
    return (
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def file_sha256(path: Path) -> str:
    """Hash one regular shard file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_array_create_only(path: Path, array: np.ndarray) -> None:
    """Write a NumPy array through an exclusive descriptor and fsync it."""
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink(missing_ok=True)
        finally:
            raise


def write_dense_shard(
    output: Path,
    records: Sequence[DenseShardRecord],
    *,
    maximum_sequence_tokens: int,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Encode records into immutable token, length, type, and sample-key arrays."""
    if output.exists():
        raise DenseShardError("Dense shard output already exists")
    if not records:
        raise DenseShardError("Dense shard requires at least one record")
    output.mkdir(parents=True)
    tokens = np.full(
        (len(records), maximum_sequence_tokens),
        PAD_ID,
        dtype=np.uint8,
    )
    lengths = np.empty(len(records), dtype=np.uint16)
    chain_types = np.empty(len(records), dtype=np.uint8)
    sample_keys = np.empty(len(records), dtype=np.uint64)
    for index, record in enumerate(records):
        encoded = encode_unpaired(record.sequence, record.chain_type)
        if len(encoded.input_ids) > maximum_sequence_tokens:
            raise DenseShardError("Encoded record exceeds the shard width")
        tokens[index, : len(encoded.input_ids)] = encoded.input_ids
        lengths[index] = len(encoded.input_ids)
        chain_types[index] = CHAIN_TYPE_TO_ID[record.chain_type]
        sample_keys[index] = record.sample_key
    arrays = {
        "chain_types.npy": chain_types,
        "lengths.npy": lengths,
        "sample_keys.npy": sample_keys,
        "tokens.npy": tokens,
    }
    files = {}
    for name, array in arrays.items():
        path = output / name
        _save_array_create_only(path, array)
        files[name] = {
            "bytes": path.stat().st_size,
            "dtype": str(array.dtype),
            "sha256": file_sha256(path),
            "shape": list(array.shape),
        }
    manifest = {
        "files": files,
        "format_version": SHARD_FORMAT,
        "maximum_sequence_tokens": maximum_sequence_tokens,
        "provenance": provenance,
        "record_count": len(records),
        "test_access": {
            "downstream_metrics_used": False,
            "sabdab2_heldout_accessed": False,
        },
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        _canonical_json_bytes(manifest)
    ).hexdigest()
    manifest_path = output / "manifest.json"
    descriptor = os.open(
        manifest_path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o444,
    )
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(_canonical_json_bytes(manifest))
        handle.flush()
        os.fsync(handle.fileno())
    output.chmod(0o555)
    return manifest


def load_dense_shard(path: Path, *, expected_manifest_sha256: str) -> DenseShard:
    """Validate all file hashes and memory-map a dense sequence shard."""
    manifest_path = path / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DenseShardError("Dense shard manifest is invalid") from error
    unsigned = dict(manifest)
    observed_payload_sha256 = unsigned.pop("manifest_sha256", None)
    if observed_payload_sha256 != expected_manifest_sha256:
        raise DenseShardError("Dense shard manifest payload digest differs")
    if hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest() != expected_manifest_sha256:
        raise DenseShardError("Dense shard manifest is not self-consistent")
    if manifest.get("format_version") != SHARD_FORMAT:
        raise DenseShardError("Dense shard format differs")
    arrays = {}
    for name, receipt in manifest["files"].items():
        file_path = path / name
        if file_sha256(file_path) != receipt["sha256"]:
            raise DenseShardError(f"Dense shard file digest differs: {name}")
        array = np.load(file_path, mmap_mode="r", allow_pickle=False)
        if list(array.shape) != receipt["shape"] or str(array.dtype) != receipt["dtype"]:
            raise DenseShardError(f"Dense shard array schema differs: {name}")
        arrays[name] = array
    record_count = manifest["record_count"]
    if any(array.shape[0] != record_count for array in arrays.values()):
        raise DenseShardError("Dense shard arrays have different record counts")
    return DenseShard(
        chain_types=arrays["chain_types.npy"],
        lengths=arrays["lengths.npy"],
        manifest=manifest,
        sample_keys=arrays["sample_keys.npy"],
        tokens=arrays["tokens.npy"],
    )
