"""Load an independently audited sharded OAS corpus without concatenation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
from typing import Any, Iterable, Mapping

import numpy as np


CORPUS_FORMAT = "oas_production_corpus_manifest_v1"
SHARD_FORMAT = "oas_production_dense_bucket_shard_v1"
AUDIT_FORMAT = "oas_production_corpus_independent_audit_v1"
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
TEST_ACCESS = {
    "downstream_metrics_used": False,
    "sabdab2_heldout_accessed": False,
}


class ProductionCorpusError(RuntimeError):
    """Report an audit, manifest, shard, or batch-index disagreement."""


@dataclass(frozen=True)
class ProductionCorpusContract:
    """Bind the exact corpus and non-authorizing gate context."""

    independent_audit_file_sha256: str
    independent_audit_payload_sha256: str
    manifest_file_sha256: str
    manifest_payload_sha256: str
    protocol_file_sha256: str
    split: str
    array_file_hashes_verified: bool = True
    training_authorized: bool = False


@dataclass(frozen=True)
class _Shard:
    """Expose one authenticated bucket shard through read-only memory maps."""

    arrays: Mapping[str, np.ndarray]
    bucket: int
    record_count: int
    root: Path


@dataclass(frozen=True)
class ProductionCorpus:
    """Provide vectorized global indexing over immutable bucket shards."""

    contract: ProductionCorpusContract
    manifest: Mapping[str, Any]
    offsets: np.ndarray
    shards: tuple[_Shard, ...]

    def __len__(self) -> int:
        return int(self.offsets[-1])

    def batch(
        self,
        indices: np.ndarray,
        *,
        fields: Iterable[str] | None = None,
    ) -> dict[str, np.ndarray]:
        """Gather one arbitrary global-index batch while preserving input order."""
        requested = np.asarray(indices)
        if requested.dtype.kind not in {"i", "u"} or requested.ndim != 1:
            raise ProductionCorpusError("Batch indices must be a one-dimensional integer array")
        if requested.dtype.kind == "u" and requested.size and requested.max() > np.iinfo(
            np.int64
        ).max:
            raise ProductionCorpusError("Batch index exceeds int64")
        requested = requested.astype(np.int64, copy=False)
        if requested.size and (requested.min() < 0 or requested.max() >= len(self)):
            raise ProductionCorpusError("Batch index is outside the corpus")
        requested_fields = tuple(
            sorted(EXPECTED_ARRAYS if fields is None else set(fields))
        )
        if not requested_fields or not set(requested_fields).issubset(EXPECTED_ARRAYS):
            raise ProductionCorpusError("Requested batch fields are invalid")
        shard_ids = np.searchsorted(self.offsets[1:], requested, side="right")
        output: dict[str, np.ndarray] = {}
        for name in requested_fields:
            shape = (len(requested), *self.shards[0].arrays[name].shape[1:])
            output[name] = np.empty(shape, dtype=self.shards[0].arrays[name].dtype)
        for shard_id in np.unique(shard_ids):
            positions = np.flatnonzero(shard_ids == shard_id)
            local = requested[positions] - self.offsets[shard_id]
            shard = self.shards[int(shard_id)]
            for name, target in output.items():
                target[positions] = shard.arrays[name][local]
        return output

    def residue_lengths(self, indices: np.ndarray) -> np.ndarray:
        """Read residue counts without materializing the corpus-wide length vector."""
        encoded_lengths = self.batch(indices, fields=("lengths.npy",))["lengths.npy"]
        residue_lengths = encoded_lengths.astype(np.int64) - 2
        if np.any(residue_lengths <= 0):
            raise ProductionCorpusError("Encoded sequence has no residue tokens")
        return residue_lengths


def _canonical_json_bytes(value: Any) -> bytes:
    """Encode finite canonical ASCII JSON."""
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


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate keys at every JSON nesting level."""
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ProductionCorpusError(f"Duplicate JSON key: {key}")
        output[key] = value
    return output


def _reject_constant(value: str) -> None:
    """Reject non-finite JSON constants."""
    raise ProductionCorpusError(f"Non-finite JSON constant: {value}")


def _file_sha256(path: Path) -> str:
    """Hash one regular non-symlink file."""
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ProductionCorpusError(f"Input is not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> tuple[dict[str, Any], bytes]:
    """Load duplicate-free finite ASCII JSON from a regular file."""
    if path.lstat().st_mode & 0o222:
        raise ProductionCorpusError(f"JSON evidence is mutable: {path}")
    _file_sha256(path)
    raw = path.read_bytes()
    try:
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_reject_duplicates,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProductionCorpusError(f"Input is not strict ASCII JSON: {path}") from error
    if not isinstance(value, dict):
        raise ProductionCorpusError(f"Input is not a JSON object: {path}")
    return value, raw


def _verify_self_hash(value: Mapping[str, Any], field: str) -> str:
    """Verify one canonical payload self-hash."""
    unsigned = dict(value)
    claimed = unsigned.pop(field, None)
    observed = hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest()
    if claimed != observed:
        raise ProductionCorpusError(f"Self hash differs: {field}")
    return observed


def _sha256(value: object, label: str) -> str:
    """Validate one lowercase SHA-256 string."""
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ProductionCorpusError(f"{label} is not a SHA-256")
    return value


def _safe_shard_root(output_root: Path, relative_root: object) -> Path:
    """Resolve one canonical shard root below the materialization output."""
    if not isinstance(relative_root, str) or not relative_root:
        raise ProductionCorpusError("Shard root is invalid")
    relative = PurePosixPath(relative_root)
    if relative.is_absolute() or ".." in relative.parts or "." in relative.parts:
        raise ProductionCorpusError("Shard root is unsafe")
    resolved_output = output_root.resolve(strict=True)
    current = resolved_output
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ProductionCorpusError("Shard path contains a symlink")
    resolved = current.resolve(strict=True)
    if not resolved.is_relative_to(resolved_output) or not resolved.is_dir():
        raise ProductionCorpusError("Shard root escapes the corpus")
    return resolved


def _assert_frozen(root: Path) -> None:
    """Require an immutable non-symlink shard tree."""
    if root.is_symlink() or not root.is_dir():
        raise ProductionCorpusError("Shard tree is invalid")
    for path in (root, *root.iterdir()):
        if path.is_symlink() or path.stat().st_mode & 0o222:
            raise ProductionCorpusError("Shard tree is mutable")


def _load_shard(
    output_root: Path,
    *,
    bucket: int,
    split: str,
    reference: Mapping[str, Any],
    verify_array_file_hashes: bool,
) -> _Shard:
    """Authenticate and memory-map one complete reducer shard."""
    root = _safe_shard_root(output_root, reference.get("relative_root"))
    _assert_frozen(root)
    manifest_path = root / "manifest.json"
    manifest, raw = _load_json(manifest_path)
    payload_sha256 = _verify_self_hash(manifest, "manifest_sha256")
    if (
        hashlib.sha256(raw).hexdigest() != reference.get("manifest_file_sha256")
        or payload_sha256 != reference.get("manifest_payload_sha256")
        or manifest.get("format_version") != SHARD_FORMAT
        or manifest.get("bucket") != bucket
        or manifest.get("split") != split
        or manifest.get("test_access") != TEST_ACCESS
        or manifest.get("record_count") != reference.get("record_count")
    ):
        raise ProductionCorpusError("Shard manifest binding differs")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or set(files) != EXPECTED_ARRAYS:
        raise ProductionCorpusError("Shard array inventory differs")
    record_count = manifest.get("record_count")
    if isinstance(record_count, bool) or not isinstance(record_count, int) or record_count < 0:
        raise ProductionCorpusError("Shard record count is invalid")
    arrays: dict[str, np.ndarray] = {}
    for name, file_reference in files.items():
        if not isinstance(file_reference, Mapping):
            raise ProductionCorpusError("Shard file reference is invalid")
        path = root / name
        if (
            path.stat().st_size != file_reference.get("bytes")
            or (
                verify_array_file_hashes
                and _file_sha256(path) != file_reference.get("sha256")
            )
        ):
            raise ProductionCorpusError(f"Shard file differs: {name}")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if (
            str(array.dtype) != file_reference.get("dtype")
            or list(array.shape) != file_reference.get("shape")
            or array.shape[0] != record_count
            or array.flags.writeable
        ):
            raise ProductionCorpusError(f"Shard array schema differs: {name}")
        arrays[name] = array
    return _Shard(arrays=arrays, bucket=bucket, record_count=record_count, root=root)


def load_production_corpus(
    output_root: Path,
    *,
    split: str,
    expected_manifest_file_sha256: str,
    expected_manifest_payload_sha256: str,
    expected_protocol_file_sha256: str,
    independent_audit_path: Path,
    expected_independent_audit_file_sha256: str,
    expected_independent_audit_payload_sha256: str,
    verify_array_file_hashes: bool = True,
) -> ProductionCorpus:
    """Authenticate the full evidence graph and open one read-only split."""
    if split not in {"train", "validation"}:
        raise ProductionCorpusError("Corpus split is invalid")
    if not isinstance(verify_array_file_hashes, bool):
        raise ProductionCorpusError("Array hash-verification mode is invalid")
    for value, label in (
        (expected_manifest_file_sha256, "Manifest file digest"),
        (expected_manifest_payload_sha256, "Manifest payload digest"),
        (expected_protocol_file_sha256, "Protocol file digest"),
        (expected_independent_audit_file_sha256, "Audit file digest"),
        (expected_independent_audit_payload_sha256, "Audit payload digest"),
    ):
        _sha256(value, label)
    manifest, manifest_raw = _load_json(output_root / "corpus_manifest.json")
    manifest_payload_sha256 = _verify_self_hash(manifest, "manifest_sha256")
    if (
        hashlib.sha256(manifest_raw).hexdigest() != expected_manifest_file_sha256
        or manifest_payload_sha256 != expected_manifest_payload_sha256
        or manifest.get("format_version") != CORPUS_FORMAT
        or manifest.get("protocol_file_sha256") != expected_protocol_file_sha256
        or manifest.get("global_order")
        != "bucket_ascending_then_sequence_sha256_ascending"
        or manifest.get("test_access") != TEST_ACCESS
        or manifest.get("authorization")
        != {
            "large_scale_pretraining": False,
            "production_corpus_use": False,
            "scientific_claim": False,
        }
    ):
        raise ProductionCorpusError("Corpus manifest identity differs")
    audit, audit_raw = _load_json(independent_audit_path)
    audit_payload_sha256 = _verify_self_hash(audit, "receipt_sha256")
    if (
        hashlib.sha256(audit_raw).hexdigest()
        != expected_independent_audit_file_sha256
        or audit_payload_sha256 != expected_independent_audit_payload_sha256
        or audit.get("format_version") != AUDIT_FORMAT
        or audit.get("inputs", {}).get("manifest_file_sha256")
        != expected_manifest_file_sha256
        or audit.get("inputs", {}).get("manifest_payload_sha256")
        != expected_manifest_payload_sha256
        or audit.get("inputs", {}).get("protocol_file_sha256")
        != expected_protocol_file_sha256
        or audit.get("authorization")
        != {
            "independent_replay": True,
            "large_scale_pretraining": False,
            "production_corpus_use": False,
            "scientific_claim": False,
        }
        or audit.get("test_access") != TEST_ACCESS
    ):
        raise ProductionCorpusError("Independent corpus audit identity differs")
    buckets = manifest.get("buckets")
    if (
        not isinstance(buckets, list)
        or len(buckets) != manifest.get("bucket_count")
        or any(
            not isinstance(item, Mapping)
            or not isinstance(item.get("shards"), Mapping)
            or set(item["shards"]) != {"train", "validation"}
            for item in buckets
        )
        or [item.get("bucket") for item in buckets] != list(range(len(buckets)))
    ):
        raise ProductionCorpusError("Corpus bucket inventory differs")
    shards = tuple(
        _load_shard(
            output_root,
            bucket=bucket["bucket"],
            split=split,
            reference=bucket["shards"][split],
            verify_array_file_hashes=verify_array_file_hashes,
        )
        for bucket in buckets
    )
    offsets = np.zeros(len(shards) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(
        np.asarray([shard.record_count for shard in shards], dtype=np.int64)
    )
    expected_count = manifest.get("counts", {}).get(f"unique_{split}_records")
    if offsets[-1] != expected_count:
        raise ProductionCorpusError("Corpus split count differs")
    return ProductionCorpus(
        contract=ProductionCorpusContract(
            independent_audit_file_sha256=expected_independent_audit_file_sha256,
            independent_audit_payload_sha256=expected_independent_audit_payload_sha256,
            manifest_file_sha256=expected_manifest_file_sha256,
            manifest_payload_sha256=expected_manifest_payload_sha256,
            protocol_file_sha256=expected_protocol_file_sha256,
            split=split,
            array_file_hashes_verified=verify_array_file_hashes,
        ),
        manifest=manifest,
        offsets=offsets,
        shards=shards,
    )
