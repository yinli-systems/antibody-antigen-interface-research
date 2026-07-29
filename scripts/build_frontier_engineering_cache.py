#!/usr/bin/env python3
"""Build an externally authorized train-only frontier engineering cache.

The controller accepts a dedicated train-only authority document. It never accepts
validation or test membership. Every source document and embedding-store inventory
is bound to a caller-owned SHA-256 anchor. Tensor bytes are acquired once, hashed,
and only then passed to restricted deserialization.

This is an engineering data-path check. Its outputs are not scientific training,
model-selection, validation, test, biological-performance, or state-of-the-art
evidence.
"""

from __future__ import annotations

import argparse
import csv
import errno
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import shutil
import stat
from typing import Any, Iterable, Mapping, Sequence


SOURCE_AUTHORITY_FORMAT = "frontier_engineering_train_authority_v2"
STORE_INVENTORY_FORMAT = "frontier_embedding_store_inventory_v2"
AUTHORIZATION_FILENAME = "AUTHORIZED_FOLD_INVENTORY.json"
AUTHORIZATION_FORMAT = "frontier_authorized_fold_inventory_v1"
CACHE_MANIFEST_FILENAME = "FOLD_LOCAL_CACHE_MANIFEST.json"
CACHE_MANIFEST_FORMAT = "frontier_fold_local_cache_manifest_v1"
EXAMPLE_ARTIFACT_FORMAT = "frontier_fold_local_example_v1"
BUILD_RECEIPT_FILENAME = "ENGINEERING_CACHE_BUILD_RECEIPT.json"
BUILD_RECEIPT_FORMAT = "frontier_engineering_cache_build_receipt_v2"
DATA_ACCESS_SCOPE = "train_only_fold_local"
VALID_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
EXPECTED_ANTIBODY_DIM = 512
EXPECTED_ANTIGEN_DIM = 1280
EXPECTED_ANTIBERTY_WEIGHTS_SHA256 = (
    "f1ae33eac8cc8784a7d4be5a600141d2fa7bc7d8d5b3f5324d64a6a63bd0f137"
)
EXPECTED_ESM_WEIGHTS_SHA256 = (
    "a08adabb949fa67ad3c14b509d04fd60368b35007b0095e3358f81200c4f4db0"
)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:+/@-]{0,511}")
_MAX_JSON_BYTES = 128 * 1024 * 1024
_MAX_CANONICAL_BYTES = 2 * 1024 * 1024 * 1024
_MAX_EMBEDDING_ARTIFACT_BYTES = 8 * 1024 * 1024 * 1024
_READ_CHUNK_BYTES = 8 * 1024 * 1024


class CacheBuildError(ValueError):
    """Raised when an engineering-cache contract fails closed."""


class _DuplicateKeyError(CacheBuildError):
    """Raised when strict JSON contains a duplicate object key."""


def sha256_bytes(value: bytes) -> str:
    """Return the SHA-256 digest of one byte string."""
    return hashlib.sha256(value).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize finite JSON deterministically."""
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256(value: Any, name: str) -> str:
    """Validate one lowercase SHA-256 digest."""
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise CacheBuildError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _positive_integer(value: Any, name: str) -> int:
    """Return a positive non-boolean integer."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CacheBuildError(f"{name} must be a positive integer")
    if value > (1 << 53) - 1:
        raise CacheBuildError(f"{name} exceeds the interoperable JSON integer range")
    return int(value)


def _nonnegative_integer(value: Any, name: str) -> int:
    """Return a non-negative non-boolean integer."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CacheBuildError(f"{name} must be a non-negative integer")
    if value > (1 << 53) - 1:
        raise CacheBuildError(f"{name} exceeds the interoperable JSON integer range")
    return int(value)


def _exact_fields(
    value: Mapping[str, Any],
    expected: set[str],
    name: str,
) -> None:
    """Require an exact object schema."""
    if type(value) is not dict:
        raise CacheBuildError(f"{name} must be a plain JSON object")
    actual = set(value)
    if actual != expected:
        raise CacheBuildError(
            f"{name} fields differ; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _identifier(value: Any, name: str) -> str:
    """Validate one ASCII identifier safe for leading-field authorization."""
    if not isinstance(value, str) or not _IDENTIFIER_PATTERN.fullmatch(value):
        raise CacheBuildError(f"{name} is not a canonical ASCII identifier")
    return value


def _reject_constant(value: str) -> None:
    """Reject non-finite JSON constants."""
    raise CacheBuildError(f"Non-finite JSON constant is forbidden: {value}")


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Build a JSON object while rejecting duplicate keys."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKeyError(f"Duplicate JSON key is forbidden: {key!r}")
        result[key] = value
    return result


def _strict_json_object(raw: bytes, name: str) -> dict[str, Any]:
    """Parse one finite UTF-8 JSON object from already authenticated bytes."""
    if raw.startswith(b"\xef\xbb\xbf"):
        raise CacheBuildError(f"{name} must not contain a UTF-8 byte-order mark")
    try:
        text = raw.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_pairs,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CacheBuildError(f"{name} is not strict UTF-8 JSON") from error
    if type(value) is not dict:
        raise CacheBuildError(f"{name} must contain one JSON object")
    return value


def _lexical_absolute(path: Path) -> Path:
    """Return an absolute path without resolving any symbolic link."""
    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def _open_directory_nofollow(path: Path) -> int:
    """Open every directory component without following symbolic links."""
    absolute = _lexical_absolute(path)
    if not absolute.is_absolute():
        raise AssertionError("Lexical absolute path construction failed")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open("/", flags)
    try:
        for component in absolute.parts[1:]:
            next_descriptor = os.open(
                component,
                flags | nofollow,
                dir_fd=descriptor,
            )
            status = os.fstat(next_descriptor)
            if not stat.S_ISDIR(status.st_mode):
                os.close(next_descriptor)
                raise CacheBuildError(f"Path component is not a directory: {absolute}")
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _status_fingerprint(status: os.stat_result) -> tuple[int, int, int, int, int]:
    """Return the replacement- and mutation-sensitive fields for one file."""
    return (
        int(status.st_dev),
        int(status.st_ino),
        int(status.st_size),
        int(status.st_mtime_ns),
        int(status.st_ctime_ns),
    )


def _read_open_regular_file(
    descriptor: int,
    *,
    name: str,
    max_bytes: int | None,
) -> tuple[bytes, os.stat_result]:
    """Read one already-open regular file and prove it stayed stable."""
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode):
        raise CacheBuildError(f"{name} must be a regular file")
    if before.st_size < 0 or (max_bytes is not None and before.st_size > max_bytes):
        raise CacheBuildError(f"{name} exceeds its maximum authenticated size")
    chunks: list[bytes] = []
    remaining = int(before.st_size)
    while remaining:
        chunk = os.read(descriptor, min(_READ_CHUNK_BYTES, remaining))
        if not chunk:
            raise CacheBuildError(f"{name} ended before its authenticated size")
        chunks.append(chunk)
        remaining -= len(chunk)
    if os.read(descriptor, 1):
        raise CacheBuildError(f"{name} grew during authenticated acquisition")
    after = os.fstat(descriptor)
    if _status_fingerprint(before) != _status_fingerprint(after):
        raise CacheBuildError(f"{name} changed during authenticated acquisition")
    payload = b"".join(chunks)
    if len(payload) != before.st_size:
        raise CacheBuildError(f"{name} byte count differs from its file metadata")
    return payload, before


def acquire_regular_file(
    path: Path,
    *,
    expected_sha256: str,
    name: str,
    expected_size_bytes: int | None = None,
    max_bytes: int | None = None,
) -> bytes:
    """Acquire, hash, and externally authenticate one regular non-symlink file."""
    expected_digest = _sha256(expected_sha256, f"{name}.expected_sha256")
    absolute = _lexical_absolute(path)
    parent_descriptor = _open_directory_nofollow(absolute.parent)
    descriptor = -1
    try:
        descriptor = os.open(
            absolute.name,
            os.O_RDONLY
            | os.O_CLOEXEC
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0),
            dir_fd=parent_descriptor,
        )
        payload, status = _read_open_regular_file(
            descriptor,
            name=name,
            max_bytes=max_bytes,
        )
    except OSError as error:
        raise CacheBuildError(f"Cannot securely acquire {name}") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_descriptor)
    if expected_size_bytes is not None and status.st_size != expected_size_bytes:
        raise CacheBuildError(f"{name} size differs from its external inventory")
    observed_digest = sha256_bytes(payload)
    if observed_digest != expected_digest:
        raise CacheBuildError(f"{name} SHA-256 differs from its external anchor")
    return payload


def _open_relative_regular_file(
    root_descriptor: int,
    relative_path: PurePosixPath,
) -> int:
    """Open one regular file below a pinned root without following links."""
    descriptor = os.dup(root_descriptor)
    try:
        for component in relative_path.parts[:-1]:
            next_descriptor = os.open(
                component,
                os.O_RDONLY
                | os.O_DIRECTORY
                | os.O_CLOEXEC
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=descriptor,
            )
            if not stat.S_ISDIR(os.fstat(next_descriptor).st_mode):
                os.close(next_descriptor)
                raise CacheBuildError("Embedding shard component is not a directory")
            os.close(descriptor)
            descriptor = next_descriptor
        file_descriptor = os.open(
            relative_path.parts[-1],
            os.O_RDONLY
            | os.O_CLOEXEC
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0),
            dir_fd=descriptor,
        )
        return file_descriptor
    finally:
        os.close(descriptor)


def acquire_store_artifact(
    root_descriptor: int,
    record: Mapping[str, Any],
    *,
    side: str,
) -> bytes:
    """Acquire and authenticate one inventory-bound embedding artifact."""
    relative_path = PurePosixPath(record["relative_path"])
    descriptor = -1
    try:
        descriptor = _open_relative_regular_file(root_descriptor, relative_path)
        payload, status = _read_open_regular_file(
            descriptor,
            name=f"{side} embedding artifact {record['sequence_key']}",
            max_bytes=record["artifact_size_bytes"],
        )
    except OSError as error:
        raise CacheBuildError(f"Cannot securely acquire {side} embedding artifact") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if status.st_size != record["artifact_size_bytes"]:
        raise CacheBuildError(f"{side} embedding artifact size differs")
    if sha256_bytes(payload) != record["artifact_sha256"]:
        raise CacheBuildError(
            f"{side} embedding artifact SHA-256 differs before deserialization"
        )
    return payload


def write_new_bytes(path: Path, payload: bytes) -> None:
    """Create one regular file and refuse replacement or symbolic links."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | os.O_CLOEXEC
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("Short write while publishing immutable artifact")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_new_json(path: Path, value: Any) -> None:
    """Create one deterministic JSON file and refuse replacement."""
    write_new_bytes(path, canonical_json_bytes(value))


def seal_tree_read_only(root: Path) -> dict[str, int | str]:
    """Seal a staging tree and verify every entry with lstat."""
    root_status = root.lstat()
    if not stat.S_ISDIR(root_status.st_mode):
        raise CacheBuildError("Staging root must remain a real directory")
    directories = [root]
    regular_files: list[Path] = []
    for current_root, directory_names, file_names in os.walk(
        root,
        topdown=True,
        followlinks=False,
    ):
        current = Path(current_root)
        for name in directory_names:
            path = current / name
            status = path.lstat()
            if not stat.S_ISDIR(status.st_mode):
                raise CacheBuildError(f"Cache tree contains a non-directory: {path}")
            directories.append(path)
        for name in file_names:
            path = current / name
            status = path.lstat()
            if not stat.S_ISREG(status.st_mode):
                raise CacheBuildError(f"Cache tree contains a special file: {path}")
            regular_files.append(path)
    for path in regular_files:
        os.chmod(path, 0o444, follow_symlinks=False)
    for path in sorted(directories, key=lambda value: len(value.parts), reverse=True):
        os.chmod(path, 0o555, follow_symlinks=False)
    for path in regular_files:
        if stat.S_IMODE(path.lstat().st_mode) != 0o444:
            raise PermissionError(f"Cache file was not sealed read-only: {path}")
    for path in directories:
        if stat.S_IMODE(path.lstat().st_mode) != 0o555:
            raise PermissionError(f"Cache directory was not sealed read-only: {path}")
    return {
        "directory_count": len(directories),
        "regular_file_count": len(regular_files),
        "directory_mode": "0555",
        "regular_file_mode": "0444",
        "symlink_count": 0,
    }


def _framed(sequence: str) -> bytes:
    encoded = sequence.encode("ascii")
    return str(len(encoded)).encode("ascii") + b":" + encoded


def antibody_identifier(vh_sequence: str, vl_sequence: str) -> str:
    """Create the protocol-defined framed antibody identifier."""
    return sha256_bytes(
        b"antibody-v1\0" + _framed(vh_sequence) + _framed(vl_sequence)
    )


def antigen_identifier(sequence: str) -> str:
    """Create the protocol-defined framed antigen identifier."""
    return sha256_bytes(b"antigen-v1\0" + _framed(sequence))


def pair_identifier(antibody_id: str, antigen_id: str) -> str:
    """Create the protocol-defined antibody-antigen pair identifier."""
    return sha256_bytes(
        b"pair-v1\0"
        + antibody_id.encode("ascii")
        + b"\0"
        + antigen_id.encode("ascii")
    )


def normalize_sequence(value: str, name: str, allow_empty: bool = False) -> str:
    """Normalize one authorized sequence under the locked canonical alphabet."""
    if not isinstance(value, str):
        raise CacheBuildError(f"{name} must be text")
    sequence = "".join(character for character in value if not character.isspace())
    sequence = sequence.upper()
    try:
        sequence.encode("ascii")
    except UnicodeEncodeError as error:
        raise CacheBuildError(f"{name} must contain only ASCII residues") from error
    if not sequence and not allow_empty:
        raise CacheBuildError(f"{name} cannot be empty")
    invalid = sorted(set(sequence).difference(VALID_AMINO_ACIDS))
    if invalid:
        raise CacheBuildError(f"{name} contains invalid residues: {invalid}")
    return sequence


class TrainRow:
    """One authority-train row after sequence normalization and framing."""

    __slots__ = (
        "instance_id",
        "antibody_id",
        "antigen_id",
        "pair_id",
        "vh_sequence",
        "vl_sequence",
        "antigen_sequence",
        "antibody_sequence_key",
        "antigen_sequence_key",
    )

    def __init__(
        self,
        *,
        instance_id: str,
        vh_sequence: str,
        vl_sequence: str,
        antigen_sequence: str,
    ) -> None:
        self.instance_id = instance_id
        self.vh_sequence = vh_sequence
        self.vl_sequence = vl_sequence
        self.antigen_sequence = antigen_sequence
        self.antibody_id = antibody_identifier(vh_sequence, vl_sequence)
        self.antigen_id = antigen_identifier(antigen_sequence)
        self.pair_id = pair_identifier(self.antibody_id, self.antigen_id)
        self.antibody_sequence_key = sha256_bytes(
            (vh_sequence + vl_sequence).encode("ascii")
        )
        self.antigen_sequence_key = sha256_bytes(antigen_sequence.encode("ascii"))


def load_train_authority(
    raw: bytes,
    *,
    authority_sha256: str,
    canonical_sha256: str,
) -> tuple[str, tuple[str, ...], dict[str, str]]:
    """Validate a caller-anchored authority containing train membership only."""
    payload = _strict_json_object(raw, "train-only authority")
    fields = {
        "format_version",
        "data_access_scope",
        "split_role",
        "fold_id",
        "canonical_manifest_sha256",
        "source_assignment_sha256",
        "dataset_admission_receipt_sha256",
        "authorized_instance_ids",
    }
    _exact_fields(payload, fields, "train-only authority")
    if payload["format_version"] != SOURCE_AUTHORITY_FORMAT:
        raise CacheBuildError("Unsupported train-only authority format")
    if payload["data_access_scope"] != DATA_ACCESS_SCOPE:
        raise CacheBuildError("Authority scope is not train-only fold-local")
    if payload["split_role"] != "train":
        raise CacheBuildError("Authority split role must be train")
    fold_id = _identifier(payload["fold_id"], "authority.fold_id")
    if (
        _sha256(
            payload["canonical_manifest_sha256"],
            "authority.canonical_manifest_sha256",
        )
        != canonical_sha256
    ):
        raise CacheBuildError("Authority is bound to a different canonical manifest")
    evidence = {
        "authority_manifest_sha256": authority_sha256,
        "source_assignment_sha256": _sha256(
            payload["source_assignment_sha256"],
            "authority.source_assignment_sha256",
        ),
        "dataset_admission_receipt_sha256": _sha256(
            payload["dataset_admission_receipt_sha256"],
            "authority.dataset_admission_receipt_sha256",
        ),
    }
    values = payload["authorized_instance_ids"]
    if not isinstance(values, list) or not values:
        raise CacheBuildError("Authority must contain a nonempty train allowlist")
    identifiers = tuple(
        _identifier(value, f"authority.authorized_instance_ids[{index}]")
        for index, value in enumerate(values)
    )
    if identifiers != tuple(sorted(identifiers)):
        raise CacheBuildError("Authority train allowlist must be sorted")
    if len(set(identifiers)) != len(identifiers):
        raise CacheBuildError("Authority train allowlist contains duplicates")
    return fold_id, identifiers, evidence


def _leading_csv_field(raw_line: str) -> str:
    """Read only the unquoted leading identifier before row authorization."""
    leading = raw_line.partition(",")[0]
    if not leading or leading.startswith('"'):
        raise CacheBuildError("Canonical INSTANCE must be an unquoted first field")
    return leading


def _require_one_record_per_physical_line(raw: bytes) -> None:
    """Reject CSV syntax that could hide record boundaries inside quoted fields."""
    field_start = 0
    unquoted = 1
    quoted = 2
    after_quote = 3
    state = field_start
    index = 0
    while index < len(raw):
        value = raw[index]
        if value == 13:
            if index + 1 >= len(raw) or raw[index + 1] != 10:
                raise CacheBuildError("Canonical manifest contains a bare carriage return")
            if state == quoted:
                raise CacheBuildError(
                    "Canonical CSV records cannot span physical lines"
                )
            state = field_start
            index += 2
            continue
        if value == 10:
            if state == quoted:
                raise CacheBuildError(
                    "Canonical CSV records cannot span physical lines"
                )
            state = field_start
            index += 1
            continue
        if state == field_start:
            if value == 34:
                state = quoted
            elif value != 44:
                state = unquoted
        elif state == unquoted:
            if value == 34:
                raise CacheBuildError(
                    "Canonical manifest contains a quote inside an unquoted field"
                )
            if value == 44:
                state = field_start
        elif state == quoted:
            if value == 34:
                state = after_quote
        elif value == 34:
            state = quoted
        elif value == 44:
            state = field_start
        else:
            raise CacheBuildError(
                "Canonical manifest contains bytes after a closing CSV quote"
            )
        index += 1
    if state == quoted:
        raise CacheBuildError("Canonical CSV records cannot span physical lines")


def load_train_rows(raw: bytes, train_ids: Sequence[str]) -> list[TrainRow]:
    """Parse sequence fields only after an ID passes exact train authorization."""
    if raw.startswith(b"\xef\xbb\xbf"):
        raise CacheBuildError("Canonical manifest must not contain a byte-order mark")
    _require_one_record_per_physical_line(raw)
    physical_lines = raw.splitlines(keepends=True)
    if not physical_lines:
        raise CacheBuildError("Canonical manifest is empty")
    records: dict[str, TrainRow] = {}
    authorized = set(train_ids)
    try:
        header_line = physical_lines[0].decode("utf-8", errors="strict")
        header = next(csv.reader([header_line], strict=True))
    except (UnicodeDecodeError, csv.Error) as error:
        raise CacheBuildError("Canonical manifest header is malformed") from error
    if len(header) != len(set(header)):
        raise CacheBuildError("Canonical manifest header contains duplicate columns")
    required = {"INSTANCE", "vh_seq", "vl_seq", "ag_seq"}
    missing = sorted(required.difference(header))
    if missing:
        raise CacheBuildError(f"Canonical manifest is missing columns: {missing}")
    if header[0] != "INSTANCE":
        raise CacheBuildError("INSTANCE must be the first canonical-manifest column")
    column = {name: header.index(name) for name in required}
    for line_number, raw_line_bytes in enumerate(physical_lines[1:], start=2):
        leading_bytes = raw_line_bytes.partition(b",")[0]
        try:
            leading = leading_bytes.decode("ascii", errors="strict")
        except UnicodeDecodeError:
            continue
        instance_id = _leading_csv_field(leading + ",")
        if instance_id not in authorized:
            continue
        try:
            raw_line = raw_line_bytes.decode("utf-8", errors="strict")
            values = next(csv.reader([raw_line], strict=True))
        except (UnicodeDecodeError, csv.Error) as error:
            raise CacheBuildError(
                f"Authorized canonical row {line_number} is malformed"
            ) from error
        if len(values) != len(header):
            raise CacheBuildError(
                f"Malformed authorized row {line_number}: "
                f"expected {len(header)} fields, observed {len(values)}"
            )
        if values[column["INSTANCE"]] != instance_id:
            raise CacheBuildError(f"Authorized row {line_number} changed its ID")
        if instance_id in records:
            raise CacheBuildError(f"Duplicate canonical row for {instance_id}")
        records[instance_id] = TrainRow(
            instance_id=instance_id,
            vh_sequence=normalize_sequence(
                values[column["vh_seq"]],
                f"{instance_id}.VH",
            ),
            vl_sequence=normalize_sequence(
                values[column["vl_seq"]],
                f"{instance_id}.VL",
                allow_empty=True,
            ),
            antigen_sequence=normalize_sequence(
                values[column["ag_seq"]],
                f"{instance_id}.antigen",
            ),
        )
    missing_ids = sorted(authorized.difference(records))
    if missing_ids:
        raise CacheBuildError(
            f"Canonical manifest is missing {len(missing_ids)} authorized rows; "
            f"first={missing_ids[:5]}"
        )
    rows = [records[instance_id] for instance_id in train_ids]
    if tuple(row.instance_id for row in rows) != tuple(train_ids):
        raise AssertionError("Train row order differs from the exact allowlist")
    return rows


def _store_relative_path(value: Any, sequence_key: str, name: str) -> PurePosixPath:
    """Validate the exact content-addressed path for one store artifact."""
    if not isinstance(value, str):
        raise CacheBuildError(f"{name} must be a relative POSIX path")
    path = PurePosixPath(value)
    expected = PurePosixPath(sequence_key[:2], f"{sequence_key}.pt")
    if path != expected or path.is_absolute() or "\\" in value:
        raise CacheBuildError(f"{name} is not the canonical content-addressed path")
    return path


def load_store_inventory(
    raw: bytes,
    *,
    side: str,
    fold_id: str,
    authority_sha256: str,
    canonical_sha256: str,
    required_sequence_keys: set[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Validate one externally anchored exact train-only store inventory."""
    payload = _strict_json_object(raw, f"{side} store inventory")
    fields = {
        "format_version",
        "data_access_scope",
        "split_role",
        "fold_id",
        "side",
        "authority_manifest_sha256",
        "canonical_manifest_sha256",
        "encoder_name",
        "encoder_weights_sha256",
        "embedding_dim",
        "artifacts",
    }
    _exact_fields(payload, fields, f"{side} store inventory")
    if payload["format_version"] != STORE_INVENTORY_FORMAT:
        raise CacheBuildError(f"Unsupported {side} store inventory format")
    expected_scalars = {
        "data_access_scope": DATA_ACCESS_SCOPE,
        "split_role": "train",
        "fold_id": fold_id,
        "side": side,
        "authority_manifest_sha256": authority_sha256,
        "canonical_manifest_sha256": canonical_sha256,
        "encoder_name": (
            "AntiBERTy_md_smooth"
            if side == "antibody"
            else "facebook/esm2_t33_650M_UR50D"
        ),
        "encoder_weights_sha256": (
            EXPECTED_ANTIBERTY_WEIGHTS_SHA256
            if side == "antibody"
            else EXPECTED_ESM_WEIGHTS_SHA256
        ),
    }
    for field, expected in expected_scalars.items():
        if payload[field] != expected:
            raise CacheBuildError(f"{side} store inventory field {field!r} differs")
    expected_embedding_dim = (
        EXPECTED_ANTIBODY_DIM if side == "antibody" else EXPECTED_ANTIGEN_DIM
    )
    if (
        _positive_integer(
            payload["embedding_dim"],
            f"{side} store inventory embedding_dim",
        )
        != expected_embedding_dim
    ):
        raise CacheBuildError(f"{side} store inventory embedding_dim differs")
    artifacts = payload["artifacts"]
    if not isinstance(artifacts, list) or not artifacts:
        raise CacheBuildError(f"{side} store inventory must contain artifacts")
    parsed: dict[str, dict[str, Any]] = {}
    observed_order: list[str] = []
    for index, raw_record in enumerate(artifacts):
        _exact_fields(
            raw_record,
            {
                "sequence_key",
                "relative_path",
                "artifact_sha256",
                "artifact_size_bytes",
            },
            f"{side}.artifacts[{index}]",
        )
        sequence_key = _sha256(
            raw_record["sequence_key"],
            f"{side}.artifacts[{index}].sequence_key",
        )
        record = {
            "sequence_key": sequence_key,
            "relative_path": _store_relative_path(
                raw_record["relative_path"],
                sequence_key,
                f"{side}.artifacts[{index}].relative_path",
            ).as_posix(),
            "artifact_sha256": _sha256(
                raw_record["artifact_sha256"],
                f"{side}.artifacts[{index}].artifact_sha256",
            ),
            "artifact_size_bytes": _positive_integer(
                raw_record["artifact_size_bytes"],
                f"{side}.artifacts[{index}].artifact_size_bytes",
            ),
        }
        if record["artifact_size_bytes"] > _MAX_EMBEDDING_ARTIFACT_BYTES:
            raise CacheBuildError(f"{side} embedding artifact exceeds the size cap")
        if sequence_key in parsed:
            raise CacheBuildError(f"{side} store inventory contains duplicate keys")
        parsed[sequence_key] = record
        observed_order.append(sequence_key)
    if observed_order != sorted(observed_order):
        raise CacheBuildError(f"{side} store inventory must be sorted by sequence key")
    if set(parsed) != required_sequence_keys:
        raise CacheBuildError(
            f"{side} store inventory is not exactly train-allowlist-equal; "
            f"missing={sorted(required_sequence_keys - set(parsed))[:5]}, "
            f"extra={sorted(set(parsed) - required_sequence_keys)[:5]}"
        )
    summary = {
        "artifact_count": len(parsed),
        "artifact_membership_sha256": _membership_sha256(parsed),
    }
    return parsed, summary


def load_antibody_embedding(
    raw: bytes,
    row: TrainRow,
    torch: Any,
) -> tuple[Any, Any]:
    """Deserialize and validate one already authenticated AntiBERTy artifact."""
    payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    expected_keys = {
        "format_version",
        "sequence_key",
        "source_sequence_sha256",
        "embedding",
        "mask",
        "vh_length",
        "vl_length",
        "embedding_dim",
        "model_name",
        "model_weights_sha256",
    }
    if type(payload) is not dict or set(payload) != expected_keys:
        raise CacheBuildError("AntiBERTy artifact schema differs")
    expected_scalars = {
        "format_version": "sabdab2_antiberty_embedding_v1",
        "sequence_key": row.antibody_sequence_key,
        "source_sequence_sha256": row.antibody_sequence_key,
        "model_name": "AntiBERTy_md_smooth",
        "model_weights_sha256": EXPECTED_ANTIBERTY_WEIGHTS_SHA256,
    }
    for key, expected in expected_scalars.items():
        if payload[key] != expected:
            raise CacheBuildError(f"AntiBERTy artifact field {key!r} differs")
    observed_dimensions = {
        "vh_length": _positive_integer(
            payload["vh_length"],
            "AntiBERTy artifact vh_length",
        ),
        "vl_length": _nonnegative_integer(
            payload["vl_length"],
            "AntiBERTy artifact vl_length",
        ),
        "embedding_dim": _positive_integer(
            payload["embedding_dim"],
            "AntiBERTy artifact embedding_dim",
        ),
    }
    expected_dimensions = {
        "vh_length": len(row.vh_sequence),
        "vl_length": len(row.vl_sequence),
        "embedding_dim": EXPECTED_ANTIBODY_DIM,
    }
    if observed_dimensions != expected_dimensions:
        raise CacheBuildError("AntiBERTy artifact dimensions differ")
    embedding = payload["embedding"]
    mask = payload["mask"]
    expected_length = len(row.vh_sequence) + len(row.vl_sequence)
    if (
        not torch.is_tensor(embedding)
        or embedding.layout != torch.strided
        or embedding.dtype != torch.float32
        or embedding.device.type != "cpu"
        or not embedding.is_contiguous()
        or tuple(embedding.shape) != (expected_length, EXPECTED_ANTIBODY_DIM)
        or not bool(torch.isfinite(embedding).all().item())
    ):
        raise CacheBuildError("AntiBERTy embedding tensor contract differs")
    if (
        not torch.is_tensor(mask)
        or mask.layout != torch.strided
        or mask.dtype != torch.bool
        or mask.device.type != "cpu"
        or not mask.is_contiguous()
        or tuple(mask.shape) != (expected_length,)
        or not bool(mask.all().item())
    ):
        raise CacheBuildError("AntiBERTy mask contract differs")
    chain_ids = torch.cat(
        (
            torch.zeros(len(row.vh_sequence), dtype=torch.long),
            torch.ones(len(row.vl_sequence), dtype=torch.long),
        )
    ).contiguous()
    return embedding.detach().contiguous(), chain_ids


def load_antigen_embedding(raw: bytes, row: TrainRow, torch: Any) -> Any:
    """Deserialize, validate, and merge one authenticated ESM-2 artifact."""
    payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    required = {
        "format_version",
        "sequence_sha256",
        "sequence_length",
        "model_name",
        "max_tokens",
        "stride",
        "starts",
        "ends",
        "chunks",
        "model_weight_sha256",
    }
    if type(payload) is not dict or set(payload) != required:
        raise CacheBuildError("ESM artifact schema differs")
    expected_scalars = {
        "format_version": "sppider_seq_embedding_v1",
        "sequence_sha256": row.antigen_sequence_key,
        "model_name": "facebook/esm2_t33_650M_UR50D",
        "max_tokens": 1024,
        "stride": 512,
        "model_weight_sha256": EXPECTED_ESM_WEIGHTS_SHA256,
    }
    for key, expected in expected_scalars.items():
        if payload[key] != expected:
            raise CacheBuildError(f"ESM artifact field {key!r} differs")
    if (
        _positive_integer(payload["sequence_length"], "ESM artifact sequence_length")
        != len(row.antigen_sequence)
    ):
        raise CacheBuildError("ESM artifact sequence length differs")
    for name, expected in (("max_tokens", 1024), ("stride", 512)):
        if _positive_integer(payload[name], f"ESM artifact {name}") != expected:
            raise CacheBuildError(f"ESM artifact {name} differs")
    starts = payload["starts"]
    ends = payload["ends"]
    chunks = payload["chunks"]
    if (
        not isinstance(starts, list)
        or not isinstance(ends, list)
        or not isinstance(chunks, list)
        or not chunks
        or len(starts) != len(ends)
        or len(starts) != len(chunks)
    ):
        raise CacheBuildError("ESM chunk metadata is inconsistent")
    length = len(row.antigen_sequence)
    accumulated = torch.zeros((length, EXPECTED_ANTIGEN_DIM), dtype=torch.float32)
    counts = torch.zeros(length, dtype=torch.int64)
    previous_start = -1
    for index, (start_value, end_value, chunk) in enumerate(
        zip(starts, ends, chunks)
    ):
        if (
            isinstance(start_value, bool)
            or not isinstance(start_value, int)
            or isinstance(end_value, bool)
            or not isinstance(end_value, int)
        ):
            raise CacheBuildError("ESM chunk intervals must use integers")
        start = int(start_value)
        end = int(end_value)
        if start < 0 or end <= start or end > length or start <= previous_start:
            raise CacheBuildError(f"ESM chunk interval {index} is invalid")
        previous_start = start
        if (
            not torch.is_tensor(chunk)
            or chunk.layout != torch.strided
            or chunk.dtype != torch.float32
            or chunk.device.type != "cpu"
            or not chunk.is_contiguous()
            or tuple(chunk.shape) != (end - start, EXPECTED_ANTIGEN_DIM)
            or not bool(torch.isfinite(chunk).all().item())
        ):
            raise CacheBuildError("ESM chunk tensor contract differs")
        accumulated[start:end] += chunk
        counts[start:end] += 1
    if not bool((counts > 0).all().item()):
        raise CacheBuildError("ESM chunks do not cover the full antigen")
    embedding = accumulated / counts.to(dtype=torch.float32).unsqueeze(-1)
    if not bool(torch.isfinite(embedding).all().item()):
        raise CacheBuildError("Merged ESM embedding is non-finite")
    return embedding.contiguous()


def _membership_sha256(values: Iterable[str]) -> str:
    """Hash a sorted unique membership using newline framing."""
    return sha256_bytes(
        "".join(f"{value}\n" for value in sorted(values)).encode("utf-8")
    )


def _serialize_torch_artifact(payload: Mapping[str, Any], torch: Any) -> bytes:
    """Serialize one tensor artifact into memory before create-only publication."""
    buffer = io.BytesIO()
    torch.save(dict(payload), buffer)
    return buffer.getvalue()


def _make_tree_owner_writable(root: Path) -> None:
    """Make a private failed staging tree removable without following symlinks."""
    try:
        status = root.lstat()
    except FileNotFoundError:
        return
    if not stat.S_ISDIR(status.st_mode):
        raise CacheBuildError("Refusing to clean a replaced staging root")
    for current_root, directory_names, file_names in os.walk(
        root,
        topdown=False,
        followlinks=False,
    ):
        current = Path(current_root)
        for name in file_names:
            path = current / name
            if stat.S_ISREG(path.lstat().st_mode):
                os.chmod(path, 0o600, follow_symlinks=False)
        for name in directory_names:
            path = current / name
            if stat.S_ISDIR(path.lstat().st_mode):
                os.chmod(path, 0o700, follow_symlinks=False)
        os.chmod(current, 0o700, follow_symlinks=False)


def _rename_noreplace(
    parent_descriptor: int,
    source_name: str,
    output_name: str,
) -> str:
    """Rename within one pinned parent without replacing an existing target."""
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is not None:
        renameat2.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        renameat2.restype = ctypes.c_int
        result = renameat2(
            parent_descriptor,
            os.fsencode(source_name),
            parent_descriptor,
            os.fsencode(output_name),
            1,
        )
        if result == 0:
            return "renameat2_noreplace"
        observed_errno = ctypes.get_errno()
        if observed_errno not in {
            errno.ENOSYS,
            errno.EINVAL,
            getattr(errno, "ENOTSUP", errno.EINVAL),
        }:
            raise OSError(observed_errno, os.strerror(observed_errno))

    renameatx_np = getattr(libc, "renameatx_np", None)
    if renameatx_np is not None:
        renameatx_np.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        renameatx_np.restype = ctypes.c_int
        result = renameatx_np(
            parent_descriptor,
            os.fsencode(source_name),
            parent_descriptor,
            os.fsencode(output_name),
            0x00000004,
        )
        if result == 0:
            return "renameatx_np_exclusive"
        observed_errno = ctypes.get_errno()
        if observed_errno not in {
            errno.ENOSYS,
            errno.EINVAL,
            getattr(errno, "ENOTSUP", errno.EINVAL),
        }:
            raise OSError(observed_errno, os.strerror(observed_errno))
    raise CacheBuildError(
        "Create-only publication requires a kernel no-replace rename primitive"
    )


def _publish_create_only(staging: Path, output: Path) -> None:
    """Publish one sealed sibling tree while refusing a pre-existing target."""
    if staging.parent != output.parent:
        raise AssertionError("Staging and output roots must be siblings")
    parent_descriptor = _open_directory_nofollow(output.parent)
    lock_name = f".{output.name}.publish.lock"
    lock_descriptor = -1
    try:
        lock_descriptor = os.open(
            lock_name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | os.O_CLOEXEC
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=parent_descriptor,
        )
        try:
            os.stat(output.name, dir_fd=parent_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(f"Refusing to reuse output root: {output}")
        _rename_noreplace(
            parent_descriptor,
            staging.name,
            output.name,
        )
        os.fsync(parent_descriptor)
    except OSError as error:
        if error.errno in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FileExistsError(f"Refusing to reuse output root: {output}") from error
        raise
    finally:
        if lock_descriptor >= 0:
            os.close(lock_descriptor)
            try:
                os.unlink(lock_name, dir_fd=parent_descriptor)
            except FileNotFoundError:
                pass
        os.close(parent_descriptor)


def _argument(args: argparse.Namespace, name: str) -> Any:
    """Return one required namespace field with a clear migration error."""
    if not hasattr(args, name):
        raise CacheBuildError(f"Missing required builder argument: {name}")
    return getattr(args, name)


def build_cache(args: argparse.Namespace) -> dict[str, Any]:
    """Materialize one externally anchored, authorization-equal engineering cache."""
    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError("PyTorch is required to build tensor artifacts") from error

    authority = _lexical_absolute(_argument(args, "authority_manifest"))
    canonical = _lexical_absolute(_argument(args, "canonical_manifest"))
    antibody_inventory_path = _lexical_absolute(
        _argument(args, "antibody_store_manifest")
    )
    antigen_inventory_path = _lexical_absolute(
        _argument(args, "antigen_store_manifest")
    )
    antibody_root = _lexical_absolute(_argument(args, "antibody_embedding_root"))
    antigen_root = _lexical_absolute(_argument(args, "antigen_embedding_root"))
    output_root = _lexical_absolute(_argument(args, "output_root"))
    expected_builder_sha256 = _sha256(
        _argument(args, "expected_builder_sha256"),
        "expected_builder_sha256",
    )
    builder_bytes = acquire_regular_file(
        Path(__file__),
        expected_sha256=expected_builder_sha256,
        name="builder source",
        max_bytes=16 * 1024 * 1024,
    )
    builder_sha256 = sha256_bytes(builder_bytes)

    expected_authority_sha256 = _sha256(
        _argument(args, "expected_authority_sha256"),
        "expected_authority_sha256",
    )
    expected_canonical_sha256 = _sha256(
        _argument(args, "expected_canonical_manifest_sha256"),
        "expected_canonical_manifest_sha256",
    )
    authority_bytes = acquire_regular_file(
        authority,
        expected_sha256=expected_authority_sha256,
        name="train-only authority",
        max_bytes=_MAX_JSON_BYTES,
    )
    canonical_bytes = acquire_regular_file(
        canonical,
        expected_sha256=expected_canonical_sha256,
        name="canonical manifest",
        max_bytes=_MAX_CANONICAL_BYTES,
    )
    fold_id, train_ids, source_evidence = load_train_authority(
        authority_bytes,
        authority_sha256=expected_authority_sha256,
        canonical_sha256=expected_canonical_sha256,
    )
    rows = load_train_rows(canonical_bytes, train_ids)
    if tuple(row.instance_id for row in rows) != train_ids:
        raise AssertionError("Materialized cache is not exact train-allowlist-equal")

    antibody_inventory_sha256 = _sha256(
        _argument(args, "expected_antibody_store_manifest_sha256"),
        "expected_antibody_store_manifest_sha256",
    )
    antigen_inventory_sha256 = _sha256(
        _argument(args, "expected_antigen_store_manifest_sha256"),
        "expected_antigen_store_manifest_sha256",
    )
    antibody_inventory_bytes = acquire_regular_file(
        antibody_inventory_path,
        expected_sha256=antibody_inventory_sha256,
        name="antibody store inventory",
        max_bytes=_MAX_JSON_BYTES,
    )
    antigen_inventory_bytes = acquire_regular_file(
        antigen_inventory_path,
        expected_sha256=antigen_inventory_sha256,
        name="antigen store inventory",
        max_bytes=_MAX_JSON_BYTES,
    )
    antibody_inventory, antibody_summary = load_store_inventory(
        antibody_inventory_bytes,
        side="antibody",
        fold_id=fold_id,
        authority_sha256=expected_authority_sha256,
        canonical_sha256=expected_canonical_sha256,
        required_sequence_keys={row.antibody_sequence_key for row in rows},
    )
    antigen_inventory, antigen_summary = load_store_inventory(
        antigen_inventory_bytes,
        side="antigen",
        fold_id=fold_id,
        authority_sha256=expected_authority_sha256,
        canonical_sha256=expected_canonical_sha256,
        required_sequence_keys={row.antigen_sequence_key for row in rows},
    )

    output_parent = output_root.parent
    parent_descriptor = _open_directory_nofollow(output_parent)
    try:
        try:
            os.stat(output_root.name, dir_fd=parent_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(f"Refusing to reuse output root: {output_root}")
        staging_name = f".{output_root.name}.building.{secrets.token_hex(16)}"
        os.mkdir(staging_name, mode=0o700, dir_fd=parent_descriptor)
        os.fsync(parent_descriptor)
    finally:
        os.close(parent_descriptor)
    staging_root = output_parent / staging_name

    antibody_root_descriptor = _open_directory_nofollow(antibody_root)
    antigen_root_descriptor = _open_directory_nofollow(antigen_root)
    published = False
    try:
        identities = [
            {
                "instance_id": row.instance_id,
                "antibody_id": row.antibody_id,
                "antigen_id": row.antigen_id,
            }
            for row in rows
        ]
        positive_pairs = sorted(
            {(row.antibody_id, row.antigen_id) for row in rows}
        )
        authorization = {
            "format_version": AUTHORIZATION_FORMAT,
            "data_access_scope": DATA_ACCESS_SCOPE,
            "split_role": "train",
            "fold_id": fold_id,
            "instances": identities,
            "positive_pairs": [list(pair) for pair in positive_pairs],
            "excluded_pairs": [],
        }
        write_new_json(staging_root / AUTHORIZATION_FILENAME, authorization)
        cache_root = staging_root / "fold_cache"
        examples_root = cache_root / "examples"
        examples_root.mkdir(parents=True, mode=0o700)

        antibody_loaded: dict[str, tuple[Any, Any]] = {}
        antigen_loaded: dict[str, Any] = {}
        cache_records: list[dict[str, Any]] = []
        for row in rows:
            if row.antibody_sequence_key not in antibody_loaded:
                antibody_raw = acquire_store_artifact(
                    antibody_root_descriptor,
                    antibody_inventory[row.antibody_sequence_key],
                    side="antibody",
                )
                antibody_loaded[row.antibody_sequence_key] = load_antibody_embedding(
                    antibody_raw,
                    row,
                    torch,
                )
            if row.antigen_sequence_key not in antigen_loaded:
                antigen_raw = acquire_store_artifact(
                    antigen_root_descriptor,
                    antigen_inventory[row.antigen_sequence_key],
                    side="antigen",
                )
                antigen_loaded[row.antigen_sequence_key] = load_antigen_embedding(
                    antigen_raw,
                    row,
                    torch,
                )
            antibody_embedding, chain_ids = antibody_loaded[
                row.antibody_sequence_key
            ]
            antigen_embedding = antigen_loaded[row.antigen_sequence_key]
            artifact_name = f"{sha256_bytes(row.instance_id.encode('ascii'))}.pt"
            artifact_path = examples_root / artifact_name
            artifact_bytes = _serialize_torch_artifact(
                {
                    "format_version": EXAMPLE_ARTIFACT_FORMAT,
                    "instance_id": row.instance_id,
                    "antibody_id": row.antibody_id,
                    "antigen_id": row.antigen_id,
                    "antibody_embeddings": antibody_embedding,
                    "antigen_embeddings": antigen_embedding,
                    "antibody_chain_ids": chain_ids,
                    "contact_target": None,
                    "contact_mask": None,
                },
                torch,
            )
            write_new_bytes(artifact_path, artifact_bytes)
            cache_records.append(
                {
                    "instance_id": row.instance_id,
                    "antibody_id": row.antibody_id,
                    "antigen_id": row.antigen_id,
                    "artifact_path": f"examples/{artifact_name}",
                    "artifact_sha256": sha256_bytes(artifact_bytes),
                    "artifact_size_bytes": len(artifact_bytes),
                    "antibody_length": len(row.vh_sequence) + len(row.vl_sequence),
                    "antigen_length": len(row.antigen_sequence),
                    "contact_supervision": False,
                }
            )
        cache_records.sort(key=lambda value: value["instance_id"])
        if [record["instance_id"] for record in cache_records] != list(train_ids):
            raise AssertionError("Cache record inventory differs from train authorization")

        authorization_sha256 = sha256_bytes(
            canonical_json_bytes(authorization)
        )
        cache_manifest = {
            "format_version": CACHE_MANIFEST_FORMAT,
            "data_access_scope": DATA_ACCESS_SCOPE,
            "split_role": "train",
            "fold_id": fold_id,
            "whole_pool_materialized": False,
            "authorization_manifest_sha256": authorization_sha256,
            "antibody_input_dim": EXPECTED_ANTIBODY_DIM,
            "antigen_input_dim": EXPECTED_ANTIGEN_DIM,
            "instances": cache_records,
            "positive_pairs": [list(pair) for pair in positive_pairs],
            "excluded_pairs": [],
        }
        cache_manifest_bytes = canonical_json_bytes(cache_manifest)
        write_new_bytes(
            cache_root / CACHE_MANIFEST_FILENAME,
            cache_manifest_bytes,
        )
        expected_directory_count = 3
        expected_regular_file_count = len(cache_records) + 3
        receipt = {
            "format_version": BUILD_RECEIPT_FORMAT,
            "status": "complete",
            "claim_boundary": {
                "engineering_data_path_verified": True,
                "scientific_training_evidence": False,
                "model_selection_evidence": False,
                "validation_or_test_evidence": False,
                "biological_performance_evidence": False,
                "state_of_the_art_evidence": False,
            },
            "fold_id": fold_id,
            "source_anchors": {
                **source_evidence,
                "canonical_manifest_sha256": expected_canonical_sha256,
                "antibody_store_manifest_sha256": antibody_inventory_sha256,
                "antigen_store_manifest_sha256": antigen_inventory_sha256,
                "builder_sha256": builder_sha256,
            },
            "selection": {
                "selection_kind": "external_exact_train_allowlist",
                "authorized_instance_count": len(train_ids),
                "selected_instance_count": len(rows),
                "selected_instance_membership_sha256": _membership_sha256(train_ids),
                "selected_pair_count": len(positive_pairs),
                "selected_antibody_count": len(
                    {row.antibody_id for row in rows}
                ),
                "selected_antigen_count": len(
                    {row.antigen_id for row in rows}
                ),
                "authorization_equal": True,
            },
            "embedding_sources": {
                "antibody": {
                    **antibody_summary,
                    "store_manifest_sha256": antibody_inventory_sha256,
                    "encoder_weights_sha256": EXPECTED_ANTIBERTY_WEIGHTS_SHA256,
                },
                "antigen": {
                    **antigen_summary,
                    "store_manifest_sha256": antigen_inventory_sha256,
                    "encoder_weights_sha256": EXPECTED_ESM_WEIGHTS_SHA256,
                },
                "hash_before_deserialization": True,
            },
            "artifacts": {
                "authorization_manifest_sha256": authorization_sha256,
                "cache_manifest_sha256": sha256_bytes(cache_manifest_bytes),
                "example_artifact_count": len(cache_records),
            },
            "immutability_policy": {
                "publication": "create_only_noreplace_under_exclusive_lock",
                "directory_mode": "0555",
                "regular_file_mode": "0444",
                "symlinks_allowed": False,
                "expected_directory_count": expected_directory_count,
                "expected_regular_file_count": expected_regular_file_count,
                "sealed_before_publication": True,
            },
        }
        write_new_json(staging_root / BUILD_RECEIPT_FILENAME, receipt)
        observed_seal = seal_tree_read_only(staging_root)
        if (
            observed_seal["directory_count"] != expected_directory_count
            or observed_seal["regular_file_count"] != expected_regular_file_count
        ):
            raise CacheBuildError("Sealed tree counts differ from the receipt")
        _publish_create_only(staging_root, output_root)
        published = True
    finally:
        os.close(antibody_root_descriptor)
        os.close(antigen_root_descriptor)
        if not published and staging_root.exists():
            _make_tree_owner_writable(staging_root)
            shutil.rmtree(staging_root)
    return receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the externally anchored create-only controller CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority-manifest", type=Path, required=True)
    parser.add_argument("--canonical-manifest", type=Path, required=True)
    parser.add_argument("--antibody-embedding-root", type=Path, required=True)
    parser.add_argument("--antigen-embedding-root", type=Path, required=True)
    parser.add_argument("--antibody-store-manifest", type=Path, required=True)
    parser.add_argument("--antigen-store-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-builder-sha256", required=True)
    parser.add_argument("--expected-authority-sha256", required=True)
    parser.add_argument("--expected-canonical-manifest-sha256", required=True)
    parser.add_argument(
        "--expected-antibody-store-manifest-sha256",
        required=True,
    )
    parser.add_argument(
        "--expected-antigen-store-manifest-sha256",
        required=True,
    )
    args = parser.parse_args(argv)
    for name in (
        "expected_builder_sha256",
        "expected_authority_sha256",
        "expected_canonical_manifest_sha256",
        "expected_antibody_store_manifest_sha256",
        "expected_antigen_store_manifest_sha256",
    ):
        try:
            _sha256(getattr(args, name), name)
        except CacheBuildError as error:
            parser.error(str(error))
    return args


def main() -> None:
    """Run the controller and print a path-free immutable receipt."""
    receipt = build_cache(parse_args())
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
