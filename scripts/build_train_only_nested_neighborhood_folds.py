#!/usr/bin/env python3
"""Build train-only nested direct-neighborhood fold candidates.

The builder accepts only an externally pinned physical train-only authority
allowlist and an exact four-column physical train-only sequence projection.
Full split files, sealed memberships, mixed manifests, extra outcome columns,
multiline CSV records, symlinks, and non-regular inputs are rejected. Every
input is consumed from the same no-follow file descriptor that was hashed.

Four exact global-alignment graphs are built over VH, non-empty VL,
heavy-then-light concatenation, and antigen views. Held-out endpoints are
selected under the locked joint policy and their direct
80%-identity/80%-reciprocal-coverage neighborhoods are excluded separately in
each fold. Long transitive paths are diagnostic only and never force global
fold equality.

The output is a create-only construction candidate. Joint assignment discovery
and externally pinned replay are separate schemas. Even a successful
independent audit is only a non-authorizing core-feasibility precursor; it
cannot satisfy the full split-domain Gate 0 requirement or authorize model
execution by itself.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import platform
import stat
import sys
import tempfile
import types
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

FORMAT_VERSION = "train_only_nested_direct80_four_view_neighborhood_folds_v5"
PROVISIONAL_FORMAT_VERSION = (
    "train_only_nested_direct80_four_view_neighborhood_folds_provisional_v5"
)
PROVISIONAL_LEDGER_FORMAT_VERSION = (
    "train_only_nested_core_feasibility_provisional_attempt_v2"
)
TRAIN_AUTHORITY_FORMAT_VERSION = "gate0_physical_train_only_allowlist_v1"
TRAIN_MANIFEST_COLUMNS = ("INSTANCE", "vh_seq", "vl_seq", "ag_seq")
LOCKED_JOINT_POLICY_SHA256 = (
    "5ddf8b4efef6ae7aed29fd529e613ee12f3b4d0caeddc23d1a320f73bca344fd"
)
REQUIRED_VIEWS = ("vh", "vl", "concatenated_antibody", "antigen")
VALID_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
HEX_DIGITS = frozenset("0123456789abcdef")
IDENTITY_NUMERATOR = 4
IDENTITY_DENOMINATOR = 5
COVERAGE_NUMERATOR = 4
COVERAGE_DENOMINATOR = 5
GAP_OPEN_PENALTY = 10.0
GAP_EXTEND_PENALTY = 1.0
EXPECTED_BIOPYTHON_VERSION = "1.83"
EXPECTED_BLOSUM62_PAYLOAD_SHA256 = (
    "2c62ca4bb7a4c3ec5fe5faee64276841028a7e6638cf846c801dd9c5e64527e2"
)
DEFAULT_SELECTION_SEED = 2026072401
DEFAULT_SELECTION_METHOD = "deterministic_greedy"
SELECTION_METHODS = ("deterministic_greedy", "joint_core_milp")
JOINT_ASSIGNMENT_MODES = (
    "provisional_discovery",
    "externally_pinned_replay",
)
REVIEWED_JOINT_MAXIMUM_BINARY_VARIABLES = 200_000
REVIEWED_JOINT_MAXIMUM_LINEAR_CONSTRAINTS = 3_000_000
REVIEWED_JOINT_MAXIMUM_NONZERO_COEFFICIENTS = 12_000_000
REVIEWED_JOINT_TIME_LIMIT_SECONDS = 43_200.0


@dataclass(frozen=True)
class SourceRecord:
    """One authorized source row after sequence-only materialization."""

    instance_id: str
    vh: str
    vl: str
    antigen: str


@dataclass(frozen=True)
class AntibodyEntity:
    """One canonical antibody entity and its three sequence views."""

    entity_id: str
    vh: str
    vl: str

    @property
    def concatenated(self) -> str:
        """Return the locked heavy-then-light view."""
        return self.vh + self.vl


@dataclass(frozen=True)
class AntigenEntity:
    """One canonical antigen entity."""

    entity_id: str
    sequence: str


@dataclass(frozen=True)
class PairGroup:
    """One indivisible exact antibody-antigen pair and its source rows."""

    pair_id: str
    antibody_id: str
    antigen_id: str
    source_instance_ids: tuple[str, ...]


@dataclass(frozen=True)
class LevelPolicy:
    """Score-free utility and retention gates for one nested level."""

    minimum_validation_pairs: int
    maximum_validation_pairs: int
    minimum_validation_endpoints_per_axis: int
    minimum_validation_components_per_axis: int
    minimum_training_pair_fraction: float
    minimum_training_endpoint_fraction: float
    maximum_quarantine_pair_fraction: float
    maximum_validation_pair_relative_range: float
    maximum_other_count_relative_range: float


@dataclass(frozen=True)
class VerifiedInput:
    """Bytes and acquisition evidence read from one verified no-follow fd."""

    reference: Mapping[str, Any]
    payload: bytes


_WORKER_ALIGNER: object | None = None


def sha256_bytes(payload: bytes) -> str:
    """Return a SHA-256 digest for bytes."""
    return hashlib.sha256(payload).hexdigest()


def _lexical_absolute(path: Path) -> Path:
    """Return an absolute path without resolving any symbolic link."""
    return Path(os.path.abspath(os.path.expanduser(os.fspath(path))))


def _require_secure_open_capabilities() -> dict[str, bool]:
    """Fail closed unless the runtime supports the locked no-follow protocol."""
    required_flags = (
        "O_NOFOLLOW",
        "O_DIRECTORY",
        "O_CLOEXEC",
        "O_NONBLOCK",
    )
    missing = [
        name
        for name in required_flags
        if not isinstance(getattr(os, name, None), int)
    ]
    dir_fd_supported = os.open in getattr(os, "supports_dir_fd", ())
    if missing or not dir_fd_supported:
        raise RuntimeError(
            "Secure fd-only acquisition capabilities are unavailable: "
            f"missing_flags={missing}, open_dir_fd={dir_fd_supported}"
        )
    return {
        **{name: True for name in required_flags},
        "open_dir_fd": True,
        "component_wise_parent_traversal": True,
    }


def _open_directory_nofollow(path: Path) -> int:
    """Open every absolute directory component without following symlinks."""
    _require_secure_open_capabilities()
    absolute = _lexical_absolute(path)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
    nofollow = os.O_NOFOLLOW
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
                raise ValueError(
                    f"Path component is not a directory: {absolute}"
                )
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _stat_fingerprint(value: os.stat_result) -> dict[str, int]:
    """Return the stat fields bound before and after one fd-only read."""
    return {
        "device": int(value.st_dev),
        "inode": int(value.st_ino),
        "mode": int(value.st_mode),
        "link_count": int(value.st_nlink),
        "uid": int(value.st_uid),
        "gid": int(value.st_gid),
        "size_bytes": int(value.st_size),
        "mtime_ns": int(value.st_mtime_ns),
        "ctime_ns": int(value.st_ctime_ns),
    }


def _read_regular_file_nofollow(
    path: Path,
    label: str,
    *,
    chunk_size: int = 8 * 1024 * 1024,
) -> tuple[str, bytes, dict[str, int], dict[str, int]]:
    """Read bytes once after no-follow traversal of every path component."""
    lexical_path = _lexical_absolute(path)
    try:
        parent_descriptor = _open_directory_nofollow(lexical_path.parent)
    except OSError as error:
        raise ValueError(
            f"{label} parent cannot be opened without following a symlink"
        ) from error
    descriptor = -1
    flags = (
        os.O_RDONLY
        | os.O_CLOEXEC
        | os.O_NOFOLLOW
        | os.O_NONBLOCK
    )
    try:
        descriptor = os.open(
            lexical_path.name,
            flags,
            dir_fd=parent_descriptor,
        )
    except OSError as error:
        os.close(parent_descriptor)
        raise ValueError(
            f"{label} cannot be acquired without following a symlink"
        ) from error
    try:
        before_stat = os.fstat(descriptor)
        if not stat.S_ISREG(before_stat.st_mode):
            raise ValueError(f"{label} must be a regular file")
        before = _stat_fingerprint(before_stat)
        chunks: list[bytes] = []
        remaining = int(before_stat.st_size)
        while remaining:
            chunk = os.read(descriptor, min(chunk_size, remaining))
            if not chunk:
                raise ValueError(
                    f"{label} ended before its authenticated size"
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ValueError(f"{label} grew during authenticated acquisition")
        after_stat = os.fstat(descriptor)
        after = _stat_fingerprint(after_stat)
        if before != after:
            raise ValueError(f"{label} changed while its verified fd was read")
        payload = b"".join(chunks)
        if len(payload) != before["size_bytes"]:
            raise ValueError(f"{label} byte count differs from its fd stat")
        return str(lexical_path), payload, before, after
    except OSError as error:
        raise ValueError(f"Cannot securely acquire {label}: {path}") from error
    finally:
        os.close(descriptor)
        os.close(parent_descriptor)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Hash bytes acquired once from a no-follow regular-file fd."""
    _, payload, _, _ = _read_regular_file_nofollow(
        path,
        "file",
        chunk_size=chunk_size,
    )
    return sha256_bytes(payload)


def canonical_json_bytes(payload: Any) -> bytes:
    """Serialize strict JSON deterministically."""
    return (
        json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_payload_sha256(payload: Any) -> str:
    """Hash scientific content with a path-independent canonical encoding."""
    return sha256_bytes(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    )


def membership_sha256(values: Iterable[str]) -> str:
    """Hash a sorted unique membership."""
    encoded = "".join(f"{value}\n" for value in sorted(set(values))).encode("utf-8")
    return sha256_bytes(encoded)


def _reject_duplicate_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    """Build a JSON object while rejecting duplicate keys."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    """Reject non-standard JSON constants."""
    raise ValueError(f"JSON contains non-finite value {value}")


def _validate_finite_json(value: Any, label: str = "JSON") -> None:
    """Reject overflowed finite-number spellings recursively."""
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{label} contains a non-finite number")
    if isinstance(value, Mapping):
        for key, child in value.items():
            _validate_finite_json(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_finite_json(child, f"{label}[{index}]")


def strict_json_object(payload_bytes: bytes, label: str) -> dict[str, Any]:
    """Load one strict JSON object from already verified bytes."""
    if payload_bytes.startswith(b"\xef\xbb\xbf"):
        raise ValueError(f"{label} must not contain a UTF-8 byte-order mark")
    try:
        text = payload_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} is not valid UTF-8") from error
    payload = json.loads(
        text,
        object_pairs_hook=_reject_duplicate_pairs,
        parse_constant=_reject_nonfinite,
    )
    _validate_finite_json(payload, label)
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def validate_sha256(value: Any, label: str) -> str:
    """Validate a lowercase hexadecimal SHA-256 digest."""
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest string")
    digest = value
    if len(digest) != 64 or any(character not in HEX_DIGITS for character in digest):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def verify_file(
    path: Path,
    expected_sha256: str,
    label: str,
) -> VerifiedInput:
    """Acquire and authenticate one input without reopening its pathname."""
    capabilities = _require_secure_open_capabilities()
    expected = validate_sha256(expected_sha256, f"{label} SHA-256")
    lexical, payload, before, after = _read_regular_file_nofollow(path, label)
    observed = sha256_bytes(payload)
    if observed != expected:
        raise ValueError(
            f"{label} SHA-256 mismatch: expected={expected}, observed={observed}"
        )
    return VerifiedInput(
        reference={
            "path": lexical,
            "sha256": observed,
            "size_bytes": len(payload),
            "stat_fingerprint_before": before,
            "stat_fingerprint_after": after,
            "single_verified_fd_read": True,
            "symlink_followed": False,
            "secure_open_capabilities": capabilities,
        },
        payload=payload,
    )


def load_verified_joint_module(verified: VerifiedInput) -> types.ModuleType:
    """Execute only the already authenticated joint-MILP source bytes."""
    source_path = str(verified.reference["path"])
    digest = str(verified.reference["sha256"])
    module_name = f"_verified_nested_joint_milp_{digest}"
    existing = sys.modules.get(module_name)
    if existing is not None:
        if not isinstance(existing, types.ModuleType):
            raise RuntimeError("Verified joint-module cache was corrupted")
        return existing
    try:
        code = compile(verified.payload, source_path, "exec", dont_inherit=True)
    except (SyntaxError, ValueError) as error:
        raise ValueError("Verified joint-MILP source cannot be compiled") from error
    module = types.ModuleType(module_name)
    module.__file__ = source_path
    module.__package__ = ""
    module.__loader__ = None
    sys.modules[module_name] = module
    try:
        exec(code, module.__dict__)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


def atomic_create_json(path: Path, payload: Any) -> None:
    """Publish one JSON artifact atomically without overwrite semantics."""
    target = path.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"Create-only target already exists: {target}")
    encoded = canonical_json_bytes(payload)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".staged", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(encoded)
            handle.flush()
            os.fchmod(handle.fileno(), 0o400)
            os.fsync(handle.fileno())
        os.link(temporary, target)
        directory_descriptor = os.open(
            target.parent, os.O_RDONLY | os.O_DIRECTORY
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def normalize_sequence(value: str, label: str, allow_empty: bool = False) -> str:
    """Apply the locked sequence normalization and alphabet validation."""
    sequence = "".join(character for character in str(value) if not character.isspace())
    sequence = sequence.upper()
    try:
        sequence.encode("ascii")
    except UnicodeEncodeError as error:
        raise ValueError(f"{label} is not ASCII") from error
    if not sequence and allow_empty:
        return ""
    if not sequence:
        raise ValueError(f"{label} is empty")
    unexpected = sorted(set(sequence) - VALID_AMINO_ACIDS)
    if unexpected:
        raise ValueError(f"{label} contains invalid residue symbols: {unexpected}")
    return sequence


def _safe_instance_id(value: Any, label: str) -> str:
    """Validate one exact string identifier."""
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    identifier = value
    if not identifier or identifier != identifier.strip():
        raise ValueError(f"{label} is empty or has surrounding whitespace")
    if any(character in identifier for character in (",", '"', "\r", "\n")):
        raise ValueError(f"{label} contains a forbidden CSV character")
    return identifier


def load_train_authority_allowlist(
    payload_bytes: bytes,
) -> tuple[set[str], dict[str, Any]]:
    """Load an exact physical train-only authority allowlist."""
    payload = strict_json_object(
        payload_bytes,
        "physical train-only authority allowlist",
    )
    expected_schema = {
        "format_version",
        "authorized_split",
        "instance_ids",
    }
    if set(payload) != expected_schema:
        raise ValueError(
            "Train-only authority schema differs; full split or sealed "
            "membership fields are forbidden"
        )
    if payload["format_version"] != TRAIN_AUTHORITY_FORMAT_VERSION:
        raise ValueError("Unsupported train-only authority format_version")
    if payload["authorized_split"] != "train":
        raise ValueError("Train-only authority must declare authorized_split=train")
    raw_ids = payload["instance_ids"]
    if not isinstance(raw_ids, list):
        raise ValueError("Train-only authority instance_ids must be a list")
    identifiers = [
        _safe_instance_id(value, "train-only authority identifier")
        for value in raw_ids
    ]
    if not identifiers:
        raise ValueError("Train-only authority membership is empty")
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Train-only authority membership contains duplicates")
    train = set(identifiers)
    return train, {
        "authority_contract": TRAIN_AUTHORITY_FORMAT_VERSION,
        "source_pool_split": "train",
        "authority_train_instance_count": len(train),
        "authority_train_membership_sha256": membership_sha256(train),
        "validation_or_test_membership_fields_present": False,
        "sealed_membership_materialized": False,
    }


def _single_physical_csv_rows(
    payload_bytes: bytes,
    label: str,
) -> list[list[str]]:
    """Parse strict one-record-per-physical-line UTF-8 CSV bytes."""
    if not payload_bytes:
        raise ValueError(f"{label} is empty")
    if payload_bytes.startswith(b"\xef\xbb\xbf"):
        raise ValueError(f"{label} must not contain a UTF-8 byte-order mark")
    if b"\0" in payload_bytes or b"\r" in payload_bytes:
        raise ValueError(f"{label} contains forbidden NUL or CR bytes")
    if not payload_bytes.endswith(b"\n"):
        raise ValueError(f"{label} must end with one LF")
    physical_lines = payload_bytes.split(b"\n")[:-1]
    if not physical_lines or any(not line for line in physical_lines):
        raise ValueError(f"{label} contains an empty physical line")
    rows: list[list[str]] = []
    for line_number, raw_line in enumerate(physical_lines, start=1):
        try:
            text = raw_line.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(
                f"{label} line {line_number} is not valid UTF-8"
            ) from error
        try:
            parsed = list(csv.reader([text], strict=True))
        except csv.Error as error:
            raise ValueError(
                f"{label} line {line_number} is not one complete CSV record"
            ) from error
        if len(parsed) != 1:
            raise ValueError(
                f"{label} line {line_number} does not contain exactly one record"
            )
        rows.append(parsed[0])
    return rows


def load_train_only_records(
    payload_bytes: bytes,
    train_ids: set[str],
) -> tuple[dict[str, SourceRecord], dict[str, Any]]:
    """Parse an exact sequence-only manifest containing train rows only."""
    rows = _single_physical_csv_rows(
        payload_bytes,
        "physical train-only canonical manifest",
    )
    header = rows[0]
    if tuple(header) != TRAIN_MANIFEST_COLUMNS:
        raise ValueError(
            "Train-only canonical manifest must contain exactly "
            "INSTANCE,vh_seq,vl_seq,ag_seq with no duplicate or extra columns"
        )
    records: dict[str, SourceRecord] = {}
    for line_number, fields in enumerate(rows[1:], start=2):
        if len(fields) != len(TRAIN_MANIFEST_COLUMNS):
            raise ValueError(
                f"Malformed physical train-only row at line {line_number}"
            )
        instance_id = _safe_instance_id(
            fields[0],
            f"physical train-only manifest line {line_number} identifier",
        )
        if instance_id not in train_ids:
            raise ValueError(
                "Physical train-only manifest contains a row outside the "
                "externally pinned train allowlist"
            )
        if instance_id in records:
            raise ValueError(
                f"Duplicate physical train-only row {instance_id!r}"
            )
        records[instance_id] = SourceRecord(
            instance_id=instance_id,
            vh=normalize_sequence(fields[1], f"{instance_id}:VH"),
            vl=normalize_sequence(
                fields[2],
                f"{instance_id}:VL",
                allow_empty=True,
            ),
            antigen=normalize_sequence(
                fields[3],
                f"{instance_id}:antigen",
            ),
        )
    missing_train = train_ids - set(records)
    if missing_train or len(records) != len(train_ids):
        raise ValueError(
            "Physical train-only manifest does not exactly equal the "
            f"train allowlist: missing_train={len(missing_train)}"
        )
    return records, {
        "input_contract": "externally_anchored_physical_train_only_v1",
        "authorized_source_split": "train",
        "authorized_rows_materialized": len(records),
        "manifest_row_count": len(records),
        "manifest_columns": list(TRAIN_MANIFEST_COLUMNS),
        "exact_sequence_only_schema_verified": True,
        "one_record_per_physical_line_verified": True,
        "full_split_input_accepted": False,
        "mixed_train_validation_test_manifest_accepted": False,
        "validation_or_test_memberships_present": 0,
        "validation_or_test_rows_present": 0,
        "validation_or_test_sequence_fields_parsed": 0,
        "validation_or_test_labels_predictions_or_scores_read": 0,
        "accepted_model_checkpoint_embedding_prediction_or_score_inputs": False,
        "sequence_columns_parsed_for": "physical_train_only_projection",
    }


def _frame(sequence: str) -> bytes:
    """Length-frame one ASCII sequence."""
    encoded = sequence.encode("ascii")
    return str(len(encoded)).encode("ascii") + b":" + encoded


def antibody_id(vh: str, vl: str) -> str:
    """Build the canonical framed antibody identifier."""
    return sha256_bytes(b"antibody-v1\0" + _frame(vh) + _frame(vl))


def antigen_id(sequence: str) -> str:
    """Build the canonical framed antigen identifier."""
    return sha256_bytes(b"antigen-v1\0" + _frame(sequence))


def pair_id(antibody_identifier: str, antigen_identifier: str) -> str:
    """Build the canonical framed exact-pair identifier."""
    return sha256_bytes(
        b"pair-v1\0"
        + antibody_identifier.encode("ascii")
        + b"\0"
        + antigen_identifier.encode("ascii")
    )


def canonicalize_inventory(
    records: Mapping[str, SourceRecord],
) -> tuple[
    dict[str, AntibodyEntity],
    dict[str, AntigenEntity],
    tuple[PairGroup, ...],
    dict[str, Any],
]:
    """Collapse authorized rows into framed entities and indivisible pairs."""
    antibodies: dict[str, AntibodyEntity] = {}
    antigens: dict[str, AntigenEntity] = {}
    sources_by_pair: dict[tuple[str, str], list[str]] = defaultdict(list)
    concatenation_to_tuples: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for instance_id in sorted(records):
        record = records[instance_id]
        a_id = antibody_id(record.vh, record.vl)
        g_id = antigen_id(record.antigen)
        antibody = AntibodyEntity(a_id, record.vh, record.vl)
        antigen = AntigenEntity(g_id, record.antigen)
        previous_antibody = antibodies.setdefault(a_id, antibody)
        previous_antigen = antigens.setdefault(g_id, antigen)
        if previous_antibody != antibody or previous_antigen != antigen:
            raise RuntimeError("Canonical digest collision detected")
        sources_by_pair[(a_id, g_id)].append(instance_id)
        concatenation_to_tuples[record.vh + record.vl].add(
            (record.vh, record.vl)
        )
    pairs = tuple(
        PairGroup(
            pair_id=pair_id(a_id, g_id),
            antibody_id=a_id,
            antigen_id=g_id,
            source_instance_ids=tuple(sorted(source_ids)),
        )
        for (a_id, g_id), source_ids in sorted(sources_by_pair.items())
    )
    pair_ids = [pair.pair_id for pair in pairs]
    if len(pair_ids) != len(set(pair_ids)):
        raise RuntimeError("Canonical exact-pair digest collision detected")
    collisions = [
        sha256_bytes(concatenation.encode("ascii"))
        for concatenation, tuples in concatenation_to_tuples.items()
        if len(tuples) > 1
    ]
    return antibodies, antigens, pairs, {
        "authority_train_source_row_count": len(records),
        "unique_antibody_entity_count": len(antibodies),
        "unique_antigen_entity_count": len(antigens),
        "unique_exact_pair_count": len(pairs),
        "antibody_membership_sha256": membership_sha256(antibodies),
        "antigen_membership_sha256": membership_sha256(antigens),
        "exact_pair_membership_sha256": membership_sha256(pair_ids),
        "raw_concatenation_collision_count": len(collisions),
        "raw_concatenation_collision_sequence_digest_sha256s": sorted(collisions),
    }


def _blosum62_payload_sha256(matrix: Any) -> str:
    """Hash the complete labeled substitution matrix."""
    alphabet = str(matrix.alphabet)
    payload = {
        "alphabet": alphabet,
        "scores": [
            [float(matrix[left, right]) for right in alphabet] for left in alphabet
        ],
    }
    return canonical_payload_sha256(payload)


def build_locked_aligner() -> object:
    """Construct and verify the pinned exact global-alignment backend."""
    try:
        import Bio
        from Bio.Align import PairwiseAligner, substitution_matrices
    except ImportError as error:
        raise RuntimeError("Biopython 1.83 is required") from error
    if Bio.__version__ != EXPECTED_BIOPYTHON_VERSION:
        raise RuntimeError(
            "Biopython version mismatch: "
            f"expected={EXPECTED_BIOPYTHON_VERSION}, observed={Bio.__version__}"
        )
    matrix = substitution_matrices.load("BLOSUM62")
    observed = _blosum62_payload_sha256(matrix)
    if observed != EXPECTED_BLOSUM62_PAYLOAD_SHA256:
        raise RuntimeError(
            "BLOSUM62 payload mismatch: "
            f"expected={EXPECTED_BLOSUM62_PAYLOAD_SHA256}, observed={observed}"
        )
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.substitution_matrix = matrix
    aligner.open_gap_score = -GAP_OPEN_PENALTY
    aligner.extend_gap_score = -GAP_EXTEND_PENALTY
    return aligner


def alignment_counts(left: str, right: str, aligner: object) -> tuple[int, int]:
    """Return identical and total residue-residue columns for first optimum."""
    alignment = next(iter(aligner.align(left, right)))
    left_blocks, right_blocks = alignment.aligned
    if len(left_blocks) != len(right_blocks):
        raise RuntimeError("Alignment block cardinalities differ")
    paired = 0
    identical = 0
    for left_block, right_block in zip(left_blocks, right_blocks):
        left_start, left_end = (int(value) for value in left_block)
        right_start, right_end = (int(value) for value in right_block)
        width = left_end - left_start
        if width <= 0 or right_end - right_start != width:
            raise RuntimeError("Alignment blocks have inconsistent widths")
        paired += width
        identical += sum(
            left_residue == right_residue
            for left_residue, right_residue in zip(
                left[left_start:left_end], right[right_start:right_end]
            )
        )
    if paired <= 0:
        raise RuntimeError("Global alignment has no residue-residue columns")
    return identical, paired


def is_safe_length_prune(left_length: int, right_length: int) -> bool:
    """Return whether 80% reciprocal coverage is mathematically impossible."""
    shorter = min(left_length, right_length)
    longer = max(left_length, right_length)
    return COVERAGE_DENOMINATOR * shorter < COVERAGE_NUMERATOR * longer


def meets_threshold(
    identical: int,
    paired: int,
    left_length: int,
    right_length: int,
) -> bool:
    """Apply inclusive 80/80 thresholds with integer cross-products."""
    return (
        IDENTITY_DENOMINATOR * identical >= IDENTITY_NUMERATOR * paired
        and COVERAGE_DENOMINATOR * paired >= COVERAGE_NUMERATOR * left_length
        and COVERAGE_DENOMINATOR * paired >= COVERAGE_NUMERATOR * right_length
    )


def symmetric_threshold_evaluation(
    left: str,
    right: str,
    aligner: object,
) -> tuple[bool, tuple[int, int], tuple[int, int]]:
    """Evaluate both alignment orientations and apply a symmetric fail-closed edge."""
    left_to_right = alignment_counts(left, right, aligner)
    right_to_left = alignment_counts(right, left, aligner)
    qualifies = (
        meets_threshold(*left_to_right, len(left), len(right))
        or meets_threshold(*right_to_left, len(right), len(left))
    )
    return qualifies, left_to_right, right_to_left


def _initialize_alignment_worker() -> None:
    """Initialize one process-local exact aligner."""
    global _WORKER_ALIGNER
    _WORKER_ALIGNER = build_locked_aligner()


def _align_chunk(
    chunk: Sequence[tuple[str, str]],
) -> tuple[int, int, list[tuple[str, str]]]:
    """Align one chunk and return its threshold-qualified sequence pairs."""
    if _WORKER_ALIGNER is None:
        raise RuntimeError("Alignment worker is not initialized")
    edges: list[tuple[str, str]] = []
    for left, right in chunk:
        qualifies, _, _ = symmetric_threshold_evaluation(
            left, right, _WORKER_ALIGNER
        )
        if qualifies:
            edges.append((left, right))
    return len(chunk), 2 * len(chunk), edges


def _chunks(
    values: Iterable[tuple[str, str]], chunk_size: int
) -> Iterator[tuple[tuple[str, str], ...]]:
    """Yield bounded deterministic chunks."""
    iterator = iter(values)
    while True:
        chunk = tuple(itertools.islice(iterator, chunk_size))
        if not chunk:
            return
        yield chunk


def _view_inventory(
    antibodies: Mapping[str, AntibodyEntity],
    antigens: Mapping[str, AntigenEntity],
    view: str,
) -> dict[str, str]:
    """Return entity-to-sequence inventory for one locked view."""
    if view == "vh":
        return {identifier: entity.vh for identifier, entity in antibodies.items()}
    if view == "vl":
        return {
            identifier: entity.vl
            for identifier, entity in antibodies.items()
            if entity.vl
        }
    if view == "concatenated_antibody":
        return {
            identifier: entity.concatenated
            for identifier, entity in antibodies.items()
        }
    if view == "antigen":
        return {
            identifier: entity.sequence
            for identifier, entity in antigens.items()
        }
    raise ValueError(f"Unknown sequence view: {view}")


def build_exact_view_graph(
    inventory: Mapping[str, str],
    *,
    workers: int,
    chunk_size: int,
) -> tuple[set[tuple[str, str]], dict[str, Any]]:
    """Build one exhaustive direct entity graph from unique sequences."""
    entities_by_sequence: dict[str, list[str]] = defaultdict(list)
    for entity_identifier, sequence in inventory.items():
        entities_by_sequence[sequence].append(entity_identifier)
    sequences = sorted(entities_by_sequence)
    exact_group_edges: set[tuple[str, str]] = set()
    for entity_ids in entities_by_sequence.values():
        for left, right in itertools.combinations(sorted(entity_ids), 2):
            exact_group_edges.add((left, right))

    candidate_count = len(sequences) * (len(sequences) - 1) // 2
    safe_pruned = 0
    alignable: list[tuple[str, str]] = []
    sequence_digests = {
        sequence: sha256_bytes(sequence.encode("ascii")) for sequence in sequences
    }
    candidate_enumeration_digest = hashlib.sha256()
    safe_prune_enumeration_digest = hashlib.sha256()
    alignment_enumeration_digest = hashlib.sha256()
    for left, right in itertools.combinations(sequences, 2):
        digest_row = (
            f"{sequence_digests[left]}\0{sequence_digests[right]}\n"
        ).encode("ascii")
        candidate_enumeration_digest.update(digest_row)
        if is_safe_length_prune(len(left), len(right)):
            safe_pruned += 1
            safe_prune_enumeration_digest.update(digest_row)
        else:
            alignable.append((left, right))
            alignment_enumeration_digest.update(digest_row)

    threshold_sequence_pairs: list[tuple[str, str]] = []
    aligned_count = 0
    directional_alignment_execution_count = 0
    if alignable:
        if workers == 1:
            aligner = build_locked_aligner()
            for left, right in alignable:
                qualifies, _, _ = symmetric_threshold_evaluation(
                    left, right, aligner
                )
                aligned_count += 1
                directional_alignment_execution_count += 2
                if qualifies:
                    threshold_sequence_pairs.append((left, right))
        else:
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_initialize_alignment_worker,
            ) as executor:
                for chunk_count, execution_count, edges in executor.map(
                    _align_chunk,
                    _chunks(alignable, chunk_size),
                    chunksize=1,
                ):
                    aligned_count += chunk_count
                    directional_alignment_execution_count += execution_count
                    threshold_sequence_pairs.extend(edges)
    if aligned_count + safe_pruned != candidate_count:
        raise RuntimeError("Exact graph all-pairs accounting is incomplete")
    if directional_alignment_execution_count != 2 * aligned_count:
        raise RuntimeError("Symmetric directional alignment accounting is incomplete")

    entity_edges = set(exact_group_edges)
    for left_sequence, right_sequence in threshold_sequence_pairs:
        for left_entity in entities_by_sequence[left_sequence]:
            for right_entity in entities_by_sequence[right_sequence]:
                low, high = sorted((left_entity, right_entity))
                if low != high:
                    entity_edges.add((low, high))
    sorted_edges = sorted(entity_edges)
    return entity_edges, {
        "entity_count": len(inventory),
        "entity_membership_sha256": membership_sha256(inventory),
        "unique_sequence_count": len(sequences),
        "unique_sequence_membership_sha256": membership_sha256(
            sha256_bytes(sequence.encode("ascii")) for sequence in sequences
        ),
        "distinct_unique_sequence_pair_count": candidate_count,
        "distinct_sequence_pair_enumeration_sha256": (
            candidate_enumeration_digest.hexdigest()
        ),
        "dynamic_programming_alignment_count": aligned_count,
        "directional_dynamic_programming_alignment_execution_count": (
            directional_alignment_execution_count
        ),
        "every_aligned_pair_evaluated_in_both_orientations": (
            directional_alignment_execution_count == 2 * aligned_count
        ),
        "dynamic_programming_alignment_pair_enumeration_sha256": (
            alignment_enumeration_digest.hexdigest()
        ),
        "provably_pruned_by_length_upper_bound_count": safe_pruned,
        "safe_length_prune_pair_enumeration_sha256": (
            safe_prune_enumeration_digest.hexdigest()
        ),
        "alignment_plus_safe_prune_equals_all_distinct_sequence_pairs": (
            aligned_count + safe_pruned == candidate_count
        ),
        "symmetric_either_orientation_threshold_edge_rule": True,
        "identical_sequence_entity_edge_count": len(exact_group_edges),
        "threshold_distinct_sequence_pair_count": len(threshold_sequence_pairs),
        "threshold_distinct_sequence_pair_membership_sha256": membership_sha256(
            (
                f"{sequence_digests[left]}\0{sequence_digests[right]}"
                for left, right in threshold_sequence_pairs
            )
        ),
        "lifted_direct_entity_edge_count": len(sorted_edges),
        "direct_entity_edges": [
            {"left_entity_id": left, "right_entity_id": right}
            for left, right in sorted_edges
        ],
        "direct_entity_edge_membership_sha256": membership_sha256(
            f"{left}\0{right}" for left, right in sorted_edges
        ),
    }


def adjacency(
    values: Iterable[str], edges: Iterable[tuple[str, str]]
) -> dict[str, set[str]]:
    """Build an undirected direct-neighbor map without transitive closure."""
    result = {value: set() for value in values}
    for left, right in edges:
        if left not in result or right not in result:
            raise ValueError("Graph edge references an unknown entity")
        result[left].add(right)
        result[right].add(left)
    return result


def canonical_component_ids(
    graph: Mapping[str, set[str]],
) -> dict[str, str]:
    """Map every full-graph entity to its canonical component representative."""
    universe = set(graph)
    for entity, neighbors in graph.items():
        unknown = neighbors.difference(universe)
        if unknown:
            raise ValueError("Graph contains a neighbor outside its universe")
        for neighbor in neighbors:
            if entity not in graph[neighbor]:
                raise ValueError("Graph is not symmetric")
    remaining = set(graph)
    result: dict[str, str] = {}
    for root in sorted(graph):
        if root not in remaining:
            continue
        remaining.remove(root)
        pending = [root]
        members: list[str] = []
        while pending:
            current = pending.pop()
            members.append(current)
            newly_reached = sorted(
                remaining.intersection(graph[current]),
                reverse=True,
            )
            for neighbor in newly_reached:
                remaining.remove(neighbor)
                pending.append(neighbor)
        representative = min(members)
        result.update({member: representative for member in members})
    if set(result) != set(graph):
        raise RuntimeError("Full-graph component traversal is incomplete")
    return result


def _closed_neighborhood(
    held_out: set[str],
    graph: Mapping[str, set[str]],
    universe: set[str],
) -> set[str]:
    """Return held-out entities plus only their direct in-universe neighbors."""
    result = set(held_out)
    for entity in held_out:
        result.update(graph[entity].intersection(universe))
    return result


def _relative_range(values: Sequence[int]) -> float:
    """Return range divided by arithmetic mean, with exact zero handling."""
    if not values:
        raise ValueError("At least one value is required")
    mean = sum(values) / len(values)
    if mean == 0.0:
        return 0.0 if min(values) == max(values) else math.inf
    return (max(values) - min(values)) / mean


def _fraction(value: float) -> Fraction:
    """Recover one declared decimal policy fraction exactly."""
    if not math.isfinite(value):
        raise ValueError("A policy fraction must be finite")
    return Fraction(str(value))


def _ceil_fraction_product(value: float, count: int) -> int:
    """Return ceil(value * count) using exact integer arithmetic."""
    ratio = _fraction(value)
    return (ratio.numerator * count + ratio.denominator - 1) // ratio.denominator


def _floor_fraction_product(value: float, count: int) -> int:
    """Return floor(value * count) using exact integer arithmetic."""
    ratio = _fraction(value)
    return ratio.numerator * count // ratio.denominator


def _relative_range_passes(values: Sequence[int], maximum: float) -> bool:
    """Compare range/mean to a decimal bound without floating-point drift."""
    ratio = _fraction(maximum)
    total = sum(values)
    if total == 0:
        return min(values) == max(values)
    return (
        len(values) * (max(values) - min(values)) * ratio.denominator
        <= ratio.numerator * total
    )


def _pair_payload(pair: PairGroup) -> dict[str, Any]:
    """Serialize one indivisible exact-pair group."""
    return {
        "pair_id": pair.pair_id,
        "antibody_id": pair.antibody_id,
        "antigen_id": pair.antigen_id,
        "source_instance_ids": list(pair.source_instance_ids),
    }


def _quarantine_reasons(
    pair: PairGroup,
    validation_antibodies: set[str],
    validation_antigens: set[str],
    closed_antibodies: set[str],
    closed_antigens: set[str],
) -> list[str]:
    """Return exhaustive score-free reasons for one quarantined pair."""
    reasons: list[str] = []
    if pair.antibody_id in validation_antibodies:
        reasons.append("held_out_antibody")
    elif pair.antibody_id in closed_antibodies:
        reasons.append("antibody_direct_neighborhood")
    if pair.antigen_id in validation_antigens:
        reasons.append("held_out_antigen")
    elif pair.antigen_id in closed_antigens:
        reasons.append("antigen_direct_neighborhood")
    antibody_closed = pair.antibody_id in closed_antibodies
    antigen_closed = pair.antigen_id in closed_antigens
    both_held_out = (
        pair.antibody_id in validation_antibodies
        and pair.antigen_id in validation_antigens
    )
    if antibody_closed != antigen_closed or (
        antibody_closed and antigen_closed and not both_held_out
    ):
        reasons.append("mixed_role_pair")
    if not reasons:
        raise RuntimeError("A quarantined pair lacks a reason")
    return reasons


def _derive_fold(
    pairs: Sequence[PairGroup],
    antibody_universe: set[str],
    antigen_universe: set[str],
    antibody_graph: Mapping[str, set[str]],
    antigen_graph: Mapping[str, set[str]],
    antibody_component_by_entity: Mapping[str, str],
    antigen_component_by_entity: Mapping[str, str],
    validation_antibodies: set[str],
    validation_antigens: set[str],
    policy: LevelPolicy,
    fold_index: int,
) -> tuple[dict[str, Any], tuple[PairGroup, ...]]:
    """Derive train, validation, and quarantine sets mechanically."""
    closed_antibodies = _closed_neighborhood(
        validation_antibodies, antibody_graph, antibody_universe
    )
    closed_antigens = _closed_neighborhood(
        validation_antigens, antigen_graph, antigen_universe
    )
    training_eligible_antibodies = antibody_universe - closed_antibodies
    training_eligible_antigens = antigen_universe - closed_antigens
    validation: list[PairGroup] = []
    training: list[PairGroup] = []
    quarantine: list[tuple[PairGroup, list[str]]] = []
    for pair in pairs:
        if (
            pair.antibody_id in validation_antibodies
            and pair.antigen_id in validation_antigens
        ):
            validation.append(pair)
        elif (
            pair.antibody_id in training_eligible_antibodies
            and pair.antigen_id in training_eligible_antigens
        ):
            training.append(pair)
        else:
            quarantine.append(
                (
                    pair,
                    _quarantine_reasons(
                        pair,
                        validation_antibodies,
                        validation_antigens,
                        closed_antibodies,
                        closed_antigens,
                    ),
                )
            )
    if len(training) + len(validation) + len(quarantine) != len(pairs):
        raise RuntimeError("Fold role accounting is incomplete")
    model_antibodies = {pair.antibody_id for pair in training}
    model_antigens = {pair.antigen_id for pair in training}
    effective_validation_antibodies = {
        pair.antibody_id for pair in validation
    }
    effective_validation_antigens = {pair.antigen_id for pair in validation}
    validation_antibody_components = {
        antibody_component_by_entity[entity]
        for entity in effective_validation_antibodies
    }
    validation_antigen_components = {
        antigen_component_by_entity[entity]
        for entity in effective_validation_antigens
    }
    gates = {
        "validation_pair_lower_bound_passed": (
            len(validation) >= policy.minimum_validation_pairs
        ),
        "validation_pair_upper_bound_passed": (
            len(validation) <= policy.maximum_validation_pairs
        ),
        "validation_antibody_count_passed": (
            len(effective_validation_antibodies)
            >= policy.minimum_validation_endpoints_per_axis
        ),
        "validation_antigen_count_passed": (
            len(effective_validation_antigens)
            >= policy.minimum_validation_endpoints_per_axis
        ),
        "validation_antibody_component_count_passed": (
            len(validation_antibody_components)
            >= policy.minimum_validation_components_per_axis
        ),
        "validation_antigen_component_count_passed": (
            len(validation_antigen_components)
            >= policy.minimum_validation_components_per_axis
        ),
        "training_pair_retention_passed": (
            len(training)
            >= _ceil_fraction_product(
                policy.minimum_training_pair_fraction,
                len(pairs),
            )
        ),
        "training_antibody_retention_passed": (
            len(model_antibodies)
            >= _ceil_fraction_product(
                policy.minimum_training_endpoint_fraction,
                len(antibody_universe),
            )
        ),
        "training_antigen_retention_passed": (
            len(model_antigens)
            >= _ceil_fraction_product(
                policy.minimum_training_endpoint_fraction,
                len(antigen_universe),
            )
        ),
        "quarantine_limit_passed": (
            len(quarantine)
            <= _floor_fraction_product(
                policy.maximum_quarantine_pair_fraction,
                len(pairs),
            )
        ),
        "every_held_out_antibody_is_effective": (
            validation_antibodies == effective_validation_antibodies
        ),
        "every_held_out_antigen_is_effective": (
            validation_antigens == effective_validation_antigens
        ),
        "model_antibody_allowlist_is_exact": (
            model_antibodies == {pair.antibody_id for pair in training}
        ),
        "model_antigen_allowlist_is_exact": (
            model_antigens == {pair.antigen_id for pair in training}
        ),
        "complete_unique_pair_accounting": (
            len(training) + len(validation) + len(quarantine) == len(pairs)
        ),
        "complete_source_row_accounting": (
            sum(len(pair.source_instance_ids) for pair in training)
            + sum(len(pair.source_instance_ids) for pair in validation)
            + sum(len(pair.source_instance_ids) for pair, _ in quarantine)
            == sum(len(pair.source_instance_ids) for pair in pairs)
        ),
    }
    return {
        "fold_index": fold_index,
        "validation_antibody_ids": sorted(validation_antibodies),
        "validation_antigen_ids": sorted(validation_antigens),
        "validation_antibody_component_ids": sorted(
            validation_antibody_components
        ),
        "validation_antigen_component_ids": sorted(
            validation_antigen_components
        ),
        "closed_direct_neighborhood_antibody_ids": sorted(closed_antibodies),
        "closed_direct_neighborhood_antigen_ids": sorted(closed_antigens),
        "training_eligible_antibody_ids": sorted(training_eligible_antibodies),
        "training_eligible_antigen_ids": sorted(training_eligible_antigens),
        "model_antibody_allowlist": sorted(model_antibodies),
        "model_antigen_allowlist": sorted(model_antigens),
        "training_pairs": [_pair_payload(pair) for pair in training],
        "validation_pairs": [_pair_payload(pair) for pair in validation],
        "quarantined_pairs": [
            {**_pair_payload(pair), "reasons": reasons}
            for pair, reasons in quarantine
        ],
        "counts": {
            "universe_pair_count": len(pairs),
            "training_pair_count": len(training),
            "validation_pair_count": len(validation),
            "quarantined_pair_count": len(quarantine),
            "universe_antibody_count": len(antibody_universe),
            "universe_antigen_count": len(antigen_universe),
            "training_antibody_count": len(model_antibodies),
            "training_antigen_count": len(model_antigens),
            "validation_antibody_count": len(effective_validation_antibodies),
            "validation_antigen_count": len(effective_validation_antigens),
            "validation_antibody_component_count": len(
                validation_antibody_components
            ),
            "validation_antigen_component_count": len(
                validation_antigen_components
            ),
            "training_source_row_count": sum(
                len(pair.source_instance_ids) for pair in training
            ),
            "validation_source_row_count": sum(
                len(pair.source_instance_ids) for pair in validation
            ),
            "quarantined_source_row_count": sum(
                len(pair.source_instance_ids) for pair, _ in quarantine
            ),
        },
        "membership_hashes": {
            "training_pair_sha256": membership_sha256(
                pair.pair_id for pair in training
            ),
            "validation_pair_sha256": membership_sha256(
                pair.pair_id for pair in validation
            ),
            "quarantined_pair_sha256": membership_sha256(
                pair.pair_id for pair, _ in quarantine
            ),
            "model_antibody_allowlist_sha256": membership_sha256(model_antibodies),
            "model_antigen_allowlist_sha256": membership_sha256(model_antigens),
            "validation_antibody_component_sha256": membership_sha256(
                validation_antibody_components
            ),
            "validation_antigen_component_sha256": membership_sha256(
                validation_antigen_components
            ),
        },
        "hard_gates": gates,
        "all_fold_hard_gates_passed": all(gates.values()),
    }, tuple(training)


def _static_pair_cost(
    pair: PairGroup,
    pairs: Sequence[PairGroup],
    antibody_graph: Mapping[str, set[str]],
    antigen_graph: Mapping[str, set[str]],
    antibody_universe: set[str],
    antigen_universe: set[str],
) -> int:
    """Count pairs excluded from training by one held-out anchor pair."""
    closed_antibodies = {pair.antibody_id} | antibody_graph[
        pair.antibody_id
    ].intersection(antibody_universe)
    closed_antigens = {pair.antigen_id} | antigen_graph[
        pair.antigen_id
    ].intersection(antigen_universe)
    return sum(
        candidate.antibody_id in closed_antibodies
        or candidate.antigen_id in closed_antigens
        for candidate in pairs
    )


def _select_held_out_sets(
    pairs: Sequence[PairGroup],
    antibody_graph: Mapping[str, set[str]],
    antigen_graph: Mapping[str, set[str]],
    policy: LevelPolicy,
    fold_count: int,
    seed_label: str,
) -> tuple[list[set[str]], list[set[str]], dict[str, Any]]:
    """Select disjoint held-out endpoints with deterministic score-free ranking."""
    antibody_universe = {pair.antibody_id for pair in pairs}
    antigen_universe = {pair.antigen_id for pair in pairs}
    static_costs = {
        pair.pair_id: _static_pair_cost(
            pair,
            pairs,
            antibody_graph,
            antigen_graph,
            antibody_universe,
            antigen_universe,
        )
        for pair in pairs
    }
    ranked: list[list[PairGroup]] = []
    for fold_index in range(fold_count):
        ranked.append(
            sorted(
                pairs,
                key=lambda pair: (
                    static_costs[pair.pair_id],
                    sha256_bytes(
                        (
                            f"{seed_label}\0{fold_index}\0{pair.pair_id}"
                        ).encode("ascii")
                    ),
                    pair.pair_id,
                ),
            )
        )
    validation_antibodies = [set() for _ in range(fold_count)]
    validation_antigens = [set() for _ in range(fold_count)]
    antibody_owner: dict[str, int] = {}
    antigen_owner: dict[str, int] = {}

    def validation_pairs(fold_index: int, a_values: set[str], g_values: set[str]) -> int:
        return sum(
            pair.antibody_id in a_values and pair.antigen_id in g_values
            for pair in pairs
        )

    stalled_folds: set[int] = set()
    iteration_count = 0
    while True:
        counts = [
            validation_pairs(index, validation_antibodies[index], validation_antigens[index])
            for index in range(fold_count)
        ]
        incomplete = [
            index
            for index, count in enumerate(counts)
            if count < policy.minimum_validation_pairs
        ]
        if not incomplete:
            break
        made_progress = False
        for fold_index in incomplete:
            chosen: PairGroup | None = None
            for pair in ranked[fold_index]:
                if antibody_owner.get(pair.antibody_id, fold_index) != fold_index:
                    continue
                if antigen_owner.get(pair.antigen_id, fold_index) != fold_index:
                    continue
                next_antibodies = validation_antibodies[fold_index] | {
                    pair.antibody_id
                }
                next_antigens = validation_antigens[fold_index] | {pair.antigen_id}
                next_count = validation_pairs(
                    fold_index, next_antibodies, next_antigens
                )
                if next_count <= counts[fold_index]:
                    continue
                if next_count > policy.maximum_validation_pairs:
                    continue
                chosen = pair
                break
            if chosen is None:
                stalled_folds.add(fold_index)
                continue
            validation_antibodies[fold_index].add(chosen.antibody_id)
            validation_antigens[fold_index].add(chosen.antigen_id)
            antibody_owner[chosen.antibody_id] = fold_index
            antigen_owner[chosen.antigen_id] = fold_index
            made_progress = True
            iteration_count += 1
        if not made_progress:
            break
        if iteration_count > 2 * len(pairs) * fold_count:
            raise RuntimeError("Held-out selection exceeded deterministic bound")
    final_counts = [
        validation_pairs(index, validation_antibodies[index], validation_antigens[index])
        for index in range(fold_count)
    ]
    return validation_antibodies, validation_antigens, {
        "algorithm": (
            "round_robin_static_direct_neighborhood_cost_then_fixed_hash_v1"
        ),
        "score_free": True,
        "seed_label": seed_label,
        "selection_step_count": iteration_count,
        "target_minimum_validation_pairs": policy.minimum_validation_pairs,
        "target_maximum_validation_pairs": policy.maximum_validation_pairs,
        "final_validation_pair_counts": final_counts,
        "stalled_fold_indices": sorted(stalled_folds),
        "target_reached_in_every_fold": all(
            policy.minimum_validation_pairs
            <= count
            <= policy.maximum_validation_pairs
            for count in final_counts
        ),
        "global_transitive_closure_used_as_fold_constraint": False,
    }


def _policy_payload(policy: LevelPolicy) -> dict[str, Any]:
    """Serialize one level policy."""
    return {
        "minimum_validation_pairs": policy.minimum_validation_pairs,
        "maximum_validation_pairs": policy.maximum_validation_pairs,
        "minimum_validation_endpoints_per_axis": (
            policy.minimum_validation_endpoints_per_axis
        ),
        "minimum_validation_components_per_axis": (
            policy.minimum_validation_components_per_axis
        ),
        "minimum_training_pair_fraction": policy.minimum_training_pair_fraction,
        "minimum_training_endpoint_fraction": (
            policy.minimum_training_endpoint_fraction
        ),
        "maximum_quarantine_pair_fraction": (
            policy.maximum_quarantine_pair_fraction
        ),
        "maximum_validation_pair_relative_range": (
            policy.maximum_validation_pair_relative_range
        ),
        "maximum_other_count_relative_range": (
            policy.maximum_other_count_relative_range
        ),
    }


def outer_policy(pair_count: int) -> LevelPolicy:
    """Return the locked outer-fold count and retention gates."""
    return LevelPolicy(
        minimum_validation_pairs=max(120, (7 * pair_count + 99) // 100),
        maximum_validation_pairs=9 * pair_count // 100,
        minimum_validation_endpoints_per_axis=60,
        minimum_validation_components_per_axis=25,
        minimum_training_pair_fraction=0.60,
        minimum_training_endpoint_fraction=0.60,
        maximum_quarantine_pair_fraction=0.33,
        maximum_validation_pair_relative_range=0.15,
        maximum_other_count_relative_range=0.20,
    )


def inner_policy(pair_count: int) -> LevelPolicy:
    """Return the locked inner-fold count and retention gates."""
    return LevelPolicy(
        minimum_validation_pairs=max(80, (7 * pair_count + 99) // 100),
        maximum_validation_pairs=9 * pair_count // 100,
        minimum_validation_endpoints_per_axis=40,
        minimum_validation_components_per_axis=20,
        minimum_training_pair_fraction=0.55,
        minimum_training_endpoint_fraction=0.55,
        maximum_quarantine_pair_fraction=0.38,
        maximum_validation_pair_relative_range=0.20,
        maximum_other_count_relative_range=0.25,
    )


def validate_level_policy(
    policy: LevelPolicy,
    pair_count: int,
    *,
    label: str,
) -> None:
    """Reject an internally impossible validation interval before selection."""
    if pair_count <= 0:
        raise ValueError(f"{label} pair universe must be positive")
    if policy.minimum_validation_pairs > policy.maximum_validation_pairs:
        raise ValueError(
            f"{label} validation-pair interval is infeasible: "
            f"minimum={policy.minimum_validation_pairs}, "
            f"maximum={policy.maximum_validation_pairs}, "
            f"pair_count={pair_count}"
        )


def _assemble_level_from_held_out_sets(
    pairs: Sequence[PairGroup],
    antibody_graph: Mapping[str, set[str]],
    antigen_graph: Mapping[str, set[str]],
    policy: LevelPolicy,
    *,
    held_out_antibodies: Sequence[set[str]],
    held_out_antigens: Sequence[set[str]],
    selection: Mapping[str, Any],
) -> tuple[dict[str, Any], list[tuple[PairGroup, ...]]]:
    """Derive and summarize one level from externally selected endpoints."""
    if not pairs:
        raise ValueError("A nested fold universe cannot be empty")
    fold_count = len(held_out_antibodies)
    if fold_count != 3 or len(held_out_antigens) != fold_count:
        raise ValueError("The nested protocol requires exactly three folds")
    validate_level_policy(policy, len(pairs), label="external-held-out-assignment")
    antibody_universe = {pair.antibody_id for pair in pairs}
    antigen_universe = {pair.antigen_id for pair in pairs}
    antibody_component_by_entity = canonical_component_ids(antibody_graph)
    antigen_component_by_entity = canonical_component_ids(antigen_graph)
    if not antibody_universe.issubset(antibody_component_by_entity):
        raise ValueError("Antibody universe leaves the complete component graph")
    if not antigen_universe.issubset(antigen_component_by_entity):
        raise ValueError("Antigen universe leaves the complete component graph")
    if any(not values.issubset(antibody_universe) for values in held_out_antibodies):
        raise ValueError("A held-out antibody is outside the level universe")
    if any(not values.issubset(antigen_universe) for values in held_out_antigens):
        raise ValueError("A held-out antigen is outside the level universe")

    folds: list[dict[str, Any]] = []
    training_sets: list[tuple[PairGroup, ...]] = []
    for fold_index in range(fold_count):
        fold, training = _derive_fold(
            pairs,
            antibody_universe,
            antigen_universe,
            antibody_graph,
            antigen_graph,
            antibody_component_by_entity,
            antigen_component_by_entity,
            held_out_antibodies[fold_index],
            held_out_antigens[fold_index],
            policy,
            fold_index,
        )
        folds.append(fold)
        training_sets.append(training)
    pairwise_disjoint_a = sum(map(len, held_out_antibodies)) == len(
        set().union(*held_out_antibodies)
    )
    pairwise_disjoint_g = sum(map(len, held_out_antigens)) == len(
        set().union(*held_out_antigens)
    )
    count_vectors = {
        "validation_pair_count": [
            fold["counts"]["validation_pair_count"] for fold in folds
        ],
        "validation_antibody_count": [
            fold["counts"]["validation_antibody_count"] for fold in folds
        ],
        "validation_antigen_count": [
            fold["counts"]["validation_antigen_count"] for fold in folds
        ],
        "training_pair_count": [
            fold["counts"]["training_pair_count"] for fold in folds
        ],
        "quarantined_pair_count": [
            fold["counts"]["quarantined_pair_count"] for fold in folds
        ],
    }
    ranges = {
        name: _relative_range(values) for name, values in count_vectors.items()
    }
    balance_gates = {
        "validation_pair_relative_range_passed": (
            _relative_range_passes(
                count_vectors["validation_pair_count"],
                policy.maximum_validation_pair_relative_range,
            )
        ),
        "validation_antibody_relative_range_passed": (
            _relative_range_passes(
                count_vectors["validation_antibody_count"],
                policy.maximum_other_count_relative_range,
            )
        ),
        "validation_antigen_relative_range_passed": (
            _relative_range_passes(
                count_vectors["validation_antigen_count"],
                policy.maximum_other_count_relative_range,
            )
        ),
        "training_pair_relative_range_passed": (
            _relative_range_passes(
                count_vectors["training_pair_count"],
                policy.maximum_other_count_relative_range,
            )
        ),
        "quarantine_pair_relative_range_passed": (
            _relative_range_passes(
                count_vectors["quarantined_pair_count"],
                policy.maximum_other_count_relative_range,
            )
        ),
    }
    target_reached = all(
        policy.minimum_validation_pairs
        <= fold["counts"]["validation_pair_count"]
        <= policy.maximum_validation_pairs
        for fold in folds
    )
    claimed_target_reached = selection.get("target_reached_in_every_fold")
    if claimed_target_reached is not None and bool(claimed_target_reached) != (
        target_reached
    ):
        raise RuntimeError("Selection receipt disagrees with derived validation counts")
    level_gates = {
        "selection_target_reached_in_every_fold": target_reached,
        "held_out_antibody_sets_pairwise_disjoint": pairwise_disjoint_a,
        "held_out_antigen_sets_pairwise_disjoint": pairwise_disjoint_g,
        "every_fold_hard_gate_passed": all(
            fold["all_fold_hard_gates_passed"] for fold in folds
        ),
        **balance_gates,
    }
    return {
        "universe": {
            "unique_pair_count": len(pairs),
            "antibody_count": len(antibody_universe),
            "antigen_count": len(antigen_universe),
            "source_row_count": sum(
                len(pair.source_instance_ids) for pair in pairs
            ),
            "pair_membership_sha256": membership_sha256(
                pair.pair_id for pair in pairs
            ),
            "full_graph_antibody_component_count": len(
                set(antibody_component_by_entity.values())
            ),
            "full_graph_antigen_component_count": len(
                set(antigen_component_by_entity.values())
            ),
            "full_graph_antibody_component_membership_sha256": membership_sha256(
                f"{entity}\0{representative}"
                for entity, representative in antibody_component_by_entity.items()
            ),
            "full_graph_antigen_component_membership_sha256": membership_sha256(
                f"{entity}\0{representative}"
                for entity, representative in antigen_component_by_entity.items()
            ),
        },
        "policy": _policy_payload(policy),
        "selection": dict(selection),
        "folds": folds,
        "count_relative_ranges": ranges,
        "level_hard_gates": level_gates,
        "all_level_hard_gates_passed": all(level_gates.values()),
    }, training_sets


def build_level(
    pairs: Sequence[PairGroup],
    antibody_graph: Mapping[str, set[str]],
    antigen_graph: Mapping[str, set[str]],
    policy: LevelPolicy,
    *,
    fold_count: int,
    seed_label: str,
) -> tuple[dict[str, Any], list[tuple[PairGroup, ...]]]:
    """Construct one complete fold level and evaluate its hard gates."""
    if not pairs:
        raise ValueError("A nested fold universe cannot be empty")
    validate_level_policy(policy, len(pairs), label=seed_label)
    held_out_a, held_out_g, selection = _select_held_out_sets(
        pairs,
        antibody_graph,
        antigen_graph,
        policy,
        fold_count,
        seed_label,
    )
    return _assemble_level_from_held_out_sets(
        pairs,
        antibody_graph,
        antigen_graph,
        policy,
        held_out_antibodies=held_out_a,
        held_out_antigens=held_out_g,
        selection=selection,
    )


def _level_policy_from_joint_policy(
    policy: Any,
    pair_count: int,
    *,
    joint_module: Any,
) -> LevelPolicy:
    """Convert an exact-rational MILP policy for one realized universe."""
    lower, upper = joint_module.validation_pair_interval(policy, pair_count)
    return LevelPolicy(
        minimum_validation_pairs=lower,
        maximum_validation_pairs=upper,
        minimum_validation_endpoints_per_axis=(
            policy.minimum_validation_endpoints_per_axis
        ),
        minimum_validation_components_per_axis=(
            policy.minimum_validation_components_per_axis
        ),
        minimum_training_pair_fraction=(
            policy.minimum_training_pair_fraction.numerator
            / policy.minimum_training_pair_fraction.denominator
        ),
        minimum_training_endpoint_fraction=(
            policy.minimum_training_endpoint_fraction.numerator
            / policy.minimum_training_endpoint_fraction.denominator
        ),
        maximum_quarantine_pair_fraction=(
            policy.maximum_quarantine_pair_fraction.numerator
            / policy.maximum_quarantine_pair_fraction.denominator
        ),
        maximum_validation_pair_relative_range=(
            policy.maximum_validation_pair_relative_range.numerator
            / policy.maximum_validation_pair_relative_range.denominator
        ),
        maximum_other_count_relative_range=(
            policy.maximum_other_count_relative_range.numerator
            / policy.maximum_other_count_relative_range.denominator
        ),
    )


def _selected_assignment_members(
    assignment: Mapping[tuple[str, ...], int],
    prefix: tuple[str, ...],
) -> set[str]:
    """Return selected terminal identifiers for one exact variable prefix."""
    return {
        key[-1]
        for key, value in assignment.items()
        if value == 1 and len(key) == len(prefix) + 1 and key[:-1] == prefix
    }


def _payload_pair_ids(fold: Mapping[str, Any], role: str) -> set[str]:
    """Return pair identifiers for one serialized fold role."""
    key_by_role = {
        "validation_pair": "validation_pairs",
        "training_pair": "training_pairs",
        "quarantine_pair": "quarantined_pairs",
    }
    return {
        str(item["pair_id"])
        for item in fold[key_by_role[role]]
    }


def _assert_fold_matches_joint_assignment(
    fold: Mapping[str, Any],
    assignment: Mapping[tuple[str, ...], int],
    prefix: tuple[str, ...],
) -> None:
    """Fail if mechanical fold derivation differs from the MILP assignment."""
    comparisons = {
        "holdout_a": set(fold["validation_antibody_ids"]),
        "holdout_g": set(fold["validation_antigen_ids"]),
        "closed_a": set(fold["closed_direct_neighborhood_antibody_ids"]),
        "closed_g": set(fold["closed_direct_neighborhood_antigen_ids"]),
        "train_present_a": set(fold["model_antibody_allowlist"]),
        "train_present_g": set(fold["model_antigen_allowlist"]),
        "validation_component_a": set(
            fold["validation_antibody_component_ids"]
        ),
        "validation_component_g": set(
            fold["validation_antigen_component_ids"]
        ),
        "validation_pair": _payload_pair_ids(fold, "validation_pair"),
        "training_pair": _payload_pair_ids(fold, "training_pair"),
        "quarantine_pair": _payload_pair_ids(fold, "quarantine_pair"),
    }
    mismatched = [
        role
        for role, observed in comparisons.items()
        if observed
        != _selected_assignment_members(assignment, (*prefix, role))
    ]
    if mismatched:
        raise RuntimeError(
            "Mechanical fold derivation differs from the joint MILP assignment: "
            + ", ".join(sorted(mismatched))
        )


def _joint_level_selection_receipt(
    assignment: Mapping[tuple[str, ...], int],
    prefixes: Sequence[tuple[str, ...]],
    policy: LevelPolicy,
) -> dict[str, Any]:
    """Build a deterministic receipt for one three-fold MILP assignment slice."""
    validation_counts = [
        len(
            _selected_assignment_members(
                assignment,
                (*prefix, "validation_pair"),
            )
        )
        for prefix in prefixes
    ]
    selected_keys = (
        "\0".join(key)
        for key, value in assignment.items()
        if value == 1
        and any(key[: len(prefix)] == prefix for prefix in prefixes)
    )
    return {
        "algorithm": "joint_3_outer_9_inner_core_feasibility_milp_v2",
        "score_free": True,
        "constant_zero_objective_only": True,
        "lexicographic_optimization_performed": False,
        "target_minimum_validation_pairs": policy.minimum_validation_pairs,
        "target_maximum_validation_pairs": policy.maximum_validation_pairs,
        "final_validation_pair_counts": validation_counts,
        "stalled_fold_indices": [],
        "target_reached_in_every_fold": all(
            policy.minimum_validation_pairs
            <= count
            <= policy.maximum_validation_pairs
            for count in validation_counts
        ),
        "selected_binary_membership_sha256": membership_sha256(selected_keys),
        "global_transitive_closure_used_as_fold_constraint": False,
    }


def build_joint_nested_levels(
    pairs: Sequence[PairGroup],
    antibody_graph: Mapping[str, set[str]],
    antigen_graph: Mapping[str, set[str]],
    *,
    resource_limits: Any,
    time_limit_seconds: float,
    joint_outer_policy: Any | None = None,
    joint_inner_policy: Any | None = None,
    joint_module: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Solve and materialize the exact joint core-feasibility assignment."""
    if time_limit_seconds <= 0:
        raise ValueError("Joint MILP time limit must be positive")
    if joint_outer_policy is None:
        joint_outer_policy = joint_module.LOCKED_OUTER_POLICY
    if joint_inner_policy is None:
        joint_inner_policy = joint_module.LOCKED_INNER_POLICY
    canonical_pairs = tuple(
        joint_module.CanonicalPair(
            pair.pair_id,
            pair.antibody_id,
            pair.antigen_id,
        )
        for pair in pairs
    )
    antibody_universe = {pair.antibody_id for pair in pairs}
    antigen_universe = {pair.antigen_id for pair in pairs}
    problem = joint_module.JointSplitProblem(
        pairs=canonical_pairs,
        antibody_direct_neighbors={
            entity: frozenset(antibody_graph[entity].intersection(antibody_universe))
            for entity in sorted(antibody_universe)
        },
        antigen_direct_neighbors={
            entity: frozenset(antigen_graph[entity].intersection(antigen_universe))
            for entity in sorted(antigen_universe)
        },
    )
    model = joint_module.build_joint_feasibility_model(
        problem,
        outer_policy=joint_outer_policy,
        inner_policy=joint_inner_policy,
        resource_limits=resource_limits,
    )
    solve = joint_module.solve_joint_feasibility(
        model,
        time_limit_seconds=time_limit_seconds,
    )
    if (
        not solve.feasible_for_unscored_split_construction
        or solve.assignment is None
    ):
        raise RuntimeError(
            "Joint core-feasibility MILP did not produce a certified assignment: "
            f"status={solve.solver_status}, message={solve.solver_message!r}"
        )
    assignment = solve.assignment
    outer_level_policy = _level_policy_from_joint_policy(
        joint_outer_policy,
        len(pairs),
        joint_module=joint_module,
    )
    outer_prefixes = [
        ("outer", str(outer_index)) for outer_index in range(3)
    ]
    outer_held_out_a = [
        _selected_assignment_members(assignment, (*prefix, "holdout_a"))
        for prefix in outer_prefixes
    ]
    outer_held_out_g = [
        _selected_assignment_members(assignment, (*prefix, "holdout_g"))
        for prefix in outer_prefixes
    ]
    outer, outer_training_sets = _assemble_level_from_held_out_sets(
        pairs,
        antibody_graph,
        antigen_graph,
        outer_level_policy,
        held_out_antibodies=outer_held_out_a,
        held_out_antigens=outer_held_out_g,
        selection=_joint_level_selection_receipt(
            assignment,
            outer_prefixes,
            outer_level_policy,
        ),
    )
    for outer_index, (fold, outer_training) in enumerate(
        zip(outer["folds"], outer_training_sets)
    ):
        outer_prefix = ("outer", str(outer_index))
        _assert_fold_matches_joint_assignment(fold, assignment, outer_prefix)
        inner_level_policy = _level_policy_from_joint_policy(
            joint_inner_policy,
            len(outer_training),
            joint_module=joint_module,
        )
        inner_prefixes = [
            ("inner", str(outer_index), str(inner_index))
            for inner_index in range(3)
        ]
        inner_held_out_a = [
            _selected_assignment_members(assignment, (*prefix, "holdout_a"))
            for prefix in inner_prefixes
        ]
        inner_held_out_g = [
            _selected_assignment_members(assignment, (*prefix, "holdout_g"))
            for prefix in inner_prefixes
        ]
        inner, _ = _assemble_level_from_held_out_sets(
            outer_training,
            antibody_graph,
            antigen_graph,
            inner_level_policy,
            held_out_antibodies=inner_held_out_a,
            held_out_antigens=inner_held_out_g,
            selection=_joint_level_selection_receipt(
                assignment,
                inner_prefixes,
                inner_level_policy,
            ),
        )
        for inner_index, inner_fold in enumerate(inner["folds"]):
            _assert_fold_matches_joint_assignment(
                inner_fold,
                assignment,
                ("inner", str(outer_index), str(inner_index)),
            )
        fold["inner_level"] = inner

    selected_assignment_sha256 = membership_sha256(
        "\0".join(key)
        for key, value in assignment.items()
        if value == 1
    )
    receipt = {
        "implementation": "joint_3_outer_9_inner_core_feasibility_milp_v2",
        "exact_nested_milp_used": True,
        "joint_outer_and_all_dependent_inner_folds_in_one_model": True,
        "outer_fold_count": 3,
        "dependent_inner_fold_count_per_outer": 3,
        "total_train_validation_boundary_count": 12,
        "exact_joint_policy": joint_module.joint_policy_payload(
            joint_outer_policy,
            joint_inner_policy,
        ),
        "exact_joint_policy_sha256": joint_module.joint_policy_sha256(
            joint_outer_policy,
            joint_inner_policy,
        ),
        "objective": "constant_zero_feasibility",
        "lexicographic_optimization_performed": False,
        "solver": solve.solver,
        "solver_status": solve.solver_status,
        "solver_termination": solve.solver_termination,
        "solver_message": solve.solver_message,
        "solver_reported_success": solve.solver_reported_success,
        "solver_solution_present": solve.solver_solution_present,
        "solver_certified_optimal_for_constant_zero_objective": (
            solve.solver_certified_optimal
        ),
        "solver_reported_mip_gap": solve.solver_reported_mip_gap,
        "solver_reported_mip_gap_is_finite_zero": (
            solve.solver_reported_mip_gap_is_finite_zero
        ),
        "maximum_integrality_error": solve.maximum_integrality_error,
        "accepted_integrality_tolerance": (
            solve.accepted_integrality_tolerance
        ),
        "exact_integer_row_audit_passed": (
            solve.exact_integer_row_audit_passed
        ),
        "certified_core_feasibility_assignment": (
            solve.feasible_for_unscored_split_construction
        ),
        "matrix_sha256": solve.matrix_sha256,
        "selected_binary_membership_sha256": selected_assignment_sha256,
        "model_size": asdict(model.estimate),
        "reviewed_resource_limits": asdict(resource_limits),
        "time_limit_seconds": time_limit_seconds,
        "mechanical_role_parity_check_passed_for_all_12_folds": True,
        "failure_semantics": (
            "non-optimal termination, nonzero reported gap, failed exact row "
            "audit, or assignment/ledger mismatch aborts publication"
        ),
        "claim_boundary": (
            "exact joint core-feasibility including full-graph component minima "
            "only; no lexicographic optimality, representativeness, or "
            "attainable-precision claim"
        ),
    }
    return outer, receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the create-only builder command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-authority-allowlist",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--train-authority-allowlist-sha256",
        required=True,
    )
    parser.add_argument(
        "--train-canonical-manifest",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--train-canonical-manifest-sha256",
        required=True,
    )
    parser.add_argument("--expected-builder-source-sha256", required=True)
    parser.add_argument("--joint-milp-source", type=Path)
    parser.add_argument("--expected-joint-milp-source-sha256")
    parser.add_argument("--expected-joint-policy-sha256")
    parser.add_argument("--expected-joint-selected-membership-sha256")
    parser.add_argument(
        "--joint-assignment-mode",
        choices=JOINT_ASSIGNMENT_MODES,
    )
    parser.add_argument("--provisional-attempt-id")
    parser.add_argument("--provisional-ledger-entry", type=Path)
    parser.add_argument("--pinned-replay-id")
    parser.add_argument("--source-provisional-attempt-id")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold-count", type=int, default=3)
    parser.add_argument(
        "--selection-method",
        choices=SELECTION_METHODS,
        default=DEFAULT_SELECTION_METHOD,
    )
    parser.add_argument("--selection-seed", type=int, default=DEFAULT_SELECTION_SEED)
    parser.add_argument(
        "--joint-maximum-binary-variables",
        type=int,
        default=REVIEWED_JOINT_MAXIMUM_BINARY_VARIABLES,
    )
    parser.add_argument(
        "--joint-maximum-linear-constraints",
        type=int,
        default=REVIEWED_JOINT_MAXIMUM_LINEAR_CONSTRAINTS,
    )
    parser.add_argument(
        "--joint-maximum-nonzero-coefficients",
        type=int,
        default=REVIEWED_JOINT_MAXIMUM_NONZERO_COEFFICIENTS,
    )
    parser.add_argument(
        "--joint-time-limit-seconds",
        type=float,
        default=REVIEWED_JOINT_TIME_LIMIT_SECONDS,
    )
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=256)
    args = parser.parse_args(argv)
    for name in (
        "train_authority_allowlist_sha256",
        "train_canonical_manifest_sha256",
        "expected_builder_source_sha256",
        "expected_joint_milp_source_sha256",
        "expected_joint_policy_sha256",
        "expected_joint_selected_membership_sha256",
    ):
        if getattr(args, name) is None:
            continue
        try:
            setattr(args, name, validate_sha256(getattr(args, name), name))
        except ValueError as error:
            parser.error(str(error))
    if (
        args.selection_method == "joint_core_milp"
        and (
            args.joint_milp_source is None
            or args.expected_joint_milp_source_sha256 is None
            or args.expected_joint_policy_sha256 is None
            or args.joint_assignment_mode is None
        )
    ):
        parser.error(
            "--joint-milp-source, --expected-joint-milp-source-sha256, "
            "--expected-joint-policy-sha256, and --joint-assignment-mode are "
            "required for joint_core_milp"
        )
    if (
        args.selection_method == "joint_core_milp"
        and args.expected_joint_policy_sha256
        != LOCKED_JOINT_POLICY_SHA256
    ):
        parser.error(
            "--expected-joint-policy-sha256 differs from the repository's "
            "locked exact 3x3 policy"
        )
    if (
        args.selection_method == "joint_core_milp"
        and args.joint_assignment_mode == "provisional_discovery"
    ):
        if args.expected_joint_selected_membership_sha256 is not None:
            parser.error(
                "provisional_discovery forbids a selected-membership anchor"
            )
        if (
            args.provisional_attempt_id is None
            or args.provisional_ledger_entry is None
        ):
            parser.error(
                "provisional_discovery requires --provisional-attempt-id and "
                "--provisional-ledger-entry"
            )
        try:
            attempt_id = _safe_instance_id(
                args.provisional_attempt_id,
                "provisional_attempt_id",
            )
        except ValueError as error:
            parser.error(str(error))
        if len(attempt_id) > 128:
            parser.error("--provisional-attempt-id exceeds 128 characters")
    if (
        args.selection_method == "joint_core_milp"
        and args.joint_assignment_mode == "externally_pinned_replay"
    ):
        if args.expected_joint_selected_membership_sha256 is None:
            parser.error(
                "externally_pinned_replay requires "
                "--expected-joint-selected-membership-sha256"
            )
        if (
            args.provisional_attempt_id is not None
            or args.provisional_ledger_entry is not None
        ):
            parser.error(
                "externally_pinned_replay forbids provisional-attempt options"
            )
        if (
            args.pinned_replay_id is None
            or args.source_provisional_attempt_id is None
        ):
            parser.error(
                "externally_pinned_replay requires --pinned-replay-id and "
                "--source-provisional-attempt-id"
            )
        for name in ("pinned_replay_id", "source_provisional_attempt_id"):
            try:
                value = _safe_instance_id(getattr(args, name), name)
            except ValueError as error:
                parser.error(str(error))
            if len(value) > 128:
                parser.error(f"--{name.replace('_', '-')} exceeds 128 characters")
        if args.pinned_replay_id == args.source_provisional_attempt_id:
            parser.error(
                "--pinned-replay-id must differ from "
                "--source-provisional-attempt-id"
            )
    if (
        args.selection_method == "joint_core_milp"
        and args.joint_assignment_mode == "provisional_discovery"
        and (
            args.pinned_replay_id is not None
            or args.source_provisional_attempt_id is not None
        )
    ):
        parser.error("provisional_discovery forbids pinned-replay options")
    if args.selection_method != "joint_core_milp" and any(
        value is not None
        for value in (
            args.joint_assignment_mode,
            args.provisional_attempt_id,
            args.provisional_ledger_entry,
            args.expected_joint_selected_membership_sha256,
            args.pinned_replay_id,
            args.source_provisional_attempt_id,
        )
    ):
        parser.error("Joint assignment options require --selection-method joint_core_milp")
    if args.fold_count != 3:
        parser.error("The nested protocol requires exactly three folds")
    if args.workers < 1 or args.chunk_size < 1:
        parser.error("--workers and --chunk-size must be positive")
    if args.selection_seed < 0:
        parser.error("--selection-seed must be non-negative")
    reviewed_bounds = {
        "joint_maximum_binary_variables": (
            REVIEWED_JOINT_MAXIMUM_BINARY_VARIABLES
        ),
        "joint_maximum_linear_constraints": (
            REVIEWED_JOINT_MAXIMUM_LINEAR_CONSTRAINTS
        ),
        "joint_maximum_nonzero_coefficients": (
            REVIEWED_JOINT_MAXIMUM_NONZERO_COEFFICIENTS
        ),
    }
    for name, reviewed_maximum in reviewed_bounds.items():
        value = getattr(args, name)
        if value <= 0 or value > reviewed_maximum:
            parser.error(
                f"--{name.replace('_', '-')} must be in [1, {reviewed_maximum}]"
            )
    if (
        args.joint_time_limit_seconds <= 0
        or args.joint_time_limit_seconds > REVIEWED_JOINT_TIME_LIMIT_SECONDS
    ):
        parser.error(
            "--joint-time-limit-seconds must be in "
            f"(0, {REVIEWED_JOINT_TIME_LIMIT_SECONDS}]"
        )
    return args


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Build the exact graph and nested fold construction candidate."""
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Create-only target already exists: {output}")
    selection_method = getattr(
        args,
        "selection_method",
        DEFAULT_SELECTION_METHOD,
    )
    if selection_method not in SELECTION_METHODS:
        raise ValueError(f"Unsupported selection method: {selection_method!r}")
    joint_assignment_mode = getattr(args, "joint_assignment_mode", None)
    provisional_ledger_entry: Path | None = None
    provisional_started_entry: Path | None = None
    provisional_started_payload: dict[str, Any] | None = None
    provisional_attempt_id: str | None = None
    pinned_replay_id: str | None = None
    source_provisional_attempt_id: str | None = None
    if selection_method == "joint_core_milp":
        if joint_assignment_mode not in JOINT_ASSIGNMENT_MODES:
            raise ValueError(
                "joint_core_milp requires an explicit joint_assignment_mode"
            )
        expected_policy = validate_sha256(
            getattr(args, "expected_joint_policy_sha256", None),
            "expected_joint_policy_sha256",
        )
        if expected_policy != LOCKED_JOINT_POLICY_SHA256:
            raise ValueError(
                "The externally supplied joint-policy digest differs from the "
                "repository's locked exact 3x3 policy"
            )
        if joint_assignment_mode == "provisional_discovery":
            if (
                getattr(args, "pinned_replay_id", None) is not None
                or getattr(args, "source_provisional_attempt_id", None)
                is not None
            ):
                raise ValueError(
                    "provisional_discovery forbids pinned-replay options"
                )
            if (
                getattr(
                    args,
                    "expected_joint_selected_membership_sha256",
                    None,
                )
                is not None
            ):
                raise ValueError(
                    "provisional_discovery forbids a selected-membership anchor"
                )
            provisional_attempt_id = _safe_instance_id(
                getattr(args, "provisional_attempt_id", None),
                "provisional_attempt_id",
            )
            if len(provisional_attempt_id) > 128:
                raise ValueError("provisional_attempt_id exceeds 128 characters")
            provisional_value = getattr(
                args,
                "provisional_ledger_entry",
                None,
            )
            if provisional_value is None:
                raise ValueError(
                    "provisional_discovery requires a create-only ledger entry"
                )
            provisional_ledger_entry = Path(
                provisional_value
            ).expanduser().resolve()
            if provisional_ledger_entry == output:
                raise ValueError(
                    "Provisional ledger entry and assignment output must differ"
                )
            if provisional_ledger_entry.exists():
                raise FileExistsError(
                    "Create-only provisional ledger entry already exists: "
                    f"{provisional_ledger_entry}"
                )
            provisional_started_entry = provisional_ledger_entry.with_name(
                f"{provisional_ledger_entry.name}.started.json"
            )
            if provisional_started_entry == output:
                raise ValueError(
                    "Provisional start ledger and assignment output must differ"
                )
            if provisional_started_entry.exists():
                raise FileExistsError(
                    "Create-only provisional start ledger already exists: "
                    f"{provisional_started_entry}"
                )
        else:
            if any(
                value is not None
                for value in (
                    getattr(args, "provisional_attempt_id", None),
                    getattr(args, "provisional_ledger_entry", None),
                )
            ):
                raise ValueError(
                    "externally_pinned_replay forbids provisional-attempt options"
                )
            pinned_replay_id = _safe_instance_id(
                getattr(args, "pinned_replay_id", None),
                "pinned_replay_id",
            )
            source_provisional_attempt_id = _safe_instance_id(
                getattr(args, "source_provisional_attempt_id", None),
                "source_provisional_attempt_id",
            )
            if (
                len(pinned_replay_id) > 128
                or len(source_provisional_attempt_id) > 128
            ):
                raise ValueError("Pinned replay identifiers exceed 128 characters")
            if pinned_replay_id == source_provisional_attempt_id:
                raise ValueError(
                    "pinned_replay_id must differ from "
                    "source_provisional_attempt_id"
                )
            validate_sha256(
                getattr(
                    args,
                    "expected_joint_selected_membership_sha256",
                    None,
                ),
                "expected_joint_selected_membership_sha256",
            )
    verified_references = {
        "train_authority_allowlist": verify_file(
            args.train_authority_allowlist,
            args.train_authority_allowlist_sha256,
            "physical train-only authority allowlist",
        ),
        "train_canonical_sequence_manifest": verify_file(
            args.train_canonical_manifest,
            args.train_canonical_manifest_sha256,
            "physical train-only canonical sequence manifest",
        ),
    }
    verified_sources = {
        "builder": verify_file(
            Path(__file__),
            args.expected_builder_source_sha256,
            "nested-neighborhood builder source",
        )
    }
    joint_module: Any | None = None
    if selection_method == "joint_core_milp":
        expected_joint_source_sha256 = getattr(
            args,
            "expected_joint_milp_source_sha256",
            None,
        )
        if expected_joint_source_sha256 is None:
            raise ValueError(
                "expected_joint_milp_source_sha256 is required for joint_core_milp"
            )
        joint_source = getattr(args, "joint_milp_source", None)
        if joint_source is None:
            raise ValueError("joint_milp_source is required for joint_core_milp")
        verified_sources["joint_core_milp"] = verify_file(
            Path(joint_source),
            expected_joint_source_sha256,
            "joint core-feasibility MILP source",
        )
        joint_module = load_verified_joint_module(
            verified_sources["joint_core_milp"]
        )
        if (
            joint_module.LOCKED_JOINT_POLICY_SHA256
            != LOCKED_JOINT_POLICY_SHA256
        ):
            raise ValueError(
                "Verified joint-MILP source does not expose the independently "
                "locked policy digest"
            )
    train_ids, authority = load_train_authority_allowlist(
        verified_references["train_authority_allowlist"].payload
    )
    records, firewall = load_train_only_records(
        verified_references["train_canonical_sequence_manifest"].payload,
        train_ids,
    )
    references = {
        key: value.reference for key, value in verified_references.items()
    }
    sources = {
        key: value.reference for key, value in verified_sources.items()
    }
    antibodies, antigens, pairs, inventory = canonicalize_inventory(records)
    if joint_assignment_mode == "provisional_discovery":
        if (
            provisional_started_entry is None
            or provisional_attempt_id is None
        ):
            raise RuntimeError("Provisional start-ledger binding was not initialized")
        provisional_started_payload = {
            "format_version": PROVISIONAL_LEDGER_FORMAT_VERSION,
            "ledger_entry_kind": "attempt_started",
            "ledger_semantics": (
                "one_create_only_start_and_one_create_only_outcome_per_attempt"
            ),
            "provisional_attempt_id": provisional_attempt_id,
            "attempt_status": "started_no_outcome_yet",
            "intended_assignment_output_path": str(output),
            "intended_outcome_ledger_path": str(provisional_ledger_entry),
            "inputs": references,
            "source_files": sources,
            "authority": authority,
            "canonical_inventory": inventory,
            "expected_joint_policy_sha256": expected_policy,
            "externally_pinned_assignment_identity": False,
            "eligible_for_independent_pinned_replay_audit": False,
            "split_domain_gate0_requirement_satisfied": False,
            "authorized_for_training": False,
            "authorized_for_model_selection": False,
            "authorized_for_nested_model_selection": False,
        }
        provisional_started_payload["scientific_payload_sha256"] = (
            canonical_payload_sha256(provisional_started_payload)
        )
        atomic_create_json(
            provisional_started_entry,
            provisional_started_payload,
        )

    edge_sets: dict[str, set[tuple[str, str]]] = {}
    graph_receipts: dict[str, dict[str, Any]] = {}
    for view in REQUIRED_VIEWS:
        edge_sets[view], graph_receipts[view] = build_exact_view_graph(
            _view_inventory(antibodies, antigens, view),
            workers=args.workers,
            chunk_size=args.chunk_size,
        )
    antibody_edges = set().union(
        edge_sets["vh"],
        edge_sets["vl"],
        edge_sets["concatenated_antibody"],
    )
    antibody_graph = adjacency(antibodies, antibody_edges)
    antigen_graph = adjacency(antigens, edge_sets["antigen"])

    if selection_method == "joint_core_milp":
        if joint_module is None:
            raise RuntimeError("Verified joint-MILP module was not loaded")
        resource_limits = joint_module.ResourceLimits(
            maximum_binary_variables=getattr(
                args,
                "joint_maximum_binary_variables",
                REVIEWED_JOINT_MAXIMUM_BINARY_VARIABLES,
            ),
            maximum_linear_constraints=getattr(
                args,
                "joint_maximum_linear_constraints",
                REVIEWED_JOINT_MAXIMUM_LINEAR_CONSTRAINTS,
            ),
            maximum_nonzero_coefficients=getattr(
                args,
                "joint_maximum_nonzero_coefficients",
                REVIEWED_JOINT_MAXIMUM_NONZERO_COEFFICIENTS,
            ),
        )
        reviewed_limits = joint_module.ResourceLimits(
            maximum_binary_variables=REVIEWED_JOINT_MAXIMUM_BINARY_VARIABLES,
            maximum_linear_constraints=(
                REVIEWED_JOINT_MAXIMUM_LINEAR_CONSTRAINTS
            ),
            maximum_nonzero_coefficients=(
                REVIEWED_JOINT_MAXIMUM_NONZERO_COEFFICIENTS
            ),
        )
        for name in (
            "maximum_binary_variables",
            "maximum_linear_constraints",
            "maximum_nonzero_coefficients",
        ):
            value = getattr(resource_limits, name)
            reviewed_maximum = getattr(reviewed_limits, name)
            if value <= 0 or value > reviewed_maximum:
                raise ValueError(
                    f"Joint resource limit {name}={value} is outside the "
                    f"reviewed range [1, {reviewed_maximum}]"
                )
        joint_time_limit_seconds = getattr(
            args,
            "joint_time_limit_seconds",
            REVIEWED_JOINT_TIME_LIMIT_SECONDS,
        )
        if (
            joint_time_limit_seconds <= 0
            or joint_time_limit_seconds > REVIEWED_JOINT_TIME_LIMIT_SECONDS
        ):
            raise ValueError(
                "Joint MILP time limit is outside the reviewed range "
                f"(0, {REVIEWED_JOINT_TIME_LIMIT_SECONDS}]"
            )
        outer, selector_receipt = build_joint_nested_levels(
            pairs,
            antibody_graph,
            antigen_graph,
            resource_limits=resource_limits,
            time_limit_seconds=joint_time_limit_seconds,
            joint_module=joint_module,
        )
        expected_policy_sha256 = getattr(
            args,
            "expected_joint_policy_sha256",
            None,
        )
        if expected_policy_sha256 is None:
            raise ValueError(
                "expected_joint_policy_sha256 is required for joint_core_milp"
            )
        expected_policy_sha256 = validate_sha256(
            expected_policy_sha256,
            "expected_joint_policy_sha256",
        )
        observed_policy_sha256 = validate_sha256(
            selector_receipt.get("exact_joint_policy_sha256"),
            "selector exact_joint_policy_sha256",
        )
        if observed_policy_sha256 != expected_policy_sha256:
            raise ValueError(
                "Joint policy SHA-256 differs from the externally frozen "
                f"policy: expected={expected_policy_sha256}, "
                f"observed={observed_policy_sha256}"
            )
        observed_selected_membership = validate_sha256(
            selector_receipt.get("selected_binary_membership_sha256"),
            "selector selected_binary_membership_sha256",
        )
        if joint_assignment_mode == "externally_pinned_replay":
            expected_selected_membership = validate_sha256(
                getattr(
                    args,
                    "expected_joint_selected_membership_sha256",
                    None,
                ),
                "expected_joint_selected_membership_sha256",
            )
            if observed_selected_membership != expected_selected_membership:
                raise ValueError(
                    "Joint assignment identity differs from the externally "
                    "frozen selected-membership digest: "
                    f"expected={expected_selected_membership}, "
                    f"observed={observed_selected_membership}"
                )
            selector_receipt.update(
                {
                    "assignment_identity_policy": (
                        "externally_pinned_selected_binary_membership_sha256"
                    ),
                    "externally_pinned_selected_binary_membership_sha256": (
                        expected_selected_membership
                    ),
                    "accepted_assignment_matches_external_identity_anchor": True,
                    "constant_zero_objective_assignment_is_solver_canonical": (
                        False
                    ),
                }
            )
        else:
            selector_receipt.update(
                {
                    "assignment_identity_policy": (
                        "provisional_unpinned_candidate_for_external_review"
                    ),
                    "externally_pinned_selected_binary_membership_sha256": None,
                    "accepted_assignment_matches_external_identity_anchor": False,
                    "constant_zero_objective_assignment_is_solver_canonical": (
                        False
                    ),
                }
            )
    else:
        root_seed = f"direct80-v2:{args.selection_seed}"
        outer, outer_training_sets = build_level(
            pairs,
            antibody_graph,
            antigen_graph,
            outer_policy(len(pairs)),
            fold_count=args.fold_count,
            seed_label=f"{root_seed}:outer",
        )
        for outer_index, outer_training in enumerate(outer_training_sets):
            if not outer_training:
                outer["folds"][outer_index]["inner_level"] = {
                    "construction_skipped": True,
                    "reason": "outer_training_universe_is_empty",
                    "all_level_hard_gates_passed": False,
                }
                continue
            inner, _ = build_level(
                outer_training,
                antibody_graph,
                antigen_graph,
                inner_policy(len(outer_training)),
                fold_count=args.fold_count,
                seed_label=f"{root_seed}:outer-{outer_index}:inner",
            )
            outer["folds"][outer_index]["inner_level"] = inner
        selector_receipt = {
            "implementation": (
                "deterministic_round_robin_static_direct_neighborhood_cost"
            ),
            "exact_nested_milp_used": False,
            "scipy_1_15_milp_assessment": (
                "capable of a fixed-level binary MILP, but a sequential "
                "outer-then-inner solve is not an exact formulation of the "
                "complete dependent 3+9 nested-feasibility problem"
            ),
            "replacement_blocker": (
                "a joint nested formulation or a complete outer-solution "
                "enumeration with proven zero-gap inner feasibility is "
                "required before exact-MILP claims are valid"
            ),
            "failure_semantics": (
                "all post-construction hard gates remain fail-closed"
            ),
        }

    all_inner_passed = all(
        fold["inner_level"].get("all_level_hard_gates_passed", False)
        for fold in outer["folds"]
    )
    construction_gates = {
        "authority_train_only_firewall_passed": (
            firewall["input_contract"]
            == "externally_anchored_physical_train_only_v1"
            and firewall["exact_sequence_only_schema_verified"]
            and firewall["one_record_per_physical_line_verified"]
            and firewall["validation_or_test_memberships_present"] == 0
            and firewall["validation_or_test_rows_present"] == 0
            and firewall[
                "validation_or_test_sequence_fields_parsed"
            ]
            == 0
            and firewall[
                "validation_or_test_labels_predictions_or_scores_read"
            ]
            == 0
        ),
        "all_four_exact_graphs_complete": (
            set(graph_receipts) == set(REQUIRED_VIEWS)
            and all(
                receipt[
                    "alignment_plus_safe_prune_equals_all_distinct_sequence_pairs"
                ]
                and receipt["every_aligned_pair_evaluated_in_both_orientations"]
                and receipt["symmetric_either_orientation_threshold_edge_rule"]
                for receipt in graph_receipts.values()
            )
        ),
        "outer_level_passed": outer["all_level_hard_gates_passed"],
        "all_inner_levels_passed": all_inner_passed,
        "exact_joint_assignment_identity_externally_pinned": (
            selection_method == "joint_core_milp"
            and selector_receipt.get(
                "accepted_assignment_matches_external_identity_anchor",
                False,
            )
            is True
            and selector_receipt.get(
                "constant_zero_objective_assignment_is_solver_canonical",
                True,
            )
            is False
        ),
        "global_transitive_closure_not_used_as_fold_constraint": True,
    }
    core_feasibility_constraints_passed = all(
        value
        for key, value in construction_gates.items()
        if key != "exact_joint_assignment_identity_externally_pinned"
    )
    if selection_method == "joint_core_milp":
        construction_profile = (
            "exact_joint_core_feasibility_direct_neighborhood_nested_"
            "construction_v2"
        )
        relation_to_v0 = (
            "implements one exact joint binary core-feasibility model with "
            "full-authority homology-component minima for all three outer and "
            "nine dependent inner folds, but does not implement the draft "
            "lexicographic objective or its full diagnostic suite"
        )
        method_limitations = [
            (
                "The constant-zero MILP proves only feasibility of the encoded "
                "core gates; it is not a lexicographic or retention optimality "
                "proof and does not canonically choose among feasible vectors."
            ),
        ]
        if joint_assignment_mode == "provisional_discovery":
            method_limitations.append(
                "This provisional assignment is deliberately unpinned and is "
                "only a candidate for an external review-and-freeze decision."
            )
        else:
            method_limitations.append(
                "Assignment identity is accepted only because its complete "
                "selected-variable membership matches an external digest anchor."
            )
    else:
        construction_profile = (
            "deterministic_symmetric_direct_neighborhood_nested_construction_v2"
        )
        relation_to_v0 = (
            "implements the direct-neighborhood equations and nested firewall, "
            "but replaces the draft joint exact MILP and its full diagnostic suite "
            "with one fixed deterministic retention-greedy construction"
        )
        method_limitations = [
            (
                "This greedy construction is not an optimality proof for the "
                "joint MILP described in the V0 draft."
            ),
            (
                "A rigorous replacement must jointly ensure outer feasibility "
                "and all three dependent inner-level feasibilities at proven "
                "zero gap."
            ),
        ]
    if (
        selection_method == "joint_core_milp"
        and joint_assignment_mode == "provisional_discovery"
    ):
        artifact_format_version = PROVISIONAL_FORMAT_VERSION
        artifact_role = "provisional_unpinned_core_feasibility_discovery"
        stage = "completed_provisional_unpinned_core_feasibility_discovery"
        authorization_reason = (
            "This create-only provisional attempt has no external assignment "
            "identity anchor and is ineligible for independent replay audit."
        )
        required_next_gate = {
            "step": "external_review_freeze_and_separate_pinned_replay",
            "required_result": (
                "An external authority independently records the published "
                "selected-membership digest, then starts a separate replay."
            ),
        }
    elif selection_method == "joint_core_milp":
        artifact_format_version = FORMAT_VERSION
        artifact_role = "externally_pinned_core_feasibility_replay"
        stage = "completed_externally_pinned_core_feasibility_replay"
        authorization_reason = (
            "Independent core-feasibility precursor audit has not yet run; "
            "full split-domain Gate 0 also requires the remaining locked V0 "
            "canonical, lexicographic, diagnostic, and authority requirements."
        )
        required_next_gate = {
            "step": "independent_core_feasibility_precursor_audit",
            "required_result": (
                "zero threshold crossings, exact accounting for all 12 "
                "boundaries, independent inventory/graph/ledger-vector "
                "reconstruction, and matrix recomputation using the shared "
                "externally pinned formulation source"
            ),
        }
    else:
        artifact_format_version = FORMAT_VERSION
        artifact_role = "non_authoritative_greedy_construction"
        stage = "completed_non_authoritative_greedy_construction"
        authorization_reason = (
            "Greedy construction is ineligible for the exact joint "
            "core-feasibility precursor audit."
        )
        required_next_gate = {
            "step": "replace_with_joint_provisional_then_pinned_replay",
            "required_result": "an externally pinned exact joint assignment",
        }
    payload: dict[str, Any] = {
        "format_version": artifact_format_version,
        "artifact_role": artifact_role,
        "stage": stage,
        "construction_profile": construction_profile,
        "relation_to_v0_preregistration_draft": relation_to_v0,
        "claim_boundary": (
            "foldwise direct exact four-view 80/80 C3/S2-like development; "
            "not global S2, transitive lineage independence, or test evaluation"
        ),
        "authorized_for_training": False,
        "authorized_for_model_selection": False,
        "authorized_for_nested_model_selection": False,
        "split_domain_gate0_requirement_satisfied": False,
        "authorization_reason": authorization_reason,
        "core_feasibility_constraints_passed": (
            core_feasibility_constraints_passed
        ),
        "construction_invariants_passed": all(construction_gates.values()),
        "construction_gates": construction_gates,
        "required_next_gate": required_next_gate,
        "workflow_identity": {
            "joint_assignment_mode": joint_assignment_mode,
            "provisional_attempt_id": provisional_attempt_id,
            "pinned_replay_id": pinned_replay_id,
            "source_provisional_attempt_id": source_provisional_attempt_id,
        },
        "inputs": references,
        "source_files": sources,
        "authority": authority,
        "data_firewall": firewall,
        "canonical_inventory": inventory,
        "exact_direct_graphs": graph_receipts,
        "combined_antibody_direct_edge_count": len(antibody_edges),
        "combined_antibody_direct_edge_membership_sha256": membership_sha256(
            f"{left}\0{right}" for left, right in sorted(antibody_edges)
        ),
        "construction": {
            "fold_count": args.fold_count,
            "selection_method": selection_method,
            "joint_assignment_mode": joint_assignment_mode,
            "selection_seed": args.selection_seed,
            "selection_seed_used": selection_method == "deterministic_greedy",
            "selection_uses_only_sequence_graph_and_positive_pair_incidence": True,
            "model_predictions_scores_losses_embeddings_or_labels_used": False,
            "global_transitive_closure_used_as_fold_constraint": False,
            "selector": selector_receipt,
            "outer_level": outer,
        },
        "alignment_policy": {
            "algorithm": "exact global Needleman-Wunsch-family dynamic programming",
            "backend": "Biopython PairwiseAligner 1.83",
            "substitution_matrix": "BLOSUM62",
            "substitution_matrix_payload_sha256": (
                EXPECTED_BLOSUM62_PAYLOAD_SHA256
            ),
            "gap_open_penalty": GAP_OPEN_PENALTY,
            "gap_extend_penalty": GAP_EXTEND_PENALTY,
            "terminal_gaps_penalized_like_internal_gaps": True,
            "identity_threshold_fraction": [4, 5],
            "reciprocal_coverage_threshold_fraction": [4, 5],
            "threshold_equality_is_edge": True,
            "threshold_comparison_uses_integer_cross_products": True,
            "score_optimal_traceback_policy": (
                "first score-optimal alignment evaluated independently in "
                "both sequence orientations"
            ),
            "symmetric_edge_definition": (
                "an undirected edge exists when either orientation passes "
                "both inclusive 80/80 gates"
            ),
            "both_orientations_required_for_every_non_pruned_distinct_pair": True,
            "only_pair_prune": (
                "min(raw lengths)/max(raw lengths) strictly below 0.8"
            ),
        },
        "runtime": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "workers": args.workers,
            "chunk_size": args.chunk_size,
            "selection_method": selection_method,
            "joint_source_executed_from_verified_bytes": (
                selection_method == "joint_core_milp"
            ),
            "self_source_preexecution_integrity_established_by_this_process": (
                False
            ),
            "self_source_preexecution_integrity_requires_private_staged_launcher": (
                True
            ),
        },
        "limitations": [
            "This builder is score-free construction code, not a performance result.",
            "The independent auditor is non-authorizing; model execution requires a separate full Gate 0 authority after every locked requirement passes.",
            "This core-feasibility artifact does not implement the full V0 lexicographic, canonical-selection, or diagnostic contract.",
            "The builder cannot attest its own pre-execution bytes; production use requires the externally inventoried private-stage launcher.",
            "Direct-neighborhood exclusion does not separate entities connected only through a long homology path.",
            *method_limitations,
            "The remaining V0 representativeness and attainable-precision gates are not claimed by this artifact.",
            "Unobserved cross-pairs are retrieval decoys, not experimentally verified non-binders.",
        ],
    }
    payload["scientific_payload_sha256"] = canonical_payload_sha256(payload)
    if joint_assignment_mode == "provisional_discovery":
        if (
            provisional_ledger_entry is None
            or provisional_attempt_id is None
            or provisional_started_entry is None
            or provisional_started_payload is None
        ):
            raise RuntimeError("Provisional ledger binding was not initialized")
        provisional_assignment_file_sha256 = sha256_bytes(
            canonical_json_bytes(payload)
        )
        ledger_entry: dict[str, Any] = {
            "format_version": PROVISIONAL_LEDGER_FORMAT_VERSION,
            "ledger_entry_kind": "attempt_outcome",
            "ledger_semantics": (
                "one_create_only_start_and_one_create_only_outcome_per_attempt"
            ),
            "provisional_attempt_id": provisional_attempt_id,
            "start_ledger_entry_path": str(provisional_started_entry),
            "start_ledger_scientific_payload_sha256": (
                provisional_started_payload["scientific_payload_sha256"]
            ),
            "attempt_status": "candidate_published",
            "provisional_assignment_output_path": str(output),
            "provisional_assignment_format_version": (
                PROVISIONAL_FORMAT_VERSION
            ),
            "provisional_assignment_file_sha256": (
                provisional_assignment_file_sha256
            ),
            "provisional_assignment_scientific_payload_sha256": payload[
                "scientific_payload_sha256"
            ],
            "selected_binary_membership_sha256": selector_receipt[
                "selected_binary_membership_sha256"
            ],
            "matrix_sha256": selector_receipt["matrix_sha256"],
            "exact_joint_policy_sha256": selector_receipt[
                "exact_joint_policy_sha256"
            ],
            "solver_evidence": {
                key: selector_receipt[key]
                for key in (
                    "solver",
                    "solver_status",
                    "solver_termination",
                    "solver_message",
                    "solver_reported_success",
                    "solver_solution_present",
                    "solver_certified_optimal_for_constant_zero_objective",
                    "solver_reported_mip_gap",
                    "solver_reported_mip_gap_is_finite_zero",
                    "maximum_integrality_error",
                    "accepted_integrality_tolerance",
                    "exact_integer_row_audit_passed",
                    "certified_core_feasibility_assignment",
                )
            },
            "inputs": references,
            "source_files": sources,
            "core_feasibility_constraints_passed": (
                core_feasibility_constraints_passed
            ),
            "externally_pinned_assignment_identity": False,
            "eligible_for_independent_pinned_replay_audit": False,
            "split_domain_gate0_requirement_satisfied": False,
            "authorized_for_training": False,
            "authorized_for_model_selection": False,
            "authorized_for_nested_model_selection": False,
            "required_next_step": (
                "external_review_freeze_then_separate_pinned_replay"
            ),
        }
        ledger_entry["scientific_payload_sha256"] = canonical_payload_sha256(
            ledger_entry
        )
        # Publish the candidate before claiming a successful outcome. A failure
        # before this write leaves only the conservative start ledger; a
        # failure after it leaves an inspectable candidate plus the start
        # ledger, never an immutable false "candidate_published" outcome.
        atomic_create_json(output, payload)
        atomic_create_json(provisional_ledger_entry, ledger_entry)
    else:
        atomic_create_json(output, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    """Run the command-line entrypoint."""
    args = parse_args(argv)
    payload = run(args)
    print(
        json.dumps(
            {
                "output": str(args.output.expanduser().resolve()),
                "scientific_payload_sha256": payload[
                    "scientific_payload_sha256"
                ],
                "construction_invariants_passed": payload[
                    "construction_invariants_passed"
                ],
                "artifact_role": payload["artifact_role"],
                "selected_binary_membership_sha256": payload[
                    "construction"
                ]["selector"].get("selected_binary_membership_sha256"),
                "provisional_ledger_entry": (
                    str(args.provisional_ledger_entry.expanduser().resolve())
                    if getattr(args, "provisional_ledger_entry", None)
                    is not None
                    else None
                ),
                "authorized_for_nested_model_selection": False,
            },
            sort_keys=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
