#!/usr/bin/env python3
"""Independently audit nested train-only direct-neighborhood folds.

This verifier does not import the builder. It accepts only an externally pinned
physical train-only authority allowlist and an exact four-column physical
train-only sequence projection. Full split files, sealed memberships, mixed
manifests, extra outcome columns, multiline CSV records, symlinks, and
non-regular inputs are rejected. It reconstructs framed entities and exact
source-row groups, verifies every outer and inner
train/validation/quarantine partition, rebuilds all four direct graphs and the
matrix emitted by the same externally pinned joint-formulation implementation
used by the builder, reconstructs the complete ledger-derived binary vector,
and exhaustively aligns every actual train-validation boundary in VH,
non-empty VL, concatenated-antibody, and antigen views. Inventory, graphs,
ledgers, and exact row evaluation are reconstructed independently; the
formulation implementation itself is shared and is not a second independent
MILP implementation.

The create-only receipt is only a core-feasibility precursor when every
independent invariant and every zero-crossing boundary check passes. It cannot
satisfy the split-domain Gate 0 requirement, authorize training, or authorize
model selection by itself.
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


FORMAT_VERSION = "train_only_nested_direct80_four_view_independent_audit_v5"
ASSIGNMENT_FORMAT_VERSION = (
    "train_only_nested_direct80_four_view_neighborhood_folds_v5"
)
TRAIN_AUTHORITY_FORMAT_VERSION = "gate0_physical_train_only_allowlist_v1"
TRAIN_MANIFEST_COLUMNS = ("INSTANCE", "vh_seq", "vl_seq", "ag_seq")
FILE_REFERENCE_FIELDS = frozenset(
    {
        "path",
        "sha256",
        "size_bytes",
        "stat_fingerprint_before",
        "stat_fingerprint_after",
        "single_verified_fd_read",
        "symlink_followed",
        "secure_open_capabilities",
    }
)
ASSIGNMENT_TOP_LEVEL_FIELDS = frozenset(
    {
        "format_version",
        "artifact_role",
        "stage",
        "construction_profile",
        "relation_to_v0_preregistration_draft",
        "claim_boundary",
        "authorized_for_training",
        "authorized_for_model_selection",
        "authorized_for_nested_model_selection",
        "split_domain_gate0_requirement_satisfied",
        "authorization_reason",
        "core_feasibility_constraints_passed",
        "construction_invariants_passed",
        "construction_gates",
        "required_next_gate",
        "workflow_identity",
        "inputs",
        "source_files",
        "authority",
        "data_firewall",
        "canonical_inventory",
        "exact_direct_graphs",
        "combined_antibody_direct_edge_count",
        "combined_antibody_direct_edge_membership_sha256",
        "construction",
        "alignment_policy",
        "runtime",
        "limitations",
        "scientific_payload_sha256",
    }
)
CONSTRUCTION_FIELDS = frozenset(
    {
        "fold_count",
        "selection_method",
        "joint_assignment_mode",
        "selection_seed",
        "selection_seed_used",
        "selection_uses_only_sequence_graph_and_positive_pair_incidence",
        "model_predictions_scores_losses_embeddings_or_labels_used",
        "global_transitive_closure_used_as_fold_constraint",
        "selector",
        "outer_level",
    }
)
LEVEL_FIELDS = frozenset(
    {
        "universe",
        "policy",
        "selection",
        "folds",
        "count_relative_ranges",
        "level_hard_gates",
        "all_level_hard_gates_passed",
    }
)
FOLD_FIELDS = frozenset(
    {
        "fold_index",
        "validation_antibody_ids",
        "validation_antigen_ids",
        "validation_antibody_component_ids",
        "validation_antigen_component_ids",
        "closed_direct_neighborhood_antibody_ids",
        "closed_direct_neighborhood_antigen_ids",
        "training_eligible_antibody_ids",
        "training_eligible_antigen_ids",
        "model_antibody_allowlist",
        "model_antigen_allowlist",
        "training_pairs",
        "validation_pairs",
        "quarantined_pairs",
        "counts",
        "membership_hashes",
        "hard_gates",
        "all_fold_hard_gates_passed",
    }
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
LOCKED_JOINT_POLICY_SHA256 = (
    "5ddf8b4efef6ae7aed29fd529e613ee12f3b4d0caeddc23d1a320f73bca344fd"
)
REVIEWED_JOINT_MAXIMUM_BINARY_VARIABLES = 200_000
REVIEWED_JOINT_MAXIMUM_LINEAR_CONSTRAINTS = 3_000_000
REVIEWED_JOINT_MAXIMUM_NONZERO_COEFFICIENTS = 12_000_000
REVIEWED_JOINT_TIME_LIMIT_SECONDS = 43_200.0


@dataclass(frozen=True)
class IndependentRow:
    """One independently reconstructed authority-train source row."""

    instance_id: str
    heavy: str
    light: str
    antigen: str


@dataclass(frozen=True)
class IndependentPair:
    """One independently reconstructed exact pair group."""

    pair_id: str
    antibody_id: str
    antigen_id: str
    source_instance_ids: tuple[str, ...]


@dataclass(frozen=True)
class Boundary:
    """One outer or inner train-validation boundary."""

    label: str
    training_pair_ids: tuple[str, ...]
    validation_pair_ids: tuple[str, ...]


@dataclass(frozen=True)
class VerifiedInput:
    """Bytes and acquisition evidence read from one verified no-follow fd."""

    reference: Mapping[str, Any]
    payload: bytes


_WORKER_ALIGNER: object | None = None


def sha256_bytes(payload: bytes) -> str:
    """Return the SHA-256 digest of bytes."""
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
    """Return stat fields that bind an inode before and after acquisition."""
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
    """Acquire one file through component-wise no-follow traversal."""
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
        raise ValueError(
            f"{label} cannot be securely acquired without following a symlink: "
            f"{path}"
        ) from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_descriptor)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Hash bytes from one component-wise no-follow regular-file fd."""
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
    """Hash one scientific payload canonically."""
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return sha256_bytes(encoded)


def membership_sha256(values: Iterable[str]) -> str:
    """Hash sorted unique membership values."""
    encoded = "".join(f"{value}\n" for value in sorted(set(values))).encode("utf-8")
    return sha256_bytes(encoded)


def _reject_duplicate_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    """Construct a mapping while rejecting duplicate JSON keys."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    """Reject non-standard JSON constants."""
    raise ValueError(f"JSON contains non-finite value {value}")


def _validate_finite_json(value: Any, label: str) -> None:
    """Reject non-finite floats recursively."""
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{label} contains a non-finite number")
    if isinstance(value, Mapping):
        for key, child in value.items():
            _validate_finite_json(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_finite_json(child, f"{label}[{index}]")


def strict_json_object(payload_bytes: bytes, label: str) -> dict[str, Any]:
    """Load one strict JSON object from already authenticated bytes."""
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
    module_name = f"_independently_verified_nested_joint_milp_{digest}"
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
    """Publish a JSON receipt atomically without clobbering."""
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


def _safe_identifier(value: Any, label: str) -> str:
    """Validate one exact string identifier."""
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    identifier = value
    if not identifier or identifier != identifier.strip():
        raise ValueError(f"{label} is empty or has surrounding whitespace")
    if any(character in identifier for character in (",", '"', "\r", "\n")):
        raise ValueError(f"{label} contains a forbidden CSV character")
    return identifier


def normalize_amino_acids(
    value: str, label: str, *, permit_empty: bool = False
) -> str:
    """Independently normalize one amino-acid sequence."""
    normalized = "".join(
        character for character in str(value) if not character.isspace()
    ).upper()
    try:
        normalized.encode("ascii")
    except UnicodeEncodeError as error:
        raise ValueError(f"{label} is not ASCII") from error
    if not normalized and permit_empty:
        return ""
    if not normalized:
        raise ValueError(f"{label} is empty")
    invalid = sorted(set(normalized) - VALID_AMINO_ACIDS)
    if invalid:
        raise ValueError(f"{label} contains invalid residue symbols: {invalid}")
    return normalized


def read_train_authority_allowlist(
    payload_bytes: bytes,
) -> tuple[set[str], dict[str, Any]]:
    """Read an exact externally anchored physical train-only allowlist."""
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
        _safe_identifier(value, "train-only authority identifier")
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


def reconstruct_train_rows(
    payload_bytes: bytes,
    train_ids: set[str],
) -> tuple[dict[str, IndependentRow], dict[str, Any]]:
    """Reconstruct an exact sequence-only manifest containing train rows only."""
    parsed_rows = _single_physical_csv_rows(
        payload_bytes,
        "physical train-only canonical manifest",
    )
    header = parsed_rows[0]
    if tuple(header) != TRAIN_MANIFEST_COLUMNS:
        raise ValueError(
            "Train-only canonical manifest must contain exactly "
            "INSTANCE,vh_seq,vl_seq,ag_seq with no duplicate or extra columns"
        )
    records: dict[str, IndependentRow] = {}
    for line_number, fields in enumerate(parsed_rows[1:], start=2):
        if len(fields) != len(TRAIN_MANIFEST_COLUMNS):
            raise ValueError(
                f"Malformed physical train-only row at line {line_number}"
            )
        instance_id = _safe_identifier(
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
        records[instance_id] = IndependentRow(
            instance_id=instance_id,
            heavy=normalize_amino_acids(fields[1], f"{instance_id}:VH"),
            light=normalize_amino_acids(
                fields[2],
                f"{instance_id}:VL",
                permit_empty=True,
            ),
            antigen=normalize_amino_acids(
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


def _length_frame(sequence: str) -> bytes:
    """Independently frame one ASCII sequence by its byte length."""
    encoded = sequence.encode("ascii")
    return str(len(encoded)).encode("ascii") + b":" + encoded


def independent_antibody_id(heavy: str, light: str) -> str:
    """Reproduce the framed antibody identifier independently."""
    return sha256_bytes(
        b"antibody-v1\0" + _length_frame(heavy) + _length_frame(light)
    )


def independent_antigen_id(sequence: str) -> str:
    """Reproduce the framed antigen identifier independently."""
    return sha256_bytes(b"antigen-v1\0" + _length_frame(sequence))


def independent_pair_id(antibody_identifier: str, antigen_identifier: str) -> str:
    """Reproduce the framed exact-pair identifier independently."""
    return sha256_bytes(
        b"pair-v1\0"
        + antibody_identifier.encode("ascii")
        + b"\0"
        + antigen_identifier.encode("ascii")
    )


def reconstruct_entities(
    rows: Mapping[str, IndependentRow],
) -> tuple[
    dict[str, tuple[str, str]],
    dict[str, str],
    dict[str, IndependentPair],
    dict[str, Any],
]:
    """Independently reconstruct entities and indivisible source groups."""
    antibodies: dict[str, tuple[str, str]] = {}
    antigens: dict[str, str] = {}
    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
    concatenations: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for instance_id in sorted(rows):
        row = rows[instance_id]
        a_id = independent_antibody_id(row.heavy, row.light)
        g_id = independent_antigen_id(row.antigen)
        previous_a = antibodies.setdefault(a_id, (row.heavy, row.light))
        previous_g = antigens.setdefault(g_id, row.antigen)
        if previous_a != (row.heavy, row.light) or previous_g != row.antigen:
            raise RuntimeError("Independent canonical digest collision detected")
        grouped[(a_id, g_id)].append(instance_id)
        concatenations[row.heavy + row.light].add((row.heavy, row.light))
    pairs: dict[str, IndependentPair] = {}
    for (a_id, g_id), source_ids in sorted(grouped.items()):
        p_id = independent_pair_id(a_id, g_id)
        if p_id in pairs:
            raise RuntimeError("Independent exact-pair digest collision detected")
        pairs[p_id] = IndependentPair(
            pair_id=p_id,
            antibody_id=a_id,
            antigen_id=g_id,
            source_instance_ids=tuple(sorted(source_ids)),
        )
    collisions = [
        sha256_bytes(sequence.encode("ascii"))
        for sequence, tuples in concatenations.items()
        if len(tuples) > 1
    ]
    return antibodies, antigens, pairs, {
        "authority_train_source_row_count": len(rows),
        "unique_antibody_entity_count": len(antibodies),
        "unique_antigen_entity_count": len(antigens),
        "unique_exact_pair_count": len(pairs),
        "antibody_membership_sha256": membership_sha256(antibodies),
        "antigen_membership_sha256": membership_sha256(antigens),
        "exact_pair_membership_sha256": membership_sha256(pairs),
        "raw_concatenation_collision_count": len(collisions),
        "raw_concatenation_collision_sequence_digest_sha256s": sorted(collisions),
    }


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    """Require one JSON mapping."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _require_exact_schema(
    value: Mapping[str, Any],
    expected: Iterable[str],
    label: str,
) -> None:
    """Reject every missing or unknown field in a versioned JSON object."""
    observed = set(value)
    required = set(expected)
    if observed != required:
        raise ValueError(
            f"{label} schema mismatch: "
            f"missing={sorted(required - observed)}, "
            f"unexpected={sorted(observed - required)}"
        )


def _list(value: Any, label: str) -> list[Any]:
    """Require one JSON list."""
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a JSON list")
    return value


def _integer(value: Any, label: str) -> int:
    """Require one non-boolean integer."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be a JSON integer")
    return value


def _number(value: Any, label: str) -> float:
    """Require one finite non-boolean number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a JSON number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _boolean(value: Any, label: str) -> bool:
    """Require one strict boolean."""
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a JSON boolean")
    return value


def _exact_json_equal(left: Any, right: Any) -> bool:
    """Compare canonical JSON bytes so booleans cannot alias integers."""
    def encode(value: Any) -> bytes:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")

    return encode(left) == encode(right)


def _canonical_sha256_list(value: Any, label: str) -> tuple[str, ...]:
    """Require a sorted duplicate-free JSON list of digest strings."""
    parsed = tuple(
        validate_sha256(item, f"{label}[{index}]")
        for index, item in enumerate(_list(value, label))
    )
    if list(parsed) != sorted(set(parsed)):
        raise ValueError(f"{label} must be a canonical sorted unique list")
    return parsed


def verify_scientific_digest(assignment: Mapping[str, Any]) -> str:
    """Recompute and verify the assignment's embedded scientific digest."""
    claimed = validate_sha256(
        assignment.get("scientific_payload_sha256"),
        "assignment scientific_payload_sha256",
    )
    scientific = dict(assignment)
    scientific.pop("scientific_payload_sha256", None)
    observed = canonical_payload_sha256(scientific)
    if claimed != observed:
        raise ValueError(
            "Assignment scientific payload SHA-256 mismatch: "
            f"claimed={claimed}, observed={observed}"
        )
    return observed


def validate_replay_assignment_envelope(
    assignment: Mapping[str, Any],
) -> None:
    """Close every non-ledger replay container against the v5 schema."""
    expected_static_strings = {
        "construction_profile": (
            "exact_joint_core_feasibility_direct_neighborhood_nested_"
            "construction_v2"
        ),
        "relation_to_v0_preregistration_draft": (
            "implements one exact joint binary core-feasibility model with "
            "full-authority homology-component minima for all three outer and "
            "nine dependent inner folds, but does not implement the draft "
            "lexicographic objective or its full diagnostic suite"
        ),
        "claim_boundary": (
            "foldwise direct exact four-view 80/80 C3/S2-like development; "
            "not global S2, transitive lineage independence, or test evaluation"
        ),
        "authorization_reason": (
            "Independent core-feasibility precursor audit has not yet run; "
            "full split-domain Gate 0 also requires the remaining locked V0 "
            "canonical, lexicographic, diagnostic, and authority requirements."
        ),
    }
    for key, expected in expected_static_strings.items():
        if assignment.get(key) != expected:
            raise ValueError(f"Assignment {key} differs from the v5 replay schema")

    for key in (
        "core_feasibility_constraints_passed",
        "construction_invariants_passed",
    ):
        if _boolean(assignment.get(key), f"assignment.{key}") is not True:
            raise ValueError(f"Assignment {key} must be true for replay audit")

    construction_gates = _mapping(
        assignment.get("construction_gates"),
        "assignment construction_gates",
    )
    expected_gate_fields = {
        "authority_train_only_firewall_passed",
        "all_four_exact_graphs_complete",
        "outer_level_passed",
        "all_inner_levels_passed",
        "exact_joint_assignment_identity_externally_pinned",
        "global_transitive_closure_not_used_as_fold_constraint",
    }
    _require_exact_schema(
        construction_gates,
        expected_gate_fields,
        "assignment construction_gates",
    )
    for key in expected_gate_fields:
        if _boolean(
            construction_gates.get(key),
            f"assignment construction_gates.{key}",
        ) is not True:
            raise ValueError(f"Assignment construction gate {key} is not true")

    required_next_gate = _mapping(
        assignment.get("required_next_gate"),
        "assignment required_next_gate",
    )
    expected_next_gate = {
        "step": "independent_core_feasibility_precursor_audit",
        "required_result": (
            "zero threshold crossings, exact accounting for all 12 "
            "boundaries, independent inventory/graph/ledger-vector "
            "reconstruction, and matrix recomputation using the shared "
            "externally pinned formulation source"
        ),
    }
    if not _exact_json_equal(required_next_gate, expected_next_gate):
        raise ValueError("Assignment required_next_gate differs from v5 replay")

    expected_alignment_policy = {
        "algorithm": (
            "exact global Needleman-Wunsch-family dynamic programming"
        ),
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
    }
    if not _exact_json_equal(
        _mapping(
            assignment.get("alignment_policy"),
            "assignment alignment_policy",
        ),
        expected_alignment_policy,
    ):
        raise ValueError("Assignment alignment_policy differs from v5 replay")

    runtime = _mapping(assignment.get("runtime"), "assignment runtime")
    runtime_fields = {
        "python_version",
        "platform",
        "machine",
        "workers",
        "chunk_size",
        "selection_method",
        "joint_source_executed_from_verified_bytes",
        "self_source_preexecution_integrity_established_by_this_process",
        "self_source_preexecution_integrity_requires_private_staged_launcher",
    }
    _require_exact_schema(runtime, runtime_fields, "assignment runtime")
    if (
        runtime.get("python_version") != platform.python_version()
        or runtime.get("platform") != platform.platform()
        or runtime.get("machine") != platform.machine()
        or runtime.get("selection_method") != "joint_core_milp"
        or _integer(runtime.get("workers"), "assignment runtime.workers") < 1
        or _integer(runtime.get("chunk_size"), "assignment runtime.chunk_size")
        < 1
        or _boolean(
            runtime.get("joint_source_executed_from_verified_bytes"),
            "assignment runtime.joint_source_executed_from_verified_bytes",
        )
        is not True
        or _boolean(
            runtime.get(
                "self_source_preexecution_integrity_established_by_this_process"
            ),
            "assignment runtime self-preexecution integrity",
        )
        is not False
        or _boolean(
            runtime.get(
                "self_source_preexecution_integrity_requires_private_staged_launcher"
            ),
            "assignment runtime private-stage requirement",
        )
        is not True
    ):
        raise ValueError("Assignment runtime receipt differs from v5 replay")

    expected_limitations = [
        "This builder is score-free construction code, not a performance result.",
        (
            "The independent auditor is non-authorizing; model execution "
            "requires a separate full Gate 0 authority after every locked "
            "requirement passes."
        ),
        (
            "This core-feasibility artifact does not implement the full V0 "
            "lexicographic, canonical-selection, or diagnostic contract."
        ),
        (
            "The builder cannot attest its own pre-execution bytes; production "
            "use requires the externally inventoried private-stage launcher."
        ),
        (
            "Direct-neighborhood exclusion does not separate entities "
            "connected only through a long homology path."
        ),
        (
            "The constant-zero MILP proves only feasibility of the encoded "
            "core gates; it is not a lexicographic or retention optimality "
            "proof and does not canonically choose among feasible vectors."
        ),
        (
            "Assignment identity is accepted only because its complete "
            "selected-variable membership matches an external digest anchor."
        ),
        (
            "The remaining V0 representativeness and attainable-precision "
            "gates are not claimed by this artifact."
        ),
        (
            "Unobserved cross-pairs are retrieval decoys, not experimentally "
            "verified non-binders."
        ),
    ]
    if _list(assignment.get("limitations"), "assignment limitations") != (
        expected_limitations
    ):
        raise ValueError("Assignment limitations differ from v5 replay")


def verify_embedded_reference(
    value: Any,
    physical_reference: Mapping[str, Any],
    label: str,
) -> None:
    """Require one assignment-embedded file reference to match physical bytes."""
    embedded = _mapping(value, label)
    if set(embedded) != FILE_REFERENCE_FIELDS:
        raise ValueError(f"{label} has an invalid file-reference schema")
    if not _exact_json_equal(embedded, physical_reference):
        raise ValueError(f"{label} differs from the independently pinned file")


def _validate_pair_payload(
    value: Any,
    expected_pairs: Mapping[str, IndependentPair],
    label: str,
    *,
    quarantine: bool,
) -> tuple[str, list[str]]:
    """Validate one builder pair payload against independent reconstruction."""
    payload = _mapping(value, label)
    required = {
        "pair_id",
        "antibody_id",
        "antigen_id",
        "source_instance_ids",
    }
    if quarantine:
        required.add("reasons")
    if set(payload) != required:
        raise ValueError(
            f"{label} schema mismatch: missing={sorted(required-set(payload))}, "
            f"unexpected={sorted(set(payload)-required)}"
        )
    p_id = validate_sha256(payload["pair_id"], f"{label}.pair_id")
    if p_id not in expected_pairs:
        raise ValueError(f"{label} references an out-of-universe pair")
    expected = expected_pairs[p_id]
    if payload["antibody_id"] != expected.antibody_id:
        raise ValueError(f"{label} antibody ID differs")
    if payload["antigen_id"] != expected.antigen_id:
        raise ValueError(f"{label} antigen ID differs")
    raw_sources = _list(payload["source_instance_ids"], f"{label}.source_instance_ids")
    sources = tuple(
        _safe_identifier(item, f"{label}.source_instance_ids")
        for item in raw_sources
    )
    if sources != expected.source_instance_ids:
        raise ValueError(f"{label} source-row group differs")
    reasons: list[str] = []
    if quarantine:
        reasons = [
            _safe_identifier(item, f"{label}.reasons[{index}]")
            for index, item in enumerate(
                _list(payload["reasons"], f"{label}.reasons")
            )
        ]
        allowed_reasons = {
            "held_out_antibody",
            "held_out_antigen",
            "antibody_direct_neighborhood",
            "antigen_direct_neighborhood",
            "mixed_role_pair",
        }
        if not reasons or len(reasons) != len(set(reasons)):
            raise ValueError(f"{label} quarantine reasons are empty or duplicated")
        if not set(reasons).issubset(allowed_reasons):
            raise ValueError(f"{label} has an unknown quarantine reason")
    return p_id, reasons


def _relative_range(values: Sequence[int]) -> float:
    """Compute count range divided by arithmetic mean."""
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


def independently_recomputed_outer_policy(
    pair_count: int,
) -> dict[str, float | int]:
    """Recompute the locked outer policy without trusting builder output."""
    if pair_count <= 0:
        raise ValueError("Outer policy pair universe must be positive")
    return {
        "minimum_validation_pairs": max(120, (7 * pair_count + 99) // 100),
        "maximum_validation_pairs": 9 * pair_count // 100,
        "minimum_validation_endpoints_per_axis": 60,
        "minimum_validation_components_per_axis": 25,
        "minimum_training_pair_fraction": 0.60,
        "minimum_training_endpoint_fraction": 0.60,
        "maximum_quarantine_pair_fraction": 0.33,
        "maximum_validation_pair_relative_range": 0.15,
        "maximum_other_count_relative_range": 0.20,
    }


def independently_recomputed_inner_policy(
    pair_count: int,
) -> dict[str, float | int]:
    """Recompute the locked inner policy without trusting builder output."""
    if pair_count <= 0:
        raise ValueError("Inner policy pair universe must be positive")
    return {
        "minimum_validation_pairs": max(80, (7 * pair_count + 99) // 100),
        "maximum_validation_pairs": 9 * pair_count // 100,
        "minimum_validation_endpoints_per_axis": 40,
        "minimum_validation_components_per_axis": 20,
        "minimum_training_pair_fraction": 0.55,
        "minimum_training_endpoint_fraction": 0.55,
        "maximum_quarantine_pair_fraction": 0.38,
        "maximum_validation_pair_relative_range": 0.20,
        "maximum_other_count_relative_range": 0.25,
    }


def _independently_recomputed_policy(
    level_kind: str,
    pair_count: int,
) -> dict[str, float | int]:
    """Dispatch to the independently locked policy for one nesting level."""
    if level_kind == "outer":
        return independently_recomputed_outer_policy(pair_count)
    if level_kind == "inner":
        return independently_recomputed_inner_policy(pair_count)
    raise ValueError(f"Unknown nested level kind: {level_kind!r}")


def _parse_policy(value: Any, label: str) -> dict[str, float | int]:
    """Parse a declared utility and retention policy with a strict schema."""
    policy = _mapping(value, label)
    required = {
        "minimum_validation_pairs",
        "maximum_validation_pairs",
        "minimum_validation_endpoints_per_axis",
        "minimum_validation_components_per_axis",
        "minimum_training_pair_fraction",
        "minimum_training_endpoint_fraction",
        "maximum_quarantine_pair_fraction",
        "maximum_validation_pair_relative_range",
        "maximum_other_count_relative_range",
    }
    if set(policy) != required:
        raise ValueError(f"{label} schema differs from the locked policy")
    result: dict[str, float | int] = {
        "minimum_validation_pairs": _integer(
            policy["minimum_validation_pairs"],
            f"{label}.minimum_validation_pairs",
        ),
        "maximum_validation_pairs": _integer(
            policy["maximum_validation_pairs"],
            f"{label}.maximum_validation_pairs",
        ),
        "minimum_validation_endpoints_per_axis": _integer(
            policy["minimum_validation_endpoints_per_axis"],
            f"{label}.minimum_validation_endpoints_per_axis",
        ),
        "minimum_validation_components_per_axis": _integer(
            policy["minimum_validation_components_per_axis"],
            f"{label}.minimum_validation_components_per_axis",
        ),
    }
    for key in (
        "minimum_training_pair_fraction",
        "minimum_training_endpoint_fraction",
        "maximum_quarantine_pair_fraction",
        "maximum_validation_pair_relative_range",
        "maximum_other_count_relative_range",
    ):
        result[key] = _number(policy[key], f"{label}.{key}")
        if not 0.0 <= float(result[key]) <= 1.0:
            raise ValueError(f"{label}.{key} must be in [0,1]")
    if int(result["minimum_validation_pairs"]) > int(
        result["maximum_validation_pairs"]
    ):
        raise ValueError(
            f"{label} validation-pair interval is infeasible: "
            f"minimum={result['minimum_validation_pairs']}, "
            f"maximum={result['maximum_validation_pairs']}"
        )
    return result


def _expected_quarantine_reasons(
    pair: IndependentPair,
    validation_antibodies: set[str],
    validation_antigens: set[str],
    closed_antibodies: set[str],
    closed_antigens: set[str],
) -> list[str]:
    """Reconstruct the canonical quarantine-reason ledger."""
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
    return reasons


def independently_reconstructed_component_ids(
    graph: Mapping[str, Sequence[str]],
) -> dict[str, str]:
    """Map a complete graph to lexical-min component representatives."""
    universe = set(graph)
    for entity, neighbors in graph.items():
        unknown = set(neighbors).difference(universe)
        if unknown:
            raise ValueError("Independent component graph has unknown neighbors")
        for neighbor in neighbors:
            if entity not in graph[neighbor]:
                raise ValueError("Independent component graph is not symmetric")
    remaining = set(universe)
    result: dict[str, str] = {}
    for root in sorted(universe):
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
    if set(result) != universe:
        raise RuntimeError("Independent component reconstruction is incomplete")
    return result


def prevalidate_level_envelope(
    value: Any,
    *,
    pair_count: int,
    label: str,
    level_kind: str,
) -> None:
    """Reject schema and locked-policy tampering before expensive alignment."""
    level = _mapping(value, label)
    _require_exact_schema(level, LEVEL_FIELDS, label)
    policy = _parse_policy(level.get("policy"), f"{label}.policy")
    recomputed_policy = _independently_recomputed_policy(level_kind, pair_count)
    if policy != recomputed_policy:
        raise ValueError(
            f"{label}.policy does not match the independently recomputed "
            f"locked {level_kind} policy for {pair_count} pairs"
        )
    folds = _list(level.get("folds"), f"{label}.folds")
    if len(folds) != 3:
        raise ValueError(f"{label} must contain exactly three folds")
    for fold_index, raw_fold in enumerate(folds):
        fold_label = f"{label}.fold_{fold_index}"
        fold = _mapping(raw_fold, fold_label)
        expected_fields = (
            FOLD_FIELDS | {"inner_level"}
            if level_kind == "outer"
            else FOLD_FIELDS
        )
        _require_exact_schema(fold, expected_fields, fold_label)
        if level_kind == "outer":
            inner = _mapping(fold.get("inner_level"), f"{fold_label}.inner")
            inner_pair_count = len(
                _list(fold.get("training_pairs"), f"{fold_label}.training_pairs")
            )
            prevalidate_level_envelope(
                inner,
                pair_count=inner_pair_count,
                label=f"{fold_label}.inner",
                level_kind="inner",
            )


def validate_level(
    value: Any,
    expected_pairs: Mapping[str, IndependentPair],
    label: str,
    *,
    level_kind: str,
    antibody_component_by_entity: Mapping[str, str],
    antigen_component_by_entity: Mapping[str, str],
) -> tuple[list[Boundary], bool, dict[str, Any]]:
    """Independently validate one outer or inner three-fold level."""
    level = _mapping(value, label)
    _require_exact_schema(level, LEVEL_FIELDS, label)
    if level_kind not in {"outer", "inner"}:
        raise ValueError(f"Unknown nested level kind: {level_kind!r}")
    universe = _mapping(level.get("universe"), f"{label}.universe")
    if _integer(universe.get("unique_pair_count"), f"{label}.pair_count") != len(
        expected_pairs
    ):
        raise ValueError(f"{label} universe pair count differs")
    if universe.get("pair_membership_sha256") != membership_sha256(expected_pairs):
        raise ValueError(f"{label} universe pair membership hash differs")
    all_antibodies = {pair.antibody_id for pair in expected_pairs.values()}
    all_antigens = {pair.antigen_id for pair in expected_pairs.values()}
    if not all_antibodies.issubset(antibody_component_by_entity):
        raise ValueError(f"{label} antibody universe leaves the full graph")
    if not all_antigens.issubset(antigen_component_by_entity):
        raise ValueError(f"{label} antigen universe leaves the full graph")
    expected_source_row_count = sum(
        len(pair.source_instance_ids) for pair in expected_pairs.values()
    )
    expected_universe_counts = {
        "unique_pair_count": len(expected_pairs),
        "antibody_count": len(all_antibodies),
        "antigen_count": len(all_antigens),
        "source_row_count": expected_source_row_count,
        "pair_membership_sha256": membership_sha256(expected_pairs),
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
    }
    if not _exact_json_equal(universe, expected_universe_counts):
        raise ValueError(f"{label} universe receipt differs independently")
    policy = _parse_policy(level.get("policy"), f"{label}.policy")
    recomputed_policy = _independently_recomputed_policy(
        level_kind,
        len(expected_pairs),
    )
    if policy != recomputed_policy:
        raise ValueError(
            f"{label}.policy does not match the independently recomputed "
            f"locked {level_kind} policy for {len(expected_pairs)} pairs"
        )
    folds = _list(level.get("folds"), f"{label}.folds")
    if len(folds) != 3:
        raise ValueError(f"{label} must contain exactly three folds")

    boundaries: list[Boundary] = []
    fold_summaries: list[dict[str, Any]] = []
    held_out_antibody_sets: list[set[str]] = []
    held_out_antigen_sets: list[set[str]] = []
    all_fold_gates_passed = True
    all_nested_children_passed = True
    for fold_index, raw_fold in enumerate(folds):
        fold_label = f"{label}.fold_{fold_index}"
        fold = _mapping(raw_fold, fold_label)
        expected_fold_fields = (
            FOLD_FIELDS | {"inner_level"}
            if level_kind == "outer"
            else FOLD_FIELDS
        )
        _require_exact_schema(fold, expected_fold_fields, fold_label)
        if _integer(fold.get("fold_index"), f"{fold_label}.fold_index") != fold_index:
            raise ValueError(f"{fold_label} index differs")
        validation_antibodies = set(
            _canonical_sha256_list(
                fold.get("validation_antibody_ids"),
                f"{fold_label}.validation_antibody_ids",
            )
        )
        validation_antigens = set(
            _canonical_sha256_list(
                fold.get("validation_antigen_ids"),
                f"{fold_label}.validation_antigen_ids",
            )
        )
        claimed_validation_antibody_components = set(
            _canonical_sha256_list(
                fold.get("validation_antibody_component_ids"),
                f"{fold_label}.validation_antibody_component_ids",
            )
        )
        claimed_validation_antigen_components = set(
            _canonical_sha256_list(
                fold.get("validation_antigen_component_ids"),
                f"{fold_label}.validation_antigen_component_ids",
            )
        )
        closed_antibodies = set(
            _canonical_sha256_list(
                fold.get("closed_direct_neighborhood_antibody_ids"),
                f"{fold_label}.closed_direct_neighborhood_antibody_ids",
            )
        )
        closed_antigens = set(
            _canonical_sha256_list(
                fold.get("closed_direct_neighborhood_antigen_ids"),
                f"{fold_label}.closed_direct_neighborhood_antigen_ids",
            )
        )
        if not validation_antibodies.issubset(all_antibodies):
            raise ValueError(f"{fold_label} has an out-of-universe antibody")
        if not validation_antigens.issubset(all_antigens):
            raise ValueError(f"{fold_label} has an out-of-universe antigen")
        if not validation_antibodies.issubset(closed_antibodies):
            raise ValueError(f"{fold_label} closed antibody set omits held-out IDs")
        if not validation_antigens.issubset(closed_antigens):
            raise ValueError(f"{fold_label} closed antigen set omits held-out IDs")
        if not closed_antibodies.issubset(all_antibodies):
            raise ValueError(f"{fold_label} closed antibody set leaves its universe")
        if not closed_antigens.issubset(all_antigens):
            raise ValueError(f"{fold_label} closed antigen set leaves its universe")
        held_out_antibody_sets.append(validation_antibodies)
        held_out_antigen_sets.append(validation_antigens)

        role_ids: dict[str, set[str]] = {}
        quarantine_reasons: dict[str, list[str]] = {}
        expected_pair_order = {
            p_id: index for index, p_id in enumerate(expected_pairs)
        }
        for role, key in (
            ("training", "training_pairs"),
            ("validation", "validation_pairs"),
            ("quarantine", "quarantined_pairs"),
        ):
            role_items = _list(fold.get(key), f"{fold_label}.{key}")
            parsed: set[str] = set()
            parsed_order: list[str] = []
            for item_index, item in enumerate(role_items):
                p_id, reasons = _validate_pair_payload(
                    item,
                    expected_pairs,
                    f"{fold_label}.{key}[{item_index}]",
                    quarantine=role == "quarantine",
                )
                if p_id in parsed:
                    raise ValueError(f"{fold_label}.{key} duplicates a pair")
                parsed.add(p_id)
                parsed_order.append(p_id)
                if role == "quarantine":
                    quarantine_reasons[p_id] = reasons
            if parsed_order != sorted(
                parsed_order,
                key=expected_pair_order.__getitem__,
            ):
                raise ValueError(
                    f"{fold_label}.{key} is not in canonical pair order"
                )
            role_ids[role] = parsed
        if (
            role_ids["training"] & role_ids["validation"]
            or role_ids["training"] & role_ids["quarantine"]
            or role_ids["validation"] & role_ids["quarantine"]
        ):
            raise ValueError(f"{fold_label} role sets overlap")
        if set().union(*role_ids.values()) != set(expected_pairs):
            raise ValueError(f"{fold_label} does not account for every exact pair")

        expected_validation = {
            p_id
            for p_id, pair in expected_pairs.items()
            if pair.antibody_id in validation_antibodies
            and pair.antigen_id in validation_antigens
        }
        if role_ids["validation"] != expected_validation:
            raise ValueError(f"{fold_label} validation pairs are not mechanical")
        effective_val_a = {
            expected_pairs[p_id].antibody_id for p_id in role_ids["validation"]
        }
        effective_val_g = {
            expected_pairs[p_id].antigen_id for p_id in role_ids["validation"]
        }
        expected_validation_antibody_components = {
            antibody_component_by_entity[entity] for entity in effective_val_a
        }
        expected_validation_antigen_components = {
            antigen_component_by_entity[entity] for entity in effective_val_g
        }
        if (
            claimed_validation_antibody_components
            != expected_validation_antibody_components
        ):
            raise ValueError(
                f"{fold_label} antibody component inventory differs from the "
                "independently reconstructed full graph"
            )
        if (
            claimed_validation_antigen_components
            != expected_validation_antigen_components
        ):
            raise ValueError(
                f"{fold_label} antigen component inventory differs from the "
                "independently reconstructed full graph"
            )
        if validation_antibodies != effective_val_a:
            raise ValueError(f"{fold_label} has dangling held-out antibodies")
        if validation_antigens != effective_val_g:
            raise ValueError(f"{fold_label} has dangling held-out antigens")

        model_a = {
            expected_pairs[p_id].antibody_id for p_id in role_ids["training"]
        }
        model_g = {
            expected_pairs[p_id].antigen_id for p_id in role_ids["training"]
        }
        claimed_model_a = set(
            _canonical_sha256_list(
                fold.get("model_antibody_allowlist"),
                f"{fold_label}.model_antibody_allowlist",
            )
        )
        claimed_model_g = set(
            _canonical_sha256_list(
                fold.get("model_antigen_allowlist"),
                f"{fold_label}.model_antigen_allowlist",
            )
        )
        if claimed_model_a != model_a or claimed_model_g != model_g:
            raise ValueError(f"{fold_label} model allowlists are not exact")
        if model_a & validation_antibodies or model_g & validation_antigens:
            raise ValueError(f"{fold_label} exact endpoints cross the boundary")

        for p_id in role_ids["quarantine"]:
            expected_reason_list = _expected_quarantine_reasons(
                expected_pairs[p_id],
                validation_antibodies,
                validation_antigens,
                closed_antibodies,
                closed_antigens,
            )
            if quarantine_reasons[p_id] != expected_reason_list:
                raise ValueError(f"{fold_label} quarantine reason ledger differs")

        training_eligible_a = set(
            _canonical_sha256_list(
                fold.get("training_eligible_antibody_ids"),
                f"{fold_label}.training_eligible_antibody_ids",
            )
        )
        training_eligible_g = set(
            _canonical_sha256_list(
                fold.get("training_eligible_antigen_ids"),
                f"{fold_label}.training_eligible_antigen_ids",
            )
        )
        if training_eligible_a != all_antibodies - closed_antibodies:
            raise ValueError(f"{fold_label} antibody eligibility set differs")
        if training_eligible_g != all_antigens - closed_antigens:
            raise ValueError(f"{fold_label} antigen eligibility set differs")
        expected_training = {
            p_id
            for p_id, pair in expected_pairs.items()
            if pair.antibody_id in training_eligible_a
            and pair.antigen_id in training_eligible_g
        }
        if role_ids["training"] != expected_training:
            raise ValueError(f"{fold_label} training pairs are not mechanical")

        training_count = len(role_ids["training"])
        validation_count = len(role_ids["validation"])
        quarantine_count = len(role_ids["quarantine"])
        source_count = sum(
            len(expected_pairs[p_id].source_instance_ids)
            for p_id in set().union(*role_ids.values())
        )
        expected_counts = {
            "universe_pair_count": len(expected_pairs),
            "training_pair_count": training_count,
            "validation_pair_count": validation_count,
            "quarantined_pair_count": quarantine_count,
            "universe_antibody_count": len(all_antibodies),
            "universe_antigen_count": len(all_antigens),
            "training_antibody_count": len(model_a),
            "training_antigen_count": len(model_g),
            "validation_antibody_count": len(effective_val_a),
            "validation_antigen_count": len(effective_val_g),
            "validation_antibody_component_count": len(
                expected_validation_antibody_components
            ),
            "validation_antigen_component_count": len(
                expected_validation_antigen_components
            ),
            "training_source_row_count": sum(
                len(expected_pairs[p_id].source_instance_ids)
                for p_id in role_ids["training"]
            ),
            "validation_source_row_count": sum(
                len(expected_pairs[p_id].source_instance_ids)
                for p_id in role_ids["validation"]
            ),
            "quarantined_source_row_count": sum(
                len(expected_pairs[p_id].source_instance_ids)
                for p_id in role_ids["quarantine"]
            ),
        }
        claimed_counts = _mapping(fold.get("counts"), f"{fold_label}.counts")
        if not _exact_json_equal(claimed_counts, expected_counts):
            raise ValueError(f"{fold_label} count receipt differs independently")
        expected_membership_hashes = {
            "training_pair_sha256": membership_sha256(role_ids["training"]),
            "validation_pair_sha256": membership_sha256(role_ids["validation"]),
            "quarantined_pair_sha256": membership_sha256(role_ids["quarantine"]),
            "model_antibody_allowlist_sha256": membership_sha256(model_a),
            "model_antigen_allowlist_sha256": membership_sha256(model_g),
            "validation_antibody_component_sha256": membership_sha256(
                expected_validation_antibody_components
            ),
            "validation_antigen_component_sha256": membership_sha256(
                expected_validation_antigen_components
            ),
        }
        claimed_hashes = _mapping(
            fold.get("membership_hashes"), f"{fold_label}.membership_hashes"
        )
        if not _exact_json_equal(claimed_hashes, expected_membership_hashes):
            raise ValueError(
                f"{fold_label} membership-hash receipt differs independently"
            )
        recomputed_gates = {
            "validation_pair_lower_bound_passed": validation_count
            >= int(policy["minimum_validation_pairs"]),
            "validation_pair_upper_bound_passed": validation_count
            <= int(policy["maximum_validation_pairs"]),
            "validation_antibody_count_passed": len(effective_val_a)
            >= int(policy["minimum_validation_endpoints_per_axis"]),
            "validation_antigen_count_passed": len(effective_val_g)
            >= int(policy["minimum_validation_endpoints_per_axis"]),
            "validation_antibody_component_count_passed": len(
                expected_validation_antibody_components
            )
            >= int(policy["minimum_validation_components_per_axis"]),
            "validation_antigen_component_count_passed": len(
                expected_validation_antigen_components
            )
            >= int(policy["minimum_validation_components_per_axis"]),
            "training_pair_retention_passed": training_count
            >= _ceil_fraction_product(
                float(policy["minimum_training_pair_fraction"]),
                len(expected_pairs),
            ),
            "training_antibody_retention_passed": len(model_a)
            >= _ceil_fraction_product(
                float(policy["minimum_training_endpoint_fraction"]),
                len(all_antibodies),
            ),
            "training_antigen_retention_passed": len(model_g)
            >= _ceil_fraction_product(
                float(policy["minimum_training_endpoint_fraction"]),
                len(all_antigens),
            ),
            "quarantine_limit_passed": quarantine_count
            <= _floor_fraction_product(
                float(policy["maximum_quarantine_pair_fraction"]),
                len(expected_pairs),
            ),
            "every_held_out_antibody_is_effective": (
                validation_antibodies == effective_val_a
            ),
            "every_held_out_antigen_is_effective": (
                validation_antigens == effective_val_g
            ),
            "model_antibody_allowlist_is_exact": claimed_model_a == model_a,
            "model_antigen_allowlist_is_exact": claimed_model_g == model_g,
            "complete_unique_pair_accounting": (
                training_count + validation_count + quarantine_count
                == len(expected_pairs)
            ),
            "complete_source_row_accounting": source_count
            == sum(
                len(pair.source_instance_ids) for pair in expected_pairs.values()
            ),
        }
        claimed_gates = _mapping(fold.get("hard_gates"), f"{fold_label}.hard_gates")
        if not _exact_json_equal(claimed_gates, recomputed_gates):
            raise ValueError(f"{fold_label} hard-gate receipt differs")
        claimed_all = _boolean(
            fold.get("all_fold_hard_gates_passed"),
            f"{fold_label}.all_fold_hard_gates_passed",
        )
        recomputed_all = all(recomputed_gates.values())
        if claimed_all != recomputed_all:
            raise ValueError(f"{fold_label} aggregate hard gate differs")
        all_fold_gates_passed = all_fold_gates_passed and recomputed_all
        boundaries.append(
            Boundary(
                label=fold_label,
                training_pair_ids=tuple(sorted(role_ids["training"])),
                validation_pair_ids=tuple(sorted(role_ids["validation"])),
            )
        )
        fold_summary = {
            "label": fold_label,
            "training_pair_count": training_count,
            "validation_pair_count": validation_count,
            "quarantined_pair_count": quarantine_count,
            "training_antibody_count": len(model_a),
            "training_antigen_count": len(model_g),
            "validation_antibody_count": len(effective_val_a),
            "validation_antigen_count": len(effective_val_g),
            "validation_antibody_component_count": len(
                expected_validation_antibody_components
            ),
            "validation_antigen_component_count": len(
                expected_validation_antigen_components
            ),
            "source_row_count": source_count,
            "all_fold_hard_gates_passed": recomputed_all,
        }
        fold_summaries.append(fold_summary)

        inner_value = fold.get("inner_level")
        if inner_value is not None:
            inner_mapping = _mapping(inner_value, f"{fold_label}.inner_level")
            if inner_mapping.get("construction_skipped"):
                all_nested_children_passed = False
            else:
                inner_expected = {
                    p_id: pair
                    for p_id, pair in expected_pairs.items()
                    if p_id in role_ids["training"]
                }
                inner_boundaries, inner_pass, inner_summary = validate_level(
                    inner_mapping,
                    inner_expected,
                    f"{fold_label}.inner",
                    level_kind="inner",
                    antibody_component_by_entity=(
                        antibody_component_by_entity
                    ),
                    antigen_component_by_entity=(
                        antigen_component_by_entity
                    ),
                )
                boundaries.extend(inner_boundaries)
                all_nested_children_passed = (
                    all_nested_children_passed and inner_pass
                )
                fold_summary["inner_summary"] = inner_summary

    disjoint_a = sum(map(len, held_out_antibody_sets)) == len(
        set().union(*held_out_antibody_sets)
    )
    disjoint_g = sum(map(len, held_out_antigen_sets)) == len(
        set().union(*held_out_antigen_sets)
    )
    count_vectors = {
        key: [int(summary[key]) for summary in fold_summaries]
        for key in (
            "validation_pair_count",
            "validation_antibody_count",
            "validation_antigen_count",
            "training_pair_count",
            "quarantined_pair_count",
        )
    }
    ranges = {key: _relative_range(values) for key, values in count_vectors.items()}
    claimed_ranges = _mapping(
        level.get("count_relative_ranges"),
        f"{label}.count_relative_ranges",
    )
    if not _exact_json_equal(claimed_ranges, ranges):
        raise ValueError(
            f"{label} count-relative-range receipt differs independently"
        )
    balance_pass = _relative_range_passes(
        count_vectors["validation_pair_count"],
        float(policy["maximum_validation_pair_relative_range"]),
    ) and all(
        _relative_range_passes(
            count_vectors[key],
            float(policy["maximum_other_count_relative_range"]),
        )
        for key in (
            "validation_antibody_count",
            "validation_antigen_count",
            "training_pair_count",
            "quarantined_pair_count",
        )
    )
    level_gates = {
        "selection_target_reached_in_every_fold": all(
            int(policy["minimum_validation_pairs"])
            <= summary["validation_pair_count"]
            <= int(policy["maximum_validation_pairs"])
            for summary in fold_summaries
        ),
        "held_out_antibody_sets_pairwise_disjoint": disjoint_a,
        "held_out_antigen_sets_pairwise_disjoint": disjoint_g,
        "every_fold_hard_gate_passed": all_fold_gates_passed,
        "validation_pair_relative_range_passed": (
            _relative_range_passes(
                count_vectors["validation_pair_count"],
                float(policy["maximum_validation_pair_relative_range"]),
            )
        ),
        "validation_antibody_relative_range_passed": (
            _relative_range_passes(
                count_vectors["validation_antibody_count"],
                float(policy["maximum_other_count_relative_range"]),
            )
        ),
        "validation_antigen_relative_range_passed": (
            _relative_range_passes(
                count_vectors["validation_antigen_count"],
                float(policy["maximum_other_count_relative_range"]),
            )
        ),
        "training_pair_relative_range_passed": (
            _relative_range_passes(
                count_vectors["training_pair_count"],
                float(policy["maximum_other_count_relative_range"]),
            )
        ),
        "quarantine_pair_relative_range_passed": (
            _relative_range_passes(
                count_vectors["quarantined_pair_count"],
                float(policy["maximum_other_count_relative_range"]),
            )
        ),
    }
    claimed_level_gates = _mapping(
        level.get("level_hard_gates"), f"{label}.level_hard_gates"
    )
    if not _exact_json_equal(claimed_level_gates, level_gates):
        raise ValueError(f"{label} level hard-gate receipt differs independently")
    current_level_passed = all(level_gates.values())
    claimed_level_passed = _boolean(
        level.get("all_level_hard_gates_passed"),
        f"{label}.all_level_hard_gates_passed",
    )
    if claimed_level_passed != current_level_passed:
        raise ValueError(f"{label} aggregate level hard gate differs")
    all_pass = current_level_passed and all_nested_children_passed
    return boundaries, all_pass, {
        "label": label,
        "level_kind": level_kind,
        "unique_pair_count": len(expected_pairs),
        "declared_policy_matches_independently_recomputed_locked_policy": True,
        "independently_recomputed_locked_policy": recomputed_policy,
        "held_out_antibody_sets_pairwise_disjoint": disjoint_a,
        "held_out_antigen_sets_pairwise_disjoint": disjoint_g,
        "count_relative_ranges": ranges,
        "balance_gates_passed": balance_pass,
        "current_level_hard_gates": level_gates,
        "current_level_hard_gates_passed": current_level_passed,
        "all_nested_children_passed": all_nested_children_passed,
        "folds": fold_summaries,
        "all_independent_level_invariants_passed": all_pass,
    }


def _blosum62_payload_sha256(matrix: Any) -> str:
    """Hash the complete labeled BLOSUM62 matrix independently."""
    alphabet = str(matrix.alphabet)
    payload = {
        "alphabet": alphabet,
        "scores": [
            [float(matrix[left, right]) for right in alphabet] for left in alphabet
        ],
    }
    return canonical_payload_sha256(payload)


def build_locked_aligner() -> object:
    """Construct the independently pinned global alignment backend."""
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


def independent_alignment_counts(
    left: str, right: str, aligner: object
) -> tuple[int, int]:
    """Count identical and total paired residues in the first optimum."""
    alignment = next(iter(aligner.align(left, right)))
    left_blocks, right_blocks = alignment.aligned
    if len(left_blocks) != len(right_blocks):
        raise RuntimeError("Alignment block cardinalities differ")
    identical = 0
    paired = 0
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
        raise RuntimeError("Global alignment has no paired residues")
    return identical, paired


def safe_length_prune(left_length: int, right_length: int) -> bool:
    """Return whether reciprocal coverage is provably below 80%."""
    shorter = min(left_length, right_length)
    longer = max(left_length, right_length)
    return COVERAGE_DENOMINATOR * shorter < COVERAGE_NUMERATOR * longer


def threshold_edge(
    identical: int, paired: int, left_length: int, right_length: int
) -> bool:
    """Apply both inclusive locked thresholds by integer arithmetic."""
    return (
        IDENTITY_DENOMINATOR * identical >= IDENTITY_NUMERATOR * paired
        and COVERAGE_DENOMINATOR * paired >= COVERAGE_NUMERATOR * left_length
        and COVERAGE_DENOMINATOR * paired >= COVERAGE_NUMERATOR * right_length
    )


def independent_symmetric_threshold_evaluation(
    left: str,
    right: str,
    aligner: object,
) -> tuple[
    bool,
    tuple[int, int],
    tuple[int, int],
]:
    """Independently apply the fail-closed edge rule in both orientations."""
    training_to_validation = independent_alignment_counts(left, right, aligner)
    validation_to_training = independent_alignment_counts(right, left, aligner)
    qualifies = (
        threshold_edge(
            *training_to_validation,
            len(left),
            len(right),
        )
        or threshold_edge(
            *validation_to_training,
            len(right),
            len(left),
        )
    )
    return qualifies, training_to_validation, validation_to_training


def coverage_passes(paired: int, left_length: int, right_length: int) -> bool:
    """Return whether both reciprocal coverage gates pass."""
    return (
        COVERAGE_DENOMINATOR * paired >= COVERAGE_NUMERATOR * left_length
        and COVERAGE_DENOMINATOR * paired >= COVERAGE_NUMERATOR * right_length
    )


def _initialize_worker() -> None:
    """Initialize one process-local independent aligner."""
    global _WORKER_ALIGNER
    _WORKER_ALIGNER = build_locked_aligner()


def _audit_chunk(
    chunk: Sequence[tuple[str, str]],
) -> tuple[
    int,
    int,
    list[tuple[str, str, int, int, int, int]],
    tuple[str, str, int, int] | None,
]:
    """Audit one chunk and return crossings plus its best covered identity."""
    if _WORKER_ALIGNER is None:
        raise RuntimeError("Audit worker is not initialized")
    crossings: list[tuple[str, str, int, int, int, int]] = []
    best: tuple[str, str, int, int] | None = None
    for left, right in chunk:
        qualifies, forward, reverse = independent_symmetric_threshold_evaluation(
            left, right, _WORKER_ALIGNER
        )
        for identical, paired in (forward, reverse):
            candidate = (
                (left, right, identical, paired)
                if coverage_passes(paired, len(left), len(right))
                else None
            )
            best = _better_best(best, candidate)
        if qualifies:
            crossings.append((left, right, *forward, *reverse))
    return len(chunk), 2 * len(chunk), crossings, best


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


def _independent_view_inventory(
    antibodies: Mapping[str, tuple[str, str]],
    antigens: Mapping[str, str],
    view: str,
) -> dict[str, str]:
    """Return the independently reconstructed entity inventory for one view."""
    if view == "vh":
        return {
            identifier: sequences[0]
            for identifier, sequences in antibodies.items()
        }
    if view == "vl":
        return {
            identifier: sequences[1]
            for identifier, sequences in antibodies.items()
            if sequences[1]
        }
    if view == "concatenated_antibody":
        return {
            identifier: sequences[0] + sequences[1]
            for identifier, sequences in antibodies.items()
        }
    if view == "antigen":
        return dict(antigens)
    raise ValueError(f"Unknown sequence view: {view}")


def independently_reconstruct_exact_view_graph(
    inventory: Mapping[str, str],
    *,
    workers: int,
    chunk_size: int,
) -> tuple[set[tuple[str, str]], dict[str, Any]]:
    """Rebuild one complete direct graph without trusting builder graph bytes."""
    entities_by_sequence: dict[str, list[str]] = defaultdict(list)
    for entity_identifier, sequence in inventory.items():
        entities_by_sequence[sequence].append(entity_identifier)
    sequences = sorted(entities_by_sequence)
    exact_group_edges: set[tuple[str, str]] = set()
    for entity_ids in entities_by_sequence.values():
        exact_group_edges.update(
            itertools.combinations(sorted(entity_ids), 2)
        )

    candidate_count = len(sequences) * (len(sequences) - 1) // 2
    safe_pruned = 0
    alignable: list[tuple[str, str]] = []
    sequence_digests = {
        sequence: sha256_bytes(sequence.encode("ascii"))
        for sequence in sequences
    }
    candidate_enumeration_digest = hashlib.sha256()
    safe_prune_enumeration_digest = hashlib.sha256()
    alignment_enumeration_digest = hashlib.sha256()
    for left, right in itertools.combinations(sequences, 2):
        digest_row = (
            f"{sequence_digests[left]}\0{sequence_digests[right]}\n"
        ).encode("ascii")
        candidate_enumeration_digest.update(digest_row)
        if safe_length_prune(len(left), len(right)):
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
                qualifies, _, _ = independent_symmetric_threshold_evaluation(
                    left,
                    right,
                    aligner,
                )
                aligned_count += 1
                directional_alignment_execution_count += 2
                if qualifies:
                    threshold_sequence_pairs.append((left, right))
        else:
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_initialize_worker,
            ) as executor:
                for count, execution_count, crossings, _ in executor.map(
                    _audit_chunk,
                    _chunks(alignable, chunk_size),
                    chunksize=1,
                ):
                    aligned_count += count
                    directional_alignment_execution_count += execution_count
                    threshold_sequence_pairs.extend(
                        (crossing[0], crossing[1])
                        for crossing in crossings
                    )
    if aligned_count + safe_pruned != candidate_count:
        raise RuntimeError("Independent exact-graph accounting is incomplete")
    if directional_alignment_execution_count != 2 * aligned_count:
        raise RuntimeError(
            "Independent exact-graph directional accounting is incomplete"
        )

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
            sequence_digests[sequence] for sequence in sequences
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
        "threshold_distinct_sequence_pair_count": len(
            threshold_sequence_pairs
        ),
        "threshold_distinct_sequence_pair_membership_sha256": membership_sha256(
            f"{sequence_digests[left]}\0{sequence_digests[right]}"
            for left, right in threshold_sequence_pairs
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


def independently_reconstruct_exact_graphs(
    assignment: Mapping[str, Any],
    antibodies: Mapping[str, tuple[str, str]],
    antigens: Mapping[str, str],
    *,
    workers: int,
    chunk_size: int,
) -> tuple[dict[str, set[tuple[str, str]]], dict[str, dict[str, Any]]]:
    """Rebuild and bind all four graph receipts used by the joint model."""
    claimed_graphs = _mapping(
        assignment.get("exact_direct_graphs"),
        "assignment exact_direct_graphs",
    )
    if set(claimed_graphs) != set(REQUIRED_VIEWS):
        raise ValueError("Assignment exact-direct-graph view schema differs")
    edges_by_view: dict[str, set[tuple[str, str]]] = {}
    receipts_by_view: dict[str, dict[str, Any]] = {}
    for view in REQUIRED_VIEWS:
        edges, receipt = independently_reconstruct_exact_view_graph(
            _independent_view_inventory(antibodies, antigens, view),
            workers=workers,
            chunk_size=chunk_size,
        )
        if not _exact_json_equal(
            _mapping(
                claimed_graphs.get(view),
                f"assignment exact_direct_graphs.{view}",
            ),
            receipt,
        ):
            raise ValueError(
                f"Assignment exact direct graph differs independently for {view}"
            )
        edges_by_view[view] = edges
        receipts_by_view[view] = receipt

    combined_antibody_edges = set().union(
        edges_by_view["vh"],
        edges_by_view["vl"],
        edges_by_view["concatenated_antibody"],
    )
    if _integer(
        assignment.get("combined_antibody_direct_edge_count"),
        "assignment combined_antibody_direct_edge_count",
    ) != len(combined_antibody_edges):
        raise ValueError("Combined antibody direct-edge count differs independently")
    claimed_combined_digest = validate_sha256(
        assignment.get("combined_antibody_direct_edge_membership_sha256"),
        "assignment combined_antibody_direct_edge_membership_sha256",
    )
    observed_combined_digest = membership_sha256(
        f"{left}\0{right}" for left, right in combined_antibody_edges
    )
    if claimed_combined_digest != observed_combined_digest:
        raise ValueError(
            "Combined antibody direct-edge membership differs independently"
        )
    return edges_by_view, receipts_by_view


def _view_sequences(
    view: str,
    pair_ids: Sequence[str],
    pairs: Mapping[str, IndependentPair],
    antibodies: Mapping[str, tuple[str, str]],
    antigens: Mapping[str, str],
) -> set[str]:
    """Return unique sequences represented by one pair role and view."""
    if view == "antigen":
        return {antigens[pairs[p_id].antigen_id] for p_id in pair_ids}
    antibody_ids = {pairs[p_id].antibody_id for p_id in pair_ids}
    if view == "vh":
        return {antibodies[a_id][0] for a_id in antibody_ids}
    if view == "vl":
        return {
            antibodies[a_id][1]
            for a_id in antibody_ids
            if antibodies[a_id][1]
        }
    if view == "concatenated_antibody":
        return {
            antibodies[a_id][0] + antibodies[a_id][1]
            for a_id in antibody_ids
        }
    raise ValueError(f"Unknown view: {view}")


def _better_best(
    current: tuple[str, str, int, int] | None,
    candidate: tuple[str, str, int, int] | None,
) -> tuple[str, str, int, int] | None:
    """Select the higher identity fraction with digest-stable ties."""
    if candidate is None:
        return current
    if current is None:
        return candidate
    if candidate[2] * current[3] > current[2] * candidate[3]:
        return candidate
    if candidate[2] * current[3] < current[2] * candidate[3]:
        return current
    current_key = (
        sha256_bytes(current[0].encode("ascii")),
        sha256_bytes(current[1].encode("ascii")),
    )
    candidate_key = (
        sha256_bytes(candidate[0].encode("ascii")),
        sha256_bytes(candidate[1].encode("ascii")),
    )
    return candidate if candidate_key < current_key else current


def _crossing_payload(
    crossing: tuple[str, str, int, int, int, int],
    boundary: str,
    view: str,
) -> dict[str, Any]:
    """Serialize redacted evidence for one symmetric threshold crossing."""
    left, right, forward_identical, forward_paired, reverse_identical, reverse_paired = (
        crossing
    )
    forward_qualifies = threshold_edge(
        forward_identical,
        forward_paired,
        len(left),
        len(right),
    )
    reverse_qualifies = threshold_edge(
        reverse_identical,
        reverse_paired,
        len(right),
        len(left),
    )
    qualifying_orientations = [
        label
        for label, qualifies in (
            ("training_to_validation", forward_qualifies),
            ("validation_to_training", reverse_qualifies),
        )
        if qualifies
    ]
    if not qualifying_orientations:
        raise RuntimeError("A crossing lacks a qualifying alignment orientation")
    if forward_qualifies:
        selected_identical, selected_paired = (
            forward_identical,
            forward_paired,
        )
        selected_orientation = "training_to_validation"
    else:
        selected_identical, selected_paired = (
            reverse_identical,
            reverse_paired,
        )
        selected_orientation = "validation_to_training"

    def directional_payload(identical: int, paired: int) -> dict[str, Any]:
        return {
            "identical_residue_pair_count": identical,
            "aligned_residue_pair_count": paired,
            "aligned_region_identity": identical / paired,
            "training_reciprocal_coverage": paired / len(left),
            "validation_reciprocal_coverage": paired / len(right),
        }

    return {
        "boundary": boundary,
        "sequence_view": view,
        "training_sequence_sha256": sha256_bytes(left.encode("ascii")),
        "validation_sequence_sha256": sha256_bytes(right.encode("ascii")),
        "training_sequence_length": len(left),
        "validation_sequence_length": len(right),
        "selected_qualifying_orientation": selected_orientation,
        "qualifying_alignment_orientations": qualifying_orientations,
        "identical_residue_pair_count": selected_identical,
        "aligned_residue_pair_count": selected_paired,
        "aligned_region_identity": selected_identical / selected_paired,
        "training_reciprocal_coverage": selected_paired / len(left),
        "validation_reciprocal_coverage": selected_paired / len(right),
        "training_to_validation_alignment": directional_payload(
            forward_identical, forward_paired
        ),
        "validation_to_training_alignment": directional_payload(
            reverse_identical, reverse_paired
        ),
        "symmetric_either_orientation_threshold_edge": True,
        "raw_sequences_or_source_instance_ids_included": False,
    }


def audit_boundary_view(
    boundary: Boundary,
    view: str,
    pairs: Mapping[str, IndependentPair],
    antibodies: Mapping[str, tuple[str, str]],
    antigens: Mapping[str, str],
    *,
    workers: int,
    chunk_size: int,
) -> dict[str, Any]:
    """Exhaustively audit one actual train-validation sequence boundary."""
    training_sequences = sorted(
        _view_sequences(
            view,
            boundary.training_pair_ids,
            pairs,
            antibodies,
            antigens,
        )
    )
    validation_sequences = sorted(
        _view_sequences(
            view,
            boundary.validation_pair_ids,
            pairs,
            antibodies,
            antigens,
        )
    )
    candidate_count = len(training_sequences) * len(validation_sequences)
    exact_matches: list[tuple[str, str, int, int]] = []
    safe_pruned = 0
    alignable: list[tuple[str, str]] = []
    sequence_digests = {
        sequence: sha256_bytes(sequence.encode("ascii"))
        for sequence in set(training_sequences).union(validation_sequences)
    }
    candidate_enumeration_digest = hashlib.sha256()
    exact_match_enumeration_digest = hashlib.sha256()
    safe_prune_enumeration_digest = hashlib.sha256()
    alignment_enumeration_digest = hashlib.sha256()
    for left in training_sequences:
        for right in validation_sequences:
            digest_row = (
                f"{sequence_digests[left]}\0{sequence_digests[right]}\n"
            ).encode("ascii")
            candidate_enumeration_digest.update(digest_row)
            if left == right:
                exact_matches.append((left, right, len(left), len(left)))
                exact_match_enumeration_digest.update(digest_row)
            elif safe_length_prune(len(left), len(right)):
                safe_pruned += 1
                safe_prune_enumeration_digest.update(digest_row)
            else:
                alignable.append((left, right))
                alignment_enumeration_digest.update(digest_row)

    aligned_count = 0
    directional_alignment_execution_count = 0
    crossings = [
        (left, right, identical, paired, identical, paired)
        for left, right, identical, paired in exact_matches
    ]
    best: tuple[str, str, int, int] | None = (
        exact_matches[0] if exact_matches else None
    )
    if alignable:
        if workers == 1:
            aligner = build_locked_aligner()
            for left, right in alignable:
                qualifies, forward, reverse = (
                    independent_symmetric_threshold_evaluation(
                        left, right, aligner
                    )
                )
                aligned_count += 1
                directional_alignment_execution_count += 2
                for identical, paired in (forward, reverse):
                    candidate = (
                        (left, right, identical, paired)
                        if coverage_passes(paired, len(left), len(right))
                        else None
                    )
                    best = _better_best(best, candidate)
                if qualifies:
                    crossings.append((left, right, *forward, *reverse))
        else:
            with ProcessPoolExecutor(
                max_workers=workers, initializer=_initialize_worker
            ) as executor:
                for count, execution_count, chunk_crossings, chunk_best in executor.map(
                    _audit_chunk,
                    _chunks(alignable, chunk_size),
                    chunksize=1,
                ):
                    aligned_count += count
                    directional_alignment_execution_count += execution_count
                    crossings.extend(chunk_crossings)
                    best = _better_best(best, chunk_best)
    if (
        aligned_count + safe_pruned + len(exact_matches)
        != candidate_count
    ):
        raise RuntimeError(f"{boundary.label}:{view} accounting is incomplete")
    if directional_alignment_execution_count != 2 * aligned_count:
        raise RuntimeError(
            f"{boundary.label}:{view} symmetric alignment accounting is incomplete"
        )
    crossing_payloads = [
        _crossing_payload(crossing, boundary.label, view)
        for crossing in sorted(
            crossings,
            key=lambda item: (
                sha256_bytes(item[0].encode("ascii")),
                sha256_bytes(item[1].encode("ascii")),
            ),
        )
    ]
    maximum = (
        None
        if best is None
        else {
            "identity_numerator": best[2],
            "identity_denominator": best[3],
            "aligned_region_identity": best[2] / best[3],
            "training_sequence_sha256": sha256_bytes(best[0].encode("ascii")),
            "validation_sequence_sha256": sha256_bytes(best[1].encode("ascii")),
        }
    )
    return {
        "boundary": boundary.label,
        "sequence_view": view,
        "training_unique_sequence_count": len(training_sequences),
        "training_unique_sequence_membership_sha256": membership_sha256(
            sequence_digests[sequence] for sequence in training_sequences
        ),
        "validation_unique_sequence_count": len(validation_sequences),
        "validation_unique_sequence_membership_sha256": membership_sha256(
            sequence_digests[sequence] for sequence in validation_sequences
        ),
        "cross_boundary_unique_sequence_pair_count": candidate_count,
        "cross_boundary_sequence_pair_enumeration_sha256": (
            candidate_enumeration_digest.hexdigest()
        ),
        "exact_sequence_match_count": len(exact_matches),
        "exact_sequence_match_pair_enumeration_sha256": (
            exact_match_enumeration_digest.hexdigest()
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
        "all_pairs_exactly_accounted": (
            aligned_count + safe_pruned + len(exact_matches) == candidate_count
        ),
        "symmetric_either_orientation_threshold_edge_rule": True,
        "threshold_crossing_edge_count": len(crossings),
        "zero_threshold_crossing_edges": not crossings,
        "maximum_identity_among_coverage_qualified_pairs": maximum,
        "crossing_edges": crossing_payloads,
        "crossing_edge_membership_sha256": membership_sha256(
            (
                f"{item['training_sequence_sha256']}\0"
                f"{item['validation_sequence_sha256']}"
            )
            for item in crossing_payloads
        ),
    }


def _fold_selected_binary_keys(
    fold: Mapping[str, Any],
    prefix: tuple[str, ...],
    label: str,
) -> set[tuple[str, ...]]:
    """Reconstruct every selected model variable represented by one ledger."""
    selected: set[tuple[str, ...]] = set()
    id_roles = {
        "holdout_a": "validation_antibody_ids",
        "holdout_g": "validation_antigen_ids",
        "validation_component_a": "validation_antibody_component_ids",
        "validation_component_g": "validation_antigen_component_ids",
        "closed_a": "closed_direct_neighborhood_antibody_ids",
        "closed_g": "closed_direct_neighborhood_antigen_ids",
        "train_present_a": "model_antibody_allowlist",
        "train_present_g": "model_antigen_allowlist",
    }
    for role, key in id_roles.items():
        for value in _list(fold.get(key), f"{label}.{key}"):
            selected.add((*prefix, role, validate_sha256(value, f"{label}.{key}")))
    pair_roles = {
        "validation_pair": "validation_pairs",
        "training_pair": "training_pairs",
        "quarantine_pair": "quarantined_pairs",
    }
    for role, key in pair_roles.items():
        for index, raw_pair in enumerate(
            _list(fold.get(key), f"{label}.{key}")
        ):
            pair = _mapping(raw_pair, f"{label}.{key}[{index}]")
            selected.add(
                (
                    *prefix,
                    role,
                    validate_sha256(
                        pair.get("pair_id"),
                        f"{label}.{key}[{index}].pair_id",
                    ),
                )
            )
    return selected


def _validate_level_selection_binding(
    level: Mapping[str, Any],
    prefix_stem: tuple[str, ...],
    label: str,
) -> tuple[set[tuple[str, ...]], str]:
    """Bind one level's selector receipt to its three published fold ledgers."""
    folds = _list(level.get("folds"), f"{label}.folds")
    if len(folds) != 3:
        raise ValueError(f"{label} must contain exactly three folds")
    selected: set[tuple[str, ...]] = set()
    validation_counts: list[int] = []
    for fold_index, raw_fold in enumerate(folds):
        fold = _mapping(raw_fold, f"{label}.fold_{fold_index}")
        prefix = (*prefix_stem, str(fold_index))
        selected.update(
            _fold_selected_binary_keys(
                fold,
                prefix,
                f"{label}.fold_{fold_index}",
            )
        )
        validation_counts.append(
            len(
                _list(
                    fold.get("validation_pairs"),
                    f"{label}.fold_{fold_index}.validation_pairs",
                )
            )
        )

    selection = _mapping(level.get("selection"), f"{label}.selection")
    expected_schema = {
        "algorithm",
        "score_free",
        "constant_zero_objective_only",
        "lexicographic_optimization_performed",
        "target_minimum_validation_pairs",
        "target_maximum_validation_pairs",
        "final_validation_pair_counts",
        "stalled_fold_indices",
        "target_reached_in_every_fold",
        "selected_binary_membership_sha256",
        "global_transitive_closure_used_as_fold_constraint",
    }
    if set(selection) != expected_schema:
        raise ValueError(f"{label}.selection schema differs")
    if selection.get("algorithm") != (
        "joint_3_outer_9_inner_core_feasibility_milp_v2"
    ):
        raise ValueError(f"{label}.selection algorithm differs")
    expected_booleans = {
        "score_free": True,
        "constant_zero_objective_only": True,
        "lexicographic_optimization_performed": False,
        "target_reached_in_every_fold": True,
        "global_transitive_closure_used_as_fold_constraint": False,
    }
    for key, expected in expected_booleans.items():
        if _boolean(selection.get(key), f"{label}.selection.{key}") is not expected:
            raise ValueError(f"{label}.selection.{key} differs")
    policy = _mapping(level.get("policy"), f"{label}.policy")
    expected_minimum = _integer(
        policy.get("minimum_validation_pairs"),
        f"{label}.policy.minimum_validation_pairs",
    )
    expected_maximum = _integer(
        policy.get("maximum_validation_pairs"),
        f"{label}.policy.maximum_validation_pairs",
    )
    if _integer(
        selection.get("target_minimum_validation_pairs"),
        f"{label}.selection.target_minimum_validation_pairs",
    ) != expected_minimum:
        raise ValueError(f"{label}.selection minimum target differs")
    if _integer(
        selection.get("target_maximum_validation_pairs"),
        f"{label}.selection.target_maximum_validation_pairs",
    ) != expected_maximum:
        raise ValueError(f"{label}.selection maximum target differs")
    claimed_counts = [
        _integer(item, f"{label}.selection.final_validation_pair_counts")
        for item in _list(
            selection.get("final_validation_pair_counts"),
            f"{label}.selection.final_validation_pair_counts",
        )
    ]
    if claimed_counts != validation_counts:
        raise ValueError(f"{label}.selection validation counts differ")
    if _list(
        selection.get("stalled_fold_indices"),
        f"{label}.selection.stalled_fold_indices",
    ):
        raise ValueError(f"{label}.selection reports stalled folds")
    observed_digest = membership_sha256(
        "\0".join(key) for key in selected
    )
    claimed_digest = validate_sha256(
        selection.get("selected_binary_membership_sha256"),
        f"{label}.selection.selected_binary_membership_sha256",
    )
    if claimed_digest != observed_digest:
        raise ValueError(
            f"{label}.selection selected binary membership differs from ledgers"
        )
    return selected, observed_digest


def reconstruct_selected_binary_membership(
    assignment: Mapping[str, Any],
) -> tuple[set[tuple[str, ...]], dict[str, str]]:
    """Reconstruct the complete 3-outer plus 9-inner selected variable set."""
    construction = _mapping(
        assignment.get("construction"),
        "assignment construction",
    )
    outer = _mapping(
        construction.get("outer_level"),
        "assignment outer level",
    )
    selected, outer_digest = _validate_level_selection_binding(
        outer,
        ("outer",),
        "outer",
    )
    level_digests = {"outer": outer_digest}
    outer_folds = _list(outer.get("folds"), "outer.folds")
    for outer_index, raw_fold in enumerate(outer_folds):
        fold = _mapping(raw_fold, f"outer.fold_{outer_index}")
        inner = _mapping(
            fold.get("inner_level"),
            f"outer.fold_{outer_index}.inner_level",
        )
        inner_selected, inner_digest = _validate_level_selection_binding(
            inner,
            ("inner", str(outer_index)),
            f"outer.fold_{outer_index}.inner",
        )
        overlap = selected.intersection(inner_selected)
        if overlap:
            raise ValueError("Outer and inner selected variable keys overlap")
        selected.update(inner_selected)
        level_digests[f"outer_{outer_index}_inner"] = inner_digest
    return selected, level_digests


def _parse_exact_ratio(
    joint_module: Any,
    value: Any,
    label: str,
) -> Any:
    raw = _list(value, label)
    if len(raw) != 2:
        raise ValueError(f"{label} must contain numerator and denominator")
    return joint_module.Ratio(
        _integer(raw[0], f"{label}.numerator"),
        _integer(raw[1], f"{label}.denominator"),
    )


def _parse_exact_level_policy(
    joint_module: Any,
    value: Any,
    label: str,
) -> Any:
    raw = _mapping(value, label)
    expected_schema = {
        "minimum_validation_pairs",
        "minimum_validation_endpoints_per_axis",
        "minimum_validation_components_per_axis",
        "validation_pair_lower_fraction",
        "validation_pair_upper_fraction",
        "minimum_training_pair_fraction",
        "minimum_training_endpoint_fraction",
        "maximum_quarantine_pair_fraction",
        "maximum_validation_pair_relative_range",
        "maximum_other_count_relative_range",
    }
    if set(raw) != expected_schema:
        raise ValueError(f"{label} exact-policy schema differs")
    return joint_module.JointLevelPolicy(
        minimum_validation_pairs=_integer(
            raw["minimum_validation_pairs"],
            f"{label}.minimum_validation_pairs",
        ),
        minimum_validation_endpoints_per_axis=_integer(
            raw["minimum_validation_endpoints_per_axis"],
            f"{label}.minimum_validation_endpoints_per_axis",
        ),
        minimum_validation_components_per_axis=_integer(
            raw["minimum_validation_components_per_axis"],
            f"{label}.minimum_validation_components_per_axis",
        ),
        validation_pair_lower_fraction=_parse_exact_ratio(
            joint_module,
            raw["validation_pair_lower_fraction"],
            f"{label}.validation_pair_lower_fraction",
        ),
        validation_pair_upper_fraction=_parse_exact_ratio(
            joint_module,
            raw["validation_pair_upper_fraction"],
            f"{label}.validation_pair_upper_fraction",
        ),
        minimum_training_pair_fraction=_parse_exact_ratio(
            joint_module,
            raw["minimum_training_pair_fraction"],
            f"{label}.minimum_training_pair_fraction",
        ),
        minimum_training_endpoint_fraction=_parse_exact_ratio(
            joint_module,
            raw["minimum_training_endpoint_fraction"],
            f"{label}.minimum_training_endpoint_fraction",
        ),
        maximum_quarantine_pair_fraction=_parse_exact_ratio(
            joint_module,
            raw["maximum_quarantine_pair_fraction"],
            f"{label}.maximum_quarantine_pair_fraction",
        ),
        maximum_validation_pair_relative_range=_parse_exact_ratio(
            joint_module,
            raw["maximum_validation_pair_relative_range"],
            f"{label}.maximum_validation_pair_relative_range",
        ),
        maximum_other_count_relative_range=_parse_exact_ratio(
            joint_module,
            raw["maximum_other_count_relative_range"],
            f"{label}.maximum_other_count_relative_range",
        ),
    )


def _adjacency_from_edges(
    entities: Iterable[str],
    edges: Iterable[tuple[str, str]],
) -> dict[str, frozenset[str]]:
    result: dict[str, set[str]] = {entity: set() for entity in entities}
    for left, right in edges:
        if left not in result or right not in result:
            raise ValueError("Independent graph edge leaves its entity universe")
        result[left].add(right)
        result[right].add(left)
    return {
        entity: frozenset(neighbors)
        for entity, neighbors in result.items()
    }


def validate_exact_joint_model_binding(
    assignment: Mapping[str, Any],
    *,
    pairs: Mapping[str, IndependentPair],
    antibodies: Mapping[str, tuple[str, str]],
    antigens: Mapping[str, str],
    edges_by_view: Mapping[str, set[tuple[str, str]]],
    joint_module: types.ModuleType,
) -> dict[str, Any]:
    """Rebuild the pinned model and audit the ledger-derived binary vector."""
    selector = _mapping(
        _mapping(assignment.get("construction"), "assignment construction").get(
            "selector"
        ),
        "assignment selector",
    )
    exact_policy = _mapping(
        selector.get("exact_joint_policy"),
        "selector exact_joint_policy",
    )
    outer_policy = _parse_exact_level_policy(
        joint_module,
        exact_policy.get("outer"),
        "selector exact_joint_policy.outer",
    )
    inner_policy = _parse_exact_level_policy(
        joint_module,
        exact_policy.get("inner"),
        "selector exact_joint_policy.inner",
    )
    if not _exact_json_equal(
        joint_module.joint_policy_payload(
            outer_policy,
            inner_policy,
        ),
        exact_policy,
    ):
        raise ValueError("Pinned joint module does not reproduce the exact policy")

    claimed_limits = _mapping(
        selector.get("reviewed_resource_limits"),
        "selector reviewed_resource_limits",
    )
    reviewed_bounds = {
        "maximum_binary_variables": (
            REVIEWED_JOINT_MAXIMUM_BINARY_VARIABLES
        ),
        "maximum_linear_constraints": (
            REVIEWED_JOINT_MAXIMUM_LINEAR_CONSTRAINTS
        ),
        "maximum_nonzero_coefficients": (
            REVIEWED_JOINT_MAXIMUM_NONZERO_COEFFICIENTS
        ),
    }
    if set(claimed_limits) != set(reviewed_bounds):
        raise ValueError("Selector reviewed-resource-limit schema differs")
    parsed_limits: dict[str, int] = {}
    for key, reviewed_maximum in reviewed_bounds.items():
        value = _integer(
            claimed_limits.get(key),
            f"selector reviewed_resource_limits.{key}",
        )
        if value <= 0 or value > reviewed_maximum:
            raise ValueError(f"Selector reviewed resource limit {key} is invalid")
        parsed_limits[key] = value
    time_limit = _number(
        selector.get("time_limit_seconds"),
        "selector time_limit_seconds",
    )
    if not 0.0 < time_limit <= REVIEWED_JOINT_TIME_LIMIT_SECONDS:
        raise ValueError("Selector time limit is outside the reviewed range")

    combined_antibody_edges = set().union(
        edges_by_view["vh"],
        edges_by_view["vl"],
        edges_by_view["concatenated_antibody"],
    )
    problem = joint_module.JointSplitProblem(
        pairs=tuple(
            joint_module.CanonicalPair(
                pair.pair_id,
                pair.antibody_id,
                pair.antigen_id,
            )
            for pair in pairs.values()
        ),
        antibody_direct_neighbors=_adjacency_from_edges(
            antibodies,
            combined_antibody_edges,
        ),
        antigen_direct_neighbors=_adjacency_from_edges(
            antigens,
            edges_by_view["antigen"],
        ),
    )
    model = joint_module.build_joint_feasibility_model(
        problem,
        outer_policy=outer_policy,
        inner_policy=inner_policy,
        resource_limits=joint_module.ResourceLimits(**parsed_limits),
    )
    claimed_matrix = validate_sha256(
        selector.get("matrix_sha256"),
        "selector matrix_sha256",
    )
    if claimed_matrix != model.matrix_sha256:
        raise ValueError(
            "Selector matrix SHA-256 differs from the independently "
            "reconstructed pinned exact model"
        )
    claimed_size = _mapping(
        selector.get("model_size"),
        "selector model_size",
    )
    if not _exact_json_equal(claimed_size, asdict(model.estimate)):
        raise ValueError(
            "Selector model-size receipt differs from the reconstructed model"
        )

    selected_keys, level_digests = reconstruct_selected_binary_membership(
        assignment
    )
    selected_digest = membership_sha256(
        "\0".join(key) for key in selected_keys
    )
    claimed_selected_digest = validate_sha256(
        selector.get("selected_binary_membership_sha256"),
        "selector selected_binary_membership_sha256",
    )
    if claimed_selected_digest != selected_digest:
        raise ValueError(
            "Selector selected binary membership differs from the 12 ledgers"
        )
    variable_keys = tuple(model.variable_keys)
    key_to_column = {
        key: column for column, key in enumerate(variable_keys)
    }
    if len(key_to_column) != len(variable_keys):
        raise RuntimeError("Reconstructed exact model contains duplicate variables")
    unknown_selected = selected_keys.difference(key_to_column)
    if unknown_selected:
        raise ValueError(
            "Ledger-derived selected variables leave the reconstructed model"
        )
    selected_columns = {key_to_column[key] for key in selected_keys}
    for row_index, row in enumerate(model.rows):
        observed = sum(
            coefficient
            for column, coefficient in row.coefficients
            if column in selected_columns
        )
        if row.lower_bound is not None and observed < row.lower_bound:
            raise ValueError(
                "Ledger-derived binary assignment violates reconstructed "
                f"model row {row_index} lower bound"
            )
        if row.upper_bound is not None and observed > row.upper_bound:
            raise ValueError(
                "Ledger-derived binary assignment violates reconstructed "
                f"model row {row_index} upper bound"
            )
    return {
        "matrix_sha256": model.matrix_sha256,
        "selected_binary_membership_sha256": selected_digest,
        "selected_binary_variable_count": len(selected_columns),
        "binary_variable_count": len(variable_keys),
        "linear_constraint_count": len(model.rows),
        "level_selected_binary_membership_sha256": level_digests,
        "matrix_recomputed_from_independent_inventory_and_exact_graphs": True,
        "shared_externally_pinned_joint_formulation_source_reused": True,
        "second_independent_joint_formulation_implementation_used": False,
        "selected_membership_recomputed_from_all_twelve_ledgers": True,
        "every_reconstructed_exact_model_row_passed": True,
        "historical_solver_execution_independently_attested": False,
        "claim_boundary": (
            "the published ledgers form a feasible binary assignment when "
            "evaluated against a matrix rebuilt from independent inputs by the "
            "same externally pinned joint-formulation implementation; this "
            "audit neither independently reimplements that formulation nor "
            "attests the historical process that found the assignment"
        ),
    }


def validate_exact_joint_selector(
    assignment: Mapping[str, Any],
    *,
    expected_joint_policy_sha256: str | None,
    expected_joint_selected_membership_sha256: str | None,
) -> tuple[bool, dict[str, Any]]:
    """Validate builder-reported solver fields before semantic model binding."""
    construction = _mapping(
        assignment.get("construction"),
        "assignment construction",
    )
    selection_method = construction.get("selection_method")
    selector = _mapping(construction.get("selector"), "assignment selector")
    if selection_method != "joint_core_milp":
        raise ValueError(
            "Only an exact joint_core_milp assignment is eligible for the "
            "core-feasibility precursor audit; greedy or sequential artifacts "
            "are rejected"
        )
    expected_selector_schema = {
        "implementation",
        "exact_nested_milp_used",
        "joint_outer_and_all_dependent_inner_folds_in_one_model",
        "outer_fold_count",
        "dependent_inner_fold_count_per_outer",
        "total_train_validation_boundary_count",
        "exact_joint_policy",
        "exact_joint_policy_sha256",
        "objective",
        "lexicographic_optimization_performed",
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
        "matrix_sha256",
        "selected_binary_membership_sha256",
        "assignment_identity_policy",
        "externally_pinned_selected_binary_membership_sha256",
        "accepted_assignment_matches_external_identity_anchor",
        "constant_zero_objective_assignment_is_solver_canonical",
        "model_size",
        "reviewed_resource_limits",
        "time_limit_seconds",
        "mechanical_role_parity_check_passed_for_all_12_folds",
        "failure_semantics",
        "claim_boundary",
    }
    if set(selector) != expected_selector_schema:
        raise ValueError("Assignment selector schema differs")
    if expected_joint_policy_sha256 is None:
        raise ValueError(
            "An external expected joint-policy SHA-256 is required for "
            "joint_core_milp audit"
        )
    expected_policy = validate_sha256(
        expected_joint_policy_sha256,
        "expected joint-policy SHA-256",
    )
    if expected_policy != LOCKED_JOINT_POLICY_SHA256:
        raise ValueError(
            "External joint-policy digest differs from the independently "
            "locked exact 3x3 policy"
        )
    exact_policy = _mapping(
        selector.get("exact_joint_policy"),
        "selector exact_joint_policy",
    )
    observed_policy = canonical_payload_sha256(exact_policy)
    claimed_policy = validate_sha256(
        selector.get("exact_joint_policy_sha256"),
        "selector exact_joint_policy_sha256",
    )
    if observed_policy != claimed_policy or claimed_policy != expected_policy:
        raise ValueError(
            "Selector exact joint-policy digest is not externally anchored: "
            f"expected={expected_policy}, claimed={claimed_policy}, "
            f"observed={observed_policy}"
        )
    expected_policy_schema = {
        "schema_version",
        "outer_fold_count",
        "dependent_inner_fold_count_per_outer",
        "outer",
        "inner",
    }
    if set(exact_policy) != expected_policy_schema:
        raise ValueError("Selector exact joint-policy schema differs")
    if (
        exact_policy["schema_version"] != "joint_nested_split_exact_policy_v2"
        or _integer(
            exact_policy["outer_fold_count"],
            "exact policy outer_fold_count",
        )
        != 3
        or _integer(
            exact_policy["dependent_inner_fold_count_per_outer"],
            "exact policy dependent_inner_fold_count_per_outer",
        )
        != 3
    ):
        raise ValueError("Selector exact joint-policy does not encode 3x3 folds")

    exact_boolean_fields = {
        "exact_nested_milp_used": True,
        "joint_outer_and_all_dependent_inner_folds_in_one_model": True,
        "lexicographic_optimization_performed": False,
        "solver_reported_success": True,
        "solver_solution_present": True,
        "solver_certified_optimal_for_constant_zero_objective": True,
        "solver_reported_mip_gap_is_finite_zero": True,
        "exact_integer_row_audit_passed": True,
        "certified_core_feasibility_assignment": True,
        "mechanical_role_parity_check_passed_for_all_12_folds": True,
        "accepted_assignment_matches_external_identity_anchor": True,
        "constant_zero_objective_assignment_is_solver_canonical": False,
    }
    for key, expected in exact_boolean_fields.items():
        if _boolean(selector.get(key), f"selector.{key}") is not expected:
            raise ValueError(f"Selector {key} does not satisfy the exact joint gate")
    exact_integer_fields = {
        "outer_fold_count": 3,
        "dependent_inner_fold_count_per_outer": 3,
        "total_train_validation_boundary_count": 12,
        "solver_status": 0,
    }
    for key, expected in exact_integer_fields.items():
        if _integer(selector.get(key), f"selector.{key}") != expected:
            raise ValueError(f"Selector {key} differs from the exact joint contract")
    exact_string_fields = {
        "implementation": "joint_3_outer_9_inner_core_feasibility_milp_v2",
        "objective": "constant_zero_feasibility",
        "solver": "scipy.optimize.milp_HiGHS",
        "solver_termination": "optimal",
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
    for key, expected in exact_string_fields.items():
        if selector.get(key) != expected:
            raise ValueError(f"Selector {key} differs from the exact joint contract")
    gap = _number(selector.get("solver_reported_mip_gap"), "selector mip_gap")
    if gap != 0.0:
        raise ValueError("Selector solver-reported MIP gap is not exactly zero")
    tolerance = _number(
        selector.get("accepted_integrality_tolerance"),
        "selector accepted_integrality_tolerance",
    )
    error = _number(
        selector.get("maximum_integrality_error"),
        "selector maximum_integrality_error",
    )
    if tolerance != 1e-7 or error < 0.0 or error > tolerance:
        raise ValueError("Selector integrality receipt is outside tolerance")
    matrix_sha256 = validate_sha256(
        selector.get("matrix_sha256"),
        "selector matrix_sha256",
    )
    selected_sha256 = validate_sha256(
        selector.get("selected_binary_membership_sha256"),
        "selector selected_binary_membership_sha256",
    )
    if expected_joint_selected_membership_sha256 is None:
        raise ValueError(
            "An external expected selected-membership SHA-256 is required"
        )
    expected_selected_sha256 = validate_sha256(
        expected_joint_selected_membership_sha256,
        "expected joint selected-membership SHA-256",
    )
    embedded_selected_anchor = validate_sha256(
        selector.get(
            "externally_pinned_selected_binary_membership_sha256"
        ),
        "selector externally pinned selected-membership SHA-256",
    )
    if (
        selector.get("assignment_identity_policy")
        != "externally_pinned_selected_binary_membership_sha256"
        or selected_sha256 != expected_selected_sha256
        or embedded_selected_anchor != expected_selected_sha256
    ):
        raise ValueError(
            "Constant-zero MILP assignment identity is not bound to the "
            "external selected-membership digest"
        )
    if selector.get("solver_message") != (
        "Optimization terminated successfully. (HiGHS Status 7: Optimal)"
    ):
        raise ValueError("Selector solver message differs from the pinned backend")
    return True, {
        "selection_method": "joint_core_milp",
        "builder_solver_receipt_fields_validated": True,
        "exact_joint_selector_verified": False,
        "exact_joint_policy_sha256": claimed_policy,
        "claimed_matrix_sha256": matrix_sha256,
        "claimed_selected_binary_membership_sha256": selected_sha256,
        "externally_pinned_selected_binary_membership_sha256": (
            expected_selected_sha256
        ),
        "constant_zero_objective_assignment_is_solver_canonical": False,
        "accepted_assignment_matches_external_identity_anchor": True,
        "solver_status": 0,
        "solver_termination": "optimal",
        "solver_reported_mip_gap": gap,
        "maximum_integrality_error": error,
        "accepted_integrality_tolerance": tolerance,
        "historical_solver_execution_independently_attested": False,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the independent auditor command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignment", type=Path, required=True)
    parser.add_argument("--assignment-sha256", required=True)
    parser.add_argument("--assignment-scientific-sha256", required=True)
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
    parser.add_argument("--expected-auditor-source-sha256", required=True)
    parser.add_argument("--builder-source", type=Path, required=True)
    parser.add_argument("--expected-builder-source-sha256", required=True)
    parser.add_argument("--joint-milp-source", type=Path, required=True)
    parser.add_argument(
        "--expected-joint-milp-source-sha256",
        required=True,
    )
    parser.add_argument("--expected-joint-policy-sha256", required=True)
    parser.add_argument(
        "--expected-joint-selected-membership-sha256",
        required=True,
    )
    parser.add_argument("--expected-pinned-replay-id", required=True)
    parser.add_argument(
        "--expected-source-provisional-attempt-id",
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=256)
    args = parser.parse_args(argv)
    for name in (
        "assignment_sha256",
        "assignment_scientific_sha256",
        "train_authority_allowlist_sha256",
        "train_canonical_manifest_sha256",
        "expected_auditor_source_sha256",
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
    if args.workers < 1 or args.chunk_size < 1:
        parser.error("--workers and --chunk-size must be positive")
    if (
        args.expected_joint_policy_sha256 is not None
        and args.expected_joint_policy_sha256 != LOCKED_JOINT_POLICY_SHA256
    ):
        parser.error(
            "--expected-joint-policy-sha256 differs from the independently "
            "locked exact 3x3 policy"
        )
    for name in (
        "expected_pinned_replay_id",
        "expected_source_provisional_attempt_id",
    ):
        try:
            value = _safe_identifier(getattr(args, name), name)
        except ValueError as error:
            parser.error(str(error))
        if len(value) > 128:
            parser.error(f"--{name.replace('_', '-')} exceeds 128 characters")
    if (
        args.expected_pinned_replay_id
        == args.expected_source_provisional_attempt_id
    ):
        parser.error(
            "--expected-pinned-replay-id must differ from "
            "--expected-source-provisional-attempt-id"
        )
    return args


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Execute the independent reconstruction and exhaustive boundary audit."""
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Create-only target already exists: {output}")
    expected_pinned_replay_id = _safe_identifier(
        getattr(args, "expected_pinned_replay_id", None),
        "expected_pinned_replay_id",
    )
    expected_source_provisional_attempt_id = _safe_identifier(
        getattr(args, "expected_source_provisional_attempt_id", None),
        "expected_source_provisional_attempt_id",
    )
    if (
        len(expected_pinned_replay_id) > 128
        or len(expected_source_provisional_attempt_id) > 128
    ):
        raise ValueError("Expected replay identifiers exceed 128 characters")
    if (
        expected_pinned_replay_id
        == expected_source_provisional_attempt_id
    ):
        raise ValueError(
            "expected_pinned_replay_id must differ from "
            "expected_source_provisional_attempt_id"
        )
    verified_references = {
        "assignment": verify_file(
            args.assignment, args.assignment_sha256, "nested assignment"
        ),
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
        "independent_auditor": verify_file(
            Path(__file__),
            args.expected_auditor_source_sha256,
            "independent nested-neighborhood auditor source",
        ),
        "externally_pinned_builder": verify_file(
            args.builder_source,
            args.expected_builder_source_sha256,
            "externally pinned builder source",
        ),
        "externally_pinned_joint_core_milp": verify_file(
            args.joint_milp_source,
            args.expected_joint_milp_source_sha256,
            "externally pinned joint-MILP source",
        ),
    }
    references = {
        key: value.reference for key, value in verified_references.items()
    }
    sources = {
        key: value.reference for key, value in verified_sources.items()
    }
    assignment = strict_json_object(
        verified_references["assignment"].payload,
        "nested assignment",
    )
    _require_exact_schema(
        assignment,
        ASSIGNMENT_TOP_LEVEL_FIELDS,
        "assignment",
    )
    if assignment.get("format_version") != ASSIGNMENT_FORMAT_VERSION:
        raise ValueError(
            "Unsupported nested assignment format_version; provisional and "
            "historical v4 and earlier artifacts are non-authoritative"
        )
    if (
        assignment.get("artifact_role")
        != "externally_pinned_core_feasibility_replay"
        or assignment.get("stage")
        != "completed_externally_pinned_core_feasibility_replay"
    ):
        raise ValueError(
            "Auditor accepts only a separately executed externally pinned replay"
        )
    if _boolean(
        assignment.get("split_domain_gate0_requirement_satisfied"),
        "assignment.split_domain_gate0_requirement_satisfied",
    ):
        raise ValueError("Core-feasibility builder falsely claims full Gate 0")
    observed_scientific = verify_scientific_digest(assignment)
    if observed_scientific != args.assignment_scientific_sha256:
        raise ValueError(
            "Externally pinned assignment scientific digest differs: "
            f"expected={args.assignment_scientific_sha256}, "
            f"observed={observed_scientific}"
        )
    validate_replay_assignment_envelope(assignment)
    if any(
        _boolean(assignment.get(key), f"assignment.{key}")
        for key in (
            "authorized_for_training",
            "authorized_for_model_selection",
            "authorized_for_nested_model_selection",
        )
    ):
        raise ValueError("Builder assignment falsely claims pre-audit authorization")
    assignment_inputs = _mapping(assignment.get("inputs"), "assignment inputs")
    if set(assignment_inputs) != {
        "train_authority_allowlist",
        "train_canonical_sequence_manifest",
    }:
        raise ValueError(
            "Assignment input schema must contain only physical train-only inputs"
        )
    verify_embedded_reference(
        assignment_inputs.get("train_authority_allowlist"),
        references["train_authority_allowlist"],
        "assignment authority reference",
    )
    verify_embedded_reference(
        assignment_inputs.get("train_canonical_sequence_manifest"),
        references["train_canonical_sequence_manifest"],
        "assignment canonical-manifest reference",
    )
    assignment_sources = _mapping(
        assignment.get("source_files"), "assignment source files"
    )
    builder_reference = _mapping(
        assignment_sources.get("builder"), "assignment builder source"
    )
    if set(builder_reference) != FILE_REFERENCE_FIELDS:
        raise ValueError("Assignment builder source has an invalid schema")
    verified_builder = sources["externally_pinned_builder"]
    if not _exact_json_equal(verified_builder, builder_reference):
        raise ValueError(
            "Assignment builder source differs from the externally pinned source"
        )
    auditor_reference = sources["independent_auditor"]
    if (
        builder_reference["path"] == auditor_reference["path"]
        or builder_reference["sha256"] == auditor_reference["sha256"]
    ):
        raise ValueError("Builder and independent auditor sources are not separated")

    construction = _mapping(
        assignment.get("construction"),
        "assignment construction",
    )
    _require_exact_schema(
        construction,
        CONSTRUCTION_FIELDS,
        "assignment construction",
    )
    if _integer(
        construction.get("fold_count"),
        "assignment construction.fold_count",
    ) != 3:
        raise ValueError("Assignment construction must contain exactly 3 folds")
    selection_seed = _integer(
        construction.get("selection_seed"),
        "assignment construction.selection_seed",
    )
    if selection_seed < 0:
        raise ValueError("Assignment construction selection_seed is negative")
    expected_construction_booleans = {
        "selection_seed_used": False,
        "selection_uses_only_sequence_graph_and_positive_pair_incidence": True,
        "model_predictions_scores_losses_embeddings_or_labels_used": False,
        "global_transitive_closure_used_as_fold_constraint": False,
    }
    for key, expected in expected_construction_booleans.items():
        if _boolean(
            construction.get(key),
            f"assignment construction.{key}",
        ) is not expected:
            raise ValueError(f"Assignment construction.{key} differs")
    selection_method = construction.get("selection_method")
    if selection_method != "joint_core_milp":
        raise ValueError(
            "Only joint_core_milp v5 artifacts are eligible; greedy and "
            "historical v4 and earlier artifacts are non-authoritative"
        )
    if construction.get("joint_assignment_mode") != "externally_pinned_replay":
        raise ValueError(
            "Auditor rejects provisional discovery and requires a separate "
            "externally pinned replay"
        )
    workflow_identity = _mapping(
        assignment.get("workflow_identity"),
        "assignment workflow_identity",
    )
    if set(workflow_identity) != {
        "joint_assignment_mode",
        "provisional_attempt_id",
        "pinned_replay_id",
        "source_provisional_attempt_id",
    }:
        raise ValueError("Assignment workflow_identity has an invalid schema")
    if (
        workflow_identity.get("joint_assignment_mode")
        != "externally_pinned_replay"
        or workflow_identity.get("provisional_attempt_id") is not None
    ):
        raise ValueError(
            "Assignment workflow_identity is not a separately numbered replay"
        )
    assignment_pinned_replay_id = _safe_identifier(
        workflow_identity.get("pinned_replay_id"),
        "assignment workflow_identity.pinned_replay_id",
    )
    assignment_source_provisional_attempt_id = _safe_identifier(
        workflow_identity.get("source_provisional_attempt_id"),
        "assignment workflow_identity.source_provisional_attempt_id",
    )
    if (
        len(assignment_pinned_replay_id) > 128
        or len(assignment_source_provisional_attempt_id) > 128
    ):
        raise ValueError("Assignment replay identifiers exceed 128 characters")
    if assignment_pinned_replay_id == assignment_source_provisional_attempt_id:
        raise ValueError(
            "Assignment replay ID must differ from its source provisional "
            "attempt ID"
        )
    if (
        assignment_pinned_replay_id != expected_pinned_replay_id
        or assignment_source_provisional_attempt_id
        != expected_source_provisional_attempt_id
    ):
        raise ValueError(
            "Assignment numbered replay identity differs from the external "
            "audit anchors"
        )
    if set(assignment_sources) != {"builder", "joint_core_milp"}:
        raise ValueError(
            "Exact joint assignment source schema must contain only the "
            "builder and joint_core_milp"
        )
    joint_reference = _mapping(
        assignment_sources.get("joint_core_milp"),
        "assignment joint-MILP source",
    )
    if set(joint_reference) != FILE_REFERENCE_FIELDS:
        raise ValueError("Assignment joint-MILP source has an invalid schema")
    verified_joint_reference = sources["externally_pinned_joint_core_milp"]
    if not _exact_json_equal(verified_joint_reference, joint_reference):
        raise ValueError(
            "Assignment joint-MILP source differs from the externally pinned source"
        )
    if (
        joint_reference["path"]
        in {builder_reference["path"], auditor_reference["path"]}
        or joint_reference["sha256"]
        in {builder_reference["sha256"], auditor_reference["sha256"]}
    ):
        raise ValueError(
            "Joint solver, builder, and auditor sources are not separated"
        )
    joint_module = load_verified_joint_module(
        verified_sources["externally_pinned_joint_core_milp"]
    )
    if joint_module.LOCKED_JOINT_POLICY_SHA256 != LOCKED_JOINT_POLICY_SHA256:
        raise ValueError(
            "Verified joint-MILP source does not expose the independently "
            "locked policy digest"
        )
    (
        selector_claims_validated,
        exact_joint_summary,
    ) = validate_exact_joint_selector(
        assignment,
        expected_joint_policy_sha256=args.expected_joint_policy_sha256,
        expected_joint_selected_membership_sha256=(
            args.expected_joint_selected_membership_sha256
        ),
    )
    exact_joint_verified: bool | None = None
    train_ids, authority = read_train_authority_allowlist(
        verified_references["train_authority_allowlist"].payload
    )
    rows, firewall = reconstruct_train_rows(
        verified_references["train_canonical_sequence_manifest"].payload,
        train_ids,
    )
    if not _exact_json_equal(
        _mapping(
            assignment.get("authority"),
            "assignment authority",
        ),
        authority,
    ):
        raise ValueError("Assignment authority receipt differs independently")
    if not _exact_json_equal(
        _mapping(
            assignment.get("data_firewall"),
            "assignment data_firewall",
        ),
        firewall,
    ):
        raise ValueError("Assignment data-firewall receipt differs independently")
    antibodies, antigens, pairs, inventory = reconstruct_entities(rows)
    if not _exact_json_equal(
        _mapping(
            assignment.get("canonical_inventory"),
            "assignment canonical inventory",
        ),
        inventory,
    ):
        raise ValueError("Assignment canonical inventory differs independently")
    outer = _mapping(
        _mapping(assignment.get("construction"), "assignment construction").get(
            "outer_level"
        ),
        "assignment outer level",
    )
    prevalidate_level_envelope(
        outer,
        pair_count=len(pairs),
        label="outer",
        level_kind="outer",
    )
    edges_by_view, independent_graph_receipts = (
        independently_reconstruct_exact_graphs(
            assignment,
            antibodies,
            antigens,
            workers=args.workers,
            chunk_size=args.chunk_size,
        )
    )
    independent_antibody_graph = _adjacency_from_edges(
        antibodies,
        set().union(
            edges_by_view["vh"],
            edges_by_view["vl"],
            edges_by_view["concatenated_antibody"],
        ),
    )
    independent_antigen_graph = _adjacency_from_edges(
        antigens,
        edges_by_view["antigen"],
    )
    antibody_component_by_entity = independently_reconstructed_component_ids(
        independent_antibody_graph
    )
    antigen_component_by_entity = independently_reconstructed_component_ids(
        independent_antigen_graph
    )
    boundaries, structural_passed, structural_summary = validate_level(
        outer,
        pairs,
        "outer",
        level_kind="outer",
        antibody_component_by_entity=antibody_component_by_entity,
        antigen_component_by_entity=antigen_component_by_entity,
    )
    if len(boundaries) != 12:
        raise ValueError(
            "Complete nested construction must expose 3 outer and 9 inner boundaries"
        )
    model_binding = validate_exact_joint_model_binding(
        assignment,
        pairs=pairs,
        antibodies=antibodies,
        antigens=antigens,
        edges_by_view=edges_by_view,
        joint_module=joint_module,
    )
    model_binding["independently_reconstructed_graph_receipt_sha256_by_view"] = {
        view: canonical_payload_sha256(independent_graph_receipts[view])
        for view in REQUIRED_VIEWS
    }
    model_binding["all_four_exact_direct_graphs_reconstructed_and_bound"] = True
    exact_joint_verified = bool(
        selector_claims_validated
        and model_binding[
            "all_four_exact_direct_graphs_reconstructed_and_bound"
        ]
        and model_binding[
            "matrix_recomputed_from_independent_inventory_and_exact_graphs"
        ]
        and model_binding[
            "selected_membership_recomputed_from_all_twelve_ledgers"
        ]
        and model_binding["every_reconstructed_exact_model_row_passed"]
    )
    exact_joint_summary.update(model_binding)
    exact_joint_summary["exact_joint_selector_verified"] = exact_joint_verified

    build_locked_aligner()
    results: list[dict[str, Any]] = []
    for boundary in boundaries:
        for view in REQUIRED_VIEWS:
            results.append(
                audit_boundary_view(
                    boundary,
                    view,
                    pairs,
                    antibodies,
                    antigens,
                    workers=args.workers,
                    chunk_size=args.chunk_size,
                )
            )
    all_views_complete = (
        len(results) == len(boundaries) * len(REQUIRED_VIEWS)
        and {
            (result["boundary"], result["sequence_view"]) for result in results
        }
        == {
            (boundary.label, view)
            for boundary in boundaries
            for view in REQUIRED_VIEWS
        }
    )
    total_candidates = sum(
        int(result["cross_boundary_unique_sequence_pair_count"])
        for result in results
    )
    total_exact = sum(int(result["exact_sequence_match_count"]) for result in results)
    total_aligned = sum(
        int(result["dynamic_programming_alignment_count"]) for result in results
    )
    total_directional_alignment_executions = sum(
        int(result["directional_dynamic_programming_alignment_execution_count"])
        for result in results
    )
    total_pruned = sum(
        int(result["provably_pruned_by_length_upper_bound_count"])
        for result in results
    )
    total_crossings = sum(
        int(result["threshold_crossing_edge_count"]) for result in results
    )
    accounting_passed = (
        total_exact + total_aligned + total_pruned == total_candidates
        and all(result["all_pairs_exactly_accounted"] for result in results)
        and total_directional_alignment_executions == 2 * total_aligned
        and all(
            result["every_aligned_pair_evaluated_in_both_orientations"]
            and result["symmetric_either_orientation_threshold_edge_rule"]
            for result in results
        )
    )
    zero_crossings = total_crossings == 0 and all(
        result["zero_threshold_crossing_edges"] for result in results
    )
    construction_claim = _boolean(
        assignment.get("construction_invariants_passed"),
        "assignment construction_invariants_passed",
    )
    independent_gates = {
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
        "independent_inventory_exactly_matches_assignment": True,
        "all_outer_and_inner_policies_match_independently_recomputed_locked_formulas": (
            structural_summary[
                "declared_policy_matches_independently_recomputed_locked_policy"
            ]
            and all(
                summary.get(
                    "declared_policy_matches_independently_recomputed_locked_policy",
                    False,
                )
                for fold in structural_summary["folds"]
                for summary in [fold.get("inner_summary", {})]
            )
        ),
        "all_outer_and_inner_partition_invariants_passed": structural_passed,
        "all_twelve_boundaries_present": len(boundaries) == 12,
        "all_four_views_completed_for_every_boundary": all_views_complete,
        "exhaustive_alignment_accounting_passed": accounting_passed,
        "zero_direct_80_80_crossing_edges_in_every_boundary_and_view": (
            zero_crossings
        ),
        "builder_construction_gates_were_passed": construction_claim,
        "exact_three_outer_nine_dependent_inner_joint_selector_verified": (
            exact_joint_verified
        ),
        "all_four_exact_direct_graphs_match_full_independent_reconstruction": (
            model_binding[
                "all_four_exact_direct_graphs_reconstructed_and_bound"
            ]
        ),
        (
            "joint_model_matrix_recomputed_from_independent_inputs_using_"
            "shared_pinned_formulation"
        ): (
            model_binding[
                "matrix_recomputed_from_independent_inventory_and_exact_graphs"
            ]
            and model_binding[
                "shared_externally_pinned_joint_formulation_source_reused"
            ]
            and not model_binding[
                "second_independent_joint_formulation_implementation_used"
            ]
        ),
        "all_twelve_ledgers_reconstruct_one_exact_model_feasible_binary_vector": (
            model_binding[
                "selected_membership_recomputed_from_all_twelve_ledgers"
            ]
            and model_binding["every_reconstructed_exact_model_row_passed"]
        ),
        "builder_and_auditor_source_bindings_are_independently_separated": True,
    }
    precursor_passed = all(independent_gates.values())
    receipt: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "stage": "completed_core_feasibility_precursor_audit",
        "valid_audit_execution": True,
        "core_feasibility_precursor_passed": precursor_passed,
        "split_domain_gate0_requirement_satisfied": False,
        "authorized_for_training": False,
        "authorized_for_model_selection": False,
        "authorized_for_nested_model_selection": False,
        "approval_status": (
            "core_feasibility_precursor_passed_non_authorizing"
            if precursor_passed
            else "core_feasibility_precursor_failed"
        ),
        "authorization_reason": (
            "This is a non-authorizing core-feasibility precursor. Full "
            "split-domain Gate 0 still requires all locked V0 canonical, "
            "lexicographic, diagnostic, and separate authority requirements."
        ),
        "claim_boundary": (
            "foldwise direct exact four-view 80/80 C3/S2-like development; "
            "not global S2, transitive lineage independence, or test evaluation"
        ),
        "independent_gates": independent_gates,
        "workflow_identity": {
            "joint_assignment_mode": "externally_pinned_replay",
            "provisional_attempt_id": None,
            "pinned_replay_id": assignment_pinned_replay_id,
            "source_provisional_attempt_id": (
                assignment_source_provisional_attempt_id
            ),
            "matches_external_audit_anchors": True,
        },
        "inputs": references,
        "source_files": sources,
        "assignment_scientific_payload_sha256": observed_scientific,
        "authority": authority,
        "data_firewall": firewall,
        "independently_reconstructed_inventory": inventory,
        "structural_audit": structural_summary,
        "exact_joint_selector_audit": exact_joint_summary,
        "boundary_count": len(boundaries),
        "required_view_count": len(REQUIRED_VIEWS),
        "boundary_view_results": results,
        "exhaustive_accounting": {
            "cross_boundary_unique_sequence_pair_count": total_candidates,
            "exact_sequence_match_count": total_exact,
            "dynamic_programming_alignment_count": total_aligned,
            "directional_dynamic_programming_alignment_execution_count": (
                total_directional_alignment_executions
            ),
            "every_aligned_pair_evaluated_in_both_orientations": (
                total_directional_alignment_executions == 2 * total_aligned
            ),
            "provably_pruned_by_length_upper_bound_count": total_pruned,
            "exact_plus_alignment_plus_safe_prune_equals_all_pairs": (
                accounting_passed
            ),
            "threshold_crossing_edge_count": total_crossings,
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
            "joint_source_executed_from_verified_bytes": True,
            "self_source_preexecution_integrity_established_by_this_process": (
                False
            ),
            "self_source_preexecution_integrity_requires_private_staged_launcher": (
                True
            ),
        },
        "publication": {
            "create_only": True,
            "raw_sequences_published": False,
            "validation_or_test_sequence_fields_parsed": False,
            "validation_or_test_labels_predictions_or_scores_read": False,
        },
        "limitations": [
            "This evidence applies only to the authority-train nested development procedure.",
            "This receipt alone never authorizes training or model selection.",
            "This receipt alone cannot satisfy the split-domain Gate 0 requirement.",
            "The auditor cannot attest its own pre-execution bytes; production use requires the externally inventoried private-stage launcher.",
            (
                "The staged three-file manifest closes only application source; "
                "the launcher, pre-execution shell state, operating system, "
                "Python runtime, scientific libraries, solver, and native "
                "dependencies remain external trust roots."
            ),
            (
                "After workflow success, the launcher publishes a conservative "
                "artifact inventory, seals and re-verifies the artifact tree, "
                "and only then publishes a separate run-root success receipt; "
                "a failed finalizer has no valid post-seal success receipt."
            ),
            "Distinct source files do not prove organizational independence or signer identity.",
            (
                "Matrix reconstruction reuses the same externally pinned "
                "joint-formulation source as the builder; inventory, graphs, "
                "ledger vector, and exact row evaluation are independent, but "
                "there is no second independent formulation implementation."
            ),
            (
                "Solver status, gap, and raw floating-point integrality fields "
                "are builder-reported; the independent audit establishes "
                "exact-model feasibility by reconstructing the matrix and "
                "ledger-derived binary vector, not by attesting historical "
                "solver execution."
            ),
            "Direct-neighborhood isolation is not transitive lineage independence.",
            "The exact selector proves core feasibility, not retention or lexicographic optimality.",
            "The remaining V0 representativeness and attainable-precision gates remain outside this receipt.",
            "No validation, test, or external performance result is produced by this audit.",
            "Unobserved cross-pairs remain retrieval decoys rather than measured non-binders.",
        ],
    }
    receipt["scientific_payload_sha256"] = canonical_payload_sha256(receipt)
    atomic_create_json(output, receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> None:
    """Run the independent auditor."""
    args = parse_args(argv)
    receipt = run(args)
    print(
        json.dumps(
            {
                "output": str(args.output.expanduser().resolve()),
                "scientific_payload_sha256": receipt[
                    "scientific_payload_sha256"
                ],
                "authorized_for_nested_model_selection": receipt[
                    "authorized_for_nested_model_selection"
                ],
                "core_feasibility_precursor_passed": receipt[
                    "core_feasibility_precursor_passed"
                ],
                "threshold_crossing_edge_count": receipt[
                    "exhaustive_accounting"
                ]["threshold_crossing_edge_count"],
            },
            sort_keys=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
