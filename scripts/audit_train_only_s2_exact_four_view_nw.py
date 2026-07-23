#!/usr/bin/env python3
"""Independently audit provisional train-only S2 folds in four sequence views.

The auditor accepts only a byte-pinned provisional assignment, its authority
split, and the canonical sequence manifest. It materializes sequence fields
only for authority-train rows, reconstructs the assignment from those rows,
and exhaustively aligns every cross-fold unique-sequence pair in VH, non-empty
VL, concatenated-antibody, and antigen views. Crossing artifacts contain only
sequence digests, lengths, fold indices, and alignment metrics.
"""

from __future__ import annotations

import argparse
import csv
import errno
import hashlib
import json
import math
import multiprocessing
import os
import platform
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Iterator, Mapping, Sequence


FORMAT_VERSION = "sabdab2_train_only_s2_exact_four_view_nw_audit_receipt_v1"
EDGE_FORMAT_VERSION = "sabdab2_train_only_s2_exact_nw_crossing_edge_v1"
ASSIGNMENT_FORMAT_VERSION = "sabdab2_train_only_two_axis_s2_fold_assignment_v1"
REQUIRED_VIEWS = ("vh", "vl", "concatenated_antibody", "antigen")
FOLD_INDICES = (0, 1, 2)
FOLD_PAIRS = ((0, 1), (0, 2), (1, 2))
VALID_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
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
HEX_DIGITS = frozenset("0123456789abcdef")


@dataclass(frozen=True)
class SequenceRecord:
    """Authority-train sequences for one canonical source row."""

    instance_id: str
    vh: str
    vl: str
    antigen: str

    @property
    def concatenated_antibody(self) -> str:
        """Return the exact heavy-then-light concatenation used by the builder."""
        return self.vh + self.vl


@dataclass(frozen=True)
class AlignmentMetrics:
    """Metrics from the first deterministic optimal global traceback."""

    score: float
    identical_residue_pair_count: int
    aligned_residue_pair_count: int
    alignment_column_count: int
    aligned_region_identity: float
    left_reciprocal_coverage: float
    right_reciprocal_coverage: float
    minimum_reciprocal_coverage: float


@dataclass(frozen=True)
class StagedArtifact:
    """One fsynced temporary artifact awaiting no-clobber publication."""

    temporary_path: Path
    target_path: Path
    sha256: str
    size_bytes: int
    edge_count: int


_WORKER_LEFT: tuple[tuple[str, str], ...] = ()
_WORKER_RIGHT: tuple[tuple[str, str], ...] = ()
_WORKER_VIEW = ""
_WORKER_LEFT_FOLD = -1
_WORKER_RIGHT_FOLD = -1
_WORKER_ALIGNER: object | None = None


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return a bounded-memory SHA-256 digest for one regular file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    """Return the SHA-256 digest of an in-memory byte string."""
    return hashlib.sha256(payload).hexdigest()


def canonical_json_bytes(payload: Any) -> bytes:
    """Serialize a JSON-compatible value deterministically and strictly."""
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
    """Hash a scientific payload with the builder's canonical convention."""
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return sha256_bytes(encoded)


def membership_sha256(values: Iterable[str]) -> str:
    """Hash sorted unique membership without exposing member identifiers."""
    encoded = "".join(f"{value}\n" for value in sorted(set(values))).encode("utf-8")
    return sha256_bytes(encoded)


def sequence_sha256(sequence: str) -> str:
    """Hash one normalized amino-acid sequence."""
    return sha256_bytes(sequence.encode("ascii"))


def validate_sha256(value: Any, label: str) -> str:
    """Validate one lowercase hexadecimal SHA-256 value."""
    digest = str(value)
    if len(digest) != 64 or any(character not in HEX_DIGITS for character in digest):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def verify_file(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    """Resolve and physically authenticate one immutable file."""
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"{label} is not a regular file: {resolved}")
    expected = validate_sha256(expected_sha256, f"{label} SHA-256")
    observed = sha256_file(resolved)
    if observed != expected:
        raise ValueError(
            f"{label} SHA-256 mismatch: expected={expected}, observed={observed}"
        )
    return {
        "path": str(resolved),
        "sha256": observed,
        "size_bytes": resolved.stat().st_size,
    }


def _reject_duplicate_pairs(
    pairs: Sequence[tuple[str, Any]],
) -> dict[str, Any]:
    """Build one object while rejecting every duplicate JSON key."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> None:
    """Reject non-standard JSON constants such as NaN and Infinity."""
    raise ValueError(f"JSON contains non-finite value {value}")


def _validate_finite_json(value: Any, label: str = "JSON") -> None:
    """Recursively reject finite-overflow floats such as 1e999."""
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{label} contains a non-finite number")
    if isinstance(value, Mapping):
        for key, child in value.items():
            _validate_finite_json(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_finite_json(child, f"{label}[{index}]")


def strict_json_object(path: Path, label: str) -> dict[str, Any]:
    """Load one JSON object with duplicate and non-finite rejection."""
    payload = json.loads(
        path.read_text(encoding="utf-8-sig"),
        object_pairs_hook=_reject_duplicate_pairs,
        parse_constant=_reject_nonfinite_constant,
    )
    _validate_finite_json(payload, label)
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def require_exact_keys(
    payload: Mapping[str, Any], required: set[str], label: str
) -> None:
    """Require an exact mapping schema without ignored fields."""
    observed = set(payload)
    if observed != required:
        raise ValueError(
            f"{label} schema mismatch: missing={sorted(required - observed)}, "
            f"unexpected={sorted(observed - required)}"
        )


def require_integer(value: Any, label: str) -> int:
    """Return one non-boolean JSON integer."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be a JSON integer")
    return value


def require_boolean(value: Any, label: str) -> bool:
    """Return one strict JSON boolean."""
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a JSON boolean")
    return value


def normalize_sequence(value: str, label: str, allow_empty: bool = False) -> str:
    """Normalize a canonical sequence and reject unsupported residues."""
    sequence = "".join(str(value).split()).upper()
    if not sequence and allow_empty:
        return ""
    if not sequence:
        raise ValueError(f"Empty sequence for {label}")
    unexpected = sorted(set(sequence) - VALID_AMINO_ACIDS)
    if unexpected:
        raise ValueError(f"Invalid residue symbols for {label}: {unexpected}")
    return sequence


def _validate_safe_instance_id(value: Any, label: str) -> str:
    """Validate an identifier that can be screened from the leading CSV field."""
    identifier = str(value)
    if not identifier:
        raise ValueError(f"{label} is empty")
    if any(character in identifier for character in (",", '"', "\r", "\n")):
        raise ValueError(f"{label} contains a CSV-unsafe character")
    return identifier


def load_authority(path: Path) -> tuple[set[str], set[str], dict[str, Any]]:
    """Load exact train membership and sealed exclusion membership."""
    payload = strict_json_object(path, "authority split")
    splits = payload.get("splits")
    if not isinstance(splits, Mapping):
        raise ValueError("Authority splits must be a JSON object")
    require_exact_keys(splits, {"train", "validation", "test"}, "authority splits")
    normalized: dict[str, list[str]] = {}
    observed: set[str] = set()
    for split_name in ("train", "validation", "test"):
        raw = splits[split_name]
        if not isinstance(raw, list):
            raise ValueError(f"Authority split {split_name!r} must be a list")
        values = [
            _validate_safe_instance_id(value, f"authority {split_name} identifier")
            for value in raw
        ]
        if len(values) != len(set(values)):
            raise ValueError(f"Authority split {split_name!r} contains duplicates")
        overlap = observed.intersection(values)
        if overlap:
            raise ValueError("Authority split memberships overlap")
        observed.update(values)
        normalized[split_name] = values
    train = set(normalized["train"])
    sealed = set(normalized["validation"]) | set(normalized["test"])
    if not train or not sealed:
        raise ValueError("Authority train and sealed pools must both be non-empty")
    return train, sealed, {
        "source_pool_split": "train",
        "authorized_instance_count": len(train),
        "authorized_membership_sha256": membership_sha256(train),
        "sealed_splits": ["validation", "test"],
        "sealed_instance_count": len(sealed),
        "sealed_membership_sha256": membership_sha256(sealed),
        "sealed_split_counts": {
            "validation": len(normalized["validation"]),
            "test": len(normalized["test"]),
        },
    }


def leading_csv_identifier(line: str) -> str:
    """Read only the first CSV field before any sequence column is parsed."""
    prefix, separator, _ = line.partition(",")
    if not separator:
        raise ValueError("Canonical manifest row has no CSV field separator")
    prefix = prefix.strip()
    if not prefix:
        raise ValueError("Canonical manifest row has an empty leading identifier")
    if prefix.startswith('"'):
        values = next(csv.reader([prefix]))
        if len(values) != 1:
            raise ValueError("Unable to parse the leading canonical identifier")
        prefix = values[0]
    return _validate_safe_instance_id(prefix, "canonical manifest identifier")


def load_authorized_sequences(
    path: Path,
    train_ids: set[str],
    sealed_ids: set[str],
) -> tuple[dict[str, SequenceRecord], dict[str, Any]]:
    """Materialize sequence fields only after authority-train authorization."""
    records: dict[str, SequenceRecord] = {}
    sealed_seen: set[str] = set()
    unknown_seen: set[str] = set()
    skipped_before_sequence_parse = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        header_line = handle.readline()
        if not header_line:
            raise ValueError("Canonical manifest is empty")
        header = next(csv.reader([header_line]))
        required = {"INSTANCE", "vh_seq", "vl_seq", "ag_seq"}
        if header[0] != "INSTANCE" or not required.issubset(header):
            raise ValueError("Canonical manifest lacks the leading ID or sequence columns")
        indices = {name: header.index(name) for name in required}
        for line_number, line in enumerate(handle, start=2):
            instance_id = leading_csv_identifier(line)
            if instance_id not in train_ids:
                skipped_before_sequence_parse += 1
                if instance_id in sealed_ids:
                    if instance_id in sealed_seen:
                        raise ValueError(
                            f"Duplicate sealed row for {instance_id!r} at line {line_number}"
                        )
                    sealed_seen.add(instance_id)
                else:
                    unknown_seen.add(instance_id)
                continue
            fields = next(csv.reader([line]))
            if len(fields) != len(header):
                raise ValueError(f"Malformed authorized row at line {line_number}")
            if str(fields[indices["INSTANCE"]]) != instance_id:
                raise ValueError(f"Leading ID mismatch at line {line_number}")
            if instance_id in records:
                raise ValueError(f"Duplicate authorized row for {instance_id!r}")
            vh = normalize_sequence(fields[indices["vh_seq"]], f"{instance_id}:VH")
            vl = normalize_sequence(
                fields[indices["vl_seq"]], f"{instance_id}:VL", allow_empty=True
            )
            antigen = normalize_sequence(
                fields[indices["ag_seq"]], f"{instance_id}:antigen"
            )
            records[instance_id] = SequenceRecord(instance_id, vh, vl, antigen)
    missing_train = train_ids - set(records)
    missing_sealed = sealed_ids - sealed_seen
    if missing_train or missing_sealed or unknown_seen:
        raise ValueError(
            "Canonical manifest does not exactly cover authority membership: "
            f"missing_train={len(missing_train)}, missing_sealed={len(missing_sealed)}, "
            f"unknown={len(unknown_seen)}"
        )
    if set(records).intersection(sealed_ids):
        raise PermissionError("A sealed sequence row was materialized")
    return records, {
        "authorized_source_split": "train",
        "authorized_rows_materialized": len(records),
        "sealed_rows_materialized": 0,
        "non_authorized_rows_skipped_before_sequence_parse": (
            skipped_before_sequence_parse
        ),
        "validation_sequence_fields_parsed": 0,
        "test_sequence_fields_parsed": 0,
        "validation_or_test_labels_read": 0,
        "validation_or_test_predictions_read": 0,
        "validation_or_test_scores_read": 0,
        "accepted_label_prediction_checkpoint_or_score_inputs": False,
        "sequence_columns_parsed_for": "authority_train_only",
    }


def _validate_embedded_file_reference(
    value: Any, label: str
) -> Mapping[str, Any]:
    """Validate one assignment-embedded path, digest, and byte-count reference."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    require_exact_keys(value, {"path", "sha256", "size_bytes"}, label)
    if not isinstance(value["path"], str) or not value["path"]:
        raise ValueError(f"{label}.path must be a non-empty string")
    validate_sha256(value["sha256"], f"{label}.sha256")
    if require_integer(value["size_bytes"], f"{label}.size_bytes") < 0:
        raise ValueError(f"{label}.size_bytes must be non-negative")
    return value


def _validate_assignment_scientific_digest(
    assignment: Mapping[str, Any], expected_scientific_sha256: str
) -> str:
    """Verify both the embedded and externally pinned scientific digest."""
    expected = validate_sha256(
        expected_scientific_sha256, "assignment scientific payload SHA-256"
    )
    claimed = validate_sha256(
        assignment.get("scientific_payload_sha256"),
        "assignment embedded scientific payload SHA-256",
    )
    scientific = dict(assignment)
    scientific.pop("scientific_payload_sha256", None)
    observed = canonical_payload_sha256(scientific)
    if claimed != expected or observed != expected:
        raise ValueError(
            "Assignment scientific payload SHA-256 mismatch: "
            f"expected={expected}, embedded={claimed}, recomputed={observed}"
        )
    return observed


def _validate_cluster_assignments(
    payload: Any,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Validate digest-to-cluster fold maps on both optimized axes."""
    if not isinstance(payload, Mapping):
        raise ValueError("cluster_assignments must be a JSON object")
    require_exact_keys(payload, {"antibody", "antigen"}, "cluster_assignments")
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for axis in ("antibody", "antigen"):
        rows = payload[axis]
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"cluster_assignments.{axis} must be a non-empty list")
        by_digest: dict[str, dict[str, Any]] = {}
        fold_by_cluster: dict[str, int] = {}
        ordered_digests: list[str] = []
        for index, raw in enumerate(rows):
            label = f"cluster_assignments.{axis}[{index}]"
            if not isinstance(raw, Mapping):
                raise ValueError(f"{label} must be a JSON object")
            require_exact_keys(
                raw,
                {
                    "sequence_sha256",
                    "cluster_id",
                    "cluster_representative_sequence_sha256",
                    "fold_index",
                },
                label,
            )
            digest = validate_sha256(raw["sequence_sha256"], f"{label}.sequence")
            representative = validate_sha256(
                raw["cluster_representative_sequence_sha256"],
                f"{label}.representative",
            )
            cluster_id = str(raw["cluster_id"])
            expected_prefix = f"{axis}_cluster_"
            if not cluster_id.startswith(expected_prefix):
                raise ValueError(f"{label}.cluster_id has an unexpected prefix")
            fold = require_integer(raw["fold_index"], f"{label}.fold_index")
            if fold not in FOLD_INDICES:
                raise ValueError(f"{label}.fold_index is outside the locked folds")
            if digest in by_digest:
                raise ValueError(f"Duplicate {axis} sequence digest in assignment")
            prior_fold = fold_by_cluster.setdefault(cluster_id, fold)
            if prior_fold != fold:
                raise ValueError(f"Cluster {cluster_id!r} spans multiple folds")
            by_digest[digest] = {
                "cluster_id": cluster_id,
                "cluster_representative_sequence_sha256": representative,
                "fold_index": fold,
            }
            ordered_digests.append(digest)
        if ordered_digests != sorted(ordered_digests):
            raise ValueError(f"cluster_assignments.{axis} is not digest-sorted")
        result[axis] = by_digest
    return result["antibody"], result["antigen"]


def _validate_retained_folds(
    payload: Any, train_ids: set[str]
) -> tuple[dict[str, int], list[Mapping[str, Any]]]:
    """Validate the three retained source ledgers and return ID-to-fold mapping."""
    if not isinstance(payload, list) or len(payload) != len(FOLD_INDICES):
        raise ValueError("retained_folds must contain exactly three entries")
    fold_by_source: dict[str, int] = {}
    normalized: list[Mapping[str, Any]] = []
    required = {
        "fold_index",
        "retained_unique_exact_pair_count",
        "retained_source_row_count",
        "retained_exact_pairs",
        "retained_exact_pair_membership_sha256",
        "retained_source_instance_ids",
        "retained_source_membership_sha256",
    }
    for expected_fold, raw in zip(FOLD_INDICES, payload):
        label = f"retained_folds[{expected_fold}]"
        if not isinstance(raw, Mapping):
            raise ValueError(f"{label} must be a JSON object")
        require_exact_keys(raw, required, label)
        fold = require_integer(raw["fold_index"], f"{label}.fold_index")
        if fold != expected_fold:
            raise ValueError("retained_folds are not in contiguous canonical order")
        identifiers_raw = raw["retained_source_instance_ids"]
        if not isinstance(identifiers_raw, list):
            raise ValueError(f"{label}.retained_source_instance_ids must be a list")
        identifiers = [
            _validate_safe_instance_id(value, f"{label} source ID")
            for value in identifiers_raw
        ]
        if identifiers != sorted(identifiers) or len(identifiers) != len(set(identifiers)):
            raise ValueError(f"{label} source IDs must be unique and sorted")
        if not set(identifiers).issubset(train_ids):
            raise PermissionError(f"{label} includes a non-authority-train identifier")
        if require_integer(raw["retained_source_row_count"], f"{label} row count") != len(
            identifiers
        ):
            raise ValueError(f"{label} source-row count is inconsistent")
        if validate_sha256(
            raw["retained_source_membership_sha256"], f"{label} source membership"
        ) != membership_sha256(identifiers):
            raise ValueError(f"{label} source membership digest is inconsistent")
        pairs = raw["retained_exact_pairs"]
        if not isinstance(pairs, list):
            raise ValueError(f"{label}.retained_exact_pairs must be a list")
        if require_integer(
            raw["retained_unique_exact_pair_count"], f"{label} exact-pair count"
        ) != len(pairs):
            raise ValueError(f"{label} exact-pair count is inconsistent")
        pair_ids: list[str] = []
        for pair_index, pair in enumerate(pairs):
            if not isinstance(pair, Mapping):
                raise ValueError(f"{label} pair {pair_index} must be a JSON object")
            pair_ids.append(
                validate_sha256(
                    pair.get("exact_pair_id_sha256"),
                    f"{label} pair {pair_index} ID",
                )
            )
        if pair_ids != sorted(pair_ids) or len(pair_ids) != len(set(pair_ids)):
            raise ValueError(f"{label} exact pairs must be unique and digest-sorted")
        if validate_sha256(
            raw["retained_exact_pair_membership_sha256"],
            f"{label} exact-pair membership",
        ) != membership_sha256(pair_ids):
            raise ValueError(f"{label} exact-pair membership digest is inconsistent")
        for identifier in identifiers:
            if identifier in fold_by_source:
                raise ValueError("A retained source row appears in multiple folds")
            fold_by_source[identifier] = fold
        normalized.append(raw)
    return fold_by_source, normalized


def _validate_discarded_source_ledger(
    payload: Any, train_ids: set[str]
) -> tuple[set[str], Mapping[str, Any]]:
    """Validate discarded-source membership without reading any sequences."""
    if not isinstance(payload, Mapping):
        raise ValueError("discarded_cross_fold_ledger must be a JSON object")
    required = {
        "discard_policy",
        "discarded_unique_exact_pair_count",
        "discarded_source_row_count",
        "discarded_cross_fold_edges",
        "discarded_source_rows",
        "discarded_exact_pair_membership_sha256",
        "discarded_source_membership_sha256",
    }
    require_exact_keys(payload, required, "discarded_cross_fold_ledger")
    if payload["discard_policy"] != (
        "discard_every_positive_edge_whose_endpoint_folds_differ"
    ):
        raise ValueError("Unexpected cross-fold discard policy")
    rows = payload["discarded_source_rows"]
    edges = payload["discarded_cross_fold_edges"]
    if not isinstance(rows, list) or not isinstance(edges, list):
        raise ValueError("Discarded rows and edges must be lists")
    identifiers: list[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"discarded source row {index} must be a JSON object")
        identifier = _validate_safe_instance_id(
            row.get("source_instance_id"), f"discarded source row {index} ID"
        )
        if identifier not in train_ids:
            raise PermissionError("Discarded ledger contains a non-authority-train ID")
        identifiers.append(identifier)
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Discarded source ledger contains duplicate rows")
    if require_integer(payload["discarded_source_row_count"], "discarded row count") != len(
        rows
    ):
        raise ValueError("Discarded source-row count is inconsistent")
    if validate_sha256(
        payload["discarded_source_membership_sha256"],
        "discarded source membership",
    ) != membership_sha256(identifiers):
        raise ValueError("Discarded source membership digest is inconsistent")
    edge_ids = [
        validate_sha256(edge.get("exact_pair_id_sha256"), f"discarded edge {index} ID")
        if isinstance(edge, Mapping)
        else (_ for _ in ()).throw(
            ValueError(f"discarded edge {index} must be a JSON object")
        )
        for index, edge in enumerate(edges)
    ]
    if edge_ids != sorted(edge_ids) or len(edge_ids) != len(set(edge_ids)):
        raise ValueError("Discarded edges must be unique and digest-sorted")
    if require_integer(
        payload["discarded_unique_exact_pair_count"], "discarded exact-pair count"
    ) != len(edges):
        raise ValueError("Discarded exact-pair count is inconsistent")
    if validate_sha256(
        payload["discarded_exact_pair_membership_sha256"],
        "discarded exact-pair membership",
    ) != membership_sha256(edge_ids):
        raise ValueError("Discarded exact-pair membership digest is inconsistent")
    return set(identifiers), payload


def load_and_validate_assignment(
    assignment_path: Path,
    expected_scientific_sha256: str,
    assignment_reference: Mapping[str, Any],
    authority_reference: Mapping[str, Any],
    manifest_reference: Mapping[str, Any],
    train_ids: set[str],
    authority_summary: Mapping[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, int],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    """Authenticate the provisional assignment and its train-only ledgers."""
    assignment = strict_json_object(assignment_path, "S2 assignment")
    top_level_keys = {
        "approval_status",
        "artifact_section_bindings",
        "authority",
        "authorized_for_model_selection",
        "authorized_for_training",
        "cluster_assignments",
        "config",
        "config_payload_sha256",
        "data_firewall",
        "determinism",
        "discarded_cross_fold_ledger",
        "entity_cluster_summary",
        "format_version",
        "inputs",
        "intended_use",
        "ledger_accounting",
        "limitations",
        "optimization_problem",
        "required_downstream_gate",
        "retained_folds",
        "runtime",
        "scientific_payload_sha256",
        "similarity_evidence",
        "solution_summary",
        "solver_result",
        "source_files",
        "stage",
        "valid_as_provisional_assignment",
    }
    require_exact_keys(assignment, top_level_keys, "S2 assignment")
    _validate_assignment_scientific_digest(assignment, expected_scientific_sha256)
    if assignment["format_version"] != ASSIGNMENT_FORMAT_VERSION:
        raise ValueError("Unsupported S2 assignment format_version")
    if assignment["stage"] != "provisional_train_only_two_axis_s2_fold_assignment":
        raise ValueError("Unexpected S2 assignment stage")
    if assignment["intended_use"] != "exact_four_view_audit_input_only":
        raise ValueError("Unexpected S2 assignment intended_use")
    if assignment["approval_status"] != (
        "unapproved_pending_exact_four_view_nw_audit"
    ):
        raise ValueError("Assignment was not in the required unapproved state")
    if not require_boolean(
        assignment["valid_as_provisional_assignment"], "valid_as_provisional_assignment"
    ):
        raise ValueError("Assignment is not valid as a provisional assignment")
    if require_boolean(
        assignment["authorized_for_training"], "authorized_for_training"
    ) or require_boolean(
        assignment["authorized_for_model_selection"],
        "authorized_for_model_selection",
    ):
        raise ValueError("The provisional assignment must not self-authorize use")

    inputs = assignment["inputs"]
    if not isinstance(inputs, Mapping):
        raise ValueError("Assignment inputs must be a JSON object")
    require_exact_keys(
        inputs,
        {
            "authority_split",
            "canonical_sequence_manifest",
            "antibody_mmseqs2_similarity",
            "antigen_mmseqs2_similarity",
            "builder_config",
        },
        "assignment inputs",
    )
    for label, reference in inputs.items():
        _validate_embedded_file_reference(reference, f"assignment inputs.{label}")
    if inputs["authority_split"]["sha256"] != authority_reference["sha256"]:
        raise ValueError("Assignment does not bind the supplied authority bytes")
    if inputs["canonical_sequence_manifest"]["sha256"] != manifest_reference["sha256"]:
        raise ValueError("Assignment does not bind the supplied canonical manifest bytes")

    config = assignment["config"]
    if not isinstance(config, Mapping):
        raise ValueError("Assignment config must be a JSON object")
    require_exact_keys(
        config,
        {
            "axis_balance_tolerance",
            "deterministic_policy",
            "fold_count",
            "format_version",
            "identity_threshold",
            "optimized_sequence_views",
            "reciprocal_coverage_threshold",
            "similarity_backend",
            "time_limit_seconds",
        },
        "assignment config",
    )
    if require_integer(config["fold_count"], "assignment fold_count") != 3:
        raise ValueError("Assignment must contain exactly three folds")
    if config["identity_threshold"] != 0.8 or config[
        "reciprocal_coverage_threshold"
    ] != 0.8:
        raise ValueError("Assignment thresholds differ from the locked 0.8 policy")
    if config["optimized_sequence_views"] != [
        "concatenated_antibody",
        "antigen",
    ]:
        raise ValueError("Assignment optimized views differ from the S2 contract")
    if validate_sha256(
        assignment["config_payload_sha256"], "assignment config payload"
    ) != canonical_payload_sha256(config):
        raise ValueError("Assignment config payload digest is inconsistent")

    gate = assignment["required_downstream_gate"]
    if not isinstance(gate, Mapping):
        raise ValueError("required_downstream_gate must be a JSON object")
    require_exact_keys(
        gate,
        {
            "gate_name",
            "required_sequence_views",
            "required_identity_threshold",
            "required_reciprocal_coverage_threshold",
            "required_result",
            "must_bind_this_artifact_file_sha256",
            "must_bind_this_artifact_scientific_payload_sha256",
            "approval_may_only_be_granted_by_separate_create_only_audit_receipt",
            "current_gate_passed",
        },
        "required_downstream_gate",
    )
    expected_gate_values = {
        "gate_name": "exact_four_view_needleman_wunsch_cross_fold_audit",
        "required_sequence_views": list(REQUIRED_VIEWS),
        "required_identity_threshold": 0.8,
        "required_reciprocal_coverage_threshold": 0.8,
        "required_result": (
            "zero_cross_fold_pairs_at_or_above_threshold_in_every_view"
        ),
        "must_bind_this_artifact_file_sha256": True,
        "must_bind_this_artifact_scientific_payload_sha256": True,
        "approval_may_only_be_granted_by_separate_create_only_audit_receipt": True,
        "current_gate_passed": False,
    }
    if dict(gate) != expected_gate_values:
        raise ValueError("Assignment downstream gate differs from the locked contract")
    if assignment["authority"] != dict(authority_summary):
        raise ValueError("Assignment authority summary differs from physical authority")

    bindings = assignment["artifact_section_bindings"]
    if not isinstance(bindings, Mapping):
        raise ValueError("artifact_section_bindings must be a JSON object")
    expected_bindings = {
        "cluster_assignments_sha256": canonical_payload_sha256(
            assignment["cluster_assignments"]
        ),
        "retained_folds_sha256": canonical_payload_sha256(
            {"retained_folds": assignment["retained_folds"]}
        ),
        "discarded_cross_fold_ledger_sha256": canonical_payload_sha256(
            assignment["discarded_cross_fold_ledger"]
        ),
        "accounting_sha256": canonical_payload_sha256(
            assignment["ledger_accounting"]
        ),
    }
    if dict(bindings) != expected_bindings:
        raise ValueError("Assignment section bindings do not match their payloads")

    antibody_map, antigen_map = _validate_cluster_assignments(
        assignment["cluster_assignments"]
    )
    fold_by_source, _ = _validate_retained_folds(
        assignment["retained_folds"], train_ids
    )
    discarded_ids, _ = _validate_discarded_source_ledger(
        assignment["discarded_cross_fold_ledger"], train_ids
    )
    if set(fold_by_source).intersection(discarded_ids):
        raise ValueError("Retained and discarded source ledgers overlap")
    if set(fold_by_source) | discarded_ids != train_ids:
        raise ValueError("Retained and discarded ledgers do not exhaust authority train")
    if assignment_reference["sha256"] != sha256_file(assignment_path):
        raise RuntimeError("Assignment bytes changed during parsing")
    return assignment, fold_by_source, antibody_map, antigen_map


def _exact_pair_id(antibody_digest: str, antigen_digest: str) -> str:
    """Reproduce the builder's collision-resistant exact-pair identifier."""
    return sequence_sha256(f"{antibody_digest}\0{antigen_digest}")


def validate_assignment_against_sequences(
    assignment: Mapping[str, Any],
    records: Mapping[str, SequenceRecord],
    fold_by_source: Mapping[str, int],
    antibody_map: Mapping[str, Mapping[str, Any]],
    antigen_map: Mapping[str, Mapping[str, Any]],
) -> None:
    """Independently reconstruct every retained and discarded assignment row."""
    digests_by_source: dict[str, tuple[str, str]] = {}
    for identifier, record in records.items():
        digests_by_source[identifier] = (
            sequence_sha256(record.concatenated_antibody),
            sequence_sha256(record.antigen),
        )
    observed_antibodies = {value[0] for value in digests_by_source.values()}
    observed_antigens = {value[1] for value in digests_by_source.values()}
    if set(antibody_map) != observed_antibodies:
        raise ValueError("Assignment antibody digest inventory differs from authority train")
    if set(antigen_map) != observed_antigens:
        raise ValueError("Assignment antigen digest inventory differs from authority train")
    for axis, mapping in (("antibody", antibody_map), ("antigen", antigen_map)):
        for digest, row in mapping.items():
            representative = str(row["cluster_representative_sequence_sha256"])
            representative_row = mapping.get(representative)
            if representative_row is None:
                raise ValueError(f"{axis} cluster representative is absent from inventory")
            if (
                representative_row["cluster_id"] != row["cluster_id"]
                or representative_row["fold_index"] != row["fold_index"]
            ):
                raise ValueError(f"{axis} cluster representative metadata is inconsistent")

    source_ids_by_pair: defaultdict[tuple[str, str], list[str]] = defaultdict(list)
    for identifier, pair in sorted(digests_by_source.items()):
        source_ids_by_pair[pair].append(identifier)

    retained_pairs_by_fold: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    retained_ids_by_fold: defaultdict[int, list[str]] = defaultdict(list)
    retained_pair_ids_by_fold: defaultdict[int, list[str]] = defaultdict(list)
    discarded_edges: list[dict[str, Any]] = []
    discarded_rows: list[dict[str, Any]] = []
    reconstructed_fold_by_source: dict[str, int] = {}
    for (antibody, antigen), source_ids in sorted(source_ids_by_pair.items()):
        antibody_row = antibody_map[antibody]
        antigen_row = antigen_map[antigen]
        antibody_fold = require_integer(
            antibody_row["fold_index"], "antibody cluster fold"
        )
        antigen_fold = require_integer(antigen_row["fold_index"], "antigen cluster fold")
        pair_id = _exact_pair_id(antibody, antigen)
        common = {
            "exact_pair_id_sha256": pair_id,
            "antibody_sequence_sha256": antibody,
            "antigen_sequence_sha256": antigen,
            "antibody_cluster_id": antibody_row["cluster_id"],
            "antigen_cluster_id": antigen_row["cluster_id"],
            "antibody_fold_index": antibody_fold,
            "antigen_fold_index": antigen_fold,
            "source_instance_ids": sorted(source_ids),
            "source_row_count": len(source_ids),
        }
        if antibody_fold == antigen_fold:
            retained_pairs_by_fold[antibody_fold].append(
                {**common, "fold_index": antibody_fold}
            )
            retained_pair_ids_by_fold[antibody_fold].append(pair_id)
            retained_ids_by_fold[antibody_fold].extend(source_ids)
            for identifier in source_ids:
                reconstructed_fold_by_source[identifier] = antibody_fold
        else:
            discarded_edges.append(
                {**common, "discard_reason": "cross_fold_cluster_assignment"}
            )
            discarded_rows.extend(
                {
                    "source_instance_id": identifier,
                    "exact_pair_id_sha256": pair_id,
                    "antibody_fold_index": antibody_fold,
                    "antigen_fold_index": antigen_fold,
                    "discard_reason": "cross_fold_cluster_assignment",
                }
                for identifier in sorted(source_ids)
            )
    if reconstructed_fold_by_source != dict(fold_by_source):
        raise ValueError("Retained source folds disagree with reconstructed axis folds")

    expected_retained_folds: list[dict[str, Any]] = []
    for fold in FOLD_INDICES:
        pairs = sorted(
            retained_pairs_by_fold[fold], key=lambda row: row["exact_pair_id_sha256"]
        )
        source_ids = sorted(retained_ids_by_fold[fold])
        pair_ids = sorted(retained_pair_ids_by_fold[fold])
        expected_retained_folds.append(
            {
                "fold_index": fold,
                "retained_unique_exact_pair_count": len(pairs),
                "retained_source_row_count": len(source_ids),
                "retained_exact_pairs": pairs,
                "retained_exact_pair_membership_sha256": membership_sha256(pair_ids),
                "retained_source_instance_ids": source_ids,
                "retained_source_membership_sha256": membership_sha256(source_ids),
            }
        )
    if assignment["retained_folds"] != expected_retained_folds:
        raise ValueError("Retained folds fail independent sequence reconstruction")

    discarded_edges.sort(key=lambda row: row["exact_pair_id_sha256"])
    discarded_rows.sort(
        key=lambda row: (row["source_instance_id"], row["exact_pair_id_sha256"])
    )
    expected_discarded = {
        "discard_policy": "discard_every_positive_edge_whose_endpoint_folds_differ",
        "discarded_unique_exact_pair_count": len(discarded_edges),
        "discarded_source_row_count": len(discarded_rows),
        "discarded_cross_fold_edges": discarded_edges,
        "discarded_source_rows": discarded_rows,
        "discarded_exact_pair_membership_sha256": membership_sha256(
            row["exact_pair_id_sha256"] for row in discarded_edges
        ),
        "discarded_source_membership_sha256": membership_sha256(
            row["source_instance_id"] for row in discarded_rows
        ),
    }
    if assignment["discarded_cross_fold_ledger"] != expected_discarded:
        raise ValueError("Discard ledger fails independent sequence reconstruction")

    retained_source_ids = sorted(reconstructed_fold_by_source)
    discarded_source_ids = sorted(row["source_instance_id"] for row in discarded_rows)
    retained_pair_ids = sorted(
        pair["exact_pair_id_sha256"]
        for fold in expected_retained_folds
        for pair in fold["retained_exact_pairs"]
    )
    discarded_pair_ids = sorted(
        row["exact_pair_id_sha256"] for row in discarded_edges
    )
    all_pair_ids = sorted(
        _exact_pair_id(antibody, antigen)
        for antibody, antigen in source_ids_by_pair
    )
    expected_accounting = {
        "valid": True,
        "all_authority_train_source_row_count": len(records),
        "retained_source_row_count": len(retained_source_ids),
        "discarded_source_row_count": len(discarded_source_ids),
        "all_unique_exact_pair_count": len(all_pair_ids),
        "retained_unique_exact_pair_count": len(retained_pair_ids),
        "discarded_unique_exact_pair_count": len(discarded_pair_ids),
        "retained_and_discarded_source_rows_disjoint": True,
        "retained_and_discarded_source_rows_exhaustive": True,
        "retained_and_discarded_exact_pairs_disjoint": True,
        "retained_and_discarded_exact_pairs_exhaustive": True,
        "all_source_membership_sha256": membership_sha256(records),
        "all_exact_pair_membership_sha256": membership_sha256(all_pair_ids),
    }
    if assignment["ledger_accounting"] != expected_accounting:
        raise ValueError("Assignment accounting fails independent reconstruction")


def build_fold_inventories(
    records: Mapping[str, SequenceRecord],
    fold_by_source: Mapping[str, int],
) -> dict[str, tuple[tuple[tuple[str, str], ...], ...]]:
    """Build digest-keyed, within-fold unique sequence inventories."""
    mutable: dict[str, list[dict[str, str]]] = {
        view: [dict() for _ in FOLD_INDICES] for view in REQUIRED_VIEWS
    }
    for identifier, fold in sorted(fold_by_source.items()):
        if fold not in FOLD_INDICES:
            raise ValueError(f"Unexpected fold index for retained source {identifier!r}")
        record = records[identifier]
        view_sequences = {
            "vh": record.vh,
            "vl": record.vl,
            "concatenated_antibody": record.concatenated_antibody,
            "antigen": record.antigen,
        }
        for view, sequence in view_sequences.items():
            if view == "vl" and not sequence:
                continue
            digest = sequence_sha256(sequence)
            previous = mutable[view][fold].setdefault(digest, sequence)
            if previous != sequence:
                raise RuntimeError("A SHA-256 sequence collision was detected")
    result: dict[str, tuple[tuple[tuple[str, str], ...], ...]] = {}
    for view in REQUIRED_VIEWS:
        folds = tuple(
            tuple(sorted(inventory.items())) for inventory in mutable[view]
        )
        if any(not inventory for inventory in folds):
            raise ValueError(f"Every fold must contain a non-empty {view} inventory")
        result[view] = folds
    return result


def inventory_sha256(
    folds: Sequence[Sequence[tuple[str, str]]],
) -> str:
    """Hash fold, sequence digest, and length without exposing raw sequences."""
    rows = "".join(
        f"{fold}\t{digest}\t{len(sequence)}\n"
        for fold, inventory in enumerate(folds)
        for digest, sequence in inventory
    )
    return sha256_bytes(rows.encode("ascii"))


def summarize_inventory(
    folds: Sequence[Sequence[tuple[str, str]]],
) -> dict[str, Any]:
    """Summarize one four-view input inventory without raw sequences."""
    per_fold = []
    for fold, inventory in enumerate(folds):
        rows = [f"{digest}\t{len(sequence)}" for digest, sequence in inventory]
        per_fold.append(
            {
                "fold_index": fold,
                "unique_sequence_count": len(inventory),
                "digest_and_length_inventory_sha256": sha256_bytes(
                    ("\n".join(rows) + "\n").encode("ascii")
                ),
            }
        )
    return {
        "folds": per_fold,
        "all_fold_digest_and_length_inventory_sha256": inventory_sha256(folds),
        "raw_sequences_or_instance_ids_included": False,
    }


def metrics_from_aligned_blocks(
    left: str,
    right: str,
    left_blocks: Sequence[Sequence[int]],
    right_blocks: Sequence[Sequence[int]],
    score: float,
) -> AlignmentMetrics:
    """Compute the locked identity and reciprocal-coverage denominators."""
    if not left or not right:
        raise ValueError("Alignment inputs must be non-empty")
    if len(left_blocks) != len(right_blocks):
        raise ValueError("Aligned block lists have different lengths")
    paired = 0
    identical = 0
    for left_block, right_block in zip(left_blocks, right_blocks):
        if len(left_block) != 2 or len(right_block) != 2:
            raise ValueError("Aligned blocks must contain start and end coordinates")
        left_start, left_end = (int(value) for value in left_block)
        right_start, right_end = (int(value) for value in right_block)
        left_width = left_end - left_start
        right_width = right_end - right_start
        if left_width <= 0 or left_width != right_width:
            raise ValueError("Aligned residue blocks must have equal positive widths")
        paired += left_width
        identical += sum(
            left_residue == right_residue
            for left_residue, right_residue in zip(
                left[left_start:left_end], right[right_start:right_end]
            )
        )
    if paired <= 0:
        raise ValueError("The selected global alignment has no residue-residue column")
    numeric_score = float(score)
    if not math.isfinite(numeric_score):
        raise ValueError("Alignment score is non-finite")
    left_coverage = paired / len(left)
    right_coverage = paired / len(right)
    return AlignmentMetrics(
        score=numeric_score,
        identical_residue_pair_count=identical,
        aligned_residue_pair_count=paired,
        alignment_column_count=len(left) + len(right) - paired,
        aligned_region_identity=identical / paired,
        left_reciprocal_coverage=left_coverage,
        right_reciprocal_coverage=right_coverage,
        minimum_reciprocal_coverage=min(left_coverage, right_coverage),
    )


def _blosum62_payload_sha256(matrix: Any) -> str:
    """Hash the complete labeled substitution matrix deterministically."""
    alphabet = str(matrix.alphabet)
    payload = {
        "alphabet": alphabet,
        "scores": [
            [float(matrix[left, right]) for right in alphabet] for left in alphabet
        ],
    }
    return canonical_payload_sha256(payload)


def build_locked_aligner() -> object:
    """Construct and verify the byte-independent locked alignment backend."""
    try:
        import Bio
        from Bio.Align import PairwiseAligner, substitution_matrices
    except ImportError as error:
        raise RuntimeError("Biopython is required for the exact NW audit") from error
    if Bio.__version__ != EXPECTED_BIOPYTHON_VERSION:
        raise RuntimeError(
            "Biopython version mismatch: "
            f"expected={EXPECTED_BIOPYTHON_VERSION}, observed={Bio.__version__}"
        )
    matrix = substitution_matrices.load("BLOSUM62")
    observed_matrix_sha256 = _blosum62_payload_sha256(matrix)
    if observed_matrix_sha256 != EXPECTED_BLOSUM62_PAYLOAD_SHA256:
        raise RuntimeError(
            "BLOSUM62 matrix payload mismatch: "
            f"expected={EXPECTED_BLOSUM62_PAYLOAD_SHA256}, "
            f"observed={observed_matrix_sha256}"
        )
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.substitution_matrix = matrix
    aligner.open_gap_score = -GAP_OPEN_PENALTY
    aligner.extend_gap_score = -GAP_EXTEND_PENALTY
    if aligner.mode != "global":
        raise AssertionError("PairwiseAligner did not retain global mode")
    if float(aligner.open_gap_score) != -GAP_OPEN_PENALTY:
        raise AssertionError("PairwiseAligner did not retain the gap-open penalty")
    if float(aligner.extend_gap_score) != -GAP_EXTEND_PENALTY:
        raise AssertionError("PairwiseAligner did not retain the gap-extend penalty")
    return aligner


def global_alignment_metrics(
    left: str, right: str, aligner: object
) -> AlignmentMetrics:
    """Return metrics for the first deterministic optimal global traceback."""
    alignment = next(iter(aligner.align(left, right)))
    aligned = alignment.aligned
    return metrics_from_aligned_blocks(
        left,
        right,
        aligned[0],
        aligned[1],
        float(alignment.score),
    )


def is_safe_length_prune(left_length: int, right_length: int) -> bool:
    """Prove that reciprocal coverage cannot reach the locked threshold."""
    if left_length <= 0 or right_length <= 0:
        raise ValueError("Sequence lengths must be positive")
    shorter = min(left_length, right_length)
    longer = max(left_length, right_length)
    return (
        COVERAGE_DENOMINATOR * shorter
        < COVERAGE_NUMERATOR * longer
    )


def meets_locked_threshold(metrics: AlignmentMetrics, left_length: int, right_length: int) -> bool:
    """Apply the locked thresholds with exact integer inequalities."""
    paired = metrics.aligned_residue_pair_count
    identical = metrics.identical_residue_pair_count
    return (
        IDENTITY_DENOMINATOR * identical >= IDENTITY_NUMERATOR * paired
        and COVERAGE_DENOMINATOR * paired >= COVERAGE_NUMERATOR * left_length
        and COVERAGE_DENOMINATOR * paired >= COVERAGE_NUMERATOR * right_length
    )


def edge_payload(
    view: str,
    left_fold: int,
    right_fold: int,
    left_digest: str,
    right_digest: str,
    left_sequence: str,
    right_sequence: str,
    metrics: AlignmentMetrics,
) -> dict[str, Any]:
    """Serialize one threshold crossing without sequences or instance IDs."""
    if view not in REQUIRED_VIEWS or (left_fold, right_fold) not in FOLD_PAIRS:
        raise ValueError("Invalid crossing-edge view or fold pair")
    return {
        "format_version": EDGE_FORMAT_VERSION,
        "sequence_view": view,
        "left_fold_index": left_fold,
        "right_fold_index": right_fold,
        "left_sequence_sha256": validate_sha256(left_digest, "left sequence digest"),
        "right_sequence_sha256": validate_sha256(right_digest, "right sequence digest"),
        "left_length": len(left_sequence),
        "right_length": len(right_sequence),
        "alignment": {
            "score": metrics.score,
            "identical_residue_pair_count": metrics.identical_residue_pair_count,
            "aligned_residue_pair_count": metrics.aligned_residue_pair_count,
            "alignment_column_count": metrics.alignment_column_count,
            "aligned_region_identity": metrics.aligned_region_identity,
            "left_reciprocal_coverage": metrics.left_reciprocal_coverage,
            "right_reciprocal_coverage": metrics.right_reciprocal_coverage,
            "minimum_reciprocal_coverage": metrics.minimum_reciprocal_coverage,
        },
        "locked_threshold_rule_met": True,
        "raw_sequences_or_instance_ids_included": False,
    }


def _initialize_worker(
    left: tuple[tuple[str, str], ...],
    right: tuple[tuple[str, str], ...],
    view: str,
    left_fold: int,
    right_fold: int,
) -> None:
    """Initialize immutable process-local alignment state."""
    global _WORKER_ALIGNER
    global _WORKER_LEFT
    global _WORKER_LEFT_FOLD
    global _WORKER_RIGHT
    global _WORKER_RIGHT_FOLD
    global _WORKER_VIEW
    _WORKER_LEFT = left
    _WORKER_RIGHT = right
    _WORKER_VIEW = view
    _WORKER_LEFT_FOLD = left_fold
    _WORKER_RIGHT_FOLD = right_fold
    _WORKER_ALIGNER = build_locked_aligner()


def _audit_flat_range(bounds: tuple[int, int]) -> dict[str, Any]:
    """Audit a deterministic contiguous range of one Cartesian product."""
    if _WORKER_ALIGNER is None:
        raise RuntimeError("Alignment worker was not initialized")
    start, end = bounds
    right_count = len(_WORKER_RIGHT)
    aligned_count = 0
    pruned_count = 0
    crossing_rows: list[dict[str, Any]] = []
    for flat_index in range(start, end):
        left_index, right_index = divmod(flat_index, right_count)
        left_digest, left_sequence = _WORKER_LEFT[left_index]
        right_digest, right_sequence = _WORKER_RIGHT[right_index]
        if is_safe_length_prune(len(left_sequence), len(right_sequence)):
            pruned_count += 1
            continue
        metrics = global_alignment_metrics(
            left_sequence, right_sequence, _WORKER_ALIGNER
        )
        aligned_count += 1
        if meets_locked_threshold(metrics, len(left_sequence), len(right_sequence)):
            crossing_rows.append(
                edge_payload(
                    _WORKER_VIEW,
                    _WORKER_LEFT_FOLD,
                    _WORKER_RIGHT_FOLD,
                    left_digest,
                    right_digest,
                    left_sequence,
                    right_sequence,
                    metrics,
                )
            )
    return {
        "candidate_pair_count": end - start,
        "dynamic_programming_alignment_count": aligned_count,
        "provably_pruned_by_length_upper_bound_count": pruned_count,
        "crossing_rows": crossing_rows,
    }


def _iter_partial_results(
    left: tuple[tuple[str, str], ...],
    right: tuple[tuple[str, str], ...],
    view: str,
    left_fold: int,
    right_fold: int,
    workers: int,
    chunk_size: int,
) -> Iterator[dict[str, Any]]:
    """Yield ordered chunk results without retaining the full Cartesian product."""
    total = len(left) * len(right)
    ranges = [
        (start, min(total, start + chunk_size))
        for start in range(0, total, chunk_size)
    ]
    initializer_args = (left, right, view, left_fold, right_fold)
    if workers == 1:
        _initialize_worker(*initializer_args)
        for bounds in ranges:
            yield _audit_flat_range(bounds)
        return
    context = multiprocessing.get_context("fork")
    with context.Pool(
        processes=workers,
        initializer=_initialize_worker,
        initargs=initializer_args,
    ) as pool:
        yield from pool.imap(_audit_flat_range, ranges, chunksize=1)


def _write_jsonl_row(handle: BinaryIO, payload: Mapping[str, Any]) -> None:
    """Write one strict, canonical, single-line JSON record."""
    encoded = (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")
    handle.write(encoded)


def audit_view_to_staged_artifact(
    view: str,
    folds: Sequence[Sequence[tuple[str, str]]],
    target_path: Path,
    workers: int,
    chunk_size: int,
) -> tuple[dict[str, Any], StagedArtifact]:
    """Exhaustively audit one view and stage every digest-only crossing edge."""
    if view not in REQUIRED_VIEWS or len(folds) != len(FOLD_INDICES):
        raise ValueError("Unexpected sequence view or fold inventory")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target_path.name}.", suffix=".staged", dir=target_path.parent
    )
    temporary = Path(temporary_name)
    pair_summaries: list[dict[str, Any]] = []
    total_candidates = 0
    total_aligned = 0
    total_pruned = 0
    total_crossings = 0
    total_exact_overlaps = 0
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            for left_fold, right_fold in FOLD_PAIRS:
                left = tuple(folds[left_fold])
                right = tuple(folds[right_fold])
                candidate_count = len(left) * len(right)
                exact_overlap_count = len(
                    {digest for digest, _ in left}
                    & {digest for digest, _ in right}
                )
                pair_aligned = 0
                pair_pruned = 0
                pair_crossings = 0
                observed_candidates = 0
                for partial in _iter_partial_results(
                    left,
                    right,
                    view,
                    left_fold,
                    right_fold,
                    workers,
                    chunk_size,
                ):
                    observed_candidates += int(partial["candidate_pair_count"])
                    pair_aligned += int(partial["dynamic_programming_alignment_count"])
                    pair_pruned += int(
                        partial["provably_pruned_by_length_upper_bound_count"]
                    )
                    crossing_rows = partial["crossing_rows"]
                    if not isinstance(crossing_rows, list):
                        raise TypeError("Worker crossing rows must be a list")
                    for row in crossing_rows:
                        _write_jsonl_row(handle, row)
                    pair_crossings += len(crossing_rows)
                if observed_candidates != candidate_count:
                    raise RuntimeError("Cartesian-product enumeration is incomplete")
                if pair_aligned + pair_pruned != candidate_count:
                    raise RuntimeError("Alignment and safe-prune accounting is incomplete")
                if exact_overlap_count > pair_crossings:
                    raise RuntimeError("An exact cross-fold overlap was not classified")
                pair_summaries.append(
                    {
                        "left_fold_index": left_fold,
                        "right_fold_index": right_fold,
                        "left_unique_sequence_count": len(left),
                        "right_unique_sequence_count": len(right),
                        "cross_fold_unique_sequence_pair_count": candidate_count,
                        "dynamic_programming_alignment_count": pair_aligned,
                        "provably_pruned_by_length_upper_bound_count": pair_pruned,
                        "exact_sequence_overlap_count": exact_overlap_count,
                        "threshold_crossing_edge_count": pair_crossings,
                    }
                )
                total_candidates += candidate_count
                total_aligned += pair_aligned
                total_pruned += pair_pruned
                total_crossings += pair_crossings
                total_exact_overlaps += exact_overlap_count
                print(
                    f"completed view={view} folds={left_fold}-{right_fold} "
                    f"pairs={candidate_count} aligned={pair_aligned} "
                    f"crossings={pair_crossings}",
                    flush=True,
                )
            handle.flush()
            os.fsync(handle.fileno())
        artifact = StagedArtifact(
            temporary_path=temporary,
            target_path=target_path,
            sha256=sha256_file(temporary),
            size_bytes=temporary.stat().st_size,
            edge_count=total_crossings,
        )
        summary = {
            "sequence_view": view,
            "valid_zero_crossing_edges": total_crossings == 0,
            "fold_pairs": pair_summaries,
            "total_cross_fold_unique_sequence_pair_count": total_candidates,
            "total_dynamic_programming_alignment_count": total_aligned,
            "total_provably_pruned_by_length_upper_bound_count": total_pruned,
            "total_exact_sequence_overlap_count": total_exact_overlaps,
            "total_threshold_crossing_edge_count": total_crossings,
        }
        return summary, artifact
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def publish_staged_no_clobber(artifact: StagedArtifact) -> None:
    """Publish a staged file with atomic hard-link no-overwrite semantics."""
    if sha256_file(artifact.temporary_path) != artifact.sha256:
        raise RuntimeError("Staged artifact changed before publication")
    try:
        os.link(artifact.temporary_path, artifact.target_path)
    except FileExistsError:
        raise
    except OSError as error:
        if error.errno == errno.EEXIST:
            raise FileExistsError(artifact.target_path) from error
        raise
    directory_descriptor = os.open(
        artifact.target_path.parent, os.O_RDONLY | os.O_DIRECTORY
    )
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)


def stage_bytes(target_path: Path, payload: bytes) -> StagedArtifact:
    """Fsync bytes to an adjacent private temporary file."""
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target_path.name}.", suffix=".staged", dir=target_path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        return StagedArtifact(
            temporary_path=temporary,
            target_path=target_path,
            sha256=sha256_bytes(payload),
            size_bytes=len(payload),
            edge_count=0,
        )
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _edge_artifact_summary(artifact: StagedArtifact) -> dict[str, Any]:
    """Return one digest-only crossing artifact reference."""
    return {
        "format": "JSON Lines; one digest-only threshold crossing per line",
        "path": str(artifact.target_path),
        "sha256": artifact.sha256,
        "size_bytes": artifact.size_bytes,
        "edge_count": artifact.edge_count,
        "raw_sequences_or_instance_ids_included": False,
        "create_only_publication": True,
    }


def approval_from_results(results: Mapping[str, Mapping[str, Any]]) -> bool:
    """Approve only complete four-view results with zero crossing edges."""
    if set(results) != set(REQUIRED_VIEWS):
        return False
    return all(
        result.get("valid_zero_crossing_edges") is True
        and result.get("total_threshold_crossing_edge_count") == 0
        for result in results.values()
    )


def runtime_provenance(workers: int, chunk_size: int) -> dict[str, Any]:
    """Describe and hash the pinned runtime component that executes alignments."""
    import Bio
    from Bio.Align import _pairwisealigner

    backend_path = Path(_pairwisealigner.__file__).resolve(strict=True)
    return {
        "python": platform.python_version(),
        "biopython": Bio.__version__,
        "platform": platform.platform(),
        "multiprocessing_start_method": "fork" if workers > 1 else "single_process",
        "worker_count": workers,
        "chunk_size": chunk_size,
        "pairwise_aligner_binary": {
            "path": str(backend_path),
            "sha256": sha256_file(backend_path),
            "size_bytes": backend_path.stat().st_size,
        },
    }


def _verify_references_unchanged(
    references: Mapping[str, Mapping[str, Any]]
) -> None:
    """Rehash every physical input and source immediately before publication."""
    for label, reference in references.items():
        observed = sha256_file(Path(str(reference["path"])))
        if observed != reference["sha256"]:
            raise RuntimeError(
                f"Bound file changed during audit: {label}; "
                f"expected={reference['sha256']}, observed={observed}"
            )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the locked audit command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignment", type=Path, required=True)
    parser.add_argument("--assignment-sha256", required=True)
    parser.add_argument("--assignment-scientific-sha256", required=True)
    parser.add_argument("--authority-split", type=Path, required=True)
    parser.add_argument("--authority-sha256", required=True)
    parser.add_argument("--canonical-manifest", type=Path, required=True)
    parser.add_argument("--canonical-manifest-sha256", required=True)
    parser.add_argument("--expected-auditor-source-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=512)
    args = parser.parse_args(argv)
    for name in (
        "assignment_sha256",
        "assignment_scientific_sha256",
        "authority_sha256",
        "canonical_manifest_sha256",
        "expected_auditor_source_sha256",
    ):
        try:
            setattr(args, name, validate_sha256(getattr(args, name), name))
        except ValueError as error:
            parser.error(str(error))
    if args.workers < 1 or args.chunk_size < 1:
        parser.error("--workers and --chunk-size must be positive")
    return args


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Execute the exact four-view audit and publish one immutable receipt."""
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    edge_targets = {
        view: output_dir / f"{view}.identity80_coverage80.crossing_edges.jsonl"
        for view in REQUIRED_VIEWS
    }
    receipt_target = output_dir / "audit_receipt.json"
    existing_targets = [
        str(path) for path in [*edge_targets.values(), receipt_target] if path.exists()
    ]
    if existing_targets:
        raise FileExistsError(
            "Create-only audit target already exists: " + ", ".join(existing_targets)
        )

    references = {
        "assignment": verify_file(
            args.assignment, args.assignment_sha256, "provisional S2 assignment"
        ),
        "authority_split": verify_file(
            args.authority_split, args.authority_sha256, "authority split"
        ),
        "canonical_sequence_manifest": verify_file(
            args.canonical_manifest,
            args.canonical_manifest_sha256,
            "canonical sequence manifest",
        ),
    }
    sources = {
        "independent_exact_four_view_auditor": verify_file(
            Path(__file__),
            args.expected_auditor_source_sha256,
            "independent exact four-view auditor source",
        )
    }
    train_ids, sealed_ids, authority_summary = load_authority(
        Path(references["authority_split"]["path"])
    )
    assignment, fold_by_source, antibody_map, antigen_map = (
        load_and_validate_assignment(
            Path(references["assignment"]["path"]),
            args.assignment_scientific_sha256,
            references["assignment"],
            references["authority_split"],
            references["canonical_sequence_manifest"],
            train_ids,
            authority_summary,
        )
    )
    records, firewall = load_authorized_sequences(
        Path(references["canonical_sequence_manifest"]["path"]),
        train_ids,
        sealed_ids,
    )
    validate_assignment_against_sequences(
        assignment,
        records,
        fold_by_source,
        antibody_map,
        antigen_map,
    )
    inventories = build_fold_inventories(records, fold_by_source)

    build_locked_aligner()
    runtime = runtime_provenance(args.workers, args.chunk_size)
    staged_edges: list[StagedArtifact] = []
    results: dict[str, dict[str, Any]] = {}
    artifacts: dict[str, dict[str, Any]] = {}
    try:
        for view in REQUIRED_VIEWS:
            summary, artifact = audit_view_to_staged_artifact(
                view,
                inventories[view],
                edge_targets[view],
                args.workers,
                args.chunk_size,
            )
            staged_edges.append(artifact)
            artifacts[view] = _edge_artifact_summary(artifact)
            results[view] = summary

        approved = approval_from_results(results)
        total_candidates = sum(
            int(result["total_cross_fold_unique_sequence_pair_count"])
            for result in results.values()
        )
        total_aligned = sum(
            int(result["total_dynamic_programming_alignment_count"])
            for result in results.values()
        )
        total_pruned = sum(
            int(result["total_provably_pruned_by_length_upper_bound_count"])
            for result in results.values()
        )
        total_crossings = sum(
            int(result["total_threshold_crossing_edge_count"])
            for result in results.values()
        )
        if total_aligned + total_pruned != total_candidates:
            raise RuntimeError("Four-view exhaustive accounting is incomplete")
        if total_crossings != sum(artifact.edge_count for artifact in staged_edges):
            raise RuntimeError("Crossing-edge artifact accounting is inconsistent")

        _verify_references_unchanged({**references, **sources})
        receipt: dict[str, Any] = {
            "format_version": FORMAT_VERSION,
            "stage": "completed_exact_train_only_s2_four_view_nw_audit",
            "intended_use": "train_only_s2_training_and_model_selection_gate",
            "valid_audit_execution": True,
            "approval_status": (
                "approved_zero_cross_fold_edges"
                if approved
                else "not_approved_cross_fold_edges_detected"
            ),
            "approved_for_training": approved,
            "approved_for_model_selection": approved,
            "validity_gate": {
                "gate_name": "exact_four_view_needleman_wunsch_cross_fold_audit",
                "required_result": (
                    "zero_cross_fold_pairs_at_or_above_threshold_in_every_view"
                ),
                "all_required_views_completed": set(results) == set(REQUIRED_VIEWS),
                "zero_crossing_edges_in_every_view": approved,
                "gate_passed": approved,
            },
            "assignment": {
                **references["assignment"],
                "scientific_payload_sha256": args.assignment_scientific_sha256,
                "provisional_assignment_was_unapproved": True,
                "independently_reconstructed_from_authority_train_sequences": True,
            },
            "inputs": {
                "authority_split": references["authority_split"],
                "canonical_sequence_manifest": references[
                    "canonical_sequence_manifest"
                ],
            },
            "source_files": sources,
            "policy": {
                "algorithm": "exact global dynamic programming (Needleman-Wunsch family)",
                "backend": "Biopython PairwiseAligner 1.83",
                "substitution_matrix": "BLOSUM62",
                "substitution_matrix_payload_sha256": (
                    EXPECTED_BLOSUM62_PAYLOAD_SHA256
                ),
                "gap_model": "affine with identically penalized terminal gaps",
                "gap_open_penalty": GAP_OPEN_PENALTY,
                "gap_extend_penalty": GAP_EXTEND_PENALTY,
                "optimal_traceback_policy": (
                    "first deterministic traceback returned by pinned Biopython 1.83"
                ),
                "aligned_region_identity_definition": (
                    "identical residue-residue columns divided by all residue-residue "
                    "columns; gap columns excluded"
                ),
                "reciprocal_coverage_definition": (
                    "paired residue-residue columns divided by each full raw sequence length"
                ),
                "identity_threshold": IDENTITY_NUMERATOR / IDENTITY_DENOMINATOR,
                "reciprocal_coverage_threshold": (
                    COVERAGE_NUMERATOR / COVERAGE_DENOMINATOR
                ),
                "threshold_comparison": (
                    "exact integer cross-products; equality is included"
                ),
                "sequence_views": list(REQUIRED_VIEWS),
                "missing_vl_policy": (
                    "exclude genuinely empty VL from the VL inventory; use VH normally"
                ),
                "enumeration": (
                    "all cross-fold unique-sequence pairs in each required view"
                ),
                "only_pair_pruning_rule": (
                    "prune only when min(raw lengths)/max(raw lengths) is strictly "
                    "below 0.8, which is a mathematical upper bound on both reciprocal "
                    "coverages"
                ),
            },
            "authority": authority_summary,
            "data_firewall": firewall,
            "inventory": {
                view: summarize_inventory(inventories[view])
                for view in REQUIRED_VIEWS
            },
            "results": results,
            "crossing_edge_artifacts": artifacts,
            "exhaustive_accounting": {
                "total_cross_fold_unique_sequence_pair_count": total_candidates,
                "total_dynamic_programming_alignment_count": total_aligned,
                "total_provably_pruned_by_length_upper_bound_count": total_pruned,
                "total_threshold_crossing_edge_count": total_crossings,
                "dynamic_alignments_plus_safe_prunes_equal_all_pairs": True,
            },
            "runtime": runtime,
            "publication": {
                "receipt_path": str(receipt_target),
                "edge_artifacts_and_receipt_are_create_only": True,
                "input_and_source_files_rehashed_immediately_before_publication": True,
                "raw_sequences_or_instance_ids_published_in_crossing_edges": False,
            },
            "limitations": [
                "This audit authorizes only authority-train S2 training and model selection; it is not model evaluation.",
                "Authority validation and frozen-test sequence fields, labels, predictions, checkpoints, and scores were not read.",
                "Passing sequence-homology isolation does not establish structural, functional, temporal, or prospective independence.",
                "The frozen protein-language-model pretraining corpus remains outside this downstream audit.",
            ],
        }
        receipt["scientific_payload_sha256"] = canonical_payload_sha256(receipt)
        staged_receipt = stage_bytes(receipt_target, canonical_json_bytes(receipt))
        try:
            for artifact in staged_edges:
                publish_staged_no_clobber(artifact)
            publish_staged_no_clobber(staged_receipt)
        finally:
            staged_receipt.temporary_path.unlink(missing_ok=True)
        for artifact in staged_edges:
            artifact.temporary_path.unlink(missing_ok=True)
        print(
            json.dumps(
                {
                    "receipt_path": str(receipt_target),
                    "receipt_sha256": sha256_file(receipt_target),
                    "scientific_payload_sha256": receipt[
                        "scientific_payload_sha256"
                    ],
                    "approved_for_training": approved,
                    "approved_for_model_selection": approved,
                    "total_threshold_crossing_edge_count": total_crossings,
                },
                indent=2,
                sort_keys=True,
                allow_nan=False,
            ),
            flush=True,
        )
        return receipt
    except BaseException:
        for artifact in staged_edges:
            artifact.temporary_path.unlink(missing_ok=True)
        raise


def main(argv: Sequence[str] | None = None) -> None:
    """Run the command-line entrypoint without printing any raw sequence."""
    run(parse_args(argv))


if __name__ == "__main__":
    main()
