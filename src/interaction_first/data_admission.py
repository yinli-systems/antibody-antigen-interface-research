"""Fail-closed metadata admission for prospective PAIRFormer datasets.

This module is deliberately data-agnostic. It validates content-addressed
metadata and independent audit receipts without downloading, opening, or
deserializing a dataset. Cryptographic binding proves that the supplied
metadata has not changed; it does not prove that an auditor inspected the
underlying bytes correctly. A later gate must verify every referenced manifest
against the actual artifacts before any model can consume them.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import re
from typing import Any, Mapping
import unicodedata
from urllib.parse import unquote, urlparse


ADMISSION_RECEIPT_FORMAT = "pairformer_dataset_admission_receipt_v1"
SHA_MANIFEST_RECEIPT_FORMAT = "pairformer_sha256_manifest_receipt_v1"
LINEAGE_RECEIPT_FORMAT = "pairformer_row_lineage_receipt_v1"
DUPLICATE_AUDIT_RECEIPT_FORMAT = "pairformer_cross_source_duplicate_audit_v1"
SOURCE_REGISTRY_ENTRY_FORMAT = "pairformer_source_registry_entry_v1"
CANONICAL_JSON_POLICY = (
    "utf8_json_nfc_safeint_sorted_keys_compact_ordered_arrays_newline_v2"
)
MANIFEST_ENTRY_ORDER_POLICY = "lexicographic_stable_row_id_v1"
LICENSE_CONTRACT_DOMAIN = "pairformer_license_contract_v1"
EXACT_DUPLICATE_METHOD = "sha256_canonical_sequence_v1"
NEAR_DUPLICATE_METHOD = "global_identity_full_coverage_v1"
NEAR_DUPLICATE_THRESHOLD = 0.8

SEMANTIC_TYPES = frozenset(
    {
        "repertoire_sequence",
        "measured_interaction",
        "predicted_or_synthetic",
    }
)
RECORD_UNITS = frozenset(
    {"sequence", "interaction_pair", "structure", "residue_pair"}
)
MOLECULAR_AXES = ("antibody", "antigen")
REVISION_TYPES = frozenset(
    {
        "version_control_commit",
        "versioned_release",
        "doi_version",
        "immutable_snapshot",
    }
)
ROUTING_TRACKS = frozenset(
    {
        "general_research",
        "noncommercial_isolated",
        "restricted_no_redistribution",
    }
)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_SAFE_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:+-]{0,255}")
_UTC_TIMESTAMP_PATTERN = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T"
    r"[0-9]{2}:[0-9]{2}:[0-9]{2}Z"
)
_MUTABLE_REVISION_NAMES = frozenset(
    {"head", "latest", "main", "master", "current", "rolling", "trunk", "unversioned"}
)
_GLOB_CHARACTERS = frozenset("*?[]{}")
_CAS_PREFIX = "cas://sha256/"
_MAX_SAFE_JSON_INTEGER = 2**53 - 1
_DOI_REVISION_PATTERN = re.compile(
    r"10\.[0-9]{4,9}/[a-z0-9][a-z0-9._;()/:+-]{0,240}"
)


class DataAdmissionError(ValueError):
    """Raised when prospective data cannot pass the locked admission contract."""


@dataclass(frozen=True)
class AdmissionCounts:
    """Audited record and token counts across the complete admission pipeline."""

    raw_records: int
    normalized_records: int
    qc_passed_records: int
    deduplicated_records: int
    admitted_records: int
    experimental_label_records: int
    predicted_structure_records: int
    synthetic_decoy_records: int
    generated_or_predicted_records: int
    residue_tokens: int
    antibody_entities: int
    antigen_entities: int


@dataclass(frozen=True)
class DatasetAdmission:
    """Immutable result of a successful metadata-only admission decision."""

    dataset_id: str
    source_id: str
    source_name: str
    source_revision: str
    acquired_at_utc: str
    semantic_type: str
    record_unit: str
    molecular_axes: tuple[str, ...]
    routing_track: str
    contamination_policy: str
    noncommercial_contaminated: bool
    counts: AdmissionCounts
    raw_manifest_sha256: str
    normalized_manifest_sha256: str
    qc_passed_manifest_sha256: str
    deduplicated_manifest_sha256: str
    admitted_manifest_sha256: str
    lineage_manifest_sha256: str
    duplicate_audit_receipt_sha256: tuple[tuple[str, str], ...]
    source_registry_entry_sha256: str
    admission_policy_sha256: str
    license_contract_sha256: str
    attribution_required: bool
    admission_created_at_utc: str
    validated_not_after_utc: str
    admission_receipt_sha256: str


@dataclass(frozen=True)
class _ManifestAudit:
    """Validated internal manifest receipt fields used for cross-receipt checks."""

    manifest_sha256: str
    receipt_sha256: str
    producer_code_sha256: str
    created_at_utc: str


@dataclass(frozen=True)
class _LineageAudit:
    """Validated internal lineage receipt fields used for cross-receipt checks."""

    manifest_sha256: str
    receipt_sha256: str
    transform_code_sha256: str
    created_at_utc: str


@dataclass(frozen=True)
class _DuplicateAudit:
    """Validated internal duplicate-audit fields used for axis isolation."""

    receipt_sha256: str
    candidate_entity_manifest_sha256: str
    comparison_registry_sha256: str
    resolution_manifest_sha256: str
    auditor_code_sha256: str
    created_at_utc: str


def _validate_plain_json(value: Any, name: str) -> None:
    """Reject non-portable or ambiguous values before canonical serialization."""
    if value is None or type(value) is bool:
        return
    if type(value) is int:
        if abs(value) > _MAX_SAFE_JSON_INTEGER:
            raise DataAdmissionError(
                f"{name} integer exceeds the interoperable JSON safe range"
            )
        return
    if type(value) is float:
        if not math.isfinite(value) or (
            value == 0.0 and math.copysign(1.0, value) < 0.0
        ):
            raise DataAdmissionError(
                f"{name} must not contain NaN, infinity, or negative zero"
            )
        return
    if type(value) is str:
        if unicodedata.normalize("NFC", value) != value:
            raise DataAdmissionError(f"{name} strings must use Unicode NFC")
        if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
            raise DataAdmissionError(f"{name} cannot contain lone surrogates")
        return
    if type(value) is list:
        for index, item in enumerate(value):
            _validate_plain_json(item, f"{name}[{index}]")
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise DataAdmissionError(f"{name} object keys must be strings")
            _validate_plain_json(key, f"{name}.<key>")
            _validate_plain_json(item, f"{name}.{key}")
        return
    raise DataAdmissionError(
        f"{name} must contain only plain interoperable JSON values"
    )


def _canonical_json_value(value: Any, name: str) -> Any:
    """Return a detached canonical JSON value or reject invalid metadata."""
    _validate_plain_json(value, name)
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        canonical = json.loads(encoded)
    except (TypeError, ValueError, OverflowError) as error:
        raise DataAdmissionError(
            f"{name} must be finite and JSON-compatible"
        ) from error
    return canonical


def _canonical_json_bytes(value: Any) -> bytes:
    """Serialize with sorted object keys while preserving every array order."""
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


def _sha256_bytes(value: bytes) -> str:
    """Return the lowercase SHA-256 digest of bytes."""
    return hashlib.sha256(value).hexdigest()


def _exact_fields(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    """Require an exact mapping schema so unknown fields cannot be ignored."""
    actual = set(value)
    if actual != expected:
        missing = sorted(expected.difference(actual))
        extra = sorted(actual.difference(expected))
        raise DataAdmissionError(
            f"{name} fields do not match the locked schema; "
            f"missing={missing}, extra={extra}"
        )


def _mapping(value: Any, name: str) -> dict[str, Any]:
    """Require a plain JSON object."""
    if type(value) is not dict:
        raise DataAdmissionError(f"{name} must be a JSON object")
    return value


def _nonempty_string(value: Any, name: str) -> str:
    """Validate one nonempty, trimmed string."""
    if not isinstance(value, str) or not value or value.strip() != value:
        raise DataAdmissionError(f"{name} must be a nonempty trimmed string")
    return value


def _identifier(value: Any, name: str) -> str:
    """Validate a safe, path-free identifier."""
    result = _nonempty_string(value, name)
    if not _SAFE_IDENTIFIER_PATTERN.fullmatch(result):
        raise DataAdmissionError(
            f"{name} must be a path-free identifier containing only ASCII "
            "letters, digits, dots, underscores, colons, pluses, or dashes"
        )
    return result


def _sha256(value: Any, name: str) -> str:
    """Validate a lowercase SHA-256 digest."""
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise DataAdmissionError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _boolean(value: Any, name: str) -> bool:
    """Require an actual JSON boolean."""
    if not isinstance(value, bool):
        raise DataAdmissionError(f"{name} must be a boolean")
    return value


def _nonnegative_integer(value: Any, name: str) -> int:
    """Return a non-negative, non-boolean integer."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DataAdmissionError(f"{name} must be a non-negative integer")
    return int(value)


def _positive_integer(value: Any, name: str) -> int:
    """Return a positive, non-boolean integer."""
    result = _nonnegative_integer(value, name)
    if result == 0:
        raise DataAdmissionError(f"{name} must be a positive integer")
    return result


def _closed_unit_interval(value: Any, name: str) -> float:
    """Return a finite scalar in the interval (0, 1]."""
    if type(value) not in (int, float) or type(value) is bool:
        raise DataAdmissionError(f"{name} must be a finite scalar in (0, 1]")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0 or result > 1.0:
        raise DataAdmissionError(f"{name} must be a finite scalar in (0, 1]")
    return result


def _utc_timestamp(value: Any, name: str) -> str:
    """Validate an explicit whole-second UTC timestamp."""
    result = _nonempty_string(value, name)
    if not _UTC_TIMESTAMP_PATTERN.fullmatch(result):
        raise DataAdmissionError(
            f"{name} must use the exact UTC form YYYY-MM-DDTHH:MM:SSZ"
        )
    try:
        parsed = datetime.fromisoformat(result.replace("Z", "+00:00"))
    except ValueError as error:
        raise DataAdmissionError(f"{name} is not a valid UTC timestamp") from error
    if parsed.tzinfo != timezone.utc:
        raise DataAdmissionError(f"{name} must be in UTC")
    return result


def _https_url(value: Any, name: str) -> str:
    """Validate a canonical, credential-free HTTPS reference URL."""
    result = _nonempty_string(value, name)
    if (
        any(character in result for character in _GLOB_CHARACTERS)
        or any(ord(character) < 0x20 or ord(character) > 0x7E for character in result)
        or "\\" in result
    ):
        raise DataAdmissionError(
            f"{name} must be printable ASCII without globs or backslashes"
        )
    try:
        parsed = urlparse(result)
        parsed_port = parsed.port
        parsed_hostname = parsed.hostname
    except ValueError as error:
        raise DataAdmissionError(f"{name} is not a valid HTTPS URL") from error
    decoded_path = unquote(parsed.path)
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or parsed.query
        or parsed.params
        or parsed_port is not None
        or parsed_hostname is None
        or parsed_hostname != parsed_hostname.lower()
        or parsed.netloc != parsed_hostname
        or not re.fullmatch(
            r"(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+"
            r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?",
            parsed_hostname,
        )
        or "\\" in decoded_path
        or "%" in parsed.path
        or any(
            ord(character) < 0x20 or ord(character) == 0x7F
            for character in decoded_path
        )
        or any(segment in {".", ".."} for segment in decoded_path.split("/"))
        or "//" in parsed.path
    ):
        raise DataAdmissionError(
            f"{name} must be a canonical HTTPS URL without credentials, "
            "queries, fragments, ports, or path traversal"
        )
    return result


def _content_addressed_uri(value: Any, digest: str, name: str) -> str:
    """Require the exact immutable URI for a supplied digest."""
    result = _nonempty_string(value, name)
    expected = f"{_CAS_PREFIX}{digest}"
    if result != expected:
        raise DataAdmissionError(
            f"{name} must be the exact content-addressed URI {expected!r}; "
            "mutable paths and globs are forbidden"
        )
    return result


def make_content_addressed_receipt(
    receipt_format: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a canonical in-memory receipt without reading or writing files.

    The digest covers the format, the immutable declaration, and the complete
    canonical payload. Object keys are sorted recursively, JSON arrays retain
    their supplied order, separators are compact, and one trailing newline is
    hashed. The URI therefore changes after any payload or array-order mutation.
    """
    format_value = _identifier(receipt_format, "receipt_format")
    payload_value = _canonical_json_value(
        _mapping(payload, "payload"),
        "payload",
    )
    body = {
        "receipt_format": format_value,
        "immutable": True,
        "payload": payload_value,
    }
    digest = _sha256_bytes(_canonical_json_bytes(body))
    return {
        **body,
        "receipt_sha256": digest,
        "receipt_uri": f"{_CAS_PREFIX}{digest}",
    }


def _validated_receipt(
    value: Any,
    *,
    expected_format: str,
    name: str,
) -> tuple[dict[str, Any], str]:
    """Verify one immutable receipt and return its canonical payload and digest."""
    receipt = _mapping(value, name)
    _exact_fields(
        receipt,
        {
            "receipt_format",
            "immutable",
            "payload",
            "receipt_sha256",
            "receipt_uri",
        },
        name,
    )
    if receipt["receipt_format"] != expected_format:
        raise DataAdmissionError(
            f"{name}.receipt_format must equal {expected_format!r}"
        )
    if receipt["immutable"] is not True:
        raise DataAdmissionError(f"{name} must declare immutable=true")
    payload = _canonical_json_value(
        _mapping(receipt["payload"], f"{name}.payload"),
        f"{name}.payload",
    )
    body = {
        "receipt_format": expected_format,
        "immutable": True,
        "payload": payload,
    }
    calculated = _sha256_bytes(_canonical_json_bytes(body))
    declared = _sha256(receipt["receipt_sha256"], f"{name}.receipt_sha256")
    if declared != calculated:
        raise DataAdmissionError(
            f"{name} receipt SHA-256 does not match its canonical payload"
        )
    _content_addressed_uri(
        receipt["receipt_uri"],
        declared,
        f"{name}.receipt_uri",
    )
    return payload, declared


def _validate_revision(value: Any, revision_type: str, name: str) -> str:
    """Validate an immutable revision using its declared namespace."""
    if revision_type == "version_control_commit":
        revision = _nonempty_string(value, name)
        if not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", revision):
            raise DataAdmissionError(
                "A version_control_commit revision must be a full lowercase "
                "40- or 64-character commit digest"
            )
        return revision
    if revision_type == "immutable_snapshot":
        return _sha256(value, name)
    if revision_type == "doi_version":
        revision = _nonempty_string(value, name)
        if not _DOI_REVISION_PATTERN.fullmatch(revision):
            raise DataAdmissionError(
                f"{name} must be a canonical lowercase versioned DOI"
            )
        return revision

    revision = _identifier(value, name)
    folded = revision.casefold()
    if (
        folded in _MUTABLE_REVISION_NAMES
        or re.fullmatch(
            r"(?:head|latest|main|master|current|rolling|trunk|unversioned)"
            r"[._:+-]?[0-9]*",
            folded,
        )
        or not any(character.isdigit() for character in revision)
    ):
        raise DataAdmissionError(
            f"{name} must be a pinned versioned release with a numeric component"
        )
    return revision


def _validate_source(
    value: Any,
    *,
    dataset_id: str,
) -> tuple[str, str, str, str, str, str, tuple[str, ...]]:
    """Validate pinned source identity and scientific semantics."""
    source = _mapping(value, "admission.payload.source")
    _exact_fields(
        source,
        {
            "source_id",
            "source_name",
            "landing_page_url",
            "revision",
            "revision_type",
            "acquired_at_utc",
            "semantic_type",
            "record_unit",
            "molecular_axes",
        },
        "admission.payload.source",
    )
    source_id = _identifier(
        source["source_id"],
        "admission.payload.source.source_id",
    )
    source_name = _nonempty_string(
        source["source_name"],
        "admission.payload.source.source_name",
    )
    _https_url(
        source["landing_page_url"],
        "admission.payload.source.landing_page_url",
    )
    revision_type = source["revision_type"]
    if revision_type not in REVISION_TYPES:
        raise DataAdmissionError(
            "admission.payload.source.revision_type is unsupported"
        )
    revision = _validate_revision(
        source["revision"],
        revision_type,
        "admission.payload.source.revision",
    )
    acquired_at = _utc_timestamp(
        source["acquired_at_utc"],
        "admission.payload.source.acquired_at_utc",
    )
    semantic_type = source["semantic_type"]
    if semantic_type not in SEMANTIC_TYPES:
        raise DataAdmissionError(
            "admission.payload.source.semantic_type is unsupported"
        )
    record_unit = source["record_unit"]
    if record_unit not in RECORD_UNITS:
        raise DataAdmissionError(
            "admission.payload.source.record_unit is unsupported"
        )
    raw_axes = source["molecular_axes"]
    if not isinstance(raw_axes, list) or not raw_axes:
        raise DataAdmissionError(
            "admission.payload.source.molecular_axes must be a nonempty array"
        )
    axes = tuple(raw_axes)
    if (
        len(set(axes)) != len(axes)
        or any(axis not in MOLECULAR_AXES for axis in axes)
        or axes != tuple(axis for axis in MOLECULAR_AXES if axis in axes)
    ):
        raise DataAdmissionError(
            "admission.payload.source.molecular_axes must be a unique canonical "
            "subsequence of ['antibody', 'antigen']"
        )

    if semantic_type == "repertoire_sequence" and record_unit != "sequence":
        raise DataAdmissionError(
            "repertoire_sequence data must declare record_unit='sequence'"
        )
    if semantic_type == "measured_interaction" and (
        record_unit != "interaction_pair" or axes != MOLECULAR_AXES
    ):
        raise DataAdmissionError(
            "measured_interaction data must declare antibody-antigen "
            "interaction_pair records"
        )
    return (
        source_id,
        source_name,
        revision,
        acquired_at,
        semantic_type,
        record_unit,
        axes,
    )


def _validate_counts(value: Any) -> AdmissionCounts:
    """Validate pipeline count monotonicity and return immutable counts."""
    counts = _mapping(value, "admission.payload.counts")
    field_names = {
        "raw_records",
        "normalized_records",
        "qc_passed_records",
        "deduplicated_records",
        "admitted_records",
        "experimental_label_records",
        "predicted_structure_records",
        "synthetic_decoy_records",
        "generated_or_predicted_records",
        "residue_tokens",
        "antibody_entities",
        "antigen_entities",
    }
    _exact_fields(counts, field_names, "admission.payload.counts")
    parsed = {
        name: _nonnegative_integer(
            counts[name],
            f"admission.payload.counts.{name}",
        )
        for name in field_names
    }
    for name in ("raw_records", "admitted_records", "residue_tokens"):
        if parsed[name] == 0:
            raise DataAdmissionError(
                f"admission.payload.counts.{name} must be positive"
            )
    pipeline = (
        parsed["raw_records"],
        parsed["normalized_records"],
        parsed["qc_passed_records"],
        parsed["deduplicated_records"],
        parsed["admitted_records"],
    )
    if any(right > left for left, right in zip(pipeline, pipeline[1:])):
        raise DataAdmissionError(
            "Record counts must satisfy raw >= normalized >= QC >= "
            "deduplicated >= admitted"
        )
    admitted = parsed["admitted_records"]
    for name in (
        "experimental_label_records",
        "predicted_structure_records",
        "synthetic_decoy_records",
        "generated_or_predicted_records",
    ):
        if parsed[name] > admitted:
            raise DataAdmissionError(
                f"admission.payload.counts.{name} cannot exceed admitted_records"
            )
    if (
        parsed["experimental_label_records"]
        + parsed["synthetic_decoy_records"]
        > admitted
    ):
        raise DataAdmissionError(
            "Experimental labels and synthetic decoys must identify disjoint "
            "admitted records"
        )
    if parsed["residue_tokens"] < admitted:
        raise DataAdmissionError(
            "residue_tokens cannot be smaller than admitted_records"
        )
    for axis in MOLECULAR_AXES:
        entity_count = parsed[f"{axis}_entities"]
        if entity_count > admitted:
            raise DataAdmissionError(
                f"{axis}_entities cannot exceed admitted_records"
            )
    return AdmissionCounts(**parsed)


def _validate_semantic_counts(
    semantic_type: str,
    counts: AdmissionCounts,
    *,
    record_unit: str,
    axes: tuple[str, ...],
) -> None:
    """Reject semantic labels that contradict audited generation categories."""
    for axis in MOLECULAR_AXES:
        entity_count = getattr(counts, f"{axis}_entities")
        if (axis in axes) != (entity_count > 0):
            raise DataAdmissionError(
                f"{axis}_entities must be positive exactly when the axis is present"
            )
    if semantic_type == "repertoire_sequence":
        if counts.generated_or_predicted_records != 0:
            raise DataAdmissionError(
                "Repertoire sequence records cannot be called generated or predicted"
            )
        if counts.experimental_label_records != 0:
            raise DataAdmissionError(
                "Repertoire sequences cannot be called experimental "
                "interaction labels"
            )
        if counts.synthetic_decoy_records != 0:
            raise DataAdmissionError(
                "A repertoire_sequence source cannot contain synthetic decoys"
            )
        if axes != ("antibody",):
            raise DataAdmissionError(
                "repertoire_sequence data must contain only the antibody axis"
            )
        if counts.antibody_entities != counts.admitted_records:
            raise DataAdmissionError(
                "Deduplicated repertoire sequence records must map one-to-one "
                "to admitted antibody entities"
            )
    elif semantic_type == "measured_interaction":
        if counts.generated_or_predicted_records != 0:
            raise DataAdmissionError(
                "Measured interaction records cannot be called generated or predicted"
            )
        if counts.experimental_label_records == 0:
            raise DataAdmissionError(
                "measured_interaction data must contain experimental labels"
            )
        if counts.synthetic_decoy_records != 0:
            raise DataAdmissionError(
                "Synthetic decoys require a separate predicted_or_synthetic "
                "admission receipt"
            )
    else:
        if counts.experimental_label_records != 0:
            raise DataAdmissionError(
                "Generated or predicted data cannot be called experimental"
            )
        if counts.generated_or_predicted_records != counts.admitted_records:
            raise DataAdmissionError(
                "Every predicted_or_synthetic record must be counted as generated "
                "or predicted"
            )
        if (
            counts.predicted_structure_records == 0
            and counts.synthetic_decoy_records == 0
        ):
            raise DataAdmissionError(
                "predicted_or_synthetic data must identify predicted structures "
                "or synthetic decoys"
            )
        if counts.synthetic_decoy_records and (
            record_unit != "interaction_pair" or axes != MOLECULAR_AXES
        ):
            raise DataAdmissionError(
                "Synthetic decoys must be antibody-antigen interaction_pair records"
            )


def _validate_label_semantics(
    value: Any,
    *,
    semantic_type: str,
    counts: AdmissionCounts,
) -> None:
    """Prevent unknown or generated pairs from masquerading as measured labels."""
    semantics = _mapping(value, "admission.payload.label_semantics")
    _exact_fields(
        semantics,
        {
            "experimental_positive_records",
            "experimental_nonbinder_records",
            "unknown_pair_records",
            "generated_negative_records",
            "unknown_pairs_used_as_negative",
        },
        "admission.payload.label_semantics",
    )
    experimental_positive = _nonnegative_integer(
        semantics["experimental_positive_records"],
        "admission.payload.label_semantics.experimental_positive_records",
    )
    experimental_nonbinder = _nonnegative_integer(
        semantics["experimental_nonbinder_records"],
        "admission.payload.label_semantics.experimental_nonbinder_records",
    )
    unknown_pairs = _nonnegative_integer(
        semantics["unknown_pair_records"],
        "admission.payload.label_semantics.unknown_pair_records",
    )
    generated_negative = _nonnegative_integer(
        semantics["generated_negative_records"],
        "admission.payload.label_semantics.generated_negative_records",
    )
    used_as_negative = _boolean(
        semantics["unknown_pairs_used_as_negative"],
        "admission.payload.label_semantics.unknown_pairs_used_as_negative",
    )
    if used_as_negative:
        raise DataAdmissionError(
            "Unknown antibody-antigen pairs are unlabeled, not confirmed nonbinders"
        )
    if (
        experimental_positive + experimental_nonbinder
        != counts.experimental_label_records
    ):
        raise DataAdmissionError(
            "Experimental positive and confirmed-nonbinder counts must exactly "
            "partition experimental_label_records"
        )
    if generated_negative != counts.synthetic_decoy_records:
        raise DataAdmissionError(
            "Generated negative records must exactly match synthetic_decoy_records"
        )
    if (
        experimental_positive
        + experimental_nonbinder
        + unknown_pairs
        + generated_negative
        > counts.admitted_records
    ):
        raise DataAdmissionError(
            "Disjoint label-semantic counts cannot exceed admitted_records"
        )
    if semantic_type == "measured_interaction" and (
        unknown_pairs != 0
        or generated_negative != 0
        or counts.experimental_label_records != counts.admitted_records
    ):
        raise DataAdmissionError(
            "measured_interaction records must all be experimentally measured; "
            "unknown or generated pairs require separate admissions"
        )
    if semantic_type == "repertoire_sequence" and unknown_pairs != 0:
        raise DataAdmissionError(
            "A repertoire sequence source cannot declare antibody-antigen "
            "unknown-pair records"
        )
    if semantic_type == "repertoire_sequence" and (
        experimental_positive != 0
        or experimental_nonbinder != 0
        or generated_negative != 0
    ):
        raise DataAdmissionError(
            "A repertoire sequence source cannot declare interaction labels"
        )
    if semantic_type == "predicted_or_synthetic" and (
        experimental_positive != 0 or experimental_nonbinder != 0
    ):
        raise DataAdmissionError(
            "Generated or predicted artifacts cannot carry experimental labels"
        )


def _validate_sha_manifest_receipt(
    value: Any,
    *,
    name: str,
    dataset_id: str,
    stage: str,
    expected_count: int,
) -> _ManifestAudit:
    """Validate a content manifest receipt and return manifest and receipt hashes."""
    payload, receipt_sha256 = _validated_receipt(
        value,
        expected_format=SHA_MANIFEST_RECEIPT_FORMAT,
        name=name,
    )
    _exact_fields(
        payload,
        {
            "dataset_id",
            "stage",
            "manifest_sha256",
            "manifest_uri",
            "entry_count",
            "total_size_bytes",
            "producer_code_sha256",
            "canonicalization_policy",
            "entry_order_policy",
            "created_at_utc",
        },
        f"{name}.payload",
    )
    if payload["dataset_id"] != dataset_id:
        raise DataAdmissionError(f"{name} is bound to the wrong dataset_id")
    if payload["stage"] != stage:
        raise DataAdmissionError(f"{name}.payload.stage must equal {stage!r}")
    manifest_sha256 = _sha256(
        payload["manifest_sha256"],
        f"{name}.payload.manifest_sha256",
    )
    _content_addressed_uri(
        payload["manifest_uri"],
        manifest_sha256,
        f"{name}.payload.manifest_uri",
    )
    entry_count = _nonnegative_integer(
        payload["entry_count"],
        f"{name}.payload.entry_count",
    )
    if entry_count != expected_count:
        raise DataAdmissionError(
            f"{name}.payload.entry_count does not match the audited {stage} count"
        )
    total_size_bytes = _nonnegative_integer(
        payload["total_size_bytes"],
        f"{name}.payload.total_size_bytes",
    )
    if entry_count > 0 and total_size_bytes == 0:
        raise DataAdmissionError(
            f"{name}.payload.total_size_bytes must be positive for a nonempty "
            "manifest"
        )
    producer_code_sha256 = _sha256(
        payload["producer_code_sha256"],
        f"{name}.payload.producer_code_sha256",
    )
    if payload["canonicalization_policy"] != CANONICAL_JSON_POLICY:
        raise DataAdmissionError(
            f"{name}.payload.canonicalization_policy must equal "
            f"{CANONICAL_JSON_POLICY!r}"
        )
    if payload["entry_order_policy"] != MANIFEST_ENTRY_ORDER_POLICY:
        raise DataAdmissionError(
            f"{name}.payload.entry_order_policy must equal "
            f"{MANIFEST_ENTRY_ORDER_POLICY!r}"
        )
    created_at_utc = _utc_timestamp(
        payload["created_at_utc"],
        f"{name}.payload.created_at_utc",
    )
    return _ManifestAudit(
        manifest_sha256=manifest_sha256,
        receipt_sha256=receipt_sha256,
        producer_code_sha256=producer_code_sha256,
        created_at_utc=created_at_utc,
    )


def _validate_lineage_receipt(
    value: Any,
    *,
    dataset_id: str,
    stage_manifest_sha256: Mapping[str, str],
    stage_counts: Mapping[str, int],
) -> _LineageAudit:
    """Validate complete source-to-normalized row lineage."""
    name = "admission.payload.lineage_receipt"
    payload, receipt_sha256 = _validated_receipt(
        value,
        expected_format=LINEAGE_RECEIPT_FORMAT,
        name=name,
    )
    _exact_fields(
        payload,
        {
            "dataset_id",
            "raw_manifest_sha256",
            "normalized_manifest_sha256",
            "qc_passed_manifest_sha256",
            "deduplicated_manifest_sha256",
            "admitted_manifest_sha256",
            "lineage_manifest_sha256",
            "lineage_manifest_uri",
            "normalized_row_count",
            "qc_passed_row_count",
            "deduplicated_row_count",
            "admitted_row_count",
            "normalized_rows_with_lineage_count",
            "qc_passed_rows_with_lineage_count",
            "deduplicated_rows_with_lineage_count",
            "admitted_rows_with_lineage_count",
            "lineage_edge_count",
            "missing_normalized_row_count",
            "missing_qc_passed_row_count",
            "missing_deduplicated_row_count",
            "missing_admitted_row_count",
            "orphan_source_reference_count",
            "duplicate_raw_row_id_count",
            "duplicate_normalized_row_id_count",
            "duplicate_qc_passed_row_id_count",
            "duplicate_deduplicated_row_id_count",
            "duplicate_admitted_row_id_count",
            "transform_code_sha256",
            "created_at_utc",
        },
        f"{name}.payload",
    )
    if payload["dataset_id"] != dataset_id:
        raise DataAdmissionError(f"{name} is bound to the wrong dataset_id")
    for stage in (
        "raw",
        "normalized",
        "qc_passed",
        "deduplicated",
        "admitted",
    ):
        field_name = f"{stage}_manifest_sha256"
        if payload[field_name] != stage_manifest_sha256[stage]:
            raise DataAdmissionError(
                f"{name} is bound to the wrong {stage} manifest"
            )
    lineage_manifest_sha256 = _sha256(
        payload["lineage_manifest_sha256"],
        f"{name}.payload.lineage_manifest_sha256",
    )
    _content_addressed_uri(
        payload["lineage_manifest_uri"],
        lineage_manifest_sha256,
        f"{name}.payload.lineage_manifest_uri",
    )
    child_stages = ("normalized", "qc_passed", "deduplicated", "admitted")
    declared_counts: dict[str, int] = {}
    covered_counts: dict[str, int] = {}
    for stage in child_stages:
        declared_counts[stage] = _nonnegative_integer(
            payload[f"{stage}_row_count"],
            f"{name}.payload.{stage}_row_count",
        )
        covered_counts[stage] = _nonnegative_integer(
            payload[f"{stage}_rows_with_lineage_count"],
            f"{name}.payload.{stage}_rows_with_lineage_count",
        )
        if declared_counts[stage] != stage_counts[stage]:
            raise DataAdmissionError(
                f"{name} {stage} row count does not match its manifest"
            )
        if covered_counts[stage] != declared_counts[stage]:
            raise DataAdmissionError(
                f"Every {stage} row must have at least one parent-row lineage edge"
            )
    lineage_edges = _nonnegative_integer(
        payload["lineage_edge_count"],
        f"{name}.payload.lineage_edge_count",
    )
    if lineage_edges < sum(covered_counts.values()):
        raise DataAdmissionError(
            "The lineage edge ledger cannot cover all child rows"
        )
    zero_fields = (
        "missing_normalized_row_count",
        "missing_qc_passed_row_count",
        "missing_deduplicated_row_count",
        "missing_admitted_row_count",
        "orphan_source_reference_count",
        "duplicate_raw_row_id_count",
        "duplicate_normalized_row_id_count",
        "duplicate_qc_passed_row_id_count",
        "duplicate_deduplicated_row_id_count",
        "duplicate_admitted_row_id_count",
    )
    for field_name in zero_fields:
        if (
            _nonnegative_integer(
                payload[field_name],
                f"{name}.payload.{field_name}",
            )
            != 0
        ):
            raise DataAdmissionError(
                f"{name}.payload.{field_name} must equal zero"
            )
    transform_code_sha256 = _sha256(
        payload["transform_code_sha256"],
        f"{name}.payload.transform_code_sha256",
    )
    created_at_utc = _utc_timestamp(
        payload["created_at_utc"],
        f"{name}.payload.created_at_utc",
    )
    return _LineageAudit(
        manifest_sha256=lineage_manifest_sha256,
        receipt_sha256=receipt_sha256,
        transform_code_sha256=transform_code_sha256,
        created_at_utc=created_at_utc,
    )


def _validate_license(value: Any) -> dict[str, Any]:
    """Validate license text binding and use/redistribution permissions."""
    license_value = _mapping(value, "admission.payload.license")
    _exact_fields(
        license_value,
        {
            "license_name",
            "spdx_expression",
            "license_url",
            "license_text_sha256",
            "license_text_uri",
            "raw_redistribution_allowed",
            "normalized_redistribution_allowed",
            "commercial_use_allowed",
            "derivatives_allowed",
            "attribution_required",
            "share_alike_required",
        },
        "admission.payload.license",
    )
    _nonempty_string(
        license_value["license_name"],
        "admission.payload.license.license_name",
    )
    spdx_expression = _nonempty_string(
        license_value["spdx_expression"],
        "admission.payload.license.spdx_expression",
    )
    if spdx_expression.casefold() in {
        "none",
        "unknown",
        "unlicensed",
        "tbd",
        "proprietary",
    }:
        raise DataAdmissionError(
            "Unlicensed or unresolved license metadata cannot be admitted"
        )
    if not re.fullmatch(
        r"(?:LicenseRef-[A-Za-z0-9.-]+|"
        r"[A-Za-z0-9.+-]+(?:\s+(?:AND|OR|WITH)\s+[A-Za-z0-9.+-]+)*)",
        spdx_expression,
    ):
        raise DataAdmissionError(
            "admission.payload.license.spdx_expression is not a supported "
            "parseable SPDX expression; custom terms require LicenseRef-*"
        )
    _https_url(
        license_value["license_url"],
        "admission.payload.license.license_url",
    )
    text_sha256 = _sha256(
        license_value["license_text_sha256"],
        "admission.payload.license.license_text_sha256",
    )
    _content_addressed_uri(
        license_value["license_text_uri"],
        text_sha256,
        "admission.payload.license.license_text_uri",
    )
    for field_name in (
        "raw_redistribution_allowed",
        "normalized_redistribution_allowed",
        "commercial_use_allowed",
        "derivatives_allowed",
        "attribution_required",
        "share_alike_required",
    ):
        _boolean(
            license_value[field_name],
            f"admission.payload.license.{field_name}",
        )
    folded_spdx = spdx_expression.casefold()
    if (
        "-nc-" in folded_spdx or folded_spdx.endswith("-nc")
    ) and license_value["commercial_use_allowed"]:
        raise DataAdmissionError(
            "The SPDX expression contains an NC restriction but "
            "commercial_use_allowed=true"
        )
    if (
        "-sa-" in folded_spdx or folded_spdx.endswith("-sa")
    ) and not license_value["share_alike_required"]:
        raise DataAdmissionError(
            "The SPDX expression contains an SA restriction but "
            "share_alike_required=false"
        )
    if not license_value["derivatives_allowed"]:
        raise DataAdmissionError(
            "The source license does not allow the normalized derivative "
            "required by this admission"
        )
    return license_value


def _domain_contract_sha256(domain: str, value: Any) -> str:
    """Hash a canonical value with an explicit semantic domain separator."""
    canonical = _canonical_json_value(value, f"{domain}.value")
    return _sha256_bytes(
        _canonical_json_bytes(
            {
                "contract_domain": domain,
                "value": canonical,
            }
        )
    )


def _source_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return the acquisition-independent source fields bound by the registry."""
    return {
        field: value[field]
        for field in (
            "source_id",
            "source_name",
            "landing_page_url",
            "revision",
            "revision_type",
            "semantic_type",
            "record_unit",
            "molecular_axes",
        )
    }


def make_source_registry_entry(
    *,
    dataset_id: str,
    source: Mapping[str, Any],
    license_value: Mapping[str, Any],
    admission_policy_sha256: str,
    comparison_registry_sha256_by_axis: Mapping[str, Any],
    created_at_utc: str,
) -> dict[str, Any]:
    """Build a source-registry receipt for caller-side review and freezing.

    Building an entry does not approve it. Admission requires the caller to
    separately supply the exact frozen receipt digest.
    """
    dataset = _identifier(dataset_id, "source_registry.dataset_id")
    source_value = _canonical_json_value(
        _mapping(source, "source_registry.source"),
        "source_registry.source",
    )
    _validate_source(source_value, dataset_id=dataset)
    license_contract = _validate_license(
        _canonical_json_value(
            _mapping(license_value, "source_registry.license"),
            "source_registry.license",
        )
    )
    policy_sha256 = _sha256(
        admission_policy_sha256,
        "source_registry.admission_policy_sha256",
    )
    comparison_registries = _mapping(
        comparison_registry_sha256_by_axis,
        "source_registry.comparison_registry_sha256_by_axis",
    )
    _exact_fields(
        comparison_registries,
        set(MOLECULAR_AXES),
        "source_registry.comparison_registry_sha256_by_axis",
    )
    comparison_registry_contract = {
        axis: _sha256(
            comparison_registries[axis],
            f"source_registry.comparison_registry_sha256_by_axis.{axis}",
        )
        for axis in MOLECULAR_AXES
    }
    if len(set(comparison_registry_contract.values())) != len(MOLECULAR_AXES):
        raise DataAdmissionError(
            "Antibody and antigen comparison registries must be domain-isolated"
        )
    created = _utc_timestamp(created_at_utc, "source_registry.created_at_utc")
    payload = {
        "dataset_id": dataset,
        "source_contract": _source_contract(source_value),
        "license_contract_sha256": _domain_contract_sha256(
            LICENSE_CONTRACT_DOMAIN,
            license_contract,
        ),
        "admission_policy_sha256": policy_sha256,
        "comparison_registry_sha256_by_axis": comparison_registry_contract,
        "created_at_utc": created,
    }
    return make_content_addressed_receipt(SOURCE_REGISTRY_ENTRY_FORMAT, payload)


def _validate_source_registry_entry(
    value: Any,
    *,
    expected_receipt_sha256: str,
    expected_admission_policy_sha256: str,
    dataset_id: str,
    source: Mapping[str, Any],
    license_contract_sha256: str,
) -> tuple[str, str, tuple[tuple[str, str], ...]]:
    """Validate a caller-frozen registry entry against admission declarations."""
    name = "source_registry_entry"
    payload, receipt_sha256 = _validated_receipt(
        value,
        expected_format=SOURCE_REGISTRY_ENTRY_FORMAT,
        name=name,
    )
    expected_digest = _sha256(
        expected_receipt_sha256,
        "expected_source_registry_entry_sha256",
    )
    if receipt_sha256 != expected_digest:
        raise DataAdmissionError(
            "The source-registry entry does not match the caller-frozen digest"
        )
    _exact_fields(
        payload,
        {
            "dataset_id",
            "source_contract",
            "license_contract_sha256",
            "admission_policy_sha256",
            "comparison_registry_sha256_by_axis",
            "created_at_utc",
        },
        f"{name}.payload",
    )
    if payload["dataset_id"] != dataset_id:
        raise DataAdmissionError(f"{name} is bound to the wrong dataset_id")
    declared_source = _mapping(
        payload["source_contract"],
        f"{name}.payload.source_contract",
    )
    expected_source = _source_contract(source)
    _exact_fields(
        declared_source,
        set(expected_source),
        f"{name}.payload.source_contract",
    )
    if declared_source != expected_source:
        raise DataAdmissionError(
            "Admission source declarations differ from the caller-frozen "
            "source-registry contract"
        )
    if payload["license_contract_sha256"] != license_contract_sha256:
        raise DataAdmissionError(
            "Admission license declarations differ from the caller-frozen "
            "source-registry contract"
        )
    if payload["admission_policy_sha256"] != expected_admission_policy_sha256:
        raise DataAdmissionError(
            "The source-registry entry is bound to a different admission policy"
        )
    comparison_registries = _mapping(
        payload["comparison_registry_sha256_by_axis"],
        f"{name}.payload.comparison_registry_sha256_by_axis",
    )
    _exact_fields(
        comparison_registries,
        set(MOLECULAR_AXES),
        f"{name}.payload.comparison_registry_sha256_by_axis",
    )
    comparison_registry_contract = tuple(
        (
            axis,
            _sha256(
                comparison_registries[axis],
                f"{name}.payload.comparison_registry_sha256_by_axis.{axis}",
            ),
        )
        for axis in MOLECULAR_AXES
    )
    if len({digest for _, digest in comparison_registry_contract}) != len(
        MOLECULAR_AXES
    ):
        raise DataAdmissionError(
            "Antibody and antigen comparison registries must be domain-isolated"
        )
    created = _utc_timestamp(
        payload["created_at_utc"],
        f"{name}.payload.created_at_utc",
    )
    return receipt_sha256, created, comparison_registry_contract


def _validate_routing(
    value: Any,
    *,
    license_value: Mapping[str, Any],
) -> tuple[str, str, bool]:
    """Validate requested use and deterministic contamination routing."""
    routing = _mapping(value, "admission.payload.routing")
    _exact_fields(
        routing,
        {
            "track",
            "commercial_use_requested",
            "redistribute_raw_requested",
            "redistribute_normalized_requested",
            "propagate_noncommercial_restriction",
            "propagate_share_alike_restriction",
            "propagate_redistribution_restrictions",
            "propagate_attribution_requirement",
        },
        "admission.payload.routing",
    )
    track = routing["track"]
    if track not in ROUTING_TRACKS:
        raise DataAdmissionError("admission.payload.routing.track is unsupported")
    commercial_requested = _boolean(
        routing["commercial_use_requested"],
        "admission.payload.routing.commercial_use_requested",
    )
    raw_requested = _boolean(
        routing["redistribute_raw_requested"],
        "admission.payload.routing.redistribute_raw_requested",
    )
    normalized_requested = _boolean(
        routing["redistribute_normalized_requested"],
        "admission.payload.routing.redistribute_normalized_requested",
    )
    propagate_noncommercial = _boolean(
        routing["propagate_noncommercial_restriction"],
        "admission.payload.routing.propagate_noncommercial_restriction",
    )
    propagate_share_alike = _boolean(
        routing["propagate_share_alike_restriction"],
        "admission.payload.routing.propagate_share_alike_restriction",
    )
    propagate_redistribution = _boolean(
        routing["propagate_redistribution_restrictions"],
        "admission.payload.routing.propagate_redistribution_restrictions",
    )
    propagate_attribution = _boolean(
        routing["propagate_attribution_requirement"],
        "admission.payload.routing.propagate_attribution_requirement",
    )
    if commercial_requested and not license_value["commercial_use_allowed"]:
        raise DataAdmissionError(
            "Commercial use was requested for a noncommercial source"
        )
    if raw_requested and not license_value["raw_redistribution_allowed"]:
        raise DataAdmissionError(
            "Raw redistribution was requested but the source license forbids it"
        )
    if (
        normalized_requested
        and not license_value["normalized_redistribution_allowed"]
    ):
        raise DataAdmissionError(
            "Normalized redistribution was requested but the source license "
            "forbids it"
        )

    noncommercial = not license_value["commercial_use_allowed"]
    share_alike = license_value["share_alike_required"]
    redistribution_restricted = not (
        license_value["raw_redistribution_allowed"]
        and license_value["normalized_redistribution_allowed"]
    )
    if noncommercial:
        expected_track = "noncommercial_isolated"
    elif redistribution_restricted:
        expected_track = "restricted_no_redistribution"
    else:
        expected_track = "general_research"
    if (
        track != expected_track
        or propagate_noncommercial != noncommercial
        or propagate_share_alike != share_alike
        or propagate_redistribution != redistribution_restricted
        or propagate_attribution != license_value["attribution_required"]
    ):
        raise DataAdmissionError(
            "Routing does not conservatively propagate the source license; "
            f"expected track={expected_track!r}, "
            f"noncommercial={noncommercial}, share_alike={share_alike}, "
            f"redistribution_restricted={redistribution_restricted}, "
            f"attribution={license_value['attribution_required']}"
        )
    active_restrictions = [
        name
        for name, active in (
            ("noncommercial", noncommercial),
            ("share_alike", share_alike),
            ("redistribution_restricted", redistribution_restricted),
        )
        if active
    ]
    policy = "none" if not active_restrictions else "+".join(active_restrictions)
    return track, policy, noncommercial


def _validate_duplicate_axis_receipt(
    value: Any,
    *,
    axis: str,
    dataset_id: str,
    admitted_manifest_sha256: str,
    axis_present: bool,
    expected_candidate_entity_count: int,
) -> _DuplicateAudit:
    """Validate one molecular-axis exact and near-duplicate audit."""
    name = f"admission.payload.duplicate_audit_receipts.{axis}"
    payload, receipt_sha256 = _validated_receipt(
        value,
        expected_format=DUPLICATE_AUDIT_RECEIPT_FORMAT,
        name=name,
    )
    _exact_fields(
        payload,
        {
            "dataset_id",
            "axis",
            "cross_source",
            "candidate_manifest_sha256",
            "candidate_entity_manifest_sha256",
            "candidate_entity_manifest_uri",
            "candidate_entity_count",
            "comparison_registry_sha256",
            "comparison_registry_uri",
            "comparison_source_count",
            "comparison_count",
            "exact_duplicate_method",
            "near_duplicate_method",
            "near_duplicate_threshold",
            "exact_duplicate_count",
            "near_duplicate_count_excluding_exact",
            "resolved_duplicate_count",
            "unresolved_duplicate_count",
            "resolution_action_counts",
            "resolution_manifest_sha256",
            "resolution_manifest_uri",
            "auditor_code_sha256",
            "created_at_utc",
        },
        f"{name}.payload",
    )
    if payload["dataset_id"] != dataset_id:
        raise DataAdmissionError(f"{name} is bound to the wrong dataset_id")
    if payload["axis"] != axis:
        raise DataAdmissionError(f"{name}.payload.axis must equal {axis!r}")
    if payload["cross_source"] is not True:
        raise DataAdmissionError(f"{name} must declare cross_source=true")
    if payload["candidate_manifest_sha256"] != admitted_manifest_sha256:
        raise DataAdmissionError(
            f"{name} is bound to the wrong admitted candidate manifest"
        )
    candidate_count = _nonnegative_integer(
        payload["candidate_entity_count"],
        f"{name}.payload.candidate_entity_count",
    )
    if candidate_count != expected_candidate_entity_count:
        raise DataAdmissionError(
            f"{name}.payload.candidate_entity_count does not match the audited "
            f"{axis} entity count"
        )
    if axis_present and candidate_count == 0:
        raise DataAdmissionError(
            f"{name} must audit at least one candidate entity for a present axis"
        )
    if not axis_present and candidate_count != 0:
        raise DataAdmissionError(
            f"{name} declares candidate entities for an absent molecular axis"
        )
    candidate_entity_manifest_sha256 = _sha256(
        payload["candidate_entity_manifest_sha256"],
        f"{name}.payload.candidate_entity_manifest_sha256",
    )
    _content_addressed_uri(
        payload["candidate_entity_manifest_uri"],
        candidate_entity_manifest_sha256,
        f"{name}.payload.candidate_entity_manifest_uri",
    )
    registry_sha256 = _sha256(
        payload["comparison_registry_sha256"],
        f"{name}.payload.comparison_registry_sha256",
    )
    _content_addressed_uri(
        payload["comparison_registry_uri"],
        registry_sha256,
        f"{name}.payload.comparison_registry_uri",
    )
    comparison_source_count = _positive_integer(
        payload["comparison_source_count"],
        f"{name}.payload.comparison_source_count",
    )
    comparison_count = _nonnegative_integer(
        payload["comparison_count"],
        f"{name}.payload.comparison_count",
    )
    exact_method = _nonempty_string(
        payload["exact_duplicate_method"],
        f"{name}.payload.exact_duplicate_method",
    )
    near_method = _nonempty_string(
        payload["near_duplicate_method"],
        f"{name}.payload.near_duplicate_method",
    )
    if exact_method != EXACT_DUPLICATE_METHOD:
        raise DataAdmissionError(
            f"{name}.payload.exact_duplicate_method must equal "
            f"{EXACT_DUPLICATE_METHOD!r}"
        )
    if near_method != NEAR_DUPLICATE_METHOD:
        raise DataAdmissionError(
            f"{name}.payload.near_duplicate_method must equal "
            f"{NEAR_DUPLICATE_METHOD!r}"
        )
    near_threshold = _closed_unit_interval(
        payload["near_duplicate_threshold"],
        f"{name}.payload.near_duplicate_threshold",
    )
    if near_threshold != NEAR_DUPLICATE_THRESHOLD:
        raise DataAdmissionError(
            f"{name}.payload.near_duplicate_threshold must equal "
            f"{NEAR_DUPLICATE_THRESHOLD}"
        )
    if (
        axis_present
        and comparison_count < candidate_count * comparison_source_count
    ):
        raise DataAdmissionError(
            f"{name} must cover every candidate across every comparison source"
        )
    exact_count = _nonnegative_integer(
        payload["exact_duplicate_count"],
        f"{name}.payload.exact_duplicate_count",
    )
    near_count = _nonnegative_integer(
        payload["near_duplicate_count_excluding_exact"],
        f"{name}.payload.near_duplicate_count_excluding_exact",
    )
    if exact_count + near_count > comparison_count:
        raise DataAdmissionError(
            f"{name} duplicate counts cannot exceed comparison_count"
        )
    resolved = _nonnegative_integer(
        payload["resolved_duplicate_count"],
        f"{name}.payload.resolved_duplicate_count",
    )
    unresolved = _nonnegative_integer(
        payload["unresolved_duplicate_count"],
        f"{name}.payload.unresolved_duplicate_count",
    )
    if resolved + unresolved != exact_count + near_count:
        raise DataAdmissionError(
            f"{name} resolved and unresolved counts do not cover all duplicates"
        )
    if unresolved != 0:
        raise DataAdmissionError(
            f"{name} contains unresolved cross-source duplicates"
        )
    actions = _mapping(
        payload["resolution_action_counts"],
        f"{name}.payload.resolution_action_counts",
    )
    _exact_fields(
        actions,
        {"quarantined", "merged", "linked_to_split_component"},
        f"{name}.payload.resolution_action_counts",
    )
    action_total = sum(
        _nonnegative_integer(
            actions[action],
            f"{name}.payload.resolution_action_counts.{action}",
        )
        for action in ("quarantined", "merged", "linked_to_split_component")
    )
    if action_total != resolved:
        raise DataAdmissionError(
            f"{name} resolution action counts do not match resolved duplicates"
        )
    resolution_sha256 = _sha256(
        payload["resolution_manifest_sha256"],
        f"{name}.payload.resolution_manifest_sha256",
    )
    _content_addressed_uri(
        payload["resolution_manifest_uri"],
        resolution_sha256,
        f"{name}.payload.resolution_manifest_uri",
    )
    auditor_code_sha256 = _sha256(
        payload["auditor_code_sha256"],
        f"{name}.payload.auditor_code_sha256",
    )
    created_at_utc = _utc_timestamp(
        payload["created_at_utc"],
        f"{name}.payload.created_at_utc",
    )
    return _DuplicateAudit(
        receipt_sha256=receipt_sha256,
        candidate_entity_manifest_sha256=candidate_entity_manifest_sha256,
        comparison_registry_sha256=registry_sha256,
        resolution_manifest_sha256=resolution_sha256,
        auditor_code_sha256=auditor_code_sha256,
        created_at_utc=created_at_utc,
    )


def _require_time_order(
    earlier: str,
    later: str,
    *,
    relationship: str,
) -> None:
    """Require two canonical UTC timestamps to respect causal ordering."""
    if earlier > later:
        raise DataAdmissionError(
            f"Timestamp order is invalid: {relationship}"
        )


def _require_distinct_role_hashes(role_hashes: Mapping[str, str]) -> None:
    """Reject reuse of the same artifact bytes under incompatible semantic roles."""
    seen: dict[str, str] = {}
    for role, digest in role_hashes.items():
        previous = seen.get(digest)
        if previous is not None:
            raise DataAdmissionError(
                "One content digest cannot serve incompatible artifact roles: "
                f"{previous!r} and {role!r}"
            )
        seen[digest] = role


def validate_dataset_admission(
    receipt: Mapping[str, Any],
    *,
    expected_admission_policy_sha256: str,
    source_registry_entry: Mapping[str, Any],
    expected_source_registry_entry_sha256: str,
    not_after_utc: str,
) -> DatasetAdmission:
    """Validate and admit one metadata-bound source, or raise fail-closed.

    This function performs no filesystem or network I/O. A successful result
    authorizes only the metadata contract. It does not authorize split
    construction, model training, model selection, or test-set access.

    The expected policy hash, source-registry digest, and validation cutoff are
    caller-owned inputs. Values declared only inside the admission receipt can
    never authorize themselves.
    """
    expected_policy_sha256 = _sha256(
        expected_admission_policy_sha256,
        "expected_admission_policy_sha256",
    )
    cutoff_utc = _utc_timestamp(not_after_utc, "not_after_utc")
    payload, admission_sha256 = _validated_receipt(
        receipt,
        expected_format=ADMISSION_RECEIPT_FORMAT,
        name="admission",
    )
    _exact_fields(
        payload,
        {
            "dataset_id",
            "source",
            "counts",
            "label_semantics",
            "raw_manifest_receipt",
            "normalized_manifest_receipt",
            "qc_passed_manifest_receipt",
            "deduplicated_manifest_receipt",
            "admitted_manifest_receipt",
            "lineage_receipt",
            "license",
            "routing",
            "duplicate_audit_receipts",
            "admission_policy_sha256",
            "source_registry_entry_sha256",
            "created_at_utc",
        },
        "admission.payload",
    )
    dataset_id = _identifier(
        payload["dataset_id"],
        "admission.payload.dataset_id",
    )
    (
        source_id,
        source_name,
        revision,
        acquired_at,
        semantic_type,
        record_unit,
        axes,
    ) = _validate_source(payload["source"], dataset_id=dataset_id)
    counts = _validate_counts(payload["counts"])
    _validate_semantic_counts(
        semantic_type,
        counts,
        record_unit=record_unit,
        axes=axes,
    )
    _validate_label_semantics(
        payload["label_semantics"],
        semantic_type=semantic_type,
        counts=counts,
    )

    raw_manifest = _validate_sha_manifest_receipt(
        payload["raw_manifest_receipt"],
        name="admission.payload.raw_manifest_receipt",
        dataset_id=dataset_id,
        stage="raw",
        expected_count=counts.raw_records,
    )
    normalized_manifest = _validate_sha_manifest_receipt(
        payload["normalized_manifest_receipt"],
        name="admission.payload.normalized_manifest_receipt",
        dataset_id=dataset_id,
        stage="normalized",
        expected_count=counts.normalized_records,
    )
    qc_passed_manifest = _validate_sha_manifest_receipt(
        payload["qc_passed_manifest_receipt"],
        name="admission.payload.qc_passed_manifest_receipt",
        dataset_id=dataset_id,
        stage="qc_passed",
        expected_count=counts.qc_passed_records,
    )
    deduplicated_manifest = _validate_sha_manifest_receipt(
        payload["deduplicated_manifest_receipt"],
        name="admission.payload.deduplicated_manifest_receipt",
        dataset_id=dataset_id,
        stage="deduplicated",
        expected_count=counts.deduplicated_records,
    )
    admitted_manifest = _validate_sha_manifest_receipt(
        payload["admitted_manifest_receipt"],
        name="admission.payload.admitted_manifest_receipt",
        dataset_id=dataset_id,
        stage="admitted",
        expected_count=counts.admitted_records,
    )
    stage_manifests = {
        "raw": raw_manifest,
        "normalized": normalized_manifest,
        "qc_passed": qc_passed_manifest,
        "deduplicated": deduplicated_manifest,
        "admitted": admitted_manifest,
    }
    lineage = _validate_lineage_receipt(
        payload["lineage_receipt"],
        dataset_id=dataset_id,
        stage_manifest_sha256={
            stage: audit.manifest_sha256
            for stage, audit in stage_manifests.items()
        },
        stage_counts={
            "raw": counts.raw_records,
            "normalized": counts.normalized_records,
            "qc_passed": counts.qc_passed_records,
            "deduplicated": counts.deduplicated_records,
            "admitted": counts.admitted_records,
        },
    )
    license_value = _validate_license(payload["license"])
    license_contract_sha256 = _domain_contract_sha256(
        LICENSE_CONTRACT_DOMAIN,
        license_value,
    )
    declared_policy_sha256 = _sha256(
        payload["admission_policy_sha256"],
        "admission.payload.admission_policy_sha256",
    )
    if declared_policy_sha256 != expected_policy_sha256:
        raise DataAdmissionError(
            "Admission receipt does not match the caller-frozen admission policy"
        )
    declared_registry_sha256 = _sha256(
        payload["source_registry_entry_sha256"],
        "admission.payload.source_registry_entry_sha256",
    )
    if declared_registry_sha256 != _sha256(
        expected_source_registry_entry_sha256,
        "expected_source_registry_entry_sha256",
    ):
        raise DataAdmissionError(
            "Admission receipt does not match the caller-frozen source registry"
        )
    (
        registry_sha256,
        registry_created_at,
        comparison_registry_contract,
    ) = _validate_source_registry_entry(
        source_registry_entry,
        expected_receipt_sha256=expected_source_registry_entry_sha256,
        expected_admission_policy_sha256=expected_policy_sha256,
        dataset_id=dataset_id,
        source=payload["source"],
        license_contract_sha256=license_contract_sha256,
    )
    if registry_sha256 != declared_registry_sha256:
        raise DataAdmissionError(
            "Admission and caller disagree on the source-registry entry"
        )
    track, policy, noncommercial = _validate_routing(
        payload["routing"],
        license_value=license_value,
    )

    duplicate_receipts = _mapping(
        payload["duplicate_audit_receipts"],
        "admission.payload.duplicate_audit_receipts",
    )
    _exact_fields(
        duplicate_receipts,
        set(MOLECULAR_AXES),
        "admission.payload.duplicate_audit_receipts",
    )
    duplicate_audits = {
        axis: _validate_duplicate_axis_receipt(
            duplicate_receipts[axis],
            axis=axis,
            dataset_id=dataset_id,
            admitted_manifest_sha256=admitted_manifest.manifest_sha256,
            axis_present=axis in axes,
            expected_candidate_entity_count=getattr(
                counts,
                f"{axis}_entities",
            ),
        )
        for axis in MOLECULAR_AXES
    }
    for axis, expected_registry_sha256 in comparison_registry_contract:
        if (
            duplicate_audits[axis].comparison_registry_sha256
            != expected_registry_sha256
        ):
            raise DataAdmissionError(
                f"The {axis} duplicate audit used a comparison registry that "
                "differs from the caller-frozen source-registry contract"
            )
    audit_hashes = tuple(
        (axis, duplicate_audits[axis].receipt_sha256)
        for axis in MOLECULAR_AXES
    )
    admission_created_at = _utc_timestamp(
        payload["created_at_utc"],
        "admission.payload.created_at_utc",
    )
    _require_time_order(
        acquired_at,
        raw_manifest.created_at_utc,
        relationship="source acquisition must not follow the raw manifest",
    )
    _require_time_order(
        raw_manifest.created_at_utc,
        normalized_manifest.created_at_utc,
        relationship="raw manifest creation must not follow normalization",
    )
    _require_time_order(
        normalized_manifest.created_at_utc,
        qc_passed_manifest.created_at_utc,
        relationship="normalization must not follow QC",
    )
    _require_time_order(
        qc_passed_manifest.created_at_utc,
        deduplicated_manifest.created_at_utc,
        relationship="QC must not follow deduplication",
    )
    _require_time_order(
        deduplicated_manifest.created_at_utc,
        admitted_manifest.created_at_utc,
        relationship="deduplication must not follow admission filtering",
    )
    _require_time_order(
        admitted_manifest.created_at_utc,
        lineage.created_at_utc,
        relationship="admitted manifest creation must not follow lineage audit",
    )
    for axis, audit in duplicate_audits.items():
        _require_time_order(
            admitted_manifest.created_at_utc,
            audit.created_at_utc,
            relationship=(
                f"admitted manifest creation must not follow the {axis} "
                "duplicate audit"
            ),
        )
        _require_time_order(
            audit.created_at_utc,
            admission_created_at,
            relationship=f"the {axis} duplicate audit must precede admission",
        )
    _require_time_order(
        lineage.created_at_utc,
        admission_created_at,
        relationship="lineage audit must precede admission",
    )
    _require_time_order(
        registry_created_at,
        admission_created_at,
        relationship="source-registry entry must precede admission",
    )
    _require_time_order(
        admission_created_at,
        cutoff_utc,
        relationship="admission must not be later than the caller cutoff",
    )

    _require_distinct_role_hashes(
        {
            "raw_manifest": raw_manifest.manifest_sha256,
            "normalized_manifest": normalized_manifest.manifest_sha256,
            "qc_passed_manifest": qc_passed_manifest.manifest_sha256,
            "deduplicated_manifest": deduplicated_manifest.manifest_sha256,
            "admitted_manifest": admitted_manifest.manifest_sha256,
            "lineage_manifest": lineage.manifest_sha256,
            "license_text": license_value["license_text_sha256"],
            **{
                f"{axis}_candidate_entity_manifest":
                duplicate_audits[axis].candidate_entity_manifest_sha256
                for axis in MOLECULAR_AXES
            },
            **{
                f"{axis}_comparison_registry":
                duplicate_audits[axis].comparison_registry_sha256
                for axis in MOLECULAR_AXES
            },
            **{
                f"{axis}_duplicate_resolution":
                duplicate_audits[axis].resolution_manifest_sha256
                for axis in MOLECULAR_AXES
            },
        }
    )
    producer_hashes = {
        raw_manifest.producer_code_sha256,
        normalized_manifest.producer_code_sha256,
        qc_passed_manifest.producer_code_sha256,
        deduplicated_manifest.producer_code_sha256,
        admitted_manifest.producer_code_sha256,
        lineage.transform_code_sha256,
    }
    for axis, audit in duplicate_audits.items():
        if audit.auditor_code_sha256 in producer_hashes:
            raise DataAdmissionError(
                f"The {axis} duplicate auditor code must be independent of "
                "manifest and lineage producer code"
            )

    return DatasetAdmission(
        dataset_id=dataset_id,
        source_id=source_id,
        source_name=source_name,
        source_revision=revision,
        acquired_at_utc=acquired_at,
        semantic_type=semantic_type,
        record_unit=record_unit,
        molecular_axes=axes,
        routing_track=track,
        contamination_policy=policy,
        noncommercial_contaminated=noncommercial,
        counts=counts,
        raw_manifest_sha256=raw_manifest.manifest_sha256,
        normalized_manifest_sha256=normalized_manifest.manifest_sha256,
        qc_passed_manifest_sha256=qc_passed_manifest.manifest_sha256,
        deduplicated_manifest_sha256=deduplicated_manifest.manifest_sha256,
        admitted_manifest_sha256=admitted_manifest.manifest_sha256,
        lineage_manifest_sha256=lineage.manifest_sha256,
        duplicate_audit_receipt_sha256=audit_hashes,
        source_registry_entry_sha256=registry_sha256,
        admission_policy_sha256=expected_policy_sha256,
        license_contract_sha256=license_contract_sha256,
        attribution_required=license_value["attribution_required"],
        admission_created_at_utc=admission_created_at,
        validated_not_after_utc=cutoff_utc,
        admission_receipt_sha256=admission_sha256,
    )


__all__ = [
    "ADMISSION_RECEIPT_FORMAT",
    "CANONICAL_JSON_POLICY",
    "DUPLICATE_AUDIT_RECEIPT_FORMAT",
    "EXACT_DUPLICATE_METHOD",
    "LINEAGE_RECEIPT_FORMAT",
    "MANIFEST_ENTRY_ORDER_POLICY",
    "NEAR_DUPLICATE_METHOD",
    "NEAR_DUPLICATE_THRESHOLD",
    "SHA_MANIFEST_RECEIPT_FORMAT",
    "SOURCE_REGISTRY_ENTRY_FORMAT",
    "AdmissionCounts",
    "DataAdmissionError",
    "DatasetAdmission",
    "make_content_addressed_receipt",
    "make_source_registry_entry",
    "validate_dataset_admission",
]
