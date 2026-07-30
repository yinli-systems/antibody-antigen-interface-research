"""Strict OAS row filtering and plan-bound record selection."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .tokenizer import TokenizerError, normalize_locus, validate_residue_sequence


FILTER_FORMAT = "oas_unpaired_sequence_filter_v1"


class PreprocessingError(ValueError):
    """Report an invalid filter policy or OAS row."""


@dataclass(frozen=True)
class AcceptedSequence:
    """Represent one quality-filtered sequence before exact deduplication."""

    chain_type: str
    sequence: str
    sequence_sha256: str
    status_flags: tuple[str, ...]


@dataclass(frozen=True)
class RowDecision:
    """Return either one accepted sequence or one stable rejection reason."""

    accepted: AcceptedSequence | None
    rejection_reason: str | None


@dataclass(frozen=True)
class CompiledUnpairedRowEvaluator:
    """Apply one validated immutable policy without per-row schema checks."""

    authority_chain: str
    boolean_requirements: tuple[tuple[str, bool], ...]
    fatal_markers: tuple[str, ...]
    maximum_length: int
    minimum_length: int
    sequence_field_priority: tuple[str, ...]

    @classmethod
    def from_policy(
        cls,
        policy: Mapping[str, Any],
        *,
        authority_chain: str,
    ) -> CompiledUnpairedRowEvaluator:
        """Validate and freeze all policy fields used by the row hot path."""
        validate_filter_policy(policy)
        if authority_chain not in {"Heavy", "Light"}:
            raise PreprocessingError("Authority chain must be Heavy or Light")
        return cls(
            authority_chain=authority_chain,
            boolean_requirements=tuple(policy["boolean_requirements"].items()),
            fatal_markers=tuple(
                marker.lower()
                for marker in policy["anarci_status"][
                    "fatal_case_insensitive_markers"
                ]
            ),
            maximum_length=int(policy["maximum_length"]),
            minimum_length=int(policy["minimum_length"]),
            sequence_field_priority=tuple(policy["sequence_field_priority"]),
        )

    def evaluate(self, row: Mapping[str, Any]) -> RowDecision:
        """Evaluate one row under the already validated immutable policy."""
        for field, required in self.boolean_requirements:
            observed = _parse_oas_boolean(row.get(field))
            if observed is None:
                return RowDecision(None, f"invalid_boolean_{field}")
            if observed is not required:
                return RowDecision(None, f"failed_{field}")
        sequence = _sequence_from_row(row, list(self.sequence_field_priority))
        if sequence is None:
            return RowDecision(None, "missing_sequence")
        try:
            canonical = validate_residue_sequence(
                sequence,
                minimum_length=self.minimum_length,
                maximum_length=self.maximum_length,
            )
        except TokenizerError as error:
            message = str(error)
            reason = "invalid_length" if "length" in message else "invalid_residue"
            return RowDecision(None, reason)
        try:
            chain_type = normalize_locus(row.get("locus", ""))
        except TokenizerError:
            return RowDecision(None, "invalid_locus")
        if self.authority_chain == "Heavy" and chain_type != "heavy":
            return RowDecision(None, "authority_locus_disagreement")
        if self.authority_chain == "Light" and chain_type not in {
            "kappa",
            "lambda",
        }:
            return RowDecision(None, "authority_locus_disagreement")
        status = row.get("ANARCI_status")
        if not isinstance(status, str):
            return RowDecision(None, "missing_anarci_status")
        lowered = status.lower()
        if any(marker in lowered for marker in self.fatal_markers):
            return RowDecision(None, "fatal_anarci_status")
        sequence_sha256 = hashlib.sha256(canonical.encode("ascii")).hexdigest()
        return RowDecision(
            AcceptedSequence(
                chain_type=chain_type,
                sequence=canonical,
                sequence_sha256=sequence_sha256,
                status_flags=_status_flags(status),
            ),
            None,
        )


def load_filter_policy(path: Path) -> tuple[dict[str, Any], str]:
    """Load one byte-anchored ASCII filtering policy."""
    payload = path.read_bytes()
    try:
        policy = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PreprocessingError("Filter policy must be ASCII JSON") from error
    validate_filter_policy(policy)
    return policy, hashlib.sha256(payload).hexdigest()


def validate_filter_policy(policy: Mapping[str, Any]) -> None:
    """Validate frozen row-quality semantics."""
    if policy.get("format_version") != FILTER_FORMAT:
        raise PreprocessingError("Unexpected filter-policy version")
    minimum = policy.get("minimum_length")
    maximum = policy.get("maximum_length")
    if (
        isinstance(minimum, bool)
        or isinstance(maximum, bool)
        or not isinstance(minimum, int)
        or not isinstance(maximum, int)
        or minimum <= 0
        or minimum > maximum
    ):
        raise PreprocessingError("Filter length bounds are invalid")
    fields = policy.get("sequence_field_priority")
    if not isinstance(fields, list) or not fields or len(fields) != len(set(fields)):
        raise PreprocessingError("Sequence field priority is invalid")
    if policy.get("boolean_requirements") != {
        "productive": True,
        "stop_codon": False,
        "vj_in_frame": True,
    }:
        raise PreprocessingError("Boolean sequence-quality requirements differ")
    status = policy.get("anarci_status")
    if not isinstance(status, Mapping):
        raise PreprocessingError("ANARCI status policy is absent")
    fatal = status.get("fatal_case_insensitive_markers")
    if not isinstance(fatal, list) or not fatal or any(
        not isinstance(marker, str) or not marker for marker in fatal
    ):
        raise PreprocessingError("ANARCI fatal markers are invalid")
    if policy.get("test_access") != {
        "downstream_metrics_used": False,
        "sabdab2_heldout_accessed": False,
    }:
        raise PreprocessingError("Filter-policy test-access boundary is open")


def _parse_oas_boolean(value: Any) -> bool | None:
    """Parse explicit OAS boolean spellings without Python truthiness."""
    if isinstance(value, bool):
        return value
    if not isinstance(value, str):
        return None
    normalized = value.strip().upper()
    if normalized in {"T", "TRUE", "1"}:
        return True
    if normalized in {"F", "FALSE", "0"}:
        return False
    return None


def _sequence_from_row(
    row: Mapping[str, Any], field_priority: list[str]
) -> str | None:
    """Return the first nonempty sequence field under frozen priority."""
    for field in field_priority:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip().upper()
    return None


def _status_flags(status: str) -> tuple[str, ...]:
    """Map verbose ANARCI status text to bounded audit categories."""
    normalized = status.lower()
    flags = []
    for marker, label in (
        ("deletion", "deletion"),
        ("insertion", "insertion"),
        ("shorter than imgt defined", "shorter"),
    ):
        if marker in normalized:
            flags.append(label)
    if status.strip() and not flags:
        flags.append("other_nonfatal")
    return tuple(flags)


def evaluate_unpaired_row(
    row: Mapping[str, Any],
    *,
    authority_chain: str,
    policy: Mapping[str, Any],
) -> RowDecision:
    """Apply productive, frame, residue, locus, length, and ANARCI gates."""
    evaluator = CompiledUnpairedRowEvaluator.from_policy(
        policy,
        authority_chain=authority_chain,
    )
    return evaluator.evaluate(row)


def selected_by_plan_fraction(
    sequence_text: str,
    *,
    namespace_sha256: str,
    numerator: int,
    denominator: int,
) -> bool:
    """Apply an exact rational SHA-256 threshold before quality filtering."""
    if (
        not isinstance(namespace_sha256, str)
        or len(namespace_sha256) != 64
        or isinstance(numerator, bool)
        or isinstance(denominator, bool)
        or not isinstance(numerator, int)
        or not isinstance(denominator, int)
        or numerator <= 0
        or numerator > denominator
    ):
        raise PreprocessingError("Plan selection fraction is invalid")
    if numerator == denominator:
        return True
    try:
        namespace = bytes.fromhex(namespace_sha256)
        sequence_bytes = sequence_text.strip().upper().encode("ascii")
    except (ValueError, UnicodeEncodeError) as error:
        raise PreprocessingError("Plan selection key is not ASCII") from error
    digest_integer = int.from_bytes(
        hashlib.sha256(namespace + b"\x00" + sequence_bytes).digest(),
        "big",
    )
    return digest_integer * denominator < numerator * (1 << 256)
