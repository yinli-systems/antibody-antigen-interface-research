"""Fail-closed dual-track near-identity decontamination for AsEP development.

The audit intentionally separates evidence ingestion from expensive sequence
search.  It reconciles a gap-aware MMseqs2 search with an independent,
antibody-specific exhaustive IMGT-position search.  No downstream transfer is
authorized unless both engines prove complete query and target coverage and
pass registered sensitivity controls.  Their hit union defines contamination;
metric-dependent disagreements are retained as evidence instead of discarded.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Iterable, Mapping

from .downstream_overlap import DevelopmentChainReference, DownstreamOverlapError


FORMAT_VERSION = "oas_asep_development_near_identity_audit_v1"
MMSEQS_ENGINE = "mmseqs2_gap_aware"
IMGT_ENGINE = "imgt_position_exhaustive"
REQUIRED_ENGINES = frozenset({MMSEQS_ENGINE, IMGT_ENGINE})
REQUIRED_TARGET_SPLITS = frozenset({"train", "validation"})
REQUIRED_SENSITIVITY_CONTROLS = frozenset(
    {
        "exact_match",
        "single_substitution_above_threshold",
        "internal_indel_above_threshold",
        "terminal_indel_above_threshold",
    }
)


class NearIdentityOverlapError(DownstreamOverlapError):
    """Report malformed, ambiguous, or out-of-protocol near-identity evidence."""


def _validate_reference(reference: DevelopmentChainReference) -> None:
    try:
        reference.validate()
    except DownstreamOverlapError as error:
        raise NearIdentityOverlapError(str(error)) from error


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def reference_key(reference: DevelopmentChainReference) -> str:
    """Return a non-identifying deterministic key for a development chain."""
    _validate_reference(reference)
    payload = {
        "chain": reference.chain,
        "instance_id": reference.instance_id,
        "role": reference.role,
        "sequence_length": reference.sequence_length,
        "sequence_sha256": reference.sequence_sha256,
    }
    return "asep_" + hashlib.sha256(_canonical_json(payload)).hexdigest()


def normalize_fraction(value: str | float) -> float:
    """Normalize MMseqs-style fractions that may be emitted as 0--1 or percent."""
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise NearIdentityOverlapError("Identity evidence is not numeric") from error
    if not math.isfinite(result) or result < 0.0:
        raise NearIdentityOverlapError("Identity evidence is outside its domain")
    if result > 1.0:
        result /= 100.0
    if result > 1.0:
        raise NearIdentityOverlapError("Identity evidence is outside its domain")
    return result


@dataclass(frozen=True)
class NearIdentityThresholds:
    """Registered inclusive identity and reciprocal-coverage thresholds."""

    identity: float = 0.90
    query_coverage: float = 0.90
    target_coverage: float = 0.90

    def validate(self) -> None:
        for name, value in (
            ("identity", self.identity),
            ("query coverage", self.query_coverage),
            ("target coverage", self.target_coverage),
        ):
            if not math.isfinite(value) or not 0.0 < value <= 1.0:
                raise NearIdentityOverlapError(f"Near-identity {name} is invalid")


@dataclass(frozen=True)
class NearIdentityHit:
    """One normalized query-to-OAS alignment from one registered engine."""

    engine: str
    query_reference_key: str
    target_id: str
    target_split: str
    identity: float
    query_coverage: float
    target_coverage: float

    def validate(self) -> None:
        if self.engine not in REQUIRED_ENGINES:
            raise NearIdentityOverlapError("Near-identity engine is not registered")
        if not self.query_reference_key.startswith("asep_"):
            raise NearIdentityOverlapError("Near-identity query key is malformed")
        if not self.target_id or any(character.isspace() for character in self.target_id):
            raise NearIdentityOverlapError("Near-identity target identifier is malformed")
        if self.target_split not in REQUIRED_TARGET_SPLITS:
            raise NearIdentityOverlapError("Near-identity target split is invalid")
        for value in (self.identity, self.query_coverage, self.target_coverage):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise NearIdentityOverlapError("Near-identity hit metric is invalid")

    def qualifies(self, thresholds: NearIdentityThresholds) -> bool:
        """Apply the frozen inclusive 90/90/90 gate with numerical tolerance."""
        self.validate()
        thresholds.validate()
        tolerance = 1e-12
        return (
            self.identity + tolerance >= thresholds.identity
            and self.query_coverage + tolerance >= thresholds.query_coverage
            and self.target_coverage + tolerance >= thresholds.target_coverage
        )


@dataclass(frozen=True)
class SearchTrackEvidence:
    """Completeness and hit evidence for one independent search track."""

    engine: str
    completed: bool
    query_reference_keys: frozenset[str]
    target_splits_completed: frozenset[str]
    maximum_sequences_truncated: bool
    unrepresentable_query_keys: frozenset[str]
    sensitivity_controls: Mapping[str, bool]
    hits: tuple[NearIdentityHit, ...]

    def validate_schema(self, expected_query_keys: frozenset[str]) -> None:
        if self.engine not in REQUIRED_ENGINES:
            raise NearIdentityOverlapError("Search track engine is not registered")
        if self.query_reference_keys - expected_query_keys:
            raise NearIdentityOverlapError("Search track contains an unknown query")
        if self.unrepresentable_query_keys - expected_query_keys:
            raise NearIdentityOverlapError("Search track has an unknown unrepresentable query")
        if self.target_splits_completed - REQUIRED_TARGET_SPLITS:
            raise NearIdentityOverlapError("Search track contains an unknown target split")
        if set(self.sensitivity_controls) != REQUIRED_SENSITIVITY_CONTROLS:
            raise NearIdentityOverlapError("Search track sensitivity-control set differs")
        for hit in self.hits:
            hit.validate()
            if hit.engine != self.engine:
                raise NearIdentityOverlapError("Hit is bound to a different search engine")
            if hit.query_reference_key not in expected_query_keys:
                raise NearIdentityOverlapError("Hit contains an unknown query")

    def completion_failures(self, expected_query_keys: frozenset[str]) -> list[str]:
        """List every fail-closed reason rather than short-circuiting evidence."""
        self.validate_schema(expected_query_keys)
        failures: list[str] = []
        if not self.completed:
            failures.append("search_not_completed")
        if self.query_reference_keys != expected_query_keys:
            failures.append("query_coverage_incomplete")
        if self.target_splits_completed != REQUIRED_TARGET_SPLITS:
            failures.append("target_split_coverage_incomplete")
        if self.maximum_sequences_truncated:
            failures.append("hit_list_truncated")
        if self.unrepresentable_query_keys:
            failures.append("unrepresentable_query")
        if not all(self.sensitivity_controls.values()):
            failures.append("sensitivity_control_failed")
        return failures


def parse_mmseqs_tsv(
    lines: Iterable[str],
    *,
    target_split_by_id: Mapping[str, str],
) -> tuple[NearIdentityHit, ...]:
    """Parse the frozen five-column ``query,target,fident,qcov,tcov`` format."""
    output: list[NearIdentityHit] = []
    seen: set[tuple[str, str]] = set()
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        fields = line.rstrip("\n").split("\t")
        if len(fields) != 5:
            raise NearIdentityOverlapError(
                f"MMseqs2 row {line_number} does not have five columns"
            )
        query, target, identity, query_coverage, target_coverage = fields
        key = (query, target)
        if key in seen:
            raise NearIdentityOverlapError("MMseqs2 evidence repeats a query-target row")
        seen.add(key)
        try:
            target_split = target_split_by_id[target]
        except KeyError as error:
            raise NearIdentityOverlapError("MMseqs2 target is absent from its manifest") from error
        hit = NearIdentityHit(
            engine=MMSEQS_ENGINE,
            query_reference_key=query,
            target_id=target,
            target_split=target_split,
            identity=normalize_fraction(identity),
            query_coverage=normalize_fraction(query_coverage),
            target_coverage=normalize_fraction(target_coverage),
        )
        hit.validate()
        output.append(hit)
    return tuple(output)


def reconcile_near_identity_evidence(
    *,
    references: Iterable[DevelopmentChainReference],
    exact_overlap_instance_ids: Iterable[str],
    tracks: Iterable[SearchTrackEvidence],
    thresholds: NearIdentityThresholds = NearIdentityThresholds(),
) -> dict[str, Any]:
    """Reconcile both searches and authorize only an explicitly clean subset."""
    thresholds.validate()
    items = tuple(references)
    if not items:
        raise NearIdentityOverlapError("Development reference set is empty")
    for item in items:
        _validate_reference(item)
    identities = [(item.instance_id, item.chain) for item in items]
    if len(identities) != len(set(identities)):
        raise NearIdentityOverlapError("Development chain reference is duplicated")
    key_to_reference = {reference_key(item): item for item in items}
    if len(key_to_reference) != len(items):
        raise NearIdentityOverlapError("Development query key is duplicated")
    expected_query_keys = frozenset(key_to_reference)
    instance_ids = {item.instance_id for item in items}
    exact_contaminated = set(exact_overlap_instance_ids)
    if exact_contaminated - instance_ids:
        raise NearIdentityOverlapError("Exact-overlap evidence contains an unknown instance")

    track_by_engine: dict[str, SearchTrackEvidence] = {}
    for track in tracks:
        if track.engine in track_by_engine:
            raise NearIdentityOverlapError("Near-identity search engine is duplicated")
        track.validate_schema(expected_query_keys)
        track_by_engine[track.engine] = track
    unknown_engines = set(track_by_engine) - REQUIRED_ENGINES
    if unknown_engines:
        raise NearIdentityOverlapError("Near-identity search engine set differs")

    completion_failures: dict[str, list[str]] = {}
    qualified_keys_by_engine: dict[str, set[str]] = {}
    hit_counts_by_engine: dict[str, int] = {}
    for engine in sorted(REQUIRED_ENGINES):
        track = track_by_engine.get(engine)
        if track is None:
            completion_failures[engine] = ["search_track_missing"]
            qualified_keys_by_engine[engine] = set()
            hit_counts_by_engine[engine] = 0
            continue
        completion_failures[engine] = track.completion_failures(expected_query_keys)
        qualified = {
            hit.query_reference_key
            for hit in track.hits
            if hit.qualifies(thresholds)
        }
        qualified_keys_by_engine[engine] = qualified
        hit_counts_by_engine[engine] = sum(
            hit.qualifies(thresholds) for hit in track.hits
        )

    mmseqs_keys = qualified_keys_by_engine[MMSEQS_ENGINE]
    imgt_keys = qualified_keys_by_engine[IMGT_ENGINE]
    disagreement_keys = mmseqs_keys ^ imgt_keys
    near_contaminated_keys = mmseqs_keys | imgt_keys
    near_contaminated_instances = {
        key_to_reference[key].instance_id for key in near_contaminated_keys
    }
    all_contaminated_instances = exact_contaminated | near_contaminated_instances
    clean_instances = sorted(instance_ids - all_contaminated_instances)
    both_tracks_complete = (
        set(track_by_engine) == REQUIRED_ENGINES
        and all(not failures for failures in completion_failures.values())
    )
    agreement_passed = not disagreement_keys
    transfer_authorized = both_tracks_complete and bool(clean_instances)

    records: list[dict[str, Any]] = []
    for key, item in sorted(
        key_to_reference.items(), key=lambda pair: (pair[1].instance_id, pair[1].chain)
    ):
        records.append(
            {
                "chain": item.chain,
                "exact_overlap_instance": item.instance_id in exact_contaminated,
                "imgt_position_near_identity": key in imgt_keys,
                "instance_id": item.instance_id,
                "mmseqs2_near_identity": key in mmseqs_keys,
                "query_reference_key": key,
                "role": item.role,
                "sequence_length": item.sequence_length,
                "sequence_sha256": item.sequence_sha256,
            }
        )

    return {
        "format_version": FORMAT_VERSION,
        "thresholds": {
            "identity": thresholds.identity,
            "query_coverage": thresholds.query_coverage,
            "target_coverage": thresholds.target_coverage,
            "inclusive": True,
        },
        "counts": {
            "development_chain_references": len(items),
            "development_instances": len(instance_ids),
            "exactly_contaminated_instances": len(exact_contaminated),
            "near_identity_contaminated_chain_references": len(near_contaminated_keys),
            "near_identity_contaminated_instances": len(near_contaminated_instances),
            "decontaminated_instances": len(clean_instances),
            "engine_disagreement_chain_references": len(disagreement_keys),
        },
        "qualified_hit_counts_by_engine": hit_counts_by_engine,
        "completion_failures_by_engine": completion_failures,
        "records": records,
        "exactly_or_near_identity_contaminated_instance_ids": sorted(
            all_contaminated_instances
        ),
        "decontaminated_development_instance_ids": clean_instances,
        "engine_disagreement_query_reference_keys": sorted(disagreement_keys),
        "transfer_gate": {
            "dual_track_evidence_completed": both_tracks_complete,
            "engine_agreement_passed": agreement_passed,
            "full_development_decontamination_passed": (
                both_tracks_complete and not all_contaminated_instances
            ),
            "decontaminated_subset_required": bool(all_contaminated_instances),
            "downstream_transfer_authorized": transfer_authorized,
            "authorized_instance_ids": clean_instances if transfer_authorized else [],
            "official_or_frozen_test_accessed": False,
        },
        "claim_boundary": (
            "This gate authorizes OAS representation transfer only for the listed AsEP "
            "development instances. It does not authorize official, frozen, or external "
            "test access and does not establish downstream benefit or partner specificity."
        ),
    }


__all__ = [
    "FORMAT_VERSION",
    "IMGT_ENGINE",
    "MMSEQS_ENGINE",
    "NearIdentityHit",
    "NearIdentityOverlapError",
    "NearIdentityThresholds",
    "REQUIRED_SENSITIVITY_CONTROLS",
    "SearchTrackEvidence",
    "normalize_fraction",
    "parse_mmseqs_tsv",
    "reconcile_near_identity_evidence",
    "reference_key",
]
