"""Lossless candidate generation for exhaustive fixed-IMGT 90/90/90 search.

The final decision is always made from all 200 registered IMGT positions.  The
seed index is only an acceleration device.  After normalizing both kinds of
"no residue" byte to zero, a pair passing 90% position identity can disagree at
no more than 20 of the 200 positions.  The registered 25-way disjoint position
partition therefore guarantees at least five exact eight-byte seeds by the
pigeonhole principle.  Requiring all five remains lossless and maps directly
to exact uint64 equality on a GPU.  Every candidate is subsequently checked
for identity and reciprocal coverage over all 200 positions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Tuple

from .imgt_position import (
    CANONICAL_LENGTH,
    GAP,
    MISSING_END,
    CanonicalAlignment,
    PositionIdentity,
    position_identity,
)
from .imgt_query_export import IMGTQueryRecord
from .oas_imgt_export import IMGTTargetRecord


IDENTITY_NUMERATOR = 9
IDENTITY_DENOMINATOR = 10
QUERY_COVERAGE_NUMERATOR = 9
QUERY_COVERAGE_DENOMINATOR = 10
TARGET_COVERAGE_NUMERATOR = 9
TARGET_COVERAGE_DENOMINATOR = 10
MAXIMUM_QUALIFYING_DISAGREEMENTS = CANONICAL_LENGTH // IDENTITY_DENOMINATOR
SEED_GROUP_COUNT = 25
MINIMUM_QUALIFYING_EXACT_SEEDS = (
    SEED_GROUP_COUNT - MAXIMUM_QUALIFYING_DISAGREEMENTS
)


class IMGTExactSearchError(RuntimeError):
    """Report an invalid query index or a violated lossless-search invariant."""


def _registered_seed_groups() -> Tuple[Tuple[int, ...], ...]:
    """Create 25 disjoint, spatially distributed groups over all 200 positions."""
    permutation = tuple(
        (19 + 73 * index) % CANONICAL_LENGTH
        for index in range(CANONICAL_LENGTH)
    )
    if len(set(permutation)) != CANONICAL_LENGTH:
        raise RuntimeError("Fixed-IMGT seed permutation is not bijective")
    groups = tuple(
        tuple(sorted(permutation[group_id::SEED_GROUP_COUNT]))
        for group_id in range(SEED_GROUP_COUNT)
    )
    flattened = tuple(position for group in groups for position in group)
    if (
        len(groups) != SEED_GROUP_COUNT
        or any(len(group) != 8 for group in groups)
        or len(flattened) != CANONICAL_LENGTH
        or set(flattened) != set(range(CANONICAL_LENGTH))
    ):
        raise RuntimeError("Fixed-IMGT seed partition is not exhaustive and disjoint")
    return groups


SEED_POSITION_GROUPS = _registered_seed_groups()


def normalized_alignment_bytes(alignment: CanonicalAlignment) -> bytes:
    """Collapse gap and missing-end sentinels to the metric's same absent state."""
    alignment.validate()
    return bytes(
        GAP if value in {GAP, MISSING_END} else value for value in alignment.encoded
    )


def seed_signatures(alignment: CanonicalAlignment) -> Tuple[bytes, ...]:
    """Return exact, collision-free seed byte strings for the registered groups."""
    normalized = normalized_alignment_bytes(alignment)
    return tuple(bytes(normalized[position] for position in group) for group in SEED_POSITION_GROUPS)


def _integer_gate(metrics: PositionIdentity) -> bool:
    """Apply inclusive 90/90/90 without floating-point boundary decisions."""
    return (
        IDENTITY_DENOMINATOR * metrics.identical_positions
        >= IDENTITY_NUMERATOR * metrics.union_positions
        and QUERY_COVERAGE_DENOMINATOR * metrics.overlapping_positions
        >= QUERY_COVERAGE_NUMERATOR * metrics.query_positions
        and TARGET_COVERAGE_DENOMINATOR * metrics.overlapping_positions
        >= TARGET_COVERAGE_NUMERATOR * metrics.target_positions
    )


@dataclass(frozen=True)
class FixedIMGTMatch:
    """One fully verified target-query match at the registered 90/90/90 gate."""

    metrics: PositionIdentity
    query_ordinal: int
    target_authority_ordinal: int
    target_split: str

    def validate(self) -> None:
        if self.query_ordinal < 0 or self.target_authority_ordinal < 0:
            raise IMGTExactSearchError("Fixed-IMGT match ordinal is invalid")
        if self.target_split not in {"train", "validation"}:
            raise IMGTExactSearchError("Fixed-IMGT match split is invalid")
        if not _integer_gate(self.metrics):
            raise IMGTExactSearchError("Fixed-IMGT match does not pass 90/90/90")


@dataclass(frozen=True)
class FixedIMGTQueryIndex:
    """Small authenticated-query index used while streaming all OAS targets."""

    buckets: Mapping[Tuple[int, bytes], Tuple[int, ...]]
    queries_by_ordinal: Mapping[int, IMGTQueryRecord]

    def validate(self) -> None:
        ordinals = set(self.queries_by_ordinal)
        if not ordinals or min(ordinals) < 0:
            raise IMGTExactSearchError("Fixed-IMGT query index is empty or malformed")
        observed = set()
        for (group_id, signature), members in self.buckets.items():
            if (
                group_id not in range(SEED_GROUP_COUNT)
                or len(signature) != len(SEED_POSITION_GROUPS[group_id])
                or tuple(sorted(set(members))) != members
                or not set(members) <= ordinals
            ):
                raise IMGTExactSearchError("Fixed-IMGT seed bucket is malformed")
            observed.update(members)
        if observed != ordinals:
            raise IMGTExactSearchError("Fixed-IMGT query index coverage differs")


def build_query_index(records: Iterable[IMGTQueryRecord]) -> FixedIMGTQueryIndex:
    """Index every query under every seed without hashing or collision risk."""
    queries: Dict[int, IMGTQueryRecord] = {}
    mutable: Dict[Tuple[int, bytes], list[int]] = {}
    for record in records:
        if record.query_ordinal in queries:
            raise IMGTExactSearchError("Fixed-IMGT query ordinal is duplicated")
        record.alignment.validate()
        queries[record.query_ordinal] = record
        for group_id, signature in enumerate(seed_signatures(record.alignment)):
            mutable.setdefault((group_id, signature), []).append(record.query_ordinal)
    buckets = {
        key: tuple(sorted(members))
        for key, members in mutable.items()
    }
    result = FixedIMGTQueryIndex(
        buckets=buckets,
        queries_by_ordinal=dict(sorted(queries.items())),
    )
    result.validate()
    return result


def candidate_query_ordinals(
    target: IMGTTargetRecord,
    index: FixedIMGTQueryIndex,
) -> Tuple[int, ...]:
    """Return the lossless seed union for one target in deterministic order."""
    target.alignment.validate()
    exact_seed_counts: Dict[int, int] = {}
    for group_id, signature in enumerate(seed_signatures(target.alignment)):
        for query_ordinal in index.buckets.get((group_id, signature), ()):
            exact_seed_counts[query_ordinal] = exact_seed_counts.get(query_ordinal, 0) + 1
    return tuple(
        query_ordinal
        for query_ordinal, count in sorted(exact_seed_counts.items())
        if count >= MINIMUM_QUALIFYING_EXACT_SEEDS
    )


def verify_target_candidates(
    target: IMGTTargetRecord,
    index: FixedIMGTQueryIndex,
) -> Tuple[FixedIMGTMatch, ...]:
    """Apply full 200-position identity and reciprocal coverage to seed candidates."""
    matches = []
    for query_ordinal in candidate_query_ordinals(target, index):
        query = index.queries_by_ordinal[query_ordinal]
        metrics = position_identity(query.alignment, target.alignment)
        if _integer_gate(metrics):
            match = FixedIMGTMatch(
                metrics=metrics,
                query_ordinal=query_ordinal,
                target_authority_ordinal=target.authority_ordinal,
                target_split=target.split,
            )
            match.validate()
            matches.append(match)
    return tuple(matches)


def assert_lossless_candidate(
    query: IMGTQueryRecord,
    target: IMGTTargetRecord,
    index: FixedIMGTQueryIndex,
) -> None:
    """Runtime/test oracle: a qualifying pair must occur in the seed union."""
    metrics = position_identity(query.alignment, target.alignment)
    if not _integer_gate(metrics):
        return
    disagreements = metrics.union_positions - metrics.identical_positions
    if disagreements > MAXIMUM_QUALIFYING_DISAGREEMENTS:
        raise IMGTExactSearchError("90% identity disagreement bound was violated")
    if query.query_ordinal not in candidate_query_ordinals(target, index):
        raise IMGTExactSearchError("Lossless fixed-IMGT candidate guarantee was violated")


__all__ = [
    "FixedIMGTMatch",
    "FixedIMGTQueryIndex",
    "IDENTITY_DENOMINATOR",
    "IDENTITY_NUMERATOR",
    "IMGTExactSearchError",
    "MAXIMUM_QUALIFYING_DISAGREEMENTS",
    "MINIMUM_QUALIFYING_EXACT_SEEDS",
    "SEED_GROUP_COUNT",
    "SEED_POSITION_GROUPS",
    "assert_lossless_candidate",
    "build_query_index",
    "candidate_query_ordinals",
    "normalized_alignment_bytes",
    "seed_signatures",
    "verify_target_candidates",
]
