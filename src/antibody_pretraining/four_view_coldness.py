"""Leakage gate for antibody-and-antigen four-view cold splits.

The gate operates on content digests and precomputed sequence-similarity
relations.  It deliberately does not compute labels, inspect structures, or
read predictions.  Relation materialization (for example with MMseqs2 or an
exact aligner) is a separate authenticated stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


REQUIRED_SEQUENCE_VIEWS = (
    "vh",
    "vl",
    "concatenated_antibody",
    "antigen",
)


@dataclass(frozen=True)
class SequenceSimilarityRelation:
    """One unordered sequence relation from an authenticated all-pairs stage."""

    left_digest: str
    right_digest: str
    aligned_region_identity: float
    left_coverage: float
    right_coverage: float


@dataclass(frozen=True)
class FourViewColdnessAudit:
    status: str
    authorized: bool
    exact_crossing_count: int
    similarity_crossing_count: int
    crossing_count_by_view: Mapping[str, int]
    split_names: tuple[str, ...]
    identity_threshold: float
    coverage_threshold: float
    labels_unopened: bool
    cohort_frozen_before_audit: bool


def _validate_fraction(value: float, label: str) -> None:
    if not isinstance(value, (int, float)) or not 0.0 < float(value) <= 1.0:
        raise ValueError(f"{label} must be in (0, 1]")


def _canonical_memberships(
    memberships: Mapping[str, Sequence[tuple[str, str]]],
) -> tuple[dict[str, dict[str, frozenset[str]]], tuple[str, ...]]:
    if set(memberships) != set(REQUIRED_SEQUENCE_VIEWS):
        raise ValueError("Exactly four antibody/antigen sequence views are required")

    canonical: dict[str, dict[str, frozenset[str]]] = {}
    all_splits: set[str] = set()
    for view in REQUIRED_SEQUENCE_VIEWS:
        digest_to_splits: dict[str, set[str]] = {}
        for digest, split in memberships[view]:
            if not isinstance(digest, str) or len(digest) != 64:
                raise ValueError(f"{view} contains a non-SHA256 sequence digest")
            if not isinstance(split, str) or not split.strip():
                raise ValueError(f"{view} contains an empty split name")
            normalized_split = split.strip()
            digest_to_splits.setdefault(digest, set()).add(normalized_split)
            all_splits.add(normalized_split)
        if not digest_to_splits:
            raise ValueError(f"{view} membership is empty")
        canonical[view] = {
            digest: frozenset(splits)
            for digest, splits in sorted(digest_to_splits.items())
        }
    if len(all_splits) < 2:
        raise ValueError("Coldness audit requires at least two splits")
    return canonical, tuple(sorted(all_splits))


def audit_four_view_coldness(
    memberships: Mapping[str, Sequence[tuple[str, str]]],
    relations: Mapping[str, Iterable[SequenceSimilarityRelation]],
    *,
    identity_threshold: float = 0.40,
    coverage_threshold: float = 0.80,
    labels_unopened: bool,
    cohort_frozen_before_audit: bool,
) -> FourViewColdnessAudit:
    """Fail closed on any exact or threshold-level cross-split sequence edge.

    ``memberships`` may contain the same digest more than once.  A digest seen
    in multiple splits is an exact crossing.  ``relations`` must cover the same
    digest universe for each view; only relations meeting both the identity and
    reciprocal-coverage thresholds are relevant to the gate.

    A clean sequence audit alone is insufficient for a blind benchmark.  The
    returned authorization additionally requires that the cohort was frozen
    before this audit and that its labels remain unopened.
    """

    _validate_fraction(identity_threshold, "identity_threshold")
    _validate_fraction(coverage_threshold, "coverage_threshold")
    if set(relations) != set(REQUIRED_SEQUENCE_VIEWS):
        raise ValueError("Similarity relations must contain exactly four views")
    canonical, split_names = _canonical_memberships(memberships)

    exact_counts: dict[str, int] = {}
    similarity_counts: dict[str, int] = {}
    for view in REQUIRED_SEQUENCE_VIEWS:
        digest_to_splits = canonical[view]
        exact_counts[view] = sum(
            1 for splits in digest_to_splits.values() if len(splits) > 1
        )
        crossing_edges: set[tuple[str, str]] = set()
        for relation in relations[view]:
            if relation.left_digest == relation.right_digest:
                continue
            if (
                relation.left_digest not in digest_to_splits
                or relation.right_digest not in digest_to_splits
            ):
                raise ValueError(f"{view} relation references an unknown digest")
            values = (
                relation.aligned_region_identity,
                relation.left_coverage,
                relation.right_coverage,
            )
            if any(not isinstance(value, (int, float)) for value in values):
                raise ValueError(f"{view} relation has non-numeric statistics")
            if any(not 0.0 <= float(value) <= 1.0 for value in values):
                raise ValueError(f"{view} relation has statistics outside [0, 1]")
            if float(relation.aligned_region_identity) < identity_threshold:
                continue
            if min(float(relation.left_coverage), float(relation.right_coverage)) < (
                coverage_threshold
            ):
                continue
            left_splits = digest_to_splits[relation.left_digest]
            right_splits = digest_to_splits[relation.right_digest]
            if any(left != right for left in left_splits for right in right_splits):
                crossing_edges.add(tuple(sorted((relation.left_digest, relation.right_digest))))
        similarity_counts[view] = len(crossing_edges)

    exact_crossing_count = sum(exact_counts.values())
    similarity_crossing_count = sum(similarity_counts.values())
    crossing_count_by_view = {
        view: exact_counts[view] + similarity_counts[view]
        for view in REQUIRED_SEQUENCE_VIEWS
    }
    sequence_clean = exact_crossing_count == 0 and similarity_crossing_count == 0
    authorized = bool(
        sequence_clean and labels_unopened and cohort_frozen_before_audit
    )
    if exact_crossing_count:
        status = "FAIL_EXACT_FOUR_VIEW_CROSSING"
    elif similarity_crossing_count:
        status = "FAIL_THRESHOLD_FOUR_VIEW_CROSSING"
    elif not cohort_frozen_before_audit:
        status = "FAIL_COHORT_NOT_FROZEN_BEFORE_AUDIT"
    elif not labels_unopened:
        status = "FAIL_LABELS_ALREADY_OPENED"
    else:
        status = "PASS_FROZEN_UNOPENED_FOUR_VIEW_DOUBLE_COLD_GATE"

    return FourViewColdnessAudit(
        status=status,
        authorized=authorized,
        exact_crossing_count=exact_crossing_count,
        similarity_crossing_count=similarity_crossing_count,
        crossing_count_by_view=crossing_count_by_view,
        split_names=split_names,
        identity_threshold=float(identity_threshold),
        coverage_threshold=float(coverage_threshold),
        labels_unopened=labels_unopened,
        cohort_frozen_before_audit=cohort_frozen_before_audit,
    )
