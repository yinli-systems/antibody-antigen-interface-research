"""Readiness diagnostics for the provisional two-axis S2 fold assignment.

``protocols/train_only_s2_feasibility/s2_assignment_provisional_v1.json`` holds
a three-fold, dual-axis split of the authority-train pairs that HiGHS proved
optimal at a zero MIP gap: 1,607 of 1,830 unique exact pairs retained, zero
cross-fold interactions, and every axis degree inside the balance band. It is
marked ``unapproved_pending_exact_four_view_nw_audit`` and has stayed there.

The repository's status table describes the V6 workstream as a linear projection
reported infeasible. It does not mention this artifact, so a reader concludes
that no dual-axis split exists. One does; it is unapproved, which is a different
thing, and the distinction changes what work is worth doing next.

This module supplies the arithmetic that turns "infeasible" into a number.

Retention frontier
    A three-fold split retaining ``R`` pairs trains an outer fold on ``R - f_i``
    pairs. The largest per-fold training floor it can support is
    ``R - max(f_i)``, and because some fold must hold at least ``ceil(R / 3)``
    pairs, no assignment of ``R`` pairs supports a floor above
    ``R - ceil(R / 3)``. Inverting that gives the retention any floor requires.
    When the optimal ``R`` falls short of it, infeasibility follows from
    counting rather than from a solver, and the shortfall says how much
    retention would have to improve.

Internal consistency
    Retained and discarded pairs and rows must partition the totals, fold counts
    must sum to the retained total, and the recorded axis degrees must lie in the
    recorded band. These are cheap to recheck and bind nothing external.

Input identity
    The assignment binds its inputs by digest. ``authority_membership_report``
    separates a file whose bytes changed from a file whose *science* changed, by
    recomputing the frozen membership digests over the split it carries. A
    reformatted file with identical membership is a registration problem; a file
    whose membership moved is a scientific one.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
from collections.abc import Iterable, Mapping, Sequence


class S2ReadinessError(ValueError):
    """Raised when an artifact cannot support the requested diagnostic."""


def membership_sha256(values: Iterable[str]) -> str:
    """Hash sorted unique ASCII identifiers with the frozen LF framing.

    Mirrors ``interaction_first.score_free_authority.membership_sha256`` so this
    module stays importable without the score-free authority stack.
    """
    digest = hashlib.sha256()
    for value in sorted(set(values)):
        if not isinstance(value, str) or not value or not value.isascii():
            raise S2ReadinessError("membership values must be non-empty ASCII")
        digest.update(value.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


@dataclasses.dataclass(frozen=True)
class RetentionFrontier:
    """What per-fold training floor a retained-pair budget can support."""

    retained_pairs: int
    fold_sizes: tuple[int, ...]
    achieved_training_floor: int
    best_possible_training_floor: int
    required_floor: int | None
    retention_required_for_floor: int | None

    @property
    def floor_is_reachable(self) -> bool:
        """Whether the required floor is attainable at this retention."""
        if self.required_floor is None:
            return True
        return self.best_possible_training_floor >= self.required_floor

    @property
    def retention_shortfall(self) -> int:
        """Pairs of retention still needed for the required floor."""
        if self.retention_required_for_floor is None:
            return 0
        return max(0, self.retention_required_for_floor - self.retained_pairs)

    def as_json(self) -> dict[str, object]:
        return {
            "retained_pairs": self.retained_pairs,
            "fold_sizes": list(self.fold_sizes),
            "achieved_training_floor": self.achieved_training_floor,
            "best_possible_training_floor": self.best_possible_training_floor,
            "required_floor": self.required_floor,
            "retention_required_for_floor": self.retention_required_for_floor,
            "floor_is_reachable": self.floor_is_reachable,
            "retention_shortfall": self.retention_shortfall,
        }


def retention_required_for_floor(floor: int, fold_count: int) -> int:
    """Smallest retained-pair count able to support a per-fold training floor.

    Some fold must hold at least ``ceil(R / k)`` pairs, so the training set of
    the largest fold has at most ``R - ceil(R / k)``. The requirement is
    monotone in ``R``, so the smallest sufficient ``R`` is found by search.
    """
    if floor < 0:
        raise S2ReadinessError("floor must be nonnegative")
    if fold_count < 2:
        raise S2ReadinessError("fold_count must be at least two")
    retained = floor
    while retained - math.ceil(retained / fold_count) < floor:
        retained += 1
    return retained


def retention_frontier(
    fold_sizes: Sequence[int],
    *,
    required_floor: int | None = None,
) -> RetentionFrontier:
    """Describe the training floors a fold assignment can and cannot support."""
    sizes = tuple(int(size) for size in fold_sizes)
    if len(sizes) < 2:
        raise S2ReadinessError("at least two folds are required")
    if any(size < 0 for size in sizes):
        raise S2ReadinessError("fold sizes must be nonnegative")
    retained = sum(sizes)
    best_possible = retained - math.ceil(retained / len(sizes))
    required_retention = (
        retention_required_for_floor(required_floor, len(sizes))
        if required_floor is not None
        else None
    )
    return RetentionFrontier(
        retained_pairs=retained,
        fold_sizes=sizes,
        achieved_training_floor=retained - max(sizes),
        best_possible_training_floor=best_possible,
        required_floor=required_floor,
        retention_required_for_floor=required_retention,
    )


def consistency_report(assignment: Mapping[str, object]) -> dict[str, object]:
    """Recheck the assignment's own accounting without consulting any input."""
    try:
        accounting = dict(assignment["ledger_accounting"])  # type: ignore[arg-type]
        summary = dict(assignment["solution_summary"])  # type: ignore[arg-type]
        solver = dict(assignment["solver_result"])  # type: ignore[arg-type]
        folds = list(summary["folds"])  # type: ignore[index]
    except (KeyError, TypeError) as error:
        raise S2ReadinessError(f"assignment is missing a required section: {error}") from error

    fold_pairs = [int(fold["retained_unique_exact_pair_count"]) for fold in folds]
    fold_rows = [int(fold["retained_source_row_count"]) for fold in folds]
    bounds = dict(summary.get("axis_balance_bounds", {}))
    degrees = [
        int(fold[key])
        for fold in folds
        for key in ("assigned_antibody_unique_pair_degree", "assigned_antigen_unique_pair_degree")
    ]
    lower, upper = bounds.get("lower"), bounds.get("upper")

    checks = {
        "pairs_partition_total": (
            int(accounting["retained_unique_exact_pair_count"])
            + int(accounting["discarded_unique_exact_pair_count"])
            == int(accounting["all_unique_exact_pair_count"])
        ),
        "rows_partition_total": (
            int(accounting["retained_source_row_count"])
            + int(accounting["discarded_source_row_count"])
            == int(accounting["all_authority_train_source_row_count"])
        ),
        "fold_pairs_sum_to_retained": sum(fold_pairs)
        == int(accounting["retained_unique_exact_pair_count"]),
        "fold_rows_sum_to_retained": sum(fold_rows)
        == int(accounting["retained_source_row_count"]),
        "no_cross_fold_interactions_retained": int(summary["cross_fold_interactions_retained"]) == 0,
        "solver_reported_optimal": float(solver["mip_gap"]) == 0.0,
        "solver_objective_matches_accounting": int(
            solver["optimal_retained_unique_exact_pair_count"]
        )
        == int(accounting["retained_unique_exact_pair_count"]),
        "axis_degrees_within_recorded_bounds": (
            lower is not None
            and upper is not None
            and all(float(lower) <= degree <= float(upper) for degree in degrees)
        ),
    }
    return {
        "checks": checks,
        "all_passed": all(checks.values()),
        "fold_pair_counts": fold_pairs,
        "retained_pairs": int(accounting["retained_unique_exact_pair_count"]),
        "discarded_pairs": int(accounting["discarded_unique_exact_pair_count"]),
        "total_pairs": int(accounting["all_unique_exact_pair_count"]),
        "retained_pair_fraction": float(summary["retained_unique_exact_pair_fraction"]),
    }


def authority_membership_report(
    assignment: Mapping[str, object],
    splits: Mapping[str, Sequence[str]],
) -> dict[str, object]:
    """Compare a split file's membership against the digests the assignment binds.

    Separates three outcomes that a file digest alone cannot tell apart: the
    science is unchanged, the sealed pool is unchanged but its internal partition
    moved, or the membership itself changed.
    """
    try:
        authority = dict(assignment["authority"])  # type: ignore[arg-type]
    except (KeyError, TypeError) as error:
        raise S2ReadinessError("assignment has no authority section") from error

    source_split = str(authority["source_pool_split"])
    sealed_splits = [str(name) for name in authority["sealed_splits"]]
    for name in [source_split, *sealed_splits]:
        if name not in splits:
            raise S2ReadinessError(f"split file has no {name!r} split")

    train = list(splits[source_split])
    sealed = [identifier for name in sealed_splits for identifier in splits[name]]
    recorded_counts = {
        str(key): int(value)
        for key, value in dict(authority.get("sealed_split_counts", {})).items()
    }
    observed_counts = {name: len(splits[name]) for name in sealed_splits}

    train_digest = membership_sha256(train)
    sealed_digest = membership_sha256(sealed)
    return {
        "source_pool_split": source_split,
        "authorized_membership_matches": train_digest
        == str(authority["authorized_membership_sha256"]),
        "authorized_instance_count_matches": len(set(train))
        == int(authority["authorized_instance_count"]),
        "sealed_membership_matches": sealed_digest
        == str(authority["sealed_membership_sha256"]),
        "sealed_instance_count_matches": len(set(sealed))
        == int(authority["sealed_instance_count"]),
        "sealed_partition_matches": observed_counts == recorded_counts,
        "recorded_sealed_counts": recorded_counts,
        "observed_sealed_counts": observed_counts,
        "train_sealed_overlap": len(set(train) & set(sealed)),
        "observed_authorized_membership_sha256": train_digest,
        "observed_sealed_membership_sha256": sealed_digest,
    }
