"""Exact joint MILP formulation for nested train-only split feasibility.

This module operates only on canonical entity identifiers, positive-pair
incidence, and precomputed direct-neighborhood graphs. It does not load source
manifests, sequences, labels, model outputs, or checkpoints.

The formulation is joint: all three outer folds and all nine dependent inner
folds are represented in one binary program. Consequently, an outer assignment
cannot be accepted unless every child inner problem is feasible under the
locked policy. The default resource limits intentionally block a production
authority-train build until its sparse-matrix resource plan has been reviewed.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from itertools import combinations
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class Ratio:
    """One non-negative exact rational number."""

    numerator: int
    denominator: int

    def __post_init__(self) -> None:
        if self.numerator < 0 or self.denominator <= 0:
            raise ValueError("A ratio requires numerator >= 0 and denominator > 0")


@dataclass(frozen=True)
class JointLevelPolicy:
    """Exact integer/rational gates for one three-fold level."""

    minimum_validation_pairs: int
    minimum_validation_endpoints_per_axis: int
    validation_pair_lower_fraction: Ratio
    validation_pair_upper_fraction: Ratio
    minimum_training_pair_fraction: Ratio
    minimum_training_endpoint_fraction: Ratio
    maximum_quarantine_pair_fraction: Ratio
    maximum_validation_pair_relative_range: Ratio
    maximum_other_count_relative_range: Ratio

    def __post_init__(self) -> None:
        if self.minimum_validation_pairs < 0:
            raise ValueError("minimum_validation_pairs must be non-negative")
        if self.minimum_validation_endpoints_per_axis < 0:
            raise ValueError(
                "minimum_validation_endpoints_per_axis must be non-negative"
            )
        for name in (
            "validation_pair_lower_fraction",
            "validation_pair_upper_fraction",
            "minimum_training_pair_fraction",
            "minimum_training_endpoint_fraction",
            "maximum_quarantine_pair_fraction",
            "maximum_validation_pair_relative_range",
            "maximum_other_count_relative_range",
        ):
            ratio = getattr(self, name)
            if ratio.numerator > ratio.denominator:
                raise ValueError(f"{name} must be in [0, 1]")
        lower = self.validation_pair_lower_fraction
        upper = self.validation_pair_upper_fraction
        if lower.numerator * upper.denominator > (
            upper.numerator * lower.denominator
        ):
            raise ValueError(
                "The validation lower fraction exceeds the upper fraction"
            )


LOCKED_OUTER_POLICY = JointLevelPolicy(
    minimum_validation_pairs=120,
    minimum_validation_endpoints_per_axis=60,
    validation_pair_lower_fraction=Ratio(7, 100),
    validation_pair_upper_fraction=Ratio(9, 100),
    minimum_training_pair_fraction=Ratio(3, 5),
    minimum_training_endpoint_fraction=Ratio(3, 5),
    maximum_quarantine_pair_fraction=Ratio(33, 100),
    maximum_validation_pair_relative_range=Ratio(3, 20),
    maximum_other_count_relative_range=Ratio(1, 5),
)


LOCKED_INNER_POLICY = JointLevelPolicy(
    minimum_validation_pairs=80,
    minimum_validation_endpoints_per_axis=40,
    validation_pair_lower_fraction=Ratio(7, 100),
    validation_pair_upper_fraction=Ratio(9, 100),
    minimum_training_pair_fraction=Ratio(11, 20),
    minimum_training_endpoint_fraction=Ratio(11, 20),
    maximum_quarantine_pair_fraction=Ratio(19, 50),
    maximum_validation_pair_relative_range=Ratio(1, 5),
    maximum_other_count_relative_range=Ratio(1, 4),
)


def _ratio_payload(value: Ratio) -> list[int]:
    """Serialize one exact ratio without floating-point conversion."""
    return [value.numerator, value.denominator]


def joint_policy_payload(
    outer_policy: JointLevelPolicy,
    inner_policy: JointLevelPolicy,
) -> dict[str, object]:
    """Return the complete exact 3x3 policy binding."""

    def level_payload(policy: JointLevelPolicy) -> dict[str, object]:
        return {
            "minimum_validation_pairs": policy.minimum_validation_pairs,
            "minimum_validation_endpoints_per_axis": (
                policy.minimum_validation_endpoints_per_axis
            ),
            "validation_pair_lower_fraction": _ratio_payload(
                policy.validation_pair_lower_fraction
            ),
            "validation_pair_upper_fraction": _ratio_payload(
                policy.validation_pair_upper_fraction
            ),
            "minimum_training_pair_fraction": _ratio_payload(
                policy.minimum_training_pair_fraction
            ),
            "minimum_training_endpoint_fraction": _ratio_payload(
                policy.minimum_training_endpoint_fraction
            ),
            "maximum_quarantine_pair_fraction": _ratio_payload(
                policy.maximum_quarantine_pair_fraction
            ),
            "maximum_validation_pair_relative_range": _ratio_payload(
                policy.maximum_validation_pair_relative_range
            ),
            "maximum_other_count_relative_range": _ratio_payload(
                policy.maximum_other_count_relative_range
            ),
        }

    return {
        "schema_version": "joint_nested_split_exact_policy_v1",
        "outer_fold_count": 3,
        "dependent_inner_fold_count_per_outer": 3,
        "outer": level_payload(outer_policy),
        "inner": level_payload(inner_policy),
    }


def joint_policy_sha256(
    outer_policy: JointLevelPolicy,
    inner_policy: JointLevelPolicy,
) -> str:
    """Hash the complete exact policy with canonical JSON."""
    encoded = json.dumps(
        joint_policy_payload(outer_policy, inner_policy),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


LOCKED_JOINT_POLICY_SHA256 = joint_policy_sha256(
    LOCKED_OUTER_POLICY,
    LOCKED_INNER_POLICY,
)


@dataclass(frozen=True)
class CanonicalPair:
    """One indivisible canonical antibody-antigen positive pair."""

    pair_id: str
    antibody_id: str
    antigen_id: str


@dataclass(frozen=True)
class JointSplitProblem:
    """Metadata-only input to the joint nested formulation."""

    pairs: tuple[CanonicalPair, ...]
    antibody_direct_neighbors: Mapping[str, frozenset[str]]
    antigen_direct_neighbors: Mapping[str, frozenset[str]]


@dataclass(frozen=True)
class ModelSizeEstimate:
    """Matrix dimensions and a safe nonzero pre-allocation bound."""

    pair_count: int
    antibody_count: int
    antigen_count: int
    antibody_undirected_edge_count: int
    antigen_undirected_edge_count: int
    directed_closed_relation_count: int
    binary_variable_count: int
    linear_constraint_count: int
    nonzero_coefficient_count: int


@dataclass(frozen=True)
class ResourceLimits:
    """Fail-closed limits for the in-memory SciPy prototype."""

    maximum_binary_variables: int = 100_000
    maximum_linear_constraints: int = 500_000
    maximum_nonzero_coefficients: int = 5_000_000

    def check(self, estimate: ModelSizeEstimate) -> None:
        """Reject an unreviewed model before sparse-matrix allocation."""
        violations: list[str] = []
        if estimate.binary_variable_count > self.maximum_binary_variables:
            violations.append(
                "binary_variables="
                f"{estimate.binary_variable_count}>{self.maximum_binary_variables}"
            )
        if estimate.linear_constraint_count > self.maximum_linear_constraints:
            violations.append(
                "linear_constraints="
                f"{estimate.linear_constraint_count}>"
                f"{self.maximum_linear_constraints}"
            )
        if estimate.nonzero_coefficient_count > self.maximum_nonzero_coefficients:
            violations.append(
                "nonzero_coefficients="
                f"{estimate.nonzero_coefficient_count}>"
                f"{self.maximum_nonzero_coefficients}"
            )
        if violations:
            raise ResourceLimitError(
                "Joint MILP exceeds the reviewed prototype limits: "
                + ", ".join(violations)
            )


class ResourceLimitError(RuntimeError):
    """Raised before an unreviewed large sparse matrix is allocated."""


class PolicyArithmeticError(ValueError):
    """Raised when a fixed-universe validation interval is empty."""


@dataclass(frozen=True)
class LinearRow:
    """One integer-coefficient bounded linear row."""

    coefficients: tuple[tuple[int, int], ...]
    lower_bound: int | None
    upper_bound: int | None


@dataclass(frozen=True)
class JointMILPModel:
    """A fully materialized binary feasibility model."""

    variable_keys: tuple[tuple[str, ...], ...]
    rows: tuple[LinearRow, ...]
    estimate: ModelSizeEstimate
    matrix_sha256: str


@dataclass(frozen=True)
class JointSolveResult:
    """Fail-closed solve receipt for one unscored feasibility model."""

    solver: str
    solver_status: int
    solver_termination: str
    solver_message: str
    solver_reported_success: bool
    solver_solution_present: bool
    solver_certified_optimal: bool
    solver_reported_mip_gap: float | None
    solver_reported_mip_gap_is_finite_zero: bool
    maximum_integrality_error: float | None
    accepted_integrality_tolerance: float
    exact_integer_row_audit_passed: bool
    feasible_for_unscored_split_construction: bool
    authorized_for_training: bool
    authorized_for_model_selection: bool
    matrix_sha256: str
    binary_variable_count: int
    linear_constraint_count: int
    nonzero_coefficient_count: int
    assignment: Mapping[tuple[str, ...], int] | None


class _ModelBuilder:
    """Small deterministic integer-row builder used by the prototype."""

    def __init__(self) -> None:
        self.variable_keys: list[tuple[str, ...]] = []
        self.variable_index: dict[tuple[str, ...], int] = {}
        self.rows: list[LinearRow] = []

    def variable(self, *key: object) -> int:
        normalized = tuple(str(value) for value in key)
        existing = self.variable_index.get(normalized)
        if existing is not None:
            return existing
        index = len(self.variable_keys)
        self.variable_keys.append(normalized)
        self.variable_index[normalized] = index
        return index

    def add_row(
        self,
        coefficients: Iterable[tuple[int, int]],
        *,
        lower_bound: int | None = None,
        upper_bound: int | None = None,
    ) -> None:
        combined: dict[int, int] = {}
        for column, coefficient in coefficients:
            if coefficient:
                combined[column] = combined.get(column, 0) + int(coefficient)
        normalized = tuple(
            (column, coefficient)
            for column, coefficient in sorted(combined.items())
            if coefficient
        )
        if not normalized:
            value = 0
            if lower_bound is not None and value < lower_bound:
                raise PolicyArithmeticError("A constant lower bound is infeasible")
            if upper_bound is not None and value > upper_bound:
                raise PolicyArithmeticError("A constant upper bound is infeasible")
            return
        if (
            lower_bound is not None
            and upper_bound is not None
            and lower_bound > upper_bound
        ):
            raise ValueError("A linear row has lower_bound > upper_bound")
        self.rows.append(LinearRow(normalized, lower_bound, upper_bound))


def _ceil_ratio_product(ratio: Ratio, value: int) -> int:
    return (ratio.numerator * value + ratio.denominator - 1) // ratio.denominator


def _floor_ratio_product(ratio: Ratio, value: int) -> int:
    return ratio.numerator * value // ratio.denominator


def validation_pair_interval(
    policy: JointLevelPolicy,
    pair_count: int,
) -> tuple[int, int]:
    """Return the exact integer validation interval for a fixed universe."""
    if pair_count <= 0:
        raise ValueError("pair_count must be positive")
    lower = max(
        policy.minimum_validation_pairs,
        _ceil_ratio_product(policy.validation_pair_lower_fraction, pair_count),
    )
    upper = _floor_ratio_product(
        policy.validation_pair_upper_fraction,
        pair_count,
    )
    return lower, upper


def minimum_arithmetically_admissible_pair_count(
    policy: JointLevelPolicy,
    *,
    search_limit: int = 1_000_000,
) -> int:
    """Return the first universe size with a non-empty validation interval."""
    for pair_count in range(1, search_limit + 1):
        lower, upper = validation_pair_interval(policy, pair_count)
        if lower <= upper:
            return pair_count
    raise ValueError("No admissible pair count was found within search_limit")


def _validate_problem(
    problem: JointSplitProblem,
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    dict[str, tuple[str, ...]],
    dict[str, tuple[str, ...]],
]:
    if not problem.pairs:
        raise ValueError("The joint split pair universe is empty")
    pair_ids = [pair.pair_id for pair in problem.pairs]
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError("Pair identifiers are not unique")
    antibodies = tuple(sorted({pair.antibody_id for pair in problem.pairs}))
    antigens = tuple(sorted({pair.antigen_id for pair in problem.pairs}))

    def closed_relation(
        entities: tuple[str, ...],
        raw: Mapping[str, frozenset[str]],
        label: str,
    ) -> dict[str, tuple[str, ...]]:
        universe = set(entities)
        unknown_keys = set(raw) - universe
        if unknown_keys:
            raise ValueError(f"{label} graph contains unknown keys")
        normalized: dict[str, set[str]] = {}
        for entity in entities:
            neighbors = set(raw.get(entity, frozenset()))
            if not neighbors.issubset(universe):
                raise ValueError(f"{label} graph contains unknown neighbors")
            neighbors.add(entity)
            normalized[entity] = neighbors
        for left in entities:
            for right in normalized[left]:
                if left not in normalized[right]:
                    raise ValueError(f"{label} graph is not symmetric")
        return {
            entity: tuple(sorted(neighbors))
            for entity, neighbors in normalized.items()
        }

    antibody_closed = closed_relation(
        antibodies,
        problem.antibody_direct_neighbors,
        "antibody",
    )
    antigen_closed = closed_relation(
        antigens,
        problem.antigen_direct_neighbors,
        "antigen",
    )
    return antibodies, antigens, antibody_closed, antigen_closed


def estimate_model_size_from_counts(
    *,
    pair_count: int,
    antibody_count: int,
    antigen_count: int,
    antibody_undirected_edge_count: int,
    antigen_undirected_edge_count: int,
    outer_fold_count: int = 3,
    inner_fold_count: int = 3,
) -> ModelSizeEstimate:
    """Estimate exact dimensions without materializing entity memberships."""
    values = (
        pair_count,
        antibody_count,
        antigen_count,
        antibody_undirected_edge_count,
        antigen_undirected_edge_count,
    )
    if any(value < 0 for value in values):
        raise ValueError("Model-size counts must be non-negative")
    if pair_count == 0 or antibody_count == 0 or antigen_count == 0:
        raise ValueError("Pair and endpoint counts must be positive")
    if outer_fold_count != 3 or inner_fold_count != 3:
        raise ValueError("The locked nested protocol uses exactly 3x3 folds")
    endpoint_count = antibody_count + antigen_count
    directed_closed = endpoint_count + 2 * (
        antibody_undirected_edge_count + antigen_undirected_edge_count
    )
    level_count = outer_fold_count + outer_fold_count * inner_fold_count
    variables = 3 * (endpoint_count + pair_count) * level_count

    outer_constraints = outer_fold_count * (
        directed_closed + 3 * endpoint_count + 9 * pair_count + 9
    )
    outer_constraints += endpoint_count + 30
    inner_constraints = outer_fold_count * inner_fold_count * (
        directed_closed + 5 * endpoint_count + 11 * pair_count + 9
    )
    inner_constraints += outer_fold_count * endpoint_count
    inner_constraints += 30 * outer_fold_count

    outer_nonzeros = outer_fold_count * (
        3 * directed_closed + 5 * endpoint_count + 30 * pair_count
    )
    outer_nonzeros += outer_fold_count * endpoint_count
    outer_nonzeros += 6 * outer_fold_count * (
        3 * pair_count + endpoint_count
    )
    inner_nonzeros = outer_fold_count * inner_fold_count * (
        4 * directed_closed + 10 * endpoint_count + 41 * pair_count
    )
    inner_nonzeros += outer_fold_count * inner_fold_count * endpoint_count
    inner_nonzeros += 6 * inner_fold_count * outer_fold_count * (
        3 * pair_count + endpoint_count
    )
    return ModelSizeEstimate(
        pair_count=pair_count,
        antibody_count=antibody_count,
        antigen_count=antigen_count,
        antibody_undirected_edge_count=antibody_undirected_edge_count,
        antigen_undirected_edge_count=antigen_undirected_edge_count,
        directed_closed_relation_count=directed_closed,
        binary_variable_count=variables,
        linear_constraint_count=outer_constraints + inner_constraints,
        nonzero_coefficient_count=outer_nonzeros + inner_nonzeros,
    )


def estimate_model_size(problem: JointSplitProblem) -> ModelSizeEstimate:
    """Return exact dimensions for one validated metadata problem."""
    antibodies, antigens, antibody_closed, antigen_closed = _validate_problem(
        problem
    )
    antibody_directed = sum(len(values) - 1 for values in antibody_closed.values())
    antigen_directed = sum(len(values) - 1 for values in antigen_closed.values())
    if antibody_directed % 2 or antigen_directed % 2:
        raise ValueError("A symmetric graph has an odd directed-edge count")
    return estimate_model_size_from_counts(
        pair_count=len(problem.pairs),
        antibody_count=len(antibodies),
        antigen_count=len(antigens),
        antibody_undirected_edge_count=antibody_directed // 2,
        antigen_undirected_edge_count=antigen_directed // 2,
    )


def _add_exact_or(
    builder: _ModelBuilder,
    output: int,
    inputs: Sequence[int],
) -> None:
    for input_variable in inputs:
        builder.add_row(
            ((output, 1), (input_variable, -1)),
            lower_bound=0,
        )
    builder.add_row(
        [(output, 1), *((value, -1) for value in inputs)],
        upper_bound=0,
    )


def _add_exact_and_two(
    builder: _ModelBuilder,
    output: int,
    left: int,
    right: int,
) -> None:
    builder.add_row(((output, 1), (left, -1)), upper_bound=0)
    builder.add_row(((output, 1), (right, -1)), upper_bound=0)
    builder.add_row(
        ((output, 1), (left, -1), (right, -1)),
        lower_bound=-1,
    )


def _add_training_and(
    builder: _ModelBuilder,
    output: int,
    closed_antibody: int,
    closed_antigen: int,
    *,
    universe_pair: int | None = None,
) -> None:
    if universe_pair is not None:
        builder.add_row(((output, 1), (universe_pair, -1)), upper_bound=0)
    builder.add_row(((output, 1), (closed_antibody, 1)), upper_bound=1)
    builder.add_row(((output, 1), (closed_antigen, 1)), upper_bound=1)
    coefficients = [
        (output, 1),
        (closed_antibody, 1),
        (closed_antigen, 1),
    ]
    if universe_pair is None:
        builder.add_row(coefficients, lower_bound=1)
    else:
        coefficients.append((universe_pair, -1))
        builder.add_row(coefficients, lower_bound=0)


def _add_endpoint_presence(
    builder: _ModelBuilder,
    presence: int,
    incident_pair_variables: Sequence[int],
) -> None:
    for pair_variable in incident_pair_variables:
        builder.add_row(
            ((pair_variable, 1), (presence, -1)),
            upper_bound=0,
        )
    builder.add_row(
        [(presence, 1), *((value, -1) for value in incident_pair_variables)],
        upper_bound=0,
    )


def _add_balance_rows(
    builder: _ModelBuilder,
    count_variables: Sequence[Sequence[int]],
    maximum_relative_range: Ratio,
) -> None:
    fold_count = len(count_variables)
    for left_fold, right_fold in combinations(range(fold_count), 2):
        for sign in (1, -1):
            coefficients: list[tuple[int, int]] = []
            for fold_index, variables in enumerate(count_variables):
                base = -maximum_relative_range.numerator
                if fold_index == left_fold:
                    base += (
                        sign
                        * fold_count
                        * maximum_relative_range.denominator
                    )
                elif fold_index == right_fold:
                    base -= (
                        sign
                        * fold_count
                        * maximum_relative_range.denominator
                    )
                coefficients.extend((variable, base) for variable in variables)
            builder.add_row(coefficients, upper_bound=0)


def _add_fixed_level_policy(
    builder: _ModelBuilder,
    *,
    policy: JointLevelPolicy,
    validation_pairs: Sequence[Sequence[int]],
    validation_antibodies: Sequence[Sequence[int]],
    validation_antigens: Sequence[Sequence[int]],
    training_pairs: Sequence[Sequence[int]],
    training_antibodies: Sequence[Sequence[int]],
    training_antigens: Sequence[Sequence[int]],
    quarantine_pairs: Sequence[Sequence[int]],
    pair_count: int,
    antibody_count: int,
    antigen_count: int,
) -> None:
    lower, upper = validation_pair_interval(policy, pair_count)
    if lower > upper:
        raise PolicyArithmeticError(
            "Fixed outer validation interval is infeasible: "
            f"minimum={lower}, maximum={upper}, pair_count={pair_count}"
        )
    for fold_index in range(3):
        builder.add_row(
            ((value, 1) for value in validation_pairs[fold_index]),
            lower_bound=policy.minimum_validation_pairs,
        )
        lower_ratio = policy.validation_pair_lower_fraction
        builder.add_row(
            (
                (value, lower_ratio.denominator)
                for value in validation_pairs[fold_index]
            ),
            lower_bound=lower_ratio.numerator * pair_count,
        )
        upper_ratio = policy.validation_pair_upper_fraction
        builder.add_row(
            (
                (value, upper_ratio.denominator)
                for value in validation_pairs[fold_index]
            ),
            upper_bound=upper_ratio.numerator * pair_count,
        )
        builder.add_row(
            ((value, 1) for value in validation_antibodies[fold_index]),
            lower_bound=policy.minimum_validation_endpoints_per_axis,
        )
        builder.add_row(
            ((value, 1) for value in validation_antigens[fold_index]),
            lower_bound=policy.minimum_validation_endpoints_per_axis,
        )
        training_ratio = policy.minimum_training_pair_fraction
        builder.add_row(
            (
                (value, training_ratio.denominator)
                for value in training_pairs[fold_index]
            ),
            lower_bound=training_ratio.numerator * pair_count,
        )
        endpoint_ratio = policy.minimum_training_endpoint_fraction
        builder.add_row(
            (
                (value, endpoint_ratio.denominator)
                for value in training_antibodies[fold_index]
            ),
            lower_bound=endpoint_ratio.numerator * antibody_count,
        )
        builder.add_row(
            (
                (value, endpoint_ratio.denominator)
                for value in training_antigens[fold_index]
            ),
            lower_bound=endpoint_ratio.numerator * antigen_count,
        )
        quarantine_ratio = policy.maximum_quarantine_pair_fraction
        builder.add_row(
            (
                (value, quarantine_ratio.denominator)
                for value in quarantine_pairs[fold_index]
            ),
            upper_bound=quarantine_ratio.numerator * pair_count,
        )
    _add_balance_rows(
        builder,
        validation_pairs,
        policy.maximum_validation_pair_relative_range,
    )
    for values in (
        validation_antibodies,
        validation_antigens,
        training_pairs,
        quarantine_pairs,
    ):
        _add_balance_rows(
            builder,
            values,
            policy.maximum_other_count_relative_range,
        )


def _add_dynamic_inner_policy(
    builder: _ModelBuilder,
    *,
    policy: JointLevelPolicy,
    outer_training_pairs: Sequence[int],
    outer_training_antibodies: Sequence[int],
    outer_training_antigens: Sequence[int],
    validation_pairs: Sequence[Sequence[int]],
    validation_antibodies: Sequence[Sequence[int]],
    validation_antigens: Sequence[Sequence[int]],
    training_pairs: Sequence[Sequence[int]],
    training_antibodies: Sequence[Sequence[int]],
    training_antigens: Sequence[Sequence[int]],
    quarantine_pairs: Sequence[Sequence[int]],
) -> None:
    for inner_index in range(3):
        builder.add_row(
            ((value, 1) for value in validation_pairs[inner_index]),
            lower_bound=policy.minimum_validation_pairs,
        )
        lower_ratio = policy.validation_pair_lower_fraction
        builder.add_row(
            [
                *(
                    (value, lower_ratio.denominator)
                    for value in validation_pairs[inner_index]
                ),
                *((value, -lower_ratio.numerator) for value in outer_training_pairs),
            ],
            lower_bound=0,
        )
        upper_ratio = policy.validation_pair_upper_fraction
        builder.add_row(
            [
                *(
                    (value, upper_ratio.denominator)
                    for value in validation_pairs[inner_index]
                ),
                *((value, -upper_ratio.numerator) for value in outer_training_pairs),
            ],
            upper_bound=0,
        )
        builder.add_row(
            ((value, 1) for value in validation_antibodies[inner_index]),
            lower_bound=policy.minimum_validation_endpoints_per_axis,
        )
        builder.add_row(
            ((value, 1) for value in validation_antigens[inner_index]),
            lower_bound=policy.minimum_validation_endpoints_per_axis,
        )
        training_ratio = policy.minimum_training_pair_fraction
        builder.add_row(
            [
                *(
                    (value, training_ratio.denominator)
                    for value in training_pairs[inner_index]
                ),
                *((value, -training_ratio.numerator) for value in outer_training_pairs),
            ],
            lower_bound=0,
        )
        endpoint_ratio = policy.minimum_training_endpoint_fraction
        builder.add_row(
            [
                *(
                    (value, endpoint_ratio.denominator)
                    for value in training_antibodies[inner_index]
                ),
                *(
                    (value, -endpoint_ratio.numerator)
                    for value in outer_training_antibodies
                ),
            ],
            lower_bound=0,
        )
        builder.add_row(
            [
                *(
                    (value, endpoint_ratio.denominator)
                    for value in training_antigens[inner_index]
                ),
                *(
                    (value, -endpoint_ratio.numerator)
                    for value in outer_training_antigens
                ),
            ],
            lower_bound=0,
        )
        quarantine_ratio = policy.maximum_quarantine_pair_fraction
        builder.add_row(
            [
                *(
                    (value, quarantine_ratio.denominator)
                    for value in quarantine_pairs[inner_index]
                ),
                *((value, -quarantine_ratio.numerator) for value in outer_training_pairs),
            ],
            upper_bound=0,
        )
    _add_balance_rows(
        builder,
        validation_pairs,
        policy.maximum_validation_pair_relative_range,
    )
    for values in (
        validation_antibodies,
        validation_antigens,
        training_pairs,
        quarantine_pairs,
    ):
        _add_balance_rows(
            builder,
            values,
            policy.maximum_other_count_relative_range,
        )


def _matrix_sha256(
    variable_keys: Sequence[tuple[str, ...]],
    rows: Sequence[LinearRow],
) -> str:
    digest = hashlib.sha256()
    for index, key in enumerate(variable_keys):
        digest.update(
            json.dumps(
                [index, *key],
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("ascii")
        )
        digest.update(b"\n")
    for index, row in enumerate(rows):
        digest.update(
            json.dumps(
                [
                    index,
                    row.lower_bound,
                    row.upper_bound,
                    list(row.coefficients),
                ],
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("ascii")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def build_joint_feasibility_model(
    problem: JointSplitProblem,
    *,
    outer_policy: JointLevelPolicy = LOCKED_OUTER_POLICY,
    inner_policy: JointLevelPolicy = LOCKED_INNER_POLICY,
    resource_limits: ResourceLimits = ResourceLimits(),
) -> JointMILPModel:
    """Build the exact 3+9-level binary feasibility model."""
    antibodies, antigens, antibody_closed, antigen_closed = _validate_problem(
        problem
    )
    pairs = tuple(sorted(problem.pairs, key=lambda value: value.pair_id))
    estimate = estimate_model_size(problem)
    resource_limits.check(estimate)
    outer_lower, outer_upper = validation_pair_interval(
        outer_policy,
        len(pairs),
    )
    if outer_lower > outer_upper:
        raise PolicyArithmeticError(
            "Fixed outer validation interval is infeasible: "
            f"minimum={outer_lower}, maximum={outer_upper}, "
            f"pair_count={len(pairs)}"
        )

    incident_by_antibody = {
        antibody: tuple(
            pair_index
            for pair_index, pair in enumerate(pairs)
            if pair.antibody_id == antibody
        )
        for antibody in antibodies
    }
    incident_by_antigen = {
        antigen: tuple(
            pair_index
            for pair_index, pair in enumerate(pairs)
            if pair.antigen_id == antigen
        )
        for antigen in antigens
    }
    builder = _ModelBuilder()

    def v(*key: object) -> int:
        return builder.variable(*key)

    for outer_index in range(3):
        for antibody in antibodies:
            for role in ("holdout_a", "closed_a", "train_present_a"):
                v("outer", outer_index, role, antibody)
        for antigen in antigens:
            for role in ("holdout_g", "closed_g", "train_present_g"):
                v("outer", outer_index, role, antigen)
        for pair in pairs:
            for role in ("validation_pair", "training_pair", "quarantine_pair"):
                v("outer", outer_index, role, pair.pair_id)
        for inner_index in range(3):
            for antibody in antibodies:
                for role in ("holdout_a", "closed_a", "train_present_a"):
                    v("inner", outer_index, inner_index, role, antibody)
            for antigen in antigens:
                for role in ("holdout_g", "closed_g", "train_present_g"):
                    v("inner", outer_index, inner_index, role, antigen)
            for pair in pairs:
                for role in (
                    "validation_pair",
                    "training_pair",
                    "quarantine_pair",
                ):
                    v("inner", outer_index, inner_index, role, pair.pair_id)

    outer_validation_pairs: list[list[int]] = []
    outer_training_pairs: list[list[int]] = []
    outer_quarantine_pairs: list[list[int]] = []
    outer_holdout_antibodies: list[list[int]] = []
    outer_holdout_antigens: list[list[int]] = []
    outer_training_antibodies: list[list[int]] = []
    outer_training_antigens: list[list[int]] = []

    for outer_index in range(3):
        for antibody in antibodies:
            closed = v("outer", outer_index, "closed_a", antibody)
            selected = [
                v("outer", outer_index, "holdout_a", source)
                for source in antibody_closed[antibody]
            ]
            _add_exact_or(builder, closed, selected)
        for antigen in antigens:
            closed = v("outer", outer_index, "closed_g", antigen)
            selected = [
                v("outer", outer_index, "holdout_g", source)
                for source in antigen_closed[antigen]
            ]
            _add_exact_or(builder, closed, selected)

        for pair in pairs:
            validation = v(
                "outer",
                outer_index,
                "validation_pair",
                pair.pair_id,
            )
            training = v("outer", outer_index, "training_pair", pair.pair_id)
            quarantine = v(
                "outer",
                outer_index,
                "quarantine_pair",
                pair.pair_id,
            )
            _add_exact_and_two(
                builder,
                validation,
                v("outer", outer_index, "holdout_a", pair.antibody_id),
                v("outer", outer_index, "holdout_g", pair.antigen_id),
            )
            _add_training_and(
                builder,
                training,
                v("outer", outer_index, "closed_a", pair.antibody_id),
                v("outer", outer_index, "closed_g", pair.antigen_id),
            )
            builder.add_row(
                ((validation, 1), (training, 1), (quarantine, 1)),
                lower_bound=1,
                upper_bound=1,
            )

        for antibody in antibodies:
            incident_training = [
                v("outer", outer_index, "training_pair", pairs[index].pair_id)
                for index in incident_by_antibody[antibody]
            ]
            _add_endpoint_presence(
                builder,
                v("outer", outer_index, "train_present_a", antibody),
                incident_training,
            )
            incident_validation = [
                v("outer", outer_index, "validation_pair", pairs[index].pair_id)
                for index in incident_by_antibody[antibody]
            ]
            builder.add_row(
                [
                    (v("outer", outer_index, "holdout_a", antibody), 1),
                    *((value, -1) for value in incident_validation),
                ],
                upper_bound=0,
            )
        for antigen in antigens:
            incident_training = [
                v("outer", outer_index, "training_pair", pairs[index].pair_id)
                for index in incident_by_antigen[antigen]
            ]
            _add_endpoint_presence(
                builder,
                v("outer", outer_index, "train_present_g", antigen),
                incident_training,
            )
            incident_validation = [
                v("outer", outer_index, "validation_pair", pairs[index].pair_id)
                for index in incident_by_antigen[antigen]
            ]
            builder.add_row(
                [
                    (v("outer", outer_index, "holdout_g", antigen), 1),
                    *((value, -1) for value in incident_validation),
                ],
                upper_bound=0,
            )

        outer_validation_pairs.append(
            [
                v("outer", outer_index, "validation_pair", pair.pair_id)
                for pair in pairs
            ]
        )
        outer_training_pairs.append(
            [
                v("outer", outer_index, "training_pair", pair.pair_id)
                for pair in pairs
            ]
        )
        outer_quarantine_pairs.append(
            [
                v("outer", outer_index, "quarantine_pair", pair.pair_id)
                for pair in pairs
            ]
        )
        outer_holdout_antibodies.append(
            [v("outer", outer_index, "holdout_a", value) for value in antibodies]
        )
        outer_holdout_antigens.append(
            [v("outer", outer_index, "holdout_g", value) for value in antigens]
        )
        outer_training_antibodies.append(
            [
                v("outer", outer_index, "train_present_a", value)
                for value in antibodies
            ]
        )
        outer_training_antigens.append(
            [
                v("outer", outer_index, "train_present_g", value)
                for value in antigens
            ]
        )

    for antibody in antibodies:
        builder.add_row(
            (
                (v("outer", outer_index, "holdout_a", antibody), 1)
                for outer_index in range(3)
            ),
            upper_bound=1,
        )
    for antigen in antigens:
        builder.add_row(
            (
                (v("outer", outer_index, "holdout_g", antigen), 1)
                for outer_index in range(3)
            ),
            upper_bound=1,
        )

    _add_fixed_level_policy(
        builder,
        policy=outer_policy,
        validation_pairs=outer_validation_pairs,
        validation_antibodies=outer_holdout_antibodies,
        validation_antigens=outer_holdout_antigens,
        training_pairs=outer_training_pairs,
        training_antibodies=outer_training_antibodies,
        training_antigens=outer_training_antigens,
        quarantine_pairs=outer_quarantine_pairs,
        pair_count=len(pairs),
        antibody_count=len(antibodies),
        antigen_count=len(antigens),
    )

    for outer_index in range(3):
        inner_validation_pairs: list[list[int]] = []
        inner_training_pairs: list[list[int]] = []
        inner_quarantine_pairs: list[list[int]] = []
        inner_holdout_antibodies: list[list[int]] = []
        inner_holdout_antigens: list[list[int]] = []
        inner_training_antibodies: list[list[int]] = []
        inner_training_antigens: list[list[int]] = []
        for inner_index in range(3):
            for antibody in antibodies:
                holdout = v(
                    "inner",
                    outer_index,
                    inner_index,
                    "holdout_a",
                    antibody,
                )
                outer_present = v(
                    "outer",
                    outer_index,
                    "train_present_a",
                    antibody,
                )
                builder.add_row(
                    ((holdout, 1), (outer_present, -1)),
                    upper_bound=0,
                )
                closed = v(
                    "inner",
                    outer_index,
                    inner_index,
                    "closed_a",
                    antibody,
                )
                builder.add_row(
                    ((closed, 1), (outer_present, -1)),
                    upper_bound=0,
                )
                selected = [
                    v(
                        "inner",
                        outer_index,
                        inner_index,
                        "holdout_a",
                        source,
                    )
                    for source in antibody_closed[antibody]
                ]
                builder.add_row(
                    [(closed, 1), *((value, -1) for value in selected)],
                    upper_bound=0,
                )
                for selected_value in selected:
                    builder.add_row(
                        (
                            (closed, 1),
                            (outer_present, -1),
                            (selected_value, -1),
                        ),
                        lower_bound=-1,
                    )
            for antigen in antigens:
                holdout = v(
                    "inner",
                    outer_index,
                    inner_index,
                    "holdout_g",
                    antigen,
                )
                outer_present = v(
                    "outer",
                    outer_index,
                    "train_present_g",
                    antigen,
                )
                builder.add_row(
                    ((holdout, 1), (outer_present, -1)),
                    upper_bound=0,
                )
                closed = v(
                    "inner",
                    outer_index,
                    inner_index,
                    "closed_g",
                    antigen,
                )
                builder.add_row(
                    ((closed, 1), (outer_present, -1)),
                    upper_bound=0,
                )
                selected = [
                    v(
                        "inner",
                        outer_index,
                        inner_index,
                        "holdout_g",
                        source,
                    )
                    for source in antigen_closed[antigen]
                ]
                builder.add_row(
                    [(closed, 1), *((value, -1) for value in selected)],
                    upper_bound=0,
                )
                for selected_value in selected:
                    builder.add_row(
                        (
                            (closed, 1),
                            (outer_present, -1),
                            (selected_value, -1),
                        ),
                        lower_bound=-1,
                    )

            for pair in pairs:
                outer_training = v(
                    "outer",
                    outer_index,
                    "training_pair",
                    pair.pair_id,
                )
                validation = v(
                    "inner",
                    outer_index,
                    inner_index,
                    "validation_pair",
                    pair.pair_id,
                )
                holdout_a = v(
                    "inner",
                    outer_index,
                    inner_index,
                    "holdout_a",
                    pair.antibody_id,
                )
                holdout_g = v(
                    "inner",
                    outer_index,
                    inner_index,
                    "holdout_g",
                    pair.antigen_id,
                )
                for prerequisite in (outer_training, holdout_a, holdout_g):
                    builder.add_row(
                        ((validation, 1), (prerequisite, -1)),
                        upper_bound=0,
                    )
                builder.add_row(
                    (
                        (validation, 1),
                        (outer_training, -1),
                        (holdout_a, -1),
                        (holdout_g, -1),
                    ),
                    lower_bound=-2,
                )
                training = v(
                    "inner",
                    outer_index,
                    inner_index,
                    "training_pair",
                    pair.pair_id,
                )
                _add_training_and(
                    builder,
                    training,
                    v(
                        "inner",
                        outer_index,
                        inner_index,
                        "closed_a",
                        pair.antibody_id,
                    ),
                    v(
                        "inner",
                        outer_index,
                        inner_index,
                        "closed_g",
                        pair.antigen_id,
                    ),
                    universe_pair=outer_training,
                )
                quarantine = v(
                    "inner",
                    outer_index,
                    inner_index,
                    "quarantine_pair",
                    pair.pair_id,
                )
                builder.add_row(
                    (
                        (validation, 1),
                        (training, 1),
                        (quarantine, 1),
                        (outer_training, -1),
                    ),
                    lower_bound=0,
                    upper_bound=0,
                )

            for antibody in antibodies:
                incident_training = [
                    v(
                        "inner",
                        outer_index,
                        inner_index,
                        "training_pair",
                        pairs[index].pair_id,
                    )
                    for index in incident_by_antibody[antibody]
                ]
                _add_endpoint_presence(
                    builder,
                    v(
                        "inner",
                        outer_index,
                        inner_index,
                        "train_present_a",
                        antibody,
                    ),
                    incident_training,
                )
                incident_validation = [
                    v(
                        "inner",
                        outer_index,
                        inner_index,
                        "validation_pair",
                        pairs[index].pair_id,
                    )
                    for index in incident_by_antibody[antibody]
                ]
                builder.add_row(
                    [
                        (
                            v(
                                "inner",
                                outer_index,
                                inner_index,
                                "holdout_a",
                                antibody,
                            ),
                            1,
                        ),
                        *((value, -1) for value in incident_validation),
                    ],
                    upper_bound=0,
                )
            for antigen in antigens:
                incident_training = [
                    v(
                        "inner",
                        outer_index,
                        inner_index,
                        "training_pair",
                        pairs[index].pair_id,
                    )
                    for index in incident_by_antigen[antigen]
                ]
                _add_endpoint_presence(
                    builder,
                    v(
                        "inner",
                        outer_index,
                        inner_index,
                        "train_present_g",
                        antigen,
                    ),
                    incident_training,
                )
                incident_validation = [
                    v(
                        "inner",
                        outer_index,
                        inner_index,
                        "validation_pair",
                        pairs[index].pair_id,
                    )
                    for index in incident_by_antigen[antigen]
                ]
                builder.add_row(
                    [
                        (
                            v(
                                "inner",
                                outer_index,
                                inner_index,
                                "holdout_g",
                                antigen,
                            ),
                            1,
                        ),
                        *((value, -1) for value in incident_validation),
                    ],
                    upper_bound=0,
                )

            inner_validation_pairs.append(
                [
                    v(
                        "inner",
                        outer_index,
                        inner_index,
                        "validation_pair",
                        pair.pair_id,
                    )
                    for pair in pairs
                ]
            )
            inner_training_pairs.append(
                [
                    v(
                        "inner",
                        outer_index,
                        inner_index,
                        "training_pair",
                        pair.pair_id,
                    )
                    for pair in pairs
                ]
            )
            inner_quarantine_pairs.append(
                [
                    v(
                        "inner",
                        outer_index,
                        inner_index,
                        "quarantine_pair",
                        pair.pair_id,
                    )
                    for pair in pairs
                ]
            )
            inner_holdout_antibodies.append(
                [
                    v(
                        "inner",
                        outer_index,
                        inner_index,
                        "holdout_a",
                        value,
                    )
                    for value in antibodies
                ]
            )
            inner_holdout_antigens.append(
                [
                    v(
                        "inner",
                        outer_index,
                        inner_index,
                        "holdout_g",
                        value,
                    )
                    for value in antigens
                ]
            )
            inner_training_antibodies.append(
                [
                    v(
                        "inner",
                        outer_index,
                        inner_index,
                        "train_present_a",
                        value,
                    )
                    for value in antibodies
                ]
            )
            inner_training_antigens.append(
                [
                    v(
                        "inner",
                        outer_index,
                        inner_index,
                        "train_present_g",
                        value,
                    )
                    for value in antigens
                ]
            )

        for antibody in antibodies:
            builder.add_row(
                (
                    (
                        v(
                            "inner",
                            outer_index,
                            inner_index,
                            "holdout_a",
                            antibody,
                        ),
                        1,
                    )
                    for inner_index in range(3)
                ),
                upper_bound=1,
            )
        for antigen in antigens:
            builder.add_row(
                (
                    (
                        v(
                            "inner",
                            outer_index,
                            inner_index,
                            "holdout_g",
                            antigen,
                        ),
                        1,
                    )
                    for inner_index in range(3)
                ),
                upper_bound=1,
            )

        _add_dynamic_inner_policy(
            builder,
            policy=inner_policy,
            outer_training_pairs=outer_training_pairs[outer_index],
            outer_training_antibodies=outer_training_antibodies[outer_index],
            outer_training_antigens=outer_training_antigens[outer_index],
            validation_pairs=inner_validation_pairs,
            validation_antibodies=inner_holdout_antibodies,
            validation_antigens=inner_holdout_antigens,
            training_pairs=inner_training_pairs,
            training_antibodies=inner_training_antibodies,
            training_antigens=inner_training_antigens,
            quarantine_pairs=inner_quarantine_pairs,
        )

    if len(builder.variable_keys) != estimate.binary_variable_count:
        raise RuntimeError(
            "Variable accounting differs from the pre-allocation estimate"
        )
    if len(builder.rows) != estimate.linear_constraint_count:
        raise RuntimeError(
            "Constraint accounting differs from the pre-allocation estimate: "
            f"observed={len(builder.rows)}, expected={estimate.linear_constraint_count}"
        )
    observed_nonzeros = sum(len(row.coefficients) for row in builder.rows)
    if observed_nonzeros > estimate.nonzero_coefficient_count:
        raise RuntimeError(
            "Nonzero accounting exceeds the pre-allocation estimate: "
            f"observed={observed_nonzeros}, "
            f"expected={estimate.nonzero_coefficient_count}"
        )
    realized_estimate = replace(
        estimate,
        nonzero_coefficient_count=observed_nonzeros,
    )
    variable_keys = tuple(builder.variable_keys)
    rows = tuple(builder.rows)
    return JointMILPModel(
        variable_keys=variable_keys,
        rows=rows,
        estimate=realized_estimate,
        matrix_sha256=_matrix_sha256(variable_keys, rows),
    )


def _audit_integer_rows(
    model: JointMILPModel,
    assignment: Sequence[int],
) -> bool:
    if len(assignment) != len(model.variable_keys):
        return False
    if any(value not in (0, 1) for value in assignment):
        return False
    for row in model.rows:
        observed = sum(
            coefficient * assignment[column]
            for column, coefficient in row.coefficients
        )
        if row.lower_bound is not None and observed < row.lower_bound:
            return False
        if row.upper_bound is not None and observed > row.upper_bound:
            return False
    return True


def solve_joint_feasibility(
    model: JointMILPModel,
    *,
    time_limit_seconds: float | None = None,
) -> JointSolveResult:
    """Solve with SciPy/HiGHS and reject every non-optimal termination."""
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix

    row_indices: list[int] = []
    column_indices: list[int] = []
    values: list[int] = []
    lower: list[float] = []
    upper: list[float] = []
    for row_index, row in enumerate(model.rows):
        lower.append(
            -np.inf if row.lower_bound is None else float(row.lower_bound)
        )
        upper.append(
            np.inf if row.upper_bound is None else float(row.upper_bound)
        )
        for column, coefficient in row.coefficients:
            row_indices.append(row_index)
            column_indices.append(column)
            values.append(coefficient)
    matrix = coo_matrix(
        (
            np.asarray(values, dtype=np.float64),
            (
                np.asarray(row_indices, dtype=np.int32),
                np.asarray(column_indices, dtype=np.int32),
            ),
        ),
        shape=(len(model.rows), len(model.variable_keys)),
        dtype=np.float64,
    ).tocsc()
    options: dict[str, float] = {"mip_rel_gap": 0.0}
    if time_limit_seconds is not None:
        if not math.isfinite(time_limit_seconds) or time_limit_seconds <= 0:
            raise ValueError("time_limit_seconds must be positive and finite")
        options["time_limit"] = float(time_limit_seconds)
    result = milp(
        c=np.zeros(len(model.variable_keys), dtype=np.float64),
        integrality=np.ones(len(model.variable_keys), dtype=np.uint8),
        bounds=Bounds(
            np.zeros(len(model.variable_keys), dtype=np.float64),
            np.ones(len(model.variable_keys), dtype=np.float64),
        ),
        constraints=LinearConstraint(
            matrix,
            np.asarray(lower, dtype=np.float64),
            np.asarray(upper, dtype=np.float64),
        ),
        options=options,
    )
    reported_gap_raw = getattr(result, "mip_gap", None)
    reported_gap = (
        None if reported_gap_raw is None else float(reported_gap_raw)
    )
    status = int(result.status)
    reported_success = bool(result.success)
    termination_by_status = {
        0: "optimal",
        1: "limit_reached_without_optimality",
        2: "certified_infeasible",
        3: "reported_unbounded",
        4: "solver_error",
    }
    solver_termination = termination_by_status.get(status, "unknown_solver_status")
    if status == 0 and not reported_success:
        solver_termination = "inconsistent_optimal_status"
    solver_optimal = bool(reported_success and status == 0)
    gap_is_finite_zero = bool(
        reported_gap is not None
        and math.isfinite(reported_gap)
        and reported_gap == 0.0
    )
    exact_audit = False
    assignment_map: dict[tuple[str, ...], int] | None = None
    maximum_integrality_error: float | None = None
    integrality_tolerance = 1e-7
    if solver_optimal and result.x is not None:
        rounded = np.rint(result.x).astype(np.int8)
        maximum_integrality_error = float(np.max(np.abs(result.x - rounded)))
        if (
            math.isfinite(maximum_integrality_error)
            and maximum_integrality_error <= integrality_tolerance
        ):
            integer_values = [int(value) for value in rounded]
            exact_audit = _audit_integer_rows(model, integer_values)
            if exact_audit:
                assignment_map = {
                    key: integer_values[index]
                    for index, key in enumerate(model.variable_keys)
                }
    feasible = solver_optimal and exact_audit and gap_is_finite_zero
    if not feasible:
        assignment_map = None
    return JointSolveResult(
        solver="scipy.optimize.milp_HiGHS",
        solver_status=status,
        solver_termination=solver_termination,
        solver_message=str(result.message),
        solver_reported_success=reported_success,
        solver_solution_present=result.x is not None,
        solver_certified_optimal=solver_optimal,
        solver_reported_mip_gap=reported_gap,
        solver_reported_mip_gap_is_finite_zero=gap_is_finite_zero,
        maximum_integrality_error=maximum_integrality_error,
        accepted_integrality_tolerance=integrality_tolerance,
        exact_integer_row_audit_passed=exact_audit,
        feasible_for_unscored_split_construction=feasible,
        authorized_for_training=False,
        authorized_for_model_selection=False,
        matrix_sha256=model.matrix_sha256,
        binary_variable_count=model.estimate.binary_variable_count,
        linear_constraint_count=model.estimate.linear_constraint_count,
        nonzero_coefficient_count=model.estimate.nonzero_coefficient_count,
        assignment=assignment_map,
    )
