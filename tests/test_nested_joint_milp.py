"""Synthetic tests for the metadata-only exact joint nested MILP."""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from interaction_first.nested_joint_milp import (
    CanonicalPair,
    JointLevelPolicy,
    JointSplitProblem,
    LOCKED_JOINT_POLICY_SHA256,
    PolicyArithmeticError,
    Ratio,
    ResourceLimitError,
    ResourceLimits,
    build_joint_feasibility_model,
    estimate_model_size,
    estimate_model_size_from_counts,
    minimum_arithmetically_admissible_pair_count,
    solve_joint_feasibility,
    validation_pair_interval,
)


def _policy(
    *,
    minimum_validation_pairs: int,
    validation_upper: Ratio,
) -> JointLevelPolicy:
    return JointLevelPolicy(
        minimum_validation_pairs=minimum_validation_pairs,
        minimum_validation_endpoints_per_axis=1,
        validation_pair_lower_fraction=Ratio(1, 1000),
        validation_pair_upper_fraction=validation_upper,
        minimum_training_pair_fraction=Ratio(1, 2),
        minimum_training_endpoint_fraction=Ratio(1, 2),
        maximum_quarantine_pair_fraction=Ratio(1, 1000),
        maximum_validation_pair_relative_range=Ratio(1, 1000),
        maximum_other_count_relative_range=Ratio(1, 1000),
    )


def _independent_problem(pair_count: int = 9) -> JointSplitProblem:
    pairs = tuple(
        CanonicalPair(f"p{index}", f"a{index}", f"g{index}")
        for index in range(pair_count)
    )
    return JointSplitProblem(
        pairs=pairs,
        antibody_direct_neighbors={
            f"a{index}": frozenset() for index in range(pair_count)
        },
        antigen_direct_neighbors={
            f"g{index}": frozenset() for index in range(pair_count)
        },
    )


def _selected(
    assignment: dict[tuple[str, ...], int],
    prefix: tuple[str, ...],
) -> set[str]:
    return {
        key[-1]
        for key, value in assignment.items()
        if value == 1 and key[:-1] == prefix
    }


def test_locked_inner_arithmetic_threshold_is_889() -> None:
    """The 847-pair child is impossible without changing the locked policy."""
    from interaction_first.nested_joint_milp import LOCKED_INNER_POLICY

    assert validation_pair_interval(LOCKED_INNER_POLICY, 847) == (80, 76)
    assert validation_pair_interval(LOCKED_INNER_POLICY, 889) == (80, 80)
    assert minimum_arithmetically_admissible_pair_count(
        LOCKED_INNER_POLICY
    ) == 889


def test_locked_exact_joint_policy_has_a_stable_external_anchor() -> None:
    """The exact 3x3 policy digest must remain reviewable out of band."""
    assert LOCKED_JOINT_POLICY_SHA256 == (
        "7940bc19f2b1bc7caf687851399d4219d114933eba5caaf9e07a0d41094805e1"
    )


def test_authority_metadata_estimate_is_large_and_prototype_is_fail_closed() -> None:
    """Known train-only counts must trigger review before matrix allocation."""
    estimate = estimate_model_size_from_counts(
        pair_count=1830,
        antibody_count=1383,
        antigen_count=1172,
        antibody_undirected_edge_count=77332,
        antigen_undirected_edge_count=16273,
    )
    assert estimate.binary_variable_count == 157_860
    assert estimate.linear_constraint_count == 2_656_178
    assert estimate.nonzero_coefficient_count == 10_257_570
    with pytest.raises(ResourceLimitError):
        ResourceLimits().check(estimate)


def test_synthetic_joint_outer_and_all_nine_inner_folds_are_feasible() -> None:
    """One solve must satisfy all outer and dependent inner assignments."""
    problem = _independent_problem()
    outer_policy = _policy(
        minimum_validation_pairs=1,
        validation_upper=Ratio(1, 9),
    )
    inner_policy = _policy(
        minimum_validation_pairs=1,
        validation_upper=Ratio(1, 8),
    )
    model = build_joint_feasibility_model(
        problem,
        outer_policy=outer_policy,
        inner_policy=inner_policy,
        resource_limits=ResourceLimits(
            maximum_binary_variables=10_000,
            maximum_linear_constraints=20_000,
            maximum_nonzero_coefficients=100_000,
        ),
    )
    assert model.estimate == estimate_model_size(problem)
    result = solve_joint_feasibility(model)
    assert result.solver_certified_optimal
    assert result.solver_reported_mip_gap == 0.0
    assert result.exact_integer_row_audit_passed
    assert result.feasible_for_unscored_split_construction
    assert not result.authorized_for_training
    assert not result.authorized_for_model_selection
    assert result.assignment is not None
    assignment = dict(result.assignment)

    outer_validation_memberships: list[set[str]] = []
    for outer_index in range(3):
        validation = _selected(
            assignment,
            ("outer", str(outer_index), "validation_pair"),
        )
        training = _selected(
            assignment,
            ("outer", str(outer_index), "training_pair"),
        )
        quarantine = _selected(
            assignment,
            ("outer", str(outer_index), "quarantine_pair"),
        )
        assert len(validation) == 1
        assert len(training) == 8
        assert not quarantine
        assert validation | training == {f"p{index}" for index in range(9)}
        outer_validation_memberships.append(validation)

        inner_validation_memberships: list[set[str]] = []
        for inner_index in range(3):
            inner_validation = _selected(
                assignment,
                (
                    "inner",
                    str(outer_index),
                    str(inner_index),
                    "validation_pair",
                ),
            )
            inner_training = _selected(
                assignment,
                (
                    "inner",
                    str(outer_index),
                    str(inner_index),
                    "training_pair",
                ),
            )
            inner_quarantine = _selected(
                assignment,
                (
                    "inner",
                    str(outer_index),
                    str(inner_index),
                    "quarantine_pair",
                ),
            )
            assert len(inner_validation) == 1
            assert len(inner_training) == 7
            assert not inner_quarantine
            assert inner_validation | inner_training == training
            inner_validation_memberships.append(inner_validation)
        assert len(set().union(*inner_validation_memberships)) == 3
    assert len(set().union(*outer_validation_memberships)) == 3


def test_fixed_outer_arithmetic_failure_precedes_solver_allocation() -> None:
    """An empty fixed validation interval must fail before SciPy is called."""
    impossible_outer = _policy(
        minimum_validation_pairs=2,
        validation_upper=Ratio(1, 9),
    )
    with pytest.raises(PolicyArithmeticError, match="minimum=2, maximum=1"):
        build_joint_feasibility_model(
            _independent_problem(),
            outer_policy=impossible_outer,
            inner_policy=_policy(
                minimum_validation_pairs=1,
                validation_upper=Ratio(1, 8),
            ),
            resource_limits=ResourceLimits(
                maximum_binary_variables=10_000,
                maximum_linear_constraints=20_000,
                maximum_nonzero_coefficients=100_000,
            ),
        )


def test_outer_feasibility_cannot_hide_an_infeasible_inner_policy() -> None:
    """The joint solve must fail when every outer child has an empty interval."""
    model = build_joint_feasibility_model(
        _independent_problem(),
        outer_policy=_policy(
            minimum_validation_pairs=1,
            validation_upper=Ratio(1, 9),
        ),
        inner_policy=_policy(
            minimum_validation_pairs=2,
            validation_upper=Ratio(1, 8),
        ),
        resource_limits=ResourceLimits(
            maximum_binary_variables=10_000,
            maximum_linear_constraints=20_000,
            maximum_nonzero_coefficients=100_000,
        ),
    )
    result = solve_joint_feasibility(model)
    assert result.solver_status == 2
    assert not result.solver_certified_optimal
    assert not result.exact_integer_row_audit_passed
    assert not result.feasible_for_unscored_split_construction
    assert not result.authorized_for_training
    assert not result.authorized_for_model_selection
    assert result.assignment is None
    assert result.solver_termination == "certified_infeasible"


def test_limit_termination_is_not_mislabeled_as_infeasible() -> None:
    """A solver limit must remain distinct from a proof of infeasibility."""
    model = build_joint_feasibility_model(
        _independent_problem(),
        outer_policy=_policy(
            minimum_validation_pairs=1,
            validation_upper=Ratio(1, 9),
        ),
        inner_policy=_policy(
            minimum_validation_pairs=1,
            validation_upper=Ratio(1, 8),
        ),
        resource_limits=ResourceLimits(
            maximum_binary_variables=10_000,
            maximum_linear_constraints=20_000,
            maximum_nonzero_coefficients=100_000,
        ),
    )
    with mock.patch(
        "scipy.optimize.milp",
        return_value=SimpleNamespace(
            success=False,
            status=1,
            message="Time limit reached",
            mip_gap=0.25,
            x=None,
        ),
    ):
        result = solve_joint_feasibility(model, time_limit_seconds=1.0)
    assert result.solver_termination == "limit_reached_without_optimality"
    assert not result.solver_certified_optimal
    assert not result.feasible_for_unscored_split_construction
    assert result.assignment is None


def test_nonzero_gap_discards_an_integer_feasible_incumbent() -> None:
    """A valid incumbent cannot be published without an exact zero-gap receipt."""
    model = build_joint_feasibility_model(
        _independent_problem(),
        outer_policy=_policy(
            minimum_validation_pairs=1,
            validation_upper=Ratio(1, 9),
        ),
        inner_policy=_policy(
            minimum_validation_pairs=1,
            validation_upper=Ratio(1, 8),
        ),
        resource_limits=ResourceLimits(
            maximum_binary_variables=10_000,
            maximum_linear_constraints=20_000,
            maximum_nonzero_coefficients=100_000,
        ),
    )
    baseline = solve_joint_feasibility(model)
    assert baseline.assignment is not None
    incumbent = np.asarray(
        [baseline.assignment[key] for key in model.variable_keys],
        dtype=np.float64,
    )
    with mock.patch(
        "scipy.optimize.milp",
        return_value=SimpleNamespace(
            success=True,
            status=0,
            message="Optimal",
            mip_gap=0.01,
            x=incumbent,
        ),
    ):
        result = solve_joint_feasibility(model)
    assert result.solver_termination == "optimal"
    assert result.exact_integer_row_audit_passed
    assert not result.solver_reported_mip_gap_is_finite_zero
    assert not result.feasible_for_unscored_split_construction
    assert result.assignment is None


@pytest.mark.parametrize("invalid", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_time_limits_are_rejected_before_solver_call(invalid: float) -> None:
    """Only positive finite solver limits are valid."""
    model = build_joint_feasibility_model(
        _independent_problem(),
        outer_policy=_policy(
            minimum_validation_pairs=1,
            validation_upper=Ratio(1, 9),
        ),
        inner_policy=_policy(
            minimum_validation_pairs=1,
            validation_upper=Ratio(1, 8),
        ),
        resource_limits=ResourceLimits(
            maximum_binary_variables=10_000,
            maximum_linear_constraints=20_000,
            maximum_nonzero_coefficients=100_000,
        ),
    )
    with (
        mock.patch(
            "scipy.optimize.milp",
            side_effect=AssertionError("solver must not be called"),
        ),
        pytest.raises(ValueError, match="positive and finite"),
    ):
        solve_joint_feasibility(model, time_limit_seconds=invalid)
