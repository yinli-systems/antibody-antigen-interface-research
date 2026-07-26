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
    LOCKED_INNER_POLICY,
    LOCKED_JOINT_POLICY_SHA256,
    LOCKED_OUTER_POLICY,
    PolicyArithmeticError,
    Ratio,
    ResourceLimitError,
    ResourceLimits,
    build_joint_feasibility_model,
    estimate_model_size,
    estimate_model_size_from_counts,
    joint_policy_payload,
    minimum_arithmetically_admissible_pair_count,
    solve_joint_feasibility,
    validation_pair_interval,
)


def _policy(
    *,
    minimum_validation_pairs: int,
    validation_upper: Ratio,
    minimum_validation_components_per_axis: int = 1,
) -> JointLevelPolicy:
    return JointLevelPolicy(
        minimum_validation_pairs=minimum_validation_pairs,
        minimum_validation_endpoints_per_axis=1,
        minimum_validation_components_per_axis=(
            minimum_validation_components_per_axis
        ),
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


def _path_component_problem(*, reverse_input: bool = False) -> JointSplitProblem:
    """Return nine pairs with one three-node path on each homology axis."""
    pairs = [
        CanonicalPair(f"p{index}", f"a{index}", f"g{index}")
        for index in range(9)
    ]
    antibody_neighbors = {
        "a0": frozenset({"a1"}),
        "a1": frozenset({"a0", "a2"}),
        "a2": frozenset({"a1"}),
        **{f"a{index}": frozenset() for index in range(3, 9)},
    }
    antigen_neighbors = {
        "g0": frozenset({"g1"}),
        "g1": frozenset({"g0", "g2"}),
        "g2": frozenset({"g1"}),
        **{f"g{index}": frozenset() for index in range(3, 9)},
    }
    if reverse_input:
        pairs.reverse()
        antibody_neighbors = dict(reversed(tuple(antibody_neighbors.items())))
        antigen_neighbors = dict(reversed(tuple(antigen_neighbors.items())))
    return JointSplitProblem(
        pairs=tuple(pairs),
        antibody_direct_neighbors=antibody_neighbors,
        antigen_direct_neighbors=antigen_neighbors,
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
    assert LOCKED_OUTER_POLICY.minimum_validation_components_per_axis == 25
    assert LOCKED_INNER_POLICY.minimum_validation_components_per_axis == 20
    assert joint_policy_payload(
        LOCKED_OUTER_POLICY,
        LOCKED_INNER_POLICY,
    )["schema_version"] == "joint_nested_split_exact_policy_v2"
    assert LOCKED_JOINT_POLICY_SHA256 == (
        "5ddf8b4efef6ae7aed29fd529e613ee12f3b4d0caeddc23d1a320f73bca344fd"
    )


def test_authority_metadata_estimate_is_large_and_prototype_is_fail_closed() -> None:
    """Known train-only counts must trigger review before matrix allocation."""
    # These are the largest component counts compatible with the edge totals.
    estimate = estimate_model_size_from_counts(
        pair_count=1830,
        antibody_count=1383,
        antigen_count=1172,
        antibody_component_count=990,
        antigen_component_count=992,
        antibody_undirected_edge_count=77332,
        antigen_undirected_edge_count=16273,
    )
    assert estimate.binary_variable_count == 181_644
    assert estimate.linear_constraint_count == 2_710_646
    assert estimate.nonzero_coefficient_count == 10_397_118
    with pytest.raises(ResourceLimitError):
        ResourceLimits().check(estimate)


def test_component_size_accounting_is_exact_for_independent_graphs() -> None:
    """Singleton components must be included in every nested model level."""
    estimate = estimate_model_size(_independent_problem())
    assert estimate.antibody_component_count == 9
    assert estimate.antigen_component_count == 9
    assert estimate.binary_variable_count == 1_188
    assert estimate.linear_constraint_count == 3_078
    assert estimate.nonzero_coefficient_count == 11_367


def test_full_graph_components_are_transitive_canonical_and_order_stable() -> None:
    """Input order cannot alter component representatives or matrix bytes."""
    limits = ResourceLimits(
        maximum_binary_variables=10_000,
        maximum_linear_constraints=20_000,
        maximum_nonzero_coefficients=100_000,
    )
    policy = _policy(
        minimum_validation_pairs=1,
        validation_upper=Ratio(1, 9),
    )
    first = build_joint_feasibility_model(
        _path_component_problem(),
        outer_policy=policy,
        inner_policy=_policy(
            minimum_validation_pairs=1,
            validation_upper=Ratio(1, 8),
        ),
        resource_limits=limits,
    )
    reordered = build_joint_feasibility_model(
        _path_component_problem(reverse_input=True),
        outer_policy=policy,
        inner_policy=_policy(
            minimum_validation_pairs=1,
            validation_upper=Ratio(1, 8),
        ),
        resource_limits=limits,
    )
    expected_representatives = {
        "a0",
        "a3",
        "a4",
        "a5",
        "a6",
        "a7",
        "a8",
    }
    observed_representatives = {
        key[-1]
        for key in first.variable_keys
        if key[:3] == ("outer", "0", "validation_component_a")
    }
    assert observed_representatives == expected_representatives
    assert first.estimate.antibody_component_count == 7
    assert first.estimate.antigen_component_count == 7
    assert first.variable_keys == reordered.variable_keys
    assert first.matrix_sha256 == reordered.matrix_sha256


def test_component_presence_is_exactly_tied_to_effective_endpoints() -> None:
    """Each component binary must be the exact OR of its holdout endpoints."""
    model = build_joint_feasibility_model(
        _path_component_problem(),
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
    index = {key: value for value, key in enumerate(model.variable_keys)}

    def assert_exact_component_or(
        prefix: tuple[str, ...],
        *,
        component_role: str,
        holdout_role: str,
        entity_prefix: str,
    ) -> None:
        presence = index[
            (*prefix, component_role, f"{entity_prefix}0")
        ]
        endpoints = [
            index[(*prefix, holdout_role, f"{entity_prefix}{endpoint}")]
            for endpoint in range(3)
        ]
        row_maps = [dict(row.coefficients) for row in model.rows]
        for endpoint in endpoints:
            assert any(
                coefficients == {presence: 1, endpoint: -1}
                and row.lower_bound == 0
                and row.upper_bound is None
                for row, coefficients in zip(model.rows, row_maps)
            )
        assert any(
            coefficients
            == {presence: 1, **{endpoint: -1 for endpoint in endpoints}}
            and row.lower_bound is None
            and row.upper_bound == 0
            for row, coefficients in zip(model.rows, row_maps)
        )
        component_variables = {
            variable_index
            for key, variable_index in index.items()
            if key[:-2] == prefix and key[-2] == component_role
        }
        assert any(
            coefficients
            == {variable_index: 1 for variable_index in component_variables}
            and row.lower_bound == 1
            and row.upper_bound is None
            for row, coefficients in zip(model.rows, row_maps)
        )

    for prefix in (("outer", "0"), ("inner", "0", "0")):
        assert_exact_component_or(
            prefix,
            component_role="validation_component_a",
            holdout_role="holdout_a",
            entity_prefix="a",
        )
        assert_exact_component_or(
            prefix,
            component_role="validation_component_g",
            holdout_role="holdout_g",
            entity_prefix="g",
        )


def test_transitive_component_capacity_gate_fails_closed() -> None:
    """Endpoint count cannot substitute for distinct homology components."""
    with pytest.raises(
        PolicyArithmeticError,
        match=r"Outer antibody component gate.*minimum=8, available=7",
    ):
        build_joint_feasibility_model(
            _path_component_problem(),
            outer_policy=_policy(
                minimum_validation_pairs=1,
                validation_upper=Ratio(1, 9),
                minimum_validation_components_per_axis=8,
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
