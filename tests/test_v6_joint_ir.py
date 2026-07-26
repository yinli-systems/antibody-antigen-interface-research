"""Tests for the V6-A canonical joint IR and CP-SAT separation loop."""

from __future__ import annotations

import itertools
import os
import tempfile
from pathlib import Path

import pytest

from interaction_first.nested_joint_milp import (
    CanonicalPair,
    JointLevelPolicy,
    JointSplitProblem,
    Ratio,
    ResourceLimits,
)
from interaction_first.v6_joint_ir import (
    CanonicalJointIR,
    ExactSeparationGate,
    IRLinearConstraint,
    SeparationFailure,
    SeparationOracleResult,
    audit_integer_constraints,
    build_membership_conflict_cut,
    build_v6_joint_ir,
    solve_cp_sat_with_exact_separation,
)
from interaction_first.v6_highs_backend import solve_highs_linear_projection


HASH_A = "a" * 64
HASH_B = "b" * 64


def _generic_ir() -> CanonicalJointIR:
    variables = (("x",), ("y",))
    constraint = IRLinearConstraint(
        name="linear_projection_000000000",
        coefficients=((('y',), 1),),
        lower_bound=1,
        upper_bound=1,
    )
    gate = ExactSeparationGate(
        gate_id="distribution_smd_tv_ks",
        level_id="outer_0",
        axis="antibody",
        role="validation_vs_training",
        oracle_arithmetic="decimal50_smd_fraction_tv_ks",
        threshold_binding_sha256=HASH_A,
    )
    return CanonicalJointIR(
        variable_keys=variables,
        linear_constraints=(constraint,),
        separation_gates=(gate,),
        source_bindings=(("authority", HASH_A), ("policy", HASH_B)),
        source_linear_projection_sha256=HASH_B,
        construction_scope="synthetic_test_formulation",
        independent_raw_authority_reconstruction=True,
    )


def _failure(value: int) -> SeparationFailure:
    return SeparationFailure(
        gate_id="distribution_smd_tv_ks",
        level_id="outer_0",
        axis="antibody",
        role="validation_vs_training",
        support=((('x',), value),),
        failure_payload={"maximum_smd_decimal": "0.35", "observed": "1.0"},
    )


def _oracle_result(
    ir: CanonicalJointIR,
    failures: tuple[SeparationFailure, ...] = (),
    *,
    independent_agreement: bool = True,
    evaluated_gates: tuple[tuple[str, str, str, str], ...] | None = None,
) -> SeparationOracleResult:
    if evaluated_gates is None:
        evaluated_gates = tuple(
            sorted(
                (
                    gate.gate_id,
                    gate.level_id,
                    gate.axis,
                    gate.role,
                )
                for gate in ir.separation_gates
            )
        )
    return SeparationOracleResult(
        evaluated_gates=evaluated_gates,
        failures=failures,
        primary_statistics_receipt_sha256=HASH_A,
        independent_audit_receipt_sha256=HASH_B,
        independent_agreement=independent_agreement,
    )


def _policy(validation_upper: Ratio) -> JointLevelPolicy:
    return JointLevelPolicy(
        minimum_validation_pairs=1,
        minimum_validation_endpoints_per_axis=1,
        minimum_validation_components_per_axis=1,
        validation_pair_lower_fraction=Ratio(1, 1000),
        validation_pair_upper_fraction=validation_upper,
        minimum_training_pair_fraction=Ratio(1, 2),
        minimum_training_endpoint_fraction=Ratio(1, 2),
        maximum_quarantine_pair_fraction=Ratio(1, 1000),
        maximum_validation_pair_relative_range=Ratio(1, 1000),
        maximum_other_count_relative_range=Ratio(1, 1000),
    )


def _independent_problem() -> JointSplitProblem:
    pairs = tuple(
        CanonicalPair(f"p{index}", f"a{index}", f"g{index}")
        for index in range(9)
    )
    return JointSplitProblem(
        pairs=pairs,
        antibody_direct_neighbors={
            f"a{index}": frozenset() for index in range(9)
        },
        antigen_direct_neighbors={
            f"g{index}": frozenset() for index in range(9)
        },
    )


def test_membership_conflict_cut_excludes_only_the_failed_support_state() -> None:
    ir = _generic_ir()
    cut = build_membership_conflict_cut(
        _failure(0),
        iteration=1,
        known_variables=ir.variable_keys,
    )
    accepted = []
    for x, y in itertools.product((0, 1), repeat=2):
        value = dict(((('x',), x), (('y',), y)))
        cut_value = sum(
            coefficient * value[key] for key, coefficient in cut.coefficients
        )
        if cut_value >= cut.lower_bound:
            accepted.append((x, y))
    assert accepted == [(1, 0), (1, 1)]
    assert len(cut.cut_sha256) == 64


def test_cp_sat_separation_accepts_only_optimal_exact_oracle_candidate() -> None:
    ir = _generic_ir()

    def oracle(assignment):
        failures = () if assignment[("x",)] == 1 else (_failure(0),)
        return _oracle_result(ir, failures)

    receipt = solve_cp_sat_with_exact_separation(ir, oracle)
    assert receipt["terminal_status"] == "OPTIMAL_EXACT_ORACLE_PASS"
    assert receipt["complete_separation"] is True
    assert receipt["candidate_present"] is True
    assert receipt["integer_row_audit_passed"] is True
    assert receipt["training_authorized"] is False
    assignment = {
        tuple(key): value for key, value in receipt["assignment"]
    }
    assert assignment[("x",)] == 1
    assert assignment[("y",)] == 1
    assert audit_integer_constraints(ir, assignment)


def test_cp_sat_rejects_oracle_support_that_disagrees_with_candidate() -> None:
    ir = _generic_ir()

    def invalid_oracle(assignment):
        return _oracle_result(ir, (_failure(1 - assignment[("x",)]),))

    receipt = solve_cp_sat_with_exact_separation(
        ir,
        invalid_oracle,
        maximum_iterations=2,
    )
    assert receipt["terminal_status"] == "SEPARATION_SUPPORT_STATE_DISAGREEMENT"
    assert receipt["complete_feasible_region_certified"] is False
    assert receipt["training_authorized"] is False


def test_cp_sat_rejects_incomplete_or_disagreeing_exact_audits() -> None:
    ir = _generic_ir()
    incomplete = solve_cp_sat_with_exact_separation(
        ir,
        lambda assignment: _oracle_result(ir, evaluated_gates=()),
    )
    assert incomplete["terminal_status"] == "INCOMPLETE_EXACT_GATE_AUDIT"
    assert incomplete["complete_feasible_region_certified"] is False
    disagreement = solve_cp_sat_with_exact_separation(
        ir,
        lambda assignment: _oracle_result(ir, independent_agreement=False),
    )
    assert disagreement["terminal_status"] == "INDEPENDENT_AUDIT_DISAGREEMENT"
    assert disagreement["complete_feasible_region_certified"] is False


def test_joint_ir_is_order_stable_and_preserves_every_linear_projection_row() -> None:
    problem = _independent_problem()
    limits = ResourceLimits(
        maximum_binary_variables=10_000,
        maximum_linear_constraints=20_000,
        maximum_nonzero_coefficients=100_000,
    )
    first = build_v6_joint_ir(
        problem,
        score_free_authority_sha256=HASH_A,
        solver_policy_sha256=HASH_B,
        outer_policy=_policy(Ratio(1, 9)),
        inner_policy=_policy(Ratio(1, 8)),
        resource_limits=limits,
    )
    reversed_problem = JointSplitProblem(
        pairs=tuple(reversed(problem.pairs)),
        antibody_direct_neighbors=dict(
            reversed(tuple(problem.antibody_direct_neighbors.items()))
        ),
        antigen_direct_neighbors=dict(
            reversed(tuple(problem.antigen_direct_neighbors.items()))
        ),
    )
    second = build_v6_joint_ir(
        reversed_problem,
        score_free_authority_sha256=HASH_A,
        solver_policy_sha256=HASH_B,
        outer_policy=_policy(Ratio(1, 9)),
        inner_policy=_policy(Ratio(1, 8)),
        resource_limits=limits,
    )
    assert first.canonical_payload() == second.canonical_payload()
    assert first.ir_sha256 == second.ir_sha256
    assert first.construction_scope == "v5_linear_projection_adapter"
    assert first.independent_raw_authority_reconstruction is False
    assert len(first.variable_keys) == 1188
    assert len(first.linear_constraints) == 3078
    assert len(first.separation_gates) == 66


def test_joint_ir_cp_sat_linear_projection_finds_integer_valid_assignment() -> None:
    ir = build_v6_joint_ir(
        _independent_problem(),
        score_free_authority_sha256=HASH_A,
        solver_policy_sha256=HASH_B,
        outer_policy=_policy(Ratio(1, 9)),
        inner_policy=_policy(Ratio(1, 8)),
        resource_limits=ResourceLimits(
            maximum_binary_variables=10_000,
            maximum_linear_constraints=20_000,
            maximum_nonzero_coefficients=100_000,
        ),
    )
    receipt = solve_cp_sat_with_exact_separation(
        ir,
        lambda assignment: _oracle_result(ir),
    )
    assert receipt["terminal_status"] == "OPTIMAL_EXACT_ORACLE_PASS"
    assignment = {
        tuple(key): value for key, value in receipt["assignment"]
    }
    assert audit_integer_constraints(ir, assignment)


def test_direct_highspy_receipt_is_projection_only_and_create_only() -> None:
    ir = _generic_ir()
    with tempfile.TemporaryDirectory(dir="/private/tmp") as directory:
        output = Path(directory) / "highs_receipt"
        try:
            receipt = solve_highs_linear_projection(ir, output)
            assert receipt["linear_projection_certified"] is True
            assert receipt["complete_score_free_feasible_region_certified"] is False
            assert receipt["nonlinear_smd_domain_authorized_by_this_receipt"] is False
            assert receipt["training_authorized"] is False
            assert receipt["user_solution_callback_used"] is False
            assert receipt["mip_logging_callback_used"] is True
            assert (output / "original_model.mps").is_file()
            assert (output / "presolved_model.mps").is_file()
            assert (output / "highs_options.txt").is_file()
            assert (output / "highs.log").is_file()
            assert (output / "solution.txt").is_file()
            assert (output / "highs_info.txt").is_file()
            assert (output / "variable_dictionary.json").is_file()
            assert (output / "mip_logging_trajectory.json").is_file()
            with pytest.raises(FileExistsError, match="already exists"):
                solve_highs_linear_projection(ir, output)
        finally:
            if output.exists():
                os.chmod(output, 0o700)
