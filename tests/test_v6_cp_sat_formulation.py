"""Tests for the independent raw-authority V6-A CP-SAT formulation."""

from __future__ import annotations

import ast
import hashlib
import itertools
import json
from fractions import Fraction
from pathlib import Path

import pytest

from interaction_first.nested_joint_milp import (
    CanonicalPair,
    JointLevelPolicy,
    JointSplitProblem,
    Ratio,
    ResourceLimits,
)
from interaction_first.score_free_authority import (
    AUTHORITY_FORMAT_VERSION,
    EDGE_LEDGER_FORMAT_VERSION,
    AuthorityCounts,
    build_score_free_authority,
    canonical_json_bytes,
)
from interaction_first.v6_cp_sat_formulation import (
    DirectCPSatError,
    DirectLevelPolicy,
    build_independent_cp_sat_formulation,
    direct_problem_from_authority_payload,
    implementation_source_sha256 as direct_formulation_source_sha256,
    solve_direct_linear_projection,
)
from interaction_first.v6_joint_ir import (
    JointIRError,
    build_v6_joint_ir,
    build_v6_joint_ir_from_direct_cp_sat,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "src" / "interaction_first" / "v6_cp_sat_formulation.py"


def _raw_fixture():
    authority = canonical_json_bytes(
        {
            "format_version": AUTHORITY_FORMAT_VERSION,
            "authorized_split": "train",
            "instance_ids": ["row0"],
        }
    )
    manifest = b"INSTANCE,vh_seq,vl_seq,ag_seq\nrow0,AAAAA,DDDDD,CCCCC\n"
    ledgers = {
        view: canonical_json_bytes(
            {
                "format_version": EDGE_LEDGER_FORMAT_VERSION,
                "view": view,
                "edges": [],
            }
        )
        for view in ("vh", "vl", "concatenated_antibody", "antigen")
    }
    counts = AuthorityCounts(1, 1, 1, 1)
    bundle = build_score_free_authority(
        authority_bytes=authority,
        manifest_bytes=manifest,
        edge_ledger_bytes=ledgers,
        expected_counts=counts,
    )
    payload_bytes = canonical_json_bytes(bundle.canonical_payload())
    return payload_bytes, ledgers, counts, bundle


def _direct_policy() -> DirectLevelPolicy:
    return DirectLevelPolicy(
        minimum_validation_pairs=0,
        minimum_validation_endpoints_per_axis=0,
        minimum_validation_components_per_axis=0,
        validation_pair_lower_fraction=Fraction(0),
        validation_pair_upper_fraction=Fraction(1),
        minimum_training_pair_fraction=Fraction(0),
        minimum_training_endpoint_fraction=Fraction(0),
        maximum_quarantine_pair_fraction=Fraction(1),
        maximum_validation_pair_relative_range=Fraction(1),
        maximum_other_count_relative_range=Fraction(1),
    )


def _projection_policy() -> JointLevelPolicy:
    return JointLevelPolicy(
        minimum_validation_pairs=0,
        minimum_validation_endpoints_per_axis=0,
        minimum_validation_components_per_axis=0,
        validation_pair_lower_fraction=Ratio(0, 1),
        validation_pair_upper_fraction=Ratio(1, 1),
        minimum_training_pair_fraction=Ratio(0, 1),
        minimum_training_endpoint_fraction=Ratio(0, 1),
        maximum_quarantine_pair_fraction=Ratio(1, 1),
        maximum_validation_pair_relative_range=Ratio(1, 1),
        maximum_other_count_relative_range=Ratio(1, 1),
    )


def _solve_ir_with_fixed_bits(ir, fixed_assignment) -> str:
    from ortools.sat.python import cp_model

    model = cp_model.CpModel()
    variables = {
        key: model.new_bool_var(f"v_{index:09d}")
        for index, key in enumerate(ir.variable_keys)
    }
    for row in ir.linear_constraints:
        expression = sum(coefficient * variables[key] for key, coefficient in row.coefficients)
        if row.lower_bound is not None:
            model.add(expression >= row.lower_bound)
        if row.upper_bound is not None:
            model.add(expression <= row.upper_bound)
    for key, value in fixed_assignment.items():
        model.add(variables[key] == value)
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 2026072701
    status = solver.solve(model)
    return solver.status_name(status)


def test_direct_formulation_has_no_historical_matrix_or_highs_import() -> None:
    tree = ast.parse(SOURCE_PATH.read_text(encoding="ascii"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert not any("nested_joint_milp" in name for name in imported)
    assert not any("v6_joint_ir" in name for name in imported)
    assert not any("v6_highs_backend" in name for name in imported)


def test_direct_authority_reconstruction_binds_graph_and_rejects_reanchored_fields() -> None:
    payload_bytes, ledgers, counts, bundle = _raw_fixture()
    expected_counts = {
        "source_rows": counts.source_rows,
        "exact_pairs": counts.exact_pairs,
        "antibody_entities": counts.antibody_entities,
        "antigen_entities": counts.antigen_entities,
    }
    problem = direct_problem_from_authority_payload(
        payload_bytes,
        ledgers,
        expected_authority_payload_sha256=hashlib.sha256(payload_bytes).hexdigest(),
        expected_counts=expected_counts,
    )
    assert len(problem.pairs) == 1
    assert problem.antibody_components == ((bundle.antibodies[0].entity_id,),)
    assert problem.antigen_components == ((bundle.antigens[0].entity_id,),)

    contaminated = json.loads(payload_bytes)
    contaminated["antibodies"][0]["model_score"] = 0
    contaminated_without_digest = dict(contaminated)
    contaminated_without_digest.pop("scientific_payload_sha256")
    contaminated["scientific_payload_sha256"] = hashlib.sha256(
        canonical_json_bytes(contaminated_without_digest)
    ).hexdigest()
    contaminated_bytes = canonical_json_bytes(contaminated)
    with pytest.raises(DirectCPSatError, match="antibody row schema differs"):
        direct_problem_from_authority_payload(
            contaminated_bytes,
            ledgers,
            expected_authority_payload_sha256=hashlib.sha256(contaminated_bytes).hexdigest(),
            expected_counts=expected_counts,
        )


def test_all_64_outer_holdout_states_match_the_shared_linear_projection() -> None:
    payload_bytes, ledgers, counts, bundle = _raw_fixture()
    problem = direct_problem_from_authority_payload(
        payload_bytes,
        ledgers,
        expected_authority_payload_sha256=hashlib.sha256(payload_bytes).hexdigest(),
        expected_counts={
            "source_rows": counts.source_rows,
            "exact_pairs": counts.exact_pairs,
            "antibody_entities": counts.antibody_entities,
            "antigen_entities": counts.antigen_entities,
        },
    )
    direct = build_independent_cp_sat_formulation(
        problem,
        outer_policy=_direct_policy(),
        inner_policy=_direct_policy(),
    )
    pair = bundle.pairs[0]
    projection = build_v6_joint_ir(
        JointSplitProblem(
            pairs=(CanonicalPair(pair.pair_id, pair.antibody_id, pair.antigen_id),),
            antibody_direct_neighbors={pair.antibody_id: frozenset()},
            antigen_direct_neighbors={pair.antigen_id: frozenset()},
        ),
        score_free_authority_sha256=bundle.scientific_payload_sha256,
        solver_policy_sha256="a" * 64,
        outer_policy=_projection_policy(),
        inner_policy=_projection_policy(),
        resource_limits=ResourceLimits(10_000, 20_000, 100_000),
    )
    assert set(direct.variables) == set(projection.variable_keys)
    assert sum(dict(direct.constraint_family_counts).values()) == len(
        projection.linear_constraints
    )
    assert len(direct.model.Proto().constraints) == len(projection.linear_constraints)
    rebuilt = build_independent_cp_sat_formulation(
        problem,
        outer_policy=_direct_policy(),
        inner_policy=_direct_policy(),
    )
    assert rebuilt.formulation_sha256 == direct.formulation_sha256
    states = []
    for bits in itertools.product((0, 1), repeat=6):
        fixed = {}
        for outer in range(3):
            fixed[("outer", str(outer), "holdout_a", pair.antibody_id)] = bits[outer]
            fixed[("outer", str(outer), "holdout_g", pair.antigen_id)] = bits[outer + 3]
        direct_status = solve_direct_linear_projection(
            direct,
            fixed_assignment=fixed,
        )["status"]
        projection_status = _solve_ir_with_fixed_bits(projection, fixed)
        states.append((bits, direct_status, projection_status))
    assert all(direct_status == projection_status for _, direct_status, projection_status in states)
    assert {status for _, status, _ in states} == {"OPTIMAL", "INFEASIBLE"}


def test_direct_receipt_remains_non_authorizing_before_exact_separation() -> None:
    payload_bytes, ledgers, counts, _ = _raw_fixture()
    problem = direct_problem_from_authority_payload(
        payload_bytes,
        ledgers,
        expected_authority_payload_sha256=hashlib.sha256(payload_bytes).hexdigest(),
        expected_counts={
            "source_rows": counts.source_rows,
            "exact_pairs": counts.exact_pairs,
            "antibody_entities": counts.antibody_entities,
            "antigen_entities": counts.antigen_entities,
        },
    )
    direct = build_independent_cp_sat_formulation(
        problem,
        outer_policy=_direct_policy(),
        inner_policy=_direct_policy(),
    )
    receipt = solve_direct_linear_projection(direct)
    assert receipt["status"] == "OPTIMAL"
    assert receipt["historical_milp_matrix_consumed"] is False
    assert receipt["highs_artifact_consumed"] is False
    assert receipt["exact_score_free_separation_completed"] is False
    assert receipt["complete_score_free_feasible_region_certified"] is False
    assert receipt["training_authorized"] is False


def test_direct_cp_sat_proto_builds_authoritative_joint_ir() -> None:
    payload_bytes, ledgers, counts, _ = _raw_fixture()
    problem = direct_problem_from_authority_payload(
        payload_bytes,
        ledgers,
        expected_authority_payload_sha256=hashlib.sha256(payload_bytes).hexdigest(),
        expected_counts={
            "source_rows": counts.source_rows,
            "exact_pairs": counts.exact_pairs,
            "antibody_entities": counts.antibody_entities,
            "antigen_entities": counts.antigen_entities,
        },
    )
    direct = build_independent_cp_sat_formulation(
        problem,
        outer_policy=_direct_policy(),
        inner_policy=_direct_policy(),
    )
    ir = build_v6_joint_ir_from_direct_cp_sat(
        direct,
        solver_policy_sha256="a" * 64,
        direct_formulation_source_sha256=direct_formulation_source_sha256(),
        exact_oracle_source_sha256="b" * 64,
    )
    assert ir.independent_raw_authority_reconstruction is True
    assert ir.construction_scope == "independent_raw_authority_direct_cp_sat_proto"
    assert set(ir.variable_keys) == set(direct.variables)
    assert len(ir.linear_constraints) == len(direct.model.Proto().constraints)
    assert len(ir.separation_gates) == 66
    assert dict(ir.source_bindings)["direct_cp_sat_formulation"] == (
        direct.formulation_sha256
    )


def test_direct_cp_sat_adapter_rejects_non_linear_proto_constructs() -> None:
    payload_bytes, ledgers, counts, _ = _raw_fixture()
    problem = direct_problem_from_authority_payload(
        payload_bytes,
        ledgers,
        expected_authority_payload_sha256=hashlib.sha256(payload_bytes).hexdigest(),
        expected_counts={
            "source_rows": counts.source_rows,
            "exact_pairs": counts.exact_pairs,
            "antibody_entities": counts.antibody_entities,
            "antigen_entities": counts.antigen_entities,
        },
    )
    direct = build_independent_cp_sat_formulation(
        problem,
        outer_policy=_direct_policy(),
        inner_policy=_direct_policy(),
    )
    direct.model.add_bool_or([next(iter(direct.variables.values()))])
    with pytest.raises(JointIRError, match="nonlinear row"):
        build_v6_joint_ir_from_direct_cp_sat(
            direct,
            solver_policy_sha256="a" * 64,
            direct_formulation_source_sha256=direct_formulation_source_sha256(),
            exact_oracle_source_sha256="b" * 64,
        )
