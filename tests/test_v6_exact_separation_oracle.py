"""Tests for the complete V6-A production exact-separation oracle."""

from __future__ import annotations

from interaction_first.v6_exact_separation_oracle import (
    ProductionExactSeparationOracle,
    implementation_source_sha256,
)
from interaction_first.v6_joint_ir import (
    CanonicalJointIR,
    IRLinearConstraint,
    separation_gate_registry,
)


def _authority(pair_count: int = 1500) -> dict[str, object]:
    antibodies = []
    antigens = []
    pairs = []
    for index in range(pair_count):
        antibody_id = f"a{index:04d}"
        antigen_id = f"g{index:04d}"
        pair_id = f"p{index:04d}"
        antibodies.append(
            {
                "entity_id": antibody_id,
                "component_id": f"ca{index:04d}",
                "vh_length": 110,
                "vl_length": 105,
                "concatenated_antibody_length": 215,
                "vh_direct_degree": 0,
                "vl_direct_degree": 0,
                "concatenated_direct_degree": 0,
                "union_direct_degree": 0,
                "pair_incidence_degree": 1,
                "component_size": 1,
            }
        )
        antigens.append(
            {
                "entity_id": antigen_id,
                "component_id": f"cg{index:04d}",
                "antigen_length": 220,
                "antigen_direct_degree": 0,
                "pair_incidence_degree": 1,
                "component_size": 1,
            }
        )
        pairs.append(
            {
                "pair_id": pair_id,
                "antibody_id": antibody_id,
                "antigen_id": antigen_id,
                "source_instance_ids": [f"row{index:04d}"],
            }
        )
    return {
        "format_version": "score_free_authority_v2",
        "antibodies": antibodies,
        "antigens": antigens,
        "pairs": pairs,
        "forbidden_outcome_or_model_fields_accepted": False,
    }


def _ir_and_assignment(pair_count: int = 1500):
    pair_ids = tuple(f"p{index:04d}" for index in range(pair_count))
    assignment = {}
    for outer in range(3):
        validation = set(pair_ids[120 * outer : 120 * (outer + 1)])
        outer_training = set(pair_ids) - validation
        for pair_id in pair_ids:
            assignment[("outer", str(outer), "validation_pair", pair_id)] = int(
                pair_id in validation
            )
            assignment[("outer", str(outer), "training_pair", pair_id)] = int(
                pair_id in outer_training
            )
            assignment[("outer", str(outer), "quarantine_pair", pair_id)] = 0
        ordered_training = tuple(sorted(outer_training))
        for inner in range(3):
            inner_validation = set(
                ordered_training[100 * inner : 100 * (inner + 1)]
            )
            inner_training = outer_training - inner_validation
            for pair_id in pair_ids:
                assignment[("inner", str(outer), str(inner), "validation_pair", pair_id)] = int(
                    pair_id in inner_validation
                )
                assignment[("inner", str(outer), str(inner), "training_pair", pair_id)] = int(
                    pair_id in inner_training
                )
                assignment[("inner", str(outer), str(inner), "quarantine_pair", pair_id)] = 0
    variable_keys = tuple(sorted(assignment))
    ir = CanonicalJointIR(
        variable_keys=variable_keys,
        linear_constraints=(
            IRLinearConstraint(
                name="synthetic_membership_anchor",
                coefficients=((variable_keys[0], 1),),
                lower_bound=0,
                upper_bound=1,
            ),
        ),
        separation_gates=separation_gate_registry("b" * 64),
        source_bindings=tuple(
            sorted(
                {
                    "direct_cp_sat_formulation": "c" * 64,
                    "direct_cp_sat_formulation_source": "d" * 64,
                    "exact_separation_oracle_source": implementation_source_sha256(),
                    "score_free_authority": "a" * 64,
                    "solver_policy": "b" * 64,
                }.items()
            )
        ),
        source_linear_projection_sha256="e" * 64,
        construction_scope="synthetic_independent_raw_authority",
        independent_raw_authority_reconstruction=True,
    )
    return ir, assignment


def test_all_66_gates_pass_and_both_statistics_paths_agree() -> None:
    authority = _authority()
    ir, assignment = _ir_and_assignment()
    oracle = ProductionExactSeparationOracle(
        ir=ir,
        authority_payload=authority,
        independent_authority_payload=authority,
    )
    result = oracle(assignment)
    assert len(result.evaluated_gates) == 66
    assert result.failures == ()
    assert result.independent_agreement is True
    assert len(result.primary_statistics_receipt_sha256) == 64
    assert len(result.independent_audit_receipt_sha256) == 64
    assert len(oracle.evidence_history) == 1
    evidence = oracle.evidence_history[0]
    assert evidence["all_66_gates_evaluated"] is True
    assert evidence["independent_agreement"] is True
    assert evidence["failure_count"] == 0
    assert evidence["training_authorized"] is False

    failing_assignment = dict(assignment)
    failing_assignment[("outer", "0", "validation_pair", "p0000")] = 0
    failing_assignment[("outer", "0", "training_pair", "p0000")] = 1
    failing_result = oracle(failing_assignment)
    panel_failure = next(
        failure
        for failure in failing_result.failures
        if failure.gate_id == "panel_shape_and_chance_mrr"
        and failure.level_id == "outer_0"
    )
    assert "validation_pair_count_outside_locked_interval" in (
        panel_failure.failure_payload["panel"]["shape_failures"]
    )
    assert all(key[:2] == ("outer", "0") for key, _ in panel_failure.support)
    assert len(panel_failure.support) == 3 * 1500
    assert all(
        failing_assignment[key] == value for key, value in panel_failure.support
    )
