"""Tests for deterministic V6 constraint-family infeasibility diagnosis."""

from __future__ import annotations

import copy

import pytest

from interaction_first.v6_constraint_family_diagnostic import (
    ConstraintFamilyDiagnosticError,
    build_family_assumption_model,
    diagnose_constraint_families,
)
from interaction_first.v6_cp_sat_formulation import DirectCPModel


def _direct_model(*, infeasible: bool) -> DirectCPModel:
    cp_model = pytest.importorskip("ortools.sat.python.cp_model")
    model = cp_model.CpModel()
    value = model.new_bool_var("value")
    families = []
    model.add(value >= 0)
    families.append("domain")
    if infeasible:
        model.add(value == 0)
        families.append("zero")
        model.add(value == 1)
        families.append("one")
    return DirectCPModel(
        model=model,
        variables={("value",): value},
        authority_payload_sha256="a" * 64,
        constraint_family_counts=tuple(
            sorted(
                (family, families.count(family))
                for family in set(families)
            )
        ),
        constraint_families=tuple(families),
        formulation_sha256="b" * 64,
    )


def test_family_activation_preserves_source_rows_and_bytes() -> None:
    direct = _direct_model(infeasible=True)
    source_text = str(direct.model.Proto())
    diagnostic = build_family_assumption_model(direct)
    assert str(direct.model.Proto()) == source_text
    assert diagnostic.source_constraint_count == 3
    assert diagnostic.source_variable_count == 1
    assert len(diagnostic.model.Proto().constraints) == 3
    assert len(diagnostic.model.Proto().variables) == 4
    assert set(diagnostic.family_index) == {"domain", "one", "zero"}
    for row_index, family in enumerate(direct.constraint_families):
        assert list(
            diagnostic.model.Proto().constraints[
                row_index
            ].enforcement_literal
        ) == [diagnostic.family_index[family]]


def test_infeasible_core_is_replayed_and_deletion_minimized() -> None:
    receipt = diagnose_constraint_families(_direct_model(infeasible=True))
    assert receipt["full_family_status"] == "INFEASIBLE"
    assert receipt["core_replay_status"] == "INFEASIBLE"
    assert receipt["sufficient_core"] == ["one", "zero"]
    assert receipt["minimality_checks"] == [
        {
            "removed_family": "one",
            "remaining_family_count": 1,
            "status": "OPTIMAL",
        },
        {
            "removed_family": "zero",
            "remaining_family_count": 1,
            "status": "OPTIMAL",
        },
    ]
    assert receipt["irreducibility_proven"] is True
    assert receipt["linear_projection_infeasibility_diagnosed"] is True
    assert receipt["full_protocol_infeasibility_claim_authorized"] is False
    assert receipt["threshold_change_authorized"] is False
    assert receipt["split_gate0_authorized"] is False
    assert receipt["training_authorized"] is False
    assert receipt["outer_evaluation_authorized"] is False
    assert receipt["test_access_authorized"] is False


def test_feasible_model_produces_no_infeasibility_claim() -> None:
    receipt = diagnose_constraint_families(_direct_model(infeasible=False))
    assert receipt["full_family_status"] == "OPTIMAL"
    assert receipt["sufficient_core"] == []
    assert receipt["core_replay_status"] == "NOT_RUN"
    assert receipt["minimality_checks"] == []
    assert receipt["linear_projection_infeasibility_diagnosed"] is False
    assert receipt["irreducibility_proven"] is False


def test_diagnostic_is_deterministic() -> None:
    first = diagnose_constraint_families(_direct_model(infeasible=True))
    second = diagnose_constraint_families(_direct_model(infeasible=True))
    assert first == second
    assert len(first["receipt_sha256"]) == 64


def test_family_sequence_must_cover_every_row() -> None:
    direct = _direct_model(infeasible=True)
    malformed = DirectCPModel(
        model=direct.model,
        variables=direct.variables,
        authority_payload_sha256=direct.authority_payload_sha256,
        constraint_family_counts=direct.constraint_family_counts,
        constraint_families=direct.constraint_families[:-1],
        formulation_sha256=direct.formulation_sha256,
    )
    with pytest.raises(ConstraintFamilyDiagnosticError, match="cover"):
        build_family_assumption_model(malformed)


def test_family_count_disagreement_is_rejected() -> None:
    direct = _direct_model(infeasible=True)
    malformed = DirectCPModel(
        model=direct.model,
        variables=direct.variables,
        authority_payload_sha256=direct.authority_payload_sha256,
        constraint_family_counts=(("domain", 3),),
        constraint_families=direct.constraint_families,
        formulation_sha256=direct.formulation_sha256,
    )
    with pytest.raises(ConstraintFamilyDiagnosticError, match="counts"):
        build_family_assumption_model(malformed)


def test_inputs_are_fail_closed() -> None:
    direct = _direct_model(infeasible=True)
    with pytest.raises(ConstraintFamilyDiagnosticError, match="positive"):
        diagnose_constraint_families(
            direct,
            maximum_time_seconds_per_solve=0,
        )
    with pytest.raises(ConstraintFamilyDiagnosticError, match="boolean"):
        diagnose_constraint_families(direct, minimize_core=1)

    altered = copy.copy(direct)
    altered = DirectCPModel(
        model=altered.model,
        variables=altered.variables,
        authority_payload_sha256=altered.authority_payload_sha256,
        constraint_family_counts=altered.constraint_family_counts,
        constraint_families=altered.constraint_families,
        formulation_sha256="not-a-digest",
    )
    with pytest.raises(ConstraintFamilyDiagnosticError, match="SHA-256"):
        build_family_assumption_model(altered)
