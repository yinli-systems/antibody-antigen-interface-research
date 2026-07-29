"""Diagnose an infeasible V6 linear projection without weakening any gate.

This module adds one Boolean activation literal per recorded constraint family
to a clone of the direct CP-SAT model. With every activation assumed true, the
diagnostic model is semantically identical to the source linear projection.
CP-SAT may then return a sufficient family core. A deterministic deletion pass
can prove that no family in the reported core is individually removable.

The result is diagnostic evidence only. It cannot certify the nonlinear
score-free feasible region, authorize a split, change a threshold, start
training, select a model, or access outer/test labels.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Sequence

from .v6_cp_sat_formulation import DirectCPModel, LOCKED_RANDOM_SEED


DIAGNOSTIC_FORMAT = "v6_constraint_family_infeasibility_diagnostic_v1"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ConstraintFamilyDiagnosticError(ValueError):
    """Raised when family provenance or a diagnostic replay is malformed."""


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise ConstraintFamilyDiagnosticError(
            "diagnostic metadata must be finite ASCII JSON"
        ) from error


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256_model_binary(model: Any) -> str:
    try:
        with tempfile.TemporaryDirectory(
            prefix="v6-family-diagnostic-"
        ) as directory:
            path = Path(directory) / "model.binpb"
            if not model.export_to_file(str(path)):
                raise ConstraintFamilyDiagnosticError(
                    "CP-SAT model binary export failed"
                )
            payload = path.read_bytes()
    except (AttributeError, OSError, TypeError, ValueError) as error:
        raise ConstraintFamilyDiagnosticError(
            "CP-SAT model cannot be exported as canonical binary"
        ) from error
    if not payload:
        raise ConstraintFamilyDiagnosticError(
            "CP-SAT model binary export is empty"
        )
    return hashlib.sha256(payload).hexdigest()


def implementation_source_sha256() -> str:
    """Hash the exact diagnostic implementation bytes."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _require_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ConstraintFamilyDiagnosticError(
            f"{name} must be a lowercase 64-character SHA-256"
        )
    return value


def _positive_time_limit(value: Any | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ConstraintFamilyDiagnosticError(
            "maximum_time_seconds_per_solve must be positive and finite"
        )
    try:
        converted = float(value)
    except (TypeError, ValueError) as error:
        raise ConstraintFamilyDiagnosticError(
            "maximum_time_seconds_per_solve must be positive and finite"
        ) from error
    if not math.isfinite(converted) or converted <= 0.0:
        raise ConstraintFamilyDiagnosticError(
            "maximum_time_seconds_per_solve must be positive and finite"
        )
    return converted


@dataclass(frozen=True)
class FamilyAssumptionModel:
    """One cloned CP model with exact family-to-activation provenance."""

    model: Any
    family_variable_indices: tuple[tuple[str, int], ...]
    source_proto_sha256: str
    diagnostic_proto_sha256: str
    source_formulation_sha256: str
    source_constraint_count: int
    source_variable_count: int

    @property
    def family_index(self) -> Mapping[str, int]:
        return dict(self.family_variable_indices)


def build_family_assumption_model(
    direct_model: DirectCPModel,
) -> FamilyAssumptionModel:
    """Clone one direct model and gate every source row by its exact family."""
    if not isinstance(direct_model, DirectCPModel):
        raise ConstraintFamilyDiagnosticError(
            "direct_model must be a DirectCPModel"
        )
    _require_sha256(
        direct_model.formulation_sha256,
        "direct formulation digest",
    )
    source_proto = direct_model.model.Proto()
    source_constraint_count = len(source_proto.constraints)
    source_variable_count = len(source_proto.variables)
    families = tuple(direct_model.constraint_families)
    if (
        not families
        or len(families) != source_constraint_count
        or any(
            not isinstance(family, str)
            or not family
            or not family.isascii()
            or family.strip() != family
            for family in families
        )
    ):
        raise ConstraintFamilyDiagnosticError(
            "constraint-family sequence does not cover the source rows"
        )
    observed_counts = {
        family: families.count(family) for family in set(families)
    }
    if observed_counts != dict(direct_model.constraint_family_counts):
        raise ConstraintFamilyDiagnosticError(
            "constraint-family counts disagree with the source model"
        )
    for constraint in source_proto.constraints:
        if constraint.enforcement_literal or not constraint.has_linear():
            raise ConstraintFamilyDiagnosticError(
                "source model must contain unconditional linear rows only"
            )
    if direct_model.model.has_objective():
        raise ConstraintFamilyDiagnosticError(
            "family infeasibility diagnosis requires a satisfaction model"
        )

    working_model = direct_model.model.clone()
    family_variables = {
        family: working_model.new_bool_var(
            f"family_active_{index:03d}"
        )
        for index, family in enumerate(sorted(set(families)))
    }
    for row_index, family in enumerate(families):
        working_model.Proto().constraints[row_index].enforcement_literal.append(
            family_variables[family].index
        )
    working_model.add_assumptions(
        [family_variables[family] for family in sorted(family_variables)]
    )

    if len(working_model.Proto().constraints) != source_constraint_count:
        raise ConstraintFamilyDiagnosticError(
            "family activation changed the source constraint count"
        )
    if (
        len(working_model.Proto().variables)
        != source_variable_count + len(family_variables)
    ):
        raise ConstraintFamilyDiagnosticError(
            "family activation variable count differs"
        )
    return FamilyAssumptionModel(
        model=working_model,
        family_variable_indices=tuple(
            (family, family_variables[family].index)
            for family in sorted(family_variables)
        ),
        source_proto_sha256=_sha256_model_binary(direct_model.model),
        diagnostic_proto_sha256=_sha256_model_binary(working_model),
        source_formulation_sha256=direct_model.formulation_sha256,
        source_constraint_count=source_constraint_count,
        source_variable_count=source_variable_count,
    )


def _solve_subset(
    assumption_model: FamilyAssumptionModel,
    active_families: Sequence[str],
    *,
    maximum_time_seconds: float | None,
) -> tuple[str, Any]:
    from ortools.sat.python import cp_model

    family_index = dict(assumption_model.family_variable_indices)
    normalized = tuple(sorted(active_families))
    if len(normalized) != len(set(normalized)):
        raise ConstraintFamilyDiagnosticError(
            "active family list contains duplicates"
        )
    unknown = set(normalized) - set(family_index)
    if unknown:
        raise ConstraintFamilyDiagnosticError(
            f"active family list contains unknown values: {sorted(unknown)}"
        )
    model = assumption_model.model.clone()
    model.clear_assumptions()
    model.add_assumptions(
        [
            model.get_bool_var_from_proto_index(family_index[family])
            for family in normalized
        ]
    )
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = LOCKED_RANDOM_SEED
    solver.parameters.randomize_search = False
    solver.parameters.relative_gap_limit = 0.0
    solver.parameters.absolute_gap_limit = 0.0
    solver.parameters.core_minimization_level = 2
    solver.parameters.find_multiple_cores = True
    if maximum_time_seconds is not None:
        solver.parameters.max_time_in_seconds = maximum_time_seconds
    status = solver.solve(model)
    return solver.status_name(status), solver


def diagnose_constraint_families(
    direct_model: DirectCPModel,
    *,
    maximum_time_seconds_per_solve: float | None = None,
    minimize_core: bool = True,
) -> dict[str, Any]:
    """Return a fail-closed family-core receipt for one linear projection."""
    if not isinstance(minimize_core, bool):
        raise ConstraintFamilyDiagnosticError("minimize_core must be boolean")
    maximum_time_seconds_per_solve = _positive_time_limit(
        maximum_time_seconds_per_solve
    )
    assumption_model = build_family_assumption_model(direct_model)
    all_families = tuple(
        family for family, _ in assumption_model.family_variable_indices
    )
    full_status, full_solver = _solve_subset(
        assumption_model,
        all_families,
        maximum_time_seconds=maximum_time_seconds_per_solve,
    )
    receipt: dict[str, Any] = {
        "format_version": DIAGNOSTIC_FORMAT,
        "source_formulation_sha256": (
            assumption_model.source_formulation_sha256
        ),
        "source_proto_sha256": assumption_model.source_proto_sha256,
        "diagnostic_proto_sha256": assumption_model.diagnostic_proto_sha256,
        "diagnostic_source_sha256": implementation_source_sha256(),
        "source_constraint_count": assumption_model.source_constraint_count,
        "source_variable_count": assumption_model.source_variable_count,
        "constraint_family_count": len(all_families),
        "all_constraint_families": list(all_families),
        "solver": {
            "num_search_workers": 1,
            "random_seed": LOCKED_RANDOM_SEED,
            "randomize_search": False,
            "relative_gap_limit": 0.0,
            "absolute_gap_limit": 0.0,
            "maximum_time_seconds_per_solve": (
                maximum_time_seconds_per_solve
            ),
        },
        "full_family_status": full_status,
        "sufficient_core": [],
        "core_replay_status": "NOT_RUN",
        "deletion_attempts": [],
        "minimality_checks": [],
        "irreducibility_proven": False,
        "linear_projection_infeasibility_diagnosed": False,
        "full_protocol_infeasibility_claim_authorized": False,
        "threshold_change_authorized": False,
        "split_gate0_authorized": False,
        "training_authorized": False,
        "model_selection_authorized": False,
        "outer_evaluation_authorized": False,
        "test_access_authorized": False,
    }
    if full_status != "INFEASIBLE":
        receipt["receipt_sha256"] = _sha256_json(receipt)
        return receipt

    index_to_family = {
        index: family
        for family, index in assumption_model.family_variable_indices
    }
    core_indices = tuple(
        int(value)
        for value in full_solver.sufficient_assumptions_for_infeasibility()
    )
    if not core_indices or any(index not in index_to_family for index in core_indices):
        raise ConstraintFamilyDiagnosticError(
            "CP-SAT returned an invalid family assumption core"
        )
    core = list(
        dict.fromkeys(index_to_family[index] for index in core_indices)
    )
    core.sort()
    core_status, _ = _solve_subset(
        assumption_model,
        core,
        maximum_time_seconds=maximum_time_seconds_per_solve,
    )
    receipt["core_replay_status"] = core_status
    if core_status != "INFEASIBLE":
        receipt["sufficient_core"] = core
        receipt["receipt_sha256"] = _sha256_json(receipt)
        return receipt

    deletion_attempts = []
    minimality_checks = []
    irreducibility_proven = False
    if minimize_core:
        position = 0
        while position < len(core):
            family = core[position]
            candidate = [value for value in core if value != family]
            status, _ = _solve_subset(
                assumption_model,
                candidate,
                maximum_time_seconds=maximum_time_seconds_per_solve,
            )
            deletion_attempts.append(
                {
                    "removed_family": family,
                    "remaining_family_count": len(candidate),
                    "status": status,
                }
            )
            if status == "INFEASIBLE":
                core = candidate
            else:
                position += 1

    final_status, _ = _solve_subset(
        assumption_model,
        core,
        maximum_time_seconds=maximum_time_seconds_per_solve,
    )
    if minimize_core and final_status == "INFEASIBLE":
        for family in core:
            candidate = [value for value in core if value != family]
            status, _ = _solve_subset(
                assumption_model,
                candidate,
                maximum_time_seconds=maximum_time_seconds_per_solve,
            )
            minimality_checks.append(
                {
                    "removed_family": family,
                    "remaining_family_count": len(candidate),
                    "status": status,
                }
            )
        irreducibility_proven = bool(core) and all(
            check["status"] in {"OPTIMAL", "FEASIBLE"}
            for check in minimality_checks
        )
    receipt["sufficient_core"] = core
    receipt["core_replay_status"] = final_status
    receipt["deletion_attempts"] = deletion_attempts
    receipt["minimality_checks"] = minimality_checks
    receipt["irreducibility_proven"] = irreducibility_proven
    receipt["linear_projection_infeasibility_diagnosed"] = (
        final_status == "INFEASIBLE"
    )
    receipt["receipt_sha256"] = _sha256_json(receipt)
    return receipt


__all__ = [
    "ConstraintFamilyDiagnosticError",
    "DIAGNOSTIC_FORMAT",
    "FamilyAssumptionModel",
    "build_family_assumption_model",
    "diagnose_constraint_families",
    "implementation_source_sha256",
]
