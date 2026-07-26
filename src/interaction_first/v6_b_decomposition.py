"""Fail-closed V6-B decomposition and row-shard telemetry primitives.

This module does not solve a split. It partitions the independently rebuilt V6-A
canonical IR into one outer master and three outer-conditioned inner groups, and it
constructs deterministic telemetry-only row shards without sampling entities or pairs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .v6_joint_ir import (
    CanonicalJointIR,
    ExactSeparationGate,
    IRLinearConstraint,
    VariableKey,
)


PARTITION_FORMAT = "v6_b_canonical_decomposition_partition_v1"
TELEMETRY_FORMAT = "v6_b_full_variable_row_shard_telemetry_v1"
EXPECTED_OUTER_GATES = 18
EXPECTED_INNER_GATES_PER_GROUP = 16
EXPECTED_INNER_GROUPS = 3
EXPECTED_TOTAL_GATES = 66
LOCKED_RANDOM_SEED = 2026072701


class V6BDecompositionError(ValueError):
    """Raised when a V6-B partition or telemetry claim is not reproducible."""


def canonical_json_bytes(payload: object) -> bytes:
    """Encode strict canonical ASCII JSON with one terminal LF."""
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )


def _sha256(payload: object) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def implementation_source_sha256() -> str:
    """Hash the exact V6-B decomposition implementation bytes."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _positive_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise V6BDecompositionError(f"{label} must be a positive integer")
    return value


def _outer_index(key: VariableKey) -> int:
    if not key or key[0] not in {"outer", "inner"}:
        raise V6BDecompositionError("variable is outside the declared V6 scope")
    if len(key) < (5 if key[0] == "inner" else 4):
        raise V6BDecompositionError("variable key is shorter than its declared scope")
    try:
        value = int(key[1])
    except ValueError as error:
        raise V6BDecompositionError("outer index is not an integer") from error
    if str(value) != key[1] or value not in range(EXPECTED_INNER_GROUPS):
        raise V6BDecompositionError("outer index is outside the frozen range")
    if key[0] == "inner":
        try:
            inner = int(key[2])
        except ValueError as error:
            raise V6BDecompositionError("inner index is not an integer") from error
        if str(inner) != key[2] or inner not in range(3):
            raise V6BDecompositionError("inner index is outside the frozen range")
    return value


def _row_scope(row: IRLinearConstraint) -> tuple[str, int | None]:
    inner_groups = {
        _outer_index(key)
        for key, _ in row.coefficients
        if key[0] == "inner"
    }
    outer_groups = {
        _outer_index(key)
        for key, _ in row.coefficients
        if key[0] == "outer"
    }
    for key, _ in row.coefficients:
        _outer_index(key)
    if not inner_groups:
        return ("outer_master", None)
    if len(inner_groups) != 1:
        raise V6BDecompositionError("one row mixes independent inner groups")
    group = next(iter(inner_groups))
    if outer_groups - {group}:
        raise V6BDecompositionError("inner row references another outer group")
    return ("inner_subproblem", group)


def _gate_scope(gate: ExactSeparationGate) -> tuple[str, int | None]:
    level = gate.level_id
    if "_inner" not in level:
        return ("outer_master", None)
    parts = level.split("_")
    if len(parts) < 3 or parts[0] != "outer" or parts[2] != "inner":
        raise V6BDecompositionError("inner gate level is noncanonical")
    try:
        group = int(parts[1])
    except ValueError as error:
        raise V6BDecompositionError("inner gate outer index is invalid") from error
    if group not in range(EXPECTED_INNER_GROUPS):
        raise V6BDecompositionError("inner gate outer index is outside range")
    return ("inner_subproblem", group)


@dataclass(frozen=True)
class V6BIRPartition:
    """One lossless outer-master plus three-inner partition of a canonical IR."""

    source_ir_sha256: str
    master_variables: tuple[VariableKey, ...]
    master_rows: tuple[IRLinearConstraint, ...]
    master_gates: tuple[ExactSeparationGate, ...]
    inner_variables: tuple[tuple[int, tuple[VariableKey, ...]], ...]
    inner_rows: tuple[tuple[int, tuple[IRLinearConstraint, ...]], ...]
    inner_gates: tuple[tuple[int, tuple[ExactSeparationGate, ...]], ...]

    def canonical_payload(self) -> dict[str, object]:
        """Return a content-addressed machine payload without dropping source rows."""
        payload: dict[str, object] = {
            "format_version": PARTITION_FORMAT,
            "source_ir_sha256": self.source_ir_sha256,
            "master": {
                "variable_keys": [list(key) for key in self.master_variables],
                "row_names": [row.name for row in self.master_rows],
                "gate_signatures": [
                    [gate.gate_id, gate.level_id, gate.axis, gate.role]
                    for gate in self.master_gates
                ],
            },
            "inner": [
                {
                    "outer_fold": group,
                    "variable_keys": [list(key) for key in variables],
                    "row_names": [row.name for row in dict(self.inner_rows)[group]],
                    "gate_signatures": [
                        [gate.gate_id, gate.level_id, gate.axis, gate.role]
                        for gate in dict(self.inner_gates)[group]
                    ],
                }
                for group, variables in self.inner_variables
            ],
            "implementation_source_sha256": implementation_source_sha256(),
            "training_authorized": False,
        }
        payload["partition_sha256"] = _sha256(payload)
        return payload


def partition_joint_ir(ir: CanonicalJointIR) -> V6BIRPartition:
    """Partition every row and gate exactly once and verify lossless reconstruction."""
    master_rows = []
    inner_rows: dict[int, list[IRLinearConstraint]] = {
        group: [] for group in range(EXPECTED_INNER_GROUPS)
    }
    for row in ir.linear_constraints:
        scope, group = _row_scope(row)
        if scope == "outer_master":
            master_rows.append(row)
        else:
            assert group is not None
            inner_rows[group].append(row)

    master_gates = []
    inner_gates: dict[int, list[ExactSeparationGate]] = {
        group: [] for group in range(EXPECTED_INNER_GROUPS)
    }
    for gate in ir.separation_gates:
        scope, group = _gate_scope(gate)
        if scope == "outer_master":
            master_gates.append(gate)
        else:
            assert group is not None
            inner_gates[group].append(gate)

    if len(master_gates) != EXPECTED_OUTER_GATES:
        raise V6BDecompositionError("outer gate partition is incomplete")
    if any(
        len(inner_gates[group]) != EXPECTED_INNER_GATES_PER_GROUP
        for group in range(EXPECTED_INNER_GROUPS)
    ):
        raise V6BDecompositionError("inner gate partition is incomplete")
    if len(ir.separation_gates) != EXPECTED_TOTAL_GATES:
        raise V6BDecompositionError("source exact-gate registry is not complete")

    master_variables = tuple(
        key for key in ir.variable_keys if key[0] == "outer"
    )
    grouped_variables = []
    for group in range(EXPECTED_INNER_GROUPS):
        referenced = {
            key
            for row in inner_rows[group]
            for key, _ in row.coefficients
        }
        expected = {
            key
            for key in ir.variable_keys
            if key[0] == "inner" and _outer_index(key) == group
        }
        if not expected <= referenced:
            raise V6BDecompositionError("an inner variable is absent from its rows")
        if any(key[0] == "outer" and _outer_index(key) != group for key in referenced):
            raise V6BDecompositionError("an inner group references a foreign outer state")
        grouped_variables.append((group, tuple(sorted(referenced))))

    partitioned_rows = tuple(master_rows) + tuple(
        row for group in range(EXPECTED_INNER_GROUPS) for row in inner_rows[group]
    )
    if sorted(row.name for row in partitioned_rows) != [
        row.name for row in ir.linear_constraints
    ]:
        raise V6BDecompositionError("row partition is not lossless")
    partitioned_gates = tuple(master_gates) + tuple(
        gate for group in range(EXPECTED_INNER_GROUPS) for gate in inner_gates[group]
    )
    if sorted(
        (gate.gate_id, gate.level_id, gate.axis, gate.role)
        for gate in partitioned_gates
    ) != [
        (gate.gate_id, gate.level_id, gate.axis, gate.role)
        for gate in ir.separation_gates
    ]:
        raise V6BDecompositionError("gate partition is not lossless")
    return V6BIRPartition(
        source_ir_sha256=ir.ir_sha256,
        master_variables=master_variables,
        master_rows=tuple(master_rows),
        master_gates=tuple(master_gates),
        inner_variables=tuple(grouped_variables),
        inner_rows=tuple(
            (group, tuple(inner_rows[group]))
            for group in range(EXPECTED_INNER_GROUPS)
        ),
        inner_gates=tuple(
            (group, tuple(inner_gates[group]))
            for group in range(EXPECTED_INNER_GROUPS)
        ),
    )


def _row_family(row: IRLinearConstraint) -> str:
    scope, group = _row_scope(row)
    variable_roles = sorted({key[-2] for key, _ in row.coefficients})
    bound_shape = (
        "eq"
        if row.lower_bound is not None and row.lower_bound == row.upper_bound
        else "range"
        if row.lower_bound is not None and row.upper_bound is not None
        else "lower"
        if row.lower_bound is not None
        else "upper"
    )
    group_label = "all" if group is None else str(group)
    return "|".join((scope, group_label, "+".join(variable_roles), bound_shape))


def build_row_shard_telemetry(
    ir: CanonicalJointIR,
    *,
    serialized_full_model_bytes: bytes,
    available_memory_bytes: int,
    observed_peak_rss_bytes: int,
    fraction_numerator: int = 1,
    fraction_denominator: int = 10,
) -> dict[str, object]:
    """Select deterministic row metadata and report non-authorizing scale evidence."""
    numerator = _positive_integer(fraction_numerator, "fraction numerator")
    denominator = _positive_integer(fraction_denominator, "fraction denominator")
    if numerator > denominator:
        raise V6BDecompositionError("row fraction exceeds one")
    available = _positive_integer(available_memory_bytes, "available memory")
    peak = _positive_integer(observed_peak_rss_bytes, "observed peak RSS")
    if not isinstance(serialized_full_model_bytes, bytes) or not serialized_full_model_bytes:
        raise V6BDecompositionError("serialized full model bytes are required")

    families: dict[str, list[tuple[str, IRLinearConstraint]]] = {}
    for row in ir.linear_constraints:
        digest = _sha256(row.canonical_payload())
        families.setdefault(_row_family(row), []).append((digest, row))

    selected_rows = []
    family_payloads = []
    for family, rows in sorted(families.items()):
        canonical = sorted(rows, key=lambda item: (item[0], item[1].name))
        count = -(-(len(canonical) * numerator) // denominator)
        selected = canonical[:count]
        selected_rows.extend(row for _, row in selected)
        family_payloads.append(
            {
                "family": family,
                "full_row_count": len(canonical),
                "selected_row_count": len(selected),
                "selected_row_sha256": [digest for digest, _ in selected],
            }
        )

    full_nonzeros = sum(len(row.coefficients) for row in ir.linear_constraints)
    selected_nonzeros = sum(len(row.coefficients) for row in selected_rows)
    variable_count = len(ir.variable_keys)
    row_count = len(ir.linear_constraints)
    model_bytes = len(serialized_full_model_bytes)
    coo_bytes = full_nonzeros * 24
    csc_bytes = full_nonzeros * 16 + (variable_count + 1) * 8
    projected_peak = max(peak, model_bytes + coo_bytes + csc_bytes)
    required_with_headroom = (projected_peak * 4 + 2) // 3
    resource_headroom_passed = available >= required_with_headroom
    payload: dict[str, object] = {
        "format_version": TELEMETRY_FORMAT,
        "source_ir_sha256": ir.ir_sha256,
        "full_variable_dictionary_sha256": _sha256(
            [list(key) for key in ir.variable_keys]
        ),
        "full_variable_count": variable_count,
        "full_row_count": row_count,
        "full_nonzero_count": full_nonzeros,
        "serialized_full_model_sha256": hashlib.sha256(
            serialized_full_model_bytes
        ).hexdigest(),
        "serialized_full_model_bytes": model_bytes,
        "estimated_full_coo_bytes": coo_bytes,
        "estimated_full_csc_bytes": csc_bytes,
        "observed_peak_rss_bytes": peak,
        "projected_peak_bytes": projected_peak,
        "available_memory_bytes": available,
        "minimum_headroom_fraction": [1, 4],
        "required_available_bytes_with_headroom": required_with_headroom,
        "resource_headroom_passed": resource_headroom_passed,
        "row_selection": {
            "fraction_numerator": numerator,
            "fraction_denominator": denominator,
            "rounding": "ceiling_per_declared_family",
            "order": "lowercase_sha256_of_canonical_row_payload",
            "families": family_payloads,
            "selected_row_count": len(selected_rows),
            "selected_nonzero_count": selected_nonzeros,
            "selected_row_name_sha256": _sha256(
                sorted(row.name for row in selected_rows)
            ),
        },
        "full_variables_constructed": True,
        "full_row_metadata_constructed": True,
        "vertices_or_pairs_sampled": False,
        "solver_invoked": False,
        "feasibility_authorized": False,
        "infeasibility_authorized": False,
        "optimality_authorized": False,
        "split_gate0_authorized": False,
        "training_authorized": False,
        "implementation_source_sha256": implementation_source_sha256(),
    }
    payload["telemetry_sha256"] = _sha256(payload)
    return payload


def selected_row_hashes(telemetry: Mapping[str, object]) -> tuple[str, ...]:
    """Return every selected row digest after validating canonical uniqueness."""
    selection = telemetry.get("row_selection")
    if not isinstance(selection, Mapping):
        raise V6BDecompositionError("telemetry row selection is malformed")
    families = selection.get("families")
    if not isinstance(families, Sequence) or isinstance(families, (str, bytes)):
        raise V6BDecompositionError("telemetry families are malformed")
    result = []
    for family in families:
        if not isinstance(family, Mapping):
            raise V6BDecompositionError("telemetry family is malformed")
        digests = family.get("selected_row_sha256")
        if not isinstance(digests, Sequence) or isinstance(digests, (str, bytes)):
            raise V6BDecompositionError("selected row hashes are malformed")
        result.extend(str(value) for value in digests)
    if len(result) != len(set(result)):
        raise V6BDecompositionError("selected row hashes are duplicated")
    return tuple(result)


@dataclass(frozen=True)
class SignedSupportLiteral:
    """One outer-master Boolean state used by an inner infeasibility proof."""

    key: VariableKey
    value: int

    def __post_init__(self) -> None:
        if not self.key or self.key[0] != "outer":
            raise V6BDecompositionError("support literal must reference an outer variable")
        _outer_index(self.key)
        if self.value not in {0, 1}:
            raise V6BDecompositionError("support literal value must be binary")

    def canonical_payload(self) -> list[object]:
        return [list(self.key), self.value]


@dataclass(frozen=True)
class V6BMasterNoGoodCut:
    """One replayed logic-based cut over signed outer-state literals."""

    outer_fold: int
    support: tuple[SignedSupportLiteral, ...]
    first_response_sha256: str
    replay_response_sha256: str
    independent_replay_sha256: str
    cut_sha256: str

    def canonical_payload(self) -> dict[str, object]:
        return {
            "outer_fold": self.outer_fold,
            "support": [literal.canonical_payload() for literal in self.support],
            "first_response_sha256": self.first_response_sha256,
            "replay_response_sha256": self.replay_response_sha256,
            "independent_replay_sha256": self.independent_replay_sha256,
            "cut_sha256": self.cut_sha256,
        }


def _configure_solver(solver: object, maximum_time_seconds: float | None) -> None:
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = LOCKED_RANDOM_SEED
    solver.parameters.randomize_search = False
    solver.parameters.relative_gap_limit = 0.0
    solver.parameters.absolute_gap_limit = 0.0
    if maximum_time_seconds is not None:
        if maximum_time_seconds <= 0:
            raise V6BDecompositionError("CP-SAT time limit must be positive")
        solver.parameters.max_time_in_seconds = maximum_time_seconds


def _add_rows(
    model: object,
    variables: Mapping[VariableKey, object],
    rows: Iterable[IRLinearConstraint],
) -> None:
    for row in rows:
        expression = sum(
            coefficient * variables[key] for key, coefficient in row.coefficients
        )
        if row.lower_bound is not None:
            model.add(expression >= row.lower_bound)
        if row.upper_bound is not None:
            model.add(expression <= row.upper_bound)


def _add_no_good(
    model: object,
    variables: Mapping[VariableKey, object],
    support: Sequence[SignedSupportLiteral],
) -> None:
    if not support:
        raise V6BDecompositionError("no-good support cannot be empty")
    unknown = {literal.key for literal in support} - set(variables)
    if unknown:
        raise V6BDecompositionError("no-good support references an unknown variable")
    model.add(
        sum(
            variables[literal.key] if literal.value == 0 else 1 - variables[literal.key]
            for literal in support
        )
        >= 1
    )


def _audit_rows(
    rows: Iterable[IRLinearConstraint],
    assignment: Mapping[VariableKey, int],
) -> bool:
    if any(value not in {0, 1} for value in assignment.values()):
        return False
    for row in rows:
        if {key for key, _ in row.coefficients} - set(assignment):
            return False
        value = sum(
            coefficient * assignment[key] for key, coefficient in row.coefficients
        )
        if row.lower_bound is not None and value < row.lower_bound:
            return False
        if row.upper_bound is not None and value > row.upper_bound:
            return False
    return True


def _audit_no_good(
    support: Sequence[SignedSupportLiteral],
    assignment: Mapping[VariableKey, int],
) -> bool:
    return any(assignment[literal.key] != literal.value for literal in support)


def _response_payload(solver: object, status_name: str) -> dict[str, object]:
    payload: dict[str, object] = {
        "status": status_name,
        "wall_time_seconds": str(solver.wall_time),
        "user_time_seconds": str(solver.user_time),
        "branches": int(solver.num_branches),
        "conflicts": int(solver.num_conflicts),
        "objective_value": str(solver.objective_value),
        "best_objective_bound": str(solver.best_objective_bound),
        "response_stats": str(solver.response_stats()),
    }
    payload["response_sha256"] = _sha256(payload)
    return payload


def solve_outer_master_linear_projection(
    partition: V6BIRPartition,
    *,
    cuts: Sequence[V6BMasterNoGoodCut] = (),
    maximum_time_seconds: float | None = None,
) -> dict[str, object]:
    """Solve the outer integer projection and reject incomplete CP-SAT states."""
    from ortools.sat.python import cp_model
    import ortools

    model = cp_model.CpModel()
    variables = {
        key: model.new_bool_var(f"m_{index:09d}")
        for index, key in enumerate(partition.master_variables)
    }
    _add_rows(model, variables, partition.master_rows)
    for cut in cuts:
        _add_no_good(model, variables, cut.support)
    model.minimize(0)
    solver = cp_model.CpSolver()
    _configure_solver(solver, maximum_time_seconds)
    status = solver.solve(model)
    status_name = solver.status_name(status)
    response = _response_payload(solver, status_name)
    assignment = None
    if status == cp_model.OPTIMAL:
        assignment = {
            key: int(solver.value(variable)) for key, variable in variables.items()
        }
        if not _audit_rows(partition.master_rows, assignment) or any(
            not _audit_no_good(cut.support, assignment) for cut in cuts
        ):
            raise V6BDecompositionError("outer integer row audit disagrees")
    payload: dict[str, object] = {
        "solver": "OR-Tools CP-SAT",
        "ortools_version": ortools.__version__,
        "status": status_name,
        "accepted_terminal_status": status in {cp_model.OPTIMAL, cp_model.INFEASIBLE},
        "candidate_present": assignment is not None,
        "integer_row_audit_passed": assignment is not None,
        "response": response,
        "assignment": (
            None
            if assignment is None
            else [[list(key), assignment[key]] for key in partition.master_variables]
        ),
        "cut_sha256": [cut.cut_sha256 for cut in cuts],
        "num_search_workers": 1,
        "random_seed": LOCKED_RANDOM_SEED,
        "complete_score_free_feasible_region_authorized": False,
        "training_authorized": False,
    }
    payload["receipt_sha256"] = _sha256(payload)
    return payload


def _inner_model_with_assumptions(
    partition: V6BIRPartition,
    outer_fold: int,
    support: Sequence[SignedSupportLiteral],
) -> tuple[object, dict[VariableKey, object], dict[int, SignedSupportLiteral]]:
    from ortools.sat.python import cp_model

    if outer_fold not in range(EXPECTED_INNER_GROUPS):
        raise V6BDecompositionError("inner outer-fold index is outside range")
    variables_for_group = dict(partition.inner_variables)[outer_fold]
    rows = dict(partition.inner_rows)[outer_fold]
    model = cp_model.CpModel()
    variables = {
        key: model.new_bool_var(f"s_{index:09d}")
        for index, key in enumerate(variables_for_group)
    }
    _add_rows(model, variables, rows)
    assumptions: dict[int, SignedSupportLiteral] = {}
    for index, literal in enumerate(support):
        if literal.key not in variables or _outer_index(literal.key) != outer_fold:
            raise V6BDecompositionError("inner assumption is outside its boundary")
        assumption = model.new_bool_var(f"a_{index:09d}")
        model.add(variables[literal.key] == literal.value).only_enforce_if(assumption)
        model.add_assumption(assumption)
        assumptions[int(assumption.index)] = literal
    model.minimize(0)
    return model, variables, assumptions


def solve_inner_linear_projection(
    partition: V6BIRPartition,
    *,
    outer_fold: int,
    outer_assignment: Mapping[VariableKey, int],
    maximum_time_seconds: float | None = None,
) -> dict[str, object]:
    """Solve one inner group under signed outer assumptions and return its core."""
    from ortools.sat.python import cp_model
    import ortools

    boundary_keys = tuple(
        key
        for key in dict(partition.inner_variables)[outer_fold]
        if key[0] == "outer"
    )
    if set(outer_assignment) != set(partition.master_variables):
        raise V6BDecompositionError("outer assignment does not cover the master")
    if any(value not in {0, 1} for value in outer_assignment.values()):
        raise V6BDecompositionError("outer assignment is not binary")
    support = tuple(
        SignedSupportLiteral(key, outer_assignment[key]) for key in boundary_keys
    )
    model, variables, assumptions = _inner_model_with_assumptions(
        partition,
        outer_fold,
        support,
    )
    solver = cp_model.CpSolver()
    _configure_solver(solver, maximum_time_seconds)
    status = solver.solve(model)
    status_name = solver.status_name(status)
    response = _response_payload(solver, status_name)
    assignment = None
    core: tuple[SignedSupportLiteral, ...] = ()
    if status == cp_model.OPTIMAL:
        assignment = {
            key: int(solver.value(variable)) for key, variable in variables.items()
        }
        if not _audit_rows(dict(partition.inner_rows)[outer_fold], assignment):
            raise V6BDecompositionError("inner integer row audit disagrees")
        if any(assignment[literal.key] != literal.value for literal in support):
            raise V6BDecompositionError("inner boundary audit disagrees")
    elif status == cp_model.INFEASIBLE:
        raw_core = tuple(int(value) for value in solver.sufficient_assumptions_for_infeasibility())
        try:
            core = tuple(sorted((assumptions[index] for index in raw_core), key=lambda item: item.key))
        except KeyError as error:
            raise V6BDecompositionError(
                "CP-SAT returned an unknown or signed assumption index"
            ) from error
        if not core:
            raise V6BDecompositionError("infeasible inner model returned an empty core")
    payload: dict[str, object] = {
        "solver": "OR-Tools CP-SAT",
        "ortools_version": ortools.__version__,
        "outer_fold": outer_fold,
        "status": status_name,
        "accepted_terminal_status": status in {cp_model.OPTIMAL, cp_model.INFEASIBLE},
        "candidate_present": assignment is not None,
        "integer_row_audit_passed": assignment is not None,
        "response": response,
        "assumption_count": len(support),
        "sufficient_assumption_support": [
            literal.canonical_payload() for literal in core
        ],
        "assignment": (
            None
            if assignment is None
            else [
                [list(key), assignment[key]]
                for key in dict(partition.inner_variables)[outer_fold]
            ]
        ),
        "complete_score_free_feasible_region_authorized": False,
        "training_authorized": False,
    }
    payload["receipt_sha256"] = _sha256(payload)
    return payload


def replay_inner_infeasibility_support(
    partition: V6BIRPartition,
    *,
    outer_fold: int,
    support: Sequence[SignedSupportLiteral],
    maximum_time_seconds: float | None = None,
) -> dict[str, object]:
    """Rebuild an inner model with only the proposed support fixed and replay it."""
    from ortools.sat.python import cp_model
    import ortools

    canonical = tuple(sorted(support, key=lambda literal: literal.key))
    if not canonical or len({literal.key for literal in canonical}) != len(canonical):
        raise V6BDecompositionError("replay support must be non-empty and unique")
    model, _, _ = _inner_model_with_assumptions(partition, outer_fold, canonical)
    solver = cp_model.CpSolver()
    _configure_solver(solver, maximum_time_seconds)
    status = solver.solve(model)
    status_name = solver.status_name(status)
    response = _response_payload(solver, status_name)
    payload: dict[str, object] = {
        "solver": "OR-Tools CP-SAT",
        "ortools_version": ortools.__version__,
        "outer_fold": outer_fold,
        "support": [literal.canonical_payload() for literal in canonical],
        "support_sha256": _sha256(
            [literal.canonical_payload() for literal in canonical]
        ),
        "status": status_name,
        "replay_infeasible": status == cp_model.INFEASIBLE,
        "accepted_terminal_status": status in {cp_model.OPTIMAL, cp_model.INFEASIBLE},
        "response": response,
        "training_authorized": False,
    }
    payload["receipt_sha256"] = _sha256(payload)
    return payload


def build_replayed_master_no_good_cut(
    *,
    outer_fold: int,
    support: Sequence[SignedSupportLiteral],
    first_response_sha256: str,
    replay_response_sha256: str,
    independent_replay_receipt: Mapping[str, object],
) -> V6BMasterNoGoodCut:
    """Create a master cut only after both replay evidence hashes are present."""
    canonical = tuple(sorted(support, key=lambda literal: literal.key))
    if not canonical or len({literal.key for literal in canonical}) != len(canonical):
        raise V6BDecompositionError("cut support must be non-empty and unique")
    expected_support = [literal.canonical_payload() for literal in canonical]
    independent_payload = dict(independent_replay_receipt)
    independent_sha256 = independent_payload.pop("receipt_sha256", None)
    if (
        independent_replay_receipt.get("format_version")
        != "v6_b_independent_inner_core_replay_v1"
        or independent_replay_receipt.get("outer_fold") != outer_fold
        or independent_replay_receipt.get("support") != expected_support
        or independent_replay_receipt.get("status") != "INFEASIBLE"
        or independent_replay_receipt.get("replay_infeasible") is not True
        or independent_replay_receipt.get("accepted_terminal_status") is not True
        or independent_replay_receipt.get("v6_b_decomposition_imported") is not False
        or independent_replay_receipt.get("historical_milp_or_highs_artifact_consumed")
        is not False
        or independent_replay_receipt.get("training_authorized") is not False
    ):
        raise V6BDecompositionError("independent replay cannot support a master cut")
    if independent_sha256 != _sha256(independent_payload):
        raise V6BDecompositionError("independent replay self hash differs")
    for label, digest in (
        ("first response", first_response_sha256),
        ("replay response", replay_response_sha256),
        ("independent replay", independent_sha256),
    ):
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise V6BDecompositionError(f"{label} must be a lowercase SHA-256")
    base: dict[str, object] = {
        "outer_fold": outer_fold,
        "support": [literal.canonical_payload() for literal in canonical],
        "first_response_sha256": first_response_sha256,
        "replay_response_sha256": replay_response_sha256,
        "independent_replay_sha256": independent_sha256,
    }
    return V6BMasterNoGoodCut(
        outer_fold=outer_fold,
        support=canonical,
        first_response_sha256=first_response_sha256,
        replay_response_sha256=replay_response_sha256,
        independent_replay_sha256=independent_sha256,
        cut_sha256=_sha256(base),
    )
