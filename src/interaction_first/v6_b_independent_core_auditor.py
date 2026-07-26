"""Independent direct-proto replay for V6-B inner infeasibility supports.

This module intentionally does not import the V6-B decomposition implementation,
the solver-neutral IR, the exact statistics oracle, or a HiGHS matrix. It selects and
rebuilds one inner group's integer rows directly from the independently constructed
CP-SAT proto and canonical variable registry.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence


FORMAT_VERSION = "v6_b_independent_inner_core_replay_v1"
LOCKED_RANDOM_SEED = 2026072701


class IndependentCoreAuditError(ValueError):
    """Raised when independent proto reconstruction cannot support a cut."""


def _canonical_json_bytes(payload: object) -> bytes:
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
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def implementation_source_sha256() -> str:
    """Hash the exact independent replay implementation bytes."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _key(value: object, label: str) -> tuple[str, ...]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or not value
        or any(not isinstance(item, str) or not item or not item.isascii() for item in value)
    ):
        raise IndependentCoreAuditError(f"{label} is not a canonical variable key")
    return tuple(value)


def _outer_index(key: tuple[str, ...]) -> int:
    if key[0] not in {"outer", "inner"} or len(key) < 2:
        raise IndependentCoreAuditError("variable lies outside the outer/inner schema")
    try:
        result = int(key[1])
    except ValueError as error:
        raise IndependentCoreAuditError("outer index is not an integer") from error
    if str(result) != key[1] or result not in range(3):
        raise IndependentCoreAuditError("outer index is outside the frozen range")
    return result


def _support(
    values: Sequence[Sequence[object]],
    outer_fold: int,
) -> tuple[tuple[tuple[str, ...], int], ...]:
    result = []
    for item in values:
        if isinstance(item, (str, bytes)) or not isinstance(item, Sequence) or len(item) != 2:
            raise IndependentCoreAuditError("support item shape differs")
        key = _key(item[0], "support key")
        value = item[1]
        if key[0] != "outer" or _outer_index(key) != outer_fold:
            raise IndependentCoreAuditError("support is outside the target boundary")
        if value not in {0, 1}:
            raise IndependentCoreAuditError("support value is not binary")
        result.append((key, int(value)))
    canonical = tuple(sorted(result))
    if not canonical or len({key for key, _ in canonical}) != len(canonical):
        raise IndependentCoreAuditError("support is empty or duplicated")
    return canonical


def replay_direct_inner_support(
    direct_model: object,
    *,
    outer_fold: int,
    signed_support: Sequence[Sequence[object]],
    maximum_time_seconds: float | None = None,
) -> dict[str, object]:
    """Rebuild and replay one inner group directly from a direct CP-SAT proto."""
    from ortools.sat.python import cp_model
    import ortools

    if outer_fold not in range(3):
        raise IndependentCoreAuditError("outer fold is outside the frozen range")
    support = _support(signed_support, outer_fold)
    try:
        source_proto = direct_model.model.Proto()
        source_variables = direct_model.variables
        formulation_sha256 = str(direct_model.formulation_sha256)
        authority_sha256 = str(direct_model.authority_payload_sha256)
    except AttributeError as error:
        raise IndependentCoreAuditError("direct CP-SAT model contract is malformed") from error
    if not isinstance(source_variables, Mapping):
        raise IndependentCoreAuditError("direct variable registry is not a mapping")

    index_to_key: dict[int, tuple[str, ...]] = {}
    for raw_key, variable in source_variables.items():
        key = _key(raw_key, "direct variable")
        index = int(variable.index)
        if index < 0 or index in index_to_key:
            raise IndependentCoreAuditError("direct variable index is invalid")
        index_to_key[index] = key
    if set(index_to_key) != set(range(len(source_proto.variables))):
        raise IndependentCoreAuditError("direct variable registry does not cover proto")

    selected_rows = []
    referenced_indices = set()
    for source_index, constraint in enumerate(source_proto.constraints):
        if constraint.enforcement_literal or not constraint.has_linear():
            raise IndependentCoreAuditError("direct proto contains an unsupported row")
        linear = constraint.linear
        keys = [index_to_key[int(index)] for index in linear.vars]
        inner_groups = {
            _outer_index(key) for key in keys if key[0] == "inner"
        }
        if not inner_groups:
            continue
        if len(inner_groups) != 1:
            raise IndependentCoreAuditError("direct row mixes inner groups")
        group = next(iter(inner_groups))
        outer_groups = {_outer_index(key) for key in keys if key[0] == "outer"}
        if outer_groups - {group}:
            raise IndependentCoreAuditError("direct inner row references another outer group")
        if group != outer_fold:
            continue
        selected_rows.append((source_index, linear))
        referenced_indices.update(int(index) for index in linear.vars)
    if not selected_rows:
        raise IndependentCoreAuditError("target inner group has no integer rows")

    support_keys = {key for key, _ in support}
    direct_key_to_index = {key: index for index, key in index_to_key.items()}
    if support_keys - set(direct_key_to_index):
        raise IndependentCoreAuditError("support references an unknown direct variable")
    referenced_indices.update(direct_key_to_index[key] for key in support_keys)
    selected_indices = tuple(sorted(referenced_indices))
    model = cp_model.CpModel()
    rebuilt = {
        index: model.new_bool_var(f"i_{position:09d}")
        for position, index in enumerate(selected_indices)
    }
    negative_infinity = -(2**63)
    positive_infinity = 2**63 - 1
    row_payloads = []
    for source_index, linear in selected_rows:
        if len(linear.vars) != len(linear.coeffs) or len(linear.domain) != 2:
            raise IndependentCoreAuditError("direct linear row shape differs")
        expression = sum(
            int(coefficient) * rebuilt[int(variable_index)]
            for variable_index, coefficient in zip(
                linear.vars,
                linear.coeffs,
                strict=True,
            )
        )
        lower, upper = map(int, linear.domain)
        if lower != negative_infinity:
            model.add(expression >= lower)
        if upper != positive_infinity:
            model.add(expression <= upper)
        row_payloads.append(
            {
                "source_index": source_index,
                "variables": [
                    list(index_to_key[int(index)]) for index in linear.vars
                ],
                "coefficients": [int(value) for value in linear.coeffs],
                "domain": [lower, upper],
            }
        )
    for position, (key, value) in enumerate(support):
        variable = rebuilt[direct_key_to_index[key]]
        assumption = model.new_bool_var(f"a_{position:09d}")
        model.add(variable == value).only_enforce_if(assumption)
        model.add_assumption(assumption)
    model.minimize(0)
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = LOCKED_RANDOM_SEED
    solver.parameters.randomize_search = False
    solver.parameters.relative_gap_limit = 0.0
    solver.parameters.absolute_gap_limit = 0.0
    if maximum_time_seconds is not None:
        if maximum_time_seconds <= 0:
            raise IndependentCoreAuditError("CP-SAT time limit must be positive")
        solver.parameters.max_time_in_seconds = maximum_time_seconds
    status = solver.solve(model)
    status_name = solver.status_name(status)
    response = {
        "status": status_name,
        "wall_time_seconds": str(solver.wall_time),
        "branches": int(solver.num_branches),
        "conflicts": int(solver.num_conflicts),
        "response_stats": str(solver.response_stats()),
    }
    response["response_sha256"] = _sha256(response)
    payload: dict[str, object] = {
        "format_version": FORMAT_VERSION,
        "solver": "OR-Tools CP-SAT",
        "ortools_version": ortools.__version__,
        "outer_fold": outer_fold,
        "support": [[list(key), value] for key, value in support],
        "support_sha256": _sha256(
            [[list(key), value] for key, value in support]
        ),
        "status": status_name,
        "replay_infeasible": status == cp_model.INFEASIBLE,
        "accepted_terminal_status": status in {cp_model.OPTIMAL, cp_model.INFEASIBLE},
        "source_authority_sha256": authority_sha256,
        "source_formulation_sha256": formulation_sha256,
        "selected_row_count": len(selected_rows),
        "selected_variable_count": len(selected_indices),
        "selected_rows_sha256": _sha256(row_payloads),
        "response": response,
        "direct_cp_sat_proto_reconstructed": True,
        "v6_b_decomposition_imported": False,
        "historical_milp_or_highs_artifact_consumed": False,
        "implementation_source_sha256": implementation_source_sha256(),
        "training_authorized": False,
    }
    payload["receipt_sha256"] = _sha256(payload)
    return payload
