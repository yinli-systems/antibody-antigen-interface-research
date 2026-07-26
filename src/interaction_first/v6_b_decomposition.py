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
