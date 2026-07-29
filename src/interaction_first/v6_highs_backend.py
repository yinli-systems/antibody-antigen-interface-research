"""Direct-highspy linear-projection backend for the V6-A canonical IR."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from pathlib import Path
from typing import Mapping

from .v6_joint_ir import CanonicalJointIR, VariableKey, audit_integer_constraints


RECEIPT_FORMAT = "v6_a_highs_linear_projection_receipt_v1"
LOCKED_OPTIONS = {
    "threads": 1,
    "parallel": "off",
    "random_seed": 2026072701,
    "mip_rel_gap": 0.0,
    "mip_abs_gap": 0.0,
    "mip_feasibility_tolerance": 1e-7,
    "primal_feasibility_tolerance": 1e-7,
    "dual_feasibility_tolerance": 1e-7,
}


class HighsProjectionError(RuntimeError):
    """Raised when direct highspy cannot publish a valid projection receipt."""


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_exclusive(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o400)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _check_status(status: object, highspy: object, label: str) -> str:
    if status != highspy.HighsStatus.kOk:
        raise HighsProjectionError(f"highspy {label} failed with {status}")
    return str(status)


def _check_export_status(
    status: object,
    highspy: object,
    label: str,
    output_path: Path,
) -> str:
    """Accept an explicit export warning only when a non-empty artifact exists."""
    if status not in {highspy.HighsStatus.kOk, highspy.HighsStatus.kWarning}:
        raise HighsProjectionError(f"highspy {label} failed with {status}")
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise HighsProjectionError(
            f"highspy {label} did not create a non-empty artifact"
        )
    return str(status)


def _info_payload(info: object) -> dict[str, object]:
    names = (
        "valid",
        "mip_node_count",
        "mip_dual_bound",
        "mip_gap",
        "max_integrality_violation",
        "max_primal_infeasibility",
        "num_primal_infeasibilities",
        "objective_function_value",
        "primal_solution_status",
        "dual_solution_status",
        "simplex_iteration_count",
        "ipm_iteration_count",
    )
    payload: dict[str, object] = {}
    for name in names:
        value = getattr(info, name)
        if isinstance(value, float):
            payload[name] = str(value) if math.isfinite(value) else None
        elif hasattr(value, "value"):
            payload[name] = int(value.value)
        else:
            payload[name] = value
    return payload


def solve_highs_linear_projection(
    ir: CanonicalJointIR,
    output_directory: Path,
) -> dict[str, object]:
    """Solve and publish a create-only non-authorizing HiGHS projection receipt."""
    try:
        import highspy
        import numpy as np
    except ImportError as error:
        raise HighsProjectionError("highspy and NumPy are required") from error

    output = Path(os.path.abspath(os.path.expanduser(str(output_directory))))
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"create-only output already exists: {output}")
    output.parent.mkdir(parents=False, exist_ok=True)
    os.mkdir(output, 0o700)
    model_path = output / "original_model.mps"
    presolved_path = output / "presolved_model.mps"
    options_path = output / "highs_options.txt"
    log_path = output / "highs.log"
    solution_path = output / "solution.txt"
    info_path = output / "highs_info.txt"
    variable_path = output / "variable_dictionary.json"
    telemetry_path = output / "mip_logging_trajectory.json"
    receipt_path = output / "receipt.json"

    variable_index = {key: index for index, key in enumerate(ir.variable_keys)}
    variable_payload = {
        "format_version": "v6_a_variable_dictionary_v1",
        "ir_sha256": ir.ir_sha256,
        "variables": [
            {"index": index, "key": list(key)}
            for index, key in enumerate(ir.variable_keys)
        ],
    }
    _write_exclusive(variable_path, _canonical_json_bytes(variable_payload))

    column_count = len(ir.variable_keys)
    row_count = len(ir.linear_constraints)
    row_lower = np.array(
        [
            -highspy.kHighsInf
            if row.lower_bound is None
            else float(row.lower_bound)
            for row in ir.linear_constraints
        ],
        dtype=np.float64,
    )
    row_upper = np.array(
        [
            highspy.kHighsInf
            if row.upper_bound is None
            else float(row.upper_bound)
            for row in ir.linear_constraints
        ],
        dtype=np.float64,
    )
    starts = [0]
    indices: list[int] = []
    values: list[float] = []
    for row in ir.linear_constraints:
        for key, coefficient in row.coefficients:
            indices.append(variable_index[key])
            values.append(float(coefficient))
        starts.append(len(indices))
    highs = highspy.Highs()
    _check_status(
        highs.addVars(
            column_count,
            np.zeros(column_count, dtype=np.float64),
            np.ones(column_count, dtype=np.float64),
        ),
        highspy,
        "addVars",
    )
    _check_status(
        highs.changeColsIntegrality(
            column_count,
            np.arange(column_count, dtype=np.int32),
            np.full(
                column_count,
                highspy.HighsVarType.kInteger.value,
                dtype=np.uint8,
            ),
        ),
        highspy,
        "changeColsIntegrality",
    )
    _check_status(
        highs.addRows(
            row_count,
            row_lower,
            row_upper,
            len(indices),
            np.array(starts, dtype=np.int32),
            np.array(indices, dtype=np.int32),
            np.array(values, dtype=np.float64),
        ),
        highspy,
        "addRows",
    )
    for name, value in LOCKED_OPTIONS.items():
        _check_status(highs.setOptionValue(name, value), highspy, f"option {name}")
    _check_status(highs.setOptionValue("log_file", str(log_path)), highspy, "log_file")
    _check_status(highs.setOptionValue("log_to_console", False), highspy, "console log")
    export_statuses = {
        "write_model": _check_export_status(
            highs.writeModel(str(model_path)),
            highspy,
            "writeModel",
            model_path,
        ),
        "write_options": _check_export_status(
            highs.writeOptions(str(options_path)),
            highspy,
            "writeOptions",
            options_path,
        ),
    }

    telemetry: list[dict[str, object]] = []

    def record_mip_log(event: object) -> None:
        data = event.data_out
        telemetry.append(
            {
                "running_time": str(data.running_time),
                "mip_node_count": int(data.mip_node_count),
                "mip_primal_bound": str(data.mip_primal_bound),
                "mip_dual_bound": str(data.mip_dual_bound),
                "mip_gap": str(data.mip_gap),
            }
        )

    highs.callbacks[5].subscribe(record_mip_log)
    _check_status(highs.presolve(), highspy, "presolve")
    presolved_lp = highs.getPresolvedLp()
    export_statuses["write_presolved_model"] = _check_export_status(
        highs.writePresolvedModel(str(presolved_path)),
        highspy,
        "writePresolvedModel",
        presolved_path,
    )
    run_status = highs.run()
    _check_status(run_status, highspy, "run")
    model_status = highs.getModelStatus()
    info = highs.getInfo()
    solution = highs.getSolution()
    export_statuses["write_solution"] = _check_export_status(
        highs.writeSolution(str(solution_path), 1),
        highspy,
        "writeSolution",
        solution_path,
    )
    export_statuses["write_info"] = _check_export_status(
        highs.writeInfo(str(info_path)),
        highspy,
        "writeInfo",
        info_path,
    )
    _write_exclusive(telemetry_path, _canonical_json_bytes(telemetry))

    assignment: dict[VariableKey, int] | None = None
    exact_row_audit = False
    maximum_rounding_error: float | None = None
    if model_status == highspy.HighsModelStatus.kOptimal:
        column_values = tuple(float(value) for value in solution.col_value)
        maximum_rounding_error = max(
            abs(value - round(value)) for value in column_values
        )
        assignment = {
            key: int(round(column_values[index]))
            for index, key in enumerate(ir.variable_keys)
        }
        exact_row_audit = audit_integer_constraints(ir, assignment)
    gap = float(info.mip_gap)
    zero_gap = math.isfinite(gap) and gap == 0.0
    optimal_projection = (
        model_status == highspy.HighsModelStatus.kOptimal
        and zero_gap
        and maximum_rounding_error is not None
        and maximum_rounding_error <= LOCKED_OPTIONS["mip_feasibility_tolerance"]
        and exact_row_audit
    )
    infeasible_projection = model_status == highspy.HighsModelStatus.kInfeasible
    projection_certified = optimal_projection or infeasible_projection
    artifact_paths = [
        model_path,
        presolved_path,
        options_path,
        log_path,
        solution_path,
        info_path,
        variable_path,
        telemetry_path,
    ]
    artifacts = {
        path.name: {
            "sha256": _sha256_file(path),
            "byte_size": path.stat().st_size,
        }
        for path in artifact_paths
    }
    receipt: dict[str, object] = {
        "format_version": RECEIPT_FORMAT,
        "ir_sha256": ir.ir_sha256,
        "highspy_version": highs.version(),
        "locked_options": LOCKED_OPTIONS,
        "export_statuses": export_statuses,
        "original_dimensions": {
            "columns": column_count,
            "rows": row_count,
            "nonzeros": len(indices),
        },
        "presolved_dimensions": {
            "columns": int(presolved_lp.num_col_),
            "rows": int(presolved_lp.num_row_),
        },
        "model_status": str(model_status),
        "highs_info": _info_payload(info),
        "mip_gap_is_finite_zero": zero_gap,
        "maximum_integrality_rounding_error": (
            None if maximum_rounding_error is None else str(maximum_rounding_error)
        ),
        "exact_integer_row_audit_passed": exact_row_audit,
        "linear_projection_certified": projection_certified,
        "complete_score_free_feasible_region_certified": False,
        "nonlinear_smd_domain_authorized_by_this_receipt": False,
        "training_authorized": False,
        "model_selection_authorized": False,
        "outer_evaluation_authorized": False,
        "user_solution_callback_used": False,
        "mip_logging_callback_used": True,
        "artifacts": artifacts,
    }
    receipt["receipt_sha256"] = hashlib.sha256(
        _canonical_json_bytes(receipt)
    ).hexdigest()
    _write_exclusive(receipt_path, _canonical_json_bytes(receipt))
    for path in output.iterdir():
        os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    os.chmod(
        output,
        stat.S_IRUSR
        | stat.S_IXUSR
        | stat.S_IRGRP
        | stat.S_IXGRP
        | stat.S_IROTH
        | stat.S_IXOTH,
    )
    return receipt
