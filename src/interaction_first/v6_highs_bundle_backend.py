"""Isolated direct-highspy backend for authenticated sparse V6 bundles."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import sys
from pathlib import Path

from .v6_linear_projection_bundle import load_linear_projection_bundle


RECEIPT_FORMAT = "v6_highs_solver_neutral_bundle_projection_receipt_v1"
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


class HighsBundleAuditError(RuntimeError):
    """Raised when direct HiGHS cannot certify the authenticated projection."""


def canonical_json_bytes(payload: object) -> bytes:
    """Encode one canonical ASCII JSON object."""
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
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_exclusive(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _require_status(status: object, highspy: object, label: str) -> None:
    if status != highspy.HighsStatus.kOk:
        raise HighsBundleAuditError(f"highspy {label} failed with {status}")


def _require_artifact(
    status: object,
    highspy: object,
    label: str,
    path: Path,
) -> str:
    if status not in {highspy.HighsStatus.kOk, highspy.HighsStatus.kWarning}:
        raise HighsBundleAuditError(f"highspy {label} failed with {status}")
    if not path.is_file() or path.stat().st_size == 0:
        raise HighsBundleAuditError(f"highspy {label} did not create an artifact")
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


def _audit_optimal_integer_solution(
    column_values: object,
    arrays: object,
    *,
    columns: int,
    rows: int,
    tolerance: float,
    numpy: object,
) -> tuple[bool, float | None]:
    """Round a binary solution and replay every sparse row with Python integers."""
    values = numpy.asarray(column_values, dtype=numpy.float64)
    if values.shape != (columns,) or not numpy.all(numpy.isfinite(values)):
        return False, None
    rounded = numpy.rint(values)
    maximum_error = float(numpy.max(numpy.abs(values - rounded))) if columns else 0.0
    if maximum_error > tolerance or numpy.any((rounded < 0) | (rounded > 1)):
        return False, maximum_error
    assignment = rounded.astype(numpy.int8)
    starts = arrays["row_starts"]
    indices = arrays["column_indices"]
    coefficients = arrays["coefficients"]
    for row in range(rows):
        start = int(starts[row])
        end = int(starts[row + 1])
        activity = sum(
            int(coefficients[index]) * int(assignment[int(indices[index])])
            for index in range(start, end)
        )
        if arrays["row_lower_present"][row] and activity < int(
            arrays["row_lower"][row]
        ):
            return False, maximum_error
        if arrays["row_upper_present"][row] and activity > int(
            arrays["row_upper"][row]
        ):
            return False, maximum_error
    return True, maximum_error


def solve_highs_projection_bundle(
    *,
    bundle_directory: Path,
    expected_bundle_sha256: str,
    output_directory: Path,
    maximum_time_seconds: float,
) -> dict[str, object]:
    """Load a solver-neutral bundle and publish a non-authorizing HiGHS receipt."""
    if maximum_time_seconds <= 0:
        raise HighsBundleAuditError("time limit must be positive")
    if any(name == "ortools" or name.startswith("ortools.") for name in sys.modules):
        raise HighsBundleAuditError("OR-Tools is loaded in the isolated HiGHS process")
    try:
        import highspy
        import numpy as np
    except ImportError as error:
        raise HighsBundleAuditError("highspy and NumPy are required") from error
    bundle = load_linear_projection_bundle(
        bundle_directory,
        expected_bundle_sha256=expected_bundle_sha256,
    )
    if any(name == "ortools" or name.startswith("ortools.") for name in sys.modules):
        raise HighsBundleAuditError("bundle loading imported OR-Tools")
    output = Path(os.path.abspath(os.path.expanduser(str(output_directory))))
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"create-only HiGHS output already exists: {output}")
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise HighsBundleAuditError("HiGHS output parent is absent or symbolic")
    os.mkdir(output, 0o700)
    model_path = output / "original_model.mps"
    presolved_path = output / "presolved_model.mps"
    options_path = output / "highs_options.txt"
    log_path = output / "highs.log"
    solution_path = output / "solution.txt"
    info_path = output / "highs_info.txt"
    telemetry_path = output / "mip_logging_trajectory.json"
    receipt_path = output / "receipt.json"

    dimensions = bundle.metadata["dimensions"]
    columns = int(dimensions["columns"])
    rows = int(dimensions["rows"])
    nonzeros = int(dimensions["nonzeros"])
    arrays = bundle.arrays
    row_lower = np.where(
        arrays["row_lower_present"],
        arrays["row_lower"].astype(np.float64),
        -highspy.kHighsInf,
    )
    row_upper = np.where(
        arrays["row_upper_present"],
        arrays["row_upper"].astype(np.float64),
        highspy.kHighsInf,
    )
    starts = np.asarray(arrays["row_starts"], dtype=np.int64)
    indices = np.asarray(arrays["column_indices"], dtype=np.int32)
    values = np.asarray(arrays["coefficients"], dtype=np.float64)
    highs = highspy.Highs()
    _require_status(
        highs.addVars(
            columns,
            np.zeros(columns, dtype=np.float64),
            np.ones(columns, dtype=np.float64),
        ),
        highspy,
        "addVars",
    )
    _require_status(
        highs.changeColsIntegrality(
            columns,
            np.arange(columns, dtype=np.int32),
            np.full(columns, highspy.HighsVarType.kInteger.value, dtype=np.uint8),
        ),
        highspy,
        "changeColsIntegrality",
    )
    _require_status(
        highs.addRows(
            rows,
            row_lower,
            row_upper,
            nonzeros,
            starts,
            indices,
            values,
        ),
        highspy,
        "addRows",
    )
    locked_options = dict(LOCKED_OPTIONS)
    locked_options["time_limit"] = float(maximum_time_seconds)
    for name, value in locked_options.items():
        _require_status(highs.setOptionValue(name, value), highspy, f"option {name}")
    _require_status(highs.setOptionValue("log_file", str(log_path)), highspy, "log file")
    _require_status(highs.setOptionValue("log_to_console", False), highspy, "console log")
    export_statuses = {
        "write_model": _require_artifact(
            highs.writeModel(str(model_path)), highspy, "write model", model_path
        ),
        "write_options": _require_artifact(
            highs.writeOptions(str(options_path)),
            highspy,
            "write options",
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
    _require_status(highs.presolve(), highspy, "presolve")
    presolved = highs.getPresolvedLp()
    export_statuses["write_presolved_model"] = _require_artifact(
        highs.writePresolvedModel(str(presolved_path)),
        highspy,
        "write presolved model",
        presolved_path,
    )
    _require_status(highs.run(), highspy, "run")
    model_status = highs.getModelStatus()
    info = highs.getInfo()
    solution = highs.getSolution()
    export_statuses["write_solution"] = _require_artifact(
        highs.writeSolution(str(solution_path), 1),
        highspy,
        "write solution",
        solution_path,
    )
    export_statuses["write_info"] = _require_artifact(
        highs.writeInfo(str(info_path)), highspy, "write info", info_path
    )
    _write_exclusive(telemetry_path, canonical_json_bytes(telemetry))
    gap = float(info.mip_gap)
    zero_gap = math.isfinite(gap) and gap == 0.0
    infeasible = model_status == highspy.HighsModelStatus.kInfeasible
    exact_row_audit = False
    maximum_rounding_error = None
    if model_status == highspy.HighsModelStatus.kOptimal:
        exact_row_audit, maximum_rounding_error = _audit_optimal_integer_solution(
            solution.col_value,
            arrays,
            columns=columns,
            rows=rows,
            tolerance=LOCKED_OPTIONS["mip_feasibility_tolerance"],
            numpy=np,
        )
    optimal = (
        model_status == highspy.HighsModelStatus.kOptimal
        and zero_gap
        and exact_row_audit
    )
    certified = infeasible or optimal
    artifact_paths = (
        model_path,
        presolved_path,
        options_path,
        log_path,
        solution_path,
        info_path,
        telemetry_path,
    )
    artifacts = {
        path.name: {
            "sha256": _sha256_file(path),
            "byte_size": path.stat().st_size,
        }
        for path in artifact_paths
    }
    receipt: dict[str, object] = {
        "format_version": RECEIPT_FORMAT,
        "bundle_sha256": expected_bundle_sha256,
        "ir_sha256": bundle.metadata["ir_sha256"],
        "highspy_version": highs.version(),
        "locked_options": locked_options,
        "export_statuses": export_statuses,
        "original_dimensions": dimensions,
        "numeric_envelope": bundle.metadata["numeric_envelope"],
        "presolved_dimensions": {
            "columns": int(presolved.num_col_),
            "rows": int(presolved.num_row_),
        },
        "model_status": str(model_status),
        "highs_info": _info_payload(info),
        "mip_gap_is_finite_zero": zero_gap,
        "maximum_integrality_rounding_error": (
            None if maximum_rounding_error is None else str(maximum_rounding_error)
        ),
        "exact_integer_row_audit_passed": exact_row_audit,
        "linear_projection_infeasible": infeasible,
        "linear_projection_optimal": optimal,
        "linear_projection_certified": certified,
        "complete_score_free_feasible_region_certified": False,
        "nonlinear_smd_domain_authorized_by_this_receipt": False,
        "user_solution_callback_used": False,
        "mip_logging_callback_used": True,
        "cp_sat_or_ortools_imported": False,
        "split_gate0_authorized": False,
        "model_selection_authorized": False,
        "outer_evaluation_authorized": False,
        "training_authorized": False,
        "artifacts": artifacts,
    }
    receipt["receipt_sha256"] = hashlib.sha256(
        canonical_json_bytes(receipt)
    ).hexdigest()
    _write_exclusive(receipt_path, canonical_json_bytes(receipt))
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
