"""Validate a frozen V6-C constraint-family diagnostic registration."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Mapping


REGISTRATION_FORMAT = "v6_c_family_core_audit_registration_v1"
EXPECTED_COUNTS = {
    "source_rows": 3229,
    "observed_pairs": 1830,
    "antibodies": 1383,
    "antigens": 1172,
}
EXPECTED_DIMENSIONS = {
    "rows": 2693786,
    "columns": 163464,
    "nonzeros": 10365708,
}
EXPECTED_PARENT_STATUSES = {
    "primary_cp": "PROVEN_INFEASIBLE",
    "independent_cp": "INDEPENDENT_DIRECT_CP_INFEASIBLE_AGREEMENT",
    "direct_highs": "DIRECT_HIGHS_LINEAR_INFEASIBLE_AGREEMENT",
}
EXPECTED_PARENT_REQUIRED_FIELDS = {
    "primary_cp": {
        "direct_highs_fixed_assignment_audit_completed": False,
        "split_gate0_authorized": False,
        "terminal_status": "PROVEN_INFEASIBLE",
        "training_authorized": False,
    },
    "independent_cp": {
        "direct_highs_audit_completed": False,
        "full_protocol_infeasibility_claim_authorized": False,
        "primary_independent_linear_agreement": True,
        "split_gate0_authorized": False,
        "terminal_status": "INDEPENDENT_DIRECT_CP_INFEASIBLE_AGREEMENT",
        "training_authorized": False,
    },
    "direct_highs": {
        "complete_score_free_feasible_region_certified": False,
        "cross_solver_linear_agreement": True,
        "direct_highs_audit_completed": True,
        "full_protocol_infeasibility_claim_authorized": False,
        "model_selection_authorized": False,
        "outer_evaluation_authorized": False,
        "split_gate0_authorized": False,
        "terminal_status": "DIRECT_HIGHS_LINEAR_INFEASIBLE_AGREEMENT",
        "test_access_authorized": False,
        "training_authorized": False,
    },
}
SOURCE_LABELS = {
    "controller",
    "cp_formulation",
    "diagnostic",
    "independent_auditor",
    "joint_ir",
    "launcher",
    "production",
    "registration",
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_OBJECT = re.compile(r"^[0-9a-f]{40}$")


class V6CFamilyCoreRegistrationError(ValueError):
    """Raised when a V6-C family-core registration is not exact."""


def _require_mapping(
    value: Any,
    keys: set[str],
    name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise V6CFamilyCoreRegistrationError(f"{name} schema differs")
    return value


def _require_digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise V6CFamilyCoreRegistrationError(
            f"{name} must be a lowercase SHA-256"
        )
    return value


def _require_absolute_path(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or not value.isascii()
        or not Path(value).is_absolute()
    ):
        raise V6CFamilyCoreRegistrationError(
            f"{name} must be an absolute ASCII path"
        )
    return value


def _validate_file_reference(value: Any, name: str) -> Mapping[str, Any]:
    reference = _require_mapping(
        value,
        {"path", "file_sha256", "receipt_sha256"},
        name,
    )
    _require_absolute_path(reference["path"], f"{name} path")
    _require_digest(reference["file_sha256"], f"{name} file")
    _require_digest(reference["receipt_sha256"], f"{name} receipt")
    return reference


def validate_family_core_registration(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Reject every registration that differs from the frozen V6-C schema."""
    registration = _require_mapping(
        payload,
        {
            "authorization",
            "expected",
            "format_version",
            "limits",
            "parent_results",
            "policy",
            "runtime",
            "slurm",
            "source",
            "v6_b_registration",
        },
        "V6-C registration",
    )
    if registration["format_version"] != REGISTRATION_FORMAT:
        raise V6CFamilyCoreRegistrationError(
            "V6-C registration format differs"
        )

    authorization = _require_mapping(
        registration["authorization"],
        {
            "full_protocol_infeasibility_claim",
            "model_selection",
            "outer_evaluation",
            "split_gate0",
            "test_access",
            "threshold_change",
            "training",
        },
        "V6-C authorization",
    )
    if any(value is not False for value in authorization.values()):
        raise V6CFamilyCoreRegistrationError(
            "V6-C registration grants forbidden authority"
        )

    policy = _require_mapping(
        registration["policy"],
        {"path", "sha256"},
        "V6-C policy reference",
    )
    _require_absolute_path(policy["path"], "V6-C policy path")
    _require_digest(policy["sha256"], "V6-C policy")

    v6_b = _require_mapping(
        registration["v6_b_registration"],
        {"path", "sha256"},
        "V6-B registration reference",
    )
    _require_absolute_path(v6_b["path"], "V6-B registration path")
    _require_digest(v6_b["sha256"], "V6-B registration")

    parent_results = _require_mapping(
        registration["parent_results"],
        set(EXPECTED_PARENT_STATUSES),
        "V6-C parent results",
    )
    for label, expected_status in EXPECTED_PARENT_STATUSES.items():
        parent = _require_mapping(
            parent_results[label],
            {
                "file_sha256",
                "path",
                "receipt_sha256",
                "required_fields",
            },
            f"V6-C parent {label}",
        )
        _require_absolute_path(parent["path"], f"V6-C parent {label} path")
        _require_digest(
            parent["file_sha256"],
            f"V6-C parent {label} file",
        )
        _require_digest(
            parent["receipt_sha256"],
            f"V6-C parent {label} receipt",
        )
        required = _require_mapping(
            parent["required_fields"],
            set(EXPECTED_PARENT_REQUIRED_FIELDS[label]),
            f"V6-C parent {label} required fields",
        )
        expected_required = EXPECTED_PARENT_REQUIRED_FIELDS[label]
        if any(
            (
                required[name] is not expected_value
                if isinstance(expected_value, bool)
                else required[name] != expected_value
            )
            for name, expected_value in expected_required.items()
        ) or required["terminal_status"] != expected_status:
            raise V6CFamilyCoreRegistrationError(
                f"V6-C parent {label} required fields differ"
            )

    expected = _require_mapping(
        registration["expected"],
        {
            "authority_counts",
            "bundle_sha256",
            "dimensions",
            "source_linear_projection_sha256",
        },
        "V6-C expected identity",
    )
    authority_counts = _require_mapping(
        expected["authority_counts"],
        set(EXPECTED_COUNTS),
        "V6-C authority counts",
    )
    if any(
        isinstance(authority_counts[name], bool)
        or not isinstance(authority_counts[name], int)
        or authority_counts[name] != value
        for name, value in EXPECTED_COUNTS.items()
    ):
        raise V6CFamilyCoreRegistrationError(
            "V6-C authority counts differ"
        )
    dimensions = _require_mapping(
        expected["dimensions"],
        set(EXPECTED_DIMENSIONS),
        "V6-C projection dimensions",
    )
    if any(
        isinstance(dimensions[name], bool)
        or not isinstance(dimensions[name], int)
        or dimensions[name] != value
        for name, value in EXPECTED_DIMENSIONS.items()
    ):
        raise V6CFamilyCoreRegistrationError(
            "V6-C projection dimensions differ"
        )
    _require_digest(expected["bundle_sha256"], "V6-C projection bundle")
    _require_digest(
        expected["source_linear_projection_sha256"],
        "V6-C linear projection",
    )

    runtime = _require_mapping(
        registration["runtime"],
        {
            "cp_sat_runtime_receipt_path",
            "cp_sat_runtime_receipt_sha256",
            "libc_family",
            "libc_version",
            "machine",
            "ortools_module_sha256",
            "ortools_version",
            "python_executable",
            "python_version",
            "system",
        },
        "V6-C runtime",
    )
    for name in (
        "libc_family",
        "libc_version",
        "machine",
        "ortools_version",
        "python_version",
        "system",
    ):
        value = runtime[name]
        if not isinstance(value, str) or not value or not value.isascii():
            raise V6CFamilyCoreRegistrationError(
                f"V6-C runtime {name} is empty"
            )
    _require_absolute_path(
        runtime["python_executable"],
        "V6-C Python executable",
    )
    _require_absolute_path(
        runtime["cp_sat_runtime_receipt_path"],
        "V6-C CP-SAT runtime receipt path",
    )
    _require_digest(
        runtime["cp_sat_runtime_receipt_sha256"],
        "V6-C CP-SAT runtime receipt",
    )
    _require_digest(
        runtime["ortools_module_sha256"],
        "V6-C OR-Tools module",
    )

    source = _require_mapping(
        registration["source"],
        {"files", "implementation_commit", "source_root"},
        "V6-C source",
    )
    implementation_commit = source["implementation_commit"]
    if (
        not isinstance(implementation_commit, str)
        or _GIT_OBJECT.fullmatch(implementation_commit) is None
    ):
        raise V6CFamilyCoreRegistrationError(
            "V6-C implementation commit differs"
        )
    _require_absolute_path(source["source_root"], "V6-C source root")
    files = _require_mapping(
        source["files"],
        SOURCE_LABELS,
        "V6-C source files",
    )
    for label, reference in files.items():
        item = _require_mapping(
            reference,
            {"path", "sha256"},
            f"V6-C source {label}",
        )
        path = item["path"]
        if (
            not isinstance(path, str)
            or not path
            or not path.isascii()
            or Path(path).is_absolute()
            or ".." in Path(path).parts
        ):
            raise V6CFamilyCoreRegistrationError(
                f"V6-C source {label} path differs"
            )
        _require_digest(item["sha256"], f"V6-C source {label}")

    limits = _require_mapping(
        registration["limits"],
        {"maximum_time_seconds_per_solve", "minimize_core"},
        "V6-C limits",
    )
    maximum_time = limits["maximum_time_seconds_per_solve"]
    if (
        isinstance(maximum_time, bool)
        or not isinstance(maximum_time, int)
        or maximum_time < 1
    ):
        raise V6CFamilyCoreRegistrationError(
            "V6-C solve time limit must be a positive integer"
        )
    if limits["minimize_core"] is not True:
        raise V6CFamilyCoreRegistrationError(
            "V6-C must request deterministic core minimization"
        )

    slurm = _require_mapping(
        registration["slurm"],
        {
            "accelerator_visible_to_solver",
            "allocated_gpu_purpose",
            "cpus_per_task",
            "explicit_memory_request",
            "gpus_requested",
            "nodes",
            "partition",
            "solver_workers",
            "tasks",
            "wall_time",
        },
        "V6-C Slurm envelope",
    )
    exact_integers = {
        "nodes": 1,
        "tasks": 1,
        "solver_workers": 1,
        "gpus_requested": 1,
    }
    if (
        any(
            isinstance(slurm[name], bool)
            or not isinstance(slurm[name], int)
            or slurm[name] != expected_value
            for name, expected_value in exact_integers.items()
        )
        or slurm["accelerator_visible_to_solver"] is not False
        or slurm["allocated_gpu_purpose"]
        != "site_scheduler_requirement_only"
        or slurm["explicit_memory_request"] is not False
    ):
        raise V6CFamilyCoreRegistrationError(
            "V6-C Slurm isolation differs"
        )
    cpus = slurm["cpus_per_task"]
    if isinstance(cpus, bool) or not isinstance(cpus, int) or cpus < 1:
        raise V6CFamilyCoreRegistrationError(
            "V6-C CPU request must be positive"
        )
    for name in ("partition", "wall_time"):
        value = slurm[name]
        if not isinstance(value, str) or not value or not value.isascii():
            raise V6CFamilyCoreRegistrationError(
                f"V6-C Slurm {name} is empty"
            )
    return dict(registration)


__all__ = [
    "EXPECTED_COUNTS",
    "EXPECTED_DIMENSIONS",
    "EXPECTED_PARENT_REQUIRED_FIELDS",
    "EXPECTED_PARENT_STATUSES",
    "REGISTRATION_FORMAT",
    "SOURCE_LABELS",
    "V6CFamilyCoreRegistrationError",
    "validate_family_core_registration",
]
