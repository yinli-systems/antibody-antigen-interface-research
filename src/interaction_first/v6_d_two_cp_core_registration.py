"""Validate a frozen V6-D two-CP family-core registration."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Mapping


REGISTRATION_FORMAT = "v6_d_two_cp_core_audit_registration_v1"
EXPECTED_COUNTS = {
    "antibody_entities": 1383,
    "antigen_entities": 1172,
    "exact_pairs": 1830,
    "source_rows": 3229,
}
EXPECTED_DIMENSIONS = {
    "rows": 2693786,
    "columns": 163464,
    "nonzeros": 10365708,
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
}
SOURCE_LABELS = {
    "audit",
    "controller",
    "core_projection",
    "cp_formulation",
    "diagnostic",
    "freezer",
    "independent_auditor",
    "joint_ir",
    "launcher",
    "production",
    "registration",
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_OBJECT = re.compile(r"^[0-9a-f]{40}$")


class V6DRegistrationError(ValueError):
    """Raised when a V6-D registration differs from the frozen schema."""


def _mapping(value: Any, keys: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise V6DRegistrationError(f"{label} schema differs")
    return value


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise V6DRegistrationError(f"{label} must be a lowercase SHA-256")
    return value


def _absolute_path(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or not value.isascii()
        or not Path(value).is_absolute()
    ):
        raise V6DRegistrationError(f"{label} must be an absolute ASCII path")
    return value


def _false_authorization(value: Any) -> Mapping[str, Any]:
    authorization = _mapping(
        value,
        {
            "full_protocol_infeasibility_claim",
            "model_selection",
            "outer_evaluation",
            "split_gate0",
            "test_access",
            "threshold_change",
            "training",
        },
        "V6-D authorization",
    )
    if any(item is not False for item in authorization.values()):
        raise V6DRegistrationError("V6-D registration grants forbidden authority")
    return authorization


def validate_two_cp_core_registration(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Reject every registration that differs from the frozen V6-D schema."""
    registration = _mapping(
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
        "V6-D registration",
    )
    if registration["format_version"] != REGISTRATION_FORMAT:
        raise V6DRegistrationError("V6-D registration format differs")
    _false_authorization(registration["authorization"])

    expected = _mapping(
        registration["expected"],
        {
            "authority_counts",
            "dimensions",
            "ir_sha256",
            "source_linear_projection_sha256",
        },
        "V6-D expected identity",
    )
    counts = _mapping(
        expected["authority_counts"],
        set(EXPECTED_COUNTS),
        "V6-D authority counts",
    )
    if any(
        isinstance(counts[name], bool) or counts[name] != value
        for name, value in EXPECTED_COUNTS.items()
    ):
        raise V6DRegistrationError("V6-D authority counts differ")
    dimensions = _mapping(
        expected["dimensions"],
        set(EXPECTED_DIMENSIONS),
        "V6-D projection dimensions",
    )
    if any(
        isinstance(dimensions[name], bool) or dimensions[name] != value
        for name, value in EXPECTED_DIMENSIONS.items()
    ):
        raise V6DRegistrationError("V6-D projection dimensions differ")
    _digest(
        expected["source_linear_projection_sha256"],
        "V6-D linear projection",
    )
    _digest(expected["ir_sha256"], "V6-D joint IR")

    policy = _mapping(
        registration["policy"],
        {"path", "sha256"},
        "V6-D policy reference",
    )
    _absolute_path(policy["path"], "V6-D policy path")
    _digest(policy["sha256"], "V6-D policy")
    v6_b = _mapping(
        registration["v6_b_registration"],
        {"path", "sha256"},
        "V6-B registration reference",
    )
    _absolute_path(v6_b["path"], "V6-B registration path")
    _digest(v6_b["sha256"], "V6-B registration")

    parents = _mapping(
        registration["parent_results"],
        set(EXPECTED_PARENT_REQUIRED_FIELDS),
        "V6-D parent results",
    )
    for label, required_expected in EXPECTED_PARENT_REQUIRED_FIELDS.items():
        parent = _mapping(
            parents[label],
            {"file_sha256", "path", "receipt_sha256", "required_fields"},
            f"V6-D parent {label}",
        )
        _absolute_path(parent["path"], f"V6-D parent {label} path")
        _digest(parent["file_sha256"], f"V6-D parent {label} file")
        _digest(parent["receipt_sha256"], f"V6-D parent {label} receipt")
        required = _mapping(
            parent["required_fields"],
            set(required_expected),
            f"V6-D parent {label} required fields",
        )
        if any(
            (
                required[name] is not expected_value
                if isinstance(expected_value, bool)
                else required[name] != expected_value
            )
            for name, expected_value in required_expected.items()
        ):
            raise V6DRegistrationError(f"V6-D parent {label} required fields differ")

    runtime = _mapping(
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
        "V6-D runtime",
    )
    for name in (
        "libc_family",
        "libc_version",
        "machine",
        "ortools_version",
        "python_version",
        "system",
    ):
        if (
            not isinstance(runtime[name], str)
            or not runtime[name]
            or not runtime[name].isascii()
        ):
            raise V6DRegistrationError(f"V6-D runtime {name} is empty")
    _absolute_path(runtime["python_executable"], "V6-D Python executable")
    _absolute_path(
        runtime["cp_sat_runtime_receipt_path"],
        "V6-D CP-SAT runtime receipt path",
    )
    _digest(
        runtime["cp_sat_runtime_receipt_sha256"],
        "V6-D CP-SAT runtime receipt",
    )
    _digest(runtime["ortools_module_sha256"], "V6-D OR-Tools module")

    limits = _mapping(
        registration["limits"],
        {"maximum_time_seconds_per_solve", "minimize_core"},
        "V6-D limits",
    )
    maximum_time = limits["maximum_time_seconds_per_solve"]
    if (
        isinstance(maximum_time, bool)
        or not isinstance(maximum_time, int)
        or maximum_time < 1
    ):
        raise V6DRegistrationError("V6-D solve time limit must be positive")
    if limits["minimize_core"] is not True:
        raise V6DRegistrationError("V6-D must minimize the family core")

    source = _mapping(
        registration["source"],
        {"files", "implementation_commit", "source_root"},
        "V6-D source",
    )
    if (
        not isinstance(source["implementation_commit"], str)
        or _GIT_OBJECT.fullmatch(source["implementation_commit"]) is None
    ):
        raise V6DRegistrationError("V6-D implementation commit differs")
    _absolute_path(source["source_root"], "V6-D source root")
    files = _mapping(source["files"], SOURCE_LABELS, "V6-D source files")
    for label, reference in files.items():
        item = _mapping(
            reference,
            {"path", "sha256"},
            f"V6-D source {label}",
        )
        path = item["path"]
        if (
            not isinstance(path, str)
            or not path
            or not path.isascii()
            or Path(path).is_absolute()
            or ".." in Path(path).parts
        ):
            raise V6DRegistrationError(f"V6-D source {label} path differs")
        _digest(item["sha256"], f"V6-D source {label}")

    slurm = _mapping(
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
        "V6-D Slurm envelope",
    )
    required_slurm = {
        "accelerator_visible_to_solver": False,
        "allocated_gpu_purpose": "site_scheduler_requirement_only",
        "cpus_per_task": 8,
        "explicit_memory_request": False,
        "gpus_requested": 1,
        "nodes": 1,
        "partition": "gpu_5090",
        "solver_workers": 1,
        "tasks": 1,
        "wall_time": "24:00:00",
    }
    if any(
        (
            slurm[name] is not expected_value
            if isinstance(expected_value, bool)
            else slurm[name] != expected_value
        )
        for name, expected_value in required_slurm.items()
    ):
        raise V6DRegistrationError("V6-D Slurm isolation differs")
    return dict(registration)


__all__ = [
    "EXPECTED_COUNTS",
    "EXPECTED_DIMENSIONS",
    "EXPECTED_PARENT_REQUIRED_FIELDS",
    "REGISTRATION_FORMAT",
    "SOURCE_LABELS",
    "V6DRegistrationError",
    "validate_two_cp_core_registration",
]
