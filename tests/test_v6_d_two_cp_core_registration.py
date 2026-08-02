"""Tests for the frozen V6-D two-CP core registration."""

from __future__ import annotations

import copy

import pytest

from interaction_first.v6_d_two_cp_core_registration import (
    EXPECTED_COUNTS,
    EXPECTED_DIMENSIONS,
    EXPECTED_PARENT_REQUIRED_FIELDS,
    SOURCE_LABELS,
    V6DRegistrationError,
    validate_two_cp_core_registration,
)


def _reference() -> dict[str, object]:
    digest = "a" * 64
    return {
        "authorization": {
            "full_protocol_infeasibility_claim": False,
            "model_selection": False,
            "outer_evaluation": False,
            "split_gate0": False,
            "test_access": False,
            "threshold_change": False,
            "training": False,
        },
        "expected": {
            "authority_counts": EXPECTED_COUNTS,
            "dimensions": EXPECTED_DIMENSIONS,
            "parent_ir_sha256": digest,
            "source_linear_projection_sha256": digest,
        },
        "format_version": "v6_d_two_cp_core_audit_registration_v1",
        "limits": {
            "maximum_time_seconds_per_solve": 3600,
            "minimize_core": True,
        },
        "parent_results": {
            label: {
                "file_sha256": digest,
                "path": f"/evidence/{label}.json",
                "receipt_sha256": digest,
                "required_fields": required,
            }
            for label, required in EXPECTED_PARENT_REQUIRED_FIELDS.items()
        },
        "policy": {"path": "/protocol/v6_d.json", "sha256": digest},
        "runtime": {
            "cp_sat_runtime_receipt_path": "/runtime/receipt.json",
            "cp_sat_runtime_receipt_sha256": digest,
            "libc_family": "glibc",
            "libc_version": "2.35",
            "machine": "x86_64",
            "ortools_module_sha256": digest,
            "ortools_version": "9.15.6755",
            "python_executable": "/runtime/bin/python",
            "python_version": "3.10.12",
            "system": "Linux",
        },
        "slurm": {
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
        },
        "source": {
            "files": {
                label: {"path": f"src/{label}.py", "sha256": digest}
                for label in SOURCE_LABELS
            },
            "implementation_commit": "b" * 40,
            "source_root": "/source/v6_d",
        },
        "v6_b_registration": {
            "path": "/protocol/v6_b.json",
            "sha256": digest,
        },
    }


def test_valid_v6_d_registration_is_preserved() -> None:
    registration = _reference()
    assert validate_two_cp_core_registration(registration) == registration


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("training", "forbidden authority"),
        ("direct_parent", "parent results"),
        ("parent_status", "independent_cp required fields"),
        ("dimensions", "projection dimensions"),
        ("relative_source", "source diagnostic path"),
        ("visible_gpu", "Slurm isolation"),
        ("minimization", "minimize"),
        ("runtime_digest", "OR-Tools module"),
        ("boolean_count", "authority counts"),
    ],
)
def test_v6_d_registration_mutations_fail_closed(
    mutation: str,
    message: str,
) -> None:
    registration = copy.deepcopy(_reference())
    if mutation == "training":
        registration["authorization"]["training"] = True
    elif mutation == "direct_parent":
        registration["parent_results"]["direct_highs"] = {}
    elif mutation == "parent_status":
        registration["parent_results"]["independent_cp"]["required_fields"][
            "terminal_status"
        ] = "UNKNOWN"
    elif mutation == "dimensions":
        registration["expected"]["dimensions"]["rows"] -= 1
    elif mutation == "relative_source":
        registration["source"]["files"]["diagnostic"]["path"] = "../bad.py"
    elif mutation == "visible_gpu":
        registration["slurm"]["accelerator_visible_to_solver"] = True
    elif mutation == "minimization":
        registration["limits"]["minimize_core"] = False
    elif mutation == "runtime_digest":
        registration["runtime"]["ortools_module_sha256"] = "bad"
    elif mutation == "boolean_count":
        registration["expected"]["authority_counts"]["source_rows"] = True
    else:
        raise AssertionError(mutation)
    with pytest.raises(V6DRegistrationError, match=message):
        validate_two_cp_core_registration(registration)
