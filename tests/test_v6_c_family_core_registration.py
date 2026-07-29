"""Tests for the frozen V6-C family-core production registration."""

from __future__ import annotations

import copy

import pytest

from interaction_first.v6_c_family_core_registration import (
    EXPECTED_COUNTS,
    EXPECTED_DIMENSIONS,
    EXPECTED_PARENT_REQUIRED_FIELDS,
    EXPECTED_PARENT_STATUSES,
    SOURCE_LABELS,
    V6CFamilyCoreRegistrationError,
    validate_family_core_registration,
)


def _reference() -> dict[str, object]:
    digest = "a" * 64
    required = {
        label: {
            "file_sha256": digest,
            "path": f"/evidence/{label}.json",
            "receipt_sha256": digest,
            "required_fields": EXPECTED_PARENT_REQUIRED_FIELDS[label],
        }
        for label in EXPECTED_PARENT_STATUSES
    }
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
            "bundle_sha256": digest,
            "direct_highs_registration_sha256": digest,
            "dimensions": EXPECTED_DIMENSIONS,
            "source_linear_projection_sha256": digest,
        },
        "format_version": "v6_c_family_core_audit_registration_v1",
        "limits": {
            "maximum_time_seconds_per_solve": 3600,
            "minimize_core": True,
        },
        "parent_results": required,
        "policy": {
            "path": "/protocol/v6_c_policy.json",
            "sha256": digest,
        },
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
                label: {
                    "path": (
                        f"src/{label}.py"
                        if label not in {"controller", "launcher"}
                        else f"scripts/{label}.py"
                    ),
                    "sha256": digest,
                }
                for label in SOURCE_LABELS
            },
            "implementation_commit": "b" * 40,
            "source_root": "/source/v6_c",
        },
        "v6_b_registration": {
            "path": "/protocol/v6_b_registration.json",
            "sha256": digest,
        },
    }


def test_valid_registration_is_preserved() -> None:
    registration = _reference()
    assert validate_family_core_registration(registration) == registration


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("training", "forbidden authority"),
        ("direct_status", "direct_highs required fields"),
        ("projection_dimensions", "projection dimensions"),
        ("relative_source", "source diagnostic path"),
        ("visible_gpu", "Slurm isolation"),
        ("minimization", "core minimization"),
        ("runtime_digest", "OR-Tools module"),
        ("direct_registration_digest", "direct-HiGHS registration"),
        ("count_bool", "authority counts"),
        ("task_bool", "Slurm isolation"),
    ],
)
def test_registration_mutations_fail_closed(
    mutation: str,
    message: str,
) -> None:
    registration = copy.deepcopy(_reference())
    if mutation == "training":
        registration["authorization"]["training"] = True
    elif mutation == "direct_status":
        registration["parent_results"]["direct_highs"]["required_fields"][
            "terminal_status"
        ] = "DIRECT_HIGHS_LINEAR_UNCERTIFIED"
    elif mutation == "projection_dimensions":
        registration["expected"]["dimensions"]["rows"] -= 1
    elif mutation == "relative_source":
        registration["source"]["files"]["diagnostic"]["path"] = (
            "../diagnostic.py"
        )
    elif mutation == "visible_gpu":
        registration["slurm"]["accelerator_visible_to_solver"] = True
    elif mutation == "minimization":
        registration["limits"]["minimize_core"] = False
    elif mutation == "runtime_digest":
        registration["runtime"]["ortools_module_sha256"] = "bad"
    elif mutation == "direct_registration_digest":
        registration["expected"]["direct_highs_registration_sha256"] = "bad"
    elif mutation == "count_bool":
        registration["expected"]["authority_counts"]["source_rows"] = True
    elif mutation == "task_bool":
        registration["slurm"]["tasks"] = True
    else:
        raise AssertionError(f"unknown mutation: {mutation}")
    with pytest.raises(V6CFamilyCoreRegistrationError, match=message):
        validate_family_core_registration(registration)
