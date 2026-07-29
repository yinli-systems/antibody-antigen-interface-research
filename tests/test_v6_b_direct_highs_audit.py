"""Tests for the registered V6-B Direct-HiGHS audit chain."""

from __future__ import annotations

import copy
import hashlib
import os
import platform
from pathlib import Path
import sys

import highspy
import numpy
import pytest

from interaction_first.v6_b_direct_highs_audit import (
    DirectHighsAuditError,
)
from interaction_first.v6_b_direct_highs_audit import (
    canonical_json_bytes,
)
from interaction_first.v6_b_direct_highs_audit import (
    execute_direct_highs_audit,
)
from interaction_first.v6_b_direct_highs_audit import (
    validate_direct_highs_registration,
)
from interaction_first.v6_highs_bundle_backend import LOCKED_OPTIONS
from interaction_first.v6_joint_ir import CanonicalJointIR, IRLinearConstraint
from interaction_first.v6_linear_projection_bundle import (
    export_linear_projection_bundle,
)


ROOT = Path(__file__).resolve().parents[1]
HASH = "a" * 64
COMMIT = "b" * 40


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_canonical(path: Path, payload: dict) -> tuple[str, str]:
    unsigned = dict(payload)
    receipt_sha256 = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    payload = {**unsigned, "receipt_sha256": receipt_sha256}
    encoded = canonical_json_bytes(payload)
    path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest(), receipt_sha256


def _infeasible_ir() -> CanonicalJointIR:
    variable = ("x",)
    return CanonicalJointIR(
        variable_keys=(variable,),
        linear_constraints=(
            IRLinearConstraint("force_one", ((variable, 1),), 1, 1),
            IRLinearConstraint("force_zero", ((variable, 1),), 0, 0),
        ),
        separation_gates=(),
        source_bindings=(("source", HASH),),
        source_linear_projection_sha256=HASH,
        construction_scope="test",
        independent_raw_authority_reconstruction=True,
    )


def _registration(tmp_path: Path) -> dict:
    bundle_directory = tmp_path / "bundle"
    bundle = export_linear_projection_bundle(_infeasible_ir(), bundle_directory)
    bundle_json_sha256 = _sha256_file(bundle_directory / "bundle.json")
    bundle_export_path = tmp_path / "bundle_export_receipt.json"
    bundle_export_file_sha256, bundle_export_receipt_sha256 = _write_canonical(
        bundle_export_path,
        {
            "format_version": "v6_b_linear_projection_bundle_export_receipt_v1",
            "terminal_status": "FULL_LINEAR_PROJECTION_BUNDLE_EXPORTED",
            "highs_solver_invoked": False,
            "highspy_imported": False,
            "split_gate0_authorized": False,
            "training_authorized": False,
            "ir_sha256": bundle["ir_sha256"],
            "bundle": {
                "directory": str(bundle_directory.resolve()),
                "bundle_sha256": bundle["bundle_sha256"],
                "bundle_json_file_sha256": bundle_json_sha256,
                "artifacts": bundle["artifacts"],
            },
        },
    )
    primary_path = tmp_path / "primary.json"
    primary_file_sha256, primary_receipt_sha256 = _write_canonical(
        primary_path,
        {
            "format_version": "v6_b_production_final_receipt_v1",
            "decomposition_terminal_status": "PROVEN_INFEASIBLE",
            "master_status": "INFEASIBLE",
            "primary_controller_proven_infeasible": True,
            "full_protocol_infeasibility_claim_authorized": False,
            "split_gate0_authorized": False,
            "training_authorized": False,
        },
    )
    independent_path = tmp_path / "independent.json"
    independent_file_sha256, independent_receipt_sha256 = _write_canonical(
        independent_path,
        {
            "format_version": "v6_b_independent_cp_audit_receipt_v1",
            "projection_status": "INFEASIBLE",
            "linear_agreement_with_primary": True,
            "full_protocol_infeasibility_claim_authorized": False,
            "split_gate0_authorized": False,
            "training_authorized": False,
        },
    )
    runtime_path = tmp_path / "runtime.json"
    libc_family, libc_version = platform.libc_ver()
    runtime_file_sha256, runtime_receipt_sha256 = _write_canonical(
        runtime_path,
        {
            "format_version": "v6_a_solver_runtime_lock_v1",
            "backend": "highs",
            "python": {
                "executable": sys.executable,
                "version": platform.python_version(),
            },
            "solver_smoke": {
                "native_library_process_isolation": True,
                "highs": {
                    "highs_version": highspy.Highs().version(),
                    "locked_options": LOCKED_OPTIONS,
                },
            },
            "training_authorized": False,
        },
    )
    source_files = {
        "backend": "src/interaction_first/v6_highs_bundle_backend.py",
        "bundle": "src/interaction_first/v6_linear_projection_bundle.py",
        "controller": "scripts/run_v6_b_direct_highs_audit.py",
        "launcher": "scripts/submit_v6_b_direct_highs_audit.sbatch",
    }
    return {
        "format_version": "v6_b_direct_highs_audit_registration_v1",
        "authorization": {
            "full_protocol_infeasibility_claim": False,
            "split_gate0": False,
            "training": False,
            "model_selection": False,
            "outer_evaluation": False,
            "test_access": False,
        },
        "bundle_export": {
            "receipt_path": str(bundle_export_path.resolve()),
            "receipt_file_sha256": bundle_export_file_sha256,
            "receipt_sha256": bundle_export_receipt_sha256,
            "bundle_directory": str(bundle_directory.resolve()),
            "bundle_sha256": bundle["bundle_sha256"],
            "bundle_json_file_sha256": bundle_json_sha256,
        },
        "primary_cp_result": {
            "path": str(primary_path.resolve()),
            "file_sha256": primary_file_sha256,
            "receipt_sha256": primary_receipt_sha256,
            "format_version": "v6_b_production_final_receipt_v1",
            "required_fields": {
                "decomposition_terminal_status": "PROVEN_INFEASIBLE",
                "master_status": "INFEASIBLE",
                "primary_controller_proven_infeasible": True,
                "full_protocol_infeasibility_claim_authorized": False,
                "split_gate0_authorized": False,
                "training_authorized": False,
            },
        },
        "independent_cp_result": {
            "path": str(independent_path.resolve()),
            "file_sha256": independent_file_sha256,
            "receipt_sha256": independent_receipt_sha256,
            "format_version": "v6_b_independent_cp_audit_receipt_v1",
            "required_fields": {
                "projection_status": "INFEASIBLE",
                "linear_agreement_with_primary": True,
                "full_protocol_infeasibility_claim_authorized": False,
                "split_gate0_authorized": False,
                "training_authorized": False,
            },
        },
        "runtime": {
            "receipt_path": str(runtime_path.resolve()),
            "receipt_file_sha256": runtime_file_sha256,
            "receipt_sha256": runtime_receipt_sha256,
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "highspy_version": highspy.Highs().version(),
            "numpy_version": numpy.__version__,
            "system": platform.system(),
            "machine": platform.machine(),
            "libc_family": libc_family,
            "libc_version": libc_version,
            "native_process_isolation": True,
        },
        "expected": {
            "ir_sha256": bundle["ir_sha256"],
            "source_linear_projection_sha256": HASH,
            "dimensions": bundle["dimensions"],
            "cp_consensus_status": "INFEASIBLE",
        },
        "source": {
            "source_root": str(ROOT),
            "implementation_commit": COMMIT,
            "files": {
                label: {
                    "path": relative_path,
                    "sha256": _sha256_file(ROOT / relative_path),
                }
                for label, relative_path in source_files.items()
            },
        },
        "limits": {"maximum_time_seconds": 30},
        "slurm": {
            "partition": "gpu_5090",
            "nodes": 1,
            "tasks": 1,
            "gpus_requested": 1,
            "allocated_gpu_purpose": "site_scheduler_requirement_only",
            "accelerator_visible_to_solver": False,
            "cpus_per_task": 8,
            "solver_threads": 1,
            "explicit_memory_request": False,
            "wall_time": "08:00:00",
        },
    }


def _make_writable(path: Path) -> None:
    if not path.exists():
        return
    for root, directories, files in os.walk(path):
        os.chmod(root, 0o700)
        for name in directories:
            os.chmod(Path(root) / name, 0o700)
        for name in files:
            os.chmod(Path(root) / name, 0o600)


def test_registered_direct_highs_audit_agrees_only_on_linear_infeasibility(
    tmp_path: Path,
) -> None:
    registration = _registration(tmp_path)
    output = tmp_path / "audit"
    try:
        receipt = execute_direct_highs_audit(
            registration=registration,
            registration_sha256=hashlib.sha256(
                canonical_json_bytes(registration)
            ).hexdigest(),
            controller_sha256=registration["source"]["files"]["controller"][
                "sha256"
            ],
            output_directory=output,
        )
        assert (
            receipt["terminal_status"]
            == "DIRECT_HIGHS_LINEAR_INFEASIBLE_AGREEMENT"
        )
        assert receipt["cross_solver_linear_agreement"] is True
        assert receipt["direct_highs_audit_completed"] is True
        assert receipt["solver_receipt"]["linear_projection_infeasible"] is True
        assert receipt["full_protocol_infeasibility_claim_authorized"] is False
        assert receipt["complete_score_free_feasible_region_certified"] is False
        assert receipt["split_gate0_authorized"] is False
        assert receipt["training_authorized"] is False
        assert receipt["test_access_authorized"] is False
        assert (output / "solver" / "original_model.mps").is_file()
        assert (output / "solver" / "presolved_model.mps").is_file()
    finally:
        _make_writable(tmp_path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("training_authorized", "authorizes science"),
        ("primary_status", "primary CP required fields differ"),
        ("bundle_digest", "bundle export reference is malformed"),
        ("threads", "Slurm envelope"),
    ],
)
def test_registration_mutations_fail_closed(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    registration = _registration(tmp_path)
    altered = copy.deepcopy(registration)
    if mutation == "training_authorized":
        altered["authorization"]["training"] = True
    elif mutation == "primary_status":
        altered["primary_cp_result"]["required_fields"]["master_status"] = "OPTIMAL"
    elif mutation == "bundle_digest":
        altered["bundle_export"]["bundle_sha256"] = "not-a-digest"
    elif mutation == "threads":
        altered["slurm"]["solver_threads"] = 2
    else:
        raise AssertionError(f"Unhandled mutation: {mutation}")
    with pytest.raises(DirectHighsAuditError, match=message):
        validate_direct_highs_registration(altered)
