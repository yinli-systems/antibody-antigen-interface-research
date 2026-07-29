#!/usr/bin/env python3
"""Freeze one canonical V6-C family-core audit registration."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping

from interaction_first import v6_b_production as production
from interaction_first import v6_c_family_core_audit as audit
from interaction_first.v6_c_family_core_registration import (
    EXPECTED_COUNTS,
    EXPECTED_DIMENSIONS,
    EXPECTED_PARENT_REQUIRED_FIELDS,
    SOURCE_LABELS,
    validate_family_core_registration,
)


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_OBJECT = re.compile(r"^[0-9a-f]{40}$")
SOURCE_PATHS = {
    "audit": "src/interaction_first/v6_c_family_core_audit.py",
    "controller": "scripts/run_v6_c_family_core_audit.py",
    "cp_formulation": "src/interaction_first/v6_cp_sat_formulation.py",
    "diagnostic": "src/interaction_first/v6_constraint_family_diagnostic.py",
    "freezer": "scripts/freeze_v6_c_family_core_registration.py",
    "independent_auditor": (
        "src/interaction_first/score_free_independent_auditor.py"
    ),
    "joint_ir": "src/interaction_first/v6_joint_ir.py",
    "launcher": "scripts/submit_v6_c_family_core_audit.sbatch",
    "production": "src/interaction_first/v6_b_production.py",
    "registration": (
        "src/interaction_first/v6_c_family_core_registration.py"
    ),
}


def parse_args() -> argparse.Namespace:
    """Parse explicit source, parent, identity, and output anchors."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--v6-b-registration", type=Path, required=True)
    parser.add_argument("--primary-cp-result", type=Path, required=True)
    parser.add_argument("--independent-cp-result", type=Path, required=True)
    parser.add_argument("--direct-highs-result", type=Path, required=True)
    parser.add_argument("--source-linear-projection-sha256", required=True)
    parser.add_argument(
        "--maximum-time-seconds-per-solve",
        type=int,
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _read_canonical(path: Path, label: str) -> tuple[bytes, dict[str, Any]]:
    payload = path.read_bytes()
    parsed = production.strict_json(payload, label)
    return payload, parsed


def _verified_receipt_reference(
    path: Path,
    required_fields: Mapping[str, object],
    label: str,
) -> dict[str, object]:
    payload, receipt = _read_canonical(path, label)
    claimed = receipt.get("receipt_sha256")
    unsigned = dict(receipt)
    unsigned.pop("receipt_sha256", None)
    if (
        not isinstance(claimed, str)
        or _SHA256.fullmatch(claimed) is None
        or audit.sha256_payload(unsigned) != claimed
    ):
        raise ValueError(f"{label} self hash differs")
    for name, expected in required_fields.items():
        observed = receipt.get(name)
        if (
            observed is not expected
            if isinstance(expected, bool)
            else observed != expected
        ):
            raise ValueError(f"{label} required field {name} differs")
    return {
        "file_sha256": hashlib.sha256(payload).hexdigest(),
        "path": str(path.resolve(strict=True)),
        "receipt_sha256": claimed,
        "required_fields": dict(required_fields),
    }


def _source_inventory(source_root: Path) -> dict[str, object]:
    if set(SOURCE_PATHS) != SOURCE_LABELS:
        raise ValueError("V6-C source inventory labels differ")
    inventory = {}
    for label, relative in SOURCE_PATHS.items():
        path = source_root / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"V6-C source {label} is absent or symbolic")
        inventory[label] = {
            "path": relative,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    return inventory


def _write_exclusive(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(
        os,
        "O_CLOEXEC",
        0,
    )
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


def freeze_registration(arguments: argparse.Namespace) -> dict[str, object]:
    """Authenticate all current evidence and create one immutable registration."""
    if arguments.source_root.is_symlink():
        raise ValueError("V6-C source root must not be symbolic")
    source_root = arguments.source_root.resolve(strict=True)
    if not source_root.is_dir():
        raise ValueError("V6-C source root is absent or symbolic")
    if (
        not isinstance(arguments.implementation_commit, str)
        or _GIT_OBJECT.fullmatch(arguments.implementation_commit) is None
    ):
        raise ValueError("V6-C implementation commit is malformed")
    if (
        not isinstance(arguments.source_linear_projection_sha256, str)
        or _SHA256.fullmatch(
            arguments.source_linear_projection_sha256
        )
        is None
    ):
        raise ValueError("V6-C linear projection digest is malformed")
    if (
        isinstance(arguments.maximum_time_seconds_per_solve, bool)
        or arguments.maximum_time_seconds_per_solve < 1
    ):
        raise ValueError("V6-C solve time limit must be positive")
    output = Path(
        os.path.abspath(os.path.expanduser(str(arguments.output)))
    )
    if output.exists() or output.is_symlink():
        raise FileExistsError(
            f"create-only V6-C registration exists: {output}"
        )
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise ValueError("V6-C registration parent is absent or symbolic")

    v6_b_bytes, v6_b = _read_canonical(
        arguments.v6_b_registration,
        "V6-B production registration",
    )
    v6_b = production.validate_registration(v6_b)
    v6_b_sha256 = hashlib.sha256(v6_b_bytes).hexdigest()
    if production._sha256_payload(v6_b) != v6_b_sha256:
        raise ValueError("V6-B production registration digest differs")

    parents = {
        "primary_cp": _verified_receipt_reference(
            arguments.primary_cp_result,
            EXPECTED_PARENT_REQUIRED_FIELDS["primary_cp"],
            "primary CP result",
        ),
        "independent_cp": _verified_receipt_reference(
            arguments.independent_cp_result,
            EXPECTED_PARENT_REQUIRED_FIELDS["independent_cp"],
            "independent CP result",
        ),
        "direct_highs": _verified_receipt_reference(
            arguments.direct_highs_result,
            EXPECTED_PARENT_REQUIRED_FIELDS["direct_highs"],
            "direct-HiGHS result",
        ),
    }
    direct_payload = production.strict_json(
        production.read_verified(
            arguments.direct_highs_result,
            str(parents["direct_highs"]["file_sha256"]),
            "direct-HiGHS result",
        ),
        "direct-HiGHS result",
    )
    bundle_receipt = direct_payload.get("bundle_export_receipt")
    if not isinstance(bundle_receipt, Mapping):
        raise ValueError("direct-HiGHS bundle receipt is absent")
    bundle_sha256 = bundle_receipt.get("bundle_sha256")
    if not isinstance(bundle_sha256, str) or _SHA256.fullmatch(
        bundle_sha256
    ) is None:
        raise ValueError("direct-HiGHS bundle digest is malformed")

    runtime = v6_b["runtime"]
    platform_abi = runtime.get("platform_abi")
    if not isinstance(platform_abi, Mapping):
        raise ValueError("V6-B runtime lacks the locked platform ABI")
    policy_path = (
        source_root
        / "protocols"
        / "v6_c_linear_infeasibility_diagnostic_policy_v1.json"
    )
    if not policy_path.is_file() or policy_path.is_symlink():
        raise ValueError("V6-C policy is absent or symbolic")
    registration = {
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
            "bundle_sha256": bundle_sha256,
            "dimensions": EXPECTED_DIMENSIONS,
            "source_linear_projection_sha256": (
                arguments.source_linear_projection_sha256
            ),
        },
        "format_version": "v6_c_family_core_audit_registration_v1",
        "limits": {
            "maximum_time_seconds_per_solve": (
                arguments.maximum_time_seconds_per_solve
            ),
            "minimize_core": True,
        },
        "parent_results": parents,
        "policy": {
            "path": str(policy_path),
            "sha256": hashlib.sha256(policy_path.read_bytes()).hexdigest(),
        },
        "runtime": {
            "cp_sat_runtime_receipt_path": (
                runtime["cp_sat_runtime_receipt_path"]
            ),
            "cp_sat_runtime_receipt_sha256": (
                runtime["cp_sat_runtime_receipt_sha256"]
            ),
            "libc_family": platform_abi["libc_family"],
            "libc_version": platform_abi["libc_version"],
            "machine": platform_abi["machine"],
            "ortools_module_sha256": (
                platform_abi["ortools_module_sha256"]
            ),
            "ortools_version": runtime["ortools_version"],
            "python_executable": runtime["python_executable"],
            "python_version": runtime["python_version"],
            "system": platform_abi["system"],
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
            "files": _source_inventory(source_root),
            "implementation_commit": arguments.implementation_commit,
            "source_root": str(source_root),
        },
        "v6_b_registration": {
            "path": str(arguments.v6_b_registration.resolve(strict=True)),
            "sha256": v6_b_sha256,
        },
    }
    checked = validate_family_core_registration(registration)
    audit._authenticate_policy(checked)
    audit._authenticate_parent_results(
        checked,
        v6_b_registration_sha256=v6_b_sha256,
    )
    audit._authenticate_runtime(checked)
    controller = source_root / SOURCE_PATHS["controller"]
    audit._authenticate_sources(
        checked,
        controller_path=controller,
        controller_sha256=checked["source"]["files"]["controller"]["sha256"],
    )
    payload = audit.canonical_json_bytes(checked)
    _write_exclusive(output, payload)
    return {
        "authorization": checked["authorization"],
        "file_sha256": hashlib.sha256(payload).hexdigest(),
        "output": str(output),
        "registration": checked,
    }


def main() -> int:
    """Freeze the registration and print only bounded evidence."""
    frozen = freeze_registration(parse_args())
    print(
        json.dumps(
            {
                "file_sha256": frozen["file_sha256"],
                "full_protocol_infeasibility_claim_authorized": False,
                "output": frozen["output"],
                "split_gate0_authorized": False,
                "test_access_authorized": False,
                "threshold_change_authorized": False,
                "training_authorized": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
