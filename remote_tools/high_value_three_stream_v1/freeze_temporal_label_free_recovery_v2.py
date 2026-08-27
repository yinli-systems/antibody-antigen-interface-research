#!/usr/bin/env python3
"""Freeze second label-free recovery with case-sensitive transfer amendment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-protocol", type=Path, required=True)
    parser.add_argument("--parent-protocol-sha256", required=True)
    parser.add_argument("--execution-plan", type=Path, required=True)
    parser.add_argument("--execution-plan-sha256", required=True)
    parser.add_argument("--recovery-v1", type=Path, required=True)
    parser.add_argument("--recovery-v1-sha256", required=True)
    parser.add_argument("--transfer-amendment", type=Path, required=True)
    parser.add_argument("--transfer-amendment-sha256", required=True)
    parser.add_argument("--failed-stdout", type=Path, required=True)
    parser.add_argument("--failed-stderr", type=Path, required=True)
    parser.add_argument("--failed-job-id", type=int, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.output, args.receipt):
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
    for path, expected, label in (
        (args.parent_protocol, args.parent_protocol_sha256, "parent protocol"),
        (args.execution_plan, args.execution_plan_sha256, "execution plan"),
        (args.recovery_v1, args.recovery_v1_sha256, "recovery-v1 plan"),
        (
            args.transfer_amendment,
            args.transfer_amendment_sha256,
            "transfer amendment",
        ),
    ):
        if sha256_file(path) != expected:
            raise ValueError(f"{label} SHA-256 differs")
    parent = json.loads(args.parent_protocol.read_text(encoding="utf-8"))
    execution = json.loads(args.execution_plan.read_text(encoding="utf-8"))
    recovery_v1 = json.loads(args.recovery_v1.read_text(encoding="utf-8"))
    amendment = json.loads(args.transfer_amendment.read_text(encoding="utf-8"))
    if (
        parent.get("selection", {}).get("allowed") is not False
        or execution.get("gate", {}).get("thresholds_may_not_be_relaxed_after_label_free_screen")
        is not True
        or recovery_v1.get("status")
        != "FROZEN_AFTER_LABEL_FREE_INTERFACE_FAILURE_BEFORE_RECOVERY"
        or recovery_v1.get("audit", {}).get("task_labels_read") is not False
        or amendment.get("status")
        != "PASS_CASE_SENSITIVE_TRANSFER_CORRECTION_FROZEN"
        or amendment.get("audit", {}).get("task_labels_read") is not False
        or amendment.get("audit", {}).get("temporal_opening_count") != 0
    ):
        raise ValueError("recovery-v2 authority differs")
    base = args.source_root / "prepare_temporal_double_cold_v1.py"
    wrapper = args.source_root / "prepare_temporal_double_cold_recovery_v1.py"
    verifier = args.source_root / "verify_temporal_label_free_recovery_v1.py"
    label_free_sbatch = args.source_root / "slurm_temporal_label_free_recovery_v1.sbatch"
    open_sbatch = args.source_root / "slurm_temporal_freeze_materialize_recovery_v1.sbatch"
    frozen_open_sbatch = args.source_root / "slurm_temporal_freeze_materialize_v1.sbatch"
    plan = {
        "format_version": "framecontact_temporal_label_free_recovery_plan_v2",
        "status": "FROZEN_AFTER_TWO_LABEL_FREE_INTERFACE_FAILURES_BEFORE_RECOVERY",
        "parent_protocol_sha256": args.parent_protocol_sha256,
        "execution_plan_sha256": args.execution_plan_sha256,
        "supersedes_recovery_plan_sha256": args.recovery_v1_sha256,
        "cif_transfer_amendment": {
            "path": str(args.transfer_amendment),
            "sha256": args.transfer_amendment_sha256,
            "status": amendment["status"],
            "corrected_instance": amendment["instance"],
            "corrected_cif_sha256": amendment["corrected"]["sha256"],
        },
        "second_failed_attempt": {
            "job_id": args.failed_job_id,
            "state": "FAILED",
            "exit_code": "1:0",
            "stdout_sha256": sha256_file(args.failed_stdout),
            "stderr_sha256": sha256_file(args.failed_stderr),
            "failure": "case-insensitive transfer collision detected before sequence alignment",
            "labels_or_cross_chain_distances_reached": False,
        },
        "recovery": {
            "base_script_sha256": sha256_file(base),
            "wrapper_script_sha256": sha256_file(wrapper),
            "verifier_script_sha256": sha256_file(verifier),
            "label_free_sbatch_sha256": sha256_file(label_free_sbatch),
            "open_sbatch_sha256": sha256_file(open_sbatch),
            "frozen_open_sbatch_sha256": sha256_file(frozen_open_sbatch),
            "semantic_change": recovery_v1["recovery"]["semantic_change"],
            "transfer_change": "replace only the aliased uppercase H/L cropped CIF with its exact case-sensitive API response",
            "allowed_scope": "label-free antigen polymer classification and one case-sensitive transfer correction only",
            "model_checkpoint_or_threshold_change": False,
            "gate_threshold_change": False,
        },
        "gate": execution["gate"],
        "measurement": execution["measurement"],
        "opening": execution["opening"],
        "audit": {
            "task_labels_read": False,
            "cross_chain_distances_computed": False,
            "contact_labels_computed": False,
            "selection_performed": False,
            "temporal_opening_count": 0,
        },
    }
    atomic_write(args.output, canonical_bytes(plan))
    receipt = {
        "format_version": "framecontact_temporal_label_free_recovery_plan_receipt_v2",
        "status": plan["status"],
        "plan_sha256": sha256_file(args.output),
        "supersedes_recovery_plan_sha256": args.recovery_v1_sha256,
        "transfer_amendment_sha256": args.transfer_amendment_sha256,
        "failed_job_id": args.failed_job_id,
        "task_labels_read": False,
        "temporal_opening_count": 0,
    }
    atomic_write(args.receipt, canonical_bytes(receipt))
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
