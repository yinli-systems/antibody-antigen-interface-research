#!/usr/bin/env python3
"""Freeze a fail-closed temporal label-free cohort gate result."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


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
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--execution-plan", type=Path, required=True)
    parser.add_argument("--execution-plan-sha256", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--recovery-receipt", type=Path, required=True)
    parser.add_argument("--failed-stdout", type=Path, required=True)
    parser.add_argument("--failed-stderr", type=Path, required=True)
    parser.add_argument("--failed-job-id", type=int, required=True)
    parser.add_argument("--cancelled-eval-job-id", required=True)
    parser.add_argument("--cancelled-analyzer-job-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    if sha256_file(args.protocol) != args.protocol_sha256:
        raise RuntimeError("protocol SHA differs")
    if sha256_file(args.execution_plan) != args.execution_plan_sha256:
        raise RuntimeError("execution-plan SHA differs")
    protocol = json.loads(args.protocol.read_text())
    plan = json.loads(args.execution_plan.read_text())
    manifest = json.loads(args.manifest.read_text())
    recovery = json.loads(args.recovery_receipt.read_text())
    counts = manifest.get("counts", {})
    gate = plan.get("gate", {})
    if (
        protocol.get("authorization", {}).get("temporal_opening_count_maximum") != 1
        or protocol.get("authorization", {}).get("old_exposed_test_may_be_reopened") is not False
        or gate.get("minimum_double_cold_instances") != 12
        or gate.get("minimum_double_cold_unique_pdbs") != 10
        or gate.get("thresholds_may_not_be_relaxed_after_label_free_screen") is not True
        or counts.get("double_cold_instances") != 15
        or counts.get("double_cold_unique_pdbs") != 9
        or manifest.get("audit", {}).get("task_labels_read") is not False
        or manifest.get("audit", {}).get("cross_chain_distances_computed") is not False
        or recovery.get("task_labels_read") is not False
        or recovery.get("gate_thresholds_changed") is not False
    ):
        raise RuntimeError("temporal gate evidence differs")

    result = {
        "format_version": "framecontact_temporal_double_cold_gate_result_v1",
        "status": "FAIL_CLOSED_INSUFFICIENT_UNIQUE_PDBS_BEFORE_OPENING",
        "protocol_sha256": args.protocol_sha256,
        "execution_plan_sha256": args.execution_plan_sha256,
        "label_free_manifest_sha256": sha256_file(args.manifest),
        "label_free_recovery_receipt_sha256": sha256_file(args.recovery_receipt),
        "gate": {
            "observed_double_cold_instances": 15,
            "required_double_cold_instances": 12,
            "instances_pass": True,
            "observed_double_cold_unique_pdbs": 9,
            "required_double_cold_unique_pdbs": 10,
            "unique_pdbs_pass": False,
            "overall_pass": False,
            "thresholds_changed": False,
        },
        "jobs": {
            "failed_materialization": args.failed_job_id,
            "cancelled_evaluation_array": args.cancelled_eval_job_id,
            "cancelled_analyzer": args.cancelled_analyzer_job_id,
        },
        "failure_evidence": {
            "stdout_sha256": sha256_file(args.failed_stdout),
            "stderr_sha256": sha256_file(args.failed_stderr),
        },
        "audit": {
            "temporal_opening_count": 0,
            "task_labels_read": False,
            "cross_chain_distances_computed": False,
            "checkpoint_selection_performed": False,
            "development_feedback_performed": False,
            "old_exposed_test_accessed": False,
            "temporal_measurement_authorized": False,
        },
    }
    atomic_write(args.output, canonical_bytes(result))
    receipt = {
        "format_version": "framecontact_temporal_double_cold_gate_result_receipt_v1",
        "status": result["status"],
        "result_sha256": sha256_file(args.output),
        "protocol_sha256": args.protocol_sha256,
        "temporal_opening_count": 0,
        "task_labels_read": False,
        "temporal_measurement_authorized": False,
    }
    atomic_write(args.receipt, canonical_bytes(receipt))
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
