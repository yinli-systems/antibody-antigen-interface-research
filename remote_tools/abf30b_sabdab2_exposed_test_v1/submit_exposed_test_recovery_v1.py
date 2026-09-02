#!/usr/bin/env python3
"""Submit the receipt-adapter recovery continuation for the frozen test chain."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("ascii")


def submit(path: Path, dependency: str) -> str:
    output = subprocess.check_output(["sbatch", "--parsable", f"--dependency=afterok:{dependency}", str(path)], text=True).strip()
    job = output.split(";", 1)[0]
    if not job.isdecimal():
        raise RuntimeError(f"unexpected sbatch output: {output}")
    return job


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tools-root", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--structure-job", required=True)
    args = parser.parse_args()
    recovery_path = args.campaign_root / "authority/CHAIN_MAP_RECEIPT_RECOVERY_PROTOCOL_V1.json"
    recovery_receipt_path = args.campaign_root / "authority/CHAIN_MAP_RECEIPT_RECOVERY_PROTOCOL_V1_RECEIPT.json"
    adapter_path = args.campaign_root / "label_free_inputs_v1/TEST_INPUT_PLAN_V1_CHAIN_MAP_ADAPTER.json"
    original_submission_path = args.campaign_root / "authority/EXPOSED_TEST_CHAIN_SUBMISSION_RECEIPT_V1.json"
    output = args.campaign_root / "authority/EXPOSED_TEST_CHAIN_RECOVERY_SUBMISSION_RECEIPT_V1.json"
    for path in (recovery_path, recovery_receipt_path, adapter_path, original_submission_path):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"recovery authority missing: {path}")
    recovery = json.loads(recovery_path.read_text(encoding="utf-8"))
    recovery_receipt = json.loads(recovery_receipt_path.read_text(encoding="utf-8"))
    adapter = json.loads(adapter_path.read_text(encoding="utf-8"))
    recovery_sha = sha256_file(recovery_path)
    if (
        recovery.get("status") != "FROZEN_BEFORE_EXPOSED_TEST_LABEL_ARTIFACT_READ"
        or recovery_receipt.get("protocol_sha256") != recovery_sha
        or adapter.get("recovery_protocol_sha256") != recovery_sha
        or adapter.get("task_labels_read") is not False
        or recovery.get("scientific_protocol_changed") is not False
        or recovery.get("cumulative_test_access_ordinal") != 3
    ):
        raise ValueError("recovery authority differs")
    scripts = {
        "embeddings": args.tools_root / "materialize_test_embeddings_recovery_v1.sbatch",
        "measurement_freeze": args.tools_root / "freeze_test_measurement_v1.sbatch",
        "evaluation": args.tools_root / "evaluate_exposed_test_v1.sbatch",
        "analyzer": args.tools_root / "analyze_exposed_test_v1.sbatch",
    }
    jobs: dict[str, str] = {}
    dependencies: dict[str, str] = {}
    previous = args.structure_job
    for name, script in scripts.items():
        jobs[name] = submit(script, previous)
        dependencies[jobs[name]] = f"afterok:{previous}"
        previous = jobs[name]
    payload = {
        "format_version": "abf30b_sabdab2_exposed_test_chain_recovery_submission_receipt_v1",
        "status": "SUBMITTED_FIELD_ALIAS_RECOVERY_CONTINUATION",
        "recovery_protocol_sha256": recovery_sha,
        "adapter_sha256": sha256_file(adapter_path),
        "original_submission_receipt_sha256": sha256_file(original_submission_path),
        "retained_structure_job": args.structure_job,
        "cancelled_unstarted_jobs": ["1500380", "1500381", "1500382", "1500383"],
        "scripts": {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in scripts.items()},
        "jobs": jobs,
        "dependencies": dependencies,
        "scientific_protocol_changed": False,
        "test_label_artifacts_read_before_recovery_freeze": False,
        "selection_performed_on_test": False,
        "fresh_blind_test": False,
        "cumulative_test_access_ordinal": 3,
    }
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    raw = canonical_bytes(payload)
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
    print(json.dumps({"status": payload["status"], "jobs": jobs, "receipt_sha256": hashlib.sha256(raw).hexdigest()}, sort_keys=True))


if __name__ == "__main__":
    main()
