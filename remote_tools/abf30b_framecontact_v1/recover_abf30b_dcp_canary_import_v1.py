#!/usr/bin/env python3
"""Record the import-only DCP-canary failure and submit the SHA-bound recovery."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any


EXPECTED_FAILURE = "ModuleNotFoundError: No module named 'antibody_pretraining'"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("ascii")


def publish(path: Path, value: Any) -> str:
    payload = canonical_bytes(value)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
    return hashlib.sha256(payload).hexdigest()


def job_state(job_id: str) -> tuple[str, str]:
    output = subprocess.check_output(
        [
            "sacct",
            "-X",
            "-n",
            "-P",
            "-j",
            job_id,
            "--format=State,ExitCode",
        ],
        text=True,
    )
    rows = [line.split("|")[:2] for line in output.splitlines() if line.strip()]
    if not rows:
        raise RuntimeError(f"no accounting row for {job_id}")
    return rows[0][0].split("+", 1)[0], rows[0][1]


def submit(path: Path, dependency: str | None = None) -> str:
    command = ["sbatch", "--parsable"]
    if dependency is not None:
        command.append(f"--dependency=afterok:{dependency}")
    command.append(str(path))
    job_id = subprocess.check_output(command, text=True).strip().split(";", 1)[0]
    if not job_id.isdecimal():
        raise RuntimeError(f"unexpected sbatch output: {job_id!r}")
    return job_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tools-root", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--failed-canary-job", default="1496535")
    parser.add_argument("--dead-materialize-job", default="1496536")
    args = parser.parse_args()
    authority = args.campaign_root / "authority"
    initial_receipt_path = authority / "ABF30B_INITIAL_CHAIN_SUBMISSION_RECEIPT_V1.json"
    chain_receipt_path = authority / "SABDAB2_DEVELOPMENT_ABF30B_CHAIN_MAP_V1_RECEIPT.json"
    failed_stdout = args.campaign_root / "logs" / f"dcp-canary-{args.failed_canary_job}.out"
    failed_stderr = args.campaign_root / "logs" / f"dcp-canary-{args.failed_canary_job}.err"
    failure_path = authority / "ABF30B_DCP_CANARY_IMPORT_FAILURE_V1.json"
    recovery_path = authority / "ABF30B_DCP_CANARY_RECOVERY_SUBMISSION_RECEIPT_V1.json"
    for output in (failure_path, recovery_path):
        if output.exists() or output.is_symlink():
            raise FileExistsError(output)

    initial = json.loads(initial_receipt_path.read_text())
    chain_receipt = json.loads(chain_receipt_path.read_text())
    if (
        initial.get("jobs", {}).get("dcp_canary") != args.failed_canary_job
        or initial.get("jobs", {}).get("materialize") != args.dead_materialize_job
        or chain_receipt.get("status") != "PASS_LABEL_FREE_ANARCII_CHAIN_TYPING"
    ):
        raise ValueError("initial submission or chain-map receipt differs")
    failed_state, failed_exit = job_state(args.failed_canary_job)
    dead_state, dead_exit = job_state(args.dead_materialize_job)
    stderr_text = failed_stderr.read_text()
    if (
        failed_state != "FAILED"
        or failed_exit != "1:0"
        or dead_state != "PENDING"
        or EXPECTED_FAILURE not in stderr_text
        or (authority / "ABF30B_DCP_EMBEDDING_CANARY_V1.json").exists()
        or (authority / "ABF30B_DCP_EMBEDDING_CANARY_V1_RECEIPT.json").exists()
    ):
        raise ValueError("failure is not the frozen import-only failure")

    failure = {
        "format_version": "abf30b_dcp_canary_import_failure_v1",
        "status": "FAIL_CLOSED_IMPORT_PATH_BEFORE_DCP_RESTORE",
        "failed_job": args.failed_canary_job,
        "failed_state": failed_state,
        "failed_exit_code": failed_exit,
        "stdout_sha256": sha256_file(failed_stdout),
        "stderr_sha256": sha256_file(failed_stderr),
        "failure_signature": EXPECTED_FAILURE,
        "dcp_restore_started": False,
        "model_inference_started": False,
        "task_labels_read": False,
        "test_loaded": False,
        "temporal_holdout_loaded": False,
    }
    failure_sha = publish(failure_path, failure)

    # The dependent job can never run because its afterok predecessor failed.
    subprocess.run(["scancel", args.dead_materialize_job], check=True)
    canary_script = args.tools_root / "canary_abf30b_dcp_embedding_v1.sbatch"
    materialize_script = args.tools_root / "materialize_abf30b_sabdab2_embeddings_v1.sbatch"
    for script in (canary_script, materialize_script):
        if script.is_symlink() or not script.is_file():
            raise ValueError(f"missing recovery script: {script}")
    canary_job = submit(canary_script)
    materialize_job = submit(materialize_script, canary_job)
    recovery = {
        "format_version": "abf30b_dcp_canary_recovery_submission_receipt_v1",
        "status": "SUBMITTED_IMPORT_PATH_RECOVERY_WITH_UNCHANGED_SCIENTIFIC_PROTOCOL",
        "failure_receipt_sha256": failure_sha,
        "chain_map_sha256": chain_receipt["chain_map_sha256"],
        "change_scope": "PYTHONPATH corrected to frozen pretraining source snapshot and model/tokenizer source SHA checks added",
        "scientific_protocol_changed": False,
        "checkpoint_changed": False,
        "model_changed": False,
        "data_changed": False,
        "scripts": {
            "dcp_canary": {
                "path": str(canary_script),
                "sha256": sha256_file(canary_script),
            },
            "materialize": {
                "path": str(materialize_script),
                "sha256": sha256_file(materialize_script),
            },
        },
        "jobs": {"dcp_canary": canary_job, "materialize": materialize_job},
        "dependencies": {materialize_job: f"afterok:{canary_job}"},
        "cancelled_dead_dependency_job": args.dead_materialize_job,
        "task_labels_read": False,
        "test_loaded": False,
        "temporal_holdout_loaded": False,
    }
    recovery_sha = publish(recovery_path, recovery)
    print(
        json.dumps(
            {
                "failure_receipt_sha256": failure_sha,
                "recovery_receipt_sha256": recovery_sha,
                **recovery["jobs"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
