#!/usr/bin/env python3
"""Recover a scheduler-only dependency rejection after verifying completed embeddings."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any


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
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
    return hashlib.sha256(payload).hexdigest()


def submit(path: Path, dependency: str) -> str:
    output = subprocess.check_output(
        ["sbatch", "--parsable", f"--dependency=afterok:{dependency}", str(path)],
        text=True,
    ).strip()
    job_id = output.split(";", 1)[0]
    if not job_id.isdecimal():
        raise RuntimeError(f"unexpected sbatch output: {output!r}")
    return job_id


def completed_zero(job_id: str) -> bool:
    output = subprocess.check_output(
        ["sacct", "-X", "-j", job_id, "--noheader", "--parsable2", "--format=JobIDRaw,State,ExitCode"],
        text=True,
    )
    rows = [line.split("|") for line in output.splitlines() if line.strip()]
    return any(row[:3] == [job_id, "COMPLETED", "0:0"] for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tools-root", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--development-analyzer-job", default="1496570")
    parser.add_argument("--sable-embedding-job", default="1496583")
    args = parser.parse_args()
    authority = args.campaign_root / "authority/sable_measurement_v1"
    failure_path = authority / "SYSTEMS_FAILURE_DEPENDENCY_SUBMISSION_V1.json"
    recovery_path = authority / "ABF30B_SABLE_MEASUREMENT_RECOVERY_SUBMISSION_RECEIPT_V1.json"
    for path in (failure_path, recovery_path):
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)

    label_free_submission_path = (
        args.campaign_root
        / "authority/sable_label_free_v1/ABF30B_SABLE_LABEL_FREE_SUBMISSION_RECEIPT_V1.json"
    )
    embedding_root = (
        args.campaign_root / "embeddings/sable_abf30b_antibody_esm2_antigen_v1"
    )
    embedding_manifest_path = embedding_root / "SABLE_ESM_EMBEDDING_MANIFEST.json"
    embedding_receipt_path = embedding_root / "SABLE_ABF30B_MIXED_EMBEDDING_RECEIPT.json"
    label_free = json.loads(label_free_submission_path.read_text())
    embedding = json.loads(embedding_manifest_path.read_text())
    embedding_receipt = json.loads(embedding_receipt_path.read_text())
    if (
        label_free.get("jobs", {}).get("embeddings") != args.sable_embedding_job
        or not completed_zero(args.sable_embedding_job)
        or embedding.get("status") != "PASS_LABEL_FREE_SABLE_MIXED_PARTNER_EMBEDDINGS"
        or embedding_receipt.get("manifest_sha256") != sha256_file(embedding_manifest_path)
        or embedding_receipt.get("bound_labels_read") is not False
        or embedding.get("authority", {}).get("selection_or_tuning") is not False
    ):
        raise ValueError("completed SABLE embedding authority differs")

    failure = {
        "format_version": "abf30b_sable_dependency_submission_failure_v1",
        "status": "FAIL_CLOSED_SCHEDULER_DEPENDENCY_SUBMISSION_REJECTED",
        "failed_command_dependency": f"afterok:{args.development_analyzer_job}:{args.sable_embedding_job}",
        "stderr": "sbatch: error: Batch job submission failed: Job dependency problem",
        "jobs_submitted_by_failed_attempt": 0,
        "sable_embedding_job": args.sable_embedding_job,
        "sable_embedding_job_completed_zero": True,
        "sable_embedding_manifest_sha256": embedding_receipt["manifest_sha256"],
        "bound_labels_read_by_failed_attempt": False,
        "performance_read_by_failed_attempt": False,
        "scientific_protocol_changed": False,
    }
    failure_sha = publish(failure_path, failure)

    scripts = {
        "bind_authorize": args.tools_root / "bind_authorize_abf30b_sable_measurement_v1.sbatch",
        "evaluate": args.tools_root / "evaluate_abf30b_sable_component_v1.sbatch",
        "analyze": args.tools_root / "analyze_abf30b_sable_measurement_v1.sbatch",
    }
    for path in scripts.values():
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"submission script missing or unsafe: {path}")
    bind_job = submit(scripts["bind_authorize"], args.development_analyzer_job)
    evaluate_job = submit(scripts["evaluate"], bind_job)
    analyze_job = submit(scripts["analyze"], evaluate_job)
    recovery = {
        "format_version": "abf30b_sable_dependency_submission_recovery_receipt_v1",
        "status": "RECOVERED_SCHEDULER_DEPENDENCY_SUBMISSION_WITH_VERIFIED_COMPLETED_EMBEDDING",
        "failure_receipt_sha256": failure_sha,
        "completed_embedding_job": args.sable_embedding_job,
        "completed_embedding_manifest_sha256": embedding_receipt["manifest_sha256"],
        "remaining_upstream_dependency": f"afterok:{args.development_analyzer_job}",
        "scripts": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in scripts.items()
        },
        "jobs": {
            "bind_authorize": bind_job,
            "evaluation_array": evaluate_job,
            "analyzer": analyze_job,
        },
        "dependencies": {
            bind_job: f"afterok:{args.development_analyzer_job}",
            evaluate_job: f"afterok:{bind_job}",
            analyze_job: f"afterok:{evaluate_job}",
        },
        "scientific_protocol_changed": False,
        "fresh_blind_test": False,
        "measurement_only": True,
        "new_bound_label_opening": False,
        "selection_or_tuning": False,
        "old_sabdab2_test_loaded": False,
        "temporal_holdout_loaded": False,
    }
    recovery_sha = publish(recovery_path, recovery)
    print(json.dumps({"recovery_receipt_sha256": recovery_sha, **recovery["jobs"]}, sort_keys=True))


if __name__ == "__main__":
    main()
