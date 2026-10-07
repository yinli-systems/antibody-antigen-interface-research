#!/usr/bin/env python3
"""Submit the frozen ABF late-stage checkpoint-scaling dependency chain."""

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
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
    return hashlib.sha256(payload).hexdigest()


def submit(path: Path, dependency: str | None) -> str:
    command = ["sbatch", "--parsable"]
    if dependency:
        command.append(f"--dependency=afterok:{dependency}")
    command.append(str(path))
    output = subprocess.check_output(command, text=True).strip()
    job_id = output.split(";", 1)[0]
    if not job_id.isdecimal():
        raise RuntimeError(f"unexpected sbatch output: {output!r}")
    return job_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tools-root", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    args = parser.parse_args()
    protocol_path = args.campaign_root / "authority/ABF30B_LATE_STAGE_SCALING_PROTOCOL_V1.json"
    protocol_receipt_path = (
        args.campaign_root
        / "authority/ABF30B_LATE_STAGE_SCALING_PROTOCOL_V1_RECEIPT.json"
    )
    output_path = args.campaign_root / "authority/LATE_STAGE_SCALING_SUBMISSION_RECEIPT_V1.json"
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(output_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol_receipt = json.loads(protocol_receipt_path.read_text(encoding="utf-8"))
    protocol_sha = sha256_file(protocol_path)
    if (
        protocol.get("status") != "FROZEN_BEFORE_LATE_STAGE_SCALING_EXECUTION"
        or protocol_receipt.get("protocol_sha256") != protocol_sha
        or protocol_receipt.get("frozen_before_new_dcp_restore") is not True
        or protocol.get("task_labels_read_during_freeze") is not False
        or protocol.get("test_loaded") is not False
        or protocol.get("temporal_holdout_loaded") is not False
    ):
        raise ValueError("scaling protocol authority differs")
    scripts = {
        "embeddings": args.tools_root / "materialize_checkpoint_embeddings_v1.sbatch",
        "execution_freeze": args.tools_root / "freeze_checkpoint_executions_v1.sbatch",
        "training": args.tools_root / "slurm_abf_checkpoint_development_v1.sbatch",
        "analyzer": args.tools_root / "analyze_late_stage_scaling_v1.sbatch",
    }
    for path in scripts.values():
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"submission script missing or symlinked: {path}")
    embedding_job = submit(scripts["embeddings"], None)
    freeze_job = submit(scripts["execution_freeze"], embedding_job)
    training_job = submit(scripts["training"], freeze_job)
    analyzer_job = submit(scripts["analyzer"], training_job)
    payload = {
        "format_version": "abf30b_late_stage_scaling_submission_receipt_v1",
        "status": "SUBMITTED_FROZEN_LATE_STAGE_SCALING_CHAIN",
        "scaling_protocol_sha256": protocol_sha,
        "scripts": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in scripts.items()
        },
        "jobs": {
            "embedding_array": embedding_job,
            "execution_freeze_array": freeze_job,
            "training_array": training_job,
            "analyzer": analyzer_job,
        },
        "dependencies": {
            freeze_job: f"afterok:{embedding_job}",
            training_job: f"afterok:{freeze_job}",
            analyzer_job: f"afterok:{training_job}",
        },
        "new_checkpoint_count": 2,
        "new_development_run_count": 12,
        "terminal_checkpoint_runs_reused": 6,
        "external_measurement_submitted": False,
        "old_sabdab2_test_loaded": False,
        "temporal_holdout_loaded": False,
    }
    receipt_sha = publish(output_path, payload)
    print(json.dumps({"receipt_sha256": receipt_sha, **payload["jobs"]}, sort_keys=True))


if __name__ == "__main__":
    main()
