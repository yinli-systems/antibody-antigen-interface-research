#!/usr/bin/env python3
"""Submit the frozen development execution, six-run array, and analyzer chain."""

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


def submit(path: Path, dependency: str) -> str:
    output = subprocess.check_output(
        ["sbatch", "--parsable", f"--dependency=afterok:{dependency}", str(path)],
        text=True,
    ).strip()
    job_id = output.split(";", 1)[0]
    if not job_id.isdecimal():
        raise RuntimeError(f"unexpected sbatch output: {output!r}")
    return job_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tools-root", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--embedding-job", default="1496562")
    args = parser.parse_args()
    authority = args.campaign_root / "authority"
    recovery_path = authority / "ABF30B_DCP_CANARY_RECOVERY_SUBMISSION_RECEIPT_V1.json"
    transfer_path = authority / "ABF30B_FRAMECONTACT_TRANSFER_PROTOCOL_V1_RECEIPT.json"
    output_path = authority / "ABF30B_DEVELOPMENT_CHAIN_SUBMISSION_RECEIPT_V1.json"
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(output_path)
    recovery = json.loads(recovery_path.read_text())
    transfer = json.loads(transfer_path.read_text())
    if (
        recovery.get("jobs", {}).get("materialize") != args.embedding_job
        or recovery.get("scientific_protocol_changed") is not False
        or transfer.get("frozen_before_model_specific_sable_measurement") is not True
        or transfer.get("test_loaded") is not False
    ):
        raise ValueError("upstream submission authority differs")
    scripts = {
        "freeze": args.tools_root / "freeze_abf30b_development_execution_v1.sbatch",
        "training": args.tools_root / "slurm_abf30b_framecontact_development_v1.sbatch",
        "analyzer": args.tools_root / "analyze_abf30b_framecontact_development_v1.sbatch",
    }
    for script in scripts.values():
        if script.is_symlink() or not script.is_file():
            raise ValueError(f"submission script missing or symlinked: {script}")
    freeze_job = submit(scripts["freeze"], args.embedding_job)
    training_job = submit(scripts["training"], freeze_job)
    analyzer_job = submit(scripts["analyzer"], training_job)
    payload = {
        "format_version": "abf30b_framecontact_development_chain_submission_receipt_v1",
        "status": "SUBMITTED_FROZEN_DEVELOPMENT_CHAIN_BEFORE_MODEL_SPECIFIC_EXTERNAL_OPENING",
        "transfer_protocol_sha256": transfer["protocol_sha256"],
        "upstream_embedding_job": args.embedding_job,
        "scripts": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in scripts.items()
        },
        "jobs": {
            "execution_freeze": freeze_job,
            "training_array": training_job,
            "analyzer": analyzer_job,
        },
        "dependencies": {
            freeze_job: f"afterok:{args.embedding_job}",
            training_job: f"afterok:{freeze_job}",
            analyzer_job: f"afterok:{training_job}",
        },
        "six_run_mapping": {
            "0": ["baseline", 17],
            "1": ["baseline", 29],
            "2": ["baseline", 43],
            "3": ["edge_topk2_supervised", 17],
            "4": ["edge_topk2_supervised", 29],
            "5": ["edge_topk2_supervised", 43],
        },
        "external_opening_submitted": False,
        "old_sabdab2_test_loaded": False,
        "temporal_holdout_loaded": False,
    }
    receipt_sha = publish(output_path, payload)
    print(json.dumps({"receipt_sha256": receipt_sha, **payload["jobs"]}, sort_keys=True))


if __name__ == "__main__":
    main()
