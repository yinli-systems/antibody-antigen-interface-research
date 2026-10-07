#!/usr/bin/env python3
"""Submit the frozen chain-map, DCP-canary, and embedding-materialization chain."""

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
    parser.add_argument("--authority-root", type=Path, required=True)
    args = parser.parse_args()
    protocol = args.authority_root / "ABF30B_FRAMECONTACT_TRANSFER_PROTOCOL_V1.json"
    protocol_receipt = (
        args.authority_root / "ABF30B_FRAMECONTACT_TRANSFER_PROTOCOL_V1_RECEIPT.json"
    )
    submission = args.authority_root / "ABF30B_INITIAL_CHAIN_SUBMISSION_RECEIPT_V1.json"
    if submission.exists() or submission.is_symlink():
        raise FileExistsError(submission)
    receipt = json.loads(protocol_receipt.read_text())
    if (
        sha256_file(protocol) != receipt.get("protocol_sha256")
        or receipt.get("frozen_before_chain_typing") is not True
        or receipt.get("frozen_before_dcp_restore") is not True
    ):
        raise ValueError("frozen protocol receipt differs")
    scripts = {
        "chain_map": args.tools_root / "build_abf30b_chain_map_v1.sbatch",
        "dcp_canary": args.tools_root / "canary_abf30b_dcp_embedding_v1.sbatch",
        "materialize": args.tools_root / "materialize_abf30b_sabdab2_embeddings_v1.sbatch",
    }
    for path in scripts.values():
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"submission script missing or symlinked: {path}")
    chain_job = submit(scripts["chain_map"])
    canary_job = submit(scripts["dcp_canary"], chain_job)
    materialize_job = submit(scripts["materialize"], canary_job)
    value = {
        "format_version": "abf30b_initial_chain_submission_receipt_v1",
        "status": "SUBMITTED_FROZEN_INITIAL_CHAIN",
        "protocol_sha256": receipt["protocol_sha256"],
        "scripts": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in scripts.items()
        },
        "jobs": {
            "chain_map": chain_job,
            "dcp_canary": canary_job,
            "materialize": materialize_job,
        },
        "dependencies": {
            canary_job: f"afterok:{chain_job}",
            materialize_job: f"afterok:{canary_job}",
        },
        "old_sabdab2_test_loaded": False,
        "temporal_holdout_loaded": False,
    }
    receipt_sha = publish(submission, value)
    print(json.dumps({"receipt_sha256": receipt_sha, **value["jobs"]}, sort_keys=True))


if __name__ == "__main__":
    main()
