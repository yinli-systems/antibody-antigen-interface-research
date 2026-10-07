#!/usr/bin/env python3
"""Submit SABLE label-free chain typing and mixed-embedding materialization."""

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
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("ascii")


def publish(path: Path, value: Any) -> str:
    payload = canonical_bytes(value)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle: handle.write(payload)
    return hashlib.sha256(payload).hexdigest()


def submit(path: Path, dependency: str | None = None) -> str:
    command = ["sbatch", "--parsable"]
    if dependency is not None: command.append(f"--dependency=afterok:{dependency}")
    command.append(str(path))
    job_id = subprocess.check_output(command, text=True).strip().split(";", 1)[0]
    if not job_id.isdecimal(): raise RuntimeError(f"unexpected sbatch output: {job_id!r}")
    return job_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tools-root", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    args = parser.parse_args()
    authority = args.campaign_root / "authority/sable_label_free_v1"
    protocol_path = authority / "ABF30B_SABLE_LABEL_FREE_PROTOCOL_V1.json"
    receipt_path = authority / "ABF30B_SABLE_LABEL_FREE_PROTOCOL_V1_RECEIPT.json"
    output_path = authority / "ABF30B_SABLE_LABEL_FREE_SUBMISSION_RECEIPT_V1.json"
    if output_path.exists() or output_path.is_symlink(): raise FileExistsError(output_path)
    receipt = json.loads(receipt_path.read_text())
    if (
        sha256_file(protocol_path) != receipt.get("protocol_sha256")
        or receipt.get("frozen_before_chain_typing") is not True
        or receipt.get("bound_labels_read") is not False
    ):
        raise ValueError("SABLE label-free protocol receipt differs")
    scripts = {
        "chain_map": args.tools_root / "build_abf30b_sable_chain_map_v1.sbatch",
        "embeddings": args.tools_root / "materialize_abf30b_sable_embeddings_v1.sbatch",
    }
    for path in scripts.values():
        if path.is_symlink() or not path.is_file(): raise ValueError(f"script missing: {path}")
    chain_job = submit(scripts["chain_map"])
    embedding_job = submit(scripts["embeddings"], chain_job)
    payload = {
        "format_version": "abf30b_sable_label_free_submission_receipt_v1",
        "status": "SUBMITTED_SABLE_LABEL_FREE_PREPARATION",
        "protocol_sha256": receipt["protocol_sha256"],
        "scripts": {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in scripts.items()},
        "jobs": {"chain_map": chain_job, "embeddings": embedding_job},
        "dependencies": {embedding_job: f"afterok:{chain_job}"},
        "bound_labels_read": False,
        "performance_read": False,
        "model_specific_measurement_submitted": False,
        "test_loaded": False,
        "temporal_holdout_loaded": False,
    }
    receipt_sha = publish(output_path, payload)
    print(json.dumps({"receipt_sha256": receipt_sha, **payload["jobs"]}, sort_keys=True))


if __name__ == "__main__": main()
