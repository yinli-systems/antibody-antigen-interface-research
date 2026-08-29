#!/usr/bin/env python3
"""Freeze the pre-load safetensors path failure and authorize one exact retry."""

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
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--authority-sha256", required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--array-job-id", type=int, required=True)
    parser.add_argument("--failed-output-root", type=Path, required=True)
    parser.add_argument("--retry-sbatch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sha256_file(args.authority) != args.authority_sha256:
        raise RuntimeError("runtime authority SHA differs")
    logs = sorted(args.log_root.glob(f"runtime-canary-{args.array_job_id}_*.err"))
    if (
        len(logs) != 2
        or any("No module named 'safetensors'" not in path.read_text() for path in logs)
        or (args.failed_output_root.exists() and any(args.failed_output_root.iterdir()))
    ):
        raise RuntimeError("pre-load failure evidence differs")
    receipt = {
        "format_version": "paired_esmc_runtime_recovery_receipt_v1",
        "status": "AUTHORIZED_SINGLE_PATH_ONLY_RUNTIME_RETRY",
        "authority_sha256": args.authority_sha256,
        "failed_array_job_id": args.array_job_id,
        "stderr_sha256": [sha256_file(path) for path in logs],
        "failure_class": "safetensors_site_packages_path_before_checkpoint_open",
        "failed_output_root_empty": True,
        "retry_change": {
            "old_site": "/data/home/scxi253/.local/lib/python3.10/site-packages",
            "new_site": "/ssd/scxi253/model_deployment/envs/epiformer/lib/python3.10/site-packages",
            "new_output_root": "runtime-canary-v2",
            "sbatch_sha256": sha256_file(args.retry_sbatch),
        },
        "checkpoint_or_runtime_source_changed": False,
        "test_access": False,
        "training_started": False,
    }
    atomic_write(args.output, canonical_bytes(receipt))
    print(json.dumps({**receipt, "receipt_sha256": sha256_file(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
