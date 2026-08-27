#!/usr/bin/env python3
"""Freeze the post-strict-load meta-buffer failure before a new authority."""

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-authority", type=Path, required=True)
    parser.add_argument("--old-authority-sha256", required=True)
    parser.add_argument("--new-authority", type=Path, required=True)
    parser.add_argument("--new-authority-sha256", required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--array-job-id", type=int, required=True)
    parser.add_argument("--failed-output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sha256_file(args.old_authority) != args.old_authority_sha256:
        raise RuntimeError("old authority SHA differs")
    if sha256_file(args.new_authority) != args.new_authority_sha256:
        raise RuntimeError("new authority SHA differs")
    logs = sorted(args.log_root.glob(f"runtime-canary-{args.array_job_id}_*.err"))
    expected = "Cannot copy out of meta tensor"
    if (
        len(logs) != 2
        or any(expected not in path.read_text() for path in logs)
        or (args.failed_output_root.exists() and any(args.failed_output_root.iterdir()))
    ):
        raise RuntimeError("meta-buffer failure evidence differs")
    receipt = {
        "format_version": "paired_esmc_runtime_meta_buffer_failure_v1",
        "status": "FAIL_CLOSED_BEFORE_INFERENCE_SUPERSEDED_BY_NEW_RUNTIME_AUTHORITY",
        "old_authority_sha256": args.old_authority_sha256,
        "new_authority_sha256": args.new_authority_sha256,
        "failed_array_job_id": args.array_job_id,
        "stderr_sha256": [sha256_file(path) for path in logs],
        "failure_class": "nonpersistent_rope_buffer_remained_meta_after_strict_parameter_load",
        "strict_parameter_load_reached": True,
        "failed_output_root_empty": True,
        "checkpoint_changed": False,
        "fix": "materialize the official nonpersistent RoPE inv_freq formula on CPU before GPU transfer",
        "test_access": False,
        "training_started": False,
    }
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(canonical_bytes(receipt))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, args.output)
    print(json.dumps({**receipt, "receipt_sha256": sha256_file(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
