#!/usr/bin/env python3
"""Freeze a single-scope recovery for the PyTorch DDP first-no_sync assertion."""

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


def atomic_write(path: Path, value: object) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(canonical_bytes(value)); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-protocol", type=Path, required=True)
    parser.add_argument("--v1-receipt", type=Path, required=True)
    parser.add_argument("--failure", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    expected_v1_protocol = "1f93208277675924e8a874d08c7ba3ad2c0632683b298bc35b81eb4f3607a05c"
    expected_v1_receipt = "52f38a454d110f043eeb2fa169d6cdccbf5a4c251abfebce9503ff5bfd800a50"
    if sha256_file(args.v1_protocol) != expected_v1_protocol or sha256_file(args.v1_receipt) != expected_v1_receipt:
        raise RuntimeError("V1 canary authority differs")
    failure = json.loads(args.failure.read_text())
    if (
        failure.get("status") != "FAIL_CLOSED_PYTORCH_DDP_STATIC_GRAPH_FIRST_NO_SYNC_ASSERT"
        or failure.get("outputs_written") is not False
        or failure.get("checkpoint_written") is not False
        or failure.get("task_labels_read") is not False
        or failure.get("sabdab2_test_access") is not False
    ):
        raise RuntimeError("V1 failure evidence differs")
    protocol = json.loads(args.v1_protocol.read_text())
    protocol["format_version"] = "paired_esmc_adaptation_canary_protocol_v2"
    protocol["protocol_status"] = "frozen_measurement_only_recovery_after_ddp_first_no_sync_assert"
    protocol["supersedes_failure"] = {
        "v1_protocol_sha256": expected_v1_protocol,
        "v1_receipt_sha256": expected_v1_receipt,
        "failure_receipt_uri": str(args.failure),
        "failure_receipt_sha256": sha256_file(args.failure),
        "failed_job_array": "1474947_[0-1]",
        "failure_before_optimizer_update": True,
    }
    protocol["system_recovery"] = {
        "single_change": (
            "Synchronize all four microbatches of optimizer update zero so DDP static_graph "
            "initializes outside no_sync; updates 1-11 retain the frozen no_sync accumulation."
        ),
        "gradient_equivalence": (
            "Averaging each rank's microbatch gradient before accumulation is algebraically "
            "equivalent to averaging the accumulated rank-local gradient once."
        ),
        "public_issue": "https://github.com/pytorch/pytorch/issues/143580",
        "data_model_mask_optimizer_seed_batch_and_update_count_unchanged": True,
    }
    protocol["implementation"]["runner_sha256"] = sha256_file(args.runner)
    protocol["implementation"]["launcher_sha256"] = sha256_file(args.launcher)
    protocol["claim_boundary"]["measurement_only_system_recovery"] = True
    atomic_write(args.output, protocol)
    receipt = {
        "format_version": "paired_esmc_adaptation_canary_protocol_receipt_v2",
        "status": "FROZEN_MEASUREMENT_ONLY_DDP_RECOVERY_BEFORE_TRAINING",
        "protocol_sha256": sha256_file(args.output),
        "v1_failure_receipt_sha256": sha256_file(args.failure),
        "production_training_authorized": False,
        "task_labels_read": False,
        "sabdab2_test_access": False,
    }
    atomic_write(args.receipt, receipt)
    print(json.dumps({"protocol_sha256": receipt["protocol_sha256"], "receipt_sha256": sha256_file(args.receipt)}, sort_keys=True))


if __name__ == "__main__":
    main()
