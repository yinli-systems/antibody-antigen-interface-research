#!/usr/bin/env python3
"""Derive the chain-map receipt alias without reading labels or changing data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("ascii")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery-protocol", type=Path, required=True)
    parser.add_argument("--recovery-protocol-sha256", required=True)
    parser.add_argument("--input-plan", type=Path, required=True)
    parser.add_argument("--input-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    if sha256_file(args.recovery_protocol) != args.recovery_protocol_sha256:
        raise ValueError("recovery protocol SHA differs")
    recovery = json.loads(args.recovery_protocol.read_text(encoding="utf-8"))
    plan = json.loads(args.input_plan.read_text(encoding="utf-8"))
    receipt = json.loads(args.input_receipt.read_text(encoding="utf-8"))
    if (
        recovery.get("status") != "FROZEN_BEFORE_EXPOSED_TEST_LABEL_ARTIFACT_READ"
        or recovery.get("input_plan_sha256") != sha256_file(args.input_plan)
        or recovery.get("input_receipt_sha256") != sha256_file(args.input_receipt)
        or plan.get("status") != "PASS_EXPOSED_TEST_LABEL_FREE_INPUT_PLAN"
        or receipt.get("plan_sha256") != sha256_file(args.input_plan)
        or receipt.get("task_labels_loaded") is not False
        or receipt.get("test_access") is not True
    ):
        raise ValueError("receipt adapter authority differs")
    payload = dict(receipt)
    payload.update({
        "format_version": "sabdab2_exposed_test_input_plan_chain_map_adapter_v1",
        "status": "PASS_EXPOSED_TEST_SEQUENCE_RECEIPT_FIELD_ALIAS_ADAPTER",
        "source_receipt_sha256": sha256_file(args.input_receipt),
        "recovery_protocol_sha256": args.recovery_protocol_sha256,
        "task_labels_read": False,
    })
    raw = canonical_bytes(payload)
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
    print(json.dumps({"status": payload["status"], "adapter_sha256": hashlib.sha256(raw).hexdigest()}, sort_keys=True))


if __name__ == "__main__":
    main()
