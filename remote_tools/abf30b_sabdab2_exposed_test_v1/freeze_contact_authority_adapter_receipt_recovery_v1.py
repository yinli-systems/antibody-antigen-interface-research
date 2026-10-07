#!/usr/bin/env python3
"""Freeze the structure-receipt field-alias recovery before label reads."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


PARENT_RECOVERY_SHA256 = "5d18d8d753b52b4dfaf385766f68f176351f41b98806990e577ecf5b923fb3f3"
PARENT_RECEIPT_SHA256 = "6475828bd74f2a10dfa924050311bf610af638fc04430d6151eebc356b72ec7e"
FAILED_STDERR_SHA256 = "8087614ce4b08b3bccf2201007c63b2c2cc0726083142394339bd7b03dada419"
EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
STRUCTURE_RECEIPT_SHA256 = "e043be4d266ea72833b9c7fc7fa7d19d1f22f46a97555d5bb221416f145ecb69"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("ascii")


def publish(path: Path, value: Any) -> str:
    payload = canonical_bytes(value)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-recovery", type=Path, required=True)
    parser.add_argument("--parent-receipt", type=Path, required=True)
    parser.add_argument("--structure-receipt", type=Path, required=True)
    parser.add_argument("--failed-stderr", type=Path, required=True)
    parser.add_argument("--failed-stdout", type=Path, required=True)
    parser.add_argument("--adapter-output-root", type=Path, required=True)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    args = parser.parse_args()
    if (
        sha256_file(args.parent_recovery) != PARENT_RECOVERY_SHA256
        or sha256_file(args.parent_receipt) != PARENT_RECEIPT_SHA256
        or sha256_file(args.structure_receipt) != STRUCTURE_RECEIPT_SHA256
        or sha256_file(args.failed_stderr) != FAILED_STDERR_SHA256
        or sha256_file(args.failed_stdout) != EMPTY_SHA256
        or args.adapter_output_root.exists()
        or args.adapter_output_root.is_symlink()
    ):
        raise ValueError("field-alias failure authority differs")
    parent = json.loads(args.parent_recovery.read_text(encoding="utf-8"))
    structure_receipt = json.loads(args.structure_receipt.read_text(encoding="utf-8"))
    failure_text = args.failed_stderr.read_text(encoding="utf-8")
    if (
        parent.get("status") != "FROZEN_AFTER_CONTACT_AUTHORITY_FAILURE_FOR_SAME_OPENING"
        or parent.get("same_opening_retry") is not True
        or "input authority content differs" not in failure_text
        or structure_receipt.get("status") != "PASS_EXPOSED_TEST_PREDICTED_STRUCTURE_CACHE"
        or structure_receipt.get("task_labels_read") is not False
        or structure_receipt.get("temporal_holdout_access") is not False
        or "selection_or_tuning" in structure_receipt
    ):
        raise ValueError("field-alias recovery content differs")
    sources = []
    for path in sorted(args.source, key=lambda item: str(item.resolve())):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"recovery source missing: {path}")
        sources.append({"path": str(path.resolve()), "sha256": sha256_file(path)})
    recovery = {
        "format_version": "abf30b_sabdab2_exposed_test_contact_authority_adapter_receipt_recovery_v1",
        "status": "FROZEN_AFTER_STRUCTURE_RECEIPT_FIELD_ALIAS_FAILURE_FOR_SAME_OPENING",
        "parent_recovery_sha256": PARENT_RECOVERY_SHA256,
        "failed_prepare_job_id": "1501359",
        "cancelled_unstarted_evaluation_job_id": "1501360",
        "cancelled_unstarted_analyzer_job_id": "1501361",
        "failed_stderr_sha256": FAILED_STDERR_SHA256,
        "failure_before_contact_artifact_read": True,
        "failure_class": "optional selection_or_tuning field absent from structure receipt",
        "recovery_change": "verify selection_or_tuning=false from authenticated structure manifest and require explicit receipt task-label/temporal fields",
        "structure_receipt_sha256": STRUCTURE_RECEIPT_SHA256,
        "adapter_output_root": str(args.adapter_output_root.resolve()),
        "same_opening_retry": True,
        "cumulative_test_access_ordinal": 3,
        "fresh_blind_test": False,
        "scientific_protocol_changed": False,
        "cohort_changed": False,
        "checkpoint_or_routing_changed": False,
        "calibration_changed": False,
        "labels_regenerated": False,
        "selection_performed_on_test": False,
        "sources": sources,
    }
    recovery_sha = publish(args.output, recovery)
    receipt_sha = publish(args.receipt_output, {
        "format_version": "abf30b_sabdab2_exposed_test_contact_authority_adapter_receipt_recovery_receipt_v1",
        "status": recovery["status"],
        "recovery_sha256": recovery_sha,
        "failure_before_contact_artifact_read": True,
        "same_opening_retry": True,
        "cumulative_test_access_ordinal": 3,
    })
    print(json.dumps({"status": recovery["status"], "recovery_sha256": recovery_sha, "receipt_sha256": receipt_sha}, sort_keys=True))


if __name__ == "__main__":
    main()
