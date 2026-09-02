#!/usr/bin/env python3
"""Freeze a field-alias-only sequence-receipt recovery before label opening."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
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
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prep-protocol", type=Path, required=True)
    parser.add_argument("--prep-receipt", type=Path, required=True)
    parser.add_argument("--input-plan", type=Path, required=True)
    parser.add_argument("--input-receipt", type=Path, required=True)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    args = parser.parse_args()
    prep_sha = sha256_file(args.prep_protocol)
    prep = json.loads(args.prep_protocol.read_text(encoding="utf-8"))
    prep_receipt = json.loads(args.prep_receipt.read_text(encoding="utf-8"))
    plan_sha = sha256_file(args.input_plan)
    plan = json.loads(args.input_plan.read_text(encoding="utf-8"))
    input_receipt_sha = sha256_file(args.input_receipt)
    input_receipt = json.loads(args.input_receipt.read_text(encoding="utf-8"))
    if (
        prep.get("status") != "FROZEN_BEFORE_EXPOSED_TEST_STRUCTURE_GENERATION_AND_LABEL_ARTIFACT_READ"
        or prep_receipt.get("protocol_sha256") != prep_sha
        or prep_receipt.get("frozen_before_test_label_artifact_read") is not True
        or plan.get("status") != "PASS_EXPOSED_TEST_LABEL_FREE_INPUT_PLAN"
        or input_receipt.get("plan_sha256") != plan_sha
        or input_receipt.get("task_labels_loaded") is not False
        or input_receipt.get("test_access") is not True
    ):
        raise ValueError("pre-label recovery authority differs")
    sources = []
    for path in sorted(args.source, key=lambda item: str(item.resolve())):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"recovery source missing or symlinked: {path}")
        sources.append({"path": str(path.resolve()), "sha256": sha256_file(path)})
    protocol = {
        "format_version": "abf30b_sabdab2_exposed_test_chain_map_receipt_recovery_v1",
        "status": "FROZEN_BEFORE_EXPOSED_TEST_LABEL_ARTIFACT_READ",
        "prep_protocol_sha256": prep_sha,
        "input_plan_sha256": plan_sha,
        "input_receipt_sha256": input_receipt_sha,
        "recovery_type": "pure-data field alias adapter",
        "source_field": "task_labels_loaded",
        "derived_alias_field": "task_labels_read",
        "derived_alias_value": False,
        "scientific_protocol_changed": False,
        "cohort_changed": False,
        "checkpoint_or_routing_changed": False,
        "test_label_artifacts_read": False,
        "fresh_blind_test": False,
        "cumulative_test_access_ordinal": 3,
        "sources": sources,
    }
    protocol_sha = publish(args.output, protocol)
    receipt_sha = publish(args.receipt_output, {
        "format_version": "abf30b_sabdab2_exposed_test_chain_map_receipt_recovery_receipt_v1",
        "status": protocol["status"],
        "protocol_sha256": protocol_sha,
        "frozen_before_test_label_artifact_read": True,
        "scientific_protocol_changed": False,
        "test_label_artifacts_read": False,
    })
    print(json.dumps({"status": protocol["status"], "protocol_sha256": protocol_sha, "receipt_sha256": receipt_sha}, sort_keys=True))


if __name__ == "__main__":
    main()
