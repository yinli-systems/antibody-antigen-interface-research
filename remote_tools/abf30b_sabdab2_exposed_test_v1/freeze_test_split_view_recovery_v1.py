#!/usr/bin/env python3
"""Freeze the same-opening evaluator split-view execution recovery."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


PROTOCOL_SHA256 = "ed2b5d6a535518a6f8b3e72e764d7c37606e2214a5068eae0cef11dc2966d196"
BASELINE_STDERR_SHA256 = "e2945641833d8f1aa6532c24dd2d3b23b5ba6240d754bdf35e461c37df0162df"
TOPK2_STDERR_SHA256 = "32b11d35b67121fef56e0c12b4dba69b2c4a784525ef3fae855a398501ca6aed"
TEST_SPLIT_VIEW_SHA256 = "84acf74ce3debc3e176c996fd04b75510daf84183060d5d8353fc17758705a46"


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
    parser.add_argument("--measurement-protocol", type=Path, required=True)
    parser.add_argument("--structure-root", type=Path, required=True)
    parser.add_argument("--failed-stderr", type=Path, action="append", required=True)
    parser.add_argument("--failed-result-root", type=Path, required=True)
    parser.add_argument("--original-evaluator", type=Path, required=True)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.measurement_protocol.read_text(encoding="utf-8"))
    manifest_path = args.structure_root / "STRUCTURE_CACHE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    test_split = args.structure_root / "TEST_SPLIT_VIEW.csv"
    if (
        sha256_file(args.measurement_protocol) != PROTOCOL_SHA256
        or protocol.get("status") != "FROZEN_BEFORE_EXPOSED_TEST_LABEL_OPENING"
        or protocol.get("claim_boundary", {}).get("cumulative_test_access_ordinal") != 3
        or protocol.get("execution", {}).get("evaluator_sha256") != sha256_file(args.original_evaluator)
        or manifest.get("counts", {}).get("roles") != {"test": 549}
        or manifest.get("authority", {}).get("test_split_view_sha256") != TEST_SPLIT_VIEW_SHA256
        or sha256_file(test_split) != TEST_SPLIT_VIEW_SHA256
        or args.failed_result_root.exists()
        or args.failed_result_root.is_symlink()
    ):
        raise ValueError("split-view recovery authority differs")
    if len(args.failed_stderr) != 6:
        raise ValueError("six failed stderr files are required")
    failure_rows = []
    for index, path in enumerate(args.failed_stderr):
        expected = BASELINE_STDERR_SHA256 if index < 3 else TOPK2_STDERR_SHA256
        text = path.read_text(encoding="utf-8")
        if (
            sha256_file(path) != expected
            or "Split view and cohort cache identifiers differ" not in text
            or "Sabdab2ContactDataset" in text
        ):
            raise ValueError(f"failed evaluator evidence differs: {index}")
        failure_rows.append({"task_index": index, "stderr_sha256": expected})
    sources = []
    for path in sorted(args.source, key=lambda item: str(item.resolve())):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"recovery source missing: {path}")
        sources.append({"path": str(path.resolve()), "sha256": sha256_file(path)})
    recovery = {
        "format_version": "abf30b_sabdab2_exposed_test_split_view_recovery_v1",
        "status": "FROZEN_AFTER_PRELABEL_SPLIT_VIEW_FAILURE_FOR_SAME_OPENING",
        "measurement_protocol_sha256": PROTOCOL_SHA256,
        "original_evaluation_array_job_id": "1500929",
        "failed_components": failure_rows,
        "failure_before_contact_dataset_construction": True,
        "contact_label_artifacts_read": False,
        "failure_class": "full split supplied to test-only cohort cache",
        "recovery_change": "use manifest-authenticated test-only split view",
        "test_split_view_sha256": TEST_SPLIT_VIEW_SHA256,
        "test_complex_count": 549,
        "same_opening_retry": True,
        "cumulative_test_access_ordinal": 3,
        "fresh_blind_test": False,
        "scientific_protocol_changed": False,
        "cohort_changed": False,
        "checkpoint_or_routing_changed": False,
        "calibration_changed": False,
        "selection_performed_on_test": False,
        "sources": sources,
    }
    recovery_sha = publish(args.output, recovery)
    receipt_sha = publish(args.receipt_output, {
        "format_version": "abf30b_sabdab2_exposed_test_split_view_recovery_receipt_v1",
        "status": recovery["status"],
        "recovery_sha256": recovery_sha,
        "six_failures_verified": True,
        "contact_label_artifacts_read_before_failure": False,
        "same_opening_retry": True,
        "cumulative_test_access_ordinal": 3,
    })
    print(json.dumps({"status": recovery["status"], "recovery_sha256": recovery_sha, "receipt_sha256": receipt_sha}, sort_keys=True))


if __name__ == "__main__":
    main()
