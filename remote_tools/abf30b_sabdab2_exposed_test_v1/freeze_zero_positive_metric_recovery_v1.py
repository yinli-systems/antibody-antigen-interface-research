#!/usr/bin/env python3
"""Freeze eligible-complex AP recovery after zero-positive test failures."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


MEASUREMENT_PROTOCOL_SHA256 = "ed2b5d6a535518a6f8b3e72e764d7c37606e2214a5068eae0cef11dc2966d196"
CONTACT_EVALUATION_RECOVERY_SHA256 = "29fec12f6af8b5b31648ee32397238dd23959574b540d359d87c987940be159b"
CONTACT_EVALUATION_RECOVERY_RECEIPT_SHA256 = "244828e54b2b7055b07f559b0753677b0b9c7ac877067a3110ed3630ab879ef9"
BASELINE_STDERR_SHA256 = "1f84ddf7a0ddd0b36fdffb3d61245ac063f3ee318178135c7eed6c0018acf41b"
TOPK2_STDERR_SHA256 = "d0765a284debfab373df6d2f87c62557b97e43b9a488cea9195a2612a9ed97a8"


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
    parser.add_argument("--contact-evaluation-recovery", type=Path, required=True)
    parser.add_argument("--contact-evaluation-recovery-receipt", type=Path, required=True)
    parser.add_argument("--failed-stderr", type=Path, action="append", required=True)
    parser.add_argument("--failed-result-root", type=Path, required=True)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    args = parser.parse_args()
    if (
        sha256_file(args.measurement_protocol) != MEASUREMENT_PROTOCOL_SHA256
        or sha256_file(args.contact_evaluation_recovery) != CONTACT_EVALUATION_RECOVERY_SHA256
        or sha256_file(args.contact_evaluation_recovery_receipt) != CONTACT_EVALUATION_RECOVERY_RECEIPT_SHA256
    ):
        raise ValueError("parent measurement authority differs")
    parent = json.loads(args.contact_evaluation_recovery.read_text(encoding="utf-8"))
    if (
        parent.get("status") != "FROZEN_AFTER_TEST_CONTACT_AUTHORITY_ADAPTER_FOR_SAME_OPENING"
        or parent.get("adapter_contact_manifest_sha256") != "ad7201ecdf339b16a666addbaa5f775a4a1da03fdf0ea5bcdd68e8aa0f6e8e41"
        or parent.get("same_opening_retry") is not True
        or parent.get("scientific_protocol_changed") is not False
    ):
        raise ValueError("parent recovery content differs")
    if len(args.failed_stderr) != 6:
        raise ValueError("six failed stderr files are required")
    failures = []
    for index, path in enumerate(args.failed_stderr):
        expected = BASELINE_STDERR_SHA256 if index < 3 else TOPK2_STDERR_SHA256
        text = path.read_text(encoding="utf-8")
        if sha256_file(path) != expected or "ranked pair metrics require at least one positive contact" not in text:
            raise ValueError(f"zero-positive failure evidence differs: {index}")
        failures.append({"task_index": index, "stderr_sha256": expected})
    if args.failed_result_root.exists():
        published = list(args.failed_result_root.rglob("COMPONENT_RESULT.json")) + list(
            args.failed_result_root.rglob("COMPONENT_RECEIPT.json")
        )
        if published:
            raise ValueError("failed evaluation unexpectedly published component outputs")
    sources = []
    for path in sorted(args.source, key=lambda item: str(item.resolve())):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"recovery source missing: {path}")
        sources.append({"path": str(path.resolve()), "sha256": sha256_file(path)})
    recovery = {
        "format_version": "abf30b_sabdab2_exposed_test_zero_positive_metric_recovery_v1",
        "status": "FROZEN_AFTER_ZERO_POSITIVE_METRIC_FAILURE_FOR_SAME_OPENING",
        "measurement_protocol_sha256": MEASUREMENT_PROTOCOL_SHA256,
        "contact_authority_evaluation_recovery_sha256": CONTACT_EVALUATION_RECOVERY_SHA256,
        "failed_evaluation_array_job_id": "1501446",
        "cancelled_unstarted_analyzer_job_id": "1501447",
        "failed_components": failures,
        "failure_after_contact_artifact_read": True,
        "component_results_published": False,
        "failure_class": "ranked per-complex contact metric undefined for zero-positive complexes",
        "recovery_change": "retain all samples in pooled metrics; exclude zero-positive complexes only from per-complex AP and ranked-contact summaries",
        "pooled_metrics_retain_all_samples": True,
        "per_complex_ap_denominator": "complexes with at least one positive label for the evaluated task",
        "ranked_contact_denominator": "complexes with at least one positive contact",
        "partner_retrieval_unchanged": True,
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
        "format_version": "abf30b_sabdab2_exposed_test_zero_positive_metric_recovery_receipt_v1",
        "status": recovery["status"],
        "recovery_sha256": recovery_sha,
        "six_failures_verified": True,
        "component_results_published_before_failure": False,
        "same_opening_retry": True,
        "cumulative_test_access_ordinal": 3,
    })
    print(json.dumps({"status": recovery["status"], "recovery_sha256": recovery_sha, "receipt_sha256": receipt_sha}, sort_keys=True))


if __name__ == "__main__":
    main()
