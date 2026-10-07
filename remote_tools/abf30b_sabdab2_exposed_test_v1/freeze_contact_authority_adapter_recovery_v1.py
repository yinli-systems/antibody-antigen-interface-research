#!/usr/bin/env python3
"""Freeze a same-opening, test-only contact-authority adapter recovery."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


MEASUREMENT_PROTOCOL_SHA256 = "ed2b5d6a535518a6f8b3e72e764d7c37606e2214a5068eae0cef11dc2966d196"
SPLIT_RECOVERY_SHA256 = "0a011f02ed8f4684701dfdb52018b3cc354e6a663e454143fd613c3a573c7e15"
SPLIT_RECOVERY_RECEIPT_SHA256 = "1ace2185125ab611c8ef9edbda9d8905bd5878f2c8586e29fa9715eb89fba364"
SOURCE_CONTACT_MANIFEST_SHA256 = "d91ed42dcc05ee17cc880f7fa6124a7699f11a3beea75aab13f0b9502c274a33"
STRUCTURE_MANIFEST_SHA256 = "b13ecb8a3b184f64a0ff509ddefea41a6aa7da226f710d3ddf79c55c84eef4cb"
STRUCTURE_RECEIPT_SHA256 = "e043be4d266ea72833b9c7fc7fa7d19d1f22f46a97555d5bb221416f145ecb69"
TEST_SPLIT_VIEW_SHA256 = "84acf74ce3debc3e176c996fd04b75510daf84183060d5d8353fc17758705a46"
BASELINE_STDERR_SHA256 = "8a80852f1d913bd3d99a725a705e6bb4c60e880ce4c81f570e7e8efc17dd0d83"
TOPK2_STDERR_SHA256 = "9aaef597a0d82f9b54db3cbec5ab512af85b60ca8ab7fa1013236cc1caf77a81"


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


def require(path: Path, expected: str) -> None:
    if path.is_symlink() or not path.is_file() or sha256_file(path) != expected:
        raise ValueError(f"authority SHA differs: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurement-protocol", type=Path, required=True)
    parser.add_argument("--split-recovery", type=Path, required=True)
    parser.add_argument("--split-recovery-receipt", type=Path, required=True)
    parser.add_argument("--source-contact-root", type=Path, required=True)
    parser.add_argument("--structure-root", type=Path, required=True)
    parser.add_argument("--failed-stderr", type=Path, action="append", required=True)
    parser.add_argument("--failed-result-root", type=Path, required=True)
    parser.add_argument("--adapter-output-root", type=Path, required=True)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    args = parser.parse_args()

    require(args.measurement_protocol, MEASUREMENT_PROTOCOL_SHA256)
    require(args.split_recovery, SPLIT_RECOVERY_SHA256)
    require(args.split_recovery_receipt, SPLIT_RECOVERY_RECEIPT_SHA256)
    contact_manifest_path = args.source_contact_root / "CONTACT_CACHE_MANIFEST.json"
    structure_manifest_path = args.structure_root / "STRUCTURE_CACHE_MANIFEST.json"
    structure_receipt_path = args.structure_root / "STRUCTURE_CACHE_RECEIPT.json"
    test_split_path = args.structure_root / "TEST_SPLIT_VIEW.csv"
    require(contact_manifest_path, SOURCE_CONTACT_MANIFEST_SHA256)
    require(structure_manifest_path, STRUCTURE_MANIFEST_SHA256)
    require(structure_receipt_path, STRUCTURE_RECEIPT_SHA256)
    require(test_split_path, TEST_SPLIT_VIEW_SHA256)

    protocol = json.loads(args.measurement_protocol.read_text(encoding="utf-8"))
    split_recovery = json.loads(args.split_recovery.read_text(encoding="utf-8"))
    structure = json.loads(structure_manifest_path.read_text(encoding="utf-8"))
    if (
        protocol.get("status") != "FROZEN_BEFORE_EXPOSED_TEST_LABEL_OPENING"
        or protocol.get("claim_boundary", {}).get("cumulative_test_access_ordinal") != 3
        or protocol.get("claim_boundary", {}).get("selection_or_tuning_from_this_measurement") is not False
        or split_recovery.get("same_opening_retry") is not True
        or split_recovery.get("scientific_protocol_changed") is not False
        or structure.get("counts", {}).get("roles") != {"test": 549}
        or structure.get("authority", {}).get("test_split_view_sha256") != TEST_SPLIT_VIEW_SHA256
        or structure.get("authority", {}).get("experimental_bound_complex_coordinates_read") is not False
        or structure.get("authority", {}).get("cross_partner_coordinates_or_distances_used") is not False
    ):
        raise ValueError("measurement or structure authority differs")
    if args.adapter_output_root.exists() or args.adapter_output_root.is_symlink():
        raise FileExistsError(args.adapter_output_root)
    if len(args.failed_stderr) != 6:
        raise ValueError("six failed stderr files are required")
    failure_rows = []
    for index, path in enumerate(args.failed_stderr):
        expected = BASELINE_STDERR_SHA256 if index < 3 else TOPK2_STDERR_SHA256
        text = path.read_text(encoding="utf-8")
        if (
            sha256_file(path) != expected
            or "Contact and structure cache authorities differ" not in text
            or ".npz" in text
        ):
            raise ValueError(f"failed evaluator evidence differs: {index}")
        failure_rows.append({"task_index": index, "stderr_sha256": expected})
    if args.failed_result_root.exists():
        forbidden = list(args.failed_result_root.rglob("COMPONENT_RESULT.json")) + list(
            args.failed_result_root.rglob("COMPONENT_RECEIPT.json")
        )
        if forbidden:
            raise ValueError("failed evaluation unexpectedly published component outputs")

    sources = []
    for path in sorted(args.source, key=lambda item: str(item.resolve())):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"recovery source missing: {path}")
        sources.append({"path": str(path.resolve()), "sha256": sha256_file(path)})
    if len({row["path"] for row in sources}) != len(sources):
        raise ValueError("duplicate recovery source")

    recovery = {
        "format_version": "abf30b_sabdab2_exposed_test_contact_authority_adapter_recovery_v1",
        "status": "FROZEN_AFTER_CONTACT_AUTHORITY_FAILURE_FOR_SAME_OPENING",
        "measurement_protocol_sha256": MEASUREMENT_PROTOCOL_SHA256,
        "split_view_recovery_sha256": SPLIT_RECOVERY_SHA256,
        "failed_evaluation_array_job_id": "1501081",
        "cancelled_unstarted_analyzer_job_id": "1501082",
        "failed_components": failure_rows,
        "failure_class": "source contact manifest bound to experimental structure authority",
        "failure_before_contact_artifact_read": True,
        "contact_manifest_metadata_read": True,
        "contact_label_artifacts_read_before_failure": False,
        "recovery_change": "test-only metadata authority adapter with byte-identical hard-linked pair-label artifacts",
        "source_contact_manifest_sha256": SOURCE_CONTACT_MANIFEST_SHA256,
        "source_contact_structure_manifest_sha256": "edf74fd704b8b3b4ec9f7c9779e6a548b51e6d18d182642b5323ebc9ea2033f0",
        "target_structure_manifest_sha256": STRUCTURE_MANIFEST_SHA256,
        "target_structure_receipt_sha256": STRUCTURE_RECEIPT_SHA256,
        "test_split_view_sha256": TEST_SPLIT_VIEW_SHA256,
        "adapter_output_root": str(args.adapter_output_root.resolve()),
        "test_complex_count": 549,
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
    receipt_sha = publish(
        args.receipt_output,
        {
            "format_version": "abf30b_sabdab2_exposed_test_contact_authority_adapter_recovery_receipt_v1",
            "status": recovery["status"],
            "recovery_sha256": recovery_sha,
            "six_failures_verified": True,
            "failure_before_contact_artifact_read": True,
            "same_opening_retry": True,
            "cumulative_test_access_ordinal": 3,
        },
    )
    print(json.dumps({"status": recovery["status"], "recovery_sha256": recovery_sha, "receipt_sha256": receipt_sha}, sort_keys=True))


if __name__ == "__main__":
    main()
