#!/usr/bin/env python3
"""Bind a verified test contact adapter for the same frozen opening."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


MEASUREMENT_PROTOCOL_SHA256 = "ed2b5d6a535518a6f8b3e72e764d7c37606e2214a5068eae0cef11dc2966d196"
SPLIT_RECOVERY_SHA256 = "0a011f02ed8f4684701dfdb52018b3cc354e6a663e454143fd613c3a573c7e15"
STRUCTURE_MANIFEST_SHA256 = "b13ecb8a3b184f64a0ff509ddefea41a6aa7da226f710d3ddf79c55c84eef4cb"
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
    parser.add_argument("--split-recovery", type=Path, required=True)
    parser.add_argument("--adapter-recovery", type=Path, required=True)
    parser.add_argument("--adapter-recovery-receipt", type=Path, required=True)
    parser.add_argument("--adapter-root", type=Path, required=True)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    args = parser.parse_args()
    if sha256_file(args.measurement_protocol) != MEASUREMENT_PROTOCOL_SHA256:
        raise ValueError("measurement protocol SHA differs")
    if sha256_file(args.split_recovery) != SPLIT_RECOVERY_SHA256:
        raise ValueError("split recovery SHA differs")
    adapter_recovery = json.loads(args.adapter_recovery.read_text(encoding="utf-8"))
    adapter_recovery_receipt = json.loads(args.adapter_recovery_receipt.read_text(encoding="utf-8"))
    adapter_recovery_sha = sha256_file(args.adapter_recovery)
    if (
        adapter_recovery_receipt.get("recovery_sha256") != adapter_recovery_sha
        or adapter_recovery.get("status") not in {
            "FROZEN_AFTER_CONTACT_AUTHORITY_FAILURE_FOR_SAME_OPENING",
            "FROZEN_AFTER_STRUCTURE_RECEIPT_FIELD_ALIAS_FAILURE_FOR_SAME_OPENING",
        }
        or adapter_recovery.get("same_opening_retry") is not True
        or adapter_recovery.get("scientific_protocol_changed") is not False
    ):
        raise ValueError("adapter recovery authority differs")
    manifest_path = args.adapter_root / "CONTACT_CACHE_MANIFEST.json"
    receipt_path = args.adapter_root / "CONTACT_CACHE_RECEIPT.json"
    manifest_sha = sha256_file(manifest_path)
    receipt_sha = sha256_file(receipt_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("status") != "PASS_PROVENANCE_PRESERVING_TEST_CONTACT_AUTHORITY_ADAPTER"
        or receipt.get("manifest_sha256") != manifest_sha
        or receipt.get("recovery_protocol_sha256") != adapter_recovery_sha
        or receipt.get("record_count") != 549
        or receipt.get("all_pair_artifact_sha256_verified") is not True
        or receipt.get("all_pair_artifacts_hard_linked") is not True
        or receipt.get("pair_labels_regenerated") is not False
        or receipt.get("pair_label_artifacts_modified") is not False
        or manifest.get("counts", {}).get("by_role") != {"test": 549}
        or manifest.get("authority", {}).get("structure_manifest_sha256") != STRUCTURE_MANIFEST_SHA256
        or manifest.get("authority", {}).get("test_membership_source") != "authenticated predicted structure manifest test role"
        or manifest.get("authority", {}).get("experimental_bound_coordinates_used_as_model_input") is not False
        or manifest.get("authority", {}).get("selection_or_tuning") is not False
    ):
        raise ValueError("adapter output authority differs")
    sources = []
    for path in sorted(args.source, key=lambda item: str(item.resolve())):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"execution source missing: {path}")
        sources.append({"path": str(path.resolve()), "sha256": sha256_file(path)})
    recovery = {
        "format_version": "abf30b_sabdab2_exposed_test_contact_authority_evaluation_recovery_v1",
        "status": "FROZEN_AFTER_TEST_CONTACT_AUTHORITY_ADAPTER_FOR_SAME_OPENING",
        "measurement_protocol_sha256": MEASUREMENT_PROTOCOL_SHA256,
        "split_view_recovery_sha256": SPLIT_RECOVERY_SHA256,
        "adapter_recovery_sha256": adapter_recovery_sha,
        "adapter_contact_root": str(args.adapter_root.resolve()),
        "adapter_contact_manifest_sha256": manifest_sha,
        "adapter_contact_receipt_sha256": receipt_sha,
        "target_structure_manifest_sha256": STRUCTURE_MANIFEST_SHA256,
        "test_split_view_sha256": TEST_SPLIT_VIEW_SHA256,
        "test_complex_count": 549,
        "pair_label_artifacts_reused_byte_identically": True,
        "labels_regenerated": False,
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
    recovery_receipt_sha = publish(
        args.receipt_output,
        {
            "format_version": "abf30b_sabdab2_exposed_test_contact_authority_evaluation_recovery_receipt_v1",
            "status": recovery["status"],
            "recovery_sha256": recovery_sha,
            "adapter_contact_manifest_sha256": manifest_sha,
            "same_opening_retry": True,
            "cumulative_test_access_ordinal": 3,
        },
    )
    print(json.dumps({
        "status": recovery["status"],
        "recovery_sha256": recovery_sha,
        "receipt_sha256": recovery_receipt_sha,
        "adapter_contact_manifest_sha256": manifest_sha,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
