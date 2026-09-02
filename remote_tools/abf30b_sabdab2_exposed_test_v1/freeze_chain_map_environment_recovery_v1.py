#!/usr/bin/env python3
"""Freeze the original two-environment chain-map/embedding topology recovery."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


FAILED_STDERR_SHA256 = "e0bdedd2801cd346d4657126ae11c56ba917b976d5d78e3e8128b22b15397874"
STRUCTURE_MANIFEST_SHA256 = "b13ecb8a3b184f64a0ff509ddefea41a6aa7da226f710d3ddf79c55c84eef4cb"
STRUCTURE_RECEIPT_SHA256 = "e043be4d266ea72833b9c7fc7fa7d19d1f22f46a97555d5bb221416f145ecb69"
STRUCTURE_BINDING_SHA256 = "165690c13a2d75f50dab8262cf48fbf3e1089e4f9cc5895ddfa4b85e3e09aa06"


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
    parser.add_argument("--failed-stderr", type=Path, required=True)
    parser.add_argument("--structure-root", type=Path, required=True)
    parser.add_argument("--receipt-adapter", type=Path, required=True)
    parser.add_argument("--anarcii-python", type=Path, required=True)
    parser.add_argument("--contact-python", type=Path, required=True)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    args = parser.parse_args()
    prep = json.loads(args.prep_protocol.read_text(encoding="utf-8"))
    adapter = json.loads(args.receipt_adapter.read_text(encoding="utf-8"))
    stderr_text = args.failed_stderr.read_text(encoding="utf-8")
    if (
        prep.get("status") != "FROZEN_BEFORE_EXPOSED_TEST_STRUCTURE_GENERATION_AND_LABEL_ARTIFACT_READ"
        or sha256_file(args.failed_stderr) != FAILED_STDERR_SHA256
        or "ModuleNotFoundError: No module named 'anarcii'" not in stderr_text
        or adapter.get("task_labels_read") is not False
        or adapter.get("task_labels_loaded") is not False
        or not args.anarcii_python.is_file()
        or not args.contact_python.is_file()
        or sha256_file(args.structure_root / "STRUCTURE_CACHE_MANIFEST.json") != STRUCTURE_MANIFEST_SHA256
        or sha256_file(args.structure_root / "STRUCTURE_CACHE_RECEIPT.json") != STRUCTURE_RECEIPT_SHA256
        or sha256_file(args.structure_root / "TEST_STRUCTURE_CACHE_BINDING_V1.json") != STRUCTURE_BINDING_SHA256
    ):
        raise ValueError("environment recovery authority differs")
    sources = []
    for path in sorted(args.source, key=lambda item: str(item.resolve())):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"source missing: {path}")
        sources.append({"path": str(path.resolve()), "sha256": sha256_file(path)})
    protocol = {
        "format_version": "abf30b_sabdab2_exposed_test_chain_map_environment_recovery_v1",
        "status": "FROZEN_AFTER_CHAIN_MAP_ENVIRONMENT_FAILURE_BEFORE_LABEL_ARTIFACT_READ",
        "prep_protocol_sha256": sha256_file(args.prep_protocol),
        "failed_job_id": "1500727",
        "failed_job_exit_code": "1:0",
        "failed_stderr_sha256": FAILED_STDERR_SHA256,
        "failure_before_chain_typing": True,
        "failure_before_dcp_restore": True,
        "failure_class": "ANARCII unavailable in contact environment",
        "recovery_topology": [
            {"stage": "chain_map", "python": str(args.anarcii_python.resolve()), "environment": "ANARCII 2.0.8 frozen"},
            {"stage": "embedding", "python": str(args.contact_python.resolve()), "environment": "contact Python 3.10 frozen"},
        ],
        "structure_manifest_sha256": STRUCTURE_MANIFEST_SHA256,
        "structure_receipt_sha256": STRUCTURE_RECEIPT_SHA256,
        "structure_binding_sha256": STRUCTURE_BINDING_SHA256,
        "receipt_adapter_sha256": sha256_file(args.receipt_adapter),
        "scientific_protocol_changed": False,
        "cohort_changed": False,
        "structure_or_prediction_files_changed": False,
        "checkpoint_or_routing_changed": False,
        "test_label_artifacts_read": False,
        "fresh_blind_test": False,
        "cumulative_test_access_ordinal": 3,
        "sources": sources,
    }
    protocol_sha = publish(args.output, protocol)
    receipt_sha = publish(args.receipt_output, {
        "format_version": "abf30b_sabdab2_exposed_test_chain_map_environment_recovery_receipt_v1",
        "status": protocol["status"],
        "protocol_sha256": protocol_sha,
        "structure_cache_verified": True,
        "scientific_protocol_changed": False,
        "test_label_artifacts_read": False,
    })
    print(json.dumps({"status": protocol["status"], "protocol_sha256": protocol_sha, "receipt_sha256": receipt_sha}, sort_keys=True))


if __name__ == "__main__":
    main()
