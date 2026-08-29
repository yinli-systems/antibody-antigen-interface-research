#!/usr/bin/env python3
"""Validate and seal an immutable FrameContact protocol receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SCHEMA = "framecontact_primary_protocol_v1"
RECEIPT_SCHEMA = "framecontact_protocol_freeze_receipt_v1"


def canonical_bytes(value):
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.receipt.exists() or arguments.receipt.is_symlink():
        raise FileExistsError(f"Refusing to overwrite frozen receipt: {arguments.receipt}")
    raw = arguments.protocol.read_bytes()
    payload = json.loads(raw)
    if payload.get("schema_version") != SCHEMA:
        raise ValueError("Unknown protocol schema")
    boundary = payload.get("evidence_boundary_at_freeze", {})
    if boundary.get("framecontact_pair_training_started") is not False:
        raise ValueError("Protocol is not eligible for a pre-training freeze")
    if boundary.get("new_antigen_fold_test_labels_accessed") is not False:
        raise ValueError("Antigen-fold test labels were already accessed")
    if len(payload.get("model_selection", {}).get("seeds", [])) < 5:
        raise ValueError("At least five seeds must be frozen")
    if "success_rule" not in payload.get("primary_endpoint", {}):
        raise ValueError("Primary endpoint lacks a success rule")
    canonical_sha = hashlib.sha256(canonical_bytes(payload)).hexdigest()
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "protocol_path": str(arguments.protocol),
        "protocol_file_sha256": hashlib.sha256(raw).hexdigest(),
        "protocol_canonical_sha256": canonical_sha,
        "schema_validated": True,
        "pretraining_boundary_validated": True,
        "test_access_boundary_validated": True,
        "amendment_rule": "create a new protocol version; never overwrite v1",
    }
    arguments.receipt.parent.mkdir(parents=True, exist_ok=True)
    arguments.receipt.write_bytes(canonical_bytes(receipt))
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
