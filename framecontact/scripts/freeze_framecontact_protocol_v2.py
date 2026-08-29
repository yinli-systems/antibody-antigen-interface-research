#!/usr/bin/env python3
"""Validate and seal FrameContact protocol v2 without mutating v1."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SCHEMA = "framecontact_primary_protocol_v2"
RECEIPT_SCHEMA = "framecontact_protocol_freeze_receipt_v2"


def canonical_bytes(value):
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--parent-protocol", type=Path, required=True)
    parser.add_argument("--parent-receipt", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.receipt.exists() or arguments.receipt.is_symlink():
        raise FileExistsError(f"Refusing to overwrite frozen receipt: {arguments.receipt}")
    raw = arguments.protocol.read_bytes()
    payload = json.loads(raw)
    if payload.get("schema_version") != SCHEMA:
        raise ValueError("Unknown v2 protocol schema")
    parent = payload.get("parent_protocol", {})
    if sha256_file(arguments.parent_protocol) != parent.get("sha256"):
        raise ValueError("Parent protocol digest differs")
    if sha256_file(arguments.parent_receipt) != parent.get("receipt_sha256"):
        raise ValueError("Parent receipt digest differs")
    boundary = payload.get("evidence_boundary_at_freeze", {})
    for key in (
        "framecontact_pair_training_started",
        "new_antigen_fold_test_labels_accessed",
        "new_external_test_predictions_accessed",
    ):
        if boundary.get(key) is not False:
            raise ValueError(f"Protocol is not eligible for freeze: {key}")
    panel = payload.get("data", {}).get("reciprocal_partner_panel", {})
    if panel.get("format_version") != "framecontact_reciprocal_partner_panel_v2":
        raise ValueError("Protocol does not require reciprocal panel v2")
    if payload.get("training_objective", {}).get("specificity_score_source") != "interaction_only":
        raise ValueError("v2 must exclude unary propensity from specificity scoring")
    if len(payload.get("model_selection", {}).get("seeds", [])) < 5:
        raise ValueError("At least five seeds must be frozen")
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "protocol_path": str(arguments.protocol),
        "protocol_file_sha256": hashlib.sha256(raw).hexdigest(),
        "protocol_canonical_sha256": hashlib.sha256(canonical_bytes(payload)).hexdigest(),
        "parent_protocol_sha256": parent["sha256"],
        "parent_receipt_sha256": parent["receipt_sha256"],
        "reciprocal_panel_validated": True,
        "interaction_only_specificity_validated": True,
        "pretraining_boundary_validated": True,
        "test_access_boundary_validated": True,
    }
    arguments.receipt.parent.mkdir(parents=True, exist_ok=True)
    arguments.receipt.write_bytes(canonical_bytes(receipt))
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
