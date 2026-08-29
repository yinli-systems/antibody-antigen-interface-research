#!/usr/bin/env python3
"""Seal the development-only FrameContact residue-head ablation protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--model-parent", type=Path, required=True)
    parser.add_argument("--split-parent", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.receipt.exists() or arguments.receipt.is_symlink():
        raise FileExistsError(f"Refusing to overwrite receipt: {arguments.receipt}")
    raw = arguments.protocol.read_bytes()
    protocol = json.loads(raw)
    if protocol.get("schema_version") != "framecontact_residue_head_exploratory_protocol_v1":
        raise ValueError("Unknown residue-head protocol schema")
    parents = protocol.get("parents", {})
    if _sha(arguments.model_parent) != parents.get("model_protocol", {}).get("sha256"):
        raise ValueError("Model parent digest differs")
    if _sha(arguments.split_parent) != parents.get("antigen_fold_protocol", {}).get("sha256"):
        raise ValueError("Split parent digest differs")
    boundary = protocol.get("evidence_boundary_at_freeze", {})
    for key in (
        "this_residue_head_ablation_training_started",
        "this_residue_head_ablation_validation_predictions_accessed",
        "sabdab2_antigen_fold_test_predictions_accessed",
        "sabdab2_antigen_fold_test_metrics_accessed",
        "asep_official_test_predictions_accessed",
    ):
        if boundary.get(key) is not False:
            raise ValueError(f"Ineligible evidence boundary: {key}")
    if protocol.get("development_matrix") != [
        "framecontact_unary_only",
        "framecontact_contact_marginal",
        "framecontact_contact_hybrid",
    ]:
        raise ValueError("Residue-head matrix differs")
    if len(protocol.get("model_selection", {}).get("seeds", [])) != 5:
        raise ValueError("Exactly five seeds must be frozen")
    if protocol.get("data", {}).get("test_role_forbidden") is not True:
        raise ValueError("Test role must remain forbidden")
    receipt = {
        "schema_version": "framecontact_residue_head_exploratory_receipt_v1",
        "protocol_path": str(arguments.protocol),
        "protocol_file_sha256": hashlib.sha256(raw).hexdigest(),
        "protocol_canonical_sha256": hashlib.sha256(_canonical(protocol)).hexdigest(),
        "parent_sha256": {
            "model": _sha(arguments.model_parent),
            "antigen_fold": _sha(arguments.split_parent),
        },
        "three_condition_matrix_validated": True,
        "five_seed_rule_validated": True,
        "development_only_boundary_validated": True,
        "test_prediction_boundary_validated": True,
    }
    arguments.receipt.parent.mkdir(parents=True, exist_ok=True)
    arguments.receipt.write_bytes(_canonical(receipt))
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
