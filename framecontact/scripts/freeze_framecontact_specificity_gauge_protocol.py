#!/usr/bin/env python3
"""Seal the development-only FrameContact specificity-gauge protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED_MATRIX = [
    "framecontact_gauge_marginal",
    "framecontact_gauge_hybrid",
    "framecontact_gauge_hybrid_contrastive",
]


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
    parser.add_argument("--residue-head-parent", type=Path, required=True)
    parser.add_argument("--split-parent", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.receipt.exists() or arguments.receipt.is_symlink():
        raise FileExistsError(f"Refusing to overwrite receipt: {arguments.receipt}")

    raw = arguments.protocol.read_bytes()
    protocol = json.loads(raw)
    if (
        protocol.get("schema_version")
        != "framecontact_specificity_gauge_exploratory_protocol_v1"
    ):
        raise ValueError("Unknown specificity-gauge protocol schema")

    parents = protocol.get("parents", {})
    residue_sha = _sha(arguments.residue_head_parent)
    split_sha = _sha(arguments.split_parent)
    if residue_sha != parents.get("residue_head_protocol", {}).get("sha256"):
        raise ValueError("Residue-head parent digest differs")
    if split_sha != parents.get("antigen_fold_protocol", {}).get("sha256"):
        raise ValueError("Antigen-fold parent digest differs")

    if protocol.get("development_matrix") != EXPECTED_MATRIX:
        raise ValueError("Specificity-gauge matrix differs")
    if protocol.get("model_selection", {}).get("seeds") != [17, 42, 73, 101, 137]:
        raise ValueError("The frozen five-seed panel differs")
    if protocol.get("data", {}).get("test_role_forbidden") is not True:
        raise ValueError("Test role must remain forbidden")

    boundary = protocol.get("evidence_boundary_at_freeze", {})
    for key in (
        "this_gauge_ablation_training_started",
        "this_gauge_ablation_validation_predictions_accessed",
        "sabdab2_antigen_fold_test_predictions_accessed",
        "sabdab2_antigen_fold_test_metrics_accessed",
        "asep_official_test_predictions_accessed",
    ):
        if boundary.get(key) is not False:
            raise ValueError(f"Ineligible evidence boundary: {key}")

    discipline = protocol.get("single_variable_discipline", {})
    for key in (
        "factor_changed_by_gauge_marginal",
        "factor_changed_by_gauge_hybrid",
        "factor_changed_by_gauge_hybrid_contrastive",
        "forbidden_comparison",
    ):
        if not discipline.get(key):
            raise ValueError(f"Missing single-variable contract: {key}")

    receipt = {
        "schema_version": "framecontact_specificity_gauge_exploratory_receipt_v1",
        "protocol_path": str(arguments.protocol),
        "protocol_file_sha256": hashlib.sha256(raw).hexdigest(),
        "protocol_canonical_sha256": hashlib.sha256(_canonical(protocol)).hexdigest(),
        "parent_sha256": {
            "residue_head": residue_sha,
            "antigen_fold": split_sha,
        },
        "three_condition_matrix_validated": True,
        "five_seed_rule_validated": True,
        "single_variable_discipline_validated": True,
        "development_only_boundary_validated": True,
        "test_prediction_boundary_validated": True,
    }
    arguments.receipt.parent.mkdir(parents=True, exist_ok=True)
    arguments.receipt.write_bytes(_canonical(receipt))
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
