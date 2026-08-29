#!/usr/bin/env python3
"""Seal the execution-complete development-only specificity gauge protocol."""

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
EXPECTED_SEEDS = [17, 42, 73, 101, 137]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--v2-protocol", type=Path, required=True)
    parser.add_argument("--v2-receipt", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.receipt.exists() or arguments.receipt.is_symlink():
        raise FileExistsError(f"Refusing to overwrite receipt: {arguments.receipt}")

    raw = arguments.protocol.read_bytes()
    protocol = json.loads(raw)
    if protocol.get("schema_version") != (
        "framecontact_specificity_gauge_exploratory_protocol_v3"
    ):
        raise ValueError("Unknown specificity-gauge v3 protocol schema")
    parent_sha = {
        "repaired_v2_protocol": _sha(arguments.v2_protocol),
        "repaired_v2_receipt": _sha(arguments.v2_receipt),
    }
    for key, digest in parent_sha.items():
        if protocol.get("parents", {}).get(key, {}).get("sha256") != digest:
            raise ValueError(f"Parent digest differs: {key}")

    if protocol.get("development_matrix") != EXPECTED_MATRIX:
        raise ValueError("Specificity-gauge v3 matrix differs")
    if protocol.get("model_selection", {}).get("seeds") != EXPECTED_SEEDS:
        raise ValueError("The frozen five-seed panel differs")
    data = protocol.get("data", {})
    panel = data.get("reciprocal_partner_panel")
    if not isinstance(panel, dict):
        raise ValueError("Executable reciprocal partner-panel contract is missing")
    if panel.get("format_version") != "framecontact_reciprocal_partner_panel_v2":
        raise ValueError("Reciprocal partner-panel format differs")
    if panel.get("size_per_query") != 31:
        raise ValueError("Reciprocal partner-panel size differs")
    if data.get("test_role_forbidden") is not True:
        raise ValueError("Test role must remain forbidden")
    implementation = protocol.get("implementation", {})
    for key in (
        "d_model",
        "num_heads",
        "num_self_layers",
        "learning_rate",
        "batch_size",
        "precision",
    ):
        if key not in implementation:
            raise ValueError(f"Execution contract lacks {key}")
    objective = protocol.get("training_objective", {})
    by_condition = objective.get("loss_weights_by_condition", {})
    if sorted(by_condition) != sorted(EXPECTED_MATRIX):
        raise ValueError("Condition-specific loss contracts differ")
    if objective.get("specificity_margin") != 0.05:
        raise ValueError("Specificity margin differs")
    boundary = protocol.get("evidence_boundary_at_freeze", {})
    for key in (
        "v2_valid_training_epoch_completed",
        "v2_validation_predictions_accessed",
        "v3_training_started",
        "v3_validation_predictions_accessed",
        "sabdab2_antigen_fold_test_predictions_accessed",
        "sabdab2_antigen_fold_test_metrics_accessed",
        "asep_official_test_predictions_accessed",
    ):
        if boundary.get(key) is not False:
            raise ValueError(f"Ineligible evidence boundary: {key}")
    if protocol.get("v2_execution_failure", {}).get("slurm_job") != "1349889":
        raise ValueError("V2 execution failure is not authenticated")

    receipt = {
        "schema_version": "framecontact_specificity_gauge_exploratory_receipt_v3",
        "protocol_path": str(arguments.protocol),
        "protocol_file_sha256": hashlib.sha256(raw).hexdigest(),
        "protocol_canonical_sha256": hashlib.sha256(_canonical(protocol)).hexdigest(),
        "parent_sha256": parent_sha,
        "v2_pretraining_failure_boundary_validated": True,
        "reciprocal_panel_contract_validated": True,
        "implementation_contract_validated": True,
        "condition_loss_contracts_validated": True,
        "five_seed_rule_validated": True,
        "development_only_boundary_validated": True,
        "test_prediction_boundary_validated": True,
    }
    arguments.receipt.parent.mkdir(parents=True, exist_ok=True)
    arguments.receipt.write_bytes(_canonical(receipt))
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
