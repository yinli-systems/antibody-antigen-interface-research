#!/usr/bin/env python3
"""Freeze every model and scorer decision before external test labels are mounted."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from interaction_first.partner_panel import PARTNER_PANEL_FORMAT  # noqa: E402
from interaction_first.reciprocal_partner_panel import (  # noqa: E402
    RECIPROCAL_PARTNER_PANEL_FORMAT,
)
from interaction_first.sabdab2_structure import (  # noqa: E402
    authenticate_file,
    canonical_json_bytes,
    sha256_file,
)
from interaction_first.structural_novelty import STRUCTURAL_AUDIT_FORMAT  # noqa: E402


FORMAT_V1 = "framecontact_external_opening_receipt_v1"
FORMAT_V2 = "framecontact_external_opening_receipt_v2"
SEEDS = (17, 42, 73, 101, 137)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-summary", type=Path, required=True)
    parser.add_argument("--development-summary-sha256", required=True)
    parser.add_argument("--structural-audit", type=Path, required=True)
    parser.add_argument("--structural-audit-sha256", required=True)
    parser.add_argument("--test-partner-panel", type=Path, required=True)
    parser.add_argument("--test-partner-panel-sha256", required=True)
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument("--thresholds-sha256", required=True)
    parser.add_argument("--checkpoints", type=Path, nargs=5, required=True)
    parser.add_argument("--checkpoint-receipts", type=Path, nargs=5, required=True)
    parser.add_argument("--test-truth-authority-sha256", required=True)
    parser.add_argument("--scorer-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.output.exists() or arguments.output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite opening receipt: {arguments.output}")
    authenticate_file(
        arguments.development_summary,
        arguments.development_summary_sha256,
        "development summary",
    )
    development = json.loads(arguments.development_summary.read_text(encoding="utf-8"))
    if (
        development.get("format_version")
        != "framecontact_development_multiseed_summary_v1"
        or development.get("matrix_complete") is not True
        or development.get("external_opening_ready") is not False
    ):
        raise ValueError("Development matrix is not a verified pre-opening authority")
    authenticate_file(
        arguments.structural_audit,
        arguments.structural_audit_sha256,
        "structural audit",
    )
    structural = json.loads(arguments.structural_audit.read_text(encoding="utf-8"))
    if (
        structural.get("format_version") != STRUCTURAL_AUDIT_FORMAT
        or structural.get("valid") is not True
        or any(value != 0 for value in structural.get("violations", {}).values())
        or structural.get("authority", {}).get("test_labels_read") is not False
    ):
        raise ValueError("Structural novelty audit is not clean")
    authenticate_file(
        arguments.test_partner_panel,
        arguments.test_partner_panel_sha256,
        "test partner panel",
    )
    panel = json.loads(arguments.test_partner_panel.read_text(encoding="utf-8"))
    panel_format = panel.get("format_version")
    if panel_format == PARTNER_PANEL_FORMAT:
        panel_size = panel.get("panel_size")
        opening_format = FORMAT_V1
    elif panel_format == RECIPROCAL_PARTNER_PANEL_FORMAT:
        panel_size = panel.get("panel_size_per_direction")
        opening_format = FORMAT_V2
    else:
        panel_size = None
        opening_format = None
    if (
        opening_format is None
        or panel.get("role") != "test"
        or panel_size != 31
        or panel.get("authority", {}).get("task_label_columns_read") is not False
    ):
        raise ValueError("Test partner panel is ineligible")
    if opening_format == FORMAT_V2 and development.get(
        "specificity_score_sources"
    ) != ["interaction_only"]:
        raise ValueError(
            "Reciprocal opening requires interaction-only development scoring"
        )
    authenticate_file(arguments.thresholds, arguments.thresholds_sha256, "thresholds")
    thresholds = json.loads(arguments.thresholds.read_text(encoding="utf-8"))
    if (
        thresholds.get("format_version") != "framecontact_validation_thresholds_v1"
        or thresholds.get("selected_on_validation_only") is not True
    ):
        raise ValueError("Thresholds were not frozen from validation only")
    checkpoint_records = []
    for expected_seed, checkpoint_path, receipt_path in zip(
        SEEDS, arguments.checkpoints, arguments.checkpoint_receipts
    ):
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        checkpoint_sha = sha256_file(checkpoint_path)
        if (
            receipt.get("seed") != expected_seed
            or receipt.get("selected_checkpoint_sha256") != checkpoint_sha
            or receipt.get("test_loaded") is not False
            or receipt.get("external_test_loaded") is not False
        ):
            raise ValueError(f"Checkpoint receipt failed for seed {expected_seed}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if (
            checkpoint.get("format_version") != "framecontact_development_training_v1"
            or checkpoint.get("experiment") != "framecontact"
            or checkpoint.get("seed") != expected_seed
        ):
            raise ValueError(f"Checkpoint payload failed for seed {expected_seed}")
        if opening_format == FORMAT_V2 and checkpoint.get(
            "specificity_score_source"
        ) != "interaction_only":
            raise ValueError(
                f"Checkpoint {expected_seed} did not use interaction-only specificity"
            )
        checkpoint_records.append(
            {
                "seed": expected_seed,
                "checkpoint_sha256": checkpoint_sha,
                "run_receipt_sha256": sha256_file(receipt_path),
                "validation_temperature": float(checkpoint["validation_temperature"]),
            }
        )
    if arguments.scorer_source.resolve() != (
        REPOSITORY_ROOT / "scripts" / "score_framecontact_external.py"
    ).resolve():
        raise ValueError("Unexpected external scorer source path")
    scorer_sha = sha256_file(arguments.scorer_source)
    payload = {
        "format_version": opening_format,
        "eligible_to_open": True,
        "test_open_count_before": 0,
        "test_labels_read_at_freeze": False,
        "development_summary_sha256": arguments.development_summary_sha256,
        "structural_audit_sha256": arguments.structural_audit_sha256,
        "test_partner_panel_sha256": arguments.test_partner_panel_sha256,
        "test_partner_panel_size": 31,
        "test_partner_panel_directions": 2 if opening_format == FORMAT_V2 else 1,
        "thresholds_sha256": arguments.thresholds_sha256,
        "test_truth_authority_sha256": arguments.test_truth_authority_sha256,
        "scorer_source_sha256": scorer_sha,
        "ensemble_rule": (
            "mean calibrated probabilities for contact and residue tasks; mean raw "
            "interaction logits then sigmoid for both partner-retrieval directions"
            if opening_format == FORMAT_V2
            else "arithmetic_mean_of_five_validation_temperature_scaled_probabilities"
        ),
        "checkpoints": checkpoint_records,
        "primary_endpoint": (
            "fixed_antigen_cognate_antibody_interaction_margin"
            if opening_format == FORMAT_V2
            else "cluster_weighted_cognate_compatibility_margin"
        ),
        "specificity_score_source": (
            "interaction_only" if opening_format == FORMAT_V2 else "total_contact"
        ),
        "bootstrap": {"unit": "antigen_cluster", "replicates": 10000, "seed": 20260730},
        "claim_boundary": (
            "Pre-opening decision receipt only. No test features, labels, predictions, or "
            "performance values were read to create it."
        ),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_bytes(canonical_json_bytes(payload))
    print(json.dumps({"output": str(arguments.output), "sha256": sha256_file(arguments.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
