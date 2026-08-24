#!/usr/bin/env python3
"""Freeze paired control/matched-local development-screen protocols."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


def canonical_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _authenticated(path: Path, expected: str, label: str) -> None:
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"{label} SHA-256 differs: {observed}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matched-local-manifest", type=Path, required=True)
    parser.add_argument("--matched-local-manifest-sha256", required=True)
    parser.add_argument("--matched-local-audit", type=Path, required=True)
    parser.add_argument("--matched-local-audit-sha256", required=True)
    parser.add_argument("--screen-split", type=Path, required=True)
    parser.add_argument("--screen-split-sha256", required=True)
    parser.add_argument("--train-panel", type=Path, required=True)
    parser.add_argument("--train-panel-sha256", required=True)
    parser.add_argument("--validation-panel", type=Path, required=True)
    parser.add_argument("--validation-panel-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args()
    for path, digest, label in (
        (
            arguments.matched_local_manifest,
            arguments.matched_local_manifest_sha256,
            "matched-local manifest",
        ),
        (
            arguments.matched_local_audit,
            arguments.matched_local_audit_sha256,
            "matched-local held-out audit",
        ),
        (arguments.screen_split, arguments.screen_split_sha256, "screen split"),
        (arguments.train_panel, arguments.train_panel_sha256, "training panel"),
        (
            arguments.validation_panel,
            arguments.validation_panel_sha256,
            "validation panel",
        ),
    ):
        _authenticated(path, digest, label)
    decoys = json.loads(arguments.matched_local_manifest.read_text(encoding="utf-8"))
    if (
        decoys.get("authority", {}).get("task_labels_read_for_roles") != ["train"]
        or decoys.get("authority", {}).get("validation_labels_read") is not False
        or decoys.get("authority", {}).get("test_labels_read") is not False
        or decoys.get("authority", {}).get("model_predictions_read") is not False
    ):
        raise ValueError("Matched-local authority is not train-only")
    audit = json.loads(arguments.matched_local_audit.read_text(encoding="utf-8"))
    if (
        audit.get("status") != "PASS"
        or audit.get("manifest_sha256") != arguments.matched_local_manifest_sha256
        or audit.get("heldout_after_caliper_freeze", {}).get("instance_count") != 508
        or float(
            audit.get("heldout_after_caliper_freeze", {}).get(
                "strict_pair_coverage", -1.0
            )
        )
        < 0.95
        or audit.get("validation_labels_read") is not False
        or audit.get("test_labels_read") is not False
        or audit.get("model_predictions_read") is not False
    ):
        raise ValueError("Matched-local held-out coverage audit is not a frozen PASS")
    train_panel = json.loads(arguments.train_panel.read_text(encoding="utf-8"))
    validation_panel = json.loads(
        arguments.validation_panel.read_text(encoding="utf-8")
    )
    for panel, role, count in (
        (train_panel, "train", 512),
        (validation_panel, "validation", 128),
    ):
        if (
            panel.get("role") != role
            or panel.get("panel_size_per_direction") != 31
            or len(panel.get("entries", [])) != count
            or panel.get("authority", {}).get("task_label_columns_read") is not False
        ):
            raise ValueError(f"Frozen {role} partner panel is invalid")
    if arguments.output_root.exists() or arguments.output_root.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {arguments.output_root}")
    arguments.output_root.mkdir(parents=True)
    base = {
        "schema_version": "sabdab2_matched_local_screen_protocol_v2",
        "protocol_status": "frozen_before_training",
        "data": {
            "dataset_backend": "sabdab2",
            "input_feature_mode": "external_embedding_cache",
            "screen_split_sha256": arguments.screen_split_sha256,
            "candidate_grid_invariant": "complete_cartesian_product_exactly_once",
            "contact_manifest_sha256": "d91ed42dcc05ee17cc880f7fa6124a7699f11a3beea75aab13f0b9502c274a33",
            "structure_manifest_sha256": "edf74fd704b8b3b4ec9f7c9779e6a548b51e6d18d182642b5323ebc9ea2033f0",
            "embedding_manifest_sha256": "9ecbc08815166fb1a2a4a72f86280ffb808a162faf9521838039fb67c05cd545",
            "development_screen": {
                "train_count": 512,
                "validation_count": 128,
                "exclude_train_validation_antigen_cluster_overlap": True,
                "test_features_or_labels_loaded": False,
            },
            "reciprocal_partner_panel": {
                "format_version": "framecontact_reciprocal_partner_panel_v2",
                "size_per_query": 31,
                "train_sha256": arguments.train_panel_sha256,
                "validation_sha256": arguments.validation_panel_sha256,
            },
        },
        "model_selection": {
            "seeds": [17],
            "maximum_epochs": 12,
            "early_stopping_patience": 3,
            "evaluate_partner_retrieval": True,
            "primary_metric": "contact_mean_per_complex_auprc",
            "checkpoint_rule": (
                "maximize validation contact mean-per-complex AUPRC; break ties by "
                "minimum-direction cluster-weighted partner margin and earlier epoch"
            ),
        },
        "tasks": {
            "winner": {
                "name": "epiformer_style_same_protocol",
                "experiment": "epiformer_style_same_protocol",
                "distance_loss_weight": 0.1,
                "contrastive_loss_weight": 0.0,
            }
        },
        "implementation": {
            "d_model": 128,
            "num_heads": 8,
            "num_self_layers": 2,
            "num_rbf": 16,
            "neighbor_k": 32,
            "neighbor_radius_angstrom": 16.0,
            "sequence_window": 32,
            "dropout": 0.1,
            "learning_rate": 0.0003,
            "weight_decay": 0.0001,
            "batch_size": 1,
            "evaluation_batch_size": 1,
            "precision": "bf16",
            "calibration_max_pairs": 500000,
            "offload_saved_tensors": True,
            "contact_focal_gamma": 1.0,
            "epitope_dice_weight": 1.0,
            "epitope_count_weight": 0.2,
            "contrastive_dim": 64,
            "experiment": "epiformer_style_same_protocol",
        },
        "training_objective": {
            "contact": "equal-complex observed-class-balanced focal BCE",
            "marginals": "balanced BCE plus epitope Dice and prevalence matching",
            "specificity": "partner-swap training disabled; reciprocal panel is evaluation-only",
            "specificity_score_source": "interaction_only",
            "matched_local_margin": 0.05,
            "loss_weights": {
                "contact": 1.0,
                "paratope": 0.1,
                "epitope": 0.5,
                "specificity": 0.0,
                "distance": 0.1,
                "contrastive": 0.0,
            },
            "specificity_margin": 0.05,
            "score_fraction": 0.01,
        },
        "screen_gate": {
            "reference_arm": "control",
            "candidate_arm": "matched_local",
            "promotion_requires": (
                "matched-local contact mean-per-complex AUPRC strictly exceeds the "
                "paired control at seed 17"
            ),
            "maximum_epitope_mean_per_complex_auprc_regression": 0.01,
            "maximum_paratope_mean_per_complex_auprc_regression": 0.01,
            "maximum_minimum_direction_cluster_weighted_margin_regression": 0.01,
            "passing_this_gate_authorizes_only_a_three_seed_development_screen": True,
        },
        "claim_boundary": {
            "development_only": True,
            "test_loaded": False,
            "old_sabdab2_test_may_not_be_used_for_selection": True,
            "future_test_requires_new_temporal_holdout": True,
            "screen_is_not_a_final_baseline": True,
            "official_baseline_replication": False,
        },
    }
    outputs = {}
    for arm, weight in (("control", 0.0), ("matched_local", 0.25)):
        protocol = copy.deepcopy(base)
        protocol["arm"] = arm
        protocol["training_objective"]["loss_weights"]["matched_local"] = weight
        if weight:
            protocol["data"][
                "matched_local_decoy_manifest_sha256"
            ] = arguments.matched_local_manifest_sha256
            protocol["data"][
                "matched_local_heldout_coverage_audit_sha256"
            ] = arguments.matched_local_audit_sha256
            protocol["training_objective"]["matched_local"] = (
                "same-antigen strict matched non-epitope motif ranking; equal decoy "
                "mass within complex and equal complex mass"
            )
        else:
            protocol["training_objective"]["matched_local"] = "disabled"
        path = arguments.output_root / f"PROTOCOL_{arm.upper()}_V2.json"
        path.write_bytes(canonical_json_bytes(protocol))
        receipt = {
            "format_version": "sabdab2_matched_local_screen_protocol_receipt_v2",
            "arm": arm,
            "frozen_before_training": True,
            "protocol_file_sha256": sha256_file(path),
            "test_loaded": False,
        }
        receipt_path = arguments.output_root / f"PROTOCOL_{arm.upper()}_V2_RECEIPT.json"
        receipt_path.write_bytes(canonical_json_bytes(receipt))
        outputs[arm] = {
            "protocol_sha256": receipt["protocol_file_sha256"],
            "receipt_sha256": sha256_file(receipt_path),
        }
    print(json.dumps(outputs, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
