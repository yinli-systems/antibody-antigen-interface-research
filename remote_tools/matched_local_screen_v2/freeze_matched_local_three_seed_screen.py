#!/usr/bin/env python3
"""Freeze the paired three-seed matched-local development screen."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


SEEDS = [17, 29, 43]


def canonical_bytes(payload: object) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def authenticate(path: Path, expected: str, label: str) -> None:
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"{label} SHA-256 differs: {observed}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed17-result", type=Path, required=True)
    parser.add_argument("--seed17-result-sha256", required=True)
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
    args = parser.parse_args()

    for path, digest, label in (
        (args.seed17_result, args.seed17_result_sha256, "seed-17 gate result"),
        (args.matched_local_manifest, args.matched_local_manifest_sha256, "matched-local manifest"),
        (args.matched_local_audit, args.matched_local_audit_sha256, "matched-local audit"),
        (args.screen_split, args.screen_split_sha256, "screen split"),
        (args.train_panel, args.train_panel_sha256, "training panel"),
        (args.validation_panel, args.validation_panel_sha256, "validation panel"),
    ):
        authenticate(path, digest, label)

    seed17 = json.loads(args.seed17_result.read_text())
    if (
        seed17.get("status") != "PASS_AUDIT"
        or seed17.get("promotion_status") != "PASS"
        or seed17.get("promotion_scope_if_pass") != "three_seed_frozen_development_screen_only"
        or seed17.get("authority", {}).get("test_loaded") is not False
    ):
        raise ValueError("seed-17 gate did not authorize this screen")

    audit = json.loads(args.matched_local_audit.read_text())
    if (
        audit.get("status") != "PASS"
        or audit.get("manifest_sha256") != args.matched_local_manifest_sha256
        or audit.get("heldout_after_caliper_freeze", {}).get("instance_count") != 508
        or float(audit.get("heldout_after_caliper_freeze", {}).get("strict_pair_coverage", -1)) < 0.95
        or audit.get("validation_labels_read") is not False
        or audit.get("test_labels_read") is not False
    ):
        raise ValueError("matched-local audit is not an eligible frozen PASS")

    for panel_path, role, count in (
        (args.train_panel, "train", 512),
        (args.validation_panel, "validation", 128),
    ):
        panel = json.loads(panel_path.read_text())
        if (
            panel.get("role") != role
            or panel.get("panel_size_per_direction") != 7
            or len(panel.get("entries", [])) != count
            or panel.get("authority", {}).get("task_label_columns_read") is not False
        ):
            raise ValueError(f"invalid frozen {role} panel")

    if args.output_root.exists() or args.output_root.is_symlink():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    authority_root = args.output_root / "authority"
    authority_root.mkdir(parents=True)

    base = {
        "schema_version": "sabdab2_matched_local_three_seed_screen_protocol_v1",
        "protocol_status": "frozen_before_training",
        "arm": None,
        "data": {
            "dataset_backend": "sabdab2",
            "input_feature_mode": "external_embedding_cache",
            "screen_split_sha256": args.screen_split_sha256,
            "train_count": 512,
            "validation_count": 128,
            "candidate_grid_invariant": "complete_cartesian_product_exactly_once",
            "contact_manifest_sha256": "d91ed42dcc05ee17cc880f7fa6124a7699f11a3beea75aab13f0b9502c274a33",
            "structure_manifest_sha256": "edf74fd704b8b3b4ec9f7c9779e6a548b51e6d18d182642b5323ebc9ea2033f0",
            "embedding_manifest_sha256": "9ecbc08815166fb1a2a4a72f86280ffb808a162faf9521838039fb67c05cd545",
            "reciprocal_partner_panel": {
                "format_version": "framecontact_reciprocal_partner_panel_v2",
                "size_per_query": 7,
                "train_sha256": args.train_panel_sha256,
                "validation_sha256": args.validation_panel_sha256,
            },
        },
        "model_selection": {
            "seeds": SEEDS,
            "maximum_epochs": 12,
            "early_stopping_patience": 3,
            "primary_metric": "contact_mean_per_complex_auprc",
            "checkpoint_rule": "maximize validation contact mean-per-complex AUPRC; break ties by minimum-direction cluster-weighted partner margin and earlier epoch",
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
            "experiment": "epiformer_style_same_protocol",
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
        },
        "training_objective": {
            "loss_weights": {
                "contact": 1.0,
                "paratope": 0.1,
                "epitope": 0.5,
                "specificity": 0.0,
                "distance": 0.1,
                "contrastive": 0.0,
                "matched_local": None,
            },
            "matched_local_margin": 0.05,
            "specificity_margin": 0.05,
            "score_fraction": 0.01,
            "contact_focal_gamma": 1.0,
            "epitope_dice_weight": 1.0,
            "epitope_count_weight": 0.2,
        },
        "promotion_gate": {
            "paired_median_contact_delta_strictly_positive": True,
            "minimum_contact_improving_seed_count": 2,
            "maximum_single_seed_contact_regression": 0.01,
            "maximum_median_epitope_regression": 0.01,
            "maximum_median_paratope_regression": 0.01,
            "maximum_median_partner_margin_regression": 0.01,
            "passing_scope": "full_2565_549_frozen_development_only",
        },
        "authority": {
            "seed17_gate_result_sha256": args.seed17_result_sha256,
            "matched_local_manifest_sha256": args.matched_local_manifest_sha256,
            "matched_local_audit_sha256": args.matched_local_audit_sha256,
            "test_loaded": False,
            "old_test_selection_forbidden": True,
            "temporal_holdout_authorized": False,
        },
    }

    outputs = {}
    for arm, weight in (("control", 0.0), ("matched_local", 0.25)):
        protocol = copy.deepcopy(base)
        protocol["arm"] = arm
        protocol["training_objective"]["loss_weights"]["matched_local"] = weight
        if weight:
            protocol["data"]["matched_local_decoy_manifest_sha256"] = args.matched_local_manifest_sha256
        path = authority_root / f"PROTOCOL_{arm.upper()}_V1.json"
        path.write_bytes(canonical_bytes(protocol))
        receipt = {
            "format_version": "sabdab2_matched_local_three_seed_protocol_receipt_v1",
            "arm": arm,
            "frozen_before_training": True,
            "protocol_file_sha256": sha256_file(path),
            "test_loaded": False,
        }
        receipt_path = authority_root / f"PROTOCOL_{arm.upper()}_V1_RECEIPT.json"
        receipt_path.write_bytes(canonical_bytes(receipt))
        outputs[arm] = {
            "protocol_sha256": receipt["protocol_file_sha256"],
            "receipt_sha256": sha256_file(receipt_path),
        }

    campaign = {
        "format_version": "sabdab2_matched_local_three_seed_campaign_authority_v1",
        "arms": outputs,
        "seeds": SEEDS,
        "array_mapping": [
            {"task": task, "arm": arm, "seed": seed}
            for task, (seed, arm) in enumerate(
                (seed, arm) for seed in SEEDS for arm in ("control", "matched_local")
            )
        ],
        "test_loaded": False,
        "temporal_holdout_authorized": False,
    }
    campaign_path = authority_root / "THREE_SEED_CAMPAIGN_AUTHORITY_V1.json"
    campaign_path.write_bytes(canonical_bytes(campaign))
    print(json.dumps({"campaign_authority_sha256": sha256_file(campaign_path), "arms": outputs}, sort_keys=True))


if __name__ == "__main__":
    main()
