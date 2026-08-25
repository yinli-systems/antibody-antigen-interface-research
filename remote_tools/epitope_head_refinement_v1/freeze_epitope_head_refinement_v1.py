#!/usr/bin/env python3
"""Freeze a one-seed, development-only epitope-head continuation protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


INITIAL_CHECKPOINT_SHA256 = (
    "5c2448c28a335dac0dcfa03d97a63be7331d664c3181c83a38bb563ca61b9a22"
)
SCREEN_SPLIT_SHA256 = (
    "4ff0965a0f8dde3913a176ac0411b40c9d2db3d1cb2684c393d6b01886b806cd"
)
TRAIN_PANEL_SHA256 = (
    "2ba7b5a3d62b731e81583ff09acca30e8be56c0452166cbba822a811dcd20a24"
)
VALIDATION_PANEL_SHA256 = (
    "c7140727a062d1f42a6e27613aa2708075f3aab65815f3a18c6a0caaf1435c11"
)
CONTACT_MANIFEST_SHA256 = (
    "d91ed42dcc05ee17cc880f7fa6124a7699f11a3beea75aab13f0b9502c274a33"
)
STRUCTURE_MANIFEST_SHA256 = (
    "edf74fd704b8b3b4ec9f7c9779e6a548b51e6d18d182642b5323ebc9ea2033f0"
)
EMBEDDING_MANIFEST_SHA256 = (
    "9ecbc08815166fb1a2a4a72f86280ffb808a162faf9521838039fb67c05cd545"
)
EPIFORMER_DICE_TO_BCE = 0.1992184401407531
EPIFORMER_COUNT_TO_BCE = 0.06957874447576923
EPIFORMER_LEARNING_RATE = 0.00006464091462
EPIFORMER_WEIGHT_DECAY = 0.000098653968


def canonical_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()


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


def require_false(value: object, label: str) -> None:
    if value is not False:
        raise ValueError(f"{label} must be false")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-result", type=Path, required=True)
    parser.add_argument("--base-result-sha256", required=True)
    parser.add_argument("--base-receipt", type=Path, required=True)
    parser.add_argument("--base-receipt-sha256", required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--screen-split", type=Path, required=True)
    parser.add_argument("--train-panel", type=Path, required=True)
    parser.add_argument("--validation-panel", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    for path, digest, label in (
        (args.base_result, args.base_result_sha256, "base result"),
        (args.base_receipt, args.base_receipt_sha256, "base receipt"),
        (args.initial_checkpoint, INITIAL_CHECKPOINT_SHA256, "initial checkpoint"),
        (args.screen_split, SCREEN_SPLIT_SHA256, "screen split"),
        (args.train_panel, TRAIN_PANEL_SHA256, "training panel"),
        (args.validation_panel, VALIDATION_PANEL_SHA256, "validation panel"),
        (args.source_manifest, args.source_manifest_sha256, "source manifest"),
    ):
        authenticate(path, digest, label)

    base_result = json.loads(args.base_result.read_text(encoding="utf-8"))
    base_receipt = json.loads(args.base_receipt.read_text(encoding="utf-8"))
    base_authority = base_result.get("authority", {})
    base_validation = base_result.get("selected_checkpoint_validation", {})
    if (
        base_result.get("experiment") != "epiformer_style_same_protocol"
        or int(base_receipt.get("seed", -1)) != 17
        or int(base_receipt.get("selected_epoch", -1)) != 8
        or base_receipt.get("selected_checkpoint_sha256")
        != INITIAL_CHECKPOINT_SHA256
        or base_receipt.get("result_sha256") != args.base_result_sha256
        or base_authority.get("split_manifest_sha256") != SCREEN_SPLIT_SHA256
        or base_authority.get("training_partner_panel_sha256")
        != TRAIN_PANEL_SHA256
        or base_authority.get("validation_partner_panel_sha256")
        != VALIDATION_PANEL_SHA256
        or base_authority.get("contact_manifest_sha256")
        != CONTACT_MANIFEST_SHA256
        or base_authority.get("structure_manifest_sha256")
        != STRUCTURE_MANIFEST_SHA256
        or base_authority.get("embedding_manifest_sha256")
        != EMBEDDING_MANIFEST_SHA256
    ):
        raise ValueError("base development authority differs")
    for label, value in (
        ("base result test_loaded", base_authority.get("test_loaded")),
        ("base result external_test_loaded", base_authority.get("external_test_loaded")),
        ("base receipt test_loaded", base_receipt.get("test_loaded")),
        ("base receipt external_test_loaded", base_receipt.get("external_test_loaded")),
    ):
        require_false(value, label)
    baseline_metrics = {
        name: float(base_validation[name]["mean_per_complex_auprc"])
        for name in ("contact", "epitope", "paratope")
    }
    if baseline_metrics != {
        "contact": 0.024830761686624868,
        "epitope": 0.2532489185677912,
        "paratope": 0.6756408447799143,
    }:
        raise ValueError(f"base validation metrics differ: {baseline_metrics}")

    if args.output_root.exists() or args.output_root.is_symlink():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    authority = args.output_root / "authority"
    authority.mkdir(parents=True)
    frozen_source_manifest = authority / "SOURCE_MANIFEST_EPITOPE_HEAD_REFINEMENT_V1.json"
    frozen_source_manifest.write_bytes(args.source_manifest.read_bytes())
    authenticate(
        frozen_source_manifest,
        args.source_manifest_sha256,
        "frozen source manifest",
    )
    protocol = {
        "schema_version": "sabdab2_epitope_head_refinement_protocol_v1",
        "protocol_status": "frozen_before_training",
        "arm": "joint_checkpoint_epitope_head_only",
        "data": {
            "dataset_backend": "sabdab2",
            "input_feature_mode": "external_embedding_cache",
            "screen_split_sha256": SCREEN_SPLIT_SHA256,
            "candidate_grid_invariant": "complete_cartesian_product_exactly_once",
            "contact_manifest_sha256": CONTACT_MANIFEST_SHA256,
            "structure_manifest_sha256": STRUCTURE_MANIFEST_SHA256,
            "embedding_manifest_sha256": EMBEDDING_MANIFEST_SHA256,
            "development_screen": {
                "train_count": 512,
                "validation_count": 128,
                "exclude_train_validation_antigen_cluster_overlap": True,
                "test_features_or_labels_loaded": False,
            },
            "reciprocal_partner_panel": {
                "format_version": "framecontact_reciprocal_partner_panel_v2",
                "size_per_query": 7,
                "train_sha256": TRAIN_PANEL_SHA256,
                "validation_sha256": VALIDATION_PANEL_SHA256,
            },
        },
        "continuation": {
            "training_policy": "epitope_head_only",
            "initial_checkpoint_sha256": INITIAL_CHECKPOINT_SHA256,
            "initial_checkpoint_seed": 17,
            "initial_checkpoint_selected_epoch": 8,
            "trainable_parameter_names": [
                "epitope_head.weight",
                "epitope_head.bias",
            ],
            "frozen_parameter_policy": "all_state_except_epitope_head_immutable",
            "backbone_mode_during_training": "eval",
        },
        "model_selection": {
            "seeds": [17],
            "maximum_epochs": 5,
            "early_stopping_patience": 2,
            "primary_metric": "epitope_mean_per_complex_auprc",
            "checkpoint_rule": (
                "maximize validation epitope mean-per-complex AUPRC; break ties by "
                "contact mean-per-complex AUPRC and earlier epoch"
            ),
        },
        "tasks": {
            "winner": {
                "name": "epiformer_style_same_protocol",
                "experiment": "epiformer_style_same_protocol",
                "distance_loss_weight": 0.0,
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
            "learning_rate": EPIFORMER_LEARNING_RATE,
            "weight_decay": EPIFORMER_WEIGHT_DECAY,
            "batch_size": 1,
            "evaluation_batch_size": 1,
            "precision": "bf16",
            "calibration_max_pairs": 500000,
            "offload_saved_tensors": False,
            "contact_focal_gamma": 1.0,
            "epitope_dice_weight": EPIFORMER_DICE_TO_BCE,
            "epitope_count_weight": EPIFORMER_COUNT_TO_BCE,
        },
        "training_objective": {
            "loss_weights": {
                "contact": 0.0,
                "paratope": 0.0,
                "epitope": 1.0,
                "specificity": 0.0,
                "distance": 0.0,
                "contrastive": 0.0,
                "matched_local": 0.0,
            },
            "specificity_margin": 0.05,
            "score_fraction": 0.01,
            "matched_local_margin": 0.05,
        },
        "promotion_gate": {
            "minimum_epitope_mean_per_complex_auprc_delta": 0.005,
            "maximum_absolute_contact_metric_drift": 1e-12,
            "maximum_absolute_paratope_metric_drift": 1e-12,
            "frozen_state_must_be_byte_identical": True,
            "promotion_scope": "seeds_29_43_development_only",
        },
        "baseline": {
            "result_sha256": args.base_result_sha256,
            "receipt_sha256": args.base_receipt_sha256,
            "selected_checkpoint_sha256": INITIAL_CHECKPOINT_SHA256,
            "selected_epoch": 8,
            "validation_mean_per_complex_auprc": baseline_metrics,
        },
        "research_basis": {
            "epiformer_loss_ratios": {
                "dice_to_bce": EPIFORMER_DICE_TO_BCE,
                "count_to_bce": EPIFORMER_COUNT_TO_BCE,
            },
            "epiformer_optimizer": {
                "learning_rate": EPIFORMER_LEARNING_RATE,
                "weight_decay": EPIFORMER_WEIGHT_DECAY,
            },
            "adaptation": (
                "Ratios and optimizer values only; the existing balanced per-complex "
                "epitope BCE and frozen same-protocol inputs are retained."
            ),
        },
        "authority": {
            "source_manifest_sha256": args.source_manifest_sha256,
            "base_result_sha256": args.base_result_sha256,
            "base_receipt_sha256": args.base_receipt_sha256,
        },
        "claim_boundary": {
            "development_only": True,
            "test_loaded": False,
            "temporal_holdout_loaded": False,
            "full_development_authorized": False,
            "only_seed_17_authorized": True,
        },
    }
    protocol_path = authority / "PROTOCOL_EPITOPE_HEAD_REFINEMENT_V1.json"
    protocol_path.write_bytes(canonical_bytes(protocol))
    receipt = {
        "format_version": "sabdab2_epitope_head_refinement_protocol_receipt_v1",
        "protocol_file_sha256": sha256_file(protocol_path),
        "source_manifest_sha256": args.source_manifest_sha256,
        "initial_checkpoint_sha256": INITIAL_CHECKPOINT_SHA256,
        "base_result_sha256": args.base_result_sha256,
        "base_receipt_sha256": args.base_receipt_sha256,
        "frozen_before_training": True,
        "test_loaded": False,
        "temporal_holdout_loaded": False,
    }
    receipt_path = authority / "PROTOCOL_EPITOPE_HEAD_REFINEMENT_V1_RECEIPT.json"
    receipt_path.write_bytes(canonical_bytes(receipt))
    campaign = {
        "format_version": "sabdab2_epitope_head_refinement_campaign_authority_v1",
        "seed": 17,
        "protocol_sha256": receipt["protocol_file_sha256"],
        "protocol_receipt_sha256": sha256_file(receipt_path),
        "source_manifest_sha256": args.source_manifest_sha256,
        "initial_checkpoint_sha256": INITIAL_CHECKPOINT_SHA256,
        "test_loaded": False,
        "temporal_holdout_loaded": False,
    }
    campaign_path = authority / "CAMPAIGN_AUTHORITY_V1.json"
    campaign_path.write_bytes(canonical_bytes(campaign))
    print(
        json.dumps(
            {
                "campaign_authority_sha256": sha256_file(campaign_path),
                "protocol_sha256": receipt["protocol_file_sha256"],
                "protocol_receipt_sha256": sha256_file(receipt_path),
                "baseline_metrics": baseline_metrics,
                "test_loaded": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
