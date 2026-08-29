#!/usr/bin/env python3
"""Seal the development-only binary-contact/distogram/triangle comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


PROTOCOL_SCHEMA = "framecontact_distogram_exploratory_protocol_v1"
RECEIPT_SCHEMA = "framecontact_distogram_exploratory_receipt_v1"
SEEDS = [17, 42, 73, 101, 137]
MATRIX = ["framecontact_distogram", "framecontact_distogram_triangle"]


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


def _authenticated_json(path: Path, expected_sha256: str, name: str) -> dict:
    if _sha(path) != expected_sha256:
        raise ValueError(f"{name} SHA-256 differs")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{name} is not a JSON object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--model-parent", type=Path, required=True)
    parser.add_argument("--split-parent", type=Path, required=True)
    parser.add_argument("--residue-head-parent", type=Path, required=True)
    parser.add_argument("--pair-geometry-manifest", type=Path, required=True)
    parser.add_argument("--pair-geometry-manifest-sha256", required=True)
    parser.add_argument("--development-selection", type=Path, required=True)
    parser.add_argument("--development-selection-sha256", required=True)
    parser.add_argument("--interrupted-access-audit", type=Path, required=True)
    parser.add_argument("--interrupted-access-audit-sha256", required=True)
    arguments = parser.parse_args()
    for output in (arguments.protocol, arguments.receipt):
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"Refusing to overwrite frozen output: {output}")

    geometry = _authenticated_json(
        arguments.pair_geometry_manifest,
        arguments.pair_geometry_manifest_sha256,
        "pair geometry manifest",
    )
    selection = _authenticated_json(
        arguments.development_selection,
        arguments.development_selection_sha256,
        "development selection",
    )
    interrupted = _authenticated_json(
        arguments.interrupted_access_audit,
        arguments.interrupted_access_audit_sha256,
        "interrupted access audit",
    )
    if geometry.get("counts", {}).get("records") != 3114:
        raise ValueError("pair geometry manifest must contain 3,114 development records")
    authority = geometry.get("authority", {})
    if (
        authority.get("selection_manifest_sha256")
        != arguments.development_selection_sha256
        or authority.get("selection_roles") != ["train", "validation"]
        or authority.get("selection_excluded_roles") != ["test"]
        or authority.get("frozen_test_pair_labels_read") is not False
    ):
        raise ValueError("pair geometry manifest lacks its development-only proof")
    if selection.get("counts") != {
        "selected": 3114,
        "train": 2565,
        "validation": 549,
        "excluded_test": 549,
    }:
        raise ValueError("development selection counts differ")
    if (
        interrupted.get("completed_artifact_role_counts", {}).get("test", 0) <= 0
        or interrupted.get("published_output_exists") is not False
        or interrupted.get("model_predictions_scores_or_metrics_accessed") is not False
    ):
        raise ValueError("interrupted access boundary is not represented truthfully")

    base_weights = {
        "contact": 1.0,
        "paratope": 0.25,
        "epitope": 0.25,
        "specificity": 0.25,
        "distogram": 0.5,
    }
    protocol = {
        "schema_version": PROTOCOL_SCHEMA,
        "protocol_status": (
            "development_only_frozen_after_prior_validation_and_test_label_"
            "preprocessing_access_but_before_this_training"
        ),
        "frozen_on": "2026-08-01",
        "parents": {
            "model_protocol": {
                "path": str(arguments.model_parent),
                "sha256": _sha(arguments.model_parent),
            },
            "antigen_fold_protocol": {
                "path": str(arguments.split_parent),
                "sha256": _sha(arguments.split_parent),
            },
            "residue_head_protocol": {
                "path": str(arguments.residue_head_parent),
                "sha256": _sha(arguments.residue_head_parent),
                "binary_reference_condition": "framecontact_contact_hybrid",
            },
        },
        "evidence_boundary_at_freeze": {
            "prior_validation_predictions_accessed": True,
            "current_sabdab2_test_pair_labels_read_by_interrupted_preprocessing": True,
            "current_sabdab2_test_model_predictions_accessed": False,
            "current_sabdab2_test_metrics_accessed": False,
            "this_distogram_training_started": False,
            "this_distogram_validation_predictions_accessed": False,
            "current_sabdab2_test_is_confirmatory": False,
            "independent_external_test_predictions_accessed": False,
        },
        "research_question": (
            "Does a single distogram head whose near-distance bins exactly project to "
            "contact probability outperform a binary contact head, and does a pose-free "
            "local triangle constraint improve precision without materially reducing recall?"
        ),
        "primary_hypothesis": (
            "Distogram supervision will improve validation contact AUPRC and calibration; "
            "CDR-by-monomer-surface-patch triangle consistency will preferentially reduce "
            "geometrically isolated false positives, increasing P@K and fixed-threshold "
            "precision while retaining at least 95% of the paired-seed recall of the "
            "unregularized distogram condition."
        ),
        "data": {
            "dataset_backend": "sabdab2",
            "split": "antigen-fold v3 train/validation roles only",
            "pair_geometry_manifest_sha256": arguments.pair_geometry_manifest_sha256,
            "pair_geometry_selection_sha256": arguments.development_selection_sha256,
            "pair_geometry_records": 3114,
            "test_role_forbidden": True,
            "reciprocal_partner_panel": {
                "format_version": "framecontact_reciprocal_partner_panel_v2",
                "size_per_query": 31,
                "train_sha256": (
                    "fd64b540ed0788d2e89d9c487d9cbde3cc56c4cb0225f29013007b2c3884efd6"
                ),
                "validation_sha256": (
                    "222a22e1c640b679df435e51e00f3550a61a6f880a31efb1807883b520aa30f5"
                ),
            },
        },
        "development_matrix": MATRIX,
        "binary_reference": {
            "condition": "framecontact_contact_hybrid",
            "source_protocol_sha256": _sha(arguments.residue_head_parent),
            "comparison_rule": "reuse all five pre-registered seed-matched development runs",
        },
        "controlled_architecture": {
            "backbone": (
                "the same two parameter-unshared full-relative-pose intra-chain encoders "
                "and late bidirectional cross-attention used by the binary hybrid reference"
            ),
            "pair_factorization": (
                "one distogram head; contact probability is the exact sum of registered "
                "near-distance-bin probabilities, never a separately parameterized mask head"
            ),
            "residue_head": "zero-gated intra-chain unary plus noisy-OR contact marginal",
            "triangle_region": (
                "expanded IMGT CDR residues by isolated-antigen surface residues inside a "
                "12-Angstrom patch seeded by intra-chain epitope-unary logits"
            ),
            "triangle_geometry": (
                "symmetric reverse triangle inequality using only known intra-chain "
                "C-alpha distances and predicted inter-chain expected distances"
            ),
        },
        "training_objective": {
            "loss_weights_by_condition": {
                "framecontact_distogram": {**base_weights, "triangle": 0.0},
                "framecontact_distogram_triangle": {
                    **base_weights,
                    "triangle": 0.1,
                },
            },
            "specificity_margin": 0.05,
            "score_fraction": 0.01,
            "specificity_score_source": "interaction_only",
            "wrong_pair_rule": (
                "wrong partners are ranking examples and never receive fabricated all-zero "
                "contact or distance labels"
            ),
        },
        "model_selection": {
            "seeds": SEEDS,
            "checkpoint_rule": (
                "maximize the minimum validation cluster-weighted directional interaction-"
                "only margin; break ties within 0.0001 by validation mean per-complex "
                "contact AUPRC, then earlier epoch"
            ),
            "maximum_epochs": 200,
            "early_stopping_patience": 20,
            "selection_unit": "seed and condition; current SAbDab2 test is unavailable",
            "test_role_forbidden": True,
        },
        "epoch_metrics": {
            "threshold_free": [
                "contact pooled and mean per-complex AUPRC",
                "epitope and paratope pooled and macro AUPRC",
                "contact P@10, P@Lmin/10, P@Lmin/5, P@Lmin and recalls",
            ],
            "fixed_operating_point": "raw logit zero for every epoch and condition",
            "fixed_operating_point_metrics": [
                "precision",
                "recall",
                "F1",
                "MCC",
                "predicted-positive ratio",
            ],
            "calibration": ["pooled Brier", "ECE-15", "validation temperature"],
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
            "calibration_max_pairs": 2000000,
            "offload_saved_tensors": True,
            "triangle_neighbor_radius_angstrom": 12.0,
            "triangle_neighbor_k": 8,
            "triangle_slack_angstrom": 0.5,
            "antigen_patch_top_seeds": 1,
            "antigen_patch_radius_angstrom": 12.0,
        },
        "promotion_rule": (
            "Report five paired seeds against the frozen binary hybrid reference. The "
            "triangle condition is promoted only if contact AUPRC/P@K and precision improve "
            "without recall ratio below 0.95 or a materially worse reciprocal margin."
        ),
        "promotion_thresholds": {
            "distogram_vs_binary_hybrid": {
                "mean_contact_mean_per_complex_auprc_delta_strictly_above": 0.0,
                "mean_contact_pooled_brier_delta_at_most": 0.0,
                "mean_selection_margin_delta_at_least": -0.01,
            },
            "triangle_vs_distogram": {
                "mean_contact_mean_per_complex_auprc_delta_strictly_above": 0.0,
                "mean_contact_p_at_lmin_over_5_delta_strictly_above": 0.0,
                "mean_fixed_raw_precision_delta_strictly_above": 0.0,
                "mean_fixed_raw_recall_retention_ratio_at_least": 0.95,
                "mean_selection_margin_delta_at_least": -0.01,
            },
        },
        "access_audit": {
            "interrupted_access_audit_sha256": (
                arguments.interrupted_access_audit_sha256
            ),
            "confirmatory_replacement_required": True,
        },
        "claim_boundary": (
            "Exploratory train/validation comparison only. The current SAbDab2 test cannot "
            "support a never-read or confirmatory claim; no external generalization, "
            "benchmark improvement, or state-of-the-art claim is authorized."
        ),
    }
    raw = _canonical(protocol)
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "protocol_path": str(arguments.protocol),
        "protocol_file_sha256": hashlib.sha256(raw).hexdigest(),
        "protocol_canonical_sha256": hashlib.sha256(_canonical(protocol)).hexdigest(),
        "parent_sha256": {
            "model": _sha(arguments.model_parent),
            "antigen_fold": _sha(arguments.split_parent),
            "residue_head": _sha(arguments.residue_head_parent),
            "pair_geometry": arguments.pair_geometry_manifest_sha256,
            "development_selection": arguments.development_selection_sha256,
            "interrupted_access_audit": arguments.interrupted_access_audit_sha256,
        },
        "two_condition_five_seed_matrix_validated": True,
        "condition_specific_loss_weights_validated": True,
        "development_only_boundary_validated": True,
        "current_sabdab2_test_confirmatory_status": False,
    }
    arguments.protocol.parent.mkdir(parents=True, exist_ok=True)
    arguments.receipt.parent.mkdir(parents=True, exist_ok=True)
    protocol_tmp = arguments.protocol.with_name(f".{arguments.protocol.name}.tmp")
    receipt_tmp = arguments.receipt.with_name(f".{arguments.receipt.name}.tmp")
    try:
        protocol_tmp.write_bytes(raw)
        receipt_tmp.write_bytes(_canonical(receipt))
        os.replace(protocol_tmp, arguments.protocol)
        os.replace(receipt_tmp, arguments.receipt)
    finally:
        for path in (protocol_tmp, receipt_tmp):
            if path.exists():
                path.unlink()
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
