#!/usr/bin/env python3
"""Freeze a two-arm, development-only joint-loss balance screen."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any


BASELINE_RESULT_SHA256 = "38bbf1a4ade66b01abd81a925ac89464725c67564bc2c4c352701d6d8a0f8184"
BASELINE_RECEIPT_SHA256 = "f63cf0e8f7cf24f1224bbd0b1890cc259fae0de96e66e47b30d23efa61494623"
PRIOR_WINNER_RESULT_SHA256 = "491c27569027c2761e79a7a91059d8df791f441b7f50357bb8180db2e819360d"
PRIOR_WINNER_RECEIPT_SHA256 = "3305b07ab351362191722f0f51e0d14781cfffc603b9128d13c636647942862d"
ARMS = {
    "paratope025": 0.25,
    "paratope050": 0.50,
}


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


def finite(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("non-finite value")
    return result


def read_authenticated_run(
    root: Path, result_sha256: str, receipt_sha256: str, label: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    result_path = root / "FRAMECONTACT_DEVELOPMENT_RESULT.json"
    receipt_path = root / "RUN_RECEIPT.json"
    authenticate(result_path, result_sha256, f"{label} result")
    authenticate(receipt_path, receipt_sha256, f"{label} receipt")
    result = json.loads(result_path.read_text())
    receipt = json.loads(receipt_path.read_text())
    if (
        receipt.get("result_sha256") != result_sha256
        or receipt.get("test_loaded") is not False
        or result.get("authority", {}).get("test_loaded") is not False
        or result.get("authority", {}).get("external_test_loaded") is not False
    ):
        raise ValueError(f"{label} authority differs")
    return result, receipt


def selected_metrics(result: dict[str, Any]) -> dict[str, Any]:
    selected = result["selected_checkpoint_validation"]
    return {
        "selected_epoch": int(result["selection"]["best_epoch"]),
        "epitope_mean_per_complex_auprc": finite(
            selected["epitope"]["mean_per_complex_auprc"]
        ),
        "contact_mean_per_complex_auprc": finite(
            selected["contact"]["mean_per_complex_auprc"]
        ),
        "paratope_mean_per_complex_auprc": finite(
            selected["paratope"]["mean_per_complex_auprc"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-analysis", type=Path, required=True)
    parser.add_argument("--prior-analysis-sha256", required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--prior-winner-root", type=Path, required=True)
    parser.add_argument("--screen-split", type=Path, required=True)
    parser.add_argument("--screen-split-sha256", required=True)
    parser.add_argument("--train-panel", type=Path, required=True)
    parser.add_argument("--train-panel-sha256", required=True)
    parser.add_argument("--validation-panel", type=Path, required=True)
    parser.add_argument("--validation-panel-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    for path, digest, label in (
        (args.prior_analysis, args.prior_analysis_sha256, "prior analysis"),
        (args.screen_split, args.screen_split_sha256, "screen split"),
        (args.train_panel, args.train_panel_sha256, "training panel"),
        (args.validation_panel, args.validation_panel_sha256, "validation panel"),
    ):
        authenticate(path, digest, label)

    prior_analysis = json.loads(args.prior_analysis.read_text())
    if (
        prior_analysis.get("status") != "PASS_AUDIT"
        or prior_analysis.get("winner") != "dropout020_count020"
        or prior_analysis.get("next_gate", {}).get(
            "winner_seeds_29_and_43_authorized"
        )
        is not False
        or prior_analysis.get("authority", {}).get("test_loaded") is not False
        or prior_analysis.get("authority", {}).get("temporal_holdout_loaded")
        is not False
    ):
        raise ValueError("prior analysis does not authorize this screen")

    baseline_result, baseline_receipt = read_authenticated_run(
        args.baseline_root / "joint_current" / "seed-17",
        BASELINE_RESULT_SHA256,
        BASELINE_RECEIPT_SHA256,
        "baseline",
    )
    prior_result, prior_receipt = read_authenticated_run(
        args.prior_winner_root,
        PRIOR_WINNER_RESULT_SHA256,
        PRIOR_WINNER_RECEIPT_SHA256,
        "prior winner",
    )
    for result, dropout, count_weight, label in (
        (baseline_result, 0.1, 0.2, "baseline"),
        (prior_result, 0.2, 0.2, "prior winner"),
    ):
        config = result["configuration"]
        if (
            result.get("experiment") != "epiformer_style_same_protocol"
            or config.get("seed") != 17
            or not math.isclose(finite(config["contact_loss_weight"]), 1.0)
            or not math.isclose(finite(config["paratope_loss_weight"]), 0.1)
            or not math.isclose(finite(config["epitope_loss_weight"]), 0.5)
            or not math.isclose(finite(config["dropout"]), dropout)
            or not math.isclose(finite(config["epitope_dice_weight"]), 1.0)
            or not math.isclose(
                finite(config["epitope_count_weight"]), count_weight
            )
            or config.get("early_stopping_patience") != 3
        ):
            raise ValueError(f"{label} configuration differs")

    prior_selected_epoch = int(prior_result["selection"]["best_epoch"])
    prior_train = next(
        row["train"]
        for row in prior_result["history"]
        if int(row["epoch"]) == prior_selected_epoch
    )
    loss_diagnostic = {
        "selected_epoch": prior_selected_epoch,
        "unweighted": {
            "contact": finite(prior_train["contact_loss"]),
            "paratope": finite(prior_train["paratope_loss"]),
            "epitope": finite(prior_train["epitope_loss"]),
        },
        "weighted": {
            "contact": finite(prior_train["contact_loss"]),
            "paratope": 0.1 * finite(prior_train["paratope_loss"]),
            "epitope": 0.5 * finite(prior_train["epitope_loss"]),
        },
    }

    if args.output_root.exists() or args.output_root.is_symlink():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    authority_root = args.output_root / "authority"
    authority_root.mkdir(parents=True)

    baseline = selected_metrics(baseline_result)
    baseline.update(
        {
            "result_sha256": BASELINE_RESULT_SHA256,
            "receipt_sha256": BASELINE_RECEIPT_SHA256,
        }
    )
    prior_winner = selected_metrics(prior_result)
    prior_winner.update(
        {
            "result_sha256": PRIOR_WINNER_RESULT_SHA256,
            "receipt_sha256": PRIOR_WINNER_RECEIPT_SHA256,
        }
    )
    base = {
        "format_version": "sabdab2_joint_loss_balance_screen_protocol_v1",
        "protocol_status": "frozen_before_training",
        "arm": None,
        "seed": 17,
        "data": {
            "dataset_backend": "sabdab2",
            "input_feature_mode": "external_embedding_cache",
            "train_count": 512,
            "validation_count": 128,
            "screen_split_sha256": args.screen_split_sha256,
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
            "maximum_epochs": 12,
            "early_stopping_patience": 3,
            "primary_metric": "epitope_mean_per_complex_auprc",
            "tie_break_metric": "contact_mean_per_complex_auprc",
            "checkpoint_rule": "maximize validation epitope mean-per-complex AUPRC; break ties by contact mean-per-complex AUPRC and earlier epoch",
            "seeds": [17],
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
            "dropout": 0.2,
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
                "paratope": None,
                "epitope": 0.5,
                "specificity": 0.0,
                "distance": 0.1,
                "contrastive": 0.0,
                "matched_local": 0.0,
            },
            "specificity_margin": 0.05,
            "score_fraction": 0.01,
            "matched_local_margin": 0.05,
        },
        "joint_objective": {
            "contact_loss_weight": 1.0,
            "paratope_loss_weight": None,
            "epitope_loss_weight": 0.5,
            "distance_loss_weight": 0.1,
            "epitope_dice_weight": 1.0,
            "epitope_count_weight": 0.2,
        },
        "screen_design": {
            "single_changed_factor": "paratope_loss_weight",
            "control_reused_from_frozen_joint_current": baseline,
            "prior_epitope_candidate": prior_winner,
            "prior_selected_epoch_loss_diagnostic": loss_diagnostic,
            "promotion_gate": {
                "minimum_epitope_delta_vs_control": 0.0075,
                "maximum_epitope_regression_vs_prior_candidate": 0.005,
                "maximum_contact_regression_vs_control": 0.005,
                "maximum_paratope_regression_vs_control": 0.01,
                "next_step_if_pass": "run_only_winner_on_seeds_29_and_43",
            },
        },
        "claim_boundary": {
            "development_only": True,
            "test_loaded": False,
            "temporal_holdout_loaded": False,
            "single_seed_screen_cannot_authorize_full_development_or_test": True,
        },
        "authority": {
            "prior_analysis_sha256": args.prior_analysis_sha256,
            "baseline_result_sha256": BASELINE_RESULT_SHA256,
            "prior_winner_result_sha256": PRIOR_WINNER_RESULT_SHA256,
        },
    }

    arm_outputs: dict[str, Any] = {}
    for arm, paratope_weight in ARMS.items():
        protocol = copy.deepcopy(base)
        protocol["arm"] = arm
        protocol["training_objective"]["loss_weights"]["paratope"] = paratope_weight
        protocol["joint_objective"]["paratope_loss_weight"] = paratope_weight
        protocol_path = authority_root / f"PROTOCOL_{arm.upper()}_V1.json"
        protocol_path.write_bytes(canonical_bytes(protocol))
        receipt = {
            "format_version": "sabdab2_joint_loss_balance_protocol_receipt_v1",
            "arm": arm,
            "seed": 17,
            "frozen_before_training": True,
            "protocol_file_sha256": sha256_file(protocol_path),
            "test_loaded": False,
            "temporal_holdout_loaded": False,
        }
        receipt_path = authority_root / f"PROTOCOL_{arm.upper()}_V1_RECEIPT.json"
        receipt_path.write_bytes(canonical_bytes(receipt))
        arm_outputs[arm] = {
            "paratope_loss_weight": paratope_weight,
            "protocol_sha256": receipt["protocol_file_sha256"],
            "receipt_sha256": sha256_file(receipt_path),
        }

    campaign = {
        "format_version": "sabdab2_joint_loss_balance_campaign_authority_v1",
        "array_mapping": [
            {"task": task, "arm": arm, "seed": 17}
            for task, arm in enumerate(ARMS)
        ],
        "arms": arm_outputs,
        "baseline": baseline,
        "prior_epitope_candidate": prior_winner,
        "prior_analysis_sha256": args.prior_analysis_sha256,
        "test_loaded": False,
        "temporal_holdout_loaded": False,
    }
    campaign_path = authority_root / "CAMPAIGN_AUTHORITY_V1.json"
    campaign_path.write_bytes(canonical_bytes(campaign))
    print(
        json.dumps(
            {
                "campaign_authority_sha256": sha256_file(campaign_path),
                "arms": arm_outputs,
                "weighted_loss_diagnostic": loss_diagnostic["weighted"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
