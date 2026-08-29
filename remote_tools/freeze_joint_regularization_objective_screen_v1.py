#!/usr/bin/env python3
"""Freeze a minimal development-only regularization/objective screen."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any


SEEDS = (17, 29, 43)
BASELINE_RESULT_SHA256 = {
    17: "38bbf1a4ade66b01abd81a925ac89464725c67564bc2c4c352701d6d8a0f8184",
    29: "5ec2248f51d9690a3d8ce161ea767f3232fa302c3f7dfbb430c4ba97fafb2c86",
    43: "17fdd96cfc1b5c3f15e9736dd356f186fb7ba32574565915fcdcaed3106affc0",
}
BASELINE_RECEIPT_SHA256 = {
    17: "f63cf0e8f7cf24f1224bbd0b1890cc259fae0de96e66e47b30d23efa61494623",
    29: "fdbdad3813093b30952a0f5624f08fd814ecd7a345c657367291b42e012911f2",
    43: "7e8012ca213296e80d6eb0756876ceb02bd7eaf3e167507ee145488c6a9dad9a",
}
ARMS = {
    "dropout020_count020": {"dropout": 0.2, "epitope_count_weight": 0.2},
    "dropout010_count000": {"dropout": 0.1, "epitope_count_weight": 0.0},
    "dropout020_count000": {"dropout": 0.2, "epitope_count_weight": 0.0},
}


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


def finite(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("non-finite metric")
    return result


def replay_patience(history: list[dict[str, Any]], patience: int) -> dict[str, Any]:
    best_value = -math.inf
    best_epoch = 0
    stale = 0
    stop_epoch = int(history[-1]["epoch"])
    for row in history:
        epoch = int(row["epoch"])
        value = finite(row["validation"]["epitope"]["mean_per_complex_auprc"])
        if value > best_value:
            best_value = value
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                stop_epoch = epoch
                break
    return {"patience": patience, "stop_epoch": stop_epoch, "best_epoch": best_epoch, "best_epitope_auprc": best_value}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-result", type=Path, required=True)
    parser.add_argument("--diagnostic-result-sha256", required=True)
    parser.add_argument("--curve-audit", type=Path, required=True)
    parser.add_argument("--curve-audit-sha256", required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--screen-split", type=Path, required=True)
    parser.add_argument("--screen-split-sha256", required=True)
    parser.add_argument("--train-panel", type=Path, required=True)
    parser.add_argument("--train-panel-sha256", required=True)
    parser.add_argument("--validation-panel", type=Path, required=True)
    parser.add_argument("--validation-panel-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    for path, digest, label in (
        (args.diagnostic_result, args.diagnostic_result_sha256, "joint diagnostic"),
        (args.curve_audit, args.curve_audit_sha256, "curve audit"),
        (args.screen_split, args.screen_split_sha256, "screen split"),
        (args.train_panel, args.train_panel_sha256, "training panel"),
        (args.validation_panel, args.validation_panel_sha256, "validation panel"),
    ):
        authenticate(path, digest, label)

    diagnostic = json.loads(args.diagnostic_result.read_text())
    curve_audit = json.loads(args.curve_audit.read_text())
    if (
        diagnostic.get("status") != "PASS_AUDIT"
        or diagnostic.get("best_median_epitope_arm") != "joint_current"
        or diagnostic.get("diagnosis", {}).get("late_overfitting_in_all_arms") is not True
        or diagnostic.get("diagnosis", {}).get("contact_task_competition_supported") is not False
        or diagnostic.get("diagnosis", {}).get("paratope_task_competition_supported") is not False
        or diagnostic.get("authority", {}).get("test_loaded") is not False
        or diagnostic.get("authority", {}).get("temporal_holdout_loaded") is not False
    ):
        raise ValueError("joint diagnostic does not authorize this development screen")
    if (
        curve_audit.get("status") != "PASS_AUDIT"
        or curve_audit.get("diagnostic_gate", {}).get("early_peak_present") is not True
        or curve_audit.get("diagnostic_gate", {}).get("late_overfitting_signature_present") is not True
        or curve_audit.get("authority", {}).get("test_loaded") is not False
    ):
        raise ValueError("curve audit does not authorize this development screen")

    baseline: dict[str, Any] = {}
    replay: dict[str, Any] = {}
    for seed in SEEDS:
        result_path = args.baseline_root / "joint_current" / f"seed-{seed}" / "FRAMECONTACT_DEVELOPMENT_RESULT.json"
        receipt_path = args.baseline_root / "joint_current" / f"seed-{seed}" / "RUN_RECEIPT.json"
        authenticate(result_path, BASELINE_RESULT_SHA256[seed], f"baseline result seed {seed}")
        authenticate(receipt_path, BASELINE_RECEIPT_SHA256[seed], f"baseline receipt seed {seed}")
        result = json.loads(result_path.read_text())
        receipt = json.loads(receipt_path.read_text())
        config = result["configuration"]
        if (
            result.get("experiment") != "epiformer_style_same_protocol"
            or config.get("seed") != seed
            or not math.isclose(finite(config["contact_loss_weight"]), 1.0)
            or not math.isclose(finite(config["paratope_loss_weight"]), 0.1)
            or not math.isclose(finite(config["epitope_loss_weight"]), 0.5)
            or not math.isclose(finite(config["dropout"]), 0.1)
            or not math.isclose(finite(config["epitope_dice_weight"]), 1.0)
            or not math.isclose(finite(config["epitope_count_weight"]), 0.2)
            or config.get("early_stopping_patience") != 3
            or result.get("authority", {}).get("test_loaded") is not False
            or result.get("authority", {}).get("external_test_loaded") is not False
            or receipt.get("result_sha256") != BASELINE_RESULT_SHA256[seed]
            or receipt.get("test_loaded") is not False
        ):
            raise ValueError(f"baseline authority differs for seed {seed}")
        selected = result["selected_checkpoint_validation"]
        baseline[str(seed)] = {
            "result_sha256": BASELINE_RESULT_SHA256[seed],
            "receipt_sha256": BASELINE_RECEIPT_SHA256[seed],
            "selected_epoch": int(result["selection"]["best_epoch"]),
            "epitope_mean_per_complex_auprc": finite(selected["epitope"]["mean_per_complex_auprc"]),
            "contact_mean_per_complex_auprc": finite(selected["contact"]["mean_per_complex_auprc"]),
            "paratope_mean_per_complex_auprc": finite(selected["paratope"]["mean_per_complex_auprc"]),
        }
        replay[str(seed)] = {str(p): replay_patience(result["history"], p) for p in (1, 2, 3)}

    if args.output_root.exists() or args.output_root.is_symlink():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    authority_root = args.output_root / "authority"
    authority_root.mkdir(parents=True)

    early_stop = {
        "format_version": "sabdab2_joint_early_stopping_replay_v1",
        "status": "PASS_AUDIT",
        "baseline": baseline,
        "replay": replay,
        "decision": {
            "retain_patience": 3,
            "reason_code": "LOWER_PATIENCE_MISSES_LATER_VALIDATION_PEAK_FOR_AT_LEAST_ONE_SEED",
            "new_gpu_run_required": False,
        },
        "authority": {"test_loaded": False, "temporal_holdout_loaded": False},
    }
    early_stop_path = authority_root / "EARLY_STOPPING_REPLAY_V1.json"
    early_stop_path.write_bytes(canonical_bytes(early_stop))

    base = {
        "format_version": "sabdab2_joint_regularization_objective_screen_protocol_v1",
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
            "training_partner_panel_sha256": args.train_panel_sha256,
            "validation_partner_panel_sha256": args.validation_panel_sha256,
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
            "dropout": None,
            "learning_rate": 0.0003,
            "weight_decay": 0.0001,
            "batch_size": 1,
            "evaluation_batch_size": 1,
            "precision": "bf16",
            "calibration_max_pairs": 500000,
            "offload_saved_tensors": True,
            "contact_focal_gamma": 1.0,
            "epitope_dice_weight": 1.0,
            "epitope_count_weight": None,
        },
        "training_objective": {
            "loss_weights": {
                "contact": 1.0,
                "paratope": 0.1,
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
            "paratope_loss_weight": 0.1,
            "epitope_loss_weight": 0.5,
            "distance_loss_weight": 0.1,
            "epitope_dice_weight": 1.0,
            "epitope_count_weight": None,
        },
        "regularization": {"dropout": None, "weight_decay": 0.0001},
        "screen_design": {
            "control_reused_from_frozen_joint_current": baseline["17"],
            "factors": {"dropout": [0.1, 0.2], "epitope_count_weight": [0.0, 0.2]},
            "omitted_cell": "dropout010_count020_frozen_control",
            "promotion_gate": {
                "minimum_epitope_delta_vs_control": 0.005,
                "maximum_contact_regression": 0.01,
                "maximum_paratope_regression": 0.01,
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
            "diagnostic_result_sha256": args.diagnostic_result_sha256,
            "curve_audit_sha256": args.curve_audit_sha256,
            "early_stopping_replay_sha256": sha256_file(early_stop_path),
        },
    }

    arm_outputs: dict[str, Any] = {}
    for arm, factors in ARMS.items():
        protocol = copy.deepcopy(base)
        protocol["arm"] = arm
        protocol["regularization"]["dropout"] = factors["dropout"]
        protocol["joint_objective"]["epitope_count_weight"] = factors["epitope_count_weight"]
        protocol["implementation"]["dropout"] = factors["dropout"]
        protocol["implementation"]["epitope_count_weight"] = factors["epitope_count_weight"]
        protocol_path = authority_root / f"PROTOCOL_{arm.upper()}_V1.json"
        protocol_path.write_bytes(canonical_bytes(protocol))
        receipt = {
            "format_version": "sabdab2_joint_regularization_objective_protocol_receipt_v1",
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
            "protocol_sha256": receipt["protocol_file_sha256"],
            "receipt_sha256": sha256_file(receipt_path),
        }

    campaign = {
        "format_version": "sabdab2_joint_regularization_objective_campaign_authority_v1",
        "array_mapping": [
            {"task": task, "arm": arm, "seed": 17}
            for task, arm in enumerate(ARMS)
        ],
        "arms": arm_outputs,
        "baseline": baseline,
        "early_stopping_replay_sha256": sha256_file(early_stop_path),
        "test_loaded": False,
        "temporal_holdout_loaded": False,
    }
    campaign_path = authority_root / "CAMPAIGN_AUTHORITY_V1.json"
    campaign_path.write_bytes(canonical_bytes(campaign))
    print(json.dumps({
        "campaign_authority_sha256": sha256_file(campaign_path),
        "early_stopping_replay_sha256": sha256_file(early_stop_path),
        "arms": arm_outputs,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
