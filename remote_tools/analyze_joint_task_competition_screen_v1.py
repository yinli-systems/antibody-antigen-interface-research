#!/usr/bin/env python3
"""Audit the 2x2 task-weight screen and diagnose epitope interference."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


SEEDS = (17, 29, 43)
EXPECTED_CAMPAIGN = "cc1892e720f98567dc62ebf4e85343cd7deb1d6d862b50d5934dc0acf01d30ab"
EXPECTED_SOURCE = "f649fb5a16cda6ff79eae05b2d2a3fedd6ffd02bd7b0b1cba1f1d0385a9a2f81"
ARMS = {
    "epitope_only": (0.0, 0.0, "34cd4150894d7429ff9bd5d11d777c34627106f00d7d2d0545bcdae0e4271012", "1a32103081e19bb0f1bcd6a49cce29b8b90ae81ae847f924d05859aa64686900"),
    "epitope_paratope": (0.0, 0.1, "668c5b29ca814e8e00c1aefb4eaa87ee515a3fbe3bfbf6fffc8d3ceeb789c19e", "0170a1561b8a9176012f2a2642ab214521fc1a4d4a2327065486791076ea672b"),
    "epitope_contact": (1.0, 0.0, "d8016c6da54116c2b579da21ef1092fbe137911844938721adc75fb43d1a3f8f", "c12b13949b50057bbb646a32b089dc5b15575580279c3bd1eaa2a7050aa6bb2d"),
    "joint_current": (1.0, 0.1, "2678f09ff99639862acb75147b100deffb14046bdded4d9c06ca31181bfae335", "dd4ea39df0e60f7b538997c2a1177113f831c04a73b74555613860a50f2d8069"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("non-finite metric")
    return result


def canonical_bytes(payload: object) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def median(values: list[float]) -> float:
    return float(statistics.median(values))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    campaign_path = args.run_root / "authority" / "CAMPAIGN_AUTHORITY_V1.json"
    if sha256_file(campaign_path) != EXPECTED_CAMPAIGN:
        raise ValueError("campaign authority differs")
    campaign = json.loads(campaign_path.read_text())
    if campaign.get("seeds") != list(SEEDS) or campaign.get("test_loaded") is not False:
        raise ValueError("campaign scope differs")

    arms: dict[str, Any] = {}
    for arm, (contact_weight, paratope_weight, protocol_sha, protocol_receipt_sha) in ARMS.items():
        runs = {}
        for seed in SEEDS:
            root = args.run_root / arm / f"seed-{seed}"
            result_path = root / "FRAMECONTACT_DEVELOPMENT_RESULT.json"
            receipt_path = root / "RUN_RECEIPT.json"
            result = json.loads(result_path.read_text())
            receipt = json.loads(receipt_path.read_text())
            result_sha = sha256_file(result_path)
            config = result["configuration"]
            authority = result["authority"]
            selected = result["selected_checkpoint_validation"]
            if (
                result.get("experiment") != "epiformer_style_same_protocol"
                or result.get("data", {}).get("train_complexes") != 512
                or result.get("data", {}).get("validation_complexes") != 128
                or config.get("seed") != seed
                or not math.isclose(finite(config["contact_loss_weight"]), contact_weight, abs_tol=1e-15)
                or not math.isclose(finite(config["paratope_loss_weight"]), paratope_weight, abs_tol=1e-15)
                or not math.isclose(finite(config["epitope_loss_weight"]), 0.5, abs_tol=1e-15)
                or result.get("selection", {}).get("primary_metric") != "epitope_mean_per_complex_auprc"
                or authority.get("protocol_file_sha256") != protocol_sha
                or authority.get("protocol_receipt_sha256") != protocol_receipt_sha
                or authority.get("source_manifest_sha256") != EXPECTED_SOURCE
                or authority.get("test_loaded") is not False
                or authority.get("external_test_loaded") is not False
                or receipt.get("result_sha256") != result_sha
                or receipt.get("test_loaded") is not False
                or receipt.get("external_test_loaded") is not False
            ):
                raise ValueError(f"authority differs for {arm} seed {seed}")
            history = result.get("history", [])
            if not history:
                raise ValueError(f"empty history for {arm} seed {seed}")
            curve = [
                {
                    "epoch": int(row["epoch"]),
                    "train_total_loss": finite(row["train"]["loss"]),
                    "train_epitope_loss": finite(row["train"]["epitope_loss"]),
                    "validation_epitope_mean_per_complex_auprc": finite(row["validation"]["epitope"]["mean_per_complex_auprc"]),
                }
                for row in history
            ]
            selected_epitope = finite(selected["epitope"]["mean_per_complex_auprc"])
            if not math.isclose(selected_epitope, finite(result["selection"]["best_validation_primary_metric"]), abs_tol=1e-12):
                raise ValueError(f"selected primary differs for {arm} seed {seed}")
            runs[str(seed)] = {
                "result_sha256": result_sha,
                "selected_checkpoint_sha256": receipt["selected_checkpoint_sha256"],
                "selected_epoch": int(result["selection"]["best_epoch"]),
                "epochs_completed": int(result["selection"]["epochs_completed"]),
                "epitope_mean_per_complex_auprc": selected_epitope,
                "contact_mean_per_complex_auprc": finite(selected["contact"]["mean_per_complex_auprc"]),
                "paratope_mean_per_complex_auprc": finite(selected["paratope"]["mean_per_complex_auprc"]),
                "final_minus_best_epitope_auprc": curve[-1]["validation_epitope_mean_per_complex_auprc"] - selected_epitope,
                "curve": curve,
            }
        arms[arm] = {
            "runs": runs,
            "median_epitope_mean_per_complex_auprc": median([runs[str(seed)]["epitope_mean_per_complex_auprc"] for seed in SEEDS]),
            "median_best_epoch": median([float(runs[str(seed)]["selected_epoch"]) for seed in SEEDS]),
            "median_final_minus_best_epitope_auprc": median([runs[str(seed)]["final_minus_best_epitope_auprc"] for seed in SEEDS]),
        }

    def paired_delta(positive: str, negative: str) -> dict[str, float]:
        return {
            str(seed): arms[positive]["runs"][str(seed)]["epitope_mean_per_complex_auprc"] - arms[negative]["runs"][str(seed)]["epitope_mean_per_complex_auprc"]
            for seed in SEEDS
        }

    effects = {
        "contact_when_paratope_absent": paired_delta("epitope_contact", "epitope_only"),
        "contact_when_paratope_present": paired_delta("joint_current", "epitope_paratope"),
        "paratope_when_contact_absent": paired_delta("epitope_paratope", "epitope_only"),
        "paratope_when_contact_present": paired_delta("joint_current", "epitope_contact"),
    }
    interaction = {
        str(seed): effects["contact_when_paratope_present"][str(seed)] - effects["contact_when_paratope_absent"][str(seed)]
        for seed in SEEDS
    }
    effect_summary = {
        name: {
            "median": median(list(values.values())),
            "positive_seed_count": sum(value > 0 for value in values.values()),
            "per_seed": values,
        }
        for name, values in effects.items()
    }
    effect_summary["contact_paratope_interaction"] = {
        "median": median(list(interaction.values())),
        "positive_seed_count": sum(value > 0 for value in interaction.values()),
        "per_seed": interaction,
    }
    joint = arms["joint_current"]["runs"]
    alternatives = [arm for arm in ARMS if arm != "joint_current"]
    winner = max(ARMS, key=lambda arm: arms[arm]["median_epitope_mean_per_complex_auprc"])
    winner_deltas = {
        str(seed): arms[winner]["runs"][str(seed)]["epitope_mean_per_complex_auprc"] - joint[str(seed)]["epitope_mean_per_complex_auprc"]
        for seed in SEEDS
    }
    stable_winner = (
        winner != "joint_current"
        and median(list(winner_deltas.values())) >= 0.005
        and sum(value > 0 for value in winner_deltas.values()) >= 2
    )
    payload = {
        "format_version": "sabdab2_joint_task_competition_screen_result_v1",
        "status": "PASS_AUDIT",
        "arms": arms,
        "factorial_epitope_effects": effect_summary,
        "best_median_epitope_arm": winner,
        "best_arm_minus_joint_current": {
            "median": median(list(winner_deltas.values())),
            "positive_seed_count": sum(value > 0 for value in winner_deltas.values()),
            "per_seed": winner_deltas,
        },
        "diagnosis": {
            "contact_task_competition_supported": effect_summary["contact_when_paratope_absent"]["median"] < 0 and effect_summary["contact_when_paratope_present"]["median"] < 0,
            "paratope_task_competition_supported": effect_summary["paratope_when_contact_absent"]["median"] < 0 and effect_summary["paratope_when_contact_present"]["median"] < 0,
            "late_overfitting_in_all_arms": all(arms[arm]["median_final_minus_best_epitope_auprc"] < 0 for arm in ARMS),
            "stable_epitope_optimized_winner": stable_winner,
        },
        "next_gate": {
            "full_2565_549_paired_winner_vs_joint_authorized": stable_winner,
            "gradient_conflict_claim_authorized": False,
            "test_or_temporal_holdout_authorized": False,
        },
        "authority": {"campaign_authority_sha256": EXPECTED_CAMPAIGN, "test_loaded": False, "temporal_holdout_loaded": False},
    }
    args.output.write_bytes(canonical_bytes(payload))
    print(json.dumps({"winner": winner, "winner_minus_joint": payload["best_arm_minus_joint_current"], "diagnosis": payload["diagnosis"], "next_gate": payload["next_gate"]}, sort_keys=True))


if __name__ == "__main__":
    main()
