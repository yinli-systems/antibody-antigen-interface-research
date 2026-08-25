#!/usr/bin/env python3
"""Audit the frozen seed-17 regularization/objective screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


EXPECTED_CAMPAIGN = "64779b5b7cece143359d9d9a95cdb7244db718f69a4cc31cadd72f0793714405"
EXPECTED_SOURCE = "f649fb5a16cda6ff79eae05b2d2a3fedd6ffd02bd7b0b1cba1f1d0385a9a2f81"
BASELINE_RESULT_SHA256 = "38bbf1a4ade66b01abd81a925ac89464725c67564bc2c4c352701d6d8a0f8184"
BASELINE_RECEIPT_SHA256 = "f63cf0e8f7cf24f1224bbd0b1890cc259fae0de96e66e47b30d23efa61494623"
ARMS = {
    "dropout020_count020": (0.2, 0.2, "3fb7d3e5176f657cb96831a2cf925c13eadddea2043f1659ee93da76011cd275", "baf00d571ddc6be34fdea6b015cefccccfb63ec5d0dd656c95efb9c86d0c20e8"),
    "dropout010_count000": (0.1, 0.0, "b51e51a7ad51f17bdbdafb2f6f5902bb3aa50d8099a8b26c01d4cd4ef4f0536c", "36116d2767bc4a2a764a286be317240528b9c33e160ab711538da2b3a1d17530"),
    "dropout020_count000": (0.2, 0.0, "1bf5308e7b49882f68231fd897a9df7243fc23bf2c433e8bdeae57260404fc2a", "aa5eaf8134cd97538d84fef374e15a479f1e01342b8c9e54ea0f4ca8ff5c0e58"),
}


def canonical_bytes(payload: object) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


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


def read_run(root: Path, expected_result_sha: str | None = None, expected_receipt_sha: str | None = None) -> tuple[dict[str, Any], dict[str, Any], str]:
    result_path = root / "FRAMECONTACT_DEVELOPMENT_RESULT.json"
    receipt_path = root / "RUN_RECEIPT.json"
    result_sha = sha256_file(result_path)
    receipt_sha = sha256_file(receipt_path)
    if expected_result_sha is not None and result_sha != expected_result_sha:
        raise ValueError(f"result SHA differs: {root}")
    if expected_receipt_sha is not None and receipt_sha != expected_receipt_sha:
        raise ValueError(f"receipt SHA differs: {root}")
    result = json.loads(result_path.read_text())
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("result_sha256") != result_sha:
        raise ValueError(f"receipt does not authenticate result: {root}")
    return result, receipt, result_sha


def summarize(result: dict[str, Any], result_sha: str, receipt: dict[str, Any]) -> dict[str, Any]:
    selected = result["selected_checkpoint_validation"]
    history = result.get("history", [])
    if not history:
        raise ValueError("empty history")
    best = finite(selected["epitope"]["mean_per_complex_auprc"])
    final = finite(history[-1]["validation"]["epitope"]["mean_per_complex_auprc"])
    return {
        "result_sha256": result_sha,
        "selected_checkpoint_sha256": receipt["selected_checkpoint_sha256"],
        "selected_epoch": int(result["selection"]["best_epoch"]),
        "epochs_completed": int(result["selection"]["epochs_completed"]),
        "epitope_mean_per_complex_auprc": best,
        "contact_mean_per_complex_auprc": finite(selected["contact"]["mean_per_complex_auprc"]),
        "paratope_mean_per_complex_auprc": finite(selected["paratope"]["mean_per_complex_auprc"]),
        "final_minus_best_epitope_auprc": final - best,
        "curve": [
            {
                "epoch": int(row["epoch"]),
                "train_total_loss": finite(row["train"]["loss"]),
                "train_epitope_loss": finite(row["train"]["epitope_loss"]),
                "validation_epitope_mean_per_complex_auprc": finite(row["validation"]["epitope"]["mean_per_complex_auprc"]),
            }
            for row in history
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)

    campaign_path = args.run_root / "authority" / "CAMPAIGN_AUTHORITY_V1.json"
    if sha256_file(campaign_path) != EXPECTED_CAMPAIGN:
        raise ValueError("campaign authority differs")
    campaign = json.loads(campaign_path.read_text())
    if campaign.get("test_loaded") is not False or campaign.get("temporal_holdout_loaded") is not False:
        raise ValueError("campaign scope differs")

    baseline_result, baseline_receipt, baseline_sha = read_run(
        args.baseline_root / "joint_current" / "seed-17",
        BASELINE_RESULT_SHA256,
        BASELINE_RECEIPT_SHA256,
    )
    baseline_config = baseline_result["configuration"]
    if (
        baseline_result.get("experiment") != "epiformer_style_same_protocol"
        or baseline_config.get("seed") != 17
        or not math.isclose(finite(baseline_config["dropout"]), 0.1)
        or not math.isclose(finite(baseline_config["epitope_count_weight"]), 0.2)
        or baseline_result.get("authority", {}).get("test_loaded") is not False
        or baseline_receipt.get("test_loaded") is not False
    ):
        raise ValueError("baseline authority differs")
    results = {"dropout010_count020_control": summarize(baseline_result, baseline_sha, baseline_receipt)}

    for arm, (dropout, count_weight, protocol_sha, protocol_receipt_sha) in ARMS.items():
        result, receipt, result_sha = read_run(args.run_root / arm / "seed-17")
        config = result["configuration"]
        authority = result["authority"]
        if (
            result.get("experiment") != "epiformer_style_same_protocol"
            or result.get("data", {}).get("train_complexes") != 512
            or result.get("data", {}).get("validation_complexes") != 128
            or config.get("seed") != 17
            or not math.isclose(finite(config["dropout"]), dropout)
            or not math.isclose(finite(config["weight_decay"]), 0.0001)
            or not math.isclose(finite(config["contact_loss_weight"]), 1.0)
            or not math.isclose(finite(config["paratope_loss_weight"]), 0.1)
            or not math.isclose(finite(config["epitope_loss_weight"]), 0.5)
            or not math.isclose(finite(config["epitope_dice_weight"]), 1.0)
            or not math.isclose(finite(config["epitope_count_weight"]), count_weight)
            or config.get("early_stopping_patience") != 3
            or authority.get("protocol_file_sha256") != protocol_sha
            or authority.get("protocol_receipt_sha256") != protocol_receipt_sha
            or authority.get("source_manifest_sha256") != EXPECTED_SOURCE
            or authority.get("test_loaded") is not False
            or authority.get("external_test_loaded") is not False
            or receipt.get("test_loaded") is not False
            or receipt.get("external_test_loaded") is not False
        ):
            raise ValueError(f"authority differs for {arm}")
        results[arm] = summarize(result, result_sha, receipt)

    control = results["dropout010_count020_control"]
    deltas: dict[str, Any] = {}
    for arm, row in results.items():
        if arm == "dropout010_count020_control":
            continue
        deltas[arm] = {
            "epitope": row["epitope_mean_per_complex_auprc"] - control["epitope_mean_per_complex_auprc"],
            "contact": row["contact_mean_per_complex_auprc"] - control["contact_mean_per_complex_auprc"],
            "paratope": row["paratope_mean_per_complex_auprc"] - control["paratope_mean_per_complex_auprc"],
            "final_minus_best_epitope_auprc_change": row["final_minus_best_epitope_auprc"] - control["final_minus_best_epitope_auprc"],
        }

    winner = max(results, key=lambda arm: results[arm]["epitope_mean_per_complex_auprc"])
    winner_delta = {"epitope": 0.0, "contact": 0.0, "paratope": 0.0} if winner == "dropout010_count020_control" else deltas[winner]
    promote = (
        winner != "dropout010_count020_control"
        and winner_delta["epitope"] >= 0.005
        and winner_delta["contact"] >= -0.01
        and winner_delta["paratope"] >= -0.01
    )
    factorial = {
        "dropout_effect_when_count020": results["dropout020_count020"]["epitope_mean_per_complex_auprc"] - control["epitope_mean_per_complex_auprc"],
        "remove_count_effect_when_dropout010": results["dropout010_count000"]["epitope_mean_per_complex_auprc"] - control["epitope_mean_per_complex_auprc"],
        "dropout_effect_when_count000": results["dropout020_count000"]["epitope_mean_per_complex_auprc"] - results["dropout010_count000"]["epitope_mean_per_complex_auprc"],
        "dropout_count_interaction": (
            results["dropout020_count000"]["epitope_mean_per_complex_auprc"]
            - results["dropout010_count000"]["epitope_mean_per_complex_auprc"]
            - results["dropout020_count020"]["epitope_mean_per_complex_auprc"]
            + control["epitope_mean_per_complex_auprc"]
        ),
    }
    payload = {
        "format_version": "sabdab2_joint_regularization_objective_screen_result_v1",
        "status": "PASS_AUDIT",
        "runs": results,
        "deltas_vs_control": deltas,
        "factorial_epitope_effects": factorial,
        "winner": winner,
        "winner_delta_vs_control": winner_delta,
        "next_gate": {
            "winner_seeds_29_and_43_authorized": promote,
            "full_2565_549_development_authorized": False,
            "test_or_temporal_holdout_authorized": False,
        },
        "authority": {
            "campaign_authority_sha256": EXPECTED_CAMPAIGN,
            "test_loaded": False,
            "temporal_holdout_loaded": False,
        },
    }
    args.output.write_bytes(canonical_bytes(payload))
    print(json.dumps({"winner": winner, "winner_delta_vs_control": winner_delta, "next_gate": payload["next_gate"]}, sort_keys=True))


if __name__ == "__main__":
    main()
