#!/usr/bin/env python3
"""Audit the frozen seed-17 joint-loss balance screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


EXPECTED_CAMPAIGN = "b1b490ae2c009f9f5105b201e85722b82f8b73da2419a0717092abcf584529a4"
EXPECTED_SOURCE = "f649fb5a16cda6ff79eae05b2d2a3fedd6ffd02bd7b0b1cba1f1d0385a9a2f81"
BASELINE_RESULT_SHA256 = "38bbf1a4ade66b01abd81a925ac89464725c67564bc2c4c352701d6d8a0f8184"
BASELINE_RECEIPT_SHA256 = "f63cf0e8f7cf24f1224bbd0b1890cc259fae0de96e66e47b30d23efa61494623"
PRIOR_WINNER_RESULT_SHA256 = "491c27569027c2761e79a7a91059d8df791f441b7f50357bb8180db2e819360d"
PRIOR_WINNER_RECEIPT_SHA256 = "3305b07ab351362191722f0f51e0d14781cfffc603b9128d13c636647942862d"
ARMS = {
    "paratope025": (0.25, "5d0e887b00e800e7babd5402bbceaffabe8fdb32cbc5c21cebc76f925b1c9a16", "725b06bcbd2e8f626304665f0f315b71cb7f547f1ed410a4bec061153b570e0d"),
    "paratope050": (0.50, "b03b48dc7b6fc5c78ea7a406ee3fcfe5f76dad98fa477e31a92deec9bde5adec", "1088dfc71fd379f1c05f0e046a980e21c0545bdb691f024d29bebcebbfe43a4a"),
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


def finite(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("non-finite metric")
    return result


def read_run(
    root: Path, expected_result_sha: str | None = None, expected_receipt_sha: str | None = None
) -> tuple[dict[str, Any], dict[str, Any], str]:
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
    return {
        "result_sha256": result_sha,
        "selected_checkpoint_sha256": receipt["selected_checkpoint_sha256"],
        "selected_epoch": int(result["selection"]["best_epoch"]),
        "epochs_completed": int(result["selection"]["epochs_completed"]),
        "epitope_mean_per_complex_auprc": finite(selected["epitope"]["mean_per_complex_auprc"]),
        "contact_mean_per_complex_auprc": finite(selected["contact"]["mean_per_complex_auprc"]),
        "paratope_mean_per_complex_auprc": finite(selected["paratope"]["mean_per_complex_auprc"]),
        "curve": [
            {
                "epoch": int(row["epoch"]),
                "train_total_loss": finite(row["train"]["loss"]),
                "train_contact_loss": finite(row["train"]["contact_loss"]),
                "train_paratope_loss": finite(row["train"]["paratope_loss"]),
                "train_epitope_loss": finite(row["train"]["epitope_loss"]),
                "validation_contact_mean_per_complex_auprc": finite(row["validation"]["contact"]["mean_per_complex_auprc"]),
                "validation_paratope_mean_per_complex_auprc": finite(row["validation"]["paratope"]["mean_per_complex_auprc"]),
                "validation_epitope_mean_per_complex_auprc": finite(row["validation"]["epitope"]["mean_per_complex_auprc"]),
            }
            for row in history
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--prior-winner-root", type=Path, required=True)
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
    prior_result, prior_receipt, prior_sha = read_run(
        args.prior_winner_root,
        PRIOR_WINNER_RESULT_SHA256,
        PRIOR_WINNER_RECEIPT_SHA256,
    )
    results = {
        "frozen_control": summarize(baseline_result, baseline_sha, baseline_receipt),
        "prior_epitope_candidate": summarize(prior_result, prior_sha, prior_receipt),
    }

    for arm, (paratope_weight, protocol_sha, protocol_receipt_sha) in ARMS.items():
        result, receipt, result_sha = read_run(args.run_root / arm / "seed-17")
        config = result["configuration"]
        authority = result["authority"]
        if (
            result.get("experiment") != "epiformer_style_same_protocol"
            or result.get("data", {}).get("train_complexes") != 512
            or result.get("data", {}).get("validation_complexes") != 128
            or config.get("seed") != 17
            or not math.isclose(finite(config["dropout"]), 0.2)
            or not math.isclose(finite(config["weight_decay"]), 0.0001)
            or not math.isclose(finite(config["contact_loss_weight"]), 1.0)
            or not math.isclose(finite(config["paratope_loss_weight"]), paratope_weight)
            or not math.isclose(finite(config["epitope_loss_weight"]), 0.5)
            or not math.isclose(finite(config["epitope_dice_weight"]), 1.0)
            or not math.isclose(finite(config["epitope_count_weight"]), 0.2)
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

    control = results["frozen_control"]
    prior = results["prior_epitope_candidate"]
    deltas: dict[str, Any] = {}
    passing: list[str] = []
    for arm in ARMS:
        row = results[arm]
        deltas[arm] = {
            "epitope_vs_control": row["epitope_mean_per_complex_auprc"] - control["epitope_mean_per_complex_auprc"],
            "epitope_vs_prior_candidate": row["epitope_mean_per_complex_auprc"] - prior["epitope_mean_per_complex_auprc"],
            "contact_vs_control": row["contact_mean_per_complex_auprc"] - control["contact_mean_per_complex_auprc"],
            "paratope_vs_control": row["paratope_mean_per_complex_auprc"] - control["paratope_mean_per_complex_auprc"],
        }
        gate = {
            "epitope_delta_vs_control": deltas[arm]["epitope_vs_control"] >= 0.0075,
            "epitope_regression_vs_prior_candidate": deltas[arm]["epitope_vs_prior_candidate"] >= -0.005,
            "contact_no_regression": deltas[arm]["contact_vs_control"] >= -0.005,
            "paratope_no_regression": deltas[arm]["paratope_vs_control"] >= -0.01,
        }
        deltas[arm]["gate"] = gate
        deltas[arm]["passes"] = all(gate.values())
        if deltas[arm]["passes"]:
            passing.append(arm)

    winner = max(
        passing,
        key=lambda arm: (
            results[arm]["epitope_mean_per_complex_auprc"],
            results[arm]["paratope_mean_per_complex_auprc"],
            results[arm]["contact_mean_per_complex_auprc"],
        ),
        default=None,
    )
    payload = {
        "format_version": "sabdab2_joint_loss_balance_screen_result_v1",
        "status": "PASS_AUDIT_PROMOTE_TWO_SEEDS" if winner else "PASS_AUDIT_NO_PROMOTION",
        "runs": results,
        "deltas_and_gates": deltas,
        "winner": winner,
        "next_gate": {
            "winner_seeds_29_and_43_authorized": winner is not None,
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
    print(json.dumps({"status": payload["status"], "winner": winner, "deltas_and_gates": deltas, "next_gate": payload["next_gate"]}, sort_keys=True))


if __name__ == "__main__":
    main()
