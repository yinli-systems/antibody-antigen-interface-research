#!/usr/bin/env python3
"""Audit the frozen seed-17 top-k2 edge-readout development screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


CAMPAIGN_SHA256 = "eaa5247b57e2fa02661f7079b11a838c39190b295a01547a628d71a4129520bc"
PROTOCOL_SHA256 = "cac9957ec6c34d27f2c29cbde8d012ccfcb1ca779d4ea378376429d1a238ca94"
PROTOCOL_RECEIPT_SHA256 = "d0c233e4570bdbcda715a93ec9753f4b0fd4131a5d579d6ac07c0d01bdbfbc51"
PARENT_CAMPAIGN_SHA256 = "c9ae16f3f43d9fd72260a181364cb516accf4f557f5eaeda0d3221735a1258b7"
PARENT_RESULT_SHA256 = "7a8226cd94efe8078b9a9f32f8e2754beade34d7f437ce4965c348541ae36094"
SPLIT_SHA256 = "39a32540dd4a6d9abcc8414c23dfbe52d446e9154c06ce58921a0dcbd5039686"
TRAIN_PANEL_SHA256 = "c209b72cfe174c63fa986cb59e8dbf6a3831f128493b0c992a604ddf6a6cfca6"
VALIDATION_PANEL_SHA256 = "86d613c1917d3a8d3fe6da49439cb0a13ab9f44affd2ea27e30043f3d71ae6e2"
BASELINE_RESULT_SHA256 = "6e6da8ee8613130cc9c3a002c9f18a75e7476cfdce201b6cbec63d2ed48cf2f7"
BASELINE_CHECKPOINT_SHA256 = "d4fddde5827eb76837b7496b52bf8959a9a60e9f69c47f4d50f641ffd12a0162"
BASELINE_RECEIPT_SHA256 = "abcdc742e59ff9312fc9b99c8589817eaf479818699d4faaedba5acab38f830a"
BASELINE_PROTOCOL_SHA256 = "7cb1ab917137dfe1de821a1eea846fd22e656a12c1d3dd99a45121bf9f77e181"
BASELINE_PROTOCOL_RECEIPT_SHA256 = "2b48039ea9e39d866e07ac8c718403109509d3db012b713bbfc1c8b028696db5"
BASELINE_SOURCE_MANIFEST_SHA256 = "f649fb5a16cda6ff79eae05b2d2a3fedd6ffd02bd7b0b1cba1f1d0385a9a2f81"
CANDIDATE_SOURCE_MANIFEST_SHA256 = "f09790fdb389bf88c248d1936f8340956ec3692cea8269af8ee4a9716aa2fa55"
FRAMECONTACT_SHA256 = "121b3e06e24db0eeb79db100a3870d19b299b44f77355b590c97ab45cc8f38fe"
METRICS = ("contact", "epitope", "paratope")


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


def require_sha(path: Path, expected: str, label: str) -> None:
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"{label} SHA-256 differs: {observed}")


def finite(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("non-finite metric")
    return result


def audit_result(
    run_root: Path,
    *,
    expected_result_sha256: str | None,
    expected_checkpoint_sha256: str | None,
    expected_receipt_sha256: str | None,
    expected_protocol_sha256: str,
    expected_protocol_receipt_sha256: str,
    expected_source_manifest_sha256: str,
) -> dict[str, Any]:
    result_path = run_root / "FRAMECONTACT_DEVELOPMENT_RESULT.json"
    checkpoint_path = run_root / "FRAMECONTACT_SELECTED.pt"
    receipt_path = run_root / "RUN_RECEIPT.json"
    result_sha256 = sha256_file(result_path)
    checkpoint_sha256 = sha256_file(checkpoint_path)
    receipt_sha256 = sha256_file(receipt_path)
    if expected_result_sha256 and result_sha256 != expected_result_sha256:
        raise ValueError("baseline result SHA-256 differs")
    if expected_checkpoint_sha256 and checkpoint_sha256 != expected_checkpoint_sha256:
        raise ValueError("baseline checkpoint SHA-256 differs")
    if expected_receipt_sha256 and receipt_sha256 != expected_receipt_sha256:
        raise ValueError("baseline receipt SHA-256 differs")

    result = json.loads(result_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    authority = result.get("authority", {})
    configuration = result.get("configuration", {})
    data = result.get("data", {})
    if (
        receipt.get("result_sha256") != result_sha256
        or receipt.get("selected_checkpoint_sha256") != checkpoint_sha256
        or receipt.get("test_loaded") is not False
        or receipt.get("external_test_loaded") is not False
        or result.get("experiment") != "epiformer_style_same_protocol"
        or data.get("train_complexes") != 512
        or data.get("validation_complexes") != 63
        or configuration.get("seed") != 17
        or configuration.get("epochs") != 12
        or configuration.get("early_stopping_patience") != 3
        or configuration.get("dataset_backend") != "sabdab2"
        or configuration.get("device") != "cuda"
        or configuration.get("precision") != "bf16"
        or not math.isclose(finite(configuration["dropout"]), 0.1)
        or not math.isclose(finite(configuration["learning_rate"]), 0.0003)
        or not math.isclose(finite(configuration["weight_decay"]), 0.0001)
        or not math.isclose(finite(configuration["contact_loss_weight"]), 1.0)
        or not math.isclose(finite(configuration["epitope_loss_weight"]), 0.5)
        or not math.isclose(finite(configuration["paratope_loss_weight"]), 0.1)
        or not math.isclose(finite(configuration["distance_loss_weight"]), 0.1)
        or not math.isclose(finite(configuration["epitope_dice_weight"]), 1.0)
        or not math.isclose(finite(configuration["epitope_count_weight"]), 0.2)
        or authority.get("protocol_file_sha256") != expected_protocol_sha256
        or authority.get("protocol_receipt_sha256")
        != expected_protocol_receipt_sha256
        or authority.get("source_manifest_sha256")
        != expected_source_manifest_sha256
        or authority.get("split_manifest_sha256") != SPLIT_SHA256
        or authority.get("training_partner_panel_sha256") != TRAIN_PANEL_SHA256
        or authority.get("validation_partner_panel_sha256")
        != VALIDATION_PANEL_SHA256
        or authority.get("test_loaded") is not False
        or authority.get("external_test_loaded") is not False
    ):
        raise ValueError("result authority or configuration differs")

    history = result.get("history", [])
    selection = result.get("selection", {})
    if not history or int(selection.get("epochs_completed", -1)) != len(history):
        raise ValueError("history differs")
    selected = result["selected_checkpoint_validation"]
    return {
        "result_sha256": result_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "receipt_sha256": receipt_sha256,
        "selected_epoch": int(selection["best_epoch"]),
        "epochs_completed": int(selection["epochs_completed"]),
        "metrics": {
            name: finite(selected[name]["mean_per_complex_auprc"])
            for name in METRICS
        },
        "calibration": {
            name: {
                "brier": finite(selected[name]["brier"]),
                "ece_15": finite(selected[name]["ece_15"]),
            }
            for name in METRICS
        },
        "curve": [
            {
                "epoch": int(row["epoch"]),
                "train_total_loss": finite(row["train"]["loss"]),
                "validation": {
                    name: finite(row["validation"][name]["mean_per_complex_auprc"])
                    for name in METRICS
                },
            }
            for row in history
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)

    parent_authority = args.parent_root / "authority"
    authority = args.run_root / "authority"
    require_sha(
        parent_authority / "CAMPAIGN_EDGE_MEAN_TRAINING_V1.json",
        PARENT_CAMPAIGN_SHA256,
        "parent campaign",
    )
    require_sha(
        args.parent_root / "EDGE_MEAN_TRAINING_SELECTION_RESULT_V1.json",
        PARENT_RESULT_SHA256,
        "parent selection result",
    )
    require_sha(
        parent_authority / "split-train512-selection63.csv",
        SPLIT_SHA256,
        "selection split",
    )
    require_sha(
        parent_authority / "train512-reciprocal-6.json",
        TRAIN_PANEL_SHA256,
        "training partner panel",
    )
    require_sha(
        parent_authority / "validation-selection63-reciprocal-6.json",
        VALIDATION_PANEL_SHA256,
        "validation partner panel",
    )
    require_sha(
        authority / "CAMPAIGN_EDGE_TOPK2_TRAINING_V1.json",
        CAMPAIGN_SHA256,
        "top-k2 campaign",
    )
    require_sha(
        authority / "PROTOCOL_EDGE_TOPK2_SUPERVISED_V1.json",
        PROTOCOL_SHA256,
        "top-k2 protocol",
    )
    require_sha(
        authority / "PROTOCOL_EDGE_TOPK2_SUPERVISED_V1_RECEIPT.json",
        PROTOCOL_RECEIPT_SHA256,
        "top-k2 protocol receipt",
    )

    campaign = json.loads(
        (authority / "CAMPAIGN_EDGE_TOPK2_TRAINING_V1.json").read_text()
    )
    protocol = json.loads(
        (authority / "PROTOCOL_EDGE_TOPK2_SUPERVISED_V1.json").read_text()
    )
    protocol_receipt = json.loads(
        (authority / "PROTOCOL_EDGE_TOPK2_SUPERVISED_V1_RECEIPT.json").read_text()
    )
    if (
        campaign.get("status") != "FROZEN_BEFORE_TRAINING"
        or campaign.get("arm") != "edge_topk2_supervised"
        or campaign.get("selection_seed") != 17
        or campaign.get("protocol_sha256") != PROTOCOL_SHA256
        or campaign.get("protocol_receipt_sha256") != PROTOCOL_RECEIPT_SHA256
        or campaign.get("source_manifest_sha256")
        != CANDIDATE_SOURCE_MANIFEST_SHA256
        or campaign.get("framecontact_sha256") != FRAMECONTACT_SHA256
        or campaign.get("baseline_result_sha256") != BASELINE_RESULT_SHA256
        or campaign.get("confirmation_panel_loaded") is not False
        or campaign.get("test_loaded") is not False
        or campaign.get("temporal_holdout_loaded") is not False
        or protocol.get("protocol_status") != "frozen_before_training"
        or protocol.get("arm") != "edge_topk2_supervised"
        or protocol.get("epitope_readout", {}).get("rule")
        != "masked_top_k_mean_2_edge_probability_no_direct_head"
        or protocol.get("epitope_readout", {}).get("single_factor_only") is not True
        or protocol.get("claim_boundary", {}).get("confirmation_panel_loaded")
        is not False
        or protocol.get("claim_boundary", {}).get("test_loaded") is not False
        or protocol.get("claim_boundary", {}).get("temporal_holdout_loaded")
        is not False
        or protocol_receipt.get("protocol_file_sha256") != PROTOCOL_SHA256
        or protocol_receipt.get("source_manifest_sha256")
        != CANDIDATE_SOURCE_MANIFEST_SHA256
        or protocol_receipt.get("training_split_sha256") != SPLIT_SHA256
        or protocol_receipt.get("training_partner_panel_sha256")
        != TRAIN_PANEL_SHA256
        or protocol_receipt.get("validation_partner_panel_sha256")
        != VALIDATION_PANEL_SHA256
        or protocol_receipt.get("confirmation_panel_loaded") is not False
        or protocol_receipt.get("test_loaded") is not False
        or protocol_receipt.get("temporal_holdout_loaded") is not False
    ):
        raise ValueError("frozen top-k2 authority differs")

    baseline = audit_result(
        args.parent_root / "runs/baseline_current/seed-17",
        expected_result_sha256=BASELINE_RESULT_SHA256,
        expected_checkpoint_sha256=BASELINE_CHECKPOINT_SHA256,
        expected_receipt_sha256=BASELINE_RECEIPT_SHA256,
        expected_protocol_sha256=BASELINE_PROTOCOL_SHA256,
        expected_protocol_receipt_sha256=BASELINE_PROTOCOL_RECEIPT_SHA256,
        expected_source_manifest_sha256=BASELINE_SOURCE_MANIFEST_SHA256,
    )
    candidate = audit_result(
        args.run_root / "runs/edge_topk2_supervised/seed-17",
        expected_result_sha256=None,
        expected_checkpoint_sha256=None,
        expected_receipt_sha256=None,
        expected_protocol_sha256=PROTOCOL_SHA256,
        expected_protocol_receipt_sha256=PROTOCOL_RECEIPT_SHA256,
        expected_source_manifest_sha256=CANDIDATE_SOURCE_MANIFEST_SHA256,
    )
    delta = {
        name: candidate["metrics"][name] - baseline["metrics"][name]
        for name in METRICS
    }
    checks = {
        "epitope_delta_at_least_0_005": delta["epitope"] >= 0.005,
        "contact_regression_at_most_0_01": delta["contact"] >= -0.01,
        "paratope_regression_at_most_0_01": delta["paratope"] >= -0.01,
    }
    passed = all(checks.values())
    payload = {
        "format_version": "sabdab2_edge_topk2_training_selection_result_v1",
        "status": "PASS_FREEZE_THREE_SEED_CONFIRMATION" if passed else "FAIL_CLEAN",
        "runs": {"baseline_current": baseline, "edge_topk2_supervised": candidate},
        "delta_candidate_minus_baseline": delta,
        "gate_checks": checks,
        "authorization": {
            "freeze_three_seed_confirmation_development_protocol": passed,
            "three_seed_confirmation_started": False,
            "full_development": False,
            "old_test": False,
            "temporal_holdout": False,
        },
        "authority": {
            "campaign_sha256": CAMPAIGN_SHA256,
            "protocol_sha256": PROTOCOL_SHA256,
            "protocol_receipt_sha256": PROTOCOL_RECEIPT_SHA256,
            "parent_campaign_sha256": PARENT_CAMPAIGN_SHA256,
            "parent_result_sha256": PARENT_RESULT_SHA256,
            "selection_panel_only": True,
            "confirmation_panel_loaded": False,
            "test_loaded": False,
            "temporal_holdout_loaded": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_bytes(payload))
    print(
        json.dumps(
            {
                "status": payload["status"],
                "delta": delta,
                "output_sha256": sha256_file(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
