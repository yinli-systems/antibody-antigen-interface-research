#!/usr/bin/env python3
"""Audit the frozen seed-17 edge-mean training selection screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


EXPECTED_CAMPAIGN_SHA256 = (
    "c9ae16f3f43d9fd72260a181364cb516accf4f557f5eaeda0d3221735a1258b7"
)
EXPECTED_SPLIT_SHA256 = (
    "39a32540dd4a6d9abcc8414c23dfbe52d446e9154c06ce58921a0dcbd5039686"
)
EXPECTED_TRAIN_PANEL_SHA256 = (
    "c209b72cfe174c63fa986cb59e8dbf6a3831f128493b0c992a604ddf6a6cfca6"
)
EXPECTED_VALIDATION_PANEL_SHA256 = (
    "86d613c1917d3a8d3fe6da49439cb0a13ab9f44affd2ea27e30043f3d71ae6e2"
)
EXPECTED_ARMS = {
    "baseline_current": {
        "protocol_sha256": (
            "7cb1ab917137dfe1de821a1eea846fd22e656a12c1d3dd99a45121bf9f77e181"
        ),
        "protocol_receipt_sha256": (
            "2b48039ea9e39d866e07ac8c718403109509d3db012b713bbfc1c8b028696db5"
        ),
        "source_manifest_sha256": (
            "f649fb5a16cda6ff79eae05b2d2a3fedd6ffd02bd7b0b1cba1f1d0385a9a2f81"
        ),
    },
    "edge_mean_supervised": {
        "protocol_sha256": (
            "699b7e22ddd9cdc885d0ed4e6d8e18b2b02b914ea1317cdfa93de4d330ccdbc7"
        ),
        "protocol_receipt_sha256": (
            "a77e9cb0085b84dce40eca184b469a7a4560e034ffc51f8ff369ed020aaed533"
        ),
        "source_manifest_sha256": (
            "993205cff450c8585bf79757fa0e97f5bbc96a5cf48b8df39bbed822d624c6a3"
        ),
    },
}
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


def finite(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("non-finite metric")
    return result


def protocol_path(authority: Path, arm: str) -> Path:
    return authority / f"PROTOCOL_{arm.upper()}_V1.json"


def protocol_receipt_path(authority: Path, arm: str) -> Path:
    return authority / f"PROTOCOL_{arm.upper()}_V1_RECEIPT.json"


def read_run(run_root: Path, authority: Path, arm: str) -> dict[str, Any]:
    expected = EXPECTED_ARMS[arm]
    result_path = run_root / "FRAMECONTACT_DEVELOPMENT_RESULT.json"
    receipt_path = run_root / "RUN_RECEIPT.json"
    checkpoint_path = run_root / "FRAMECONTACT_SELECTED.pt"
    result_sha256 = sha256_file(result_path)
    receipt_sha256 = sha256_file(receipt_path)
    checkpoint_sha256 = sha256_file(checkpoint_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if (
        receipt.get("result_sha256") != result_sha256
        or receipt.get("selected_checkpoint_sha256") != checkpoint_sha256
        or receipt.get("test_loaded") is not False
        or receipt.get("external_test_loaded") is not False
    ):
        raise ValueError(f"run receipt differs for {arm}")

    protocol = protocol_path(authority, arm)
    protocol_receipt = protocol_receipt_path(authority, arm)
    if (
        sha256_file(protocol) != expected["protocol_sha256"]
        or sha256_file(protocol_receipt) != expected["protocol_receipt_sha256"]
    ):
        raise ValueError(f"frozen protocol differs for {arm}")
    frozen = json.loads(protocol.read_text(encoding="utf-8"))
    frozen_receipt = json.loads(protocol_receipt.read_text(encoding="utf-8"))
    configuration = result.get("configuration", {})
    result_authority = result.get("authority", {})
    data = result.get("data", {})
    if (
        frozen.get("protocol_status") != "frozen_before_training"
        or frozen.get("arm") != arm
        or frozen.get("claim_boundary", {}).get("confirmation_panel_loaded")
        is not False
        or frozen.get("claim_boundary", {}).get("test_loaded") is not False
        or frozen.get("claim_boundary", {}).get("temporal_holdout_loaded")
        is not False
        or frozen_receipt.get("protocol_file_sha256")
        != expected["protocol_sha256"]
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
        or result_authority.get("protocol_file_sha256")
        != expected["protocol_sha256"]
        or result_authority.get("protocol_receipt_sha256")
        != expected["protocol_receipt_sha256"]
        or result_authority.get("source_manifest_sha256")
        != expected["source_manifest_sha256"]
        or result_authority.get("split_manifest_sha256") != EXPECTED_SPLIT_SHA256
        or result_authority.get("training_partner_panel_sha256")
        != EXPECTED_TRAIN_PANEL_SHA256
        or result_authority.get("validation_partner_panel_sha256")
        != EXPECTED_VALIDATION_PANEL_SHA256
        or result_authority.get("test_loaded") is not False
        or result_authority.get("external_test_loaded") is not False
    ):
        raise ValueError(f"result authority or configuration differs for {arm}")

    selected = result["selected_checkpoint_validation"]
    history = result.get("history", [])
    selection = result.get("selection", {})
    if not history or int(selection.get("epochs_completed", -1)) != len(history):
        raise ValueError(f"history differs for {arm}")
    metrics = {
        name: finite(selected[name]["mean_per_complex_auprc"])
        for name in METRICS
    }
    return {
        "result_sha256": result_sha256,
        "receipt_sha256": receipt_sha256,
        "selected_checkpoint_sha256": checkpoint_sha256,
        "selected_epoch": int(selection["best_epoch"]),
        "epochs_completed": int(selection["epochs_completed"]),
        "metrics": metrics,
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
                "train_epitope_loss": finite(row["train"]["epitope_loss"]),
                "validation": {
                    name: finite(
                        row["validation"][name]["mean_per_complex_auprc"]
                    )
                    for name in METRICS
                },
            }
            for row in history
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)

    authority = args.run_root / "authority"
    campaign_path = authority / "CAMPAIGN_EDGE_MEAN_TRAINING_V1.json"
    if sha256_file(campaign_path) != EXPECTED_CAMPAIGN_SHA256:
        raise ValueError("campaign authority differs")
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    if (
        campaign.get("status") != "FROZEN_BEFORE_TRAINING"
        or campaign.get("test_loaded") is not False
        or campaign.get("temporal_holdout_loaded") is not False
        or campaign.get("training_split_sha256") != EXPECTED_SPLIT_SHA256
        or campaign.get("training_partner_panel_sha256")
        != EXPECTED_TRAIN_PANEL_SHA256
        or campaign.get("validation_partner_panel_sha256")
        != EXPECTED_VALIDATION_PANEL_SHA256
    ):
        raise ValueError("campaign scope differs")

    runs = {
        arm: read_run(args.run_root / "runs" / arm / "seed-17", authority, arm)
        for arm in EXPECTED_ARMS
    }
    baseline = runs["baseline_current"]["metrics"]
    candidate = runs["edge_mean_supervised"]["metrics"]
    delta = {name: candidate[name] - baseline[name] for name in METRICS}
    checks = {
        "epitope_delta_at_least_0_005": delta["epitope"] >= 0.005,
        "contact_regression_at_most_0_01": delta["contact"] >= -0.01,
        "paratope_regression_at_most_0_01": delta["paratope"] >= -0.01,
    }
    passed = all(checks.values())
    payload = {
        "format_version": "sabdab2_edge_mean_training_selection_result_v1",
        "status": "PASS_OPEN_CONFIRMATION" if passed else "FAIL_CLEAN",
        "runs": runs,
        "delta_candidate_minus_baseline": delta,
        "gate_checks": checks,
        "authorization": {
            "selection_checkpoint_confirmation_panel_measurement_only": passed,
            "seeds_29_43_development_only": False,
            "full_development": False,
            "old_test": False,
            "temporal_holdout": False,
        },
        "authority": {
            "campaign_sha256": EXPECTED_CAMPAIGN_SHA256,
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
