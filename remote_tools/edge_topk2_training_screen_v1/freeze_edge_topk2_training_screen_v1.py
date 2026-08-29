#!/usr/bin/env python3
"""Freeze a single-factor EpiFormer top-k2 edge-readout development screen."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


PARENT_CAMPAIGN_SHA256 = "c9ae16f3f43d9fd72260a181364cb516accf4f557f5eaeda0d3221735a1258b7"
PARENT_SELECTION_RESULT_SHA256 = "7a8226cd94efe8078b9a9f32f8e2754beade34d7f437ce4965c348541ae36094"
PARENT_CANDIDATE_PROTOCOL_SHA256 = "699b7e22ddd9cdc885d0ed4e6d8e18b2b02b914ea1317cdfa93de4d330ccdbc7"
SPLIT_SHA256 = "39a32540dd4a6d9abcc8414c23dfbe52d446e9154c06ce58921a0dcbd5039686"
TRAIN_PANEL_SHA256 = "c209b72cfe174c63fa986cb59e8dbf6a3831f128493b0c992a604ddf6a6cfca6"
VALIDATION_PANEL_SHA256 = "86d613c1917d3a8d3fe6da49439cb0a13ab9f44affd2ea27e30043f3d71ae6e2"
BASELINE_RESULT_SHA256 = "6e6da8ee8613130cc9c3a002c9f18a75e7476cfdce201b6cbec63d2ed48cf2f7"
BASELINE_CHECKPOINT_SHA256 = "d4fddde5827eb76837b7496b52bf8959a9a60e9f69c47f4d50f641ffd12a0162"
BASELINE_RECEIPT_SHA256 = "abcdc742e59ff9312fc9b99c8589817eaf479818699d4faaedba5acab38f830a"
SOURCE_MANIFEST_SHA256 = "f09790fdb389bf88c248d1936f8340956ec3692cea8269af8ee4a9716aa2fa55"
FRAMECONTACT_SHA256 = "121b3e06e24db0eeb79db100a3870d19b299b44f77355b590c97ab45cc8f38fe"


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


def canonical_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()


def write_new(path: Path, payload: object) -> str:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.write_bytes(canonical_bytes(payload))
    return sha256_file(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-root", type=Path, required=True)
    parser.add_argument("--candidate-source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    authority = args.parent_root / "authority"
    baseline = args.parent_root / "runs/baseline_current/seed-17"
    candidate_manifest = args.candidate_source_root / "SOURCE_MANIFEST_EDGE_TOPK2_V1.json"
    checks = (
        (authority / "CAMPAIGN_EDGE_MEAN_TRAINING_V1.json", PARENT_CAMPAIGN_SHA256, "parent campaign"),
        (args.parent_root / "EDGE_MEAN_TRAINING_SELECTION_RESULT_V1.json", PARENT_SELECTION_RESULT_SHA256, "parent result"),
        (authority / "PROTOCOL_EDGE_MEAN_SUPERVISED_V1.json", PARENT_CANDIDATE_PROTOCOL_SHA256, "parent candidate protocol"),
        (authority / "split-train512-selection63.csv", SPLIT_SHA256, "selection split"),
        (authority / "train512-reciprocal-6.json", TRAIN_PANEL_SHA256, "train panel"),
        (authority / "validation-selection63-reciprocal-6.json", VALIDATION_PANEL_SHA256, "validation panel"),
        (baseline / "FRAMECONTACT_DEVELOPMENT_RESULT.json", BASELINE_RESULT_SHA256, "baseline result"),
        (baseline / "FRAMECONTACT_SELECTED.pt", BASELINE_CHECKPOINT_SHA256, "baseline checkpoint"),
        (baseline / "RUN_RECEIPT.json", BASELINE_RECEIPT_SHA256, "baseline receipt"),
        (candidate_manifest, SOURCE_MANIFEST_SHA256, "candidate source manifest"),
        (args.candidate_source_root / "src/interaction_first/framecontact.py", FRAMECONTACT_SHA256, "candidate FrameContact"),
    )
    for path, expected, label in checks:
        authenticate(path, expected, label)

    parent_result = json.loads((args.parent_root / "EDGE_MEAN_TRAINING_SELECTION_RESULT_V1.json").read_text())
    if parent_result.get("status") != "FAIL_CLEAN":
        raise ValueError("parent edge-mean result is not the frozen clean failure")
    authorization = parent_result.get("authorization", {})
    if any(
        authorization.get(key) is not False
        for key in (
            "selection_checkpoint_confirmation_panel_measurement_only",
            "seeds_29_43_development_only",
            "full_development",
            "old_test",
            "temporal_holdout",
        )
    ):
        raise ValueError("parent result authorization differs")

    if args.output_root.exists() or args.output_root.is_symlink():
        raise FileExistsError(args.output_root)
    output_authority = args.output_root / "authority"
    output_authority.mkdir(parents=True)
    (args.output_root / "logs").mkdir()
    (args.output_root / "runs").mkdir()

    protocol = copy.deepcopy(json.loads((authority / "PROTOCOL_EDGE_MEAN_SUPERVISED_V1.json").read_text()))
    protocol["schema_version"] = "sabdab2_edge_topk2_training_screen_protocol_v1"
    protocol["arm"] = "edge_topk2_supervised"
    protocol["authority"] = {
        "parent_campaign_sha256": PARENT_CAMPAIGN_SHA256,
        "parent_selection_result_sha256": PARENT_SELECTION_RESULT_SHA256,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "framecontact_sha256": FRAMECONTACT_SHA256,
        "baseline_result_sha256": BASELINE_RESULT_SHA256,
        "baseline_checkpoint_sha256": BASELINE_CHECKPOINT_SHA256,
        "baseline_receipt_sha256": BASELINE_RECEIPT_SHA256,
    }
    protocol["tasks"]["winner"]["name"] = "edge_topk2_supervised"
    protocol["epitope_readout"] = {
        "rule": "masked_top_k_mean_2_edge_probability_no_direct_head",
        "upstream_reference": "official EpiFormer decoder sampling_strat=top_k_mean_2",
        "single_factor_only": True,
        "parent_rule": "paper_mean_edge_probability_no_direct_head",
        "joint_contact_paratope_epitope_training_preserved": True,
        "loss_weights_unchanged": True,
        "training_budget_unchanged": True,
    }
    protocol["claim_boundary"].update({
        "selection_panel_only": True,
        "confirmation_panel_loaded": False,
        "test_loaded": False,
        "temporal_holdout_loaded": False,
    })
    protocol_path = output_authority / "PROTOCOL_EDGE_TOPK2_SUPERVISED_V1.json"
    protocol_sha256 = write_new(protocol_path, protocol)
    receipt = {
        "format_version": "sabdab2_edge_topk2_training_protocol_receipt_v1",
        "protocol_file_sha256": protocol_sha256,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "training_split_sha256": SPLIT_SHA256,
        "training_partner_panel_sha256": TRAIN_PANEL_SHA256,
        "validation_partner_panel_sha256": VALIDATION_PANEL_SHA256,
        "frozen_before_training": True,
        "confirmation_panel_loaded": False,
        "test_loaded": False,
        "temporal_holdout_loaded": False,
    }
    receipt_sha256 = write_new(output_authority / "PROTOCOL_EDGE_TOPK2_SUPERVISED_V1_RECEIPT.json", receipt)
    campaign = {
        "format_version": "sabdab2_edge_topk2_training_campaign_v1",
        "status": "FROZEN_BEFORE_TRAINING",
        "arm": "edge_topk2_supervised",
        "single_factor": "masked mean probability to masked top-2 mean probability",
        "protocol_sha256": protocol_sha256,
        "protocol_receipt_sha256": receipt_sha256,
        "parent_campaign_sha256": PARENT_CAMPAIGN_SHA256,
        "parent_selection_result_sha256": PARENT_SELECTION_RESULT_SHA256,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "framecontact_sha256": FRAMECONTACT_SHA256,
        "baseline_result_sha256": BASELINE_RESULT_SHA256,
        "selection_seed": 17,
        "promotion_gate": {
            "epitope_delta_minimum": 0.005,
            "contact_delta_minimum": -0.01,
            "paratope_delta_minimum": -0.01,
            "next_scope": "three_seed_confirmation_panel_only",
        },
        "confirmation_panel_loaded": False,
        "test_loaded": False,
        "temporal_holdout_loaded": False,
    }
    campaign_sha256 = write_new(output_authority / "CAMPAIGN_EDGE_TOPK2_TRAINING_V1.json", campaign)
    print(json.dumps({"status": campaign["status"], "campaign_sha256": campaign_sha256, "protocol_sha256": protocol_sha256, "test_loaded": False}, sort_keys=True))


if __name__ == "__main__":
    main()
