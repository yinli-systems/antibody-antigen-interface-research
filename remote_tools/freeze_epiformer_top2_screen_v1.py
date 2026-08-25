#!/usr/bin/env python3
"""Freeze a single-factor EpiFormer top-2 development screen."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


PRIOR_ANALYSIS_SHA256 = "c24352ea9766cd564dab09cbc444084c4b9c091576d7ed5f5600442760884a3d"
BASE_PROTOCOL_SHA256 = "17d85b0e31d59b9fc69c3960750dfcc50d2f28cab2a189db1f993b1203648374"
SOURCE_MANIFEST_SHA256 = "757373282bec0340f3f0171b7471a881fbd4f854a088bc81b72315e9bba24157"
SOURCE_TREE_SHA256 = "c059d79117ed22433f02b24363a54c23cce531aeefad4663feeef357da07538f"
BASELINE_RESULT_SHA256 = "38bbf1a4ade66b01abd81a925ac89464725c67564bc2c4c352701d6d8a0f8184"
PRIOR_CANDIDATE_RESULT_SHA256 = "491c27569027c2761e79a7a91059d8df791f441b7f50357bb8180db2e819360d"


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


def write_new(path: Path, payload: object) -> str:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.write_bytes(canonical_bytes(payload))
    return sha256_file(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-analysis", type=Path, required=True)
    parser.add_argument("--base-protocol", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    authenticate(args.prior_analysis, PRIOR_ANALYSIS_SHA256, "prior analysis")
    authenticate(args.base_protocol, BASE_PROTOCOL_SHA256, "base protocol")
    authenticate(args.source_manifest, SOURCE_MANIFEST_SHA256, "source manifest")

    prior_analysis = json.loads(args.prior_analysis.read_text())
    if (
        prior_analysis.get("status") != "PASS_AUDIT_NO_PROMOTION"
        or prior_analysis.get("winner") is not None
        or prior_analysis.get("authority", {}).get("test_loaded") is not False
        or prior_analysis.get("authority", {}).get("temporal_holdout_loaded")
        is not False
    ):
        raise ValueError("prior loss-balance screen authority differs")
    source_manifest = json.loads(args.source_manifest.read_text())
    if (
        source_manifest.get("tree_sha256") != SOURCE_TREE_SHA256
        or source_manifest.get("record_count") != 160
        or not str(source_manifest.get("source_root", "")).endswith(
            "/epiformer_top2_training_source_v1"
        )
    ):
        raise ValueError("top-2 source manifest differs")

    protocol = copy.deepcopy(json.loads(args.base_protocol.read_text()))
    if (
        protocol.get("protocol_status") != "frozen_before_training"
        or protocol.get("arm") != "dropout020_count020"
        or protocol.get("seed") != 17
        or protocol["tasks"]["winner"]["experiment"]
        != "epiformer_style_same_protocol"
        or protocol["joint_objective"]
        != {
            "contact_loss_weight": 1.0,
            "distance_loss_weight": 0.1,
            "epitope_count_weight": 0.2,
            "epitope_dice_weight": 1.0,
            "epitope_loss_weight": 0.5,
            "paratope_loss_weight": 0.1,
        }
        or protocol["implementation"]["dropout"] != 0.2
    ):
        raise ValueError("base protocol differs")

    if args.output_root.exists() or args.output_root.is_symlink():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    authority_root = args.output_root / "authority"
    authority_root.mkdir(parents=True)

    protocol["format_version"] = "sabdab2_epiformer_top2_screen_protocol_v1"
    protocol["arm"] = "epiformer_top2"
    protocol["authority"] = {
        "prior_loss_balance_analysis_sha256": PRIOR_ANALYSIS_SHA256,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "source_tree_sha256": SOURCE_TREE_SHA256,
        "base_protocol_sha256": BASE_PROTOCOL_SHA256,
        "official_epiformer_decoder": "https://github.com/mansoor181/epiformer/blob/main/model/decoder.py",
    }
    protocol["implementation"]["contact_marginalization"] = {
        "input": "sigmoid_contact_probability",
        "pooling": "masked_top_k_mean",
        "k": 2,
        "output": "stable_logit_added_to_direct_residue_head",
        "changed_for_this_screen": True,
    }
    protocol["screen_design"] = {
        "single_changed_factor": "contact_to_residue_marginalization",
        "unchanged": [
            "dataset",
            "candidate_grid",
            "split",
            "embeddings",
            "architecture_except_marginalization",
            "joint_objective",
            "optimizer",
            "seed",
            "early_stopping",
        ],
        "frozen_control_result_sha256": BASELINE_RESULT_SHA256,
        "prior_epitope_candidate_result_sha256": PRIOR_CANDIDATE_RESULT_SHA256,
        "promotion_gate": {
            "minimum_epitope_delta_vs_control": 0.0075,
            "maximum_epitope_regression_vs_prior_candidate": 0.005,
            "maximum_contact_regression_vs_control": 0.005,
            "maximum_paratope_regression_vs_control": 0.01,
            "next_step_if_pass": "run_only_this_arm_on_seeds_29_and_43",
        },
    }
    protocol["claim_boundary"] = {
        "development_only": True,
        "test_loaded": False,
        "temporal_holdout_loaded": False,
        "single_seed_screen_cannot_authorize_full_development_or_test": True,
    }

    protocol_path = authority_root / "PROTOCOL_EPIFORMER_TOP2_V1.json"
    protocol_sha = write_new(protocol_path, protocol)
    receipt = {
        "format_version": "sabdab2_epiformer_top2_protocol_receipt_v1",
        "arm": "epiformer_top2",
        "seed": 17,
        "frozen_before_training": True,
        "protocol_file_sha256": protocol_sha,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "source_tree_sha256": SOURCE_TREE_SHA256,
        "test_loaded": False,
        "temporal_holdout_loaded": False,
    }
    receipt_path = authority_root / "PROTOCOL_EPIFORMER_TOP2_V1_RECEIPT.json"
    receipt_sha = write_new(receipt_path, receipt)
    campaign = {
        "format_version": "sabdab2_epiformer_top2_campaign_authority_v1",
        "protocol_status": "frozen_before_training",
        "protocol_sha256": protocol_sha,
        "protocol_receipt_sha256": receipt_sha,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "source_tree_sha256": SOURCE_TREE_SHA256,
        "prior_loss_balance_analysis_sha256": PRIOR_ANALYSIS_SHA256,
        "test_loaded": False,
        "temporal_holdout_loaded": False,
    }
    campaign_path = authority_root / "CAMPAIGN_AUTHORITY_V1.json"
    campaign_sha = write_new(campaign_path, campaign)
    (args.output_root / "logs").mkdir()
    print(
        json.dumps(
            {
                "status": "PASS_FROZEN_BEFORE_TRAINING",
                "protocol_sha256": protocol_sha,
                "protocol_receipt_sha256": receipt_sha,
                "campaign_authority_sha256": campaign_sha,
                "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
