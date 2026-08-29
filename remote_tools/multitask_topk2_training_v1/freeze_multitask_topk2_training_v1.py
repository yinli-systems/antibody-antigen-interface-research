#!/usr/bin/env python3
"""Freeze two single-factor seed-17 TopK2 transfer retraining arms."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import tempfile
from pathlib import Path


PARENT_PROTOCOL_SHA256 = "a00a51a4617a581c57a74f854defd6c18d8258180659775b1c5044be10f98c5c"
SOURCE_MANIFEST_SHA256 = "ddd43e25f29c22f71bb5816b4ac1d2739dad59f345c485b2fc7f51ce9655769e"
FULL_SPLIT_SHA256 = "fd1f516839ec9aabc85531ccfa96754887a13d2f54a3867f68bdc8f55d046ce6"
TRAIN_PANEL_SHA256 = "fd64b540ed0788d2e89d9c487d9cbde3cc56c4cb0225f29013007b2c3884efd6"
VALIDATION_PANEL_SHA256 = "222a22e1c640b679df435e51e00f3550a61a6f880a31efb1807883b520aa30f5"
CONTROL_PROTOCOL_SHA256 = "9375c09dfd7c6bcdd8f0cb9d4cc9f49f70cd34dcecb9cb57652de14eee4542fd"
CONTROL_RESULT_SHA256 = "7e14a264170b7965cb2ea08c8c11b177e3929b54b61a817d602ffeb70c5e295d"
CONTROL_RECEIPT_SHA256 = "6ca896ecbab5604aaddef02c508a58479cd228cf855733a030bb9c331e728b7d"
ARMS = {
    "paratope_topk2": {"paratope_pooling": "topk_mean", "paratope_k": 2, "contact_readout": "pair"},
    "contact_topk2_context": {"paratope_pooling": "logmeanexp", "paratope_k": None, "contact_readout": "topk_context_equal_logit"},
}


def canonical_bytes(payload: object) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha(path: Path, expected: str, label: str) -> None:
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"{label} SHA differs: {observed}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-protocol", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--full-split", type=Path, required=True)
    parser.add_argument("--train-panel", type=Path, required=True)
    parser.add_argument("--validation-panel", type=Path, required=True)
    parser.add_argument("--control-protocol", type=Path, required=True)
    parser.add_argument("--control-result", type=Path, required=True)
    parser.add_argument("--control-receipt", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--analyzer", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    for path, expected, label in (
        (args.parent_protocol, PARENT_PROTOCOL_SHA256, "parent protocol"),
        (args.source_manifest, SOURCE_MANIFEST_SHA256, "source manifest"),
        (args.full_split, FULL_SPLIT_SHA256, "full split"),
        (args.train_panel, TRAIN_PANEL_SHA256, "training panel"),
        (args.validation_panel, VALIDATION_PANEL_SHA256, "validation panel"),
        (args.control_protocol, CONTROL_PROTOCOL_SHA256, "control protocol"),
        (args.control_result, CONTROL_RESULT_SHA256, "control result"),
        (args.control_receipt, CONTROL_RECEIPT_SHA256, "control receipt"),
    ):
        require_sha(path, expected, label)
    if args.output_root.exists() or args.output_root.is_symlink():
        raise FileExistsError(args.output_root)
    parent = json.loads(args.parent_protocol.read_text())
    if parent["protocol_status"] != "frozen_before_full_development_training":
        raise ValueError("parent protocol differs")
    control = json.loads(args.control_result.read_text())
    required_control_keys = {
        "contact:topk2_context_equal_logit-minus-baseline_pair",
        "paratope:node_plus_topk2-minus-baseline_node_plus_logmeanexp",
    }
    if set(control["paired_complex_bootstrap"]) != required_control_keys | {
        "paratope:topk2_edge_only-minus-baseline_node_plus_logmeanexp"
    }:
        raise ValueError("fixed-checkpoint control differs")
    for key in required_control_keys:
        interval = control["paired_complex_bootstrap"][key]["ci95_percentile"]
        if control["paired_complex_bootstrap"][key]["point_estimate"] <= 0 or interval[0] <= 0:
            raise ValueError("strict retraining was not authorized by the frozen control")

    temporary = Path(tempfile.mkdtemp(prefix=".multitask-topk2-training-", dir=args.output_root.parent))
    try:
        authority = temporary / "authority"
        authority.mkdir(parents=True)
        (temporary / "logs").mkdir()
        (temporary / "runs").mkdir()
        arm_records = {}
        for arm, settings in ARMS.items():
            protocol = copy.deepcopy(parent)
            protocol["schema_version"] = "sabdab2_multitask_topk2_training_v1"
            protocol["protocol_status"] = "frozen_before_seed17_training"
            protocol["arm"] = arm
            protocol["model_selection"]["seeds"] = [17]
            protocol["epitope_readout"] = {
                "pooling": "topk_mean",
                "k": 2,
                "fixed_across_arms": True,
            }
            protocol["paratope_readout"] = {
                "pooling": settings["paratope_pooling"],
                "k": settings["paratope_k"],
                "node_logit_retained": True,
            }
            protocol["contact_readout"] = {
                "mode": settings["contact_readout"],
                "pair_index_preserved": True,
                "equal_logit_weights": settings["contact_readout"] == "topk_context_equal_logit",
            }
            protocol["single_factor"] = arm
            protocol["execution_sources"] = {
                "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
                "launcher_sha256": sha256_file(args.launcher),
                "analyzer_sha256": sha256_file(args.analyzer),
                "parent_protocol_sha256": PARENT_PROTOCOL_SHA256,
                "fixed_checkpoint_control_protocol_sha256": CONTROL_PROTOCOL_SHA256,
                "fixed_checkpoint_control_result_sha256": CONTROL_RESULT_SHA256,
            }
            protocol["claim_boundary"] = {
                "development_only": True,
                "full_validation_is_checkpoint_selection_data": True,
                "single_seed_screen": True,
                "test_loaded": False,
                "temporal_holdout_loaded": False,
            }
            protocol_path = authority / f"PROTOCOL_{arm.upper()}_V1.json"
            protocol_path.write_bytes(canonical_bytes(protocol))
            protocol_sha = sha256_file(protocol_path)
            receipt = {
                "format_version": "sabdab2_multitask_topk2_training_protocol_receipt_v1",
                "status": protocol["protocol_status"],
                "arm": arm,
                "protocol_file_sha256": protocol_sha,
                "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
                "split_sha256": FULL_SPLIT_SHA256,
                "training_panel_sha256": TRAIN_PANEL_SHA256,
                "validation_panel_sha256": VALIDATION_PANEL_SHA256,
                "test_loaded": False,
                "temporal_holdout_loaded": False,
            }
            receipt_path = authority / f"PROTOCOL_{arm.upper()}_V1_RECEIPT.json"
            receipt_path.write_bytes(canonical_bytes(receipt))
            arm_records[arm] = {
                **settings,
                "protocol_path": str(args.output_root / "authority" / protocol_path.name),
                "protocol_sha256": protocol_sha,
                "receipt_path": str(args.output_root / "authority" / receipt_path.name),
                "receipt_sha256": sha256_file(receipt_path),
            }
        campaign = {
            "format_version": "sabdab2_multitask_topk2_training_campaign_v1",
            "status": "FROZEN_BEFORE_SEED17_TRAINING",
            "arms": arm_records,
            "reference": {
                "arm": "existing_topk2_seed17",
                "result_sha256": "bbefaf74c4c785009470cd4f51c255dfc04286ada86139a7125e0a8f182caa42",
                "checkpoint_sha256": "6f029a705e05147f4f8275aa879fe34f23df3abb33ce2586d73d80172e8f524d",
            },
            "claim_boundary": {
                "single_seed_development_screen": True,
                "test_loaded": False,
                "temporal_holdout_loaded": False,
                "not_final_model_selection": True,
            },
        }
        campaign_path = authority / "CAMPAIGN_MULTITASK_TOPK2_TRAINING_V1.json"
        campaign_path.write_bytes(canonical_bytes(campaign))
        campaign_sha = sha256_file(campaign_path)
        temporary.replace(args.output_root)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps({"status": campaign["status"], "campaign_sha256": campaign_sha}))


if __name__ == "__main__":
    main()
