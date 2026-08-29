#!/usr/bin/env python3
"""Freeze paired three-seed full-development protocols after a passing screen."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any


BASE_FULL_PROTOCOL_SHA256 = "ce244ab5e677c2819ce4f4e9702568a54e3a094ebd89f1ac144fd73040748780"
SCREEN_RESULT_SHA256 = "284b20b97caa5aa354440b5e86cbd0b8cee98f7bee4bfa67d3c61d42d0183f42"
JOINT_MANIFEST_SHA256 = "d081d1752acf0e94757b248141a1c8861b76fab66a25fcaca89580fbee768323"
FEATURE_ENERGY_AUDIT_SHA256 = "aaa674a411d1373ed4e307086f7613cbb086e2ec1455a47e5538598f470c5fed"
FULL_SPLIT_SHA256 = "fd1f516839ec9aabc85531ccfa96754887a13d2f54a3867f68bdc8f55d046ce6"
JOINT_RESIDUAL_SCALE = 0.6800912466498086


def _canonical(payload: Any) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-full-protocol", type=Path, required=True)
    parser.add_argument("--screen-result", type=Path, required=True)
    parser.add_argument("--joint-manifest", type=Path, required=True)
    parser.add_argument("--feature-energy-audit", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args()
    for path, expected, name in (
        (arguments.base_full_protocol, BASE_FULL_PROTOCOL_SHA256, "base full protocol"),
        (arguments.screen_result, SCREEN_RESULT_SHA256, "screen result"),
        (arguments.joint_manifest, JOINT_MANIFEST_SHA256, "joint manifest"),
        (arguments.feature_energy_audit, FEATURE_ENERGY_AUDIT_SHA256, "energy audit"),
    ):
        if _sha256(path) != expected:
            raise ValueError(f"{name} SHA-256 differs")
    screen = json.loads(arguments.screen_result.read_text(encoding="utf-8"))
    if (
        screen.get("status") != "PASS_AUDIT"
        or screen.get("promotion_status") != "PASS"
        or not all(screen.get("promotion_checks", {}).values())
        or screen.get("authority", {}).get("test_loaded") is not False
        or screen.get("authority", {}).get("full_split_sha256") != FULL_SPLIT_SHA256
    ):
        raise ValueError("Energy-balanced screen did not authorize full development")
    joint = json.loads(arguments.joint_manifest.read_text(encoding="utf-8"))
    if (
        joint.get("status") != "COMPLETE_TRAIN_VALIDATION_ONLY"
        or joint.get("role_counts") != {"train": 2565, "validation": 549}
        or joint.get("test_access") is not False
        or joint.get("authority", {}).get("test_loaded") is not False
        or joint.get("authority", {}).get("task_labels_read") is not False
        or joint.get("authority", {}).get("split_sha256") != FULL_SPLIT_SHA256
    ):
        raise ValueError("Joint embedding authority is not complete development-only data")
    if arguments.output_root.exists() or arguments.output_root.is_symlink():
        raise FileExistsError(arguments.output_root)
    arguments.output_root.mkdir(parents=True)

    base = json.loads(arguments.base_full_protocol.read_text(encoding="utf-8"))
    base["schema_version"] = "sabdab2_paired_full_development_v2"
    base["protocol_status"] = "frozen_before_full_development_training"
    base["model_selection"]["seeds"] = [17, 29, 43]
    base["claim_boundary"]["screen_result_sha256"] = SCREEN_RESULT_SHA256
    base["claim_boundary"]["screen_promotion_status"] = "PASS"
    base["claim_boundary"]["paired_baseline_retrained_under_same_protocol"] = True
    base["claim_boundary"]["test_loaded"] = False
    base["claim_boundary"]["future_temporal_holdout_forbidden_before_full_development_freeze"] = True
    base["full_development_gate"] = {
        "reference_arm": "epiformer_style_same_protocol",
        "candidate_arm": "abpairfm_epiformer_energy_balanced_residual_same_protocol",
        "promotion_requires": (
            "candidate three-seed median contact and epitope mean-per-complex AUPRC "
            "both strictly exceed the paired baseline"
        ),
        "maximum_single_seed_epitope_regression": 0.01,
        "maximum_single_seed_contact_regression": 0.005,
        "test_loaded": False,
    }

    arms = {
        "baseline": {
            "experiment": "epiformer_style_same_protocol",
            "feature_mode": "external_embedding_cache",
            "implementation_boundary": (
                "same-protocol EpiFormer-style architectural adaptation; not the official "
                "EpiFormer implementation"
            ),
        },
        "energy_balanced": {
            "experiment": "abpairfm_epiformer_energy_balanced_residual_same_protocol",
            "feature_mode": "abpairfm_joint_energy_balanced_residual_v1",
            "implementation_boundary": (
                "same-protocol EpiFormer-style adapter with frozen ESM-2 features and "
                "train-only energy-balanced authenticated ABPairFM joint hidden states"
            ),
        },
    }
    summary = {}
    for arm, specification in arms.items():
        protocol = copy.deepcopy(base)
        protocol["arm"] = arm
        protocol["data"]["input_feature_mode"] = specification["feature_mode"]
        protocol["tasks"]["winner"] = {
            "name": specification["experiment"],
            "experiment": specification["experiment"],
            "distance_loss_weight": 0.1,
            "contrastive_loss_weight": 0.0,
            "implementation_boundary": specification["implementation_boundary"],
        }
        if arm == "energy_balanced":
            protocol["implementation"]["joint_residual_scale"] = JOINT_RESIDUAL_SCALE
            protocol["data"]["joint_embedding"] = {
                "manifest_sha256": JOINT_MANIFEST_SHA256,
                "source_split_sha256": FULL_SPLIT_SHA256,
                "checkpoint_content_sha256": joint["checkpoint"]["content_sha256"],
                "joint_residual_scale": JOINT_RESIDUAL_SCALE,
                "feature_energy_audit_sha256": FEATURE_ENERGY_AUDIT_SHA256,
                "task_labels_read": False,
                "test_loaded": False,
            }
        path = arguments.output_root / f"PROTOCOL_{arm.upper()}_FULL_V2.json"
        path.write_bytes(_canonical(protocol))
        receipt = {
            "format_version": "sabdab2_paired_full_development_receipt_v2",
            "arm": arm,
            "protocol_file_sha256": _sha256(path),
            "screen_result_sha256": SCREEN_RESULT_SHA256,
            "frozen_before_training": True,
            "test_loaded": False,
        }
        receipt_path = arguments.output_root / f"PROTOCOL_{arm.upper()}_FULL_V2_RECEIPT.json"
        receipt_path.write_bytes(_canonical(receipt))
        summary[arm] = {
            "protocol_sha256": receipt["protocol_file_sha256"],
            "receipt_sha256": _sha256(receipt_path),
        }
    promotion = {
        "format_version": "abpairfm_energy_balanced_full_development_promotion_v1",
        "screen_result_sha256": SCREEN_RESULT_SHA256,
        "screen_aggregate": screen["aggregate"],
        "screen_promotion_checks": screen["promotion_checks"],
        "full_split_sha256": FULL_SPLIT_SHA256,
        "joint_manifest_sha256": JOINT_MANIFEST_SHA256,
        "feature_energy_audit_sha256": FEATURE_ENERGY_AUDIT_SHA256,
        "joint_residual_scale": JOINT_RESIDUAL_SCALE,
        "arms": summary,
        "test_loaded": False,
    }
    promotion_path = arguments.output_root / "FULL_DEVELOPMENT_PROMOTION_MANIFEST_V1.json"
    promotion_path.write_bytes(_canonical(promotion))
    print(json.dumps({**summary, "promotion_sha256": _sha256(promotion_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
