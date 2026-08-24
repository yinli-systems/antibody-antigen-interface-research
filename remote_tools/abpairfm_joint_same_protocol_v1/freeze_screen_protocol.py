#!/usr/bin/env python3
"""Freeze the three-seed ABPairFM joint-feature SAbDab2 development screen."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


BASE_PROTOCOL_SHA256 = "c8e262f6a6fdadc4807687904fc8ca3e41ad962a12e1e6caab810567ba795364"
FULL_SPLIT_SHA256 = "fd1f516839ec9aabc85531ccfa96754887a13d2f54a3867f68bdc8f55d046ce6"
SCREEN_SPLIT_SHA256 = "4ff0965a0f8dde3913a176ac0411b40c9d2db3d1cb2684c393d6b01886b806cd"
REFERENCE_EPITOPE_AUPRC = 0.2532489185677912
REFERENCE_CONTACT_AUPRC = 0.024830761686624868


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
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
    parser.add_argument("--base-protocol", type=Path, required=True)
    parser.add_argument("--joint-manifest", type=Path, required=True)
    parser.add_argument("--expected-joint-manifest-sha256", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    arguments = parser.parse_args()

    if arguments.protocol.exists() or arguments.receipt.exists():
        raise FileExistsError("Protocol and receipt must be create-only")
    if _sha256(arguments.base_protocol) != BASE_PROTOCOL_SHA256:
        raise ValueError("Base same-protocol authority differs")
    if _sha256(arguments.joint_manifest) != arguments.expected_joint_manifest_sha256:
        raise ValueError("Joint embedding manifest SHA-256 differs")

    joint = json.loads(arguments.joint_manifest.read_text(encoding="utf-8"))
    if (
        joint.get("format_version")
        != "sabdab2_abpairfm_joint_embedding_manifest_v1"
        or joint.get("status") != "COMPLETE_TRAIN_VALIDATION_ONLY"
        or joint.get("role_counts") != {"train": 2565, "validation": 549}
        or joint.get("test_access") is not False
        or joint.get("authority", {}).get("task_labels_read") is not False
        or joint.get("authority", {}).get("test_loaded") is not False
        or joint.get("authority", {}).get("split_sha256") != FULL_SPLIT_SHA256
    ):
        raise ValueError("Joint embedding authority is not development-only and complete")

    protocol = json.loads(arguments.base_protocol.read_text(encoding="utf-8"))
    protocol["schema_version"] = "abpairfm_joint_same_protocol_screen_v1"
    protocol["protocol_status"] = "frozen_before_training"
    protocol["data"]["input_feature_mode"] = "abpairfm_joint_cross_chain_v1"
    protocol["data"]["joint_embedding"] = {
        "format_version": joint["format_version"],
        "manifest_sha256": arguments.expected_joint_manifest_sha256,
        "source_split_sha256": FULL_SPLIT_SHA256,
        "checkpoint_content_sha256": joint["checkpoint"]["content_sha256"],
        "task_labels_read": False,
        "test_loaded": False,
    }
    protocol["model_selection"]["seeds"] = [17, 29, 43]
    protocol["tasks"] = {
        "winner": {
            "name": "abpairfm_epiformer_style_same_protocol",
            "experiment": "abpairfm_epiformer_style_same_protocol",
            "distance_loss_weight": 0.1,
            "contrastive_loss_weight": 0.0,
            "implementation_boundary": (
                "Frozen EpiFormer-style same-protocol adapter with authenticated joint "
                "ABPairFM cross-chain hidden states; not the official EpiFormer implementation."
            ),
        }
    }
    protocol["screen_gate"] = {
        "reference": "epiformer_style_same_protocol_seed_17",
        "reference_epitope_mean_per_complex_auprc": REFERENCE_EPITOPE_AUPRC,
        "reference_contact_mean_per_complex_auprc": REFERENCE_CONTACT_AUPRC,
        "promotion_requires": (
            "three-seed median epitope and contact mean-per-complex AUPRC both strictly "
            "exceed the frozen reference"
        ),
        "maximum_single_seed_epitope_regression": 0.01,
        "maximum_single_seed_contact_regression": 0.005,
    }
    protocol["claim_boundary"] = {
        "development_only": True,
        "test_loaded": False,
        "screen_is_not_a_final_baseline": True,
        "official_baseline_replication": False,
        "old_sabdab2_test_permanently_closed": True,
        "future_temporal_holdout_forbidden_before_full_development_freeze": True,
    }

    protocol_raw = _canonical(protocol)
    arguments.protocol.parent.mkdir(parents=True, exist_ok=True)
    arguments.protocol.write_bytes(protocol_raw)
    receipt = {
        "format_version": "sabdab2_same_protocol_development_receipt_v1",
        "protocol_file_sha256": hashlib.sha256(protocol_raw).hexdigest(),
        "protocol_canonical_sha256": hashlib.sha256(_canonical(protocol)).hexdigest(),
        "joint_embedding_manifest_sha256": arguments.expected_joint_manifest_sha256,
        "screen_split_sha256": SCREEN_SPLIT_SHA256,
        "test_loaded": False,
        "frozen_before_training": True,
    }
    arguments.receipt.write_bytes(_canonical(receipt))
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
