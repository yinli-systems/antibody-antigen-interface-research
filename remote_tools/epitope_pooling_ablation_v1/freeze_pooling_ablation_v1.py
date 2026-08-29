#!/usr/bin/env python3
"""Freeze a seed-17 full-development epitope-pooling ablation."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import tempfile
from pathlib import Path


PARENT_PROTOCOL_SHA256 = "a00a51a4617a581c57a74f854defd6c18d8258180659775b1c5044be10f98c5c"
SOURCE_MANIFEST_SHA256 = "10536e3a71d982d2cc6d8b82eefb6c7d51059fab0bf7f486fdbd266ada420348"
FULL_SPLIT_SHA256 = "fd1f516839ec9aabc85531ccfa96754887a13d2f54a3867f68bdc8f55d046ce6"
TRAIN_PANEL_SHA256 = "fd64b540ed0788d2e89d9c487d9cbde3cc56c4cb0225f29013007b2c3884efd6"
VALIDATION_PANEL_SHA256 = "222a22e1c640b679df435e51e00f3550a61a6f880a31efb1807883b520aa30f5"
ARMS = {
    "topk1_max": ("topk_mean", 1),
    "topk2": ("topk_mean", 2),
    "topk4": ("topk_mean", 4),
    "topk8": ("topk_mean", 8),
    "noisy_or": ("noisy_or", None),
    "logmeanexp": ("logmeanexp", None),
    "learned_softmax": ("learned_softmax", None),
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


def require_sha(path: Path, expected: str, label: str) -> None:
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"{label} SHA-256 differs: {observed}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-protocol", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--full-split", type=Path, required=True)
    parser.add_argument("--train-panel", type=Path, required=True)
    parser.add_argument("--validation-panel", type=Path, required=True)
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
    ):
        require_sha(path, expected, label)
    if args.output_root.exists() or args.output_root.is_symlink():
        raise FileExistsError(args.output_root)

    parent = json.loads(args.parent_protocol.read_text(encoding="utf-8"))
    if (
        parent.get("protocol_status") != "frozen_before_full_development_training"
        or parent.get("claim_boundary", {}).get("test_loaded") is not False
        or parent.get("claim_boundary", {}).get("temporal_holdout_loaded") is not False
    ):
        raise ValueError("parent full-development protocol differs")

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{args.output_root.name}.", dir=args.output_root.parent)
    )
    try:
        authority = temporary / "authority"
        authority.mkdir(parents=True)
        (temporary / "logs").mkdir()
        (temporary / "runs").mkdir()
        protocol_records = {}
        for arm, (pooling, k) in ARMS.items():
            protocol = copy.deepcopy(parent)
            protocol["schema_version"] = "sabdab2_epitope_pooling_ablation_v1"
            protocol["protocol_status"] = "frozen_before_seed17_training"
            protocol["arm"] = arm
            protocol["model_selection"]["seeds"] = [17]
            protocol["epitope_readout"] = {
                "pooling": pooling,
                "k": k,
                "probability_domain_topk": pooling == "topk_mean",
                "topk1_is_exact_max_alias": arm == "topk1_max",
                "single_factor_only": True,
            }
            protocol["claim_boundary"] = {
                "development_only": True,
                "full_validation_is_checkpoint_selection_data": True,
                "single_seed_screen": True,
                "test_loaded": False,
                "temporal_holdout_loaded": False,
            }
            protocol["execution_sources"] = {
                "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
                "launcher_sha256": sha256_file(args.launcher),
                "analyzer_sha256": sha256_file(args.analyzer),
                "parent_protocol_sha256": PARENT_PROTOCOL_SHA256,
            }
            protocol_path = authority / f"PROTOCOL_{arm.upper()}_V1.json"
            protocol_path.write_bytes(canonical_bytes(protocol))
            protocol_sha = sha256_file(protocol_path)
            receipt = {
                "format_version": "sabdab2_epitope_pooling_protocol_receipt_v1",
                "status": protocol["protocol_status"],
                "arm": arm,
                "pooling": pooling,
                "k": k,
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
            protocol_records[arm] = {
                "pooling": pooling,
                "k": k,
                "protocol_path": str(args.output_root / "authority" / protocol_path.name),
                "protocol_sha256": protocol_sha,
                "receipt_path": str(args.output_root / "authority" / receipt_path.name),
                "receipt_sha256": sha256_file(receipt_path),
            }
        campaign = {
            "format_version": "sabdab2_epitope_pooling_ablation_campaign_v1",
            "status": "FROZEN_BEFORE_SEED17_TRAINING",
            "question": "fixed-protocol epitope pooling sensitivity",
            "arms": protocol_records,
            "equivalences": {"topk1_max": "masked maximum over observed candidates"},
            "selection_seed": 17,
            "claim_boundary": {
                "single_seed_development_screen": True,
                "not_a_test_result": True,
                "no_optimal_k_claim_from_this_screen_alone": True,
                "test_loaded": False,
                "temporal_holdout_loaded": False,
            },
        }
        campaign_path = authority / "CAMPAIGN_EPITOPE_POOLING_ABLATION_V1.json"
        campaign_path.write_bytes(canonical_bytes(campaign))
        campaign_sha = sha256_file(campaign_path)
        temporary.replace(args.output_root)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(
        json.dumps(
            {
                "status": campaign["status"],
                "campaign_sha256": campaign_sha,
                "arm_count": len(ARMS),
                "test_loaded": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
