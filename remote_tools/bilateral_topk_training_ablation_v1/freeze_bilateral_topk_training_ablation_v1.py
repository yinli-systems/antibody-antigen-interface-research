#!/usr/bin/env python3
"""Freeze an eight-arm, seed-17, from-scratch bilateral Top-K ablation."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any


K_VALUES = (1, 2, 4, 8)


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--parent-protocol", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--fixed-readout-analysis", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--analyzer", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists() or args.output_root.is_symlink():
        raise FileExistsError(args.output_root)

    parent = json.loads(args.parent_protocol.read_text(encoding="utf-8"))
    if parent["protocol_status"] != "frozen_before_training_no_privileged_distance_v1":
        raise ValueError("parent protocol differs")
    if parent["training_objective"]["loss_weights"]["distance"] != 0.0:
        raise ValueError("parent uses distance auxiliary")
    if parent["claim_boundary"]["test_loaded"] is not False:
        raise ValueError("parent accessed test")
    if parent["claim_boundary"]["temporal_holdout_loaded"] is not False:
        raise ValueError("parent accessed temporal holdout")
    fixed = json.loads(args.fixed_readout_analysis.read_text(encoding="utf-8"))
    if fixed["status"] != "PASS_BILATERAL_TOPK_FIXED_CHECKPOINT_READOUT_AUDIT":
        raise ValueError("fixed-checkpoint screen did not pass its audit")
    if fixed["claim_boundary"]["test_loaded"] is not False:
        raise ValueError("fixed-checkpoint screen accessed test")

    arms: dict[str, dict[str, Any]] = {}
    for k in K_VALUES:
        arms[f"epitope_k{k}"] = {
            "ablation_axis": "epitope",
            "epitope_pooling": "topk_mean",
            "epitope_k": k,
            "paratope_pooling": "logmeanexp",
            "paratope_k": None,
        }
    for k in K_VALUES:
        arms[f"paratope_k{k}"] = {
            "ablation_axis": "paratope",
            "epitope_pooling": "topk_mean",
            "epitope_k": 2,
            "paratope_pooling": "topk_mean",
            "paratope_k": k,
        }

    args.output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{args.output_root.name}.", dir=args.output_root.parent)
    )
    try:
        authority = temporary / "authority"
        authority.mkdir()
        (temporary / "runs").mkdir()
        (temporary / "logs").mkdir()
        arm_records: dict[str, Any] = {}
        for arm, settings in arms.items():
            protocol = copy.deepcopy(parent)
            protocol["schema_version"] = "sabdab2_bilateral_topk_training_ablation_v1"
            protocol["protocol_status"] = "frozen_before_seed17_bilateral_topk_training"
            protocol["arm"] = arm
            protocol["model_selection"]["seeds"] = [17]
            protocol["epitope_readout"] = {
                "pooling": settings["epitope_pooling"],
                "k": settings["epitope_k"],
            }
            protocol["paratope_readout"] = {
                "pooling": settings["paratope_pooling"],
                "k": settings["paratope_k"],
                "node_logit_retained": True,
            }
            protocol["contact_readout"] = {
                "mode": "pair",
                "pair_index_preserved": True,
            }
            protocol["single_factor_ablation"] = settings["ablation_axis"]
            protocol["implementation"] = {
                **protocol.get("implementation", {}),
                "source_manifest_sha256": sha256_file(args.source_manifest),
                "launcher_sha256": sha256_file(args.launcher),
                "analyzer_sha256": sha256_file(args.analyzer),
                "parent_protocol_sha256": sha256_file(args.parent_protocol),
                "fixed_readout_analysis_sha256": sha256_file(
                    args.fixed_readout_analysis
                ),
            }
            protocol["claim_boundary"] = {
                **protocol["claim_boundary"],
                "single_seed_development_screen": True,
                "from_scratch_training_ablation": True,
                "no_optimal_k_claim": True,
                "not_final_model_selection": True,
                "test_loaded": False,
                "temporal_holdout_loaded": False,
            }
            protocol_path = authority / f"PROTOCOL_{arm.upper()}_V1.json"
            protocol_path.write_bytes(canonical_bytes(protocol))
            protocol_sha = sha256_file(protocol_path)
            receipt = {
                "format_version": "bilateral_topk_training_protocol_receipt_v1",
                "status": protocol["protocol_status"],
                "arm": arm,
                "protocol_file_sha256": protocol_sha,
                "source_manifest_sha256": sha256_file(args.source_manifest),
                "split_sha256": parent["data"]["full_split_sha256"],
                "test_loaded": False,
                "temporal_holdout_loaded": False,
            }
            receipt_path = authority / f"PROTOCOL_{arm.upper()}_V1_RECEIPT.json"
            receipt_path.write_bytes(canonical_bytes(receipt))
            arm_records[arm] = {
                **settings,
                "protocol_sha256": protocol_sha,
                "receipt_sha256": sha256_file(receipt_path),
            }

        campaign = {
            "format_version": "bilateral_topk_training_ablation_campaign_v1",
            "status": "FROZEN_BEFORE_EIGHT_ARM_SEED17_TRAINING",
            "arms": arm_records,
            "seed": 17,
            "k_values": list(K_VALUES),
            "shared_reference_arm": "epitope_k2",
            "execution": {
                "source_root": str(args.source_manifest.parent),
                "source_manifest_sha256": sha256_file(args.source_manifest),
                "trainer_sha256": sha256_file(
                    args.source_manifest.parent / "scripts/train_framecontact.py"
                ),
                "launcher_sha256": sha256_file(args.launcher),
                "analyzer_sha256": sha256_file(args.analyzer),
            },
            "authorities": {
                "parent_protocol_sha256": sha256_file(args.parent_protocol),
                "fixed_readout_analysis_sha256": sha256_file(
                    args.fixed_readout_analysis
                ),
                "structure_manifest_sha256": parent["data"][
                    "structure_manifest_sha256"
                ],
                "contact_manifest_sha256": parent["data"][
                    "contact_manifest_sha256"
                ],
                "embedding_manifest_sha256": parent["data"][
                    "embedding_manifest_sha256"
                ],
                "split_sha256": parent["data"]["full_split_sha256"],
            },
            "claim_boundary": {
                "development_only": True,
                "single_seed_screen": True,
                "from_scratch_training": True,
                "validation_is_checkpoint_selection_data": True,
                "no_optimal_k_claim": True,
                "no_test_or_temporal_access": True,
            },
        }
        campaign_path = authority / "BILATERAL_TOPK_TRAINING_CAMPAIGN_V1.json"
        campaign_path.write_bytes(canonical_bytes(campaign))
        campaign_sha = sha256_file(campaign_path)
        temporary.rename(args.output_root)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(
        json.dumps(
            {"status": campaign["status"], "campaign_sha256": campaign_sha},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
