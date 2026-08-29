#!/usr/bin/env python3
"""Freeze adapted ESM-C development embedding authority before GPU inference."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path


ANALYSIS_SHA256 = "bf13cacbfa5106218945288f9f041816b1e7e41538a1bd79f91583a64e12ceff"
ANALYSIS_RECEIPT_SHA256 = "e260967dbae93fa179974c69621d03a09cf8b483ad6ac605c486c40ba1ce8d16"
PLAN_SHA256 = "4bdd000eea6f4d4c514f995849f084a39b845f8dd94719eb7f667ab4db1ef0b2"
BASE_MANIFEST_SHA256 = "cf2b6734a4614f4c260b6f1f3f861bd732549010324d7053da237b5890f182ab"
MODELS = {
    "whole_chain_adapted": {
        "source_arm": "whole_chain_control",
        "checkpoint_sha256": "50b27675df33f843e7a9a71eca4423c449603246980e3296dfbb21fc27ffa1fb",
        "result_sha256": "6d03871ac58af2df46e13712ae127f082f4787ae98a465bd027f82a8eae91990",
    },
    "cdr_focused_adapted": {
        "source_arm": "cdr_focused",
        "checkpoint_sha256": "5fd67e390f4602f88e9a59d03a11bff332da487ebe4425bdb562bcce8a3a9d9a",
        "result_sha256": "a7d4166454bec5492c57ba3c37879f3169a1bf61ad96f4be6607c3b85205472e",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path: Path, value: object) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def require_sha(path: Path, expected: str, label: str) -> None:
    if sha256_file(path) != expected:
        raise RuntimeError(f"{label} SHA differs")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--analysis-receipt", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--materializer", type=Path, required=True)
    parser.add_argument("--canary-launcher", type=Path, required=True)
    parser.add_argument("--canary-analyzer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    for path, expected, label in (
        (args.analysis, ANALYSIS_SHA256, "adaptation analysis"),
        (args.analysis_receipt, ANALYSIS_RECEIPT_SHA256, "adaptation analysis receipt"),
        (args.plan, PLAN_SHA256, "development sequence plan"),
        (args.base_manifest, BASE_MANIFEST_SHA256, "base development embedding manifest"),
    ):
        require_sha(path, expected, label)
    analysis = json.loads(args.analysis.read_text())
    receipt = json.loads(args.analysis_receipt.read_text())
    plan = json.loads(args.plan.read_text())
    base_manifest = json.loads(args.base_manifest.read_text())
    if (
        analysis.get("status") != "PASS_FIXED_EPOCH4_CONTROLLED_ADAPTATION"
        or analysis.get("authorization", {}).get("development_embedding_materialization_authorized") is not True
        or analysis.get("authorization", {}).get("downstream_test_authorized") is not False
        or analysis.get("claim_boundary", {}).get("sabdab2_test_access") is not False
        or receipt.get("analysis_sha256") != ANALYSIS_SHA256
        or receipt.get("development_embedding_materialization_authorized") is not True
        or receipt.get("sabdab2_test_access") is not False
        or plan.get("status") != "FROZEN_TRAIN_VALIDATION_ONLY_BEFORE_EMBEDDING"
        or plan.get("authorization", {}).get("test_access") is not False
        or base_manifest.get("complete") is not True
        or base_manifest.get("model_name") != "biohub/ESMC-600M"
        or base_manifest.get("authority", {}).get("test_access") is not False
        or base_manifest.get("authority", {}).get("task_labels_read") is not False
    ):
        raise RuntimeError("adapted development embedding prerequisites differ")
    model_bindings = {}
    for role, expected in MODELS.items():
        run = analysis["per_arm"][expected["source_arm"]]
        checkpoint = Path(run["fixed_checkpoint"]["path"])
        result = checkpoint.parent / "result.json"
        require_sha(checkpoint, expected["checkpoint_sha256"], f"{role} checkpoint")
        require_sha(result, expected["result_sha256"], f"{role} result")
        if (
            run.get("result_sha256") != expected["result_sha256"]
            or run.get("fixed_checkpoint", {}).get("sha256") != expected["checkpoint_sha256"]
            or not all(run.get("checks", {}).values())
        ):
            raise RuntimeError(f"{role} analysis binding differs")
        model_bindings[role] = {
            "source_arm": expected["source_arm"],
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": expected["checkpoint_sha256"],
            "adaptation_result": str(result),
            "adaptation_result_sha256": expected["result_sha256"],
            "fixed_epoch": 4,
        }
    protocol = {
        "format_version": "sabdab2_adapted_esmc_embedding_development_protocol_v1",
        "status": "FROZEN_AFTER_ADAPTATION_BEFORE_EMBEDDING",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "adaptation_analysis_sha256": ANALYSIS_SHA256,
        "adaptation_analysis_receipt_sha256": ANALYSIS_RECEIPT_SHA256,
        "development_sequence_plan": str(args.plan),
        "development_sequence_plan_sha256": PLAN_SHA256,
        "adapted_antibody_models": model_bindings,
        "antigen_representation": {
            "source": "unchanged biohub/ESMC-600M development cache",
            "manifest": str(args.base_manifest),
            "manifest_sha256": BASE_MANIFEST_SHA256,
            "model_weight_sha256": "e4232c30fd35fe2f57051ec88a703996ac94520580b4b836894207a3d45d9ff8",
            "may_be_reembedded": False,
        },
        "embedding_contract": {
            "dimension": 1152,
            "paired_antibody_input": "VH + separator-token-30 + VL",
            "separator_removed_from_exported_residue_axis": True,
            "antibody_units_only_materialized_from_adapted_checkpoints": True,
            "base_antigen_artifacts_reused_by_verified_hardlink": True,
            "shard_count_per_arm": 4,
        },
        "source_binding": {
            "materializer_sha256": sha256_file(args.materializer),
            "canary_launcher_sha256": sha256_file(args.canary_launcher),
            "canary_analyzer_sha256": sha256_file(args.canary_analyzer),
        },
        "authorization": {
            "canary_embedding_authorized": True,
            "full_embedding_authorized_before_canary_pass": False,
            "downstream_development_training_authorized_before_full_manifest": False,
            "test_access": False,
        },
        "claim_boundary": {
            "development_only": True,
            "fixed_epoch4_adaptation_not_selected_by_downstream_metrics": True,
            "performance_gain_claim_authorized": False,
            "old_exposed_test_forbidden": True,
            "temporal_holdout_forbidden": True,
        },
        "test_access": False,
    }
    atomic_write(args.output, protocol)
    output_receipt = {
        "format_version": "sabdab2_adapted_esmc_embedding_development_protocol_receipt_v1",
        "status": protocol["status"],
        "protocol_sha256": sha256_file(args.output),
        "canary_embedding_authorized": True,
        "full_embedding_authorized": False,
        "downstream_development_training_authorized": False,
        "test_access": False,
    }
    atomic_write(args.receipt, output_receipt)
    print(json.dumps({**output_receipt, "receipt_sha256": sha256_file(args.receipt)}, sort_keys=True))


if __name__ == "__main__":
    main()
