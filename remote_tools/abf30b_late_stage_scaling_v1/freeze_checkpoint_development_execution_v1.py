#!/usr/bin/env python3
"""Freeze one checkpoint's six-run development execution for late-stage scaling."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any

import torch


ORIGINAL_ESM_MODULE_SHA256 = "2adc1d711f7d5a9d0fcfb34d4813468393d18ea45b0751fffa9d569857570dd8"
COMPATIBILITY_ESM_MODULE_SHA256 = "6124719960e135eb19f89f7841b92ceb55ceca81dafb59578530d2eca853808e"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("ascii")


def publish(path: Path, value: Any) -> str:
    payload = canonical_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
    return hashlib.sha256(payload).hexdigest()


def require_sha(path: Path, expected: str) -> None:
    if path.is_symlink() or not path.is_file() or sha256_file(path) != expected:
        raise ValueError(f"authority SHA differs: {path}")


def load_module(path: Path, name: str, expected_sha: str) -> Any:
    require_sha(path, expected_sha)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify_source(root: Path, manifest: dict[str, Any]) -> None:
    if Path(manifest["source_root"]).resolve() != root.resolve():
        raise ValueError("source root differs")
    for row in manifest["records"]:
        path = root / row["path"]
        require_sha(path, row["sha256"])
        if path.stat().st_size != int(row["size_bytes"]):
            raise ValueError(f"source size differs: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scaling-protocol", type=Path, required=True)
    parser.add_argument("--scaling-protocol-sha256", required=True)
    parser.add_argument("--checkpoint-label", required=True)
    parser.add_argument("--canary", type=Path, required=True)
    parser.add_argument("--canary-receipt", type=Path, required=True)
    parser.add_argument("--embedding-root", type=Path, required=True)
    parser.add_argument("--reference-execution-root", type=Path, required=True)
    parser.add_argument("--baseline-source-root", type=Path, required=True)
    parser.add_argument("--topk2-source-root", type=Path, required=True)
    parser.add_argument("--original-source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    require_sha(args.scaling_protocol, args.scaling_protocol_sha256)
    scaling = json.loads(args.scaling_protocol.read_text())
    checkpoint = scaling.get("checkpoints", {}).get(args.checkpoint_label, {})
    if (
        scaling.get("status") != "FROZEN_BEFORE_LATE_STAGE_SCALING_EXECUTION"
        or args.checkpoint_label not in scaling.get("execution", {}).get(
            "new_embedding_checkpoints", []
        )
        or not checkpoint.get("checkpoint_model_sha256")
    ):
        raise ValueError("scaling protocol or checkpoint label differs")
    checkpoint_sha = checkpoint["checkpoint_model_sha256"]

    canary_receipt = json.loads(args.canary_receipt.read_text())
    require_sha(args.canary, canary_receipt["result_sha256"])
    canary = json.loads(args.canary.read_text())
    embedding_manifest_path = args.embedding_root / "SABDAB2_ESM_EMBEDDING_MANIFEST.json"
    embedding_receipt_path = (
        args.embedding_root / "SABDAB2_ABF_CHECKPOINT_MIXED_EMBEDDING_RECEIPT.json"
    )
    embedding_receipt = json.loads(embedding_receipt_path.read_text())
    require_sha(embedding_manifest_path, embedding_receipt["manifest_sha256"])
    embeddings = json.loads(embedding_manifest_path.read_text())
    if (
        canary.get("status") != "PASS_DCP_EXACT_RESTORE_AND_RESIDUE_EMBEDDING_CANARY"
        or canary.get("checkpoint_label") != args.checkpoint_label
        or canary.get("checkpoint_model_sha256") != checkpoint_sha
        or canary.get("task_labels_read") is not False
        or embedding_receipt.get("checkpoint_label") != args.checkpoint_label
        or embedding_receipt.get("checkpoint_model_sha256") != checkpoint_sha
        or embedding_receipt.get("test_loaded") is not False
        or embeddings.get("antibody_model", {}).get("checkpoint_label")
        != args.checkpoint_label
        or embeddings.get("antibody_model", {}).get("checkpoint_model_sha256")
        != checkpoint_sha
        or embeddings.get("task_labels_read") is not False
    ):
        raise ValueError("canary or embedding authority differs")

    reference_campaign_path = (
        args.reference_execution_root / "ABF30B_DEVELOPMENT_EXECUTION_V1.json"
    )
    reference_receipt_path = (
        args.reference_execution_root / "ABF30B_DEVELOPMENT_EXECUTION_V1_RECEIPT.json"
    )
    reference_campaign = json.loads(reference_campaign_path.read_text())
    reference_receipt = json.loads(reference_receipt_path.read_text())
    if (
        reference_campaign.get("status") != "FROZEN_BEFORE_SIX_RUN_DEVELOPMENT_TRAINING"
        or reference_receipt.get("campaign_sha256") != sha256_file(reference_campaign_path)
        or reference_campaign.get("distance_loss_weight") != 0.0
        or reference_campaign.get("distance_label_cache_supplied") is not False
        or reference_campaign.get("test_loaded") is not False
        or reference_campaign.get("temporal_holdout_loaded") is not False
    ):
        raise ValueError("reference development execution differs")

    source_roots = {
        "baseline": args.baseline_source_root.resolve(),
        "edge_topk2_supervised": args.topk2_source_root.resolve(),
    }
    source_manifests: dict[str, dict[str, Any]] = {}
    for arm in source_roots:
        path = (
            args.reference_execution_root
            / f"SOURCE_MANIFEST_{arm.upper()}_ABF30B_V1.json"
        )
        manifest = json.loads(path.read_text())
        verify_source(source_roots[arm], manifest)
        if sha256_file(path) != reference_campaign["arms"][arm]["source_manifest_sha256"]:
            raise ValueError("reference source manifest SHA differs")
        source_manifests[arm] = manifest

    original = load_module(
        args.original_source_root / "src/interaction_first/esm_embeddings.py",
        "scaling_original_esm",
        ORIGINAL_ESM_MODULE_SHA256,
    )
    baseline_compat = load_module(
        args.baseline_source_root / "src/interaction_first/esm_embeddings.py",
        "scaling_baseline_compat",
        COMPATIBILITY_ESM_MODULE_SHA256,
    )
    load_module(
        args.topk2_source_root / "src/interaction_first/esm_embeddings.py",
        "scaling_topk2_compat",
        COMPATIBILITY_ESM_MODULE_SHA256,
    )
    antigen_record = next(row for row in embeddings["records"] if row["partner"] == "antigen")
    antibody_record = next(row for row in embeddings["records"] if row["partner"] == "antibody")
    antigen_path = args.embedding_root / antigen_record["artifact_path"]
    antibody_path = args.embedding_root / antibody_record["artifact_path"]
    antigen_original = original.load_embedding(
        antigen_path,
        expected_key=antigen_record["sequence_sha256"],
        expected_length=int(antigen_record["sequence_length"]),
    )
    antigen_compat = baseline_compat.load_embedding(
        antigen_path,
        expected_key=antigen_record["sequence_sha256"],
        expected_length=int(antigen_record["sequence_length"]),
    )
    if not torch.equal(antigen_original, antigen_compat):
        raise RuntimeError("compatibility loader changes original ESM-2 output")
    antibody_embedding = baseline_compat.load_embedding(
        antibody_path,
        expected_key=antibody_record["sequence_sha256"],
        expected_length=int(antibody_record["sequence_length"]),
    )
    if tuple(antibody_embedding.shape) != (
        int(antibody_record["sequence_length"]),
        1280,
    ):
        raise RuntimeError("compatibility loader ABF output shape differs")

    output = args.output_root.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    embedding_sha = embedding_receipt["manifest_sha256"]
    arms: dict[str, dict[str, str]] = {}
    for arm in ("baseline", "edge_topk2_supervised"):
        source_manifest_path = output / f"SOURCE_MANIFEST_{arm.upper()}_SCALING_V1.json"
        source_manifest_sha = publish(source_manifest_path, source_manifests[arm])
        reference_protocol_path = (
            args.reference_execution_root / f"PROTOCOL_{arm.upper()}_ABF30B_V1.json"
        )
        reference_protocol = json.loads(reference_protocol_path.read_text())
        value = reference_protocol
        value["schema_version"] = f"abf_checkpoint_scaling_{arm}_development_v1"
        value["protocol_status"] = "frozen_before_abf_checkpoint_scaling_training_v1"
        value["data"]["embedding_manifest_sha256"] = embedding_sha
        value["data"]["mixed_partner_embedding"] = {
            "antibody": (
                f"ABF-650M OAS checkpoint {args.checkpoint_label} residue hidden states"
            ),
            "antigen": "unchanged pinned ESM-2 t33 650M residue embeddings",
            "single_factor_relative_to_final": "pretraining token checkpoint",
            "manifest_sha256": embedding_sha,
        }
        value["claim_boundary"].update(
            {
                "claim": "Development-only late-stage checkpoint sensitivity.",
                "late_stage_scaling_only": True,
                "full_scaling_curve": False,
                "selection_or_tuning_from_external_measurement": False,
                "old_sabdab2_test_loaded": False,
                "temporal_holdout_loaded": False,
            }
        )
        value["abf_checkpoint_scaling_v1"] = {
            "scaling_protocol_sha256": args.scaling_protocol_sha256,
            "checkpoint_label": args.checkpoint_label,
            "checkpoint_model_sha256": checkpoint_sha,
            "consumed_residue_tokens": checkpoint["consumed_residue_tokens"],
            "dcp_canary_sha256": canary_receipt["result_sha256"],
            "embedding_manifest_sha256": embedding_sha,
            "source_manifest_sha256": source_manifest_sha,
        }
        protocol_output = output / f"PROTOCOL_{arm.upper()}_SCALING_V1.json"
        protocol_sha = publish(protocol_output, value)
        protocol_receipt_output = output / f"PROTOCOL_{arm.upper()}_SCALING_V1_RECEIPT.json"
        protocol_receipt_sha = publish(
            protocol_receipt_output,
            {
                "format_version": "abf_checkpoint_scaling_development_protocol_receipt_v1",
                "arm": arm,
                "checkpoint_label": args.checkpoint_label,
                "checkpoint_model_sha256": checkpoint_sha,
                "protocol_file_sha256": protocol_sha,
                "source_manifest_sha256": source_manifest_sha,
                "embedding_manifest_sha256": embedding_sha,
                "distance_loss_weight": 0.0,
                "frozen_before_training": True,
                "test_loaded": False,
            },
        )
        arms[arm] = {
            "protocol_sha256": protocol_sha,
            "protocol_receipt_sha256": protocol_receipt_sha,
            "source_manifest_sha256": source_manifest_sha,
            "source_root": str(source_roots[arm]),
        }

    compatibility = {
        "format_version": "abf_checkpoint_scaling_embedding_compatibility_canary_v1",
        "status": "PASS_AUTHORITY_AWARE_LOADER_COMPATIBILITY",
        "checkpoint_label": args.checkpoint_label,
        "checkpoint_model_sha256": checkpoint_sha,
        "original_esm_module_sha256": ORIGINAL_ESM_MODULE_SHA256,
        "compatibility_module_sha256": COMPATIBILITY_ESM_MODULE_SHA256,
        "original_esm_output_bit_equal": True,
        "abf_artifact_shape_valid": True,
        "task_labels_read": False,
        "test_loaded": False,
    }
    compatibility_sha = publish(output / "EMBEDDING_COMPATIBILITY_CANARY_V1.json", compatibility)
    campaign = {
        "format_version": "abf_checkpoint_scaling_development_execution_v1",
        "status": "FROZEN_BEFORE_SIX_RUN_LATE_STAGE_SCALING_DEVELOPMENT",
        "scaling_protocol_sha256": args.scaling_protocol_sha256,
        "checkpoint_label": args.checkpoint_label,
        "checkpoint_model_sha256": checkpoint_sha,
        "consumed_residue_tokens": checkpoint["consumed_residue_tokens"],
        "embedding_manifest_sha256": embedding_sha,
        "compatibility_canary_sha256": compatibility_sha,
        "arms": arms,
        "seeds": [17, 29, 43],
        "task_routing": {
            "contact": "baseline same seed",
            "paratope": "baseline same seed",
            "epitope": "edge_topk2_supervised same seed",
        },
        "single_factor": "pretraining token checkpoint",
        "antigen_embedding_changed": False,
        "distance_loss_weight": 0.0,
        "distance_label_cache_supplied": False,
        "test_loaded": False,
        "temporal_holdout_loaded": False,
    }
    campaign_sha = publish(output / "CHECKPOINT_DEVELOPMENT_EXECUTION_V1.json", campaign)
    receipt_sha = publish(
        output / "CHECKPOINT_DEVELOPMENT_EXECUTION_V1_RECEIPT.json",
        {
            "format_version": "abf_checkpoint_scaling_development_execution_receipt_v1",
            "status": campaign["status"],
            "campaign_sha256": campaign_sha,
            "checkpoint_label": args.checkpoint_label,
            "checkpoint_model_sha256": checkpoint_sha,
            "embedding_manifest_sha256": embedding_sha,
            "compatibility_canary_sha256": compatibility_sha,
            "frozen_before_training": True,
            "test_loaded": False,
        },
    )
    print(
        json.dumps(
            {
                "status": campaign["status"],
                "checkpoint_label": args.checkpoint_label,
                "campaign_sha256": campaign_sha,
                "receipt_sha256": receipt_sha,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
