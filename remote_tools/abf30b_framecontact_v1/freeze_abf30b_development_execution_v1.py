#!/usr/bin/env python3
"""Bind mixed embeddings, compatibility sources, and six development runs."""

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


NO_DISTANCE_CAMPAIGN_SHA256 = "9eddf93aca94c1018287620e3720f873ece3730d9e3fe85d340b372a9ef0b4d0"
ORIGINAL_ESM_MODULE_SHA256 = "2adc1d711f7d5a9d0fcfb34d4813468393d18ea45b0751fffa9d569857570dd8"
COMPATIBILITY_ESM_MODULE_SHA256 = "6124719960e135eb19f89f7841b92ceb55ceca81dafb59578530d2eca853808e"
FINAL_MODEL_SHA256 = "c36e0b1d812f97317109f1f8ab085a349beb2a782a6f4092b3074bfc1eaab3ea"


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


def eligible_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for directory in ("scripts", "src", "protocols"):
        base = root / directory
        if base.is_symlink() or not base.is_dir():
            raise ValueError(f"source directory missing or unsafe: {base}")
        for path in base.rglob("*"):
            if not path.is_file() or path.is_symlink():
                continue
            relative = path.relative_to(root)
            if "__pycache__" in relative.parts or path.suffix == ".pyc":
                continue
            files.append(path)
    return sorted(set(files), key=lambda path: path.relative_to(root).as_posix())


def source_manifest(root: Path) -> dict[str, Any]:
    root = root.resolve()
    records = [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in eligible_files(root)
    ]
    tree_sha = hashlib.sha256(
        canonical_bytes(
            [{"path": row["path"], "sha256": row["sha256"]} for row in records]
        )
    ).hexdigest()
    return {
        "format_version": "framecontact_training_source_tree_v1",
        "source_root": str(root),
        "included_directories": ["scripts", "src", "protocols"],
        "record_count": len(records),
        "tree_sha256": tree_sha,
        "records": records,
        "claim_boundary": "Authenticated training source inventory only; no metric or test claim.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--baseline-source-root", type=Path, required=True)
    parser.add_argument("--topk2-source-root", type=Path, required=True)
    parser.add_argument("--original-source-root", type=Path, required=True)
    parser.add_argument("--embedding-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    authority = args.campaign_root / "authority"
    protocol_path = authority / "ABF30B_FRAMECONTACT_TRANSFER_PROTOCOL_V1.json"
    protocol_receipt_path = authority / "ABF30B_FRAMECONTACT_TRANSFER_PROTOCOL_V1_RECEIPT.json"
    chain_path = authority / "SABDAB2_DEVELOPMENT_ABF30B_CHAIN_MAP_V1.json"
    chain_receipt_path = authority / "SABDAB2_DEVELOPMENT_ABF30B_CHAIN_MAP_V1_RECEIPT.json"
    canary_path = authority / "ABF30B_DCP_EMBEDDING_CANARY_V1.json"
    canary_receipt_path = authority / "ABF30B_DCP_EMBEDDING_CANARY_V1_RECEIPT.json"
    embedding_manifest_path = args.embedding_root / "SABDAB2_ESM_EMBEDDING_MANIFEST.json"
    embedding_receipt_path = args.embedding_root / "SABDAB2_ABF30B_MIXED_EMBEDDING_RECEIPT.json"
    protocol_receipt = json.loads(protocol_receipt_path.read_text())
    chain_receipt = json.loads(chain_receipt_path.read_text())
    canary_receipt = json.loads(canary_receipt_path.read_text())
    embedding_receipt = json.loads(embedding_receipt_path.read_text())
    for path, expected in (
        (protocol_path, protocol_receipt["protocol_sha256"]),
        (chain_path, chain_receipt["chain_map_sha256"]),
        (canary_path, canary_receipt["result_sha256"]),
        (embedding_manifest_path, embedding_receipt["manifest_sha256"]),
    ):
        require_sha(path, expected)
    protocol = json.loads(protocol_path.read_text())
    chain = json.loads(chain_path.read_text())
    canary = json.loads(canary_path.read_text())
    embeddings = json.loads(embedding_manifest_path.read_text())
    if (
        protocol.get("status")
        != "FROZEN_BEFORE_CHAIN_TYPING_DCP_RESTORE_AND_MODEL_SPECIFIC_MEASUREMENT"
        or chain.get("status") != "PASS_LABEL_FREE_ANARCII_CHAIN_TYPING"
        or canary.get("status") != "PASS_DCP_EXACT_RESTORE_AND_RESIDUE_EMBEDDING_CANARY"
        or embeddings.get("status") != "PASS_COMPLETE_DEVELOPMENT_MIXED_PARTNER_EMBEDDINGS"
        or canary.get("checkpoint_model_sha256") != FINAL_MODEL_SHA256
        or embeddings.get("antibody_model", {}).get("checkpoint_model_sha256")
        != FINAL_MODEL_SHA256
        or any(value.get("task_labels_read") is not False for value in (chain, canary, embeddings))
    ):
        raise ValueError("upstream ABF-30B transfer authority differs")

    upstream_campaign_path = (
        args.upstream_root / "authority/no_privileged_distance_v1/CAMPAIGN_NO_PRIVILEGED_DISTANCE_V1.json"
    )
    require_sha(upstream_campaign_path, NO_DISTANCE_CAMPAIGN_SHA256)
    upstream_campaign = json.loads(upstream_campaign_path.read_text())
    if upstream_campaign.get("status") != "FROZEN_BEFORE_NO_PRIVILEGED_DISTANCE_TRAINING_V1":
        raise ValueError("upstream no-distance campaign differs")

    original_module_path = args.original_source_root / "src/interaction_first/esm_embeddings.py"
    baseline_module_path = args.baseline_source_root / "src/interaction_first/esm_embeddings.py"
    topk2_module_path = args.topk2_source_root / "src/interaction_first/esm_embeddings.py"
    original = load_module(original_module_path, "abf30b_original_esm", ORIGINAL_ESM_MODULE_SHA256)
    baseline_compat = load_module(
        baseline_module_path, "abf30b_baseline_compat", COMPATIBILITY_ESM_MODULE_SHA256
    )
    load_module(topk2_module_path, "abf30b_topk2_compat", COMPATIBILITY_ESM_MODULE_SHA256)
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
    if tuple(antibody_embedding.shape) != (int(antibody_record["sequence_length"]), 1280):
        raise RuntimeError("compatibility loader ABF output shape differs")
    original_rejected_abf = False
    try:
        original.load_embedding(
            antibody_path,
            expected_key=antibody_record["sequence_sha256"],
            expected_length=int(antibody_record["sequence_length"]),
        )
    except ValueError:
        original_rejected_abf = True
    if not original_rejected_abf:
        raise RuntimeError("original ESM-only loader unexpectedly accepts ABF artifact")

    output.mkdir(parents=True)
    source_roots = {
        "baseline": args.baseline_source_root.resolve(),
        "edge_topk2_supervised": args.topk2_source_root.resolve(),
    }
    source_manifest_shas: dict[str, str] = {}
    for arm, root in source_roots.items():
        manifest_path = output / f"SOURCE_MANIFEST_{arm.upper()}_ABF30B_V1.json"
        source_manifest_shas[arm] = publish(manifest_path, source_manifest(root))

    upstream_authority = args.upstream_root / "authority/no_privileged_distance_v1"
    template_names = {
        "baseline": "PROTOCOL_PREDICTED_GEOMETRY_BASELINE_NO_PRIVILEGED_DISTANCE_V1.json",
        "edge_topk2_supervised": "PROTOCOL_PREDICTED_GEOMETRY_TOPK2_NO_PRIVILEGED_DISTANCE_V1.json",
    }
    arms: dict[str, dict[str, str]] = {}
    for arm, template_name in template_names.items():
        expected_template_sha = upstream_campaign["arms"][arm]["protocol_sha256"]
        template_path = upstream_authority / template_name
        require_sha(template_path, expected_template_sha)
        value = json.loads(template_path.read_text())
        value["schema_version"] = f"abf30b_mixed_embedding_{arm}_development_v1"
        value["protocol_status"] = "frozen_before_abf30b_mixed_embedding_training_v1"
        value["data"]["embedding_manifest_sha256"] = embedding_receipt["manifest_sha256"]
        value["data"]["mixed_partner_embedding"] = {
            "antibody": "ABF-650M OAS-30B final residue hidden states",
            "antigen": "unchanged pinned ESM-2 t33 650M residue embeddings",
            "single_factor_relative_to_control": "antibody embedding source",
            "manifest_sha256": embedding_receipt["manifest_sha256"],
        }
        value["claim_boundary"].update(
            {
                "claim": "Development-only representation-swap comparison for ABF-650M OAS-30B antibody residue embeddings.",
                "representation_swap_only": True,
                "antigen_embedding_changed": False,
                "framecontact_architecture_loss_optimization_split_and_labels_changed": False,
                "selection_or_tuning_from_external_measurement": False,
                "old_sabdab2_test_loaded": False,
                "temporal_holdout_loaded": False,
            }
        )
        value["abf30b_transfer_v1"] = {
            "transfer_protocol_sha256": protocol_receipt["protocol_sha256"],
            "chain_map_sha256": chain_receipt["chain_map_sha256"],
            "dcp_canary_sha256": canary_receipt["result_sha256"],
            "embedding_manifest_sha256": embedding_receipt["manifest_sha256"],
            "source_manifest_sha256": source_manifest_shas[arm],
            "compatibility_module_sha256": COMPATIBILITY_ESM_MODULE_SHA256,
            "control_esm_output_bit_equal": True,
            "original_loader_rejects_abf_artifact": True,
        }
        protocol_output = output / f"PROTOCOL_{arm.upper()}_ABF30B_V1.json"
        protocol_sha = publish(protocol_output, value)
        protocol_receipt_output = output / f"PROTOCOL_{arm.upper()}_ABF30B_V1_RECEIPT.json"
        protocol_receipt_sha = publish(
            protocol_receipt_output,
            {
                "format_version": "abf30b_framecontact_development_protocol_receipt_v1",
                "arm": arm,
                "protocol_file_sha256": protocol_sha,
                "source_manifest_sha256": source_manifest_shas[arm],
                "embedding_manifest_sha256": embedding_receipt["manifest_sha256"],
                "distance_loss_weight": 0.0,
                "frozen_before_training": True,
                "test_loaded": False,
            },
        )
        arms[arm] = {
            "protocol_sha256": protocol_sha,
            "protocol_receipt_sha256": protocol_receipt_sha,
            "source_manifest_sha256": source_manifest_shas[arm],
            "source_root": str(source_roots[arm]),
        }

    compatibility = {
        "format_version": "abf30b_framecontact_embedding_compatibility_canary_v1",
        "status": "PASS_AUTHORITY_AWARE_LOADER_COMPATIBILITY",
        "original_esm_module_sha256": ORIGINAL_ESM_MODULE_SHA256,
        "compatibility_module_sha256": COMPATIBILITY_ESM_MODULE_SHA256,
        "original_esm_output_bit_equal": True,
        "abf_artifact_shape_valid": True,
        "original_esm_only_loader_rejects_abf": True,
        "task_labels_read": False,
        "test_loaded": False,
    }
    compatibility_sha = publish(output / "EMBEDDING_COMPATIBILITY_CANARY_V1.json", compatibility)
    campaign = {
        "format_version": "abf30b_framecontact_development_execution_v1",
        "status": "FROZEN_BEFORE_SIX_RUN_DEVELOPMENT_TRAINING",
        "transfer_protocol_sha256": protocol_receipt["protocol_sha256"],
        "embedding_manifest_sha256": embedding_receipt["manifest_sha256"],
        "compatibility_canary_sha256": compatibility_sha,
        "arms": arms,
        "seeds": [17, 29, 43],
        "task_routing": {
            "contact": "baseline same seed",
            "paratope": "baseline same seed",
            "epitope": "edge_topk2_supervised same seed",
        },
        "single_factor": "antibody embedding source",
        "antigen_embedding_changed": False,
        "distance_loss_weight": 0.0,
        "distance_label_cache_supplied": False,
        "test_loaded": False,
        "temporal_holdout_loaded": False,
    }
    campaign_sha = publish(output / "ABF30B_DEVELOPMENT_EXECUTION_V1.json", campaign)
    receipt_sha = publish(
        output / "ABF30B_DEVELOPMENT_EXECUTION_V1_RECEIPT.json",
        {
            "format_version": "abf30b_framecontact_development_execution_receipt_v1",
            "status": campaign["status"],
            "campaign_sha256": campaign_sha,
            "embedding_manifest_sha256": embedding_receipt["manifest_sha256"],
            "compatibility_canary_sha256": compatibility_sha,
            "frozen_before_training": True,
            "test_loaded": False,
        },
    )
    print(f"campaign_sha256={campaign_sha}")
    print(f"receipt_sha256={receipt_sha}")


if __name__ == "__main__":
    main()
