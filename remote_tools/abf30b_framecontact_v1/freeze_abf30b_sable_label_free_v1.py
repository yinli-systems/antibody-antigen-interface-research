#!/usr/bin/env python3
"""Freeze SABLE label-free ABF-30B embedding materialization authorities."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


INPUT_PLAN_SHA256 = "c82c12f08c87b42d21aee31cc555373ffbc4211415d9aff302444baefc971ad6"
INPUT_PLAN_RECEIPT_SHA256 = "a6ba3357c282a8f375ab6e0890bb78d90b5cdbb41a75ff05b088796ab237b1d2"
STRUCTURE_MANIFEST_SHA256 = "795223b2ebd02d6ce37cebf7dadafe5b1f50b9b212976ba15e36af3e62f23259"
STRUCTURE_RECEIPT_SHA256 = "607ca15081d6b34687650ec1a13c378c233a48c5b21a6a71b75a188cb171b82e"
ESM2_MANIFEST_SHA256 = "67670200608ac98573c9d0329fd06fb7f5973d0a47bd96aeea5861b94955abc9"
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--sable-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists() or args.output_root.is_symlink():
        raise FileExistsError(args.output_root)
    authority = args.campaign_root / "authority"
    transfer_path = authority / "ABF30B_FRAMECONTACT_TRANSFER_PROTOCOL_V1.json"
    transfer_receipt_path = authority / "ABF30B_FRAMECONTACT_TRANSFER_PROTOCOL_V1_RECEIPT.json"
    canary_path = authority / "ABF30B_DCP_EMBEDDING_CANARY_V1.json"
    canary_receipt_path = authority / "ABF30B_DCP_EMBEDDING_CANARY_V1_RECEIPT.json"
    transfer_receipt = json.loads(transfer_receipt_path.read_text())
    canary_receipt = json.loads(canary_receipt_path.read_text())
    paths = {
        "transfer_protocol": (transfer_path, transfer_receipt["protocol_sha256"]),
        "dcp_canary": (canary_path, canary_receipt["result_sha256"]),
        "input_plan": (
            args.sable_root / "structure_prediction/af3_label_free_v1/AF3_LABEL_FREE_INPUT_PLAN.json",
            INPUT_PLAN_SHA256,
        ),
        "input_plan_receipt": (
            args.sable_root
            / "structure_prediction/af3_label_free_v1/AF3_LABEL_FREE_INPUT_PLAN_RECEIPT.json",
            INPUT_PLAN_RECEIPT_SHA256,
        ),
        "predicted_structure_manifest": (
            args.sable_root
            / "structure_cache/predicted_label_free_v1/STRUCTURE_CACHE_MANIFEST.json",
            STRUCTURE_MANIFEST_SHA256,
        ),
        "predicted_structure_receipt": (
            args.sable_root
            / "structure_cache/predicted_label_free_v1/STRUCTURE_CACHE_RECEIPT.json",
            STRUCTURE_RECEIPT_SHA256,
        ),
        "control_esm2_manifest": (
            args.sable_root
            / "embeddings/esm2_t33_650m_label_free_v1/SABLE_ESM_EMBEDDING_MANIFEST.json",
            ESM2_MANIFEST_SHA256,
        ),
    }
    for path, expected in paths.values():
        require_sha(path, expected)
    transfer = json.loads(transfer_path.read_text())
    canary = json.loads(canary_path.read_text())
    plan = json.loads(paths["input_plan"][0].read_text())
    structures = json.loads(paths["predicted_structure_manifest"][0].read_text())
    esm2 = json.loads(paths["control_esm2_manifest"][0].read_text())
    if (
        transfer.get("status")
        != "FROZEN_BEFORE_CHAIN_TYPING_DCP_RESTORE_AND_MODEL_SPECIFIC_MEASUREMENT"
        or transfer.get("external_measurement", {}).get("fresh_blind_test") is not False
        or transfer.get("external_measurement", {}).get(
            "analyst_has_seen_prior_model_results_on_dataset"
        )
        is not True
        or canary.get("status") != "PASS_DCP_EXACT_RESTORE_AND_RESIDUE_EMBEDDING_CANARY"
        or canary.get("checkpoint_model_sha256") != FINAL_MODEL_SHA256
        or plan.get("status") != "PASS_FROZEN_LABEL_FREE_AF3_INPUT_PLAN"
        or plan.get("task_labels_read") is not False
        or structures.get("status") != "PASS_LABEL_FREE_SEQUENCE_PREDICTED_STRUCTURE_CACHE"
        or structures.get("authority", {}).get("task_labels_read") is not False
        or esm2.get("status") != "PASS_LABEL_FREE_ESM2_EMBEDDINGS"
        or esm2.get("authority", {}).get("task_labels_read") is not False
    ):
        raise ValueError("SABLE label-free authority differs")
    args.output_root.mkdir(parents=True)
    payload = {
        "format_version": "abf30b_sable_label_free_protocol_v1",
        "status": "FROZEN_BEFORE_SABLE_LABEL_FREE_CHAIN_TYPING_AND_MIXED_EMBEDDINGS",
        "transfer_protocol_sha256": transfer_receipt["protocol_sha256"],
        "dcp_canary_sha256": canary_receipt["result_sha256"],
        "authorities": {name: digest for name, (_, digest) in paths.items()},
        "cohort": {"entries": 97, "unique_pdb_ids": 88},
        "single_factor": "replace antibody ESM-2 residue embeddings with ABF-650M OAS-30B final hidden states",
        "antigen_embedding_source": "unchanged pinned ESM-2 t33 650M artifacts",
        "chain_typing": "ANARCII 2.0.8; accept H/K/L and require H then K/L for paired antibodies",
        "task_labels_read": False,
        "bound_label_artifacts_read": False,
        "performance_read": False,
        "old_sabdab2_test_loaded": False,
        "temporal_holdout_loaded": False,
        "model_specific_measurement_submitted": False,
    }
    protocol_sha = publish(args.output_root / "ABF30B_SABLE_LABEL_FREE_PROTOCOL_V1.json", payload)
    receipt = {
        "format_version": "abf30b_sable_label_free_protocol_receipt_v1",
        "status": payload["status"],
        "protocol_sha256": protocol_sha,
        "frozen_before_chain_typing": True,
        "frozen_before_mixed_embedding_materialization": True,
        "bound_labels_read": False,
        "test_loaded": False,
    }
    receipt_sha = publish(
        args.output_root / "ABF30B_SABLE_LABEL_FREE_PROTOCOL_V1_RECEIPT.json", receipt
    )
    print(json.dumps({"protocol_sha256": protocol_sha, "receipt_sha256": receipt_sha}))


if __name__ == "__main__":
    main()
