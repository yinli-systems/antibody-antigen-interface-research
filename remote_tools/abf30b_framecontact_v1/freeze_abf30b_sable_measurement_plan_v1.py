#!/usr/bin/env python3
"""Freeze the ABF-30B model-specific, analyst-exposed SABLE measurement plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


TRANSFER_PROTOCOL_SHA256 = "98b3213fb2f06675f2b954139fad354d5c7ab1d6b37e190c048a5b4ad3da4ef3"
ORIGINAL_EXTERNAL_PLAN_SHA256 = "d97f0b94863fe9227d80c5675e944091809245b8fab22a89cda81148a8de172a"
STRUCTURE_MANIFEST_SHA256 = "795223b2ebd02d6ce37cebf7dadafe5b1f50b9b212976ba15e36af3e62f23259"
BOUND_LABEL_MANIFEST_SHA256 = "6e575b4468f25f17dbcffd7b34278cb765dce89a9008299e7df52198a028078e"
BOUND_LABEL_RECEIPT_SHA256 = "5834b892039c4a18a4e80cd083447866c84d3450931dca8a6b321f0d600a86c0"
FINAL_MODEL_SHA256 = "c36e0b1d812f97317109f1f8ab085a349beb2a782a6f4092b3074bfc1eaab3ea"
SEEDS = (17, 29, 43)


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
    parser.add_argument("--original-external-plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists() or args.output_root.is_symlink():
        raise FileExistsError(args.output_root)

    transfer_path = (
        args.campaign_root / "authority/ABF30B_FRAMECONTACT_TRANSFER_PROTOCOL_V1.json"
    )
    structure_path = (
        args.sable_root
        / "structure_cache/predicted_label_free_v1/STRUCTURE_CACHE_MANIFEST.json"
    )
    label_path = args.sable_root / "bound_labels_once_v1/SABLE_BOUND_LABEL_MANIFEST.json"
    label_receipt_path = (
        args.sable_root / "bound_labels_once_v1/SABLE_BOUND_LABEL_RECEIPT.json"
    )
    for path, expected in (
        (transfer_path, TRANSFER_PROTOCOL_SHA256),
        (args.original_external_plan, ORIGINAL_EXTERNAL_PLAN_SHA256),
        (structure_path, STRUCTURE_MANIFEST_SHA256),
        (label_path, BOUND_LABEL_MANIFEST_SHA256),
        (label_receipt_path, BOUND_LABEL_RECEIPT_SHA256),
    ):
        require_sha(path, expected)

    transfer = json.loads(transfer_path.read_text())
    original_plan = json.loads(args.original_external_plan.read_text())
    structures = json.loads(structure_path.read_text())
    labels = json.loads(label_path.read_text())
    label_receipt = json.loads(label_receipt_path.read_text())
    if (
        transfer.get("external_measurement", {}).get("fresh_blind_test") is not False
        or transfer.get("external_measurement", {}).get(
            "analyst_has_seen_prior_model_results_on_dataset"
        )
        is not True
        or original_plan.get("status") != "FROZEN_POST_GATE_BEFORE_BOUND_LABEL_OPENING"
        or structures.get("status") != "PASS_LABEL_FREE_SEQUENCE_PREDICTED_STRUCTURE_CACHE"
        or structures.get("authority", {}).get("task_labels_read") is not False
        or labels.get("status") != "PASS_SINGLE_BOUND_LABEL_OPENING_AND_MAPPING_GATE"
        or labels.get("mapping_gate", {}).get("passed") is not True
        or labels.get("authority", {}).get("opening_count") != 1
        or label_receipt.get("manifest_sha256") != BOUND_LABEL_MANIFEST_SHA256
        or label_receipt.get("mapping_gate_passed") is not True
    ):
        raise ValueError("SABLE measurement authority differs")

    args.output_root.mkdir(parents=True)
    plan = {
        "format_version": "abf30b_sable_analyst_exposed_measurement_plan_v1",
        "status": "FROZEN_BEFORE_MODEL_SPECIFIC_ANALYST_EXPOSED_SABLE_MEASUREMENT",
        "model": {
            "name": "ABF-650M OAS-30B final",
            "checkpoint_model_sha256": FINAL_MODEL_SHA256,
            "model_unseen_before_this_measurement": True,
        },
        "dataset": {
            "name": "SABLE",
            "version": "0.2.0",
            "doi": "10.5281/zenodo.21759053",
            "official_test_split": True,
            "entries": 97,
            "unique_pdb_ids": 88,
            "predicted_structure_manifest_sha256": STRUCTURE_MANIFEST_SHA256,
            "bound_label_manifest_sha256": BOUND_LABEL_MANIFEST_SHA256,
            "bound_label_receipt_sha256": BOUND_LABEL_RECEIPT_SHA256,
            "bound_label_opening_count": 1,
            "new_bound_label_opening_required": False,
        },
        "task_routing": {
            "contact": "baseline checkpoint from the same seed",
            "paratope": "baseline checkpoint from the same seed",
            "epitope": "edge_topk2_supervised checkpoint from the same seed",
            "seeds": list(SEEDS),
            "mixture_weight_tuning": False,
        },
        "metrics": {
            "average_precision_definition": "sklearn.metrics.average_precision_score",
            "reported_per_seed": [
                "mean per eligible entry AP",
                "pooled AP",
                "pooled AUROC",
                "precision",
                "recall",
                "F1",
                "MCC",
            ],
            "seed_summary": "median across seeds 17, 29, and 43",
            "pdb_equal_weighted_bootstrap": {
                "replicates": 10000,
                "seed": 20260830,
                "confidence_interval": [0.025, 0.975],
            },
            "temperature_and_threshold": "frozen development checkpoint values; no refit",
        },
        "evidence_boundary": {
            "fresh_blind_test": False,
            "analyst_has_seen_prior_model_results_on_dataset": True,
            "measurement_only": True,
            "checkpoint_selection_used_sable": False,
            "selection_or_tuning_from_this_measurement": False,
            "model_or_protocol_feedback": False,
            "formal_blind_test_claim_authorized": False,
            "old_sabdab2_test_loaded": False,
            "temporal_holdout_loaded": False,
        },
        "input_boundary": {
            "antibody_embedding": "ABF-650M OAS-30B final residue hidden states",
            "antigen_embedding": "unchanged pinned ESM-2 t33 650M residue embeddings",
            "geometry": "independent sequence-predicted chain-internal N/CA/C backbone",
            "experimental_bound_coordinates_used_as_model_input": False,
            "cross_partner_geometry_used_as_model_input": False,
            "experimental_bound_coordinates_used_for_labels_only": True,
        },
        "authority": {
            "transfer_protocol_sha256": TRANSFER_PROTOCOL_SHA256,
            "original_external_plan_sha256": ORIGINAL_EXTERNAL_PLAN_SHA256,
            "label_manifest_metadata_read_at_freeze": True,
            "bound_label_artifact_contents_read_at_freeze": False,
            "prior_performance_artifact_read_by_this_freezer": False,
        },
    }
    plan_sha = publish(args.output_root / "ABF30B_SABLE_MEASUREMENT_PLAN_V1.json", plan)
    receipt = {
        "format_version": "abf30b_sable_analyst_exposed_measurement_plan_receipt_v1",
        "status": plan["status"],
        "plan_sha256": plan_sha,
        "frozen_before_new_model_measurement": True,
        "fresh_blind_test": False,
        "measurement_only": True,
        "new_bound_label_opening_required": False,
        "old_sabdab2_test_loaded": False,
        "temporal_holdout_loaded": False,
    }
    receipt_sha = publish(
        args.output_root / "ABF30B_SABLE_MEASUREMENT_PLAN_V1_RECEIPT.json", receipt
    )
    print(json.dumps({"plan_sha256": plan_sha, "receipt_sha256": receipt_sha}))


if __name__ == "__main__":
    main()
