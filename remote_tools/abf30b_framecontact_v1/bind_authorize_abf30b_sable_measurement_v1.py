#!/usr/bin/env python3
"""Bind ABF-30B development checkpoints and authorize exposed SABLE measurement."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


FINAL_MODEL_SHA256 = "c36e0b1d812f97317109f1f8ab085a349beb2a782a6f4092b3074bfc1eaab3ea"
STRUCTURE_MANIFEST_SHA256 = "795223b2ebd02d6ce37cebf7dadafe5b1f50b9b212976ba15e36af3e62f23259"
BOUND_LABEL_MANIFEST_SHA256 = "6e575b4468f25f17dbcffd7b34278cb765dce89a9008299e7df52198a028078e"
BOUND_LABEL_RECEIPT_SHA256 = "5834b892039c4a18a4e80cd083447866c84d3450931dca8a6b321f0d600a86c0"
SEEDS = (17, 29, 43)
ARMS = ("baseline", "edge_topk2_supervised")


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


def finite(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("non-finite value")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--sable-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists() or args.output_root.is_symlink():
        raise FileExistsError(args.output_root)

    plan_root = args.campaign_root / "authority/sable_measurement_v1"
    plan_path = plan_root / "ABF30B_SABLE_MEASUREMENT_PLAN_V1.json"
    plan_receipt_path = plan_root / "ABF30B_SABLE_MEASUREMENT_PLAN_V1_RECEIPT.json"
    analysis_path = args.campaign_root / "ABF30B_FRAMECONTACT_DEVELOPMENT_ANALYSIS_V1.json"
    analysis_receipt_path = (
        args.campaign_root / "ABF30B_FRAMECONTACT_DEVELOPMENT_ANALYSIS_V1_RECEIPT.json"
    )
    embedding_root = (
        args.campaign_root / "embeddings/sable_abf30b_antibody_esm2_antigen_v1"
    )
    embedding_path = embedding_root / "SABLE_ESM_EMBEDDING_MANIFEST.json"
    embedding_receipt_path = embedding_root / "SABLE_ABF30B_MIXED_EMBEDDING_RECEIPT.json"
    structure_path = (
        args.sable_root
        / "structure_cache/predicted_label_free_v1/STRUCTURE_CACHE_MANIFEST.json"
    )
    label_path = args.sable_root / "bound_labels_once_v1/SABLE_BOUND_LABEL_MANIFEST.json"
    label_receipt_path = (
        args.sable_root / "bound_labels_once_v1/SABLE_BOUND_LABEL_RECEIPT.json"
    )

    plan_receipt = json.loads(plan_receipt_path.read_text())
    analysis = json.loads(analysis_path.read_text())
    analysis_receipt = json.loads(analysis_receipt_path.read_text())
    embeddings = json.loads(embedding_path.read_text())
    embedding_receipt = json.loads(embedding_receipt_path.read_text())
    labels = json.loads(label_path.read_text())
    label_receipt = json.loads(label_receipt_path.read_text())
    for path, expected in (
        (plan_path, plan_receipt["plan_sha256"]),
        (analysis_path, analysis_receipt["analysis_sha256"]),
        (embedding_path, embedding_receipt["manifest_sha256"]),
        (structure_path, STRUCTURE_MANIFEST_SHA256),
        (label_path, BOUND_LABEL_MANIFEST_SHA256),
        (label_receipt_path, BOUND_LABEL_RECEIPT_SHA256),
    ):
        require_sha(path, expected)

    plan = json.loads(plan_path.read_text())
    if (
        plan.get("status")
        != "FROZEN_BEFORE_MODEL_SPECIFIC_ANALYST_EXPOSED_SABLE_MEASUREMENT"
        or plan.get("evidence_boundary", {}).get("fresh_blind_test") is not False
        or plan.get("evidence_boundary", {}).get("measurement_only") is not True
        or plan.get("evidence_boundary", {}).get("checkpoint_selection_used_sable")
        is not False
        or analysis.get("status") != "PASS_ABF30B_ANTIBODY_REPRESENTATION_DEVELOPMENT"
        or analysis_receipt.get("all_six_runs_verified") is not True
        or analysis.get("abf30b_checkpoint_model_sha256") != FINAL_MODEL_SHA256
        or analysis.get("development_only") is not True
        or analysis.get("independent_test") is not False
        or analysis.get("distance_label_cache_supplied") is not False
        or finite(analysis.get("distance_loss_weight")) != 0.0
        or analysis.get("selection_or_tuning_from_external_measurement") is not False
        or analysis.get("old_sabdab2_test_loaded") is not False
        or analysis.get("temporal_holdout_loaded") is not False
        or embeddings.get("status") != "PASS_LABEL_FREE_SABLE_MIXED_PARTNER_EMBEDDINGS"
        or embeddings.get("antibody_model", {}).get("checkpoint_model_sha256")
        != FINAL_MODEL_SHA256
        or embeddings.get("authority", {}).get("task_labels_read") is not False
        or embeddings.get("authority", {}).get("bound_label_artifacts_read") is not False
        or embedding_receipt.get("bound_labels_read") is not False
        or labels.get("status") != "PASS_SINGLE_BOUND_LABEL_OPENING_AND_MAPPING_GATE"
        or labels.get("mapping_gate", {}).get("passed") is not True
        or labels.get("authority", {}).get("opening_count") != 1
        or label_receipt.get("manifest_sha256") != BOUND_LABEL_MANIFEST_SHA256
    ):
        raise ValueError("model-specific SABLE binding authority differs")

    runs: dict[str, dict[str, dict[str, Any]]] = {}
    for arm in ARMS:
        analysis_arm = analysis.get("run_bindings", {}).get(arm, {})
        if set(analysis_arm) != {str(seed) for seed in SEEDS}:
            raise ValueError(f"development seed binding differs: {arm}")
        runs[arm] = {}
        for seed in SEEDS:
            run_root = args.campaign_root / "runs_development_v1" / arm / f"seed-{seed}"
            result_path = run_root / "FRAMECONTACT_DEVELOPMENT_RESULT.json"
            checkpoint_path = run_root / "FRAMECONTACT_SELECTED.pt"
            receipt_path = run_root / "RUN_RECEIPT.json"
            source = analysis_arm[str(seed)]
            expected = {
                "run_root": str(run_root),
                "arm": arm,
                "seed": seed,
                "selected_epoch": int(source["selected_epoch"]),
                "epochs_completed": int(source["epochs_completed"]),
                "result_sha256": sha256_file(result_path),
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "receipt_sha256": sha256_file(receipt_path),
                "embedding_manifest_sha256": analysis["embedding_manifest_sha256"],
                "test_loaded": False,
            }
            if source != expected:
                raise ValueError(f"development run binding differs: {arm}/seed-{seed}")
            result = json.loads(result_path.read_text())
            receipt = json.loads(receipt_path.read_text())
            if (
                result.get("configuration", {}).get("seed") != seed
                or finite(result.get("configuration", {}).get("distance_loss_weight")) != 0.0
                or result.get("authority", {}).get("distance_label_manifest_sha256") is not None
                or result.get("authority", {}).get("test_loaded") is not False
                or result.get("authority", {}).get("external_test_loaded") is not False
                or receipt.get("selected_checkpoint_sha256") != expected["checkpoint_sha256"]
                or receipt.get("result_sha256") != expected["result_sha256"]
            ):
                raise ValueError(f"development run authority differs: {arm}/seed-{seed}")
            runs[arm][str(seed)] = {
                **expected,
                "result_path": str(result_path),
                "checkpoint_path": str(checkpoint_path),
                "receipt_path": str(receipt_path),
            }

    args.output_root.mkdir(parents=True)
    binding = {
        "format_version": "abf30b_sable_checkpoint_binding_v1",
        "status": "PASS_EXACT_ABF30B_DEVELOPMENT_CHECKPOINT_BINDING",
        "measurement_plan_sha256": plan_receipt["plan_sha256"],
        "development_analysis_sha256": analysis_receipt["analysis_sha256"],
        "mixed_embedding_manifest_sha256": embedding_receipt["manifest_sha256"],
        "predicted_structure_manifest_sha256": STRUCTURE_MANIFEST_SHA256,
        "bound_label_manifest_sha256": BOUND_LABEL_MANIFEST_SHA256,
        "bound_label_receipt_sha256": BOUND_LABEL_RECEIPT_SHA256,
        "abf30b_checkpoint_model_sha256": FINAL_MODEL_SHA256,
        "runs": runs,
        "seed_matched_task_routing": {
            "contact": "baseline",
            "paratope": "baseline",
            "epitope": "edge_topk2_supervised",
        },
        "development_validation_used_for_checkpoint_selection": True,
        "sable_used_for_checkpoint_selection": False,
        "distance_labels_loaded": False,
        "experimental_bound_coordinates_used_as_model_input": False,
        "selection_or_tuning": False,
        "old_sabdab2_test_loaded": False,
        "temporal_holdout_loaded": False,
    }
    binding_path = args.output_root / "ABF30B_SABLE_CHECKPOINT_BINDING_V1.json"
    binding_sha = publish(binding_path, binding)
    binding_receipt = {
        "format_version": "abf30b_sable_checkpoint_binding_receipt_v1",
        "status": binding["status"],
        "binding_sha256": binding_sha,
        "all_six_checkpoint_sha256_verified": True,
        "sable_used_for_checkpoint_selection": False,
        "old_sabdab2_test_loaded": False,
        "temporal_holdout_loaded": False,
    }
    binding_receipt_sha = publish(
        args.output_root / "ABF30B_SABLE_CHECKPOINT_BINDING_V1_RECEIPT.json",
        binding_receipt,
    )

    authorization = {
        "format_version": "abf30b_sable_analyst_exposed_measurement_authorization_v1",
        "status": "AUTHORIZED_MODEL_SPECIFIC_ANALYST_EXPOSED_MEASUREMENT_ONLY",
        "measurement_plan_sha256": plan_receipt["plan_sha256"],
        "checkpoint_binding_sha256": binding_sha,
        "mixed_embedding_manifest_sha256": embedding_receipt["manifest_sha256"],
        "predicted_structure_manifest_sha256": STRUCTURE_MANIFEST_SHA256,
        "bound_label_manifest_sha256": BOUND_LABEL_MANIFEST_SHA256,
        "bound_label_receipt_sha256": BOUND_LABEL_RECEIPT_SHA256,
        "bound_label_opening_count": 1,
        "new_bound_label_opening_authorized": False,
        "model_specific_measurement_authorized": True,
        "measurement_ordinal_on_this_dataset": 2,
        "model_unseen_before_this_measurement": True,
        "analyst_has_seen_prior_model_results_on_dataset": True,
        "fresh_blind_test": False,
        "measurement_only": True,
        "formal_blind_test_claim_authorized": False,
        "calibration_refit": False,
        "model_parameters_updated": False,
        "selection_or_tuning": False,
        "feedback_to_model_or_protocol": False,
        "old_sabdab2_test_loaded": False,
        "temporal_holdout_loaded": False,
    }
    authorization_path = (
        args.output_root / "ABF30B_SABLE_MEASUREMENT_AUTHORIZATION_V1.json"
    )
    authorization_sha = publish(authorization_path, authorization)
    authorization_receipt = {
        "format_version": "abf30b_sable_analyst_exposed_measurement_authorization_receipt_v1",
        "status": authorization["status"],
        "authorization_sha256": authorization_sha,
        "checkpoint_binding_sha256": binding_sha,
        "fresh_blind_test": False,
        "measurement_only": True,
        "new_bound_label_opening_authorized": False,
        "old_sabdab2_test_loaded": False,
        "temporal_holdout_loaded": False,
    }
    authorization_receipt_sha = publish(
        args.output_root / "ABF30B_SABLE_MEASUREMENT_AUTHORIZATION_V1_RECEIPT.json",
        authorization_receipt,
    )
    print(
        json.dumps(
            {
                "binding_sha256": binding_sha,
                "binding_receipt_sha256": binding_receipt_sha,
                "authorization_sha256": authorization_sha,
                "authorization_receipt_sha256": authorization_receipt_sha,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
