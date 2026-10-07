#!/usr/bin/env python3
"""Audit six ABF-30B development runs and bind seed-matched task routing."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import median
from typing import Any


SEEDS = (17, 29, 43)
TASKS = ("contact", "epitope", "paratope")
CONTROL_ANALYSIS_SHA256 = "1382c6fc18624b53f50657c0c0d61408fa03f0acc5aa3a812ffa3f84bd23b78c"
STRUCTURE_SHA256 = "bf6ea4a21ff303f5e0bbf0f8edca5d824cc5024ec6c41fea9213172c64a8f614"
CONTACT_SHA256 = "d772a8dce8181d161318135a1bf4947506210363bb8ed3527bfa41dc7131ad88"
SPLIT_SHA256 = "ae1d72ade95c343c186915c12f0bfe0d392877cfb5bacd89ff0e42d8422d2700"
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
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
    return hashlib.sha256(payload).hexdigest()


def finite(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite value: {value!r}")
    return result


def selected_metrics(result: dict[str, Any]) -> dict[str, float]:
    selected = result["selected_checkpoint_validation"]
    return {
        task: finite(selected[task]["mean_per_complex_auprc"])
        for task in TASKS
    }


def verify_run(
    run_root: Path,
    *,
    arm: str,
    seed: int,
    arm_authority: dict[str, Any],
    embedding_sha256: str,
) -> tuple[dict[str, float], dict[str, Any]]:
    result_path = run_root / "FRAMECONTACT_DEVELOPMENT_RESULT.json"
    checkpoint_path = run_root / "FRAMECONTACT_SELECTED.pt"
    receipt_path = run_root / "RUN_RECEIPT.json"
    for path in (result_path, checkpoint_path, receipt_path):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"run artifact missing or symlinked: {path}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    authority = result.get("authority", {})
    configuration = result.get("configuration", {})
    selection = result.get("selection", {})
    history = result.get("history", [])
    expected_protocol_receipt = arm_authority["protocol_receipt_sha256"]
    if (
        result.get("experiment") != "epiformer_style_same_protocol"
        or configuration.get("seed") != seed
        or authority.get("protocol_file_sha256") != arm_authority["protocol_sha256"]
        or authority.get("protocol_receipt_sha256") != expected_protocol_receipt
        or authority.get("source_manifest_sha256") != arm_authority["source_manifest_sha256"]
        or authority.get("embedding_manifest_sha256") != embedding_sha256
        or authority.get("structure_manifest_sha256") != STRUCTURE_SHA256
        or authority.get("contact_manifest_sha256") != CONTACT_SHA256
        or authority.get("split_manifest_sha256") != SPLIT_SHA256
        or authority.get("distance_label_manifest_sha256") is not None
        or authority.get("test_loaded") is not False
        or authority.get("external_test_loaded") is not False
        or finite(configuration.get("distance_loss_weight")) != 0.0
        or finite(configuration.get("contrastive_loss_weight")) != 0.0
        or not history
        or int(selection.get("epochs_completed", 0)) != len(history)
        or int(selection.get("best_epoch", -1)) < 1
        or int(selection.get("best_epoch", -1)) > len(history)
        or receipt.get("seed") != seed
        or receipt.get("test_loaded") is not False
        or receipt.get("external_test_loaded") is not False
        or receipt.get("result_sha256") != sha256_file(result_path)
        or receipt.get("selected_checkpoint_sha256") != sha256_file(checkpoint_path)
        or receipt.get("selected_epoch") != selection.get("best_epoch")
    ):
        raise ValueError(f"run authority differs: {run_root}")
    if arm == "edge_topk2_supervised" and configuration.get("epitope_pooling") != "topk":
        # Older trainer variants do not emit epitope_pooling; the protocol and
        # source manifest remain the primary architecture authority.
        if "epitope_pooling" in configuration:
            raise ValueError(f"TopK2 configuration differs: {run_root}")
    for row in history:
        finite(row.get("elapsed_seconds"))
        for phase in ("train", "validation"):
            for name, value in row.get(phase, {}).items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    finite(value)
        if finite(row.get("train", {}).get("distance_loss", 0.0)) != 0.0:
            raise ValueError(f"distance loss is nonzero: {run_root}")
    metrics = selected_metrics(result)
    return metrics, {
        "run_root": str(run_root),
        "arm": arm,
        "seed": seed,
        "selected_epoch": int(selection["best_epoch"]),
        "epochs_completed": int(selection["epochs_completed"]),
        "result_sha256": sha256_file(result_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "receipt_sha256": sha256_file(receipt_path),
        "embedding_manifest_sha256": embedding_sha256,
        "test_loaded": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--control-analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.output, args.receipt_output):
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
    if sha256_file(args.control_analysis) != CONTROL_ANALYSIS_SHA256:
        raise ValueError("control no-distance analysis SHA differs")

    execution_root = args.campaign_root / "authority/development_execution_v1"
    campaign_path = execution_root / "ABF30B_DEVELOPMENT_EXECUTION_V1.json"
    campaign_receipt_path = execution_root / "ABF30B_DEVELOPMENT_EXECUTION_V1_RECEIPT.json"
    embedding_receipt_path = (
        args.campaign_root
        / "embeddings/sabdab2_abf30b_antibody_esm2_antigen_v1/SABDAB2_ABF30B_MIXED_EMBEDDING_RECEIPT.json"
    )
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    campaign_receipt = json.loads(campaign_receipt_path.read_text(encoding="utf-8"))
    embedding_receipt = json.loads(embedding_receipt_path.read_text(encoding="utf-8"))
    campaign_sha = sha256_file(campaign_path)
    embedding_sha = embedding_receipt["manifest_sha256"]
    if (
        campaign.get("status") != "FROZEN_BEFORE_SIX_RUN_DEVELOPMENT_TRAINING"
        or campaign_receipt.get("status") != campaign["status"]
        or campaign_receipt.get("campaign_sha256") != campaign_sha
        or campaign.get("seeds") != list(SEEDS)
        or campaign.get("embedding_manifest_sha256") != embedding_sha
        or embedding_receipt.get("checkpoint_model_sha256") != FINAL_MODEL_SHA256
        or campaign.get("single_factor") != "antibody embedding source"
        or campaign.get("antigen_embedding_changed") is not False
        or campaign.get("distance_loss_weight") != 0.0
        or campaign.get("distance_label_cache_supplied") is not False
        or campaign.get("test_loaded") is not False
        or campaign.get("temporal_holdout_loaded") is not False
    ):
        raise ValueError("development execution authority differs")

    arm_metrics: dict[str, dict[int, dict[str, float]]] = {}
    bindings: dict[str, dict[str, dict[str, Any]]] = {}
    for arm in ("baseline", "edge_topk2_supervised"):
        arm_metrics[arm] = {}
        bindings[arm] = {}
        for seed in SEEDS:
            run_root = args.campaign_root / "runs_development_v1" / arm / f"seed-{seed}"
            metrics, binding = verify_run(
                run_root,
                arm=arm,
                seed=seed,
                arm_authority=campaign["arms"][arm],
                embedding_sha256=embedding_sha,
            )
            arm_metrics[arm][seed] = metrics
            bindings[arm][str(seed)] = binding

    routed = {
        seed: {
            "contact": arm_metrics["baseline"][seed]["contact"],
            "epitope": arm_metrics["edge_topk2_supervised"][seed]["epitope"],
            "paratope": arm_metrics["baseline"][seed]["paratope"],
        }
        for seed in SEEDS
    }
    routed_median = {
        task: float(median(routed[seed][task] for seed in SEEDS)) for task in TASKS
    }
    control = json.loads(args.control_analysis.read_text(encoding="utf-8"))
    if control.get("status") != "PASS_PREDICTED_GEOMETRY_NO_BOUND_DISTANCE_AUXILIARY_DEVELOPMENT":
        raise ValueError("control analysis status differs")
    control_by_seed = {
        int(seed): {task: finite(value) for task, value in metrics.items()}
        for seed, metrics in control["routed_metrics_by_seed"].items()
    }
    delta = {
        seed: {task: routed[seed][task] - control_by_seed[seed][task] for task in TASKS}
        for seed in SEEDS
    }
    delta_median = {
        task: float(median(delta[seed][task] for seed in SEEDS)) for task in TASKS
    }
    payload = {
        "format_version": "abf30b_framecontact_development_analysis_v1",
        "status": "PASS_ABF30B_ANTIBODY_REPRESENTATION_DEVELOPMENT",
        "campaign_sha256": campaign_sha,
        "embedding_manifest_sha256": embedding_sha,
        "abf30b_checkpoint_model_sha256": FINAL_MODEL_SHA256,
        "control_analysis_sha256": CONTROL_ANALYSIS_SHA256,
        "run_bindings": bindings,
        "baseline_metrics_by_seed": {
            str(seed): arm_metrics["baseline"][seed] for seed in SEEDS
        },
        "topk2_expert_metrics_by_seed": {
            str(seed): arm_metrics["edge_topk2_supervised"][seed] for seed in SEEDS
        },
        "routed_metrics_by_seed": {str(seed): routed[seed] for seed in SEEDS},
        "routed_median_metrics": routed_median,
        "abf30b_minus_esm2_control_by_seed": {
            str(seed): delta[seed] for seed in SEEDS
        },
        "abf30b_minus_esm2_control_median": delta_median,
        "single_factor": "antibody residue embedding source",
        "antigen_embedding_changed": False,
        "development_only": True,
        "full_validation_is_checkpoint_selection_data": True,
        "independent_test": False,
        "distance_label_cache_supplied": False,
        "distance_loss_weight": 0.0,
        "experimental_bound_structure_used_to_define_task_contact_labels": True,
        "experimental_structure_required_at_inference": False,
        "selection_or_tuning_from_external_measurement": False,
        "old_sabdab2_test_loaded": False,
        "temporal_holdout_loaded": False,
    }
    analysis_sha = publish(args.output, payload)
    receipt = {
        "format_version": "abf30b_framecontact_development_analysis_receipt_v1",
        "status": payload["status"],
        "analysis_sha256": analysis_sha,
        "campaign_sha256": campaign_sha,
        "all_six_runs_verified": True,
        "checkpoint_binding_frozen": True,
        "test_loaded": False,
        "temporal_holdout_loaded": False,
    }
    receipt_sha = publish(args.receipt_output, receipt)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "routed_median_metrics": routed_median,
                "abf30b_minus_esm2_control_median": delta_median,
                "analysis_sha256": analysis_sha,
                "receipt_sha256": receipt_sha,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
