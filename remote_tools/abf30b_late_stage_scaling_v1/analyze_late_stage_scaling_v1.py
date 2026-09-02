#!/usr/bin/env python3
"""Audit late-stage ABF checkpoint runs and summarize development scaling."""

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
NEW_CHECKPOINTS = ("segment_000058", "segment_000063")
FINAL_CHECKPOINT = "segment_000069"
STRUCTURE_SHA256 = "bf6ea4a21ff303f5e0bbf0f8edca5d824cc5024ec6c41fea9213172c64a8f614"
CONTACT_SHA256 = "d772a8dce8181d161318135a1bf4947506210363bb8ed3527bfa41dc7131ad88"
SPLIT_SHA256 = "ae1d72ade95c343c186915c12f0bfe0d392877cfb5bacd89ff0e42d8422d2700"


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
    if (
        result.get("experiment") != "epiformer_style_same_protocol"
        or configuration.get("seed") != seed
        or authority.get("protocol_file_sha256") != arm_authority["protocol_sha256"]
        or authority.get("protocol_receipt_sha256")
        != arm_authority["protocol_receipt_sha256"]
        or authority.get("source_manifest_sha256")
        != arm_authority["source_manifest_sha256"]
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
    if arm == "edge_topk2_supervised" and "epitope_pooling" in configuration:
        if configuration["epitope_pooling"] != "topk":
            raise ValueError(f"TopK2 configuration differs: {run_root}")
    for row in history:
        finite(row.get("elapsed_seconds"))
        for phase in ("train", "validation"):
            for value in row.get(phase, {}).values():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    finite(value)
        if finite(row.get("train", {}).get("distance_loss", 0.0)) != 0.0:
            raise ValueError(f"distance loss is nonzero: {run_root}")
    return selected_metrics(result), {
        "arm": arm,
        "seed": seed,
        "selected_epoch": int(selection["best_epoch"]),
        "epochs_completed": int(selection["epochs_completed"]),
        "result_sha256": sha256_file(result_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "receipt_sha256": sha256_file(receipt_path),
        "test_loaded": False,
    }


def subtract(
    left: dict[int, dict[str, float]], right: dict[int, dict[str, float]]
) -> dict[int, dict[str, float]]:
    return {
        seed: {task: left[seed][task] - right[seed][task] for task in TASKS}
        for seed in SEEDS
    }


def medians(values: dict[int, dict[str, float]]) -> dict[str, float]:
    return {
        task: float(median(values[seed][task] for seed in SEEDS)) for task in TASKS
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--scaling-protocol", type=Path, required=True)
    parser.add_argument("--scaling-protocol-sha256", required=True)
    parser.add_argument("--control-analysis", type=Path, required=True)
    parser.add_argument("--final-analysis", type=Path, required=True)
    parser.add_argument("--final-analysis-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.output, args.receipt_output):
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
    if sha256_file(args.scaling_protocol) != args.scaling_protocol_sha256:
        raise ValueError("scaling protocol SHA differs")
    protocol = json.loads(args.scaling_protocol.read_text(encoding="utf-8"))
    if (
        protocol.get("status") != "FROZEN_BEFORE_LATE_STAGE_SCALING_EXECUTION"
        or tuple(protocol.get("execution", {}).get("new_embedding_checkpoints", []))
        != NEW_CHECKPOINTS
        or protocol.get("claim_boundary", {}).get("late_stage_only") is not True
        or protocol.get("claim_boundary", {}).get("development_only") is not True
        or protocol.get("claim_boundary", {}).get("sable_or_external_measurement_repeated")
        is not False
        or protocol.get("test_loaded") is not False
        or protocol.get("temporal_holdout_loaded") is not False
    ):
        raise ValueError("scaling protocol differs")
    if sha256_file(args.control_analysis) != protocol["control_analysis_sha256"]:
        raise ValueError("control analysis SHA differs")
    final_authority = protocol["final_checkpoint_reuse"]
    if (
        sha256_file(args.final_analysis) != final_authority["analysis_sha256"]
        or sha256_file(args.final_analysis_receipt)
        != final_authority["analysis_receipt_sha256"]
        or final_authority.get("six_runs_rerun") is not False
    ):
        raise ValueError("final checkpoint reuse authority differs")

    routed_by_checkpoint: dict[str, dict[int, dict[str, float]]] = {}
    bindings: dict[str, Any] = {}
    for label in NEW_CHECKPOINTS:
        checkpoint_root = args.campaign_root / "checkpoints" / label
        execution_root = checkpoint_root / "authority/development_execution_v1"
        execution_path = execution_root / "CHECKPOINT_DEVELOPMENT_EXECUTION_V1.json"
        receipt_path = execution_root / "CHECKPOINT_DEVELOPMENT_EXECUTION_V1_RECEIPT.json"
        execution = json.loads(execution_path.read_text(encoding="utf-8"))
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        embedding_receipt_path = (
            checkpoint_root
            / "embeddings/sabdab2_abf_antibody_esm2_antigen_v1"
            / "SABDAB2_ABF_CHECKPOINT_MIXED_EMBEDDING_RECEIPT.json"
        )
        embedding_receipt = json.loads(embedding_receipt_path.read_text(encoding="utf-8"))
        embedding_sha = embedding_receipt["manifest_sha256"]
        checkpoint_authority = protocol["checkpoints"][label]
        if (
            execution.get("status")
            != "FROZEN_BEFORE_SIX_RUN_LATE_STAGE_SCALING_DEVELOPMENT"
            or receipt.get("campaign_sha256") != sha256_file(execution_path)
            or execution.get("scaling_protocol_sha256") != args.scaling_protocol_sha256
            or execution.get("checkpoint_label") != label
            or execution.get("checkpoint_model_sha256")
            != checkpoint_authority["checkpoint_model_sha256"]
            or execution.get("embedding_manifest_sha256") != embedding_sha
            or execution.get("distance_loss_weight") != 0.0
            or execution.get("distance_label_cache_supplied") is not False
            or execution.get("test_loaded") is not False
            or execution.get("temporal_holdout_loaded") is not False
        ):
            raise ValueError(f"checkpoint execution differs: {label}")
        arm_metrics: dict[str, dict[int, dict[str, float]]] = {}
        bindings[label] = {}
        for arm in ("baseline", "edge_topk2_supervised"):
            arm_metrics[arm] = {}
            bindings[label][arm] = {}
            for seed in SEEDS:
                run_root = checkpoint_root / "runs_development_v1" / arm / f"seed-{seed}"
                metrics, binding = verify_run(
                    run_root,
                    arm=arm,
                    seed=seed,
                    arm_authority=execution["arms"][arm],
                    embedding_sha256=embedding_sha,
                )
                arm_metrics[arm][seed] = metrics
                bindings[label][arm][str(seed)] = binding
        routed_by_checkpoint[label] = {
            seed: {
                "contact": arm_metrics["baseline"][seed]["contact"],
                "epitope": arm_metrics["edge_topk2_supervised"][seed]["epitope"],
                "paratope": arm_metrics["baseline"][seed]["paratope"],
            }
            for seed in SEEDS
        }

    final = json.loads(args.final_analysis.read_text(encoding="utf-8"))
    final_receipt = json.loads(args.final_analysis_receipt.read_text(encoding="utf-8"))
    if (
        final.get("status") != "PASS_ABF30B_ANTIBODY_REPRESENTATION_DEVELOPMENT"
        or final_receipt.get("analysis_sha256") != sha256_file(args.final_analysis)
        or final_receipt.get("all_six_runs_verified") is not True
        or final_receipt.get("test_loaded") is not False
        or final.get("old_sabdab2_test_loaded") is not False
        or final.get("temporal_holdout_loaded") is not False
    ):
        raise ValueError("final analysis differs")
    routed_by_checkpoint[FINAL_CHECKPOINT] = {
        int(seed): {task: finite(value) for task, value in metrics.items()}
        for seed, metrics in final["routed_metrics_by_seed"].items()
    }
    bindings[FINAL_CHECKPOINT] = {
        "reused_analysis_sha256": sha256_file(args.final_analysis),
        "reused_analysis_receipt_sha256": sha256_file(args.final_analysis_receipt),
        "six_runs_rerun": False,
    }

    control = json.loads(args.control_analysis.read_text(encoding="utf-8"))
    if control.get("status") != "PASS_PREDICTED_GEOMETRY_NO_BOUND_DISTANCE_AUXILIARY_DEVELOPMENT":
        raise ValueError("control analysis status differs")
    control_by_seed = {
        int(seed): {task: finite(value) for task, value in metrics.items()}
        for seed, metrics in control["routed_metrics_by_seed"].items()
    }
    ordered_labels = (*NEW_CHECKPOINTS, FINAL_CHECKPOINT)
    delta_vs_control = {
        label: subtract(routed_by_checkpoint[label], control_by_seed)
        for label in ordered_labels
    }
    adjacent_deltas = {
        "segment_000063_minus_segment_000058": subtract(
            routed_by_checkpoint["segment_000063"],
            routed_by_checkpoint["segment_000058"],
        ),
        "segment_000069_minus_segment_000063": subtract(
            routed_by_checkpoint["segment_000069"],
            routed_by_checkpoint["segment_000063"],
        ),
        "segment_000069_minus_segment_000058": subtract(
            routed_by_checkpoint["segment_000069"],
            routed_by_checkpoint["segment_000058"],
        ),
    }
    monotonic = {
        task: all(
            routed_by_checkpoint[ordered_labels[index]][seed][task]
            <= routed_by_checkpoint[ordered_labels[index + 1]][seed][task]
            for seed in SEEDS
            for index in range(len(ordered_labels) - 1)
        )
        for task in TASKS
    }
    payload = {
        "format_version": "abf30b_late_stage_token_scaling_development_analysis_v1",
        "status": "PASS_ABF30B_LATE_STAGE_TOKEN_SCALING_DEVELOPMENT",
        "scaling_protocol_sha256": args.scaling_protocol_sha256,
        "checkpoint_order": list(ordered_labels),
        "checkpoint_tokens_and_intrinsic_validation": {
            label: {
                "consumed_residue_tokens": protocol["checkpoints"][label][
                    "consumed_residue_tokens"
                ],
                "next_update": protocol["checkpoints"][label]["next_update"],
                "checkpoint_model_sha256": protocol["checkpoints"][label][
                    "checkpoint_model_sha256"
                ],
                "validation_masked_cross_entropy": protocol["checkpoints"][label][
                    "validation_masked_cross_entropy"
                ],
                "validation_masked_accuracy": protocol["checkpoints"][label][
                    "validation_masked_accuracy"
                ],
            }
            for label in ordered_labels
        },
        "run_bindings": bindings,
        "routed_metrics_by_checkpoint_and_seed": {
            label: {str(seed): routed_by_checkpoint[label][seed] for seed in SEEDS}
            for label in ordered_labels
        },
        "routed_median_metrics_by_checkpoint": {
            label: medians(routed_by_checkpoint[label]) for label in ordered_labels
        },
        "checkpoint_minus_esm2_control_by_seed": {
            label: {str(seed): delta_vs_control[label][seed] for seed in SEEDS}
            for label in ordered_labels
        },
        "checkpoint_minus_esm2_control_median": {
            label: medians(delta_vs_control[label]) for label in ordered_labels
        },
        "paired_adjacent_deltas_by_seed": {
            name: {str(seed): values[seed] for seed in SEEDS}
            for name, values in adjacent_deltas.items()
        },
        "paired_adjacent_delta_medians": {
            name: medians(values) for name, values in adjacent_deltas.items()
        },
        "strictly_nondecreasing_for_all_seeds": monotonic,
        "late_stage_only": True,
        "full_scaling_curve": False,
        "development_only": True,
        "full_validation_is_checkpoint_selection_data": True,
        "independent_test": False,
        "distance_label_cache_supplied": False,
        "distance_loss_weight": 0.0,
        "external_measurement_repeated": False,
        "old_sabdab2_test_loaded": False,
        "temporal_holdout_loaded": False,
    }
    analysis_sha = publish(args.output, payload)
    receipt = {
        "format_version": "abf30b_late_stage_token_scaling_analysis_receipt_v1",
        "status": payload["status"],
        "analysis_sha256": analysis_sha,
        "scaling_protocol_sha256": args.scaling_protocol_sha256,
        "new_run_count_verified": 12,
        "reused_final_run_count": 6,
        "all_result_checkpoint_receipt_hashes_verified": True,
        "all_losses_finite": True,
        "distance_loss_zero": True,
        "external_measurement_repeated": False,
        "test_loaded": False,
        "temporal_holdout_loaded": False,
    }
    receipt_sha = publish(args.receipt_output, receipt)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "routed_median_metrics_by_checkpoint": payload[
                    "routed_median_metrics_by_checkpoint"
                ],
                "paired_adjacent_delta_medians": payload[
                    "paired_adjacent_delta_medians"
                ],
                "analysis_sha256": analysis_sha,
                "receipt_sha256": receipt_sha,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
