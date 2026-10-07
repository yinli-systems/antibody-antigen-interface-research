#!/usr/bin/env python3
"""Bind exact no-distance checkpoints for one-time SABLE evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_CAMPAIGN_SHA256 = (
    "9eddf93aca94c1018287620e3720f873ece3730d9e3fe85d340b372a9ef0b4d0"
)
EXPECTED_EXTERNAL_PLAN_SHA256 = (
    "d97f0b94863fe9227d80c5675e944091809245b8fab22a89cda81148a8de172a"
)
SEEDS = (17, 29, 43)
ARMS = ("baseline", "edge_topk2_supervised")


class BindingError(ValueError):
    """Raised when a checkpoint or frozen authority differs."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def write_json(path: Path, payload: Any) -> str:
    data = canonical_bytes(payload)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--analysis-receipt", type=Path, required=True)
    parser.add_argument("--external-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.output, args.receipt_output):
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
    if sha256_file(args.external_plan) != EXPECTED_EXTERNAL_PLAN_SHA256:
        raise BindingError("frozen SABLE external execution plan SHA256 differs")
    external_plan = json.loads(args.external_plan.read_text(encoding="utf-8"))
    if (
        external_plan.get("status")
        != "FROZEN_POST_GATE_BEFORE_BOUND_LABEL_OPENING"
        or external_plan.get("bound_label_opening_authorized") is not False
        or external_plan.get("task_labels_read_at_freeze") is not False
        or external_plan.get("old_sabdab2_test_loaded") is not False
        or external_plan.get("temporal_holdout_loaded") is not False
    ):
        raise BindingError("SABLE external plan is not sealed")

    analysis = json.loads(args.analysis.read_text(encoding="utf-8"))
    receipt = json.loads(args.analysis_receipt.read_text(encoding="utf-8"))
    analysis_sha256 = sha256_file(args.analysis)
    if (
        analysis.get("status")
        != "PASS_PREDICTED_GEOMETRY_NO_BOUND_DISTANCE_AUXILIARY_DEVELOPMENT"
        or analysis.get("campaign_sha256") != EXPECTED_CAMPAIGN_SHA256
        or receipt.get("status") != analysis.get("status")
        or receipt.get("analysis_sha256") != analysis_sha256
        or receipt.get("campaign_sha256") != EXPECTED_CAMPAIGN_SHA256
        or receipt.get("all_six_runs_verified") is not True
        or analysis.get("development_only") is not True
        or analysis.get("full_validation_is_checkpoint_selection_data") is not True
        or analysis.get("independent_confirmation") is not False
        or analysis.get("distance_label_cache_supplied") is not False
        or analysis.get("distance_labels_loaded") is not False
        or analysis.get("distance_loss_weight") != 0.0
        or analysis.get("experimental_bound_geometry_used_as_model_input") is not False
        or analysis.get(
            "experimental_bound_geometry_used_for_auxiliary_training_supervision"
        )
        is not False
        or analysis.get("experimental_structure_required_at_inference") is not False
        or analysis.get("old_sabdab2_test_loaded") is not False
        or analysis.get("temporal_holdout_loaded") is not False
        or analysis.get("selection_or_tuning_from_this_study") is not False
    ):
        raise BindingError("no-distance analysis authority differs")

    source_bindings = analysis.get("run_bindings", {})
    bound_runs: dict[str, dict[str, dict[str, Any]]] = {}
    for arm in ARMS:
        arm_bindings = source_bindings.get(arm, {})
        if set(arm_bindings) != {str(seed) for seed in SEEDS}:
            raise BindingError(f"seed set differs for {arm}")
        bound_runs[arm] = {}
        for seed in SEEDS:
            binding = arm_bindings[str(seed)]
            run_root = (
                args.campaign_root
                / "runs_no_privileged_distance_v1"
                / arm
                / f"seed-{seed}"
            )
            result_path = run_root / "FRAMECONTACT_DEVELOPMENT_RESULT.json"
            checkpoint_path = run_root / "FRAMECONTACT_SELECTED.pt"
            run_receipt_path = run_root / "RUN_RECEIPT.json"
            expected = {
                "run_root": str(run_root),
                "result_sha256": sha256_file(result_path),
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "receipt_sha256": sha256_file(run_receipt_path),
                "selected_epoch": int(binding["selected_epoch"]),
            }
            if binding != expected:
                raise BindingError(f"run binding differs: {arm}/seed-{seed}")
            result = json.loads(result_path.read_text(encoding="utf-8"))
            run_receipt = json.loads(run_receipt_path.read_text(encoding="utf-8"))
            if (
                result.get("configuration", {}).get("seed") != seed
                or float(
                    result.get("configuration", {}).get("distance_loss_weight", -1.0)
                )
                != 0.0
                or result.get("authority", {}).get("distance_label_manifest_sha256")
                is not None
                or result.get("authority", {}).get("test_loaded") is not False
                or result.get("authority", {}).get("external_test_loaded") is not False
                or run_receipt.get("selected_checkpoint_sha256")
                != expected["checkpoint_sha256"]
                or run_receipt.get("test_loaded") is not False
                or run_receipt.get("external_test_loaded") is not False
            ):
                raise BindingError(f"run result authority differs: {arm}/seed-{seed}")
            bound_runs[arm][str(seed)] = {
                **expected,
                "result_path": str(result_path),
                "checkpoint_path": str(checkpoint_path),
                "receipt_path": str(run_receipt_path),
            }

    payload = {
        "format_version": "sable_v0_2_0_no_distance_checkpoint_binding_v1",
        "status": "PASS_EXACT_NO_DISTANCE_CHECKPOINT_BINDING",
        "external_execution_plan_sha256": EXPECTED_EXTERNAL_PLAN_SHA256,
        "no_distance_campaign_sha256": EXPECTED_CAMPAIGN_SHA256,
        "no_distance_analysis_path": str(args.analysis),
        "no_distance_analysis_sha256": analysis_sha256,
        "no_distance_analysis_receipt_sha256": sha256_file(args.analysis_receipt),
        "seeds": list(SEEDS),
        "runs": bound_runs,
        "seed_matched_task_routing": {
            "contact": "baseline checkpoint with the same seed",
            "paratope": "baseline checkpoint with the same seed",
            "epitope": "edge_topk2_supervised checkpoint with the same seed",
        },
        "distance_label_cache_supplied": False,
        "distance_labels_loaded": False,
        "experimental_bound_geometry_used_as_model_input": False,
        "experimental_bound_geometry_used_for_auxiliary_training_supervision": False,
        "experimental_structure_required_at_inference": False,
        "development_validation_used_for_checkpoint_selection": True,
        "performance_threshold_applied": False,
        "task_labels_read": False,
        "sable_bound_structures_downloaded": False,
        "bound_label_opening_authorized": False,
        "old_sabdab2_test_loaded": False,
        "temporal_holdout_loaded": False,
        "selection_or_tuning": False,
    }
    output_sha256 = write_json(args.output, payload)
    receipt_payload = {
        "format_version": "sable_v0_2_0_no_distance_checkpoint_binding_receipt_v1",
        "status": payload["status"],
        "binding_sha256": output_sha256,
        "analysis_sha256": analysis_sha256,
        "all_six_checkpoint_sha256_verified": True,
        "bound_label_opening_authorized": False,
        "task_labels_read": False,
        "test_loaded": False,
        "temporal_holdout_loaded": False,
    }
    receipt_sha256 = write_json(args.receipt_output, receipt_payload)
    print(
        json.dumps(
            {
                **receipt_payload,
                "receipt_sha256": receipt_sha256,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
