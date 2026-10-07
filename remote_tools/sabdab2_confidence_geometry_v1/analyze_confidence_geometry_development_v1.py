#!/usr/bin/env python3
"""Audit pLDDT-70 runs and compare them with frozen no-distance controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median
from typing import Any


SEEDS = (17, 29, 43)
TASKS = ("contact", "epitope", "paratope")
CONTROL_ANALYSIS_SHA256 = (
    "1382c6fc18624b53f50657c0c0d61408fa03f0acc5aa3a812ffa3f84bd23b78c"
)
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def finite(value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("non-finite value")
    return number


def selected_metrics(result: dict[str, Any]) -> dict[str, float]:
    selected = result["selected_checkpoint_validation"]
    return {
        task: finite(selected[task]["mean_per_complex_auprc"]) for task in TASKS
    }


def verify_run(
    root: Path,
    *,
    seed: int,
    protocol_sha256: str,
    protocol_receipt_sha256: str,
    source_sha256: str,
    split_sha256: str,
    structure_sha256: str,
    contact_sha256: str,
) -> tuple[dict[str, float], dict[str, Any]]:
    result_path = root / "FRAMECONTACT_DEVELOPMENT_RESULT.json"
    checkpoint_path = root / "FRAMECONTACT_SELECTED.pt"
    receipt_path = root / "RUN_RECEIPT.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    authority = result.get("authority", {})
    configuration = result.get("configuration", {})
    history = result.get("history", [])
    if (
        result.get("experiment") != "epiformer_style_same_protocol"
        or configuration.get("seed") != seed
        or finite(configuration.get("distance_loss_weight", -1.0)) != 0.0
        or authority.get("protocol_file_sha256") != protocol_sha256
        or authority.get("protocol_receipt_sha256") != protocol_receipt_sha256
        or authority.get("source_manifest_sha256") != source_sha256
        or authority.get("split_manifest_sha256") != split_sha256
        or authority.get("structure_manifest_sha256") != structure_sha256
        or authority.get("contact_manifest_sha256") != contact_sha256
        or authority.get("distance_label_manifest_sha256") is not None
        or authority.get("test_loaded") is not False
        or authority.get("external_test_loaded") is not False
        or receipt.get("seed") != seed
        or receipt.get("test_loaded") is not False
        or receipt.get("external_test_loaded") is not False
        or receipt.get("result_sha256") != sha256_file(result_path)
        or receipt.get("selected_checkpoint_sha256") != sha256_file(checkpoint_path)
        or receipt.get("selected_epoch") != result["selection"]["best_epoch"]
        or not history
    ):
        raise ValueError(f"run authority differs: {root}")
    for row in history:
        train = row.get("train", {})
        for key, value in train.items():
            if isinstance(value, (int, float)):
                finite(value)
        if finite(train.get("distance_loss", 0.0)) != 0.0:
            raise ValueError(f"distance loss is nonzero: {root}")
    return selected_metrics(result), {
        "run_root": str(root),
        "result_sha256": sha256_file(result_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "receipt_sha256": sha256_file(receipt_path),
        "selected_epoch": int(result["selection"]["best_epoch"]),
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
        raise ValueError("no-distance control analysis SHA-256 differs")
    control = json.loads(args.control_analysis.read_text(encoding="utf-8"))
    if (
        control.get("status")
        != "PASS_PREDICTED_GEOMETRY_NO_BOUND_DISTANCE_AUXILIARY_DEVELOPMENT"
        or control.get("distance_loss_weight") != 0.0
        or control.get("old_sabdab2_test_loaded") is not False
        or control.get("temporal_holdout_loaded") is not False
    ):
        raise ValueError("no-distance control analysis content differs")

    authority_root = args.campaign_root / "authority" / "confidence_geometry_plddt70_v1"
    checksums_path = authority_root / "AUTHORITY_SHA256SUMS.txt"
    campaign_path = authority_root / "CAMPAIGN_CONFIDENCE_GEOMETRY_PLDDT70_V1.json"
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    if (
        campaign.get("status") != "FROZEN_BEFORE_PLDDT70_DEVELOPMENT_TRAINING"
        or campaign.get("threshold") != 70.0
        or campaign.get("distance_loss_weight") != 0.0
        or campaign.get("distance_label_cache_supplied") is not False
        or campaign.get("selection_or_tuning") is not False
        or campaign.get("test_loaded") is not False
        or campaign.get("temporal_holdout_loaded") is not False
        or campaign.get("seeds") != list(SEEDS)
    ):
        raise ValueError("confidence-geometry campaign authority differs")
    for raw_line in checksums_path.read_text(encoding="utf-8").splitlines():
        expected, raw_path = raw_line.split(maxsplit=1)
        path = Path(raw_path.strip())
        if sha256_file(path) != expected:
            raise ValueError(f"authority checksum differs: {path}")

    split_sha = campaign["development_split_sha256"]
    structure_sha = campaign["model_input_structure_manifest_sha256"]
    contact_sha = campaign["contact_manifest_sha256"]
    arm_metrics: dict[str, dict[int, dict[str, float]]] = {}
    bindings: dict[str, dict[str, dict[str, Any]]] = {}
    for arm in ("baseline", "edge_topk2_supervised"):
        arm_metrics[arm] = {}
        bindings[arm] = {}
        arm_authority = campaign["arms"][arm]
        for seed in SEEDS:
            run_root = (
                args.campaign_root
                / "runs_confidence_geometry_plddt70_v1"
                / arm
                / f"seed-{seed}"
            )
            metrics, binding = verify_run(
                run_root,
                seed=seed,
                protocol_sha256=arm_authority["protocol_sha256"],
                protocol_receipt_sha256=arm_authority["receipt_sha256"],
                source_sha256=arm_authority["source_manifest_sha256"],
                split_sha256=split_sha,
                structure_sha256=structure_sha,
                contact_sha256=contact_sha,
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
    control_by_seed = {
        int(seed): {task: finite(value) for task, value in values.items()}
        for seed, values in control["routed_metrics_by_seed"].items()
    }
    delta_by_seed = {
        seed: {
            task: routed[seed][task] - control_by_seed[seed][task] for task in TASKS
        }
        for seed in SEEDS
    }
    median_metrics = {
        task: float(median(routed[seed][task] for seed in SEEDS)) for task in TASKS
    }
    median_delta = {
        task: float(median(delta_by_seed[seed][task] for seed in SEEDS))
        for task in TASKS
    }
    minimum_paratope_delta = min(delta_by_seed[seed]["paratope"] for seed in SEEDS)
    gate = campaign["promotion_gate"]
    checks = {
        "median_paratope_delta_gt_0": median_delta["paratope"]
        > float(gate["median_paratope_delta_gt"]),
        "minimum_seed_paratope_delta_gte_minus_0_005": minimum_paratope_delta
        >= float(gate["minimum_seed_paratope_delta_gte"]),
        "median_contact_delta_gte_minus_0_002": median_delta["contact"]
        >= float(gate["median_contact_delta_gte"]),
        "median_epitope_delta_gte_minus_0_005": median_delta["epitope"]
        >= float(gate["median_epitope_delta_gte"]),
    }
    payload = {
        "format_version": "sabdab2_confidence_geometry_plddt70_analysis_v1",
        "status": "PASS_CONFIDENCE_MASKED_PREDICTED_GEOMETRY_DEVELOPMENT_AUDIT",
        "campaign_sha256": sha256_file(campaign_path),
        "control_analysis_sha256": CONTROL_ANALYSIS_SHA256,
        "model_input_structure_manifest_sha256": structure_sha,
        "contact_manifest_sha256": contact_sha,
        "development_split_sha256": split_sha,
        "threshold": 70.0,
        "run_bindings": bindings,
        "baseline_metrics_by_seed": {
            str(seed): arm_metrics["baseline"][seed] for seed in SEEDS
        },
        "topk2_expert_metrics_by_seed": {
            str(seed): arm_metrics["edge_topk2_supervised"][seed] for seed in SEEDS
        },
        "routed_metrics_by_seed": {str(seed): routed[seed] for seed in SEEDS},
        "routed_median_metrics": median_metrics,
        "plddt70_minus_control_by_seed": {
            str(seed): delta_by_seed[seed] for seed in SEEDS
        },
        "plddt70_minus_control_median": median_delta,
        "promotion_checks": checks,
        "development_hypothesis_gate_pass": all(checks.values()),
        "development_only": True,
        "full_validation_is_checkpoint_selection_data": True,
        "independent_confirmation": False,
        "distance_label_cache_supplied": False,
        "distance_labels_loaded": False,
        "distance_loss_weight": 0.0,
        "experimental_bound_geometry_used_as_model_input": False,
        "experimental_bound_geometry_used_for_auxiliary_training_supervision": False,
        "experimental_bound_structure_used_to_define_task_contact_labels": True,
        "old_sabdab2_test_loaded": False,
        "temporal_holdout_loaded": False,
        "selection_or_tuning_from_this_study": False,
    }
    args.output.write_bytes(canonical_bytes(payload))
    receipt = {
        "format_version": "sabdab2_confidence_geometry_plddt70_analysis_receipt_v1",
        "status": payload["status"],
        "analysis_sha256": sha256_file(args.output),
        "campaign_sha256": payload["campaign_sha256"],
        "all_six_runs_verified": True,
        "development_hypothesis_gate_pass": payload[
            "development_hypothesis_gate_pass"
        ],
        "test_loaded": False,
        "temporal_holdout_loaded": False,
    }
    args.receipt_output.write_bytes(canonical_bytes(receipt))
    print(
        json.dumps(
            {
                **receipt,
                "routed_median_metrics": median_metrics,
                "plddt70_minus_control_median": median_delta,
                "receipt_sha256": sha256_file(args.receipt_output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
