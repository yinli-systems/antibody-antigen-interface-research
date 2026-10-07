#!/usr/bin/env python3
"""Audit six no-distance predicted-geometry development runs and route by task."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from statistics import median
from typing import Any


HERE = Path(__file__).resolve().parent
BASE_PATH = HERE / "analyze_predicted_geometry_development_v1.py"
SPEC = importlib.util.spec_from_file_location("predgeom_base_analyzer_no_dist", BASE_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(BASE_PATH)
BASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)

UPSTREAM_ANALYSIS_SHA256 = (
    "a4168a919e97b1344d718946fdd51305803773ae3355a3014ee2f05583a077b7"
)


def _metrics_by_seed(payload: dict[str, Any]) -> dict[int, dict[str, float]]:
    return {
        int(seed): {task: BASE.finite(value) for task, value in values.items()}
        for seed, values in payload["routed_metrics_by_seed"].items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--bound-reference", type=Path, required=True)
    parser.add_argument("--upstream-analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.output, args.receipt_output):
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
    if BASE.sha256_file(args.bound_reference) != BASE.BOUND_REFERENCE_SHA256:
        raise ValueError("bound-complex development reference SHA-256 differs")
    if BASE.sha256_file(args.upstream_analysis) != UPSTREAM_ANALYSIS_SHA256:
        raise ValueError("upstream predicted-geometry analysis SHA-256 differs")

    authority_root = args.campaign_root / "authority" / "no_privileged_distance_v1"
    checksums_path = authority_root / "AUTHORITY_SHA256SUMS.txt"
    campaign_path = authority_root / "CAMPAIGN_NO_PRIVILEGED_DISTANCE_V1.json"
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    if (
        campaign.get("status")
        != "FROZEN_BEFORE_NO_PRIVILEGED_DISTANCE_TRAINING_V1"
        or campaign.get("distance_loss_weight") != 0.0
        or campaign.get("distance_label_cache_supplied") is not False
        or campaign.get("distance_labels_loaded") is not False
        or campaign.get("model_architecture_changed") is not False
        or campaign.get("optimizer_or_schedule_changed") is not False
        or campaign.get("pair_contact_labels_changed") is not False
        or campaign.get("experimental_bound_geometry_used_as_model_input") is not False
        or campaign.get(
            "experimental_bound_geometry_used_for_auxiliary_training_supervision"
        )
        is not False
        or campaign.get("test_loaded") is not False
        or campaign.get("temporal_holdout_loaded") is not False
        or campaign.get("selection_or_tuning") is not False
        or campaign.get("seeds") != list(BASE.SEEDS)
    ):
        raise ValueError("no-privileged-distance campaign authority differs")
    for raw_line in checksums_path.read_text(encoding="utf-8").splitlines():
        expected, raw_path = raw_line.split(maxsplit=1)
        path = Path(raw_path.strip())
        if BASE.sha256_file(path) != expected:
            raise ValueError(f"authority checksum differs: {path}")

    split_sha256 = campaign["development_split_sha256"]
    structure_sha256 = campaign["model_input_structure_manifest_sha256"]
    contact_sha256 = campaign["contact_manifest_sha256"]
    arm_metrics: dict[str, dict[int, dict[str, float]]] = {}
    bindings: dict[str, dict[str, dict[str, Any]]] = {}
    for arm in ("baseline", "edge_topk2_supervised"):
        arm_authority = campaign["arms"][arm]
        arm_metrics[arm] = {}
        bindings[arm] = {}
        for seed in BASE.SEEDS:
            run_root = (
                args.campaign_root
                / "runs_no_privileged_distance_v1"
                / arm
                / f"seed-{seed}"
            )
            metrics, binding = BASE.verify_run(
                run_root,
                seed=seed,
                protocol_sha256=arm_authority["protocol_sha256"],
                protocol_receipt_sha256=arm_authority["receipt_sha256"],
                source_sha256=arm_authority["source_manifest_sha256"],
                split_sha256=split_sha256,
                structure_sha256=structure_sha256,
            )
            result = json.loads(
                (run_root / "FRAMECONTACT_DEVELOPMENT_RESULT.json").read_text(
                    encoding="utf-8"
                )
            )
            authority = result.get("authority", {})
            configuration = result.get("configuration", {})
            history = result.get("history", [])
            if (
                authority.get("contact_manifest_sha256") != contact_sha256
                or authority.get("distance_label_manifest_sha256") is not None
                or authority.get("test_loaded") is not False
                or authority.get("external_test_loaded") is not False
                or float(configuration.get("distance_loss_weight", -1.0)) != 0.0
                or float(configuration.get("contrastive_loss_weight", -1.0)) != 0.0
                or any(
                    float(row.get("train", {}).get("distance_loss", 0.0)) != 0.0
                    for row in history
                )
            ):
                raise ValueError(f"no-distance run authority differs: {run_root}")
            arm_metrics[arm][seed] = metrics
            bindings[arm][str(seed)] = binding

    routed = {
        seed: {
            "contact": arm_metrics["baseline"][seed]["contact"],
            "epitope": arm_metrics["edge_topk2_supervised"][seed]["epitope"],
            "paratope": arm_metrics["baseline"][seed]["paratope"],
        }
        for seed in BASE.SEEDS
    }
    routed_median = {
        task: float(median(routed[seed][task] for seed in BASE.SEEDS))
        for task in BASE.TASKS
    }
    upstream = json.loads(args.upstream_analysis.read_text(encoding="utf-8"))
    bound = json.loads(args.bound_reference.read_text(encoding="utf-8"))
    upstream_by_seed = _metrics_by_seed(upstream)
    bound_by_seed = _metrics_by_seed(bound)

    def delta(reference: dict[int, dict[str, float]]) -> dict[int, dict[str, float]]:
        return {
            seed: {
                task: routed[seed][task] - reference[seed][task]
                for task in BASE.TASKS
            }
            for seed in BASE.SEEDS
        }

    versus_upstream = delta(upstream_by_seed)
    versus_bound = delta(bound_by_seed)

    def median_delta(values: dict[int, dict[str, float]]) -> dict[str, float]:
        return {
            task: float(median(values[seed][task] for seed in BASE.SEEDS))
            for task in BASE.TASKS
        }

    payload = {
        "format_version": "sabdab2_predicted_geometry_no_privileged_distance_analysis_v1",
        "status": "PASS_PREDICTED_GEOMETRY_NO_BOUND_DISTANCE_AUXILIARY_DEVELOPMENT",
        "campaign_sha256": BASE.sha256_file(campaign_path),
        "model_input_structure_manifest_sha256": structure_sha256,
        "contact_manifest_sha256": contact_sha256,
        "development_split_sha256": split_sha256,
        "upstream_analysis_sha256": UPSTREAM_ANALYSIS_SHA256,
        "bound_reference_analysis_sha256": BASE.BOUND_REFERENCE_SHA256,
        "run_bindings": bindings,
        "baseline_metrics_by_seed": {
            str(seed): arm_metrics["baseline"][seed] for seed in BASE.SEEDS
        },
        "topk2_expert_metrics_by_seed": {
            str(seed): arm_metrics["edge_topk2_supervised"][seed]
            for seed in BASE.SEEDS
        },
        "routed_metrics_by_seed": {str(seed): routed[seed] for seed in BASE.SEEDS},
        "routed_median_metrics": routed_median,
        "no_distance_minus_distance_auxiliary_by_seed": {
            str(seed): versus_upstream[seed] for seed in BASE.SEEDS
        },
        "no_distance_minus_distance_auxiliary_median": median_delta(versus_upstream),
        "no_distance_minus_bound_reference_by_seed": {
            str(seed): versus_bound[seed] for seed in BASE.SEEDS
        },
        "no_distance_minus_bound_reference_median": median_delta(versus_bound),
        "development_only": True,
        "full_validation_is_checkpoint_selection_data": True,
        "independent_confirmation": False,
        "distance_label_cache_supplied": False,
        "distance_labels_loaded": False,
        "distance_loss_weight": 0.0,
        "experimental_bound_geometry_used_as_model_input": False,
        "experimental_bound_geometry_used_for_auxiliary_training_supervision": False,
        "experimental_bound_structure_used_to_define_task_contact_labels": True,
        "experimental_structure_required_at_inference": False,
        "old_sabdab2_test_loaded": False,
        "temporal_holdout_loaded": False,
        "selection_or_tuning_from_this_study": False,
    }
    args.output.write_bytes(BASE.canonical_bytes(payload))
    receipt = {
        "format_version": "sabdab2_predicted_geometry_no_privileged_distance_analysis_receipt_v1",
        "status": payload["status"],
        "analysis_sha256": BASE.sha256_file(args.output),
        "campaign_sha256": payload["campaign_sha256"],
        "all_six_runs_verified": True,
        "distance_labels_loaded": False,
        "test_loaded": False,
        "temporal_holdout_loaded": False,
    }
    args.receipt_output.write_bytes(BASE.canonical_bytes(receipt))
    print(
        json.dumps(
            {
                "status": payload["status"],
                "routed_median_metrics": routed_median,
                "no_distance_minus_distance_auxiliary_median": payload[
                    "no_distance_minus_distance_auxiliary_median"
                ],
                "no_distance_minus_bound_reference_median": payload[
                    "no_distance_minus_bound_reference_median"
                ],
                "analysis_sha256": receipt["analysis_sha256"],
                "receipt_sha256": BASE.sha256_file(args.receipt_output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
