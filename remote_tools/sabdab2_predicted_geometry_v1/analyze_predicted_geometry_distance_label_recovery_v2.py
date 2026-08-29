#!/usr/bin/env python3
"""Audit six corrected predicted-geometry development runs and route by task."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from statistics import median


HERE = Path(__file__).resolve().parent
BASE_PATH = HERE / "analyze_predicted_geometry_development_v1.py"
SPEC = importlib.util.spec_from_file_location("predgeom_base_analyzer_v2", BASE_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(BASE_PATH)
BASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--bound-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    if args.receipt_output.exists() or args.receipt_output.is_symlink():
        raise FileExistsError(args.receipt_output)
    if BASE.sha256_file(args.bound_reference) != BASE.BOUND_REFERENCE_SHA256:
        raise ValueError("bound-complex development reference SHA-256 differs")

    authority_root = args.campaign_root / "authority" / "distance_label_recovery_v2"
    checksums_path = authority_root / "AUTHORITY_SHA256SUMS.txt"
    campaign_path = authority_root / "CAMPAIGN_PREDICTED_GEOMETRY_DISTANCE_LABEL_RECOVERY_V2.json"
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    if (
        campaign.get("status")
        != "FROZEN_BEFORE_DISTANCE_LABEL_RECOVERY_TRAINING_V2"
        or campaign.get("implementation_corrected") is not True
        or campaign.get("scientific_protocol_changed") is not False
        or campaign.get("model_architecture_changed") is not False
        or campaign.get("loss_or_optimization_changed") is not False
        or campaign.get("distance_bin_definition_changed") is not False
        or campaign.get("invalid_run_reused") is not False
        or campaign.get("experimental_bound_geometry_used_as_model_input") is not False
        or campaign.get(
            "experimental_bound_geometry_used_for_auxiliary_training_supervision"
        )
        is not True
        or campaign.get("test_loaded") is not False
        or campaign.get("temporal_holdout_loaded") is not False
        or campaign.get("selection_or_tuning") is not False
        or campaign.get("seeds") != list(BASE.SEEDS)
    ):
        raise ValueError("distance-label recovery campaign authority differs")
    for raw_line in checksums_path.read_text(encoding="utf-8").splitlines():
        expected, raw_path = raw_line.split(maxsplit=1)
        path = Path(raw_path.strip())
        if BASE.sha256_file(path) != expected:
            raise ValueError(f"authority checksum differs: {path}")

    split_sha256 = campaign["development_split_sha256"]
    structure_sha256 = campaign["model_input_structure_manifest_sha256"]
    contact_sha256 = campaign["contact_manifest_sha256"]
    distance_sha256 = campaign["distance_label_manifest_sha256"]
    arm_metrics: dict[str, dict[int, dict[str, float]]] = {}
    bindings: dict[str, dict[str, dict]] = {}
    for arm in ("baseline", "edge_topk2_supervised"):
        arm_authority = campaign["arms"][arm]
        arm_metrics[arm] = {}
        bindings[arm] = {}
        for seed in BASE.SEEDS:
            run_root = (
                args.campaign_root
                / "runs_distance_label_recovery_v2"
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
            result_authority = result.get("authority", {})
            configuration = result.get("configuration", {})
            if (
                result_authority.get("contact_manifest_sha256") != contact_sha256
                or result_authority.get("distance_label_manifest_sha256")
                != distance_sha256
                or result_authority.get("test_loaded") is not False
                or result_authority.get("external_test_loaded") is not False
                or float(configuration.get("distance_loss_weight", -1.0)) != 0.1
                or float(configuration.get("contrastive_loss_weight", -1.0)) != 0.0
            ):
                raise ValueError(f"corrected run authority differs: {run_root}")
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
    predicted_median = {
        task: float(median(routed[seed][task] for seed in BASE.SEEDS))
        for task in BASE.TASKS
    }
    bound = json.loads(args.bound_reference.read_text(encoding="utf-8"))
    bound_by_seed = {
        int(seed): {task: BASE.finite(value) for task, value in values.items()}
        for seed, values in bound["routed_metrics_by_seed"].items()
    }
    delta_by_seed = {
        seed: {
            task: routed[seed][task] - bound_by_seed[seed][task]
            for task in BASE.TASKS
        }
        for seed in BASE.SEEDS
    }
    delta_median = {
        task: float(median(delta_by_seed[seed][task] for seed in BASE.SEEDS))
        for task in BASE.TASKS
    }
    payload = {
        "format_version": "sabdab2_predicted_geometry_distance_label_recovery_analysis_v2",
        "status": "PASS_PREDICTED_GEOMETRY_DEVELOPMENT_SENSITIVITY_CORRECTED_DISTANCE_LABELS",
        "campaign_sha256": BASE.sha256_file(campaign_path),
        "model_input_structure_manifest_sha256": structure_sha256,
        "contact_manifest_sha256": contact_sha256,
        "distance_label_manifest_sha256": distance_sha256,
        "source_bound_structure_manifest_sha256": campaign[
            "source_bound_structure_manifest_sha256"
        ],
        "development_split_sha256": split_sha256,
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
        "routed_median_metrics": predicted_median,
        "predicted_minus_bound_routed_by_seed": {
            str(seed): delta_by_seed[seed] for seed in BASE.SEEDS
        },
        "predicted_minus_bound_routed_median": delta_median,
        "development_only": True,
        "full_validation_is_checkpoint_selection_data": True,
        "independent_confirmation": False,
        "experimental_bound_geometry_used_as_model_input": False,
        "experimental_bound_geometry_used_for_auxiliary_training_supervision": True,
        "experimental_structure_required_at_inference": False,
        "invalid_run_reused": False,
        "old_sabdab2_test_loaded": False,
        "temporal_holdout_loaded": False,
        "selection_or_tuning_from_this_study": False,
    }
    args.output.write_bytes(BASE.canonical_bytes(payload))
    receipt = {
        "format_version": "sabdab2_predicted_geometry_distance_label_analysis_receipt_v2",
        "status": payload["status"],
        "analysis_sha256": BASE.sha256_file(args.output),
        "campaign_sha256": payload["campaign_sha256"],
        "distance_label_manifest_sha256": distance_sha256,
        "all_six_runs_verified": True,
        "invalid_run_reused": False,
        "test_loaded": False,
        "temporal_holdout_loaded": False,
    }
    args.receipt_output.write_bytes(BASE.canonical_bytes(receipt))
    print(
        json.dumps(
            {
                "status": payload["status"],
                "routed_median_metrics": predicted_median,
                "predicted_minus_bound_routed_median": delta_median,
                "analysis_sha256": receipt["analysis_sha256"],
                "receipt_sha256": BASE.sha256_file(args.receipt_output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
