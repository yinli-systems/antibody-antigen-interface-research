#!/usr/bin/env python3
"""Verify and analyze the frozen 18-run FrameContact development study."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from statistics import median
from typing import Any


REFERENCE_ANALYSIS_SHA256 = (
    "1382c6fc18624b53f50657c0c0d61408fa03f0acc5aa3a812ffa3f84bd23b78c"
)
SEEDS = (17, 29, 43)
TASKS = ("contact", "epitope", "paratope")
INTERVENTIONS = ("surface", "hard_negative", "combined")
CHECKPOINT_ARMS = ("baseline", "edge_topk2_supervised")


def load_base(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("predgeom_verifier", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def finite(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite metric: {value!r}")
    return result


def median_metrics(by_seed: dict[int, dict[str, float]]) -> dict[str, float]:
    return {
        task: float(median(by_seed[seed][task] for seed in SEEDS)) for task in TASKS
    }


def delta_metrics(
    values: dict[int, dict[str, float]],
    reference: dict[int, dict[str, float]],
) -> dict[int, dict[str, float]]:
    return {
        seed: {task: values[seed][task] - reference[seed][task] for task in TASKS}
        for seed in SEEDS
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--base-analyzer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.output, args.receipt_output):
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)

    base = load_base(args.base_analyzer)
    authority = args.campaign_root / "authority"
    campaign_path = authority / "CAMPAIGN_SURFACE_HARDNEGATIVE_V1.json"
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    if (
        campaign.get("status")
        != "FROZEN_BEFORE_SURFACE_HARDNEGATIVE_TRAINING_V1"
        or campaign.get("seeds") != list(SEEDS)
        or campaign.get("distance_labels_loaded") is not False
        or campaign.get("external_sable_read_or_evaluated_by_this_study") is not False
        or campaign.get("old_sabdab2_test_loaded") is not False
        or campaign.get("temporal_holdout_loaded") is not False
        or campaign.get("independent_confirmation") is not False
    ):
        raise ValueError("campaign authority differs")
    for raw_line in (authority / "AUTHORITY_SHA256SUMS.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        expected, raw_path = raw_line.split(maxsplit=1)
        path = Path(raw_path.strip())
        if base.sha256_file(path) != expected:
            raise ValueError(f"authority checksum differs: {path}")

    reference_path = (
        args.reference_root / "PREDICTED_GEOMETRY_NO_PRIVILEGED_DISTANCE_ANALYSIS_V1.json"
    )
    if base.sha256_file(reference_path) != REFERENCE_ANALYSIS_SHA256:
        raise ValueError("no-distance reference analysis differs")
    reference_payload = json.loads(reference_path.read_text(encoding="utf-8"))
    reference_by_seed = {
        int(seed): {task: finite(value) for task, value in metrics.items()}
        for seed, metrics in reference_payload["routed_metrics_by_seed"].items()
    }

    structure_sha = campaign["model_input_structure_manifest_sha256"]
    contact_sha = campaign["contact_manifest_sha256"]
    split_sha = campaign["development_split_sha256"]
    all_metrics: dict[str, dict[str, dict[int, dict[str, float]]]] = {}
    all_bindings: dict[str, dict[str, dict[str, Any]]] = {}
    for intervention in INTERVENTIONS:
        all_metrics[intervention] = {}
        all_bindings[intervention] = {}
        settings = campaign["interventions"][intervention]
        for checkpoint_arm in CHECKPOINT_ARMS:
            key = f"{checkpoint_arm}:{intervention}"
            protocol = campaign["protocols"][key]
            arm_metrics: dict[int, dict[str, float]] = {}
            arm_bindings: dict[str, Any] = {}
            for seed in SEEDS:
                run_root = (
                    args.campaign_root
                    / "runs"
                    / intervention
                    / checkpoint_arm
                    / f"seed-{seed}"
                )
                metrics, binding = base.verify_run(
                    run_root,
                    seed=seed,
                    protocol_sha256=protocol["protocol_sha256"],
                    protocol_receipt_sha256=protocol["receipt_sha256"],
                    source_sha256=protocol["source_manifest_sha256"],
                    split_sha256=split_sha,
                    structure_sha256=structure_sha,
                )
                result = json.loads(
                    (run_root / "FRAMECONTACT_DEVELOPMENT_RESULT.json").read_text(
                        encoding="utf-8"
                    )
                )
                configuration = result["configuration"]
                authority_row = result["authority"]
                history = result["history"]
                if (
                    authority_row.get("contact_manifest_sha256") != contact_sha
                    or authority_row.get("distance_label_manifest_sha256") is not None
                    or authority_row.get("test_loaded") is not False
                    or authority_row.get("external_test_loaded") is not False
                    or float(configuration.get("distance_loss_weight", -1)) != 0.0
                    or float(configuration.get("contrastive_loss_weight", -1)) != 0.0
                    or configuration.get("surface_prior_mode")
                    != settings["surface_prior_mode"]
                    or float(configuration.get("surface_radius", -1))
                    != float(settings["surface_radius_angstrom"])
                    or float(
                        configuration.get("contact_hard_negative_loss_weight", -1)
                    )
                    != float(settings["contact_hard_negative"])
                    or float(configuration.get("contact_hard_negative_fraction", -1))
                    != 0.01
                    or int(
                        configuration.get("contact_hard_negative_maximum_pairs", -1)
                    )
                    != 64
                    or float(configuration.get("contact_hard_negative_margin", -1))
                    != 0.5
                ):
                    raise ValueError(f"run authority/configuration differs: {run_root}")
                for row in history:
                    train = row["train"]
                    for name, value in train.items():
                        if name.endswith("loss"):
                            finite(value)
                    if float(train.get("distance_loss", 0.0)) != 0.0:
                        raise ValueError(f"distance loss was active: {run_root}")
                    hard_loss = float(train.get("contact_hard_negative_loss", 0.0))
                    if settings["contact_hard_negative"] == 0.0 and hard_loss != 0.0:
                        raise ValueError(f"disabled hard-negative loss nonzero: {run_root}")
                arm_metrics[seed] = {task: finite(metrics[task]) for task in TASKS}
                arm_bindings[str(seed)] = binding
            all_metrics[intervention][checkpoint_arm] = arm_metrics
            all_bindings[intervention][checkpoint_arm] = arm_bindings

    routed_by_intervention: dict[str, dict[int, dict[str, float]]] = {}
    deltas_by_intervention: dict[str, dict[int, dict[str, float]]] = {}
    median_by_intervention: dict[str, dict[str, float]] = {}
    median_delta_by_intervention: dict[str, dict[str, float]] = {}
    gate_results: dict[str, dict[str, bool]] = {}
    for intervention in INTERVENTIONS:
        routed = {
            seed: {
                "contact": all_metrics[intervention]["baseline"][seed]["contact"],
                "epitope": all_metrics[intervention]["edge_topk2_supervised"][seed][
                    "epitope"
                ],
                "paratope": all_metrics[intervention]["baseline"][seed]["paratope"],
            }
            for seed in SEEDS
        }
        deltas = delta_metrics(routed, reference_by_seed)
        routed_by_intervention[intervention] = routed
        deltas_by_intervention[intervention] = deltas
        median_by_intervention[intervention] = median_metrics(routed)
        delta_median = median_metrics(deltas)
        median_delta_by_intervention[intervention] = delta_median
        contact_gate = campaign["promotion_gates"]["contact_route"]
        epitope_gate = campaign["promotion_gates"]["epitope_route"]
        gate_results[intervention] = {
            "contact_route": (
                delta_median["contact"]
                >= float(contact_gate["contact_median_delta_minimum"])
                and delta_median["epitope"]
                >= float(contact_gate["epitope_median_delta_minimum"])
                and delta_median["paratope"]
                >= float(contact_gate["paratope_median_delta_minimum"])
            ),
            "epitope_route": (
                delta_median["contact"]
                >= float(epitope_gate["contact_median_delta_minimum"])
                and delta_median["epitope"]
                >= float(epitope_gate["epitope_median_delta_minimum"])
                and delta_median["paratope"]
                >= float(epitope_gate["paratope_median_delta_minimum"])
            ),
        }
        gate_results[intervention]["eligible_for_new_confirmation"] = any(
            gate_results[intervention].values()
        )

    priority = campaign["promotion_gates"]["development_selection_priority"]
    eligible = [
        arm for arm in priority if gate_results[arm]["eligible_for_new_confirmation"]
    ]
    selected = eligible[0] if eligible else None
    status = (
        "PASS_DEVELOPMENT_CANDIDATE_FOR_NEW_PROSPECTIVE_CONFIRMATION"
        if selected is not None
        else "FAIL_DEVELOPMENT_PROMOTION_GATES"
    )
    payload = {
        "format_version": "framecontact_surface_hardnegative_analysis_v1",
        "status": status,
        "campaign_sha256": base.sha256_file(campaign_path),
        "reference_analysis_sha256": REFERENCE_ANALYSIS_SHA256,
        "run_bindings": all_bindings,
        "metrics_by_intervention_checkpoint_arm_seed": {
            intervention: {
                arm: {str(seed): metrics for seed, metrics in by_seed.items()}
                for arm, by_seed in by_arm.items()
            }
            for intervention, by_arm in all_metrics.items()
        },
        "routed_metrics_by_intervention_seed": {
            intervention: {str(seed): metrics for seed, metrics in by_seed.items()}
            for intervention, by_seed in routed_by_intervention.items()
        },
        "routed_median_metrics_by_intervention": median_by_intervention,
        "routed_delta_vs_no_distance_by_intervention_seed": {
            intervention: {str(seed): metrics for seed, metrics in by_seed.items()}
            for intervention, by_seed in deltas_by_intervention.items()
        },
        "routed_median_delta_vs_no_distance_by_intervention": (
            median_delta_by_intervention
        ),
        "promotion_gate_results": gate_results,
        "development_selected_intervention": selected,
        "development_selection_priority": priority,
        "all_eighteen_runs_verified": True,
        "development_only": True,
        "full_validation_is_checkpoint_selection_data": True,
        "independent_confirmation": False,
        "external_sable_v0_2_0_was_exposed_before_design": True,
        "external_sable_read_or_evaluated_by_this_study": False,
        "requires_new_prospective_external_confirmation": True,
        "distance_labels_loaded": False,
        "experimental_bound_geometry_used_as_model_input": False,
        "experimental_bound_geometry_used_for_auxiliary_training_supervision": False,
        "experimental_bound_structure_used_to_define_task_contact_labels": True,
        "old_sabdab2_test_loaded": False,
        "temporal_holdout_loaded": False,
        "test_loaded": False,
    }
    args.output.write_bytes(base.canonical_bytes(payload))
    receipt = {
        "format_version": "framecontact_surface_hardnegative_analysis_receipt_v1",
        "status": status,
        "analysis_sha256": base.sha256_file(args.output),
        "campaign_sha256": payload["campaign_sha256"],
        "all_eighteen_runs_verified": True,
        "sable_v0_2_0_accessed": False,
        "test_loaded": False,
        "temporal_holdout_loaded": False,
    }
    args.receipt_output.write_bytes(base.canonical_bytes(receipt))
    print(
        json.dumps(
            {
                "status": status,
                "selected": selected,
                "median_metrics": median_by_intervention,
                "median_deltas": median_delta_by_intervention,
                "analysis_sha256": receipt["analysis_sha256"],
                "receipt_sha256": base.sha256_file(args.receipt_output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
