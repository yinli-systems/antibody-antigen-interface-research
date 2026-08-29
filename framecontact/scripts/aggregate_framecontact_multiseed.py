#!/usr/bin/env python3
"""Verify and aggregate the five-seed FrameContact development matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from interaction_first.sabdab2_structure import canonical_json_bytes, sha256_file  # noqa: E402


FORMAT = "framecontact_development_training_v1"
SUMMARY_FORMAT = "framecontact_development_multiseed_summary_v1"
DEFAULT_EXPERIMENTS = (
    "sequence_dense",
    "spatial_topology",
    "distance",
    "distance_direction",
    "full_pose",
    "frozen_frame_ipa",
    "framecontact",
    "framecontact_interleaved",
)
DEFAULT_SEEDS = (17, 42, 73, 101, 137)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiments", nargs="+", default=list(DEFAULT_EXPERIMENTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    return parser.parse_args()


def _mean_sd(values: list[float]) -> dict[str, float | int]:
    return {
        "mean": statistics.fmean(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "count": len(values),
    }


def _metrics(result: dict[str, Any]) -> dict[str, float]:
    validation = result["selected_checkpoint_validation"]
    metrics = {
        "cognate_compatibility_margin": float(validation["selection_metric"]),
        "contact_mean_complex_auprc": float(validation["contact"]["mean_per_complex_auprc"]),
        "contact_pooled_auprc": float(validation["contact"]["auprc"]),
        "paratope_mean_complex_auprc": float(validation["paratope"]["mean_per_complex_auprc"]),
        "epitope_mean_complex_auprc": float(validation["epitope"]["mean_per_complex_auprc"]),
    }
    retrieval_by_direction = validation.get("partner_retrieval_by_direction")
    if retrieval_by_direction:
        for direction, retrieval in sorted(retrieval_by_direction.items()):
            metrics[f"{direction}_margin"] = float(
                retrieval["cluster_weighted_cognate_compatibility_margin"]
            )
            metrics[f"{direction}_top1"] = float(
                retrieval["top1_retrieval_accuracy"]
            )
            metrics[f"{direction}_mrr"] = float(retrieval["mean_reciprocal_rank"])
    else:
        retrieval = validation["partner_retrieval"]
        metrics["top1_partner_retrieval"] = float(
            retrieval["top1_retrieval_accuracy"]
        )
        metrics["partner_mrr"] = float(retrieval["mean_reciprocal_rank"])
    return metrics


def main() -> None:
    arguments = _arguments()
    if arguments.output.exists() or arguments.output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite summary: {arguments.output}")
    if tuple(arguments.seeds) != DEFAULT_SEEDS:
        raise ValueError("Aggregator requires the five pre-registered seeds in order")
    if len(set(arguments.experiments)) != len(arguments.experiments):
        raise ValueError("Experiments must be unique")
    results: dict[tuple[str, int], dict[str, Any]] = {}
    common_authority: dict[str, Any] | None = None
    parameter_count_by_experiment: dict[str, int] = {}
    initial_by_experiment_seed: dict[str, dict[int, str]] = {}
    architecture_families = {}
    for experiment in arguments.experiments:
        if experiment == "framecontact_interleaved":
            family = "interleaved"
        elif experiment == "frozen_frame_ipa":
            family = "frozen_frame_ipa"
        else:
            family = "late_pair_mlp"
        architecture_families[experiment] = family
    parameter_count_by_family: dict[str, int] = {}
    initial_by_family_seed: dict[tuple[str, int], str] = {}
    for experiment in arguments.experiments:
        for seed in arguments.seeds:
            run_root = arguments.run_root / experiment / f"seed-{seed}"
            result_path = run_root / "FRAMECONTACT_DEVELOPMENT_RESULT.json"
            checkpoint_path = run_root / "FRAMECONTACT_SELECTED.pt"
            receipt_path = run_root / "RUN_RECEIPT.json"
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if (
                receipt.get("format_version") != FORMAT
                or receipt.get("seed") != seed
                or receipt.get("test_loaded") is not False
                or receipt.get("external_test_loaded") is not False
                or receipt.get("result_sha256") != sha256_file(result_path)
                or receipt.get("selected_checkpoint_sha256") != sha256_file(checkpoint_path)
            ):
                raise ValueError(f"Run receipt validation failed: {experiment}, seed {seed}")
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if (
                result.get("format_version") != FORMAT
                or result.get("experiment") != experiment
                or result.get("configuration", {}).get("seed") != seed
                or result.get("authority", {}).get("test_loaded") is not False
                or result.get("authority", {}).get("external_test_loaded") is not False
            ):
                raise ValueError(f"Result metadata validation failed: {experiment}, seed {seed}")
            authority = result["authority"]
            authority_without_panels = {
                key: value
                for key, value in authority.items()
                if key not in {"training_partner_panel_sha256", "validation_partner_panel_sha256"}
            }
            if common_authority is None:
                common_authority = authority_without_panels
            elif authority_without_panels != common_authority:
                raise ValueError("Source/protocol authorities differ across runs")
            observed_parameters = int(result["model"]["parameter_count"])
            parameter_count_by_experiment[experiment] = observed_parameters
            family = architecture_families[experiment]
            if family in parameter_count_by_family:
                if observed_parameters != parameter_count_by_family[family]:
                    raise ValueError(
                        f"Parameter counts differ within architecture family {family}"
                    )
            else:
                parameter_count_by_family[family] = observed_parameters
            initial = result["model"]["initial_state_sha256"]
            family_seed = (family, seed)
            if (
                family_seed in initial_by_family_seed
                and initial_by_family_seed[family_seed] != initial
            ):
                raise ValueError(
                    f"Initial state differs within architecture family {family}, seed {seed}"
                )
            initial_by_family_seed[family_seed] = initial
            initial_by_experiment_seed.setdefault(experiment, {})[seed] = initial
            results[(experiment, seed)] = result
    experiment_summaries = {}
    for experiment in arguments.experiments:
        per_seed = {
            str(seed): _metrics(results[(experiment, seed)])
            for seed in arguments.seeds
        }
        aggregate = {
            metric: _mean_sd([values[metric] for values in per_seed.values()])
            for metric in next(iter(per_seed.values()))
        }
        experiment_summaries[experiment] = {
            "per_seed": per_seed,
            "five_seed": aggregate,
        }
    dense = experiment_summaries.get("sequence_dense")
    paired_deltas = {}
    if dense is not None:
        for experiment, summary in experiment_summaries.items():
            if experiment == "sequence_dense":
                continue
            per_seed_delta = {
                str(seed): {
                    metric: summary["per_seed"][str(seed)][metric]
                    - dense["per_seed"][str(seed)][metric]
                    for metric in summary["per_seed"][str(seed)]
                }
                for seed in arguments.seeds
            }
            paired_deltas[experiment] = {
                "per_seed": per_seed_delta,
                "five_seed": {
                    metric: _mean_sd(
                        [per_seed_delta[str(seed)][metric] for seed in arguments.seeds]
                    )
                    for metric in next(iter(per_seed_delta.values()))
                },
            }
    payload = {
        "format_version": SUMMARY_FORMAT,
        "claim_boundary": (
            "Five-seed development train/validation aggregation only. No test, external, "
            "generalization, or benchmark-improvement claim."
        ),
        "matrix_complete": True,
        "external_opening_ready": False,
        "external_opening_blockers": [
            "new antigen-fold test authority and passing structural audit",
            "frozen five-seed ensemble prediction rule",
            "receipt-gated one-time external scorer",
            "canonical CHIMERA co-design adapter and frozen evaluation protocol",
        ],
        "authority": common_authority,
        "architecture_family_by_experiment": architecture_families,
        "model_parameter_count_by_experiment": parameter_count_by_experiment,
        "initial_state_sha256_by_experiment_and_seed": {
            experiment: {
                str(seed): value for seed, value in by_seed.items()
            }
            for experiment, by_seed in initial_by_experiment_seed.items()
        },
        "specificity_score_sources": sorted(
            {
                result["selected_checkpoint_validation"].get(
                    "specificity_score_source", "legacy_total_contact"
                )
                for result in results.values()
            }
        ),
        "experiments": experiment_summaries,
        "paired_delta_vs_sequence_dense": paired_deltas,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_bytes(canonical_json_bytes(payload))
    print(
        json.dumps(
            {
                "output": str(arguments.output),
                "sha256": sha256_file(arguments.output),
                "runs": len(results),
                "external_opening_ready": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
