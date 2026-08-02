#!/usr/bin/env python3
"""Summarize in-flight FrameContact Slurm array logs without touching test data."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping


SEEDS = (17, 42, 73, 101, 137)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--log-prefix", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--conditions", nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument(
        "--minimum-seeds",
        type=int,
        default=None,
        help="Only publish an epoch after this many seeds have logged it (default: all).",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help=(
            "Emit only per-seed epoch coverage, the latest synchronized epoch, "
            "and descriptive trajectory diagnostics."
        ),
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _mean_sd(values: Iterable[float]) -> dict[str, float | int]:
    numbers = [float(value) for value in values]
    if not numbers:
        raise ValueError("Cannot summarize an empty value sequence")
    return {
        "n": len(numbers),
        "mean": statistics.fmean(numbers),
        "sample_sd": statistics.stdev(numbers) if len(numbers) > 1 else 0.0,
        "minimum": min(numbers),
        "maximum": max(numbers),
    }


def _epoch_record_metrics(record: Mapping[str, Any]) -> dict[str, float]:
    validation = record["validation"]
    contact = validation["contact"]
    fixed = contact["fixed_raw_logit_threshold_zero"]
    train = record["train"]
    metrics = {
        "elapsed_minutes": float(record["elapsed_seconds"]) / 60.0,
        "train_loss": float(train["loss"]),
        "train_contact_loss": float(train["contact_loss"]),
        "train_distogram_loss": float(train.get("distogram_loss", 0.0)),
        "train_triangle_loss": float(train.get("triangle_loss", 0.0)),
        "train_edm_loss": float(train.get("edm_loss", 0.0)),
        "contact_mean_per_complex_auprc": float(contact["mean_per_complex_auprc"]),
        "contact_pooled_auprc": float(contact["auprc"]),
        "contact_pooled_brier": float(contact["brier"]),
        "contact_pooled_ece_15": float(contact["ece_15"]),
        "contact_selected_threshold": float(contact["threshold_selected_on_validation"]),
        "contact_selected_precision": float(contact["precision"]),
        "contact_selected_recall": float(contact["recall"]),
        "contact_selected_f1": float(contact["f1"]),
        "contact_selected_mcc": float(contact["mcc"]),
        "contact_selected_predicted_positive_ratio": float(
            contact["predicted_positive_ratio"]
        ),
        "contact_fixed_raw_precision": float(fixed["precision"]),
        "contact_fixed_raw_recall": float(fixed["recall"]),
        "contact_fixed_raw_f1": float(fixed["f1"]),
        "contact_fixed_raw_mcc": float(fixed["mcc"]),
        "contact_fixed_raw_predicted_positive_ratio": float(
            fixed["predicted_positive_ratio"]
        ),
        "selection_margin": float(validation["selection_metric"]),
    }
    for residue_name in ("paratope", "epitope"):
        residue = validation[residue_name]
        for source_name, target_name in (
            ("auprc", "pooled_auprc"),
            ("mean_per_complex_auprc", "mean_per_complex_auprc"),
            ("precision", "precision"),
            ("recall", "recall"),
            ("f1", "f1"),
            ("mcc", "mcc"),
            ("brier", "brier"),
            ("ece_15", "ece_15"),
            ("predicted_positive_ratio", "predicted_positive_ratio"),
        ):
            value = residue.get(source_name)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                metrics[f"{residue_name}_{target_name}"] = float(value)

    for cutoff, ranked in sorted(contact.get("ranked_pair_macro", {}).items()):
        for source_name, target_name in (
            ("mean_precision", "precision"),
            ("mean_recall", "recall"),
            ("mean_effective_k", "effective_k"),
        ):
            value = ranked.get(source_name)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                metrics[f"contact_ranked_{cutoff}_{target_name}"] = float(value)

    directional_margins = []
    directional_top1 = []
    directional_mrr = []
    for direction, retrieval in sorted(
        validation.get("partner_retrieval_by_direction", {}).items()
    ):
        margin = float(retrieval["cluster_weighted_cognate_compatibility_margin"])
        top1 = float(retrieval["top1_retrieval_accuracy"])
        mrr = float(retrieval["mean_reciprocal_rank"])
        metrics[f"retrieval_{direction}_margin"] = margin
        metrics[f"retrieval_{direction}_top1"] = top1
        metrics[f"retrieval_{direction}_mrr"] = mrr
        directional_margins.append(margin)
        directional_top1.append(top1)
        directional_mrr.append(mrr)
    if directional_margins:
        metrics.update(
            {
                "retrieval_mean_directional_margin": statistics.fmean(
                    directional_margins
                ),
                "retrieval_worst_directional_margin": min(directional_margins),
                "retrieval_mean_directional_top1": statistics.fmean(directional_top1),
                "retrieval_worst_directional_top1": min(directional_top1),
                "retrieval_mean_directional_mrr": statistics.fmean(directional_mrr),
                "retrieval_worst_directional_mrr": min(directional_mrr),
            }
        )
    return metrics


def _metric_direction(metric_name: str) -> str:
    if metric_name == "elapsed_minutes" or metric_name.endswith(
        ("_threshold", "_effective_k")
    ):
        return "diagnostic_only"
    if metric_name.startswith("train_") or metric_name.endswith(
        ("_brier", "_ece_15")
    ):
        return "lower_is_better"
    return "higher_is_better"


def _linear_slope(points: list[tuple[int, float]]) -> float:
    if len(points) < 2:
        return 0.0
    x_mean = statistics.fmean(epoch for epoch, _ in points)
    y_mean = statistics.fmean(value for _, value in points)
    denominator = sum((epoch - x_mean) ** 2 for epoch, _ in points)
    if denominator == 0:
        return 0.0
    return sum(
        (epoch - x_mean) * (value - y_mean) for epoch, value in points
    ) / denominator


def _trajectory_diagnostics(trajectories: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Describe synchronized multi-seed trajectories without selecting a model."""

    if not trajectories:
        return {}
    metric_names = sorted(
        set.intersection(
            *(set(entry["metrics"]) for entry in trajectories)
        )
    )
    diagnostics: dict[str, Any] = {}
    for metric_name in metric_names:
        points = [
            (int(entry["epoch"]), float(entry["metrics"][metric_name]["mean"]))
            for entry in trajectories
        ]
        direction = _metric_direction(metric_name)
        transitions = [right[1] - left[1] for left, right in zip(points, points[1:])]
        if direction == "higher_is_better":
            improving = [delta > 0 for delta in transitions]
            best_epoch, best_value = max(points, key=lambda item: (item[1], -item[0]))
        elif direction == "lower_is_better":
            improving = [delta < 0 for delta in transitions]
            best_epoch, best_value = min(points, key=lambda item: (item[1], item[0]))
        else:
            improving = []
            best_epoch = None
            best_value = None
        recent_points = points[-min(5, len(points)) :]
        diagnostics[metric_name] = {
            "optimization_direction": direction,
            "epoch_count": len(points),
            "first_epoch": points[0][0],
            "first_mean": points[0][1],
            "last_epoch": points[-1][0],
            "last_mean": points[-1][1],
            "last_minus_first": points[-1][1] - points[0][1],
            "least_squares_slope_per_epoch": _linear_slope(points),
            "recent_window_epoch_count": len(recent_points),
            "recent_window_slope_per_epoch": _linear_slope(recent_points),
            "improving_transition_fraction": (
                statistics.fmean(improving) if improving else None
            ),
            "best_epoch_descriptive_only": best_epoch,
            "best_mean_descriptive_only": best_value,
        }
    return diagnostics


def read_epoch_records(path: Path) -> dict[int, Mapping[str, Any]]:
    records: dict[int, Mapping[str, Any]] = {}
    if not path.is_file():
        return records
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
            if "epoch" not in record:
                continue
            epoch = int(record["epoch"])
            if epoch in records:
                raise ValueError(f"Duplicate epoch {epoch} in {path}")
            records[epoch] = record
    return records


def summarize_array(
    *,
    log_root: Path,
    log_prefix: str,
    job_id: str,
    conditions: list[str],
    seeds: list[int],
    minimum_seeds: int,
) -> dict[str, Any]:
    if len(set(conditions)) != len(conditions):
        raise ValueError("Conditions must be unique")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be unique")
    if minimum_seeds < 1 or minimum_seeds > len(seeds):
        raise ValueError("minimum_seeds must be between one and the seed count")

    condition_payload: dict[str, Any] = {}
    for condition_index, condition in enumerate(conditions):
        by_seed: dict[int, dict[int, Mapping[str, Any]]] = {}
        paths: dict[int, str] = {}
        for seed_index, seed in enumerate(seeds):
            task_index = condition_index * len(seeds) + seed_index
            path = log_root / f"{log_prefix}-{job_id}_{task_index}.out"
            paths[seed] = str(path)
            by_seed[seed] = read_epoch_records(path)

        observed_epochs = sorted(
            {epoch for seed_records in by_seed.values() for epoch in seed_records}
        )
        trajectories = []
        for epoch in observed_epochs:
            available = {
                seed: records[epoch]
                for seed, records in by_seed.items()
                if epoch in records
            }
            if len(available) < minimum_seeds:
                continue
            per_seed = {
                str(seed): _epoch_record_metrics(record)
                for seed, record in available.items()
            }
            metric_names = sorted(
                set.intersection(*(set(metrics) for metrics in per_seed.values()))
            )
            trajectories.append(
                {
                    "epoch": epoch,
                    "seed_count": len(per_seed),
                    "seeds": sorted(int(seed) for seed in per_seed),
                    "metrics": {
                        name: _mean_sd(metrics[name] for metrics in per_seed.values())
                        for name in metric_names
                    },
                    "per_seed": per_seed,
                }
            )

        condition_payload[condition] = {
            "log_paths_by_seed": {str(key): value for key, value in paths.items()},
            "epochs_logged_by_seed": {
                str(seed): sorted(records) for seed, records in by_seed.items()
            },
            "complete_epoch_trajectory": trajectories,
            "trajectory_diagnostics": _trajectory_diagnostics(trajectories),
        }

    return {
        "format_version": "framecontact_live_array_summary_v2",
        "claim_boundary": (
            "In-flight development train/validation logs only. This summary does not "
            "load test predictions, choose a checkpoint, or authorize generalization "
            "or benchmark claims. Trend and best-epoch fields are descriptive only."
        ),
        "job_id": str(job_id),
        "log_prefix": log_prefix,
        "seeds": seeds,
        "minimum_seeds_per_epoch": minimum_seeds,
        "conditions": condition_payload,
    }


def compact_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Return a bounded live-monitoring view without changing metric semantics."""

    conditions = {}
    for condition, payload in summary["conditions"].items():
        trajectory = payload["complete_epoch_trajectory"]
        latest = trajectory[-1] if trajectory else None
        conditions[condition] = {
            "epochs_logged_by_seed": payload["epochs_logged_by_seed"],
            "latest_synchronized_epoch": (
                None
                if latest is None
                else {
                    "epoch": latest["epoch"],
                    "seed_count": latest["seed_count"],
                    "seeds": latest["seeds"],
                    "metrics": latest["metrics"],
                }
            ),
            "trajectory_diagnostics": payload["trajectory_diagnostics"],
        }
    return {
        key: summary[key]
        for key in (
            "format_version",
            "claim_boundary",
            "job_id",
            "log_prefix",
            "seeds",
            "minimum_seeds_per_epoch",
        )
    } | {"conditions": conditions}


def main() -> None:
    arguments = parse_arguments()
    minimum_seeds = arguments.minimum_seeds or len(arguments.seeds)
    summary = summarize_array(
        log_root=arguments.log_root,
        log_prefix=arguments.log_prefix,
        job_id=arguments.job_id,
        conditions=arguments.conditions,
        seeds=arguments.seeds,
        minimum_seeds=minimum_seeds,
    )
    if arguments.compact:
        summary = compact_summary(summary)
    rendered = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if arguments.output is None:
        print(rendered, end="")
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
