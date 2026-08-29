#!/usr/bin/env python3
"""Compare two in-flight FrameContact conditions at a matched five-seed epoch."""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import statistics
from pathlib import Path
from typing import Any


SUMMARY_SCRIPT = Path(__file__).with_name("summarize_framecontact_live_array.py")
SPEC = importlib.util.spec_from_file_location("framecontact_live_summary", SUMMARY_SCRIPT)
assert SPEC is not None and SPEC.loader is not None
SUMMARY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUMMARY)


def _paired_uncertainty(deltas: list[float]) -> dict[str, Any]:
    """Return small-sample diagnostics without asymptotic assumptions."""

    if not deltas:
        raise ValueError("paired deltas must be nonempty")
    observed = abs(statistics.fmean(deltas))
    tolerance = 1e-15
    if len(deltas) <= 20:
        sign_flipped = [
            abs(
                statistics.fmean(
                    sign * value for sign, value in zip(signs, deltas)
                )
            )
            for signs in itertools.product((-1.0, 1.0), repeat=len(deltas))
        ]
        exact_p = sum(value >= observed - tolerance for value in sign_flipped) / len(
            sign_flipped
        )
        exact_permutation_count = len(sign_flipped)
    else:
        exact_p = None
        exact_permutation_count = None
    leave_one_out_means = (
        [
            statistics.fmean(deltas[:index] + deltas[index + 1 :])
            for index in range(len(deltas))
        ]
        if len(deltas) > 1
        else []
    )
    return {
        "median_delta": statistics.median(deltas),
        "positive_seed_fraction": sum(value > 0 for value in deltas) / len(deltas),
        "negative_seed_fraction": sum(value < 0 for value in deltas) / len(deltas),
        "zero_seed_fraction": sum(value == 0 for value in deltas) / len(deltas),
        "exact_two_sided_sign_flip_p": exact_p,
        "exact_sign_flip_permutation_count": exact_permutation_count,
        "minimum_attainable_two_sided_p_at_n": 2.0 / (2 ** len(deltas)),
        "leave_one_seed_out_mean_delta_range": (
            None
            if not leave_one_out_means
            else [min(leave_one_out_means), max(leave_one_out_means)]
        ),
    }


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-root", type=Path, required=True)
    for side in ("left", "right"):
        parser.add_argument(f"--{side}-label", required=True)
        parser.add_argument(f"--{side}-log-prefix", required=True)
        parser.add_argument(f"--{side}-job-id", required=True)
        parser.add_argument(f"--{side}-condition-index", type=int, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SUMMARY.SEEDS))
    parser.add_argument(
        "--metrics",
        nargs="+",
        help="Optional metric allowlist; default is every metric shared by both sides.",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _load_condition(
    *,
    log_root: Path,
    log_prefix: str,
    job_id: str,
    condition_index: int,
    seeds: list[int],
) -> tuple[dict[int, dict[int, Any]], dict[int, str]]:
    if condition_index < 0:
        raise ValueError("condition indices must be nonnegative")
    by_seed = {}
    paths = {}
    for seed_index, seed in enumerate(seeds):
        task_index = condition_index * len(seeds) + seed_index
        path = log_root / f"{log_prefix}-{job_id}_{task_index}.out"
        paths[seed] = str(path)
        by_seed[seed] = SUMMARY.read_epoch_records(path)
    return by_seed, paths


def compare_conditions(
    *,
    log_root: Path,
    left_label: str,
    left_log_prefix: str,
    left_job_id: str,
    left_condition_index: int,
    right_label: str,
    right_log_prefix: str,
    right_job_id: str,
    right_condition_index: int,
    seeds: list[int],
    metrics: list[str] | None = None,
) -> dict[str, Any]:
    if not left_label or not right_label or left_label == right_label:
        raise ValueError("left and right labels must be distinct and nonempty")
    if len(set(seeds)) != len(seeds) or not seeds:
        raise ValueError("seeds must be nonempty and unique")
    left, left_paths = _load_condition(
        log_root=log_root,
        log_prefix=left_log_prefix,
        job_id=left_job_id,
        condition_index=left_condition_index,
        seeds=seeds,
    )
    right, right_paths = _load_condition(
        log_root=log_root,
        log_prefix=right_log_prefix,
        job_id=right_job_id,
        condition_index=right_condition_index,
        seeds=seeds,
    )
    common_epochs = set.intersection(
        *(set(left[seed]) & set(right[seed]) for seed in seeds)
    )
    if not common_epochs:
        raise ValueError("no epoch is present for every seed on both sides")
    epoch = max(common_epochs)
    left_metrics = {
        seed: SUMMARY._epoch_record_metrics(left[seed][epoch]) for seed in seeds
    }
    right_metrics = {
        seed: SUMMARY._epoch_record_metrics(right[seed][epoch]) for seed in seeds
    }
    metric_names = set.intersection(
        *(set(left_metrics[seed]) & set(right_metrics[seed]) for seed in seeds)
    )
    if metrics is not None:
        requested = set(metrics)
        missing = sorted(requested - metric_names)
        if missing:
            raise ValueError(f"requested metrics are unavailable: {missing}")
        metric_names = requested
    comparisons = {}
    for name in sorted(metric_names):
        left_values = [left_metrics[seed][name] for seed in seeds]
        right_values = [right_metrics[seed][name] for seed in seeds]
        deltas = [right - left for left, right in zip(left_values, right_values)]
        comparisons[name] = {
            "optimization_direction": SUMMARY._metric_direction(name),
            "left": SUMMARY._mean_sd(left_values),
            "right": SUMMARY._mean_sd(right_values),
            "right_minus_left_paired": SUMMARY._mean_sd(deltas),
            "per_seed_right_minus_left": {
                str(seed): delta for seed, delta in zip(seeds, deltas)
            },
            "paired_small_sample_diagnostics": _paired_uncertainty(deltas),
        }
    return {
        "format_version": "framecontact_live_matched_condition_comparison_v2",
        "claim_boundary": (
            "Matched-epoch paired-seed development train/validation logs only. "
            "This comparison does not choose a checkpoint, load test predictions, "
            "or authorize generalization or benchmark claims. Exact sign-flip "
            "p-values quantify only paired seed consistency and have coarse "
            "resolution at five seeds; they are not a substitute for the frozen "
            "confirmatory analysis."
        ),
        "matched_epoch": epoch,
        "seeds": seeds,
        "delta_definition": f"{right_label} minus {left_label}",
        "left": {
            "label": left_label,
            "job_id": str(left_job_id),
            "log_prefix": left_log_prefix,
            "condition_index": left_condition_index,
            "log_paths_by_seed": {str(seed): left_paths[seed] for seed in seeds},
        },
        "right": {
            "label": right_label,
            "job_id": str(right_job_id),
            "log_prefix": right_log_prefix,
            "condition_index": right_condition_index,
            "log_paths_by_seed": {str(seed): right_paths[seed] for seed in seeds},
        },
        "metrics": comparisons,
    }


def main() -> None:
    arguments = _arguments()
    result = compare_conditions(
        log_root=arguments.log_root,
        left_label=arguments.left_label,
        left_log_prefix=arguments.left_log_prefix,
        left_job_id=arguments.left_job_id,
        left_condition_index=arguments.left_condition_index,
        right_label=arguments.right_label,
        right_log_prefix=arguments.right_log_prefix,
        right_job_id=arguments.right_job_id,
        right_condition_index=arguments.right_condition_index,
        seeds=arguments.seeds,
        metrics=arguments.metrics,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.output is None:
        print(rendered, end="")
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
