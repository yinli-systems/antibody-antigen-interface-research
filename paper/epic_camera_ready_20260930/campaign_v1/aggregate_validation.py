#!/usr/bin/env python3
"""Aggregate the frozen 12-condition x 3-seed EPIC validation campaign."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import socket
import time
from typing import Any, Mapping, Sequence

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def mean_sd(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    require(array.size > 0 and np.isfinite(array).all(), "Invalid numeric summary")
    return {
        "values": [float(value) for value in array],
        "mean": float(array.mean()),
        "sample_sd": float(array.std(ddof=1)) if array.size > 1 else None,
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "count": int(array.size),
    }


def paired_scalar(condition_values: Sequence[float], reference_values: Sequence[float],
                  *, higher_is_better: bool) -> dict[str, Any]:
    condition = np.asarray(condition_values, dtype=np.float64)
    reference = np.asarray(reference_values, dtype=np.float64)
    require(condition.shape == reference.shape and condition.size > 0,
            "Paired scalar shapes differ")
    deltas = condition - reference
    same_positive = int((deltas > 0).sum())
    same_negative = int((deltas < 0).sum())
    if same_positive == deltas.size:
        direction = "condition_higher_all_seeds"
    elif same_negative == deltas.size:
        direction = "condition_lower_all_seeds"
    elif bool((deltas == 0).all()):
        direction = "exact_tie_all_seeds"
    else:
        direction = "heterogeneous_across_seeds"
    preferred = deltas if higher_is_better else -deltas
    return {
        "condition_minus_reference_by_seed": [float(value) for value in deltas],
        "condition_minus_reference_mean": float(deltas.mean()),
        "condition_minus_reference_sample_sd": float(deltas.std(ddof=1)) if deltas.size > 1 else None,
        "raw_direction": direction,
        "higher_is_better": higher_is_better,
        "condition_preferred_seed_count": int((preferred > 0).sum()),
        "reference_preferred_seed_count": int((preferred < 0).sum()),
        "tie_seed_count": int((preferred == 0).sum()),
    }


def bootstrap_seed_complex_delta(
    condition_by_seed: np.ndarray,
    reference_by_seed: np.ndarray,
    *,
    replicates: int,
    random_seed: int,
    higher_is_better: bool,
) -> dict[str, Any]:
    require(condition_by_seed.shape == reference_by_seed.shape,
            "Bootstrap arrays differ")
    require(condition_by_seed.ndim == 2 and condition_by_seed.shape[0] >= 2 and
            condition_by_seed.shape[1] >= 2, "Bootstrap array is too small")
    delta = condition_by_seed - reference_by_seed
    require(np.isfinite(delta).all(), "Bootstrap delta contains nonfinite values")
    rng = np.random.default_rng(random_seed)
    seed_count, complex_count = delta.shape
    samples = np.empty(replicates, dtype=np.float64)
    offset = 0
    chunk_size = 500
    while offset < replicates:
        current = min(chunk_size, replicates - offset)
        complex_indices = rng.integers(0, complex_count, size=(current, complex_count))
        per_seed_means = np.stack(
            [delta[seed_index][complex_indices].mean(axis=1)
             for seed_index in range(seed_count)],
            axis=1,
        )
        seed_indices = rng.integers(0, seed_count, size=(current, seed_count))
        samples[offset:offset + current] = np.take_along_axis(
            per_seed_means, seed_indices, axis=1
        ).mean(axis=1)
        offset += current
    low, high = np.quantile(samples, [0.025, 0.975])
    seed_deltas = delta.mean(axis=1)
    if low > 0 and bool((seed_deltas > 0).all()):
        robust_raw_direction = "condition_higher"
    elif high < 0 and bool((seed_deltas < 0).all()):
        robust_raw_direction = "condition_lower"
    else:
        robust_raw_direction = "inconclusive_or_seed_heterogeneous"
    preferred_samples = samples if higher_is_better else -samples
    return {
        "condition_minus_reference_observed": float(delta.mean()),
        "condition_minus_reference_by_seed": [float(value) for value in seed_deltas],
        "ci95_seed_and_complex_bootstrap": [float(low), float(high)],
        "bootstrap_probability_condition_minus_reference_gt_zero": float((samples > 0).mean()),
        "bootstrap_probability_condition_preferred": float((preferred_samples > 0).mean()),
        "replicates": replicates,
        "seed_count": seed_count,
        "complex_count": complex_count,
        "higher_is_better": higher_is_better,
        "robust_raw_direction": robust_raw_direction,
    }


def format_mean_sd(summary: Mapping[str, Any], digits: int = 5) -> str:
    mean = float(summary["mean"])
    sd = summary["sample_sd"]
    if sd is None:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {float(sd):.{digits}f}"


def format_bootstrap(summary: Mapping[str, Any], digits: int = 5) -> str:
    value = float(summary["condition_minus_reference_observed"])
    low, high = summary["ci95_seed_and_complex_bootstrap"]
    return f"{value:+.{digits}f} [{float(low):+.{digits}f}, {float(high):+.{digits}f}]"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-protocol", required=True, type=Path)
    parser.add_argument("--campaign-protocol-sha256", required=True)
    parser.add_argument("--analysis-protocol", required=True, type=Path)
    parser.add_argument("--analysis-protocol-sha256", required=True)
    parser.add_argument("--training-manifest", required=True, type=Path)
    parser.add_argument("--training-manifest-sha256", required=True)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-markdown", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    arguments = parser.parse_args()
    started = time.monotonic()

    campaign_path = arguments.campaign_protocol.resolve()
    analysis_path = arguments.analysis_protocol.resolve()
    manifest_path = arguments.training_manifest.resolve()
    require(sha256_file(campaign_path) == arguments.campaign_protocol_sha256,
            "Campaign protocol hash differs")
    require(sha256_file(analysis_path) == arguments.analysis_protocol_sha256,
            "Analysis protocol hash differs")
    require(sha256_file(manifest_path) == arguments.training_manifest_sha256,
            "Training manifest hash differs")
    campaign = read_json(campaign_path)
    protocol = read_json(analysis_path)
    manifest = read_json(manifest_path)
    require(protocol["campaign_protocol_sha256"] == arguments.campaign_protocol_sha256,
            "Analysis protocol campaign differs")
    require(sha256_file(Path(__file__).resolve()) == protocol["implementation"]["aggregate_validation.py"],
            "Validation aggregator differs from preregistered implementation")
    require(manifest["status"] == "PASS_ALL_TRAINING_CHECKPOINTS_FROZEN_BEFORE_TEST",
            "Training checkpoints were not frozen before analysis")
    require(manifest["campaign_protocol_sha256"] == arguments.campaign_protocol_sha256,
            "Training manifest campaign differs")
    require(manifest["analysis_protocol_sha256"] == arguments.analysis_protocol_sha256,
            "Training manifest analysis protocol differs")
    require(manifest.get("test_label_artifacts_opened") is False and
            manifest.get("test_selection_performed") is False,
            "Training manifest reports test access or selection")

    conditions = [entry["name"] for entry in campaign["conditions"]]
    condition_descriptions = {entry["name"]: entry["description"]
                              for entry in campaign["conditions"]}
    seeds = [int(seed) for seed in campaign["seeds"]]
    reference = str(protocol["reference_condition"])
    require(reference in conditions, "Reference condition is absent")
    require(manifest["record_count"] == len(conditions) * len(seeds),
            "Training manifest record count differs")

    records: dict[tuple[str, int], dict[str, Any]] = {}
    validations: dict[tuple[str, int], dict[str, Any]] = {}
    results: dict[tuple[str, int], dict[str, Any]] = {}
    for record in manifest["records"]:
        key = (str(record["condition"]), int(record["seed"]))
        require(key not in records, f"Duplicate training manifest record: {key}")
        result_path = Path(record["result_path"])
        validation_path = Path(record["validation_path"])
        require(sha256_file(result_path) == record["result_sha256"],
                f"Result changed after freeze: {key}")
        require(sha256_file(validation_path) == record["validation_sha256"],
                f"Validation changed after freeze: {key}")
        records[key] = record
        results[key] = read_json(result_path)
        validations[key] = read_json(validation_path)
    require(set(records) == {(condition, seed) for condition in conditions for seed in seeds},
            "Training manifest key set differs")

    task_names = ("contact", "epitope", "paratope")
    scalar_metrics: dict[str, bool] = {
        "mpc_ap": True,
        "mpc_auroc": True,
        "fixed_05_precision": True,
        "fixed_05_recall": True,
        "brier": False,
        "ece_20bin": False,
    }
    per_complex_fields: dict[str, tuple[str, bool]] = {
        "mpc_ap": ("average_precision", True),
        "mpc_auroc": ("auroc", True),
        "fixed_05_precision": ("precision_at_05", True),
        "fixed_05_recall": ("recall_at_05", True),
    }
    condition_summaries: dict[str, Any] = {}
    for condition in conditions:
        condition_summary: dict[str, Any] = {
            "description": condition_descriptions[condition],
            "tasks": {},
            "contact_ranked_cutoffs": {},
            "best_epoch": mean_sd([results[(condition, seed)]["best_epoch"] for seed in seeds]),
            "executed_epochs": mean_sd([len(results[(condition, seed)]["trace"]) for seed in seeds]),
            "wall_seconds": mean_sd([results[(condition, seed)]["wall_seconds"] for seed in seeds]),
            "total_parameters": mean_sd([results[(condition, seed)]["total_parameters"] for seed in seeds]),
            "trainable_parameters": mean_sd([results[(condition, seed)]["trainable_parameters"] for seed in seeds]),
        }
        for task in task_names:
            condition_summary["tasks"][task] = {
                metric: mean_sd([
                    validations[(condition, seed)]["metrics"][task][metric]
                    for seed in seeds
                ])
                for metric in scalar_metrics
            }
        cutoff_names = sorted(
            validations[(condition, seeds[0])]["metrics"]["contact_ranked_cutoffs"]
        )
        for cutoff in cutoff_names:
            condition_summary["contact_ranked_cutoffs"][cutoff] = {
                metric: mean_sd([
                    validations[(condition, seed)]["metrics"]["contact_ranked_cutoffs"][cutoff][metric]
                    for seed in seeds
                ])
                for metric in ("mean_precision", "mean_recall", "mean_enrichment_over_prevalence")
            }
        condition_summaries[condition] = condition_summary

    comparisons: dict[str, Any] = {}
    replicates = int(protocol["bootstrap"]["replicates"])
    for condition in conditions:
        if condition == reference:
            continue
        comparison: dict[str, Any] = {
            "condition_description": condition_descriptions[condition],
            "reference_condition": reference,
            "tasks": {},
            "contact_ranked_cutoffs": {},
            "training": {},
        }
        for task in task_names:
            task_comparison: dict[str, Any] = {}
            for metric, higher_is_better in scalar_metrics.items():
                condition_values = [
                    validations[(condition, seed)]["metrics"][task][metric] for seed in seeds
                ]
                reference_values = [
                    validations[(reference, seed)]["metrics"][task][metric] for seed in seeds
                ]
                scalar = paired_scalar(condition_values, reference_values,
                                       higher_is_better=higher_is_better)
                if metric in per_complex_fields:
                    row_field, field_higher_is_better = per_complex_fields[metric]
                    condition_arrays: list[np.ndarray] = []
                    reference_arrays: list[np.ndarray] = []
                    common_ids_reference: list[str] | None = None
                    for seed in seeds:
                        condition_rows = {
                            str(row["instance_id"]): row[row_field]
                            for row in validations[(condition, seed)]["metrics"][task]["per_complex"]
                        }
                        reference_rows = {
                            str(row["instance_id"]): row[row_field]
                            for row in validations[(reference, seed)]["metrics"][task]["per_complex"]
                        }
                        common_ids = sorted(
                            instance_id for instance_id in condition_rows.keys() & reference_rows.keys()
                            if condition_rows[instance_id] is not None and
                            reference_rows[instance_id] is not None
                        )
                        require(len(common_ids) >= 2,
                                f"Too few paired complexes: {condition}/{task}/{metric}/{seed}")
                        if common_ids_reference is None:
                            common_ids_reference = common_ids
                        else:
                            require(common_ids == common_ids_reference,
                                    f"Paired complex authority differs by seed: {condition}/{task}/{metric}")
                        condition_arrays.append(np.asarray(
                            [condition_rows[instance_id] for instance_id in common_ids],
                            dtype=np.float64,
                        ))
                        reference_arrays.append(np.asarray(
                            [reference_rows[instance_id] for instance_id in common_ids],
                            dtype=np.float64,
                        ))
                    bootstrap_seed = int.from_bytes(
                        hashlib.sha256(
                            f"{protocol['bootstrap']['random_seed']}|{condition}|{task}|{metric}".encode()
                        ).digest()[:8], "little"
                    )
                    scalar["paired_seed_and_complex_bootstrap"] = bootstrap_seed_complex_delta(
                        np.stack(condition_arrays), np.stack(reference_arrays),
                        replicates=replicates,
                        random_seed=bootstrap_seed,
                        higher_is_better=field_higher_is_better,
                    )
                task_comparison[metric] = scalar
            comparison["tasks"][task] = task_comparison

        cutoff_names = sorted(condition_summaries[condition]["contact_ranked_cutoffs"])
        for cutoff in cutoff_names:
            comparison["contact_ranked_cutoffs"][cutoff] = {}
            for metric in ("mean_precision", "mean_recall", "mean_enrichment_over_prevalence"):
                comparison["contact_ranked_cutoffs"][cutoff][metric] = paired_scalar(
                    [validations[(condition, seed)]["metrics"]["contact_ranked_cutoffs"][cutoff][metric]
                     for seed in seeds],
                    [validations[(reference, seed)]["metrics"]["contact_ranked_cutoffs"][cutoff][metric]
                     for seed in seeds],
                    higher_is_better=True,
                )
        comparison["training"] = {
            "wall_seconds": paired_scalar(
                [results[(condition, seed)]["wall_seconds"] for seed in seeds],
                [results[(reference, seed)]["wall_seconds"] for seed in seeds],
                higher_is_better=False,
            ),
            "trainable_parameters": paired_scalar(
                [results[(condition, seed)]["trainable_parameters"] for seed in seeds],
                [results[(reference, seed)]["trainable_parameters"] for seed in seeds],
                higher_is_better=False,
            ),
        }
        comparisons[condition] = comparison

    total_wall_seconds = sum(
        float(results[(condition, seed)]["wall_seconds"])
        for condition in conditions for seed in seeds
    )
    analysis_payload: dict[str, Any] = {
        "status": "PASS_MATCHED_MULTI_SEED_VALIDATION_ANALYSIS_NO_TEST_ACCESS",
        "format_version": "epic_reviewer_closure_validation_analysis_v1",
        "evidence_class": campaign["evidence_class"],
        "claim_boundaries": protocol["claim_boundaries"],
        "campaign_protocol_sha256": arguments.campaign_protocol_sha256,
        "analysis_protocol_sha256": arguments.analysis_protocol_sha256,
        "training_manifest_sha256": arguments.training_manifest_sha256,
        "reference_condition": reference,
        "conditions": conditions,
        "seeds": seeds,
        "condition_summaries": condition_summaries,
        "paired_comparisons": comparisons,
        "resource_summary": {
            "run_count": len(conditions) * len(seeds),
            "sum_run_wall_seconds": total_wall_seconds,
            "sum_run_wall_hours": total_wall_seconds / 3600.0,
            "accelerator_per_run": "1x NVIDIA RTX 4090",
            "parallel_execution": True,
        },
        "interpretation_policy": protocol["interpretation_policy"],
        "test_label_artifacts_opened": False,
        "test_selection_performed": False,
        "host": socket.gethostname(),
        "wall_seconds": time.monotonic() - started,
    }

    lines: list[str] = [
        "# EPIC matched reviewer-ablation validation analysis",
        "",
        "**Evidence boundary.** These are post-submission, three-seed matched ablations on the frozen "
        "official-development authority with independently sequence-predicted AF3 monomer geometry. "
        "They are not the unavailable anonymous OpenDDE checkpoint, the 321-complex validation set was "
        "used for checkpoint selection, and no campaign test labels were opened. Intervals are therefore "
        "descriptive stability diagnostics, not prospective-confirmatory p-values.",
        "",
        f"Reference condition: `{reference}`. Seeds: {', '.join(map(str, seeds))}. "
        f"Crossed seed×complex bootstrap replicates: {replicates:,}.",
        "",
        "## Three-seed validation means",
        "",
        "| Condition | Contact AP | Epitope AP | Paratope AP | Contact AUROC | Wall time (s) | Trainable params |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in conditions:
        summary = condition_summaries[condition]
        lines.append(
            f"| `{condition}` | {format_mean_sd(summary['tasks']['contact']['mpc_ap'])} | "
            f"{format_mean_sd(summary['tasks']['epitope']['mpc_ap'])} | "
            f"{format_mean_sd(summary['tasks']['paratope']['mpc_ap'])} | "
            f"{format_mean_sd(summary['tasks']['contact']['mpc_auroc'])} | "
            f"{format_mean_sd(summary['wall_seconds'], 1)} | "
            f"{format_mean_sd(summary['trainable_parameters'], 0)} |"
        )
    lines.extend([
        "",
        "## Paired ablation deltas versus the full learned-gate reference",
        "",
        "Values are `ablation − full reference`; brackets are 95% crossed seed×complex bootstrap intervals. "
        "For AP, a negative value favors the full reference.",
        "",
        "| Ablation | Contact ΔAP | Epitope ΔAP | Paratope ΔAP | Stability label (contact / epitope / paratope) |",
        "|---|---:|---:|---:|---|",
    ])
    for condition in conditions:
        if condition == reference:
            continue
        comparison = comparisons[condition]
        bootstraps = [
            comparison["tasks"][task]["mpc_ap"]["paired_seed_and_complex_bootstrap"]
            for task in task_names
        ]
        lines.append(
            f"| `{condition}` | {format_bootstrap(bootstraps[0])} | "
            f"{format_bootstrap(bootstraps[1])} | {format_bootstrap(bootstraps[2])} | "
            f"{bootstraps[0]['robust_raw_direction']} / "
            f"{bootstraps[1]['robust_raw_direction']} / "
            f"{bootstraps[2]['robust_raw_direction']} |"
        )
    lines.extend([
        "",
        "## Practical contact-ranking readouts",
        "",
        "| Condition | P@10 | R@10 | P@100 | R@100 | P@Lmin | R@Lmin |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for condition in conditions:
        ranked = condition_summaries[condition]["contact_ranked_cutoffs"]
        lines.append(
            f"| `{condition}` | {format_mean_sd(ranked['10']['mean_precision'])} | "
            f"{format_mean_sd(ranked['10']['mean_recall'])} | "
            f"{format_mean_sd(ranked['100']['mean_precision'])} | "
            f"{format_mean_sd(ranked['100']['mean_recall'])} | "
            f"{format_mean_sd(ranked['lmin']['mean_precision'])} | "
            f"{format_mean_sd(ranked['lmin']['mean_recall'])} |"
        )
    lines.extend([
        "",
        "## Interpretation rule",
        "",
        "A component-level directional statement is permitted only when all three paired seed effects agree "
        "and the crossed seed×complex interval excludes zero for the named metric. Otherwise the result is "
        "reported as heterogeneous or inconclusive. No single metric is allowed to erase a trade-off across "
        "contact, epitope, paratope, calibration, or ranking budgets.",
        "",
        "Exact OpenDDE predictor substitution and prospective biological validation remain unresolved authority "
        "boundaries; this analysis must not be relabelled as either one.",
        "",
    ])

    for path in (arguments.output_json, arguments.output_markdown, arguments.receipt):
        require(not path.exists(), f"Refusing to overwrite existing artifact: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output_json.open("x") as handle:
        json.dump(analysis_payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    arguments.output_markdown.write_text("\n".join(lines))
    receipt_payload = {
        "status": "PASS_VALIDATION_ANALYSIS_FROZEN",
        "format_version": "epic_reviewer_closure_validation_analysis_receipt_v1",
        "campaign_protocol_sha256": arguments.campaign_protocol_sha256,
        "analysis_protocol_sha256": arguments.analysis_protocol_sha256,
        "training_manifest_sha256": arguments.training_manifest_sha256,
        "analysis_json_path": str(arguments.output_json.resolve()),
        "analysis_json_sha256": sha256_file(arguments.output_json),
        "analysis_markdown_path": str(arguments.output_markdown.resolve()),
        "analysis_markdown_sha256": sha256_file(arguments.output_markdown),
        "test_label_artifacts_opened": False,
        "test_selection_performed": False,
    }
    with arguments.receipt.open("x") as handle:
        json.dump(receipt_payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print("COMPLETE", arguments.output_json, sha256_file(arguments.output_json), flush=True)
    print("RECEIPT", arguments.receipt, sha256_file(arguments.receipt), flush=True)


if __name__ == "__main__":
    main()
