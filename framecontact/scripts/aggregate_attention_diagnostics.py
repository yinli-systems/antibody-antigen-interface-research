#!/usr/bin/env python3
"""Aggregate head-resolved attention diagnostics across seeds.

This post-processing step deliberately treats head indices as exchangeable across
independent seeds.  It therefore reports within-seed head dispersion and retains
the raw seed-by-head values instead of pretending that head 0 has the same role
after separate random initializations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


EXPERIMENTS = (
    "sequence_dense",
    "sequence_local",
    "spatial_topology",
    "distance",
    "distance_direction",
    "full_pose",
)
PARTNERS = ("antibody", "antigen")
FOCUS_METRICS = (
    "mean_normalized_entropy",
    "mean_ca_within_8a_conditional_attention_mass",
    "mean_ca_within_8a_mass_lift_over_uniform",
    "mean_ca_within_12a_conditional_attention_mass",
    "mean_ca_within_12a_mass_lift_over_uniform",
    "mean_ca_within_16a_conditional_attention_mass",
    "mean_ca_within_16a_mass_lift_over_uniform",
    "mean_expected_ca_distance_angstrom",
    "mean_sequence_within_8_attention_mass",
    "mean_sequence_within_16_attention_mass",
    "mean_sequence_within_32_attention_mass",
    "mean_geometry_bias_abs",
    "mean_geometry_counterfactual_total_variation",
    "mean_valid_geometry_attention_mass",
)
RESULT_FORMAT = "sabdab2_self_attention_factorized_ablation_v2"
SUMMARY_FORMAT = "sabdab2_attention_diagnostics_multiseed_v1"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=(17, 42, 73))
    parser.add_argument(
        "--experiments", nargs="+", choices=EXPERIMENTS, default=EXPERIMENTS
    )
    return parser.parse_args()


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sample_std(values: np.ndarray, *, axis: int | None = None) -> Any:
    count = values.shape[axis] if axis is not None else values.size
    if count <= 1:
        return None
    result = values.std(axis=axis, ddof=1)
    if np.ndim(result) == 0:
        return float(result)
    return [float(value) for value in result]


def _summarize_head_vectors(vectors: list[list[float]]) -> dict[str, Any]:
    array = np.asarray(vectors, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("head vectors must form a non-empty [seed, head] matrix")
    if not np.all(np.isfinite(array)):
        raise ValueError("head vectors must be finite")
    per_seed_mean = array.mean(axis=1)
    per_seed_range = np.ptp(array, axis=1)
    within_seed_std = (
        array.std(axis=1, ddof=1) if array.shape[1] > 1 else np.zeros(array.shape[0])
    )
    return {
        "values_by_seed_by_head": [
            [float(value) for value in row] for row in array
        ],
        "per_seed_head_mean": [float(value) for value in per_seed_mean],
        "per_seed_head_sample_std": [float(value) for value in within_seed_std],
        "per_seed_head_range": [float(value) for value in per_seed_range],
        "mean_of_seed_head_means": float(per_seed_mean.mean()),
        "sample_std_of_seed_head_means": _sample_std(per_seed_mean),
        "mean_within_seed_head_sample_std": float(within_seed_std.mean()),
        "mean_within_seed_head_range": float(per_seed_range.mean()),
        "pooled_seed_head_mean": float(array.mean()),
        "head_count": int(array.shape[1]),
        "seed_count": int(array.shape[0]),
    }


def main() -> None:
    arguments = _arguments()
    if len(set(arguments.seeds)) != len(arguments.seeds):
        raise ValueError("seeds must be unique")
    if len(set(arguments.experiments)) != len(arguments.experiments):
        raise ValueError("experiments must be unique")
    if arguments.output.exists() or arguments.output.is_symlink():
        raise ValueError("output already exists")

    runs: dict[tuple[int, str], dict[str, Any]] = {}
    source_files: list[dict[str, str]] = []
    parameter_counts: set[int] = set()
    manifest_hashes: set[str] = set()
    validation_axis_hashes: set[str] = set()
    initial_hashes_by_seed: dict[int, set[str]] = {
        seed: set() for seed in arguments.seeds
    }
    for seed in arguments.seeds:
        for experiment in arguments.experiments:
            result_path = (
                arguments.run_root
                / f"seed_{seed}"
                / experiment
                / "SABDAB2_SELF_ATTENTION_ABLATION.json"
            )
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            if payload.get("format_version") != RESULT_FORMAT:
                raise ValueError(f"Unexpected result format: {result_path}")
            if payload["configuration"]["seed"] != seed:
                raise ValueError(f"Seed mismatch: {result_path}")
            if payload["data"].get("native_release_test_cif_read") is not False:
                raise ValueError(f"Frozen/native test boundary differs: {result_path}")
            if len(payload["experiments"]) != 1:
                raise ValueError(f"Expected one experiment per run: {result_path}")
            run = payload["experiments"][0]
            if run["experiment"] != experiment:
                raise ValueError(f"Experiment mismatch: {result_path}")
            diagnostics = run.get("final_attention_diagnostics")
            if not isinstance(diagnostics, dict):
                raise ValueError(f"Missing attention diagnostics: {result_path}")
            runs[(seed, experiment)] = run
            source_files.append(
                {"path": str(result_path), "sha256": _sha256(result_path)}
            )
            parameter_counts.add(int(run["parameter_count"]))
            manifest_hashes.add(payload["data"]["structure_cache_manifest_sha256"])
            validation_axis_hashes.add(
                payload["data"]["validation_instance_ids_sha256"]
            )
            initial_hashes_by_seed[seed].add(run["initial_state_sha256"])
    if len(parameter_counts) != 1:
        raise ValueError("Run matrix differs in parameter count")
    if len(manifest_hashes) != 1 or len(validation_axis_hashes) != 1:
        raise ValueError("Run matrix differs in manifest or validation axis")
    if any(len(values) != 1 for values in initial_hashes_by_seed.values()):
        raise ValueError("Experiments within a seed have different initialization")

    aggregate: dict[str, Any] = {}
    for experiment in arguments.experiments:
        experiment_payload: dict[str, Any] = {}
        for partner in PARTNERS:
            per_seed_layers = [
                runs[(seed, experiment)]["final_attention_diagnostics"]["partners"][
                    partner
                ]["layers"]
                for seed in arguments.seeds
            ]
            layer_counts = {len(layers) for layers in per_seed_layers}
            if len(layer_counts) != 1:
                raise ValueError(f"Layer count differs: {experiment}, {partner}")
            layer_payloads = []
            for layer_index in range(next(iter(layer_counts))):
                seed_metrics = [
                    layers[layer_index]["metrics_per_head"]
                    for layers in per_seed_layers
                ]
                head_counts = {
                    len(metrics["mean_normalized_entropy"])
                    for metrics in seed_metrics
                }
                if len(head_counts) != 1:
                    raise ValueError(f"Head count differs: {experiment}, {partner}")
                available = set(seed_metrics[0])
                if any(set(metrics) != available for metrics in seed_metrics[1:]):
                    raise ValueError(f"Diagnostic schema differs: {experiment}, {partner}")
                selected_metrics = [
                    metric for metric in FOCUS_METRICS if metric in available
                ]
                summarized = {}
                for metric in selected_metrics:
                    vectors = [metrics[metric] for metrics in seed_metrics]
                    if any(
                        len(vector) != next(iter(head_counts)) for vector in vectors
                    ):
                        raise ValueError(
                            f"Metric head count differs: {experiment}, {partner}, {metric}"
                        )
                    summarized[metric] = _summarize_head_vectors(vectors)
                layer_payloads.append(
                    {
                        "layer_index": layer_index,
                        "head_count": next(iter(head_counts)),
                        "metrics": summarized,
                    }
                )
            experiment_payload[partner] = {"layers": layer_payloads}
        aggregate[experiment] = experiment_payload

    summary = {
        "format_version": SUMMARY_FORMAT,
        "claim_boundary": (
            "Locked development-only final-epoch attention diagnostics. Head indices are "
            "exchangeable across independently initialized seeds; use within-seed head "
            "dispersion, not index-wise semantic alignment. No causal interpretation, "
            "frozen-test claim, or benchmark-improvement claim is made."
        ),
        "seeds": list(arguments.seeds),
        "experiments": list(arguments.experiments),
        "data": {
            "structure_cache_manifest_sha256": next(iter(manifest_hashes)),
            "validation_instance_ids_sha256": next(iter(validation_axis_hashes)),
            "native_release_test_cif_read": False,
        },
        "model_parameter_count": next(iter(parameter_counts)),
        "identical_initial_state_sha256_by_seed": {
            str(seed): next(iter(values))
            for seed, values in initial_hashes_by_seed.items()
        },
        "interpretation_contract": {
            "entropy": "normalized entropy of the observed attention distribution",
            "spatial_enrichment": (
                "conditional mass on valid-frame pairs minus the uniform fraction on the "
                "same allowed attention graph"
            ),
            "geometry_counterfactual": (
                "total variation after removing only the learned additive geometry bias on "
                "the same graph; association with the bias pathway, not causality"
            ),
            "head_dependence": (
                "within-seed dispersion across exchangeable heads; head indices are not "
                "aligned across seeds"
            ),
        },
        "aggregate": aggregate,
        "source_files": source_files,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_bytes(_canonical_bytes(summary))
    receipt_path = arguments.output.with_suffix(arguments.output.suffix + ".receipt.json")
    receipt_path.write_bytes(
        _canonical_bytes(
            {
                "format_version": SUMMARY_FORMAT,
                "summary_path": str(arguments.output),
                "summary_sha256": _sha256(arguments.output),
                "run_count": len(runs),
                "seed_count": len(arguments.seeds),
                "experiment_count": len(arguments.experiments),
            }
        )
    )
    print(receipt_path.read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
