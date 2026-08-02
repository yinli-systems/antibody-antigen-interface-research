#!/usr/bin/env python3
"""Validate and aggregate the factorized multi-seed development ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score


EXPERIMENTS = (
    "sequence_dense",
    "sequence_local",
    "spatial_topology",
    "distance",
    "distance_direction",
    "full_pose",
)
COMPARISONS = (
    ("sequence_local", "sequence_dense", "sequence_local_minus_dense"),
    ("spatial_topology", "sequence_dense", "spatial_topology_minus_dense"),
    ("distance", "spatial_topology", "distance_features_minus_topology"),
    (
        "distance_direction",
        "spatial_topology",
        "distance_direction_features_minus_topology",
    ),
    ("full_pose", "spatial_topology", "full_pose_features_minus_topology"),
    ("full_pose", "distance_direction", "rotation_pose_minus_direction"),
)
RESULT_FORMAT = "sabdab2_self_attention_factorized_ablation_v2"
SUMMARY_FORMAT = "sabdab2_self_attention_multiseed_summary_v1"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=(17, 42, 73))
    parser.add_argument("--experiments", nargs="+", choices=EXPERIMENTS, default=EXPERIMENTS)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260729)
    return parser.parse_args()


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mean_std(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "values_by_seed": [float(value) for value in array],
        "mean": float(array.mean()),
        "sample_std": float(array.std(ddof=1)) if len(array) > 1 else None,
    }


def _load_prediction(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        required = {"instance_ids"}
        for axis in ("paratope", "epitope"):
            required.update(
                {f"{axis}_offsets", f"{axis}_labels", f"{axis}_probabilities"}
            )
        if set(payload.files) != required:
            raise ValueError(f"Prediction schema differs: {path}")
        result = {key: payload[key].copy() for key in required}
    instance_count = len(result["instance_ids"])
    for axis in ("paratope", "epitope"):
        offsets = result[f"{axis}_offsets"]
        labels = result[f"{axis}_labels"]
        probabilities = result[f"{axis}_probabilities"]
        if len(offsets) != instance_count + 1 or offsets[0] != 0:
            raise ValueError(f"Invalid {axis} offsets: {path}")
        if offsets[-1] != len(labels) or len(labels) != len(probabilities):
            raise ValueError(f"Invalid {axis} flattened arrays: {path}")
        if not np.all(np.diff(offsets) > 0):
            raise ValueError(f"Empty {axis} complex: {path}")
        if not np.all(np.isfinite(probabilities)) or not np.all(
            (0.0 <= probabilities) & (probabilities <= 1.0)
        ):
            raise ValueError(f"Invalid {axis} probabilities: {path}")
    return result


def _complex_auprc(prediction: dict[str, Any], axis: str) -> np.ndarray:
    offsets = prediction[f"{axis}_offsets"]
    labels = prediction[f"{axis}_labels"]
    probabilities = prediction[f"{axis}_probabilities"]
    values = []
    for start, stop in zip(offsets[:-1], offsets[1:]):
        local_labels = labels[start:stop]
        if local_labels.sum() <= 0:
            raise ValueError(f"{axis} complex has no positive residue")
        values.append(average_precision_score(local_labels, probabilities[start:stop]))
    return np.asarray(values, dtype=np.float64)


def _hierarchical_paired_interval(
    delta: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    if delta.ndim != 2 or not np.all(np.isfinite(delta)):
        raise ValueError("delta must be a finite [seed, complex] matrix")
    rng = np.random.default_rng(seed)
    seed_count, complex_count = delta.shape
    bootstrap = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sampled_seeds = rng.integers(0, seed_count, size=seed_count)
        sampled_complexes = rng.integers(0, complex_count, size=complex_count)
        bootstrap[index] = delta[np.ix_(sampled_seeds, sampled_complexes)].mean()
    low, high = np.quantile(bootstrap, (0.025, 0.975))
    return {
        "estimate": float(delta.mean()),
        "ci_95": [float(low), float(high)],
        "bootstrap_probability_positive": float((bootstrap > 0.0).mean()),
        "replicates": replicates,
        "resampling_unit": "paired development complexes nested within sampled seeds",
    }


def main() -> None:
    arguments = _arguments()
    if arguments.bootstrap_replicates <= 0:
        raise ValueError("bootstrap-replicates must be positive")
    if len(set(arguments.seeds)) != len(arguments.seeds):
        raise ValueError("seeds must be unique")
    if len(set(arguments.experiments)) != len(arguments.experiments):
        raise ValueError("experiments must be unique")
    if arguments.output.exists() or arguments.output.is_symlink():
        raise ValueError("output already exists")

    runs: dict[tuple[int, str], dict[str, Any]] = {}
    predictions: dict[tuple[int, str], dict[str, Any]] = {}
    source_files = []
    parameter_counts = set()
    manifest_hashes = set()
    validation_axis_hashes = set()
    initial_hashes_by_seed: dict[int, set[str]] = {
        seed: set() for seed in arguments.seeds
    }
    for seed in arguments.seeds:
        for experiment in arguments.experiments:
            root = arguments.run_root / f"seed_{seed}" / experiment
            result_path = root / "SABDAB2_SELF_ATTENTION_ABLATION.json"
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            if payload.get("format_version") != RESULT_FORMAT:
                raise ValueError(f"Unexpected result format: {result_path}")
            if payload["configuration"]["seed"] != seed:
                raise ValueError(f"Seed mismatch: {result_path}")
            if len(payload["experiments"]) != 1:
                raise ValueError(f"Expected one experiment per run: {result_path}")
            run = payload["experiments"][0]
            if run["experiment"] != experiment:
                raise ValueError(f"Experiment mismatch: {result_path}")
            prediction_path = root / run["validation_predictions"]
            if _sha256(prediction_path) != run["validation_predictions_sha256"]:
                raise ValueError(f"Prediction digest mismatch: {prediction_path}")
            runs[(seed, experiment)] = run
            predictions[(seed, experiment)] = _load_prediction(prediction_path)
            parameter_counts.add(run["parameter_count"])
            manifest_hashes.add(payload["data"]["structure_cache_manifest_sha256"])
            validation_axis_hashes.add(payload["data"]["validation_instance_ids_sha256"])
            initial_hashes_by_seed[seed].add(run["initial_state_sha256"])
            source_files.extend(
                (
                    {"path": str(result_path), "sha256": _sha256(result_path)},
                    {"path": str(prediction_path), "sha256": _sha256(prediction_path)},
                )
            )
    if len(parameter_counts) != 1 or len(manifest_hashes) != 1 or len(validation_axis_hashes) != 1:
        raise ValueError("Run matrix differs in parameters, manifest, or validation axis")
    if any(len(values) != 1 for values in initial_hashes_by_seed.values()):
        raise ValueError("Experiments within a seed did not share identical initialization")

    reference_key = (arguments.seeds[0], arguments.experiments[0])
    reference_prediction = predictions[reference_key]
    for key, prediction in predictions.items():
        if not np.array_equal(prediction["instance_ids"], reference_prediction["instance_ids"]):
            raise ValueError(f"Prediction instance order differs: {key}")
        for axis in ("paratope", "epitope"):
            if not np.array_equal(
                prediction[f"{axis}_offsets"], reference_prediction[f"{axis}_offsets"]
            ) or not np.array_equal(
                prediction[f"{axis}_labels"], reference_prediction[f"{axis}_labels"]
            ):
                raise ValueError(f"Prediction labels differ: {key}, {axis}")

    aggregate: dict[str, Any] = {}
    for experiment in arguments.experiments:
        records = [runs[(seed, experiment)] for seed in arguments.seeds]
        fixed = [record["history"][-1]["validation"] for record in records]
        best = [
            record["history"][record["development_best_epoch"] - 1]["validation"]
            for record in records
        ]
        aggregate[experiment] = {
            "experiment_spec": records[0]["experiment_spec"],
            "fixed_final": {
                metric: _mean_std([float(item[metric]) for item in fixed])
                for metric in ("macro_auprc", "macro_auroc", "weighted_bce")
            },
            "development_best": {
                metric: _mean_std([float(item[metric]) for item in best])
                for metric in ("macro_auprc", "macro_auroc", "weighted_bce")
            },
            "development_best_epochs": [
                int(record["development_best_epoch"]) for record in records
            ],
            "fixed_final_by_axis": {
                axis: {
                    metric: _mean_std(
                        [float(item[axis][metric]) for item in fixed]
                    )
                    for metric in ("auprc", "auroc", "mcc_at_0_5")
                }
                for axis in ("paratope", "epitope")
            },
        }

    comparisons = []
    for comparison_index, (candidate, reference, name) in enumerate(COMPARISONS):
        if candidate not in arguments.experiments or reference not in arguments.experiments:
            continue
        comparison: dict[str, Any] = {
            "name": name,
            "candidate": candidate,
            "reference": reference,
            "fixed_final_macro_auprc_delta_by_seed": [],
            "mean_complex_auprc_delta": {},
        }
        for seed in arguments.seeds:
            candidate_metric = runs[(seed, candidate)]["history"][-1]["validation"][
                "macro_auprc"
            ]
            reference_metric = runs[(seed, reference)]["history"][-1]["validation"][
                "macro_auprc"
            ]
            comparison["fixed_final_macro_auprc_delta_by_seed"].append(
                float(candidate_metric - reference_metric)
            )
        for axis_index, axis in enumerate(("paratope", "epitope")):
            delta_rows = []
            for seed in arguments.seeds:
                candidate_scores = _complex_auprc(predictions[(seed, candidate)], axis)
                reference_scores = _complex_auprc(predictions[(seed, reference)], axis)
                delta_rows.append(candidate_scores - reference_scores)
            delta = np.stack(delta_rows)
            comparison["mean_complex_auprc_delta"][axis] = _hierarchical_paired_interval(
                delta,
                replicates=arguments.bootstrap_replicates,
                seed=arguments.bootstrap_seed + 100 * comparison_index + axis_index,
            )
        comparisons.append(comparison)

    summary = {
        "format_version": SUMMARY_FORMAT,
        "claim_boundary": (
            "Locked development-only multi-seed analysis. Native-release and frozen tests "
            "remain unread; intervals quantify development stability, not generalization or "
            "benchmark improvement. Resampling is by complex because cluster IDs are not "
            "present in the prediction artifact."
        ),
        "seeds": list(arguments.seeds),
        "experiments": list(arguments.experiments),
        "data": {
            "structure_cache_manifest_sha256": next(iter(manifest_hashes)),
            "validation_instance_ids_sha256": next(iter(validation_axis_hashes)),
            "validation_complex_count": len(reference_prediction["instance_ids"]),
        },
        "model_parameter_count": next(iter(parameter_counts)),
        "identical_initial_state_sha256_by_seed": {
            str(seed): next(iter(values)) for seed, values in initial_hashes_by_seed.items()
        },
        "aggregate": aggregate,
        "paired_comparisons": comparisons,
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
                "bootstrap_replicates": arguments.bootstrap_replicates,
            }
        )
    )
    print(receipt_path.read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
