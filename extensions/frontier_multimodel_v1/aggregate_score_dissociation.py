#!/usr/bin/env python3
"""Strict complete-campaign aggregation for score-dissociation all-vs-all runs."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from extensions.frontier_multimodel_v1.score_dissociation import SCORE_COLUMNS, sha256_file, exclusive_json

CONDITIONS = ("identity:0", "chain_shift:8", "global_shuffle:0")
SEEDS = (1, 2, 3)
RANK_METRICS = (
    "mrr_tie_expected",
    "recall_at_1_tie_expected",
    "recall_at_5_tie_expected",
    "recall_at_10_tie_expected",
    "known_positive_ap_unknown_as_zero",
)
CORRELATIONS = (
    "spearman_score_vs_candidate_length",
    "spearman_score_vs_candidate_surface_proxy",
    "spearman_score_vs_candidate_node_prior",
    "spearman_score_vs_interaction_std",
)


def bootstrap_crossed(matrix: np.ndarray, seed: int, repeats: int = 5000) -> dict:
    if matrix.ndim != 2 or matrix.shape[0] != 3 or matrix.shape[1] < 10 or not np.isfinite(matrix).all():
        raise ValueError("expected complete three-seed by anchor paired matrix")
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        seed_indices = rng.integers(matrix.shape[0], size=matrix.shape[0])
        anchor_indices = rng.integers(matrix.shape[1], size=matrix.shape[1])
        values.append(float(matrix[seed_indices][:, anchor_indices].mean()))
    return {
        "mean": float(matrix.mean()),
        "ci95": [float(value) for value in np.quantile(values, [0.025, 0.975])],
        "seed_count": matrix.shape[0],
        "anchor_count": matrix.shape[1],
        "bootstrap_repeats": repeats,
        "resampling": "crossed training-seed and unique-anchor bootstrap",
    }


def cluster_bootstrap_spearman(score_delta: np.ndarray, contact_delta: np.ndarray,
                               clusters: Sequence[str], seed: int, repeats: int = 3000) -> dict:
    from scipy.stats import spearmanr
    if score_delta.shape != contact_delta.shape or score_delta.ndim != 2:
        raise ValueError("paired seed by complex matrices required")
    clusters = np.asarray(clusters, dtype=str)
    names = sorted(set(clusters))
    if len(names) < 10:
        raise ValueError("too few authority clusters")
    point_by_seed = [float(spearmanr(score_delta[index], contact_delta[index]).statistic) for index in range(score_delta.shape[0])]
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(repeats):
        chosen_clusters = rng.choice(names, size=len(names), replace=True)
        selected = np.concatenate([np.flatnonzero(clusters == name) for name in chosen_clusters])
        chosen_seeds = rng.integers(score_delta.shape[0], size=score_delta.shape[0])
        correlations = []
        for seed_index in chosen_seeds:
            left, right = score_delta[seed_index, selected], contact_delta[seed_index, selected]
            value = float(spearmanr(left, right).statistic)
            if math.isfinite(value):
                correlations.append(value)
        if correlations:
            samples.append(float(np.mean(correlations)))
    return {
        "mean_seed_spearman": float(np.mean(point_by_seed)),
        "per_seed_spearman": point_by_seed,
        "crossed_seed_cluster_ci95": [float(value) for value in np.quantile(samples, [0.025, 0.975])],
        "complex_count": score_delta.shape[1],
        "cluster_count": len(names),
        "bootstrap_repeats": repeats,
    }


def main() -> None:
    arguments = parser().parse_args()
    import pandas as pd
    from scipy.stats import spearmanr

    receipts = {}
    grids = {}
    anchors = {}
    input_hashes = {}
    for seed in SEEDS:
        for condition in CONDITIONS:
            directory = arguments.results / f"seed_{seed}" / condition.replace(":", "_")
            receipt_path = directory / "RECEIPT.json"
            receipt = json.loads(receipt_path.read_text())
            if receipt.get("status") != "PASS_FULL_UNIQUE_SEQUENCE_ALL_VS_ALL_DECOMPOSITION":
                raise RuntimeError(f"incomplete task: {receipt_path}")
            if receipt.get("training_seed") != seed or receipt.get("condition") != condition:
                raise RuntimeError(f"task identity mismatch: {receipt_path}")
            if receipt.get("protocol_sha256") != sha256_file(arguments.protocol):
                raise RuntimeError(f"protocol mismatch: {receipt_path}")
            grid_path = directory / receipt["artifacts"]["pair_grid"]
            anchor_path = directory / receipt["artifacts"]["anchor_metrics"]
            if sha256_file(grid_path) != receipt["artifacts"]["pair_grid_sha256"] or sha256_file(anchor_path) != receipt["artifacts"]["anchor_metrics_sha256"]:
                raise RuntimeError(f"artifact digest mismatch: {directory}")
            key = (seed, condition)
            receipts[key] = receipt
            grids[key] = pd.read_parquet(grid_path)
            anchors[key] = pd.read_parquet(anchor_path)
            input_hashes[str(receipt_path)] = sha256_file(receipt_path)
            input_hashes[str(grid_path)] = sha256_file(grid_path)
            input_hashes[str(anchor_path)] = sha256_file(anchor_path)

    baseline_keys = None
    for key, frame in grids.items():
        observed = list(zip(frame.ab_sequence_sha256, frame.ag_sequence_sha256))
        if len(observed) != 122 * 119 or len(set(observed)) != len(observed):
            raise RuntimeError(f"non-unique or incomplete grid {key}")
        if baseline_keys is None:
            baseline_keys = observed
        elif observed != baseline_keys:
            raise RuntimeError(f"grid ordering or candidate universe changed {key}")

    comparisons = []
    absolute = []
    for direction in sorted(anchors[(1, CONDITIONS[0])].direction.unique()):
        for score in SCORE_COLUMNS:
            for metric in RANK_METRICS:
                values_by_condition = {}
                anchors_expected = None
                for condition in CONDITIONS:
                    matrix = []
                    for seed in SEEDS:
                        selected = anchors[(seed, condition)]
                        selected = selected[(selected.direction == direction) & (selected.score_name == score)].sort_values("anchor_sequence_sha256")
                        names = selected.anchor_sequence_sha256.tolist()
                        if anchors_expected is None:
                            anchors_expected = names
                        elif names != anchors_expected:
                            raise RuntimeError("anchor universe differs across tasks")
                        matrix.append(selected[metric].to_numpy(dtype=float))
                    values_by_condition[condition] = np.asarray(matrix)
                    absolute.append({
                        "direction": direction,
                        "score_name": score,
                        "metric": metric,
                        "condition": condition,
                        **bootstrap_crossed(values_by_condition[condition], seed=int(hashlib.sha256(f"abs:{direction}:{score}:{metric}:{condition}".encode()).hexdigest()[:8], 16)),
                    })
                for condition in CONDITIONS[1:]:
                    delta = values_by_condition[condition] - values_by_condition["identity:0"]
                    comparisons.append({
                        "direction": direction,
                        "score_name": score,
                        "metric": metric,
                        "comparison": f"{condition}_minus_identity",
                        **bootstrap_crossed(delta, seed=int(hashlib.sha256(f"delta:{direction}:{score}:{metric}:{condition}".encode()).hexdigest()[:8], 16)),
                    })

    bias_correlations = []
    for direction in sorted(anchors[(1, CONDITIONS[0])].direction.unique()):
        for score in SCORE_COLUMNS:
            for condition in CONDITIONS:
                for correlation in CORRELATIONS:
                    per_seed = []
                    for seed in SEEDS:
                        selected = anchors[(seed, condition)]
                        selected = selected[(selected.direction == direction) & (selected.score_name == score)]
                        values = selected[correlation].dropna().to_numpy(dtype=float)
                        per_seed.append(float(values.mean()) if len(values) else None)
                    bias_correlations.append({
                        "direction": direction,
                        "score_name": score,
                        "condition": condition,
                        "correlation": correlation,
                        "per_seed_mean_anchor_spearman": per_seed,
                        "mean_over_seeds": float(np.mean([value for value in per_seed if value is not None])) if any(value is not None for value in per_seed) else None,
                    })

    frame = pd.read_parquet(arguments.prepare_dir / "instances.parquet").set_index("instance_id", verify_integrity=True)
    split = json.loads(arguments.split_manifest.read_text())
    validation_ids = sorted(split["splits"]["validation"])
    frame = frame.loc[validation_ids]
    clusters = [str(split["component_id_by_instance"][identifier]) for identifier in validation_ids]
    contact_links = []
    condition_map = {"identity:0": "identity:0", "chain_shift:8": "chain_shift:8", "global_shuffle:0": "global_shuffle:0"}
    prior = {}
    for seed, task_id in zip(SEEDS, (0, 1, 2)):
        payload = json.loads((arguments.correspondence_results / f"task_{task_id}" / "RESULT.json").read_text())
        if payload["task"]["training_seed"] != seed or payload["task"]["encoder"] != "esm_if1":
            raise RuntimeError("prior correspondence result mismatch")
        records = {(row["instance_id"], row["condition"]): row for row in payload["records"]}
        for identifier in validation_ids:
            for condition in CONDITIONS:
                prior[(seed, identifier, condition)] = records[(identifier, condition_map[condition])]["pair_ap"]

    for score in SCORE_COLUMNS:
        for condition in CONDITIONS[1:]:
            score_delta_rows = []
            contact_delta_rows = []
            for seed in SEEDS:
                identity_grid = grids[(seed, "identity:0")].set_index(["ab_sequence_sha256", "ag_sequence_sha256"])
                condition_grid = grids[(seed, condition)].set_index(["ab_sequence_sha256", "ag_sequence_sha256"])
                score_deltas, contact_deltas = [], []
                for identifier in validation_ids:
                    row = frame.loc[identifier]
                    key = (str(row.ab_sequence_sha256), str(row.ag_sequence_sha256))
                    score_deltas.append(float(condition_grid.loc[key, score] - identity_grid.loc[key, score]))
                    contact_deltas.append(float(prior[(seed, identifier, condition)] - prior[(seed, identifier, "identity:0")]))
                score_delta_rows.append(score_deltas)
                contact_delta_rows.append(contact_deltas)
            score_matrix = np.asarray(score_delta_rows)
            contact_matrix = np.asarray(contact_delta_rows)
            contact_links.append({
                "score_name": score,
                "comparison": f"{condition}_minus_identity",
                **cluster_bootstrap_spearman(score_matrix, contact_matrix, clusters,
                    seed=int(hashlib.sha256(f"link:{score}:{condition}".encode()).hexdigest()[:8], 16)),
                "mean_score_delta": float(score_matrix.mean()),
                "mean_contact_ap_delta": float(contact_matrix.mean()),
            })

    # Directly quantify whether the current score is primarily a spread statistic.
    mechanism = []
    for condition in CONDITIONS:
        for seed in SEEDS:
            data = grids[(seed, condition)]
            mechanism.append({
                "training_seed": seed,
                "condition": condition,
                "spearman_current_score_vs_centered_std": float(spearmanr(data.score_current_centered_lme, data.interaction_centered_std).statistic),
                "spearman_current_score_vs_centered_top16": float(spearmanr(data.score_current_centered_lme, data.interaction_centered_top16).statistic),
                "spearman_current_score_vs_pair_probability_top16": float(spearmanr(data.score_current_centered_lme, data.pair_probability_top16).statistic),
            })

    payload = {
        "status": "COMPLETE_SCORE_DISSOCIATION_ANALYSIS",
        "format_version": "score_dissociation_analysis_v1",
        "protocol_sha256": sha256_file(arguments.protocol),
        "tasks": len(receipts),
        "unique_pair_count_per_task": 122 * 119,
        "total_pair_evaluations": len(receipts) * 122 * 119,
        "absolute_metrics": absolute,
        "paired_corruption_comparisons": comparisons,
        "bias_correlations": bias_correlations,
        "contact_link_analysis": contact_links,
        "mechanism_correlations": mechanism,
        "input_sha256": input_hashes,
        "test_loaded": False,
        "training_performed": False,
        "promotion_authorized": False,
        "interpretation_boundary": [
            "all results are development validation",
            "unobserved pairs are unlabelled distractors, not non-binders",
            "pointwise intervals are exploratory and not multiplicity corrected",
            "alternative scores inspected here cannot be promoted on this evidence",
        ],
    }
    exclusive_json(arguments.output, payload)
    print(json.dumps({"status": payload["status"], "tasks": len(receipts), "comparisons": len(comparisons), "contact_links": len(contact_links), "output": str(arguments.output)}))


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--prepare-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--correspondence-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


if __name__ == "__main__":
    main()
