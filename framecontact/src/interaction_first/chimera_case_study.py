"""Exploratory, provenance-bound CHIMERA epitope case-study utilities.

The module deliberately keeps descriptive case analysis separate from the
preregistered external scorer.  It never changes checkpoints, thresholds, or
the frozen primary endpoint.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.spatial import cKDTree, distance
from scipy.stats import spearmanr
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .sabdab2_structure import sha256_file


CANDIDATES = ("sequence_dense", "spatial_topology")
SEEDS = (17, 42, 73)
PREDICTION_KEYS = {
    "instance_ids",
    "antibody_offsets",
    "antigen_offsets",
    "antibody_labels",
    "antigen_labels",
    "paratope_probabilities",
    "epitope_probabilities",
}
CORRELATE_FEATURES = (
    "antigen_length",
    "epitope_count",
    "epitope_prevalence",
    "epitope_sequence_run_count",
    "epitope_components_8a",
    "epitope_largest_component_fraction_8a",
    "epitope_radius_of_gyration_a",
    "epitope_mean_nearest_neighbor_distance_a",
    "antigen_mean_neighbor_count_16a",
    "epitope_mean_neighbor_count_16a",
    "valid_frame_fraction",
)


def load_prediction_matrix(
    prediction_root: Path, manifest: dict[str, Any]
) -> dict[tuple[str, int], dict[str, np.ndarray]]:
    """Load and cross-check the complete two-candidate, three-seed matrix."""
    records = manifest.get("predictions")
    if not isinstance(records, list) or len(records) != 6:
        raise ValueError("prediction manifest must contain exactly six runs")
    matrix: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    reference: tuple[np.ndarray, ...] | None = None
    for record in records:
        key = (str(record.get("candidate")), int(record.get("seed", -1)))
        if key in matrix or key[0] not in CANDIDATES or key[1] not in SEEDS:
            raise ValueError("prediction run matrix identity differs")
        path = prediction_root / str(record.get("prediction_path", ""))
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"prediction artifact is missing or unsafe: {path}")
        if sha256_file(path) != record.get("prediction_sha256"):
            raise ValueError(f"prediction digest differs: {path}")
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != PREDICTION_KEYS:
                raise ValueError("prediction artifact schema differs")
            arrays = {name: archive[name].copy() for name in archive.files}
        identifiers = arrays["instance_ids"]
        if identifiers.ndim != 1 or len(set(identifiers.tolist())) != len(identifiers):
            raise ValueError("prediction identifiers are invalid")
        authority = (
            identifiers,
            arrays["antibody_offsets"],
            arrays["antigen_offsets"],
            arrays["antibody_labels"],
            arrays["antigen_labels"],
        )
        if reference is None:
            reference = tuple(value.copy() for value in authority)
        elif any(not np.array_equal(left, right) for left, right in zip(reference, authority)):
            raise ValueError("prediction ordering or labels differ between runs")
        matrix[key] = arrays
    expected = {(candidate, seed) for candidate in CANDIDATES for seed in SEEDS}
    if set(matrix) != expected:
        raise ValueError("prediction run matrix is incomplete")
    return matrix


def ensemble_probabilities(
    matrix: dict[tuple[str, int], dict[str, np.ndarray]], candidate: str
) -> np.ndarray:
    return np.mean(
        np.stack(
            [matrix[(candidate, seed)]["epitope_probabilities"] for seed in SEEDS]
        ),
        axis=0,
        dtype=np.float64,
    )


def connected_component_sizes(coords: np.ndarray, threshold: float) -> list[int]:
    """Return Euclidean CA-graph component sizes at a fixed distance threshold."""
    count = len(coords)
    if count == 0:
        return []
    parent = np.arange(count)
    size = np.ones(count, dtype=np.int64)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    for left, right in cKDTree(coords).query_pairs(threshold):
        left_root, right_root = find(int(left)), find(int(right))
        if left_root == right_root:
            continue
        if size[left_root] < size[right_root]:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        size[left_root] += size[right_root]
    counts: dict[int, int] = defaultdict(int)
    for index in range(count):
        counts[find(index)] += 1
    return sorted(counts.values(), reverse=True)


def sequence_run_count(labels: np.ndarray, chain_ids: np.ndarray) -> int:
    positives = np.flatnonzero(labels == 1.0)
    if not len(positives):
        return 0
    runs = 1
    for previous, current in zip(positives[:-1], positives[1:]):
        if current != previous + 1 or chain_ids[current] != chain_ids[previous]:
            runs += 1
    return runs


def _neighbor_counts(coords: np.ndarray, radius: float) -> np.ndarray:
    if not len(coords):
        return np.empty(0, dtype=np.float64)
    neighbors = cKDTree(coords).query_ball_point(coords, radius)
    return np.asarray([len(indices) - 1 for indices in neighbors], dtype=np.float64)


def geometry_descriptors(
    backbone: np.ndarray,
    chain_ids: np.ndarray,
    frame_valid: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    """Describe antigen size, epitope dispersion, and local packing."""
    length = len(labels)
    positive = labels == 1.0
    valid = frame_valid.astype(bool, copy=False)
    ca = backbone[:, 1, :]
    finite = np.isfinite(ca).all(axis=1)
    usable = valid & finite
    positive_usable = positive & usable
    usable_coords = ca[usable].astype(np.float64, copy=False)
    positive_coords = ca[positive_usable].astype(np.float64, copy=False)
    positive_count = int(positive.sum())
    positive_usable_count = int(positive_usable.sum())

    result: dict[str, Any] = {
        "antigen_length": int(length),
        "epitope_count": positive_count,
        "epitope_prevalence": float(positive.mean()) if length else 0.0,
        "valid_frame_fraction": float(valid.mean()) if length else 0.0,
        "epitope_valid_frame_fraction": (
            float(positive_usable_count / positive_count) if positive_count else 0.0
        ),
        "epitope_sequence_run_count": sequence_run_count(labels, chain_ids),
    }
    for threshold in (6.0, 8.0, 10.0, 12.0):
        sizes = connected_component_sizes(positive_coords, threshold)
        suffix = f"{int(threshold)}a"
        result[f"epitope_components_{suffix}"] = len(sizes)
        result[f"epitope_largest_component_fraction_{suffix}"] = (
            float(sizes[0] / positive_usable_count) if sizes else 0.0
        )

    if positive_usable_count:
        center = positive_coords.mean(axis=0)
        result["epitope_radius_of_gyration_a"] = float(
            np.sqrt(np.mean(np.sum((positive_coords - center) ** 2, axis=1)))
        )
    else:
        result["epitope_radius_of_gyration_a"] = None
    if positive_usable_count >= 2:
        pairwise = distance.pdist(positive_coords)
        nearest = cKDTree(positive_coords).query(positive_coords, k=2)[0][:, 1]
        result["epitope_max_pairwise_distance_a"] = float(pairwise.max())
        result["epitope_mean_pairwise_distance_a"] = float(pairwise.mean())
        result["epitope_mean_nearest_neighbor_distance_a"] = float(nearest.mean())
    else:
        result["epitope_max_pairwise_distance_a"] = None
        result["epitope_mean_pairwise_distance_a"] = None
        result["epitope_mean_nearest_neighbor_distance_a"] = None

    all_counts = _neighbor_counts(usable_coords, 16.0)
    result["antigen_mean_neighbor_count_16a"] = (
        float(all_counts.mean()) if len(all_counts) else None
    )
    if positive_usable_count and len(usable_coords):
        tree = cKDTree(usable_coords)
        epitope_counts = np.asarray(
            [len(indices) - 1 for indices in tree.query_ball_point(positive_coords, 16.0)],
            dtype=np.float64,
        )
        result["epitope_mean_neighbor_count_16a"] = float(epitope_counts.mean())
    else:
        result["epitope_mean_neighbor_count_16a"] = None
    return result


def prediction_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    binary = (probabilities >= 0.5).astype(np.int64)
    positive_count = int(labels.sum())
    if positive_count:
        top = np.argpartition(probabilities, -positive_count)[-positive_count:]
        precision_at_label_count = float(labels[top].mean())
    else:
        precision_at_label_count = 0.0
    return {
        "auprc": (
            float(average_precision_score(labels, probabilities))
            if positive_count
            else 0.0
        ),
        "auroc": (
            float(roc_auc_score(labels, probabilities))
            if len(np.unique(labels)) == 2
            else None
        ),
        "precision_at_0_5": float(precision_score(labels, binary, zero_division=0)),
        "recall_at_0_5": float(recall_score(labels, binary, zero_division=0)),
        "f1_at_0_5": float(f1_score(labels, binary, zero_division=0)),
        "mcc_at_0_5": (
            float(matthews_corrcoef(labels, binary))
            if len(np.unique(np.concatenate((labels, binary)))) == 2
            else 0.0
        ),
        "brier": float(brier_score_loss(labels, probabilities)),
        "precision_at_label_count": precision_at_label_count,
    }


def _finite_pairs(rows: list[dict[str, Any]], feature: str) -> tuple[np.ndarray, np.ndarray]:
    pairs = [
        (row[feature], row["delta_auprc_spatial_minus_dense"])
        for row in rows
        if row.get(feature) is not None
        and np.isfinite(float(row[feature]))
        and np.isfinite(float(row["delta_auprc_spatial_minus_dense"]))
    ]
    if not pairs:
        return np.empty(0), np.empty(0)
    return np.asarray(pairs, dtype=np.float64).T


def correlation_with_group_bootstrap(
    rows: list[dict[str, Any]],
    feature: str,
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    x, y = _finite_pairs(rows, feature)
    if len(x) < 3 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return {"n": int(len(x)), "spearman_rho": None, "p_value": None, "ci_low": None, "ci_high": None}
    estimate = spearmanr(x, y)
    members: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        value = row.get(feature)
        if value is not None and np.isfinite(float(value)):
            members[str(row["antigen_sequence_sha256"])].append(index)
    group_names = sorted(members)
    rng = np.random.default_rng(seed)
    boot: list[float] = []
    for _ in range(replicates):
        sampled_groups = rng.choice(group_names, size=len(group_names), replace=True)
        sampled_rows = [rows[index] for group in sampled_groups for index in members[str(group)]]
        bx, by = _finite_pairs(sampled_rows, feature)
        if len(bx) >= 3 and len(np.unique(bx)) >= 2 and len(np.unique(by)) >= 2:
            value = float(spearmanr(bx, by).statistic)
            if np.isfinite(value):
                boot.append(value)
    low, high = (np.quantile(boot, (0.025, 0.975)) if boot else (np.nan, np.nan))
    return {
        "n": int(len(x)),
        "group_count": len(group_names),
        "spearman_rho": float(estimate.statistic),
        "p_value": float(estimate.pvalue),
        "bootstrap_replicates_requested": replicates,
        "bootstrap_replicates_valid": len(boot),
        "ci_low": float(low) if np.isfinite(low) else None,
        "ci_high": float(high) if np.isfinite(high) else None,
    }


def quantile_strata(
    rows: list[dict[str, Any]], feature: str, *, bins: int = 4
) -> dict[str, Any]:
    values = np.asarray(
        [float(row[feature]) for row in rows if row.get(feature) is not None],
        dtype=np.float64,
    )
    if not len(values):
        return {"edges": [], "bins": []}
    edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) == 1:
        edges = np.asarray([edges[0], edges[0]])
    outputs = []
    for index in range(len(edges) - 1):
        low, high = float(edges[index]), float(edges[index + 1])
        selected = [
            row
            for row in rows
            if row.get(feature) is not None
            and float(row[feature]) >= low
            and (float(row[feature]) <= high if index == len(edges) - 2 else float(row[feature]) < high)
        ]
        delta = np.asarray([row["delta_auprc_spatial_minus_dense"] for row in selected])
        outputs.append(
            {
                "bin": index + 1,
                "low_inclusive": low,
                "high_inclusive": high if index == len(edges) - 2 else None,
                "high_exclusive": None if index == len(edges) - 2 else high,
                "n": len(selected),
                "dense_mean_auprc": float(np.mean([row["sequence_dense_auprc"] for row in selected])) if selected else None,
                "spatial_mean_auprc": float(np.mean([row["spatial_topology_auprc"] for row in selected])) if selected else None,
                "mean_delta_auprc": float(delta.mean()) if len(delta) else None,
                "median_delta_auprc": float(np.median(delta)) if len(delta) else None,
            }
        )
    return {"edges": edges.tolist(), "bins": outputs}


def mean_delta_group_bootstrap(
    rows: list[dict[str, Any]], *, replicates: int, seed: int
) -> dict[str, Any]:
    if not rows:
        return {
            "n": 0,
            "group_count": 0,
            "dense_mean_auprc": None,
            "spatial_mean_auprc": None,
            "mean_delta_auprc": None,
            "median_delta_auprc": None,
            "ci_low": None,
            "ci_high": None,
        }
    members: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        members[str(row["antigen_sequence_sha256"])].append(row)
    groups = sorted(members)
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        values = [
            row["delta_auprc_spatial_minus_dense"]
            for group in sampled
            for row in members[str(group)]
        ]
        bootstrap[replicate] = float(np.mean(values))
    low, high = np.quantile(bootstrap, (0.025, 0.975))
    return {
        "n": len(rows),
        "group_count": len(groups),
        "dense_mean_auprc": float(np.mean([row["sequence_dense_auprc"] for row in rows])),
        "spatial_mean_auprc": float(np.mean([row["spatial_topology_auprc"] for row in rows])),
        "mean_delta_auprc": float(np.mean([row["delta_auprc_spatial_minus_dense"] for row in rows])),
        "median_delta_auprc": float(np.median([row["delta_auprc_spatial_minus_dense"] for row in rows])),
        "ci_low": float(low),
        "ci_high": float(high),
    }


def summarize_rows(
    rows: list[dict[str, Any]], *, bootstrap_replicates: int, seed: int
) -> dict[str, Any]:
    correlations = {
        feature: correlation_with_group_bootstrap(
            rows, feature, replicates=bootstrap_replicates, seed=seed + index
        )
        for index, feature in enumerate(CORRELATE_FEATURES)
    }
    strata = {
        feature: quantile_strata(rows, feature)
        for feature in (
            "antigen_length",
            "epitope_count",
            "epitope_prevalence",
            "epitope_components_8a",
            "epitope_radius_of_gyration_a",
        )
    }
    cohorts = {
        "all": rows,
        "antigen_length_ge_50": [row for row in rows if row["antigen_length"] >= 50],
        "epitope_prevalence_le_0_5": [
            row for row in rows if row["epitope_prevalence"] <= 0.5
        ],
        "antigen_length_ge_50_and_epitope_prevalence_le_0_5": [
            row
            for row in rows
            if row["antigen_length"] >= 50 and row["epitope_prevalence"] <= 0.5
        ],
    }
    sensitivity = {
        name: mean_delta_group_bootstrap(
            cohort, replicates=bootstrap_replicates, seed=seed + 100 + index
        )
        for index, (name, cohort) in enumerate(cohorts.items())
    }
    ranked = sorted(rows, key=lambda row: row["delta_auprc_spatial_minus_dense"])
    return {
        "complex_count": len(rows),
        "correlations": correlations,
        "quantile_strata": strata,
        "sensitivity_cohorts": sensitivity,
        "most_harmed_cases": ranked[:10],
        "most_helped_cases": list(reversed(ranked[-10:])),
    }


def analyze_instances(
    *,
    cache_root: Path,
    cache_manifest: dict[str, Any],
    prediction_matrix: dict[tuple[str, int], dict[str, np.ndarray]],
    split_name: str,
    bootstrap_replicates: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    reference = prediction_matrix[(CANDIDATES[0], SEEDS[0])]
    identifiers = [str(value) for value in reference["instance_ids"]]
    records = {str(record["instance_id"]): record for record in cache_manifest["records"]}
    if set(identifiers) != set(records):
        raise ValueError("prediction and cache instance sets differ")
    offsets = reference["antigen_offsets"]
    labels_flat = reference["antigen_labels"]
    ensemble = {
        candidate: ensemble_probabilities(prediction_matrix, candidate)
        for candidate in CANDIDATES
    }
    rows: list[dict[str, Any]] = []
    for index, identifier in enumerate(identifiers):
        record = records[identifier]
        memberships = record.get("split_memberships")
        if not isinstance(memberships, list) or split_name not in memberships:
            continue
        artifact = cache_root / str(record["artifact_path"])
        if artifact.is_symlink() or not artifact.is_file():
            raise ValueError(f"cache artifact is missing or unsafe: {artifact}")
        if sha256_file(artifact) != record["artifact_sha256"]:
            raise ValueError(f"cache artifact digest differs: {artifact}")
        with np.load(artifact, allow_pickle=False) as archive:
            antigen_sequence = str(archive["antigen_sequence"].item()).strip().upper()
            backbone = archive["antigen_backbone"].copy()
            chain_ids = archive["antigen_chain_ids"].copy()
            valid = archive["antigen_frame_valid_mask"].copy()
            cache_labels = archive["antigen_labels"].copy()
        start, end = int(offsets[index]), int(offsets[index + 1])
        labels = labels_flat[start:end]
        if not np.array_equal(labels, cache_labels):
            raise ValueError(f"prediction/cache labels differ for {identifier}")
        row: dict[str, Any] = {
            "instance_id": identifier,
            "pdb_id": identifier.split("_", 1)[0].lower(),
            "split": split_name,
            "antigen_sequence_sha256": hashlib.sha256(antigen_sequence.encode("ascii")).hexdigest(),
        }
        row.update(geometry_descriptors(backbone, chain_ids, valid, labels))
        metrics: dict[str, dict[str, Any]] = {}
        for candidate in CANDIDATES:
            values = prediction_metrics(labels, ensemble[candidate][start:end])
            metrics[candidate] = values
            for metric, value in values.items():
                row[f"{candidate}_{metric}"] = value
        for metric in ("auprc", "auroc", "f1_at_0_5", "mcc_at_0_5", "brier", "precision_at_label_count"):
            dense_value = metrics["sequence_dense"][metric]
            spatial_value = metrics["spatial_topology"][metric]
            row[f"delta_{metric}_spatial_minus_dense"] = (
                None if dense_value is None or spatial_value is None else float(spatial_value - dense_value)
            )
        rows.append(row)
    if not rows:
        raise ValueError(f"no predictions belong to split {split_name}")
    return rows, summarize_rows(rows, bootstrap_replicates=bootstrap_replicates, seed=seed)


def flatten_case_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return deterministic scalar-only CSV records."""
    return [{key: row[key] for key in sorted(row)} for row in rows]
