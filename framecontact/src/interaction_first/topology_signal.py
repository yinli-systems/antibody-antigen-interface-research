"""Development-only topology diagnostics for antibody--antigen interfaces.

The functions in this module deliberately use known residue labels.  They are
therefore *descriptive diagnostics*, not predictors and not model evaluation.
Their purpose is to test whether a proposed spatial propagation scale has a
measurable signal before spending GPU budget on a new architecture.
"""

from __future__ import annotations

import math
import statistics
from typing import Any, Iterable

import numpy as np
from sklearn.metrics import average_precision_score


def _square_boolean_matrix(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.bool_ or array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(f"{name} must be a square boolean matrix")
    return array


def spatial_relation_mask(
    backbone: np.ndarray,
    chain_ids: np.ndarray,
    frame_valid: np.ndarray,
    *,
    radius: float,
) -> np.ndarray:
    """Return a directed, self-edge-free, same-chain CA radius graph."""
    backbone = np.asarray(backbone)
    chain_ids = np.asarray(chain_ids)
    frame_valid = np.asarray(frame_valid)
    if (
        backbone.ndim != 3
        or backbone.shape[1:] != (3, 3)
        or chain_ids.shape != (backbone.shape[0],)
        or frame_valid.shape != (backbone.shape[0],)
        or frame_valid.dtype != np.bool_
    ):
        raise ValueError("backbone, chain_ids, and frame_valid axes differ")
    if not math.isfinite(float(radius)) or radius <= 0:
        raise ValueError("radius must be a positive finite scalar")
    ca = backbone[:, 1].astype(np.float64, copy=False)
    finite = np.isfinite(ca).all(axis=1)
    usable = frame_valid & finite
    delta = ca[:, None, :] - ca[None, :, :]
    squared_distance = np.einsum("ijk,ijk->ij", delta, delta)
    mask = (
        (squared_distance <= float(radius) ** 2)
        & (chain_ids[:, None] == chain_ids[None, :])
        & usable[:, None]
        & usable[None, :]
    )
    np.fill_diagonal(mask, False)
    return mask


def sequence_relation_mask(
    chain_ids: np.ndarray,
    residue_mask: np.ndarray,
    *,
    window: int,
) -> np.ndarray:
    """Return a directed, self-edge-free, same-chain sequence-window graph."""
    chain_ids = np.asarray(chain_ids)
    residue_mask = np.asarray(residue_mask)
    if (
        chain_ids.ndim != 1
        or residue_mask.shape != chain_ids.shape
        or residue_mask.dtype != np.bool_
    ):
        raise ValueError("chain_ids and residue_mask axes differ")
    if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
        raise ValueError("window must be a positive integer")
    positions = np.arange(len(chain_ids))
    mask = (
        (np.abs(positions[:, None] - positions[None, :]) <= window)
        & (chain_ids[:, None] == chain_ids[None, :])
        & residue_mask[:, None]
        & residue_mask[None, :]
    )
    np.fill_diagonal(mask, False)
    return mask


def positive_component_sizes(
    relation_mask: np.ndarray, labels: np.ndarray
) -> list[int]:
    """Connected positive-residue components under an arbitrary relation."""
    relation_mask = _square_boolean_matrix(relation_mask, "relation_mask")
    labels = np.asarray(labels)
    if labels.shape != (relation_mask.shape[0],):
        raise ValueError("labels must align with relation_mask")
    positive = np.flatnonzero(labels == 1)
    if not len(positive):
        return []
    adjacency = relation_mask | relation_mask.T
    remaining = set(int(index) for index in positive)
    sizes: list[int] = []
    while remaining:
        root = remaining.pop()
        stack = [root]
        size = 0
        while stack:
            current = stack.pop()
            size += 1
            neighbors = np.flatnonzero(adjacency[current])
            discovered = [int(index) for index in neighbors if int(index) in remaining]
            for index in discovered:
                remaining.remove(index)
            stack.extend(discovered)
        sizes.append(size)
    return sorted(sizes, reverse=True)


def relation_label_signal(
    relation_mask: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    """Measure leave-one-out label autocorrelation on a fixed relation graph.

    Neighbor labels are intentionally used in the score.  ``loo_neighbor_auprc``
    must therefore never be reported as prediction performance.
    """
    relation_mask = _square_boolean_matrix(relation_mask, "relation_mask")
    labels = np.asarray(labels, dtype=np.float64)
    if labels.shape != (relation_mask.shape[0],) or not np.isin(labels, (0, 1)).all():
        raise ValueError("labels must be aligned binary values")
    degree = relation_mask.sum(axis=1)
    query_mask = degree > 0
    query_labels = labels[query_mask]
    if not len(query_labels):
        return {
            "query_count": 0,
            "edge_count_directed": 0,
            "prevalence": None,
            "loo_neighbor_auprc": None,
            "positive_neighbor_rate": None,
            "negative_neighbor_rate": None,
            "positive_neighbor_lift": None,
            "positive_vs_negative_neighbor_rate_gap": None,
            "edge_homophily": None,
            "positive_component_count": 0,
            "positive_largest_component_fraction": None,
        }
    neighbor_positive_fraction = (
        relation_mask.astype(np.float64) @ labels
    ) / degree.clip(min=1)
    score = neighbor_positive_fraction[query_mask]
    prevalence = float(query_labels.mean())
    positive_queries = query_mask & (labels == 1)
    negative_queries = query_mask & (labels == 0)
    positive_neighbor_rate = (
        float(neighbor_positive_fraction[positive_queries].mean())
        if positive_queries.any()
        else None
    )
    negative_neighbor_rate = (
        float(neighbor_positive_fraction[negative_queries].mean())
        if negative_queries.any()
        else None
    )
    components = positive_component_sizes(relation_mask, labels)
    positive_count = int((labels == 1).sum())
    source, target = np.nonzero(relation_mask)
    edge_homophily = (
        float((labels[source] == labels[target]).mean()) if len(source) else None
    )
    return {
        "query_count": int(query_mask.sum()),
        "edge_count_directed": int(relation_mask.sum()),
        "prevalence": prevalence,
        "loo_neighbor_auprc": (
            float(average_precision_score(query_labels, score))
            if len(np.unique(query_labels)) == 2
            else None
        ),
        "positive_neighbor_rate": positive_neighbor_rate,
        "negative_neighbor_rate": negative_neighbor_rate,
        "positive_neighbor_lift": (
            positive_neighbor_rate / prevalence
            if positive_neighbor_rate is not None and prevalence > 0
            else None
        ),
        "positive_vs_negative_neighbor_rate_gap": (
            positive_neighbor_rate - negative_neighbor_rate
            if positive_neighbor_rate is not None and negative_neighbor_rate is not None
            else None
        ),
        "edge_homophily": edge_homophily,
        "positive_component_count": len(components),
        "positive_largest_component_fraction": (
            float(components[0] / positive_count)
            if components and positive_count
            else None
        ),
    }


def summarize_metric(
    values: Iterable[float | None],
    *,
    bootstrap_replicates: int,
    seed: int,
) -> dict[str, Any]:
    """Return transparent macro statistics and a complex bootstrap interval."""
    observed = np.asarray(
        [float(value) for value in values if value is not None], dtype=np.float64
    )
    if not len(observed):
        return {
            "count": 0,
            "mean": None,
            "sample_sd": None,
            "median": None,
            "ci95": None,
        }
    if bootstrap_replicates <= 0:
        raise ValueError("bootstrap_replicates must be positive")
    generator = np.random.default_rng(seed)
    indices = generator.integers(
        0, len(observed), size=(bootstrap_replicates, len(observed))
    )
    bootstrap_means = observed[indices].mean(axis=1)
    return {
        "count": int(len(observed)),
        "mean": float(observed.mean()),
        "sample_sd": (
            float(statistics.stdev(observed.tolist())) if len(observed) > 1 else 0.0
        ),
        "median": float(np.median(observed)),
        "ci95": [
            float(np.quantile(bootstrap_means, 0.025)),
            float(np.quantile(bootstrap_means, 0.975)),
        ],
    }


def paired_relation_delta(
    rows: list[dict[str, Any]],
    *,
    left: str,
    right: str,
    metric: str,
    bootstrap_replicates: int,
    seed: int,
) -> dict[str, Any]:
    """Summarize a within-complex relation difference."""
    deltas = []
    for row in rows:
        left_value = row["relations"][left][metric]
        right_value = row["relations"][right][metric]
        if left_value is not None and right_value is not None:
            deltas.append(float(left_value) - float(right_value))
    summary = summarize_metric(
        deltas, bootstrap_replicates=bootstrap_replicates, seed=seed
    )
    return {
        "left": left,
        "right": right,
        "metric": metric,
        "definition": "left minus right within the same validation complex",
        **summary,
    }


__all__ = [
    "paired_relation_delta",
    "positive_component_sizes",
    "relation_label_signal",
    "sequence_relation_mask",
    "spatial_relation_mask",
    "summarize_metric",
]
