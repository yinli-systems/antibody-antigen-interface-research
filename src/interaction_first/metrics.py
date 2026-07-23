"""Dependency-light metrics for sparse interfaces and partner controls."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence


def _numpy() -> Any:
    try:
        import numpy as np
    except ModuleNotFoundError as error:
        raise RuntimeError("NumPy is required for interaction metrics") from error
    return np


def _validated_binary_inputs(labels: Any, scores: Any) -> tuple[Any, Any]:
    np = _numpy()
    labels = np.asarray(labels, dtype=np.uint8).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if labels.size == 0 or labels.shape != scores.shape:
        raise ValueError("Metric labels and scores must be non-empty and aligned")
    if not np.isin(labels, (0, 1)).all() or not np.isfinite(scores).all():
        raise ValueError("Metrics require binary labels and finite scores")
    return labels, scores


def average_precision(labels: Any, scores: Any) -> float | None:
    """Compute tie-aware non-interpolated average precision."""
    np = _numpy()
    labels, scores = _validated_binary_inputs(labels, scores)
    positive_total = int(labels.sum())
    if positive_total == 0:
        return None
    order = np.argsort(-scores, kind="stable")
    labels = labels[order]
    scores = scores[order]
    true_positive = 0
    false_positive = 0
    result = 0.0
    start = 0
    while start < len(labels):
        end = start + 1
        while end < len(labels) and scores[end] == scores[start]:
            end += 1
        group_positive = int(labels[start:end].sum())
        group_negative = end - start - group_positive
        true_positive += group_positive
        false_positive += group_negative
        if group_positive:
            precision = true_positive / (true_positive + false_positive)
            result += (group_positive / positive_total) * precision
        start = end
    return float(result)


def roc_auc(labels: Any, scores: Any) -> float | None:
    """Compute tie-aware AUROC by integrating grouped ROC increments."""
    np = _numpy()
    labels, scores = _validated_binary_inputs(labels, scores)
    positive_total = int(labels.sum())
    negative_total = len(labels) - positive_total
    if positive_total == 0 or negative_total == 0:
        return None
    order = np.argsort(-scores, kind="stable")
    labels = labels[order]
    scores = scores[order]
    true_positive = 0
    false_positive = 0
    previous_tpr = 0.0
    previous_fpr = 0.0
    area = 0.0
    start = 0
    while start < len(labels):
        end = start + 1
        while end < len(labels) and scores[end] == scores[start]:
            end += 1
        group_positive = int(labels[start:end].sum())
        group_negative = end - start - group_positive
        true_positive += group_positive
        false_positive += group_negative
        tpr = true_positive / positive_total
        fpr = false_positive / negative_total
        area += (fpr - previous_fpr) * (tpr + previous_tpr) / 2.0
        previous_tpr = tpr
        previous_fpr = fpr
        start = end
    return float(area)


def binary_metrics(labels: Any, scores: Any, threshold: float = 0.5) -> dict[str, float | None]:
    """Return ranking and threshold metrics without hiding single-class cases."""
    np = _numpy()
    labels, scores = _validated_binary_inputs(labels, scores)
    predictions = scores >= float(threshold)
    truth = labels.astype(bool)
    tp = int(np.logical_and(predictions, truth).sum())
    tn = int(np.logical_and(~predictions, ~truth).sum())
    fp = int(np.logical_and(predictions, ~truth).sum())
    fn = int(np.logical_and(~predictions, truth).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denominator if denominator else 0.0
    return {
        "count": int(len(labels)),
        "positive_count": int(labels.sum()),
        "prevalence": float(labels.mean()),
        "auprc": average_precision(labels, scores),
        "auroc": roc_auc(labels, scores),
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mcc": float(mcc),
        "accuracy": float((predictions == truth).mean()),
    }


def contact_map_metrics(labels: Any, scores: Any, threshold: float = 0.5) -> dict[str, float | None]:
    """Add contact-map IoU and precision at the true interface size."""
    np = _numpy()
    labels, scores = _validated_binary_inputs(labels, scores)
    metrics = binary_metrics(labels, scores, threshold)
    truth = labels.astype(bool)
    predictions = scores >= threshold
    union = int(np.logical_or(truth, predictions).sum())
    intersection = int(np.logical_and(truth, predictions).sum())
    positive_count = int(truth.sum())
    if positive_count:
        order = np.argsort(-scores, kind="stable")[:positive_count]
        precision_at_true_count = float(labels[order].mean())
    else:
        precision_at_true_count = None
    metrics.update(
        {
            "contact_iou": intersection / union if union else 0.0,
            "precision_at_true_contact_count": precision_at_true_count,
        }
    )
    return metrics


def optional_mean(values: Iterable[float | None]) -> float | None:
    """Return a finite mean after omitting undefined values."""
    np = _numpy()
    array = np.asarray([value for value in values if value is not None], dtype=np.float64)
    if array.size == 0:
        return None
    if not np.isfinite(array).all():
        raise ValueError("Metric collection contains non-finite values")
    return float(array.mean())


def macro_metric_summary(records: Sequence[Mapping[str, float | None]]) -> dict[str, float | None]:
    """Average all numeric metric keys across complexes."""
    if not records:
        raise ValueError("Macro metrics require at least one record")
    excluded = {"count", "positive_count", "threshold"}
    keys = sorted(set.intersection(*(set(record) for record in records)) - excluded)
    return {key: optional_mean(record[key] for record in records) for key in keys}


def partner_ranking_metrics(
    matched_scores: Sequence[float],
    negative_scores: Sequence[Sequence[float]],
) -> dict[str, float]:
    """Evaluate matched-partner retrieval against real-sequence candidates."""
    np = _numpy()
    matched = np.asarray(matched_scores, dtype=np.float64)
    negatives = np.asarray(negative_scores, dtype=np.float64)
    if matched.ndim != 1 or negatives.ndim != 2 or matched.shape[0] != negatives.shape[0]:
        raise ValueError("Partner ranking scores are not aligned")
    if negatives.shape[1] == 0 or not np.isfinite(matched).all() or not np.isfinite(negatives).all():
        raise ValueError("Partner ranking requires finite non-empty negatives")
    ranks = 1 + (negatives >= matched[:, None]).sum(axis=1)
    result = {
        "instance_count": int(len(matched)),
        "candidate_count_per_instance": int(negatives.shape[1] + 1),
        "recall_at_1": float((ranks <= 1).mean()),
        "mean_rank": float(ranks.mean()),
        "median_rank": float(np.median(ranks)),
        "mean_matched_score": float(matched.mean()),
        "mean_hardest_negative_score": float(negatives.max(axis=1).mean()),
        "mean_matched_minus_hardest_negative": float((matched - negatives.max(axis=1)).mean()),
    }
    for cutoff in (5, 10):
        if cutoff <= negatives.shape[1] + 1:
            result[f"recall_at_{cutoff}"] = float((ranks <= cutoff).mean())
    return result


def cluster_bootstrap_mean_ci(
    values_by_instance: Mapping[str, float],
    cluster_by_instance: Mapping[str, str],
    iterations: int = 2000,
    seed: int = 42,
    confidence: float = 0.95,
) -> dict[str, float | int]:
    """Bootstrap a mean by resampling independent clusters with replacement."""
    np = _numpy()
    if iterations <= 0 or not 0.0 < confidence < 1.0:
        raise ValueError("Bootstrap iterations and confidence are invalid")
    if set(values_by_instance) - set(cluster_by_instance):
        raise ValueError("Cluster mapping is incomplete for bootstrap values")
    grouped: dict[str, list[float]] = defaultdict(list)
    for identifier, value in values_by_instance.items():
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("Bootstrap values must be finite")
        grouped[str(cluster_by_instance[identifier])].append(value)
    if not grouped:
        raise ValueError("Bootstrap requires at least one cluster")
    cluster_values = [np.asarray(values, dtype=np.float64) for _, values in sorted(grouped.items())]
    rng = np.random.default_rng(seed)
    estimates = np.empty(iterations, dtype=np.float64)
    for iteration in range(iterations):
        sampled = rng.integers(0, len(cluster_values), size=len(cluster_values))
        values = np.concatenate([cluster_values[index] for index in sampled])
        estimates[iteration] = values.mean()
    alpha = (1.0 - confidence) / 2.0
    return {
        "instance_count": len(values_by_instance),
        "cluster_count": len(cluster_values),
        "mean": float(np.mean(list(values_by_instance.values()))),
        "confidence": float(confidence),
        "lower": float(np.quantile(estimates, alpha)),
        "upper": float(np.quantile(estimates, 1.0 - alpha)),
        "iterations": int(iterations),
        "seed": int(seed),
    }
