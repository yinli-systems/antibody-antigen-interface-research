"""Dependency-light metrics for sparse interfaces and partner controls."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence


VALIDATED_MINIMUM_INDEPENDENT_CLUSTERS = 10


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


def _validated_probabilities(labels: Any, probabilities: Any) -> tuple[Any, Any]:
    labels, probabilities = _validated_binary_inputs(labels, probabilities)
    np = _numpy()
    if bool(np.logical_or(probabilities < 0.0, probabilities > 1.0).any()):
        raise ValueError("Calibration metrics require probabilities in [0, 1]")
    return labels, probabilities


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



def precision_recall_at_k(
    labels: Any,
    scores: Any,
    k_values: Sequence[int] = (10, 50, 100),
) -> dict[str, float | None]:
    """Precision@K and Recall@K for multiple K values, tie-aware."""
    np = _numpy()
    labels, scores = _validated_binary_inputs(labels, scores)
    positive_total = int(labels.sum())
    if positive_total == 0:
        result: dict[str, float | None] = {}
        for k in k_values:
            result[f"precision_at_{k}"] = None
            result[f"recall_at_{k}"] = None
        return result
    ranking = ranking_at_k(labels, scores, cutoffs=list(k_values))
    result = {}
    for k in k_values:
        p_at_k = ranking.get(f"precision_at_{k}")
        result[f"precision_at_{k}"] = p_at_k
        # recall@k = (P@k * k) / total_positives
        if p_at_k is not None:
            effective_k = min(k, len(labels))
            tp_at_k = p_at_k * effective_k
            result[f"recall_at_{k}"] = float(tp_at_k / positive_total) if positive_total else None
        else:
            result[f"recall_at_{k}"] = None
    return result


def enrichment_factor(
    labels: Any,
    scores: Any,
    fractions: Sequence[float] = (0.01, 0.05, 0.10),
) -> dict[str, float | None]:
    """Enrichment Factor at top-X% of the ranked list.

    EF@x% = (hits in top x%) / (expected hits if random).
    A perfect ranker concentrating all positives at the top gives EF = 1/fraction.
    """
    np = _numpy()
    labels, scores = _validated_binary_inputs(labels, scores)
    n = len(labels)
    positive_total = int(labels.sum())
    if positive_total == 0 or n == 0:
        return {f"ef_at_{int(f*100)}pct": None for f in fractions}
    prevalence = positive_total / n
    order = np.argsort(-scores, kind="stable")
    sorted_labels = labels[order]
    result: dict[str, float | None] = {}
    for frac in fractions:
        k = max(1, int(round(n * frac)))
        hits = int(sorted_labels[:k].sum())
        hit_rate = hits / k
        ef = hit_rate / prevalence if prevalence > 0 else None
        result[f"ef_at_{int(frac*100)}pct"] = float(ef) if ef is not None else None
    return result


def score_separation_stats(
    labels: Any,
    scores: Any,
) -> dict[str, float | None]:
    """Statistics measuring separation between positive and negative score distributions.

    Returns:
        pos_mean, neg_mean: mean scores for each class
        pos_std, neg_std: standard deviations
        cohens_d: (pos_mean - neg_mean) / pooled_std — effect size of separation
        score_gap: pos_mean - neg_mean
    """
    np = _numpy()
    labels, scores = _validated_binary_inputs(labels, scores)
    pos_mask = labels.astype(bool)
    neg_mask = ~pos_mask
    pos_scores = scores[pos_mask]
    neg_scores = scores[neg_mask]
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return {
            "pos_score_mean": None, "neg_score_mean": None,
            "pos_score_std": None, "neg_score_std": None,
            "cohens_d": None, "score_gap": None,
        }
    pos_mean = float(np.mean(pos_scores))
    neg_mean = float(np.mean(neg_scores))
    pos_std = float(np.std(pos_scores, ddof=1)) if len(pos_scores) > 1 else 0.0
    neg_std = float(np.std(neg_scores, ddof=1)) if len(neg_scores) > 1 else 0.0
    # pooled std
    n1, n2 = len(pos_scores), len(neg_scores)
    pooled_var = ((n1 - 1) * pos_std**2 + (n2 - 1) * neg_std**2) / (n1 + n2 - 2)
    pooled_std = float(np.sqrt(pooled_var)) if pooled_var > 0 else 1e-12
    cohens_d = (pos_mean - neg_mean) / pooled_std
    return {
        "pos_score_mean": pos_mean,
        "neg_score_mean": neg_mean,
        "pos_score_std": pos_std,
        "neg_score_std": neg_std,
        "cohens_d": float(cohens_d),
        "score_gap": float(pos_mean - neg_mean),
    }


def contact_map_metrics(
    labels: Any,
    scores: Any,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Add contact-map IoU and tie-invariant precision at the true interface size."""
    np = _numpy()
    labels, scores = _validated_binary_inputs(labels, scores)
    metrics = binary_metrics(labels, scores, threshold)
    truth = labels.astype(bool)
    predictions = scores >= threshold
    union = int(np.logical_or(truth, predictions).sum())
    intersection = int(np.logical_and(truth, predictions).sum())
    positive_count = int(truth.sum())
    if positive_count:
        ranking = ranking_at_k(labels, scores, cutoffs=(positive_count,))
        precision_at_true_count = float(
            ranking[f"precision_at_{positive_count}"]
        )
    else:
        precision_at_true_count = None
    # --- P@K / R@K ---
    pk_rk = precision_recall_at_k(labels, scores, k_values=(10, 50, 100))
    # --- Enrichment Factor ---
    ef = enrichment_factor(labels, scores, fractions=(0.01, 0.05, 0.10))
    # --- Score separation ---
    separation = score_separation_stats(labels, scores)
    metrics.update(
        {
            "contact_iou": intersection / union if union else 0.0,
            "precision_at_true_contact_count": precision_at_true_count,
            "precision_at_true_contact_count_tie_policy": (
                "fractional_expected_credit_at_score_boundary"
            ),
        }
    )
    metrics.update(pk_rk)
    metrics.update(ef)
    metrics.update(separation)
    return metrics


def ranking_at_k(
    labels: Any,
    scores: Any,
    cutoffs: Sequence[int] = (1, 5, 10),
) -> dict[str, Any]:
    """Return permutation-invariant expected precision and recall at fixed depths.

    If a score tie crosses a requested cutoff, every tied candidate receives
    the same fractional inclusion weight. The resulting true-positive count is
    the expectation under a uniformly random ordering inside that tie group.
    This preserves exactly ``min(k, candidate_count)`` units of support without
    allowing input order to determine the metric.
    """
    np = _numpy()
    labels, scores = _validated_binary_inputs(labels, scores)
    normalized_cutoffs: list[int] = []
    for cutoff in cutoffs:
        if isinstance(cutoff, bool) or not isinstance(cutoff, int) or cutoff <= 0:
            raise ValueError("Ranking cutoffs must be positive integers")
        normalized_cutoffs.append(cutoff)
    if len(set(normalized_cutoffs)) != len(normalized_cutoffs):
        raise ValueError("Ranking cutoffs must be unique")
    positive_total = int(labels.sum())
    result: dict[str, Any] = {
        "candidate_count": int(labels.size),
        "positive_count": positive_total,
        "tie_policy": "fractional_expected_credit_at_score_boundary",
    }
    for cutoff in normalized_cutoffs:
        selected_count = min(cutoff, int(labels.size))
        boundary_score = float(np.sort(scores)[-selected_count])
        strictly_above = scores > boundary_score
        at_boundary = scores == boundary_score
        strictly_above_count = int(strictly_above.sum())
        boundary_count = int(at_boundary.sum())
        remaining_support = selected_count - strictly_above_count
        if remaining_support <= 0 or remaining_support > boundary_count:
            raise AssertionError("Top-k boundary accounting is inconsistent")
        boundary_weight = remaining_support / boundary_count
        expected_true_positive = float(labels[strictly_above].sum()) + (
            boundary_weight * float(labels[at_boundary].sum())
        )
        result[f"precision_at_{cutoff}"] = (
            expected_true_positive / selected_count
        )
        result[f"recall_at_{cutoff}"] = (
            expected_true_positive / positive_total if positive_total else None
        )
        result[f"support_at_{cutoff}"] = selected_count
        result[f"expected_true_positive_at_{cutoff}"] = expected_true_positive
        result[f"boundary_score_at_{cutoff}"] = boundary_score
        result[f"boundary_tie_count_at_{cutoff}"] = boundary_count
        result[f"boundary_fractional_weight_at_{cutoff}"] = boundary_weight
    return result


def calibration_metrics(
    labels: Any,
    probabilities: Any,
    bins: int = 15,
    strategy: str = "equal_width",
    epsilon: float = 1e-12,
) -> dict[str, Any]:
    """Measure proper scoring rules and tie-preserving calibration error."""
    np = _numpy()
    labels, probabilities = _validated_probabilities(labels, probabilities)
    if isinstance(bins, bool) or not isinstance(bins, int) or bins <= 1:
        raise ValueError("Calibration bins must be an integer greater than one")
    if strategy not in {"equal_width", "adaptive"}:
        raise ValueError("Calibration strategy must be 'equal_width' or 'adaptive'")
    try:
        epsilon = float(epsilon)
    except (TypeError, ValueError) as error:
        raise ValueError("epsilon must be a finite probability tolerance") from error
    if not math.isfinite(epsilon) or not 0.0 < epsilon < 0.5:
        raise ValueError("epsilon must be in (0, 0.5)")

    if strategy == "equal_width":
        bin_indices = np.minimum(
            (probabilities * bins).astype(np.int64),
            bins - 1,
        )
        bin_boundaries = [
            (bin_index / bins, (bin_index + 1) / bins)
            for bin_index in range(bins)
        ]
    else:
        unique_probabilities, inverse, counts = np.unique(
            probabilities,
            return_inverse=True,
            return_counts=True,
        )
        cumulative_midpoints = np.cumsum(counts) - counts / 2.0
        unique_bin_indices = np.minimum(
            (cumulative_midpoints * bins / int(labels.size)).astype(np.int64),
            bins - 1,
        )
        bin_indices = unique_bin_indices[inverse]
        bin_boundaries = [
            (float("nan"), float("nan"))
            for _ in range(bins)
        ]

    absolute_weighted_gap = 0.0
    maximum_gap = 0.0
    populated_bins: list[dict[str, float | int]] = []
    for bin_index in range(bins):
        selected = bin_indices == bin_index
        count = int(selected.sum())
        if count == 0:
            continue
        mean_probability = float(probabilities[selected].mean())
        observed_frequency = float(labels[selected].mean())
        gap = abs(mean_probability - observed_frequency)
        absolute_weighted_gap += count * gap
        maximum_gap = max(maximum_gap, gap)
        populated_bins.append(
            {
                "index": bin_index,
                "count": count,
                "lower": (
                    bin_boundaries[bin_index][0]
                    if strategy == "equal_width"
                    else float(probabilities[selected].min())
                ),
                "upper": (
                    bin_boundaries[bin_index][1]
                    if strategy == "equal_width"
                    else float(probabilities[selected].max())
                ),
                "mean_probability": mean_probability,
                "observed_frequency": observed_frequency,
                "absolute_gap": gap,
            }
        )
    clipped = np.clip(probabilities, epsilon, 1.0 - epsilon)
    negative_log_likelihood = -np.mean(
        labels * np.log(clipped) + (1 - labels) * np.log1p(-clipped)
    )
    return {
        "count": int(labels.size),
        "bin_count": bins,
        "populated_bin_count": len(populated_bins),
        "strategy": strategy,
        "brier_score": float(np.mean((probabilities - labels) ** 2)),
        "negative_log_likelihood": float(negative_log_likelihood),
        "expected_calibration_error": absolute_weighted_gap / int(labels.size),
        "maximum_calibration_error": maximum_gap,
        "bins": populated_bins,
    }


def risk_coverage_metrics(
    labels: Any,
    probabilities: Any,
    coverages: Sequence[float] = (0.25, 0.5, 0.75, 1.0),
    *,
    decision_threshold: float = 0.5,
) -> dict[str, Any]:
    """Report selective risk with inclusive, tie-preserving confidence support.

    Predictions at or above ``decision_threshold`` are positive. Confidence is
    the absolute distance from that same decision threshold. At each requested
    coverage, the minimum number of highest-confidence examples is located and
    every example tied at the boundary confidence is included. Realized
    coverage can therefore exceed requested coverage, but never depends on
    input order.
    """
    np = _numpy()
    labels, probabilities = _validated_probabilities(labels, probabilities)
    try:
        decision_threshold = float(decision_threshold)
    except (TypeError, ValueError) as error:
        raise ValueError("decision_threshold must be a finite probability") from error
    if (
        not math.isfinite(decision_threshold)
        or not 0.0 <= decision_threshold <= 1.0
    ):
        raise ValueError("decision_threshold must be in [0, 1]")
    normalized_coverages: list[float] = []
    for coverage in coverages:
        try:
            value = float(coverage)
        except (TypeError, ValueError) as error:
            raise ValueError("Risk-coverage values must be finite fractions") from error
        if not math.isfinite(value) or not 0.0 < value <= 1.0:
            raise ValueError("Risk-coverage values must be in (0, 1]")
        normalized_coverages.append(value)
    if len(set(normalized_coverages)) != len(normalized_coverages):
        raise ValueError("Risk-coverage values must be unique")

    confidence = np.abs(probabilities - decision_threshold)
    truth = labels.astype(bool)
    predictions = probabilities >= decision_threshold
    points: list[dict[str, Any]] = []
    for coverage in normalized_coverages:
        requested_minimum_count = max(
            1,
            int(math.ceil(coverage * labels.size)),
        )
        selection_threshold = float(
            np.sort(confidence)[-requested_minimum_count]
        )
        selected = confidence >= selection_threshold
        selected_count = int(selected.sum())
        error_rate = float((predictions[selected] != truth[selected]).mean())
        points.append(
            {
                "requested_coverage": coverage,
                "requested_minimum_count": requested_minimum_count,
                "selected_count": selected_count,
                "realized_coverage": selected_count / int(labels.size),
                "classification_risk": error_rate,
                "accuracy": 1.0 - error_rate,
                "selection_confidence_threshold": selection_threshold,
                "minimum_selected_confidence": selection_threshold,
                "selection_threshold_comparator": "greater_than_or_equal",
            }
        )
    return {
        "count": int(labels.size),
        "selection_rule": (
            "include_all_examples_with_absolute_decision_margin_at_or_above_"
            "the_requested_coverage_boundary"
        ),
        "selection_tie_policy": "include_all_boundary_confidence_ties",
        "confidence_definition": (
            "absolute_probability_minus_classification_decision_threshold"
        ),
        "classification_decision_threshold": decision_threshold,
        "classification_decision_rule": (
            "probability_greater_than_or_equal_to_threshold_predicts_positive"
        ),
        "points": points,
    }


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
    excluded = {"count", "positive_count", "threshold", "tie_policy", "precision_at_true_contact_count_tie_policy"}
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
    candidate_count = int(negatives.shape[1] + 1)
    result = {
        "instance_count": int(len(matched)),
        "candidate_count_per_instance": candidate_count,
        "recall_at_1": float((ranks <= 1).mean()),
        "mean_reciprocal_rank": float((1.0 / ranks).mean()),
        "mean_rank": float(ranks.mean()),
        "median_rank": float(np.median(ranks)),
        "mean_normalized_rank": float(
            ((ranks - 1) / max(candidate_count - 1, 1)).mean()
        ),
        "mean_matched_score": float(matched.mean()),
        "mean_hardest_negative_score": float(negatives.max(axis=1).mean()),
        "mean_matched_minus_hardest_negative": float((matched - negatives.max(axis=1)).mean()),
    }
    for cutoff in (5, 10):
        if cutoff <= negatives.shape[1] + 1:
            result[f"recall_at_{cutoff}"] = float((ranks <= cutoff).mean())
    return result


def paired_cluster_bootstrap_mean_difference(
    candidate_by_instance: Mapping[str, float],
    reference_by_instance: Mapping[str, float],
    cluster_by_instance: Mapping[str, str],
    iterations: int = 10_000,
    seed: int = 42,
    confidence: float = 0.95,
    minimum_independent_clusters: int = VALIDATED_MINIMUM_INDEPENDENT_CLUSTERS,
    insufficient_cluster_policy: str = "raise",
) -> dict[str, Any]:
    """Bootstrap a paired mean only under an explicit cluster authorization gate.

    Instance and cluster identifiers must already be whitespace-normalized.
    Aliases that collapse to the same normalized identifier are rejected. By
    default, fewer than ``minimum_independent_clusters`` fails closed. An
    exploratory caller may instead request ``"mark_non_authorizing"``; the
    returned interval is then explicitly barred from inferential claims.
    """
    candidate, candidate_raw = _normalized_identifier_mapping(
        candidate_by_instance,
        name="candidate_by_instance",
    )
    reference, reference_raw = _normalized_identifier_mapping(
        reference_by_instance,
        name="reference_by_instance",
    )
    clusters, cluster_instance_raw = _normalized_identifier_mapping(
        cluster_by_instance,
        name="cluster_by_instance",
    )
    if set(candidate) != set(reference):
        raise ValueError("Paired bootstrap inputs must have identical instance IDs")
    if set(candidate) != set(clusters):
        raise ValueError(
            "Cluster mapping must have exactly the paired bootstrap instance IDs"
        )
    for identifier in candidate:
        raw_forms = {
            candidate_raw[identifier],
            reference_raw[identifier],
            cluster_instance_raw[identifier],
        }
        if len(raw_forms) != 1:
            raise ValueError(
                "Paired bootstrap instance IDs must use identical normalized forms"
            )
    if (
        isinstance(minimum_independent_clusters, bool)
        or not isinstance(minimum_independent_clusters, int)
        or minimum_independent_clusters
        < VALIDATED_MINIMUM_INDEPENDENT_CLUSTERS
    ):
        raise ValueError(
            "minimum_independent_clusters cannot be below the validated "
            f"floor of {VALIDATED_MINIMUM_INDEPENDENT_CLUSTERS}"
        )
    if insufficient_cluster_policy not in {"raise", "mark_non_authorizing"}:
        raise ValueError(
            "insufficient_cluster_policy must be 'raise' or "
            "'mark_non_authorizing'"
        )

    normalized_cluster_by_instance, cluster_count = _normalized_cluster_values(
        clusters
    )
    inferential_authorized = cluster_count >= minimum_independent_clusters
    if not inferential_authorized and insufficient_cluster_policy == "raise":
        raise ValueError(
            "Paired cluster bootstrap has fewer independent clusters than the "
            "validated minimum"
        )

    differences: dict[str, float] = {}
    for identifier in candidate:
        candidate_value = float(candidate[identifier])
        reference_value = float(reference[identifier])
        if not math.isfinite(candidate_value) or not math.isfinite(reference_value):
            raise ValueError("Paired bootstrap values must be finite")
        differences[identifier] = candidate_value - reference_value
    result = cluster_bootstrap_mean_ci(
        differences,
        normalized_cluster_by_instance,
        iterations=iterations,
        seed=seed,
        confidence=confidence,
    )
    result["candidate_mean"] = float(
        sum(float(value) for value in candidate.values())
        / len(candidate)
    )
    result["reference_mean"] = float(
        sum(float(value) for value in reference.values())
        / len(reference)
    )
    result["mean_difference"] = result.pop("mean")
    result["minimum_independent_cluster_count"] = minimum_independent_clusters
    result["validated_minimum_independent_cluster_floor"] = (
        VALIDATED_MINIMUM_INDEPENDENT_CLUSTERS
    )
    result["inferential_authorized"] = inferential_authorized
    result["authorization_reason"] = (
        "independent_cluster_minimum_satisfied"
        if inferential_authorized
        else "exploratory_only_independent_cluster_minimum_not_satisfied"
    )
    result["insufficient_cluster_policy"] = insufficient_cluster_policy
    return result


def _normalized_identifier_mapping(
    values: Mapping[Any, Any],
    *,
    name: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Normalize mapping keys while rejecting aliases and noncanonical forms."""
    normalized: dict[str, Any] = {}
    raw_by_normalized: dict[str, str] = {}
    noncanonical: list[str] = []
    for raw_identifier, value in values.items():
        if not isinstance(raw_identifier, str):
            raise ValueError(f"{name} identifiers must be strings")
        identifier = raw_identifier.strip()
        if not identifier:
            raise ValueError(f"{name} identifiers must be nonempty")
        if identifier in normalized:
            raise ValueError(
                f"{name} contains a normalized identifier collision: {identifier!r}"
            )
        normalized[identifier] = value
        raw_by_normalized[identifier] = raw_identifier
        if raw_identifier != identifier:
            noncanonical.append(raw_identifier)
    if noncanonical:
        raise ValueError(f"{name} identifiers must already be whitespace-normalized")
    return normalized, raw_by_normalized


def _normalized_cluster_values(
    cluster_by_instance: Mapping[str, Any],
) -> tuple[dict[str, str], int]:
    """Validate cluster IDs and reject distinct raw aliases of one cluster."""
    normalized: dict[str, str] = {}
    raw_by_cluster: dict[str, str] = {}
    noncanonical: list[str] = []
    for identifier, raw_cluster in cluster_by_instance.items():
        if not isinstance(raw_cluster, str):
            raise ValueError("Cluster identifiers must be strings")
        cluster = raw_cluster.strip()
        if not cluster:
            raise ValueError("Cluster identifiers must be nonempty")
        previous_raw = raw_by_cluster.get(cluster)
        if previous_raw is not None and previous_raw != raw_cluster:
            raise ValueError(
                f"Cluster mapping contains a normalized cluster collision: {cluster!r}"
            )
        raw_by_cluster[cluster] = raw_cluster
        normalized[identifier] = cluster
        if raw_cluster != cluster:
            noncanonical.append(raw_cluster)
    if noncanonical:
        raise ValueError("Cluster identifiers must already be whitespace-normalized")
    return normalized, len(raw_by_cluster)


def cluster_bootstrap_mean_ci(
    values_by_instance: Mapping[str, float],
    cluster_by_instance: Mapping[str, str],
    iterations: int = 2000,
    seed: int = 42,
    confidence: float = 0.95,
) -> dict[str, float | int]:
    """Bootstrap a mean by resampling independent clusters with replacement."""
    np = _numpy()
    try:
        confidence = float(confidence)
    except (TypeError, ValueError) as error:
        raise ValueError("Bootstrap confidence must be a finite fraction") from error
    if (
        isinstance(iterations, bool)
        or not isinstance(iterations, int)
        or iterations <= 0
        or not math.isfinite(confidence)
        or not 0.0 < confidence < 1.0
    ):
        raise ValueError("Bootstrap iterations and confidence are invalid")
    values, values_raw = _normalized_identifier_mapping(
        values_by_instance,
        name="values_by_instance",
    )
    clusters, clusters_raw = _normalized_identifier_mapping(
        cluster_by_instance,
        name="cluster_by_instance",
    )
    if set(values) != set(clusters):
        raise ValueError("Cluster mapping must exactly match bootstrap value IDs")
    for identifier in values:
        if values_raw[identifier] != clusters_raw[identifier]:
            raise ValueError(
                "Bootstrap value and cluster IDs must use identical normalized forms"
            )
    normalized_clusters, _ = _normalized_cluster_values(clusters)
    grouped: dict[str, list[float]] = defaultdict(list)
    for identifier, value in values.items():
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("Bootstrap values must be finite")
        grouped[normalized_clusters[identifier]].append(value)
    if not grouped:
        raise ValueError("Bootstrap requires at least one cluster")
    cluster_values = [np.asarray(values, dtype=np.float64) for _, values in sorted(grouped.items())]
    rng = np.random.default_rng(seed)
    estimates = np.empty(iterations, dtype=np.float64)
    for iteration in range(iterations):
        sampled = rng.integers(0, len(cluster_values), size=len(cluster_values))
        sampled_values = np.concatenate(
            [cluster_values[index] for index in sampled]
        )
        estimates[iteration] = sampled_values.mean()
    alpha = (1.0 - confidence) / 2.0
    return {
        "instance_count": len(values_by_instance),
        "cluster_count": len(cluster_values),
        "mean": float(np.mean(list(values.values()))),
        "confidence": float(confidence),
        "lower": float(np.quantile(estimates, alpha)),
        "upper": float(np.quantile(estimates, 1.0 - alpha)),
        "iterations": int(iterations),
        "seed": int(seed),
    }
