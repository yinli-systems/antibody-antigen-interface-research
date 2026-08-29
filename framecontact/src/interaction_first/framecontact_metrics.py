"""Frozen evaluation metrics for contact, calibration, and partner retrieval."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np


def _binary_arrays(labels: Any, probabilities: Any) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if labels.shape != probabilities.shape or labels.size == 0:
        raise ValueError("labels and probabilities must be nonempty and aligned")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("labels must be binary")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("probabilities must be finite and lie in [0, 1]")
    return labels.astype(np.int8, copy=False), probabilities


def expected_calibration_error(
    labels: Any, probabilities: Any, *, bins: int = 15
) -> float:
    labels, probabilities = _binary_arrays(labels, probabilities)
    if isinstance(bins, bool) or not isinstance(bins, int) or bins <= 1:
        raise ValueError("bins must be an integer greater than one")
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    assignments = np.minimum(np.searchsorted(boundaries, probabilities, side="right") - 1, bins - 1)
    assignments = np.maximum(assignments, 0)
    total = float(len(labels))
    error = 0.0
    for bin_index in range(bins):
        selected = assignments == bin_index
        if selected.any():
            confidence = float(probabilities[selected].mean())
            accuracy = float(labels[selected].mean())
            error += float(selected.sum()) / total * abs(confidence - accuracy)
    return error


def reliability_bins(
    labels: Any, probabilities: Any, *, bins: int = 15
) -> list[dict[str, float | int | None]]:
    """Return fixed-width bin statistics sufficient to render a reliability plot."""
    labels, probabilities = _binary_arrays(labels, probabilities)
    if isinstance(bins, bool) or not isinstance(bins, int) or bins <= 1:
        raise ValueError("bins must be an integer greater than one")
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    assignments = np.minimum(
        np.searchsorted(boundaries, probabilities, side="right") - 1,
        bins - 1,
    )
    assignments = np.maximum(assignments, 0)
    result = []
    for bin_index in range(bins):
        selected = assignments == bin_index
        result.append(
            {
                "lower": float(boundaries[bin_index]),
                "upper": float(boundaries[bin_index + 1]),
                "count": int(selected.sum()),
                "mean_probability": (
                    float(probabilities[selected].mean()) if selected.any() else None
                ),
                "positive_fraction": (
                    float(labels[selected].mean()) if selected.any() else None
                ),
            }
        )
    return result


def logistic_calibration_regression(
    labels: Any,
    probabilities: Any,
    *,
    maximum_iterations: int = 50,
    tolerance: float = 1e-8,
) -> dict[str, Any]:
    """Fit calibration intercept and slope by two-parameter logistic IRLS.

    The implementation uses scalar sufficient sums rather than constructing a
    ``candidate_count x 2`` design matrix, which keeps final pooled contact
    evaluation practical for tens of millions of residue pairs.
    """
    labels, probabilities = _binary_arrays(labels, probabilities)
    if (
        isinstance(maximum_iterations, bool)
        or not isinstance(maximum_iterations, int)
        or maximum_iterations <= 0
    ):
        raise ValueError("maximum_iterations must be positive")
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    if len(np.unique(labels)) != 2:
        return {
            "calibration_intercept": None,
            "calibration_slope": None,
            "calibration_regression_converged": False,
            "calibration_regression_iterations": 0,
            "calibration_regression_reason": "single_class",
        }
    epsilon = np.finfo(np.float64).eps
    clipped = np.clip(probabilities, epsilon, 1.0 - epsilon)
    predictor = np.log(clipped) - np.log1p(-clipped)
    if float(np.ptp(predictor)) <= 1e-12:
        return {
            "calibration_intercept": None,
            "calibration_slope": None,
            "calibration_regression_converged": False,
            "calibration_regression_iterations": 0,
            "calibration_regression_reason": "constant_prediction",
        }
    prevalence = float(labels.mean())
    beta = np.asarray(
        [math.log(prevalence / (1.0 - prevalence)), 1.0], dtype=np.float64
    )
    converged = False
    reason = "maximum_iterations"
    iterations = 0
    for iterations in range(1, maximum_iterations + 1):
        linear = np.clip(beta[0] + beta[1] * predictor, -50.0, 50.0)
        fitted = 1.0 / (1.0 + np.exp(-linear))
        residual = labels - fitted
        weight = fitted * (1.0 - fitted)
        gradient = np.asarray(
            [residual.sum(), np.dot(predictor, residual)], dtype=np.float64
        )
        information = np.asarray(
            [
                [weight.sum(), np.dot(predictor, weight)],
                [
                    np.dot(predictor, weight),
                    np.dot(predictor * predictor, weight),
                ],
            ],
            dtype=np.float64,
        )
        try:
            step = np.linalg.solve(information, gradient)
        except np.linalg.LinAlgError:
            reason = "singular_information"
            break
        if not np.isfinite(step).all():
            reason = "non_finite_step"
            break
        beta += step
        if float(np.max(np.abs(step))) < tolerance:
            converged = True
            reason = "converged"
            break
    return {
        "calibration_intercept": float(beta[0]),
        "calibration_slope": float(beta[1]),
        "calibration_regression_converged": converged,
        "calibration_regression_iterations": iterations,
        "calibration_regression_reason": reason,
    }


def binary_classification_metrics(
    labels: Any,
    probabilities: Any,
    *,
    threshold: float,
    include_calibration_regression: bool = False,
) -> dict[str, Any]:
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        f1_score,
        matthews_corrcoef,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    labels, probabilities = _binary_arrays(labels, probabilities)
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must lie in [0, 1]")
    predictions = probabilities >= threshold
    both_classes = len(np.unique(labels)) == 2
    result = {
        "auprc": float(average_precision_score(labels, probabilities)),
        "auroc": float(roc_auc_score(labels, probabilities)) if both_classes else None,
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, predictions)) if both_classes else None,
        "brier": float(brier_score_loss(labels, probabilities)),
        "ece_15": expected_calibration_error(labels, probabilities, bins=15),
        "reliability_bins_15": reliability_bins(labels, probabilities, bins=15),
        "positive_count": int(labels.sum()),
        "candidate_count": int(labels.size),
        "predicted_positive_count": int(predictions.sum()),
        "predicted_positive_ratio": float(predictions.mean()),
    }
    if include_calibration_regression:
        result.update(logistic_calibration_regression(labels, probabilities))
    return result


def positive_required_average_precision(
    labels: Any, probabilities: Any
) -> dict[str, Any]:
    """Return AP only when the evaluated unit contains a positive target.

    Pooled AP can legitimately retain all-negative complexes as negative
    candidates.  A per-complex AP, however, has no positive-retrieval problem
    to solve when that complex has zero positive labels.  Returning an explicit
    eligibility record prevents both sklearn's warning-driven ``0.0`` convention
    and silent sample deletion.
    """
    from sklearn.metrics import average_precision_score

    labels, probabilities = _binary_arrays(labels, probabilities)
    positive_count = int(labels.sum())
    if positive_count == 0:
        return {
            "eligible": False,
            "value": None,
            "ineligible_reason": "zero_positive_labels",
            "positive_count": 0,
            "candidate_count": int(labels.size),
        }
    return {
        "eligible": True,
        "value": float(average_precision_score(labels, probabilities)),
        "ineligible_reason": None,
        "positive_count": positive_count,
        "candidate_count": int(labels.size),
    }


def ranked_pair_metrics(
    labels: Any,
    probabilities: Any,
    *,
    antibody_length: int,
    antigen_length: int,
) -> dict[str, Any]:
    """Return deterministic interface-contact precision/recall at fixed cutoffs.

    Cross-chain contact maps have two sequence lengths.  To avoid silently
    choosing one side, normalized cutoffs use ``Lmin = min(L_ab, L_ag)`` and
    the output records every requested and effective cutoff.
    """
    labels, probabilities = _binary_arrays(labels, probabilities)
    for value, name in (
        (antibody_length, "antibody_length"),
        (antigen_length, "antigen_length"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    positive_count = int(labels.sum())
    if positive_count == 0:
        raise ValueError("ranked pair metrics require at least one positive contact")
    length_minimum = min(antibody_length, antigen_length)
    requested = {
        "10": 10,
        "20": 20,
        "50": 50,
        "lmin_over_10": max(1, math.ceil(length_minimum / 10)),
        "lmin_over_5": max(1, math.ceil(length_minimum / 5)),
        "lmin": length_minimum,
    }
    order = np.argsort(-probabilities, kind="stable")
    cutoffs = {}
    for name, requested_k in requested.items():
        effective_k = min(int(requested_k), int(labels.size))
        true_positives = int(labels[order[:effective_k]].sum())
        cutoffs[name] = {
            "requested_k": int(requested_k),
            "effective_k": effective_k,
            "true_positives": true_positives,
            "precision": true_positives / effective_k,
            "recall": true_positives / positive_count,
        }
    return {
        "length_definition": "Lmin=min(antibody_length,antigen_length)",
        "antibody_length": antibody_length,
        "antigen_length": antigen_length,
        "positive_count": positive_count,
        "candidate_count": int(labels.size),
        "cutoffs": cutoffs,
    }


def partner_retrieval_metrics(
    cognate_scores: Any,
    wrong_scores: Any,
    antigen_cluster_ids: Sequence[str],
) -> dict[str, Any]:
    cognate = np.asarray(cognate_scores, dtype=np.float64).reshape(-1)
    wrong = np.asarray(wrong_scores, dtype=np.float64)
    if wrong.ndim != 2 or wrong.shape[0] != cognate.size or wrong.shape[1] == 0:
        raise ValueError("wrong_scores must have shape [complex, noncognate_partner]")
    if len(antigen_cluster_ids) != cognate.size:
        raise ValueError("antigen_cluster_ids must align with complexes")
    if not np.isfinite(cognate).all() or not np.isfinite(wrong).all():
        raise ValueError("partner scores must be finite")
    margins = cognate - wrong.mean(axis=1)
    ranks = 1 + (wrong > cognate[:, None]).sum(axis=1) + 0.5 * (
        wrong == cognate[:, None]
    ).sum(axis=1)
    clusters: dict[str, list[int]] = {}
    for index, cluster in enumerate(antigen_cluster_ids):
        if not cluster:
            raise ValueError("antigen cluster identifiers must be nonempty")
        clusters.setdefault(str(cluster), []).append(index)
    cluster_margins = {
        cluster: float(margins[indices].mean())
        for cluster, indices in clusters.items()
    }
    return {
        "cluster_weighted_cognate_compatibility_margin": float(
            np.mean(list(cluster_margins.values()))
        ),
        "complex_weighted_cognate_compatibility_margin": float(margins.mean()),
        "top1_retrieval_accuracy": float((ranks == 1).mean()),
        "mean_reciprocal_rank": float((1.0 / ranks).mean()),
        "median_cognate_rank": float(np.median(ranks)),
        "candidate_count_per_query": int(wrong.shape[1] + 1),
        "complex_count": int(cognate.size),
        "antigen_cluster_count": len(clusters),
        "per_complex_margin": margins.tolist(),
        "per_complex_rank": ranks.tolist(),
        "cluster_margins": cluster_margins,
    }


def bootstrap_cluster_weighted_margin(
    cluster_margins: dict[str, float],
    *,
    replicates: int = 10_000,
    seed: int = 20_260_730,
) -> dict[str, float | int]:
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates <= 0:
        raise ValueError("replicates must be positive")
    values = np.asarray(list(cluster_margins.values()), dtype=np.float64)
    if values.size < 2 or not np.isfinite(values).all():
        raise ValueError("at least two finite antigen-cluster margins are required")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(values), size=(replicates, len(values)))
    samples = values[indices].mean(axis=1)
    return {
        "estimate": float(values.mean()),
        "ci95_lower": float(np.quantile(samples, 0.025)),
        "ci95_upper": float(np.quantile(samples, 0.975)),
        "replicates": replicates,
        "seed": seed,
        "cluster_count": int(len(values)),
    }


def bootstrap_group_weighted_values(
    values_by_instance: Mapping[str, float],
    group_by_instance: Mapping[str, str],
    *,
    replicates: int = 10_000,
    seed: int = 20_260_730,
) -> dict[str, Any]:
    """Bootstrap group means while retaining equal weight per unique group."""
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates <= 0:
        raise ValueError("replicates must be positive")
    if not values_by_instance:
        raise ValueError("values_by_instance cannot be empty")
    if not set(values_by_instance).issubset(group_by_instance):
        raise ValueError("Every scored instance must have a group identifier")
    grouped: dict[str, list[float]] = {}
    for identifier, raw_value in values_by_instance.items():
        value = float(raw_value)
        group = str(group_by_instance[identifier])
        if not math.isfinite(value) or not group:
            raise ValueError("Group bootstrap values and groups must be valid")
        grouped.setdefault(group, []).append(value)
    group_means = {
        group: float(np.mean(values)) for group, values in sorted(grouped.items())
    }
    values = np.asarray(list(group_means.values()), dtype=np.float64)
    if len(values) < 2:
        raise ValueError("At least two groups are required for bootstrap")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(values), size=(replicates, len(values)))
    samples = values[indices].mean(axis=1)
    return {
        "estimate": float(values.mean()),
        "ci95_lower": float(np.quantile(samples, 0.025)),
        "ci95_upper": float(np.quantile(samples, 0.975)),
        "replicates": replicates,
        "seed": seed,
        "eligible_complex_count": len(values_by_instance),
        "group_count": len(values),
        "group_means": group_means,
    }


__all__ = [
    "binary_classification_metrics",
    "bootstrap_cluster_weighted_margin",
    "bootstrap_group_weighted_values",
    "expected_calibration_error",
    "logistic_calibration_regression",
    "partner_retrieval_metrics",
    "positive_required_average_precision",
    "ranked_pair_metrics",
    "reliability_bins",
]
