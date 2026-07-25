"""Tests for calibration, ranking, and paired uncertainty metrics."""

from __future__ import annotations

import math

import numpy as np
import pytest

from interaction_first.metrics import (
    calibration_metrics,
    contact_map_metrics,
    paired_cluster_bootstrap_mean_difference,
    partner_ranking_metrics,
    ranking_at_k,
    risk_coverage_metrics,
)


def test_ranking_at_k_reports_precision_and_recall() -> None:
    result = ranking_at_k(
        [1, 0, 1, 0],
        [0.9, 0.8, 0.7, 0.1],
        cutoffs=(1, 2, 4),
    )
    assert result["precision_at_1"] == 1.0
    assert result["recall_at_1"] == 0.5
    assert result["precision_at_2"] == 0.5
    assert result["recall_at_2"] == 0.5
    assert result["recall_at_4"] == 1.0


def test_ranking_at_k_fractional_boundary_is_permutation_invariant() -> None:
    labels = np.asarray([1, 0, 1, 0])
    scores = np.asarray([0.9, 0.8, 0.8, 0.1])
    first = ranking_at_k(labels, scores, cutoffs=(2,))
    permutation = np.asarray([2, 0, 3, 1])
    second = ranking_at_k(
        labels[permutation],
        scores[permutation],
        cutoffs=(2,),
    )

    assert first == second
    assert first["precision_at_2"] == pytest.approx(0.75)
    assert first["recall_at_2"] == pytest.approx(0.75)
    assert first["expected_true_positive_at_2"] == pytest.approx(1.5)
    assert first["boundary_fractional_weight_at_2"] == pytest.approx(0.5)
    assert first["tie_policy"] == (
        "fractional_expected_credit_at_score_boundary"
    )


def test_precision_at_true_contact_count_is_tie_invariant() -> None:
    labels = np.asarray([1, 1, 0, 0])
    scores = np.asarray([0.9, 0.5, 0.5, 0.1])
    first = contact_map_metrics(labels, scores)
    permutation = np.asarray([2, 0, 3, 1])
    second = contact_map_metrics(labels[permutation], scores[permutation])

    assert first["precision_at_true_contact_count"] == pytest.approx(0.75)
    assert second["precision_at_true_contact_count"] == pytest.approx(0.75)
    assert first["precision_at_true_contact_count_tie_policy"] == (
        "fractional_expected_credit_at_score_boundary"
    )


def test_ranking_at_k_rejects_duplicate_or_invalid_cutoffs() -> None:
    with pytest.raises(ValueError, match="unique"):
        ranking_at_k([0, 1], [0.1, 0.9], cutoffs=(1, 1))
    with pytest.raises(ValueError, match="positive integers"):
        ranking_at_k([0, 1], [0.1, 0.9], cutoffs=(0,))


def test_calibration_metrics_match_an_exact_small_example() -> None:
    result = calibration_metrics(
        [0, 0, 1, 1],
        [0.1, 0.3, 0.7, 0.9],
        bins=2,
    )
    assert result["brier_score"] == pytest.approx(0.05)
    assert result["negative_log_likelihood"] == pytest.approx(
        -(
            math.log(0.9)
            + math.log(0.7)
            + math.log(0.7)
            + math.log(0.9)
        )
        / 4.0,
    )
    assert result["expected_calibration_error"] == pytest.approx(0.2)
    assert result["maximum_calibration_error"] == pytest.approx(0.2)
    assert result["populated_bin_count"] == 2


def test_adaptive_calibration_preserves_probability_ties() -> None:
    result = calibration_metrics(
        [0, 1, 0, 1],
        [0.5, 0.5, 0.5, 0.5],
        bins=3,
        strategy="adaptive",
    )
    assert result["populated_bin_count"] == 1
    assert result["expected_calibration_error"] == 0.0
    assert result["bins"][0]["count"] == 4


def test_calibration_metrics_reject_nonprobabilities() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        calibration_metrics([0, 1], [-0.1, 1.1])
    with pytest.raises(ValueError, match="strategy"):
        calibration_metrics([0, 1], [0.1, 0.9], strategy="quantile")


def test_risk_coverage_selects_the_most_confident_predictions_first() -> None:
    result = risk_coverage_metrics(
        [0, 1, 1, 0],
        [0.01, 0.99, 0.51, 0.90],
        coverages=(0.5, 1.0),
    )
    assert result["points"][0]["selected_count"] == 2
    assert result["points"][0]["classification_risk"] == 0.0
    assert result["points"][1]["classification_risk"] == 0.25
    assert result["classification_decision_threshold"] == 0.5
    assert result["selection_tie_policy"] == (
        "include_all_boundary_confidence_ties"
    )


def test_risk_coverage_includes_all_boundary_ties_independent_of_order() -> None:
    labels = np.asarray([1, 0, 1, 0])
    probabilities = np.asarray([0.75, 0.25, 0.625, 0.375])
    first = risk_coverage_metrics(
        labels,
        probabilities,
        coverages=(0.25,),
    )
    permutation = np.asarray([1, 3, 0, 2])
    second = risk_coverage_metrics(
        labels[permutation],
        probabilities[permutation],
        coverages=(0.25,),
    )

    assert first == second
    point = first["points"][0]
    assert point["requested_minimum_count"] == 1
    assert point["selected_count"] == 2
    assert point["realized_coverage"] == pytest.approx(0.5)
    assert point["selection_threshold_comparator"] == "greater_than_or_equal"


def test_risk_coverage_uses_one_explicit_decision_threshold_semantics() -> None:
    result = risk_coverage_metrics(
        [0, 1, 1],
        [0.69, 0.70, 0.90],
        coverages=(1.0,),
        decision_threshold=0.7,
    )
    assert result["classification_decision_threshold"] == pytest.approx(0.7)
    assert result["classification_decision_rule"] == (
        "probability_greater_than_or_equal_to_threshold_predicts_positive"
    )
    assert result["points"][0]["classification_risk"] == 0.0


def test_risk_coverage_rejects_duplicate_or_invalid_coverages() -> None:
    with pytest.raises(ValueError, match="unique"):
        risk_coverage_metrics([0, 1], [0.1, 0.9], coverages=(0.5, 0.5))
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        risk_coverage_metrics([0, 1], [0.1, 0.9], coverages=(0.0,))


def test_partner_ranking_includes_mrr_and_normalized_rank() -> None:
    result = partner_ranking_metrics(
        matched_scores=[0.8, 0.2],
        negative_scores=[[0.1, 0.0], [0.9, 0.1]],
    )
    assert result["mean_reciprocal_rank"] == pytest.approx(0.75)
    assert result["mean_normalized_rank"] == pytest.approx(0.25)


def test_paired_cluster_bootstrap_is_reproducible_and_paired() -> None:
    candidate = {"a": 0.8, "b": 0.7, "c": 0.6}
    reference = {"a": 0.3, "b": 0.4, "c": 0.5}
    clusters = {"a": "x", "b": "x", "c": "y"}
    first = paired_cluster_bootstrap_mean_difference(
        candidate,
        reference,
        clusters,
        iterations=200,
        seed=11,
        insufficient_cluster_policy="mark_non_authorizing",
    )
    second = paired_cluster_bootstrap_mean_difference(
        candidate,
        reference,
        clusters,
        iterations=200,
        seed=11,
        insufficient_cluster_policy="mark_non_authorizing",
    )
    assert first == second
    assert first["candidate_mean"] == pytest.approx(0.7)
    assert first["reference_mean"] == pytest.approx(0.4)
    assert first["mean_difference"] == pytest.approx(0.3)
    assert first["inferential_authorized"] is False
    assert math.isfinite(float(first["lower"]))
    assert math.isfinite(float(first["upper"]))


def test_paired_cluster_bootstrap_rejects_unpaired_ids() -> None:
    with pytest.raises(ValueError, match="identical instance IDs"):
        paired_cluster_bootstrap_mean_difference(
            {"a": 0.1},
            {"b": 0.1},
            {"a": "x"},
            iterations=10,
        )


def test_paired_cluster_bootstrap_rejects_normalized_identifier_collisions() -> None:
    with pytest.raises(ValueError, match="normalized identifier collision"):
        paired_cluster_bootstrap_mean_difference(
            {"a": 0.2, " a ": 0.3},
            {"a": 0.1, " a ": 0.1},
            {"a": "x", " a ": "y"},
            iterations=10,
        )


def test_paired_cluster_bootstrap_rejects_normalized_cluster_collisions() -> None:
    with pytest.raises(ValueError, match="normalized cluster collision"):
        paired_cluster_bootstrap_mean_difference(
            {"a": 0.2, "b": 0.3},
            {"a": 0.1, "b": 0.1},
            {"a": "cluster", "b": " cluster "},
            iterations=10,
        )


def test_paired_cluster_bootstrap_rejects_lowering_validated_cluster_floor() -> None:
    with pytest.raises(ValueError, match="validated floor of 10"):
        paired_cluster_bootstrap_mean_difference(
            {"a": 0.2, "b": 0.3},
            {"a": 0.1, "b": 0.1},
            {"a": "x", "b": "y"},
            iterations=10,
            minimum_independent_clusters=2,
        )
