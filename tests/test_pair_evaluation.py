"""Tests for complete pair-level evaluation against baselines."""

from __future__ import annotations

import numpy as np
import pytest

from interaction_first.pair_evaluation import (
    ComplexPrediction,
    PairEvaluationError,
    enrichment_factor,
    evaluate_pair_predictions,
    evaluate_single,
    outer_product_scores,
    score_separation_stats,
)


def _perfect(size: int = 4) -> ComplexPrediction:
    labels = np.zeros((size, size))
    labels[0, 0] = labels[1, 1] = 1.0
    return ComplexPrediction("perfect", labels, labels.copy())


# --- the baselines -------------------------------------------------------


def test_outer_product_is_the_product_of_the_two_marginals() -> None:
    grid = outer_product_scores([0.5, 1.0], [0.2, 0.4, 0.6])

    assert grid.shape == (2, 3)
    assert grid[1, 2] == pytest.approx(0.6)
    assert grid[0, 0] == pytest.approx(0.1)


def test_outer_product_carries_no_partner_specific_information() -> None:
    """Its ranking is determined entirely by the two marginals."""
    grid = outer_product_scores([0.9, 0.1], [0.9, 0.1])

    # The best pair is the two most interface-prone residues, whatever pairs.
    assert grid.argmax() == 0
    assert grid[0, 1] == pytest.approx(grid[1, 0])


def test_a_constant_score_scores_average_precision_equal_to_prevalence() -> None:
    labels = np.array([1.0, 0.0, 0.0, 0.0])

    record = evaluate_single(labels, np.zeros(4))

    assert record["auprc"] == pytest.approx(record["prevalence"])


def test_prevalence_baseline_is_reported_and_equals_the_positive_rate() -> None:
    report = evaluate_pair_predictions([_perfect()])

    assert report["baselines"]["prevalence"]["macro"]["auprc"] == pytest.approx(2 / 16)


def test_outer_product_baseline_appears_only_with_marginals() -> None:
    without = evaluate_pair_predictions([_perfect()])
    labels = np.zeros((2, 2))
    labels[0, 0] = 1.0
    with_marginals = evaluate_pair_predictions(
        [
            ComplexPrediction(
                "c", labels, labels.copy(),
                epitope_probability=[0.9, 0.1], paratope_probability=[0.9, 0.1],
            )
        ]
    )

    assert "outer_product" not in without["baselines"]
    assert "outer_product" in with_marginals["baselines"]


def test_a_model_that_only_reproduces_marginals_does_not_beat_the_outer_product() -> None:
    """The inversion the project measures: no partner information, no gain."""
    epitope, paratope = np.array([0.9, 0.2, 0.1]), np.array([0.8, 0.3])
    scores = np.outer(epitope, paratope)
    labels = np.zeros((3, 2))
    labels[0, 0] = labels[2, 1] = 1.0

    report = evaluate_pair_predictions(
        [
            ComplexPrediction(
                "c", labels, scores,
                epitope_probability=epitope, paratope_probability=paratope,
            )
        ]
    )

    assert report["versus_baseline"]["outer_product"]["macro_auprc"] == pytest.approx(0.0)
    assert report["versus_baseline"]["outer_product"]["model_beats_baseline_macro_auprc"] is False


def test_a_partner_specific_model_does_beat_the_outer_product() -> None:
    epitope, paratope = np.array([0.5, 0.5]), np.array([0.5, 0.5])
    labels = np.array([[1.0, 0.0], [0.0, 1.0]])  # diagonal: pure partner specificity

    report = evaluate_pair_predictions(
        [
            ComplexPrediction(
                "c", labels, labels.copy(),
                epitope_probability=epitope, paratope_probability=paratope,
            )
        ]
    )

    assert report["versus_baseline"]["outer_product"]["model_beats_baseline_macro_auprc"] is True


# --- aggregation ---------------------------------------------------------


def test_macro_weights_every_complex_equally_and_micro_does_not() -> None:
    small = np.zeros((2, 2))
    small[0, 0] = 1.0
    large = np.zeros((10, 10))
    large[0, 0] = 1.0
    predictions = [
        ComplexPrediction("small", small, small.copy()),          # perfect
        ComplexPrediction("large", large, -large.copy()),         # inverted
    ]

    report = evaluate_pair_predictions(predictions)

    # Macro averages one good and one bad complex; micro is dominated by the
    # 100-pair complex, so the two disagree.
    assert report["macro"]["auprc"] != pytest.approx(report["micro"]["auprc"])
    assert report["complex_count"] == 2
    assert len(report["per_complex"]) == 2


def test_aggregate_records_how_many_complexes_reported_each_metric() -> None:
    report = evaluate_pair_predictions([_perfect(), _perfect()])

    assert report["macro"]["auprc_reported_by"] == 2


def test_per_complex_records_carry_their_identifier() -> None:
    report = evaluate_pair_predictions([_perfect()])

    assert report["per_complex"][0]["identifier"] == "perfect"


# --- candidate mask ------------------------------------------------------


def test_masked_pairs_are_excluded_rather_than_counted_as_negatives() -> None:
    labels = np.zeros((2, 2))
    labels[0, 0] = 1.0
    mask = np.array([[True, True], [False, False]])

    record = evaluate_pair_predictions(
        [ComplexPrediction("c", labels, labels.copy(), candidate_mask=mask)]
    )["per_complex"][0]

    assert record["candidate_count"] == 2
    assert record["prevalence"] == pytest.approx(0.5)


def test_an_empty_candidate_mask_is_rejected() -> None:
    labels = np.zeros((2, 2))
    with pytest.raises(PairEvaluationError, match="selects no pair"):
        evaluate_pair_predictions(
            [ComplexPrediction("c", labels, labels, candidate_mask=np.zeros((2, 2), dtype=bool))]
        )


def test_shape_disagreements_are_rejected() -> None:
    with pytest.raises(PairEvaluationError, match="differ"):
        evaluate_pair_predictions(
            [ComplexPrediction("c", np.zeros((2, 2)), np.zeros((3, 3)))]
        )
    with pytest.raises(PairEvaluationError, match="two-dimensional"):
        evaluate_pair_predictions([ComplexPrediction("c", np.zeros(4), np.zeros(4))])


def test_a_mismatched_outer_product_is_rejected() -> None:
    labels = np.zeros((2, 2))
    labels[0, 0] = 1.0
    with pytest.raises(PairEvaluationError, match="does not match labels"):
        evaluate_pair_predictions(
            [
                ComplexPrediction(
                    "c", labels, labels.copy(),
                    epitope_probability=[0.5, 0.5, 0.5],  # wrong length
                    paratope_probability=[0.5, 0.5],
                )
            ]
        )


def test_no_predictions_is_rejected() -> None:
    with pytest.raises(PairEvaluationError, match="at least one complex"):
        evaluate_pair_predictions([])


# --- individual metrics --------------------------------------------------


def test_enrichment_is_one_for_random_and_the_reciprocal_fraction_at_best() -> None:
    labels = np.zeros(100)
    labels[:10] = 1.0
    perfect = enrichment_factor(labels, labels, fractions=(0.10,))
    constant = enrichment_factor(labels, np.zeros(100), fractions=(0.10,))

    assert perfect["ef_at_10pct"] == pytest.approx(10.0)
    assert constant["ef_at_10pct"] == pytest.approx(1.0)


def test_enrichment_is_tie_aware() -> None:
    """A tie group crossing the cutoff must not be resolved by input order."""
    labels = np.array([1.0, 0.0, 1.0, 0.0])
    forward = enrichment_factor(labels, np.zeros(4), fractions=(0.5,))
    reversed_order = enrichment_factor(labels[::-1], np.zeros(4), fractions=(0.5,))

    assert forward["ef_at_50pct"] == pytest.approx(reversed_order["ef_at_50pct"])


def test_enrichment_is_none_without_positives() -> None:
    assert enrichment_factor(np.zeros(10), np.ones(10))["ef_at_1pct"] is None


def test_cohens_d_is_zero_when_the_classes_score_alike() -> None:
    labels = np.array([1.0, 0.0, 1.0, 0.0])
    scores = np.array([0.5, 0.5, 0.6, 0.6])

    assert score_separation_stats(labels, scores)["cohens_d"] == pytest.approx(0.0)


def test_cohens_d_is_positive_when_positives_score_higher() -> None:
    labels = np.array([1.0, 1.0, 0.0, 0.0])
    scores = np.array([0.9, 0.8, 0.2, 0.1])

    stats = score_separation_stats(labels, scores)

    assert stats["cohens_d"] > 1.0
    assert stats["score_gap"] == pytest.approx(0.7)


def test_separation_is_none_for_a_single_class() -> None:
    assert score_separation_stats(np.ones(4), np.ones(4))["cohens_d"] is None


def test_precision_at_the_true_count_is_reported() -> None:
    record = evaluate_single(np.array([1.0, 1.0, 0.0, 0.0]), np.array([0.9, 0.8, 0.2, 0.1]))

    assert record["precision_at_true_count"] == pytest.approx(1.0)


def test_cutoffs_larger_than_the_candidate_set_are_skipped() -> None:
    record = evaluate_single(np.array([1.0, 0.0]), np.array([0.9, 0.1]))

    assert "precision_at_1" in record
    assert "precision_at_100" not in record
