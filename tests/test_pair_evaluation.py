"""Tests for complete pair-level evaluation against baselines."""

from __future__ import annotations

import numpy as np
import pytest

from interaction_first.pair_evaluation import (
    ComplexPrediction,
    balanced_accuracy_metrics,
    bedroc_score,
    best_threshold_metrics,
    discrimination_metrics,
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


# --- early recognition ---------------------------------------------------


def _ranked(total: int = 10000, actives: int = 200):
    labels = np.zeros(total)
    labels[:actives] = 1.0
    perfect = np.concatenate([np.linspace(1, 0.9, actives), np.linspace(0.5, 0, total - actives)])
    worst = np.concatenate([np.linspace(0.1, 0, actives), np.linspace(1, 0.2, total - actives)])
    return labels, perfect, worst


def test_bedroc_is_one_for_a_perfect_ranking_and_zero_for_the_worst() -> None:
    labels, perfect, worst = _ranked()

    assert bedroc_score(labels, perfect) == pytest.approx(1.0, abs=1e-9)
    assert bedroc_score(labels, worst) == pytest.approx(0.0, abs=1e-9)


def test_bedroc_stays_inside_the_unit_interval() -> None:
    labels, perfect, worst = _ranked()
    random = np.random.default_rng(0).random(labels.size)

    for scores in (perfect, worst, random):
        assert 0.0 <= bedroc_score(labels, scores) <= 1.0


def test_bedroc_rewards_early_hits_more_than_auprc_does() -> None:
    """Two rankings with equal actives, one front-loaded: BEDROC separates them."""
    labels = np.zeros(1000)
    labels[:20] = 1.0
    early = np.zeros(1000)
    early[:20] = 1.0                       # all actives first
    late = np.zeros(1000)
    late[:20] = 0.5                        # actives mid-pack
    late[500:520] = 1.0                    # decoys ranked above them

    assert bedroc_score(labels, early) > bedroc_score(labels, late)


def test_bedroc_needs_both_classes() -> None:
    assert bedroc_score(np.zeros(10), np.ones(10)) is None
    assert bedroc_score(np.ones(10), np.ones(10)) is None
    assert bedroc_score(np.array([1.0, 0.0]), np.array([1.0, 0.0]), alpha=0.0) is None


def test_higher_alpha_concentrates_weight_further_forward() -> None:
    labels = np.zeros(1000)
    labels[:10] = 1.0
    scores = np.zeros(1000)
    scores[:10] = 0.9          # actives at the very top
    scores[10:] = np.linspace(0.8, 0, 990)

    assert bedroc_score(labels, scores, alpha=80.0) >= bedroc_score(labels, scores, alpha=8.0)


# --- balanced accuracy and thresholds ------------------------------------


def test_balanced_accuracy_averages_sensitivity_and_specificity() -> None:
    labels = np.array([1.0, 1.0, 0.0, 0.0])
    scores = np.array([0.9, 0.1, 0.9, 0.1])

    record = balanced_accuracy_metrics(labels, scores, threshold=0.5)

    assert record["sensitivity"] == pytest.approx(0.5)
    assert record["specificity"] == pytest.approx(0.5)
    assert record["balanced_accuracy"] == pytest.approx(0.5)


def test_balanced_accuracy_is_none_for_a_single_class() -> None:
    assert balanced_accuracy_metrics(np.ones(4), np.ones(4))["specificity"] is None


def test_best_threshold_finds_a_better_operating_point_than_a_fixed_one() -> None:
    """A fixed 0.5 conflates ranking quality with calibration."""
    labels = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    scores = np.array([0.30, 0.28, 0.10, 0.08, 0.05, 0.01])  # ranks perfectly, all below 0.5

    fixed = binary_metrics_f1(labels, scores, 0.5)
    best = best_threshold_metrics(labels, scores)

    assert fixed == pytest.approx(0.0)
    assert best["best_f1"] == pytest.approx(1.0)
    assert best["best_f1_threshold"] < 0.5


def binary_metrics_f1(labels, scores, threshold):
    from interaction_first.metrics import binary_metrics

    return binary_metrics(labels, scores, threshold=threshold)["f1"]


def test_best_threshold_is_none_without_positives() -> None:
    assert best_threshold_metrics(np.zeros(5), np.linspace(0, 1, 5))["best_f1"] is None


# --- cognate versus decoy discrimination ---------------------------------


def test_discrimination_separates_cognate_from_decoy() -> None:
    rng = np.random.default_rng(0)

    record = discrimination_metrics(rng.normal(3.0, 1.0, 200), rng.normal(0.0, 1.0, 2000))

    assert record["discrimination_auroc"] > 0.95
    assert record["cohens_d"] > 2.0
    assert record["cognate_count"] == 200
    assert record["decoy_count"] == 2000


def test_discrimination_is_chance_when_the_two_are_drawn_alike() -> None:
    rng = np.random.default_rng(1)

    record = discrimination_metrics(rng.normal(0, 1, 500), rng.normal(0, 1, 500))

    assert record["discrimination_auroc"] == pytest.approx(0.5, abs=0.06)


def test_discrimination_requires_both_sides() -> None:
    with pytest.raises(PairEvaluationError, match="both cognate and decoy"):
        discrimination_metrics([], [1.0, 2.0])


def test_new_metrics_appear_in_the_per_complex_record() -> None:
    report = evaluate_pair_predictions([_perfect()])
    record = report["per_complex"][0]

    for key in ("balanced_accuracy", "specificity", "best_f1", "bedroc_alpha20"):
        assert key in record
