"""The corruption ladder's classification rule.

The curve itself will be read once. The rule that turns it into ``cliff``,
``slope`` or ``coarse-only`` therefore has to be right before the reading, not
adjusted after it, so it is exercised here on constructed ladders whose intended
answer is obvious.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

from aggregate_corruption_curve import (  # noqa: E402
    CLIFF_FRACTION,
    GLOBAL_KEY,
    SLOPE_FRACTION,
    classify,
    read_metric,
    summarise,
)


def ladder(means):
    """A ladder where every named k is damaged by the stated mean amount.

    Keyed by integer k, so it is passed as a mapping rather than as keywords.
    """
    return {
        k: {"mean_difference": value, "damaged": value < 0.0,
            "bootstrap_ci95": [value - 0.001, value - 0.0005]}
        for k, value in means.items()
    }


def test_a_one_residue_error_costing_most_of_everything_is_a_cliff():
    verdict = classify(ladder({1: -0.090, 2: -0.095, 4: -0.098, 8: -0.099,
                                 16: -0.100, GLOBAL_KEY: -0.100}))
    assert verdict["verdict"] == "cliff"
    assert verdict["k1_fraction_of_global"] >= CLIFF_FRACTION


def test_damage_that_accumulates_with_displacement_is_a_slope():
    verdict = classify(ladder({1: -0.010, 2: -0.025, 4: -0.045, 8: -0.070,
                                 16: -0.090, GLOBAL_KEY: -0.100}))
    assert verdict["verdict"] == "slope"
    assert verdict["monotone"]


def test_a_harmless_single_residue_error_is_coarse_only():
    points = ladder({1: -0.001, 2: -0.030, 4: -0.060, 8: -0.080,
                       16: -0.090, GLOBAL_KEY: -0.100})
    points[1]["damaged"] = False
    assert classify(points)["verdict"] == "coarse-only"


def test_the_band_between_the_boundaries_is_reported_not_resolved():
    """0.5D < |d1| < 0.8D matches none of the three predictions.

    The protocol says that outcome is reported as intermediate rather than
    settled by moving a boundary, so the rule must not quietly round it into
    the nearer named class.
    """
    verdict = classify(ladder({1: -0.065, 2: -0.075, 4: -0.085, 8: -0.090,
                                 16: -0.095, GLOBAL_KEY: -0.100}))
    assert verdict["verdict"] == "intermediate"
    assert SLOPE_FRACTION < verdict["k1_fraction_of_global"] < CLIFF_FRACTION


def test_a_small_k1_with_a_non_monotone_ladder_is_not_called_a_slope():
    verdict = classify(ladder({1: -0.010, 2: -0.060, 4: -0.030, 8: -0.070,
                                 16: -0.090, GLOBAL_KEY: -0.100}))
    assert verdict["verdict"] == "intermediate"
    assert not verdict["monotone"]
    assert verdict["monotonicity_violations"][0]["from_k"] == 2


def test_monotonicity_violations_are_printed_with_whether_they_are_noise():
    points = ladder({1: -0.010, 2: -0.060, 4: -0.030, 8: -0.070,
                       16: -0.090, GLOBAL_KEY: -0.100})
    points[4]["bootstrap_ci95"] = [-0.050, 0.010]
    violation = classify(points)["monotonicity_violations"][0]
    assert violation["to_k"] == 4
    assert violation["covers_zero"] is True


def test_a_global_permutation_that_does_no_damage_leaves_nothing_to_scale_by():
    verdict = classify(ladder({1: -0.010, GLOBAL_KEY: 0.0}))
    assert verdict["verdict"] == "not classifiable"


def test_an_incomplete_ladder_is_refused_rather_than_guessed():
    assert classify(ladder({2: -0.05, GLOBAL_KEY: -0.1}))["verdict"] == "not classifiable"
    assert classify(ladder({1: -0.05}))["verdict"] == "not classifiable"


def test_the_endpoint_is_read_from_the_pair_macro_block(tmp_path):
    """Receipts nest the endpoint under pair.macro; a sibling must not shadow it."""
    receipt = tmp_path / "evaluation.json"
    receipt.write_text(
        '{"nodes": {"epitope": {"auprc": 0.9}}, "pair": {"macro": {"auprc": 0.25},'
        ' "micro": {"auprc": 0.31}}}',
        encoding="utf-8",
    )
    assert read_metric(receipt, "auprc") == 0.25


def test_an_unreadable_receipt_yields_nothing_rather_than_a_zero(tmp_path):
    """A missing seed must drop out of the pairing, not enter it as no damage."""
    assert read_metric(tmp_path / "absent.json", "auprc") is None
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    assert read_metric(broken, "auprc") is None


def test_the_pairing_counts_only_seeds_present_on_both_sides():
    rows = [{"seed": 1, "difference": -0.02}, {"seed": 2, "difference": -0.03},
            {"seed": 3, "difference": 0.01}]
    summary = summarise(rows)
    assert summary["seed_count"] == 3
    assert summary["negative_seed_count"] == 2
    assert summary["seeds"] == [1, 2, 3]
    assert not summary["damaged"], "two of three negative cannot clear the sign bound"
