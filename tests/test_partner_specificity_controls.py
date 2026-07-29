"""Focused tests for hard-negative partner-specificity diagnostics."""

from __future__ import annotations

import importlib.util

import pytest

from interaction_first.partner_controls import (
    partner_specificity_control_metrics,
    validate_partner_score_support,
)


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("numpy") is None,
    reason="NumPy is unavailable",
)


def test_named_controls_use_conservative_tie_ranks() -> None:
    result = partner_specificity_control_metrics(
        matched_scores=[3.0, 2.0, 1.0],
        controls_by_name={
            "random_derangement": [
                [1.0, 2.0],
                [1.0, 2.0],
                [0.0, 0.5],
            ],
            "similarity_matched": [2.5, 2.5, 1.5],
        },
    )

    random_metrics = result["by_control"]["random_derangement"]
    assert random_metrics["recall_at_1"] == pytest.approx(2.0 / 3.0)
    assert random_metrics["matched_tie_rate"] == pytest.approx(1.0 / 3.0)
    assert random_metrics["mean_rank"] == pytest.approx(4.0 / 3.0)

    strict_metrics = result["all_controls"]
    assert strict_metrics["control_candidates_per_instance"] == 3
    assert strict_metrics["recall_at_1"] == pytest.approx(1.0 / 3.0)
    assert strict_metrics["matched_strict_win_rate"] == pytest.approx(1.0 / 3.0)
    assert strict_metrics["mean_matched_minus_hardest_control"] == pytest.approx(
        (0.5 - 0.5 - 0.5) / 3.0
    )


def test_partner_control_inputs_fail_closed() -> None:
    with pytest.raises(ValueError, match="At least one"):
        partner_specificity_control_metrics([1.0], {})
    with pytest.raises(ValueError, match="shape"):
        partner_specificity_control_metrics(
            [1.0, 2.0],
            {"misaligned": [[0.0, 1.0]]},
        )
    with pytest.raises(ValueError, match="finite"):
        partner_specificity_control_metrics(
            [1.0, 2.0],
            {"nonfinite": [0.0, float("nan")]},
        )


def test_candidate_support_audit_exposes_incomparable_controls() -> None:
    result = validate_partner_score_support(
        matched_candidate_counts=[100, 80, 120],
        control_candidate_counts={
            "exact": [100, 80, 120],
            "mismatched": [200, 80, 60],
        },
    )

    assert not result["all_controls_exact_support_match"]
    assert result["by_control"]["exact"]["exact_support_match"]
    assert not result["by_control"]["mismatched"]["exact_support_match"]
    assert result["by_control"]["mismatched"][
        "matched_instance_fraction"
    ] == pytest.approx(1.0 / 3.0)
    assert result["by_control"]["mismatched"][
        "maximum_relative_count_gap"
    ] == pytest.approx(0.5)
