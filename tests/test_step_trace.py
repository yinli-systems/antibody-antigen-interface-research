"""Locating a sign change in optimisation steps.

The measurement exists because two epoch-level means disagreed, so the rule that
reads it has to be stricter than "the average went positive". A single noisy
step must not declare a crossing, and a crossing must not be declared where
nothing was ever negative.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

from aggregate_step_trace import (  # noqa: E402
    COSINE_KEY,
    EARLY_FRACTION,
    OPENING,
    analyse,
    classify,
    read_trace,
    smooth,
)


def crossing_series(negative_for=60, total=400, low=-0.08, high=0.55):
    return [low] * negative_for + [high] * (total - negative_for)


def test_the_trailing_mean_starts_at_the_first_value():
    """Otherwise the opening steps would be diluted by a window of nothing."""
    smoothed = smooth([1.0, 1.0, 1.0], window=10)
    assert smoothed[0] == pytest.approx(1.0)
    assert smoothed[2] == pytest.approx(1.0)


def test_a_run_that_starts_negative_and_stays_positive_has_a_crossing():
    result = analyse(crossing_series())
    assert result["starts_negative"]
    assert result["crossed"]
    assert 60 <= result["crossing_step"] <= 75, "the crossing lands near the change"


def test_a_run_that_is_positive_throughout_reports_no_crossing():
    result = analyse([0.5] * 300)
    assert not result["starts_negative"]
    assert result["crossing_step"] is None
    assert not result["crossed"]


def test_a_single_negative_step_late_on_does_not_undo_a_crossing_forever():
    """A crossing is durable, so one dip inside the smoothing window is absorbed."""
    series = crossing_series(negative_for=40, total=400)
    series[300] = -0.6
    result = analyse(series)
    assert result["crossed"], "one step against nine positive ones is not a reversal"


def test_a_sustained_return_to_conflict_pushes_the_crossing_late():
    """Recovering near the end is a crossing, but not an early transient.

    The crossing is defined as the point after which conflict never returns, so
    a long negative stretch at step 300 moves it there rather than removing it.
    What must not happen is that this gets classified alongside a run which
    settles in the first few dozen steps.
    """
    series = crossing_series(negative_for=40, total=400)
    series[300:340] = [-0.4] * 40
    result = analyse(series)
    assert result["crossed"]
    assert result["crossing_step"] > 300, "forty consecutive negative steps is not noise"
    verdict = classify({f"s{i}": result for i in range(3)})
    assert verdict["verdict"] == "late crossing"
    assert verdict["seeds_crossing_early"] == 0


def test_the_opening_window_is_the_one_the_smoke_covered():
    """The question came from a 40-step run, so 40 steps is what is compared."""
    assert OPENING == 40
    result = analyse([-0.05] * 40 + [0.6] * 360)
    assert result["opening_mean"] == pytest.approx(-0.05)


def test_every_seed_crossing_is_called_transient():
    per_seed = {f"s{i}": analyse(crossing_series()) for i in range(3)}
    verdict = classify(per_seed)
    assert verdict["verdict"] == "transient"
    assert verdict["seeds_crossing"] == 3


def test_no_seed_opening_negative_indicts_the_smoke_not_the_model():
    per_seed = {f"s{i}": analyse([0.5] * 300) for i in range(3)}
    verdict = classify(per_seed)
    assert verdict["verdict"] == "positive throughout"
    assert "never described training" in verdict["reason"]


def test_persistent_sign_changes_without_a_crossing_are_called_oscillating():
    # Biased slightly negative: a perfectly balanced alternation averages to
    # exactly zero, which is not negative, and the fixture would then be testing
    # the positive-throughout branch instead.
    alternating = [0.3 if index % 2 else -0.4 for index in range(400)]
    per_seed = {f"s{i}": analyse(alternating) for i in range(3)}
    verdict = classify(per_seed)
    assert verdict["verdict"] == "oscillating"


def test_seeds_disagreeing_are_reported_as_mixed_not_averaged_into_a_story():
    per_seed = {"s0": analyse(crossing_series()), "s1": analyse([0.5] * 300),
                "s2": analyse(crossing_series())}
    verdict = classify(per_seed)
    assert verdict["verdict"] == "mixed"
    assert verdict["seeds_crossing"] == 2
    assert verdict["seeds_opening_negative"] == 2


def test_a_trace_without_the_cosine_is_skipped(tmp_path):
    path = tmp_path / "balancer_trace.json"
    path.write_text(json.dumps({"steps": [{"something_else": 0.1}]}), encoding="utf-8")
    assert read_trace(path) is None


def test_a_usable_trace_is_read_in_step_order(tmp_path):
    path = tmp_path / "balancer_trace.json"
    path.write_text(json.dumps({"steps": [{COSINE_KEY: v} for v in (-0.1, 0.2, 0.4)]}),
                    encoding="utf-8")
    assert read_trace(path) == [-0.1, 0.2, 0.4]


def test_an_early_crossing_and_a_late_one_are_not_the_same_verdict():
    """The distinction the late-crossing branch exists to make."""
    early = {f"s{i}": analyse(crossing_series(negative_for=60, total=400))
             for i in range(3)}
    assert classify(early)["verdict"] == "transient"
    assert all(step <= EARLY_FRACTION * 400 for step in classify(early)["crossing_steps"])


def test_a_transient_shorter_than_the_opening_window_is_still_detected():
    """The blind spot the short probe closes.

    Twenty negative steps followed by positive ones leaves the 40-step mean
    positive, so a rule reading only that window would record the run as never
    having started in conflict -- missing exactly what it was built to find.
    """
    series = crossing_series(negative_for=20, total=400, low=-0.30, high=0.55)
    assert sum(series[:40]) / 40 > 0.0, "the fixture must defeat the long window"
    result = analyse(series)
    assert result["short_opening_mean"] < 0.0
    assert result["starts_negative"], "the short probe has to catch it"
    assert result["crossed"]
