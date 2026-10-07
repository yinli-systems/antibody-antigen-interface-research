"""How far a single-point node metric moves with its threshold.

Published epitope numbers are single points, and a point on a threshold sweep
says as much about the threshold as about the model. This bounds how much of a
gap the operating point can explain -- for the model being swept, and only for
that one.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

numpy = pytest.importorskip("numpy")

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
_SPEC = importlib.util.spec_from_file_location(
    "node_threshold_sensitivity_under_test",
    REPOSITORY_ROOT / "scripts" / "node_threshold_sensitivity.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

operating_points = _MODULE.operating_points
sweep = _MODULE.sweep


def test_a_perfectly_separated_split_reaches_one() -> None:
    labels = [False] * 50 + [True] * 50
    scores = [0.1] * 50 + [0.9] * 50

    best = max(row["mcc"] for row in sweep(labels, scores))

    assert best == pytest.approx(1.0)


def test_a_useless_score_stays_near_zero() -> None:
    labels = [n % 2 == 0 for n in range(100)]
    scores = [0.5] * 100

    assert max(abs(row["mcc"]) for row in sweep(labels, scores)) < 0.2


def test_the_sweep_covers_the_open_interval_without_degenerate_ends() -> None:
    """A threshold of 0 or 1 predicts everything or nothing and has no MCC."""
    rows = sweep([True, False], [0.9, 0.1])

    assert 0.0 < rows[0]["threshold"] < rows[-1]["threshold"] < 1.0


def test_prevalence_matching_predicts_as_many_positives_as_exist() -> None:
    """The one operating point two methods can share without agreeing on anything."""
    labels = [True] * 20 + [False] * 80
    scores = list(numpy.linspace(0.99, 0.01, 100))

    points = operating_points(sweep(labels, scores), labels)

    assert abs(points["prevalence_matched"]["predicted_positive"] - 20) <= 1


def test_the_fixed_point_is_the_one_nearest_a_half() -> None:
    labels = [True] * 20 + [False] * 80
    scores = list(numpy.linspace(0.99, 0.01, 100))

    points = operating_points(sweep(labels, scores), labels)

    assert points["fixed"]["threshold"] == pytest.approx(0.5, abs=0.01)


def test_the_best_point_is_at_least_the_fixed_one() -> None:
    """The distance between them is what the convention is worth."""
    labels = [True] * 20 + [False] * 80
    scores = list(numpy.linspace(0.99, 0.01, 100))

    points = operating_points(sweep(labels, scores), labels)

    assert points["best"]["mcc"] >= points["fixed"]["mcc"]


def test_a_score_distribution_shifted_high_moves_the_best_threshold_up() -> None:
    """The observed case: the optimum sits at 0.85 while the metric is read at 0.5."""
    labels = [True] * 20 + [False] * 80
    scores = [0.95] * 20 + [0.7] * 80

    points = operating_points(sweep(labels, scores), labels)

    assert points["best"]["threshold"] > 0.7
    assert points["fixed"]["mcc"] < points["best"]["mcc"]


def test_precision_and_recall_are_reported_at_each_point() -> None:
    """Without them a reader cannot tell which operating point a number came from."""
    labels = [True] * 20 + [False] * 80
    scores = list(numpy.linspace(0.99, 0.01, 100))

    points = operating_points(sweep(labels, scores), labels)

    for entry in points.values():
        assert 0.0 <= entry["precision"] <= 1.0
        assert 0.0 <= entry["recall"] <= 1.0


def test_no_positives_gives_a_defined_but_zero_correlation() -> None:
    """A degenerate split must not raise inside a sweep over many seeds."""
    rows = sweep([False] * 10, list(numpy.linspace(0.0, 1.0, 10)))

    assert all(row["mcc"] == 0.0 for row in rows)
