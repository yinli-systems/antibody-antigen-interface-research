"""The quantity and the rule behind the `d_model` choice.

This choice is made once, before the single registered ratio-test reading, and
the protocol states the rule so that it cannot be exercised after the fact. What
can still go wrong is the arithmetic underneath it, so that is pinned here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

from aggregate_selection_metric_confirmation import (  # noqa: E402
    ROLE,
    macro_mcc_curve,
    matthews,
)


def test_matthews_agrees_with_the_reference_implementation():
    sklearn = pytest.importorskip("sklearn.metrics")
    truth = [1, 1, 0, 0, 1, 0, 1, 0, 0, 1]
    predicted = [1, 0, 0, 1, 1, 0, 1, 0, 1, 1]
    true_positive = sum(1 for t, p in zip(truth, predicted) if t and p)
    false_positive = sum(1 for t, p in zip(truth, predicted) if not t and p)
    true_negative = sum(1 for t, p in zip(truth, predicted) if not t and not p)
    false_negative = sum(1 for t, p in zip(truth, predicted) if t and not p)
    assert matthews(true_positive, false_positive, true_negative, false_negative) == \
        pytest.approx(sklearn.matthews_corrcoef(truth, predicted))


def test_a_prediction_with_no_negatives_scores_zero_rather_than_raising():
    """An empty margin makes MCC undefined; the convention here is zero.

    Per-complex averaging means one degenerate complex would otherwise take the
    whole split's number down with a NaN.
    """
    assert matthews(5, 0, 0, 0) == 0.0
    assert matthews(0, 0, 5, 0) == 0.0


def frame(rows):
    pandas = pytest.importorskip("pandas")
    return pandas.DataFrame(
        [{"instance_id": i, "role": ROLE, "res_idx": n, "label": l, "score": s}
         for i, entries in rows.items() for n, (l, s) in enumerate(entries)]
    )


def test_complexes_are_averaged_not_pooled():
    """A big easy complex must not outvote a small hard one.

    Pooling would let the 100-residue complex dominate; per-complex averaging
    gives the two equal weight, and the protocol names the per-complex figure.
    """
    easy = [(n < 50, 0.9 if n < 50 else 0.1) for n in range(100)]
    hard = [(n < 2, 0.4 if n < 2 else 0.6) for n in range(4)]
    _, curve = macro_mcc_curve(frame({"easy": easy, "hard": hard}))
    best = max(curve)
    assert best == pytest.approx(0.5, abs=0.02), (
        "a perfect complex and a fully inverted one should average to about half"
    )


def test_the_curve_finds_a_threshold_the_fixed_convention_misses():
    """Scores squeezed below 0.5 score zero at 0.5 and well above it elsewhere."""
    rows = {name: [(n < 3, 0.30 if n < 3 else 0.10) for n in range(10)]
            for name in ("a", "b", "c")}
    thresholds, curve = macro_mcc_curve(frame(rows))
    at_half = curve[min(range(len(thresholds)), key=lambda i: abs(thresholds[i] - 0.5))]
    assert at_half == 0.0
    assert max(curve) == pytest.approx(1.0)


def choose(means, interval):
    """The protocol's rule, restated here so the test does not import the printout."""
    decided = interval[0] > 0.0 or interval[1] < 0.0
    return max((128, 256), key=lambda width: means[width]) if decided else 128


def test_an_undecided_difference_takes_the_smaller_model_even_when_it_looks_worse():
    assert choose({128: 0.2600, 256: 0.2845}, (-0.004, 0.052)) == 128


def test_a_decided_difference_takes_the_argmax_in_either_direction():
    assert choose({128: 0.2600, 256: 0.2845}, (0.008, 0.041)) == 256
    assert choose({128: 0.2845, 256: 0.2600}, (-0.041, -0.008)) == 128


def test_an_interval_touching_zero_counts_as_undecided():
    """`lo > 0 or hi < 0` is strict; a bound exactly at zero does not decide."""
    assert choose({128: 0.26, 256: 0.29}, (0.0, 0.06)) == 128
