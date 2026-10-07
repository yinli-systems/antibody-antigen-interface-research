"""Rank correlation and selection regret for checkpoint-selection signals.

The claim these support -- that a signal is or is not aligned with node
generalisation -- replaces an earlier one that inferred alignment from the size
of a gradient. That inference was wrong, so the replacement is worth pinning.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

from selection_signal_alignment import (  # noqa: E402
    MINIMUM_EPOCHS,
    analyse,
    read_trajectory,
    spearman,
)


def test_spearman_matches_the_reference_implementation():
    stats = pytest.importorskip("scipy.stats")
    first = [1.0, 3.0, 2.0, 5.0, 4.0]
    second = [2.0, 1.0, 4.0, 3.0, 5.0]
    assert spearman(first, second) == pytest.approx(stats.spearmanr(first, second).statistic)


def test_ties_get_average_ranks_rather_than_sort_order():
    stats = pytest.importorskip("scipy.stats")
    first = [1.0, 1.0, 2.0, 3.0, 3.0]
    second = [5.0, 4.0, 3.0, 2.0, 1.0]
    assert spearman(first, second) == pytest.approx(stats.spearmanr(first, second).statistic)


def test_a_flat_signal_has_no_correlation_rather_than_a_spurious_one():
    assert spearman([2.0, 2.0, 2.0, 2.0], [1.0, 2.0, 3.0, 4.0]) is None


def test_a_signal_that_orders_epochs_perfectly_has_no_regret():
    """A loss falling exactly as the metric rises picks the oracle epoch."""
    trajectory = {"metric": [0.10, 0.20, 0.30, 0.40],
                  "signals": {"total": [4.0, 3.0, 2.0, 1.0]}}
    result = analyse(trajectory)
    assert result["oracle_epoch"] == 4
    assert result["signals"]["total"]["chosen_epoch"] == 4
    assert result["signals"]["total"]["regret"] == pytest.approx(0.0)
    assert result["signals"]["total"]["spearman"] == pytest.approx(1.0)


def test_a_signal_anticorrelated_with_the_metric_pays_the_full_regret():
    """A loss that keeps falling while the metric falls picks the worst epoch."""
    trajectory = {"metric": [0.40, 0.30, 0.20, 0.10],
                  "signals": {"partner_rank": [4.0, 3.0, 2.0, 1.0]}}
    result = analyse(trajectory)
    signal = result["signals"]["partner_rank"]
    assert signal["chosen_epoch"] == 4
    assert signal["regret"] == pytest.approx(0.30)
    assert signal["spearman"] == pytest.approx(-1.0)


def test_losses_are_oriented_so_that_positive_always_means_aligned():
    """Every recorded signal is minimised; the sign convention must absorb that.

    Without the negation a perfectly useful loss would report rho = -1, and the
    reader would have to remember which signals were which way round.
    """
    falling = {"metric": [0.1, 0.2, 0.3], "signals": {"total": [3.0, 2.0, 1.0]}}
    assert analyse(falling)["signals"]["total"]["spearman"] == pytest.approx(1.0)


def test_regret_is_zero_only_at_the_oracle_and_never_negative():
    trajectory = {"metric": [0.20, 0.35, 0.25],
                  "signals": {"a": [1.0, 3.0, 2.0], "b": [3.0, 2.0, 1.0]}}
    result = analyse(trajectory)
    for signal in result["signals"].values():
        assert signal["regret"] >= 0.0
    assert result["signals"]["a"]["regret"] == pytest.approx(0.15)


def write_history(path: Path, epochs, target="node_epitope_mcc"):
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"history": [
        {"epoch": index + 1, "validation": {
            target: metric, "total": total, "partner_rank": rank,
            "edge": 1.0, "epitope_bce": 1.0,
        }}
        for index, (metric, total, rank) in enumerate(epochs)
    ]}), encoding="utf-8")


def test_a_trajectory_without_the_target_is_refused_not_defaulted(tmp_path):
    """Older runs recorded losses only; they must drop out, not read as zero."""
    history = tmp_path / "train" / "history.json"
    write_history(history, [(0.1, 3.0, 4.0), (0.2, 2.0, 3.0), (0.3, 1.0, 2.0)],
                  target="something_else")
    assert read_trajectory(history, "node_epitope_mcc") is None


def test_a_trajectory_too_short_to_correlate_is_dropped(tmp_path):
    history = tmp_path / "train" / "history.json"
    write_history(history, [(0.1, 3.0, 4.0), (0.2, 2.0, 3.0)][:MINIMUM_EPOCHS - 1])
    assert read_trajectory(history, "node_epitope_mcc") is None


def test_a_usable_trajectory_is_read_with_every_signal(tmp_path):
    history = tmp_path / "train" / "history.json"
    write_history(history, [(0.1, 3.0, 4.0), (0.2, 2.0, 3.0), (0.3, 1.0, 2.0)])
    trajectory = read_trajectory(history, "node_epitope_mcc")
    assert trajectory["metric"] == [0.1, 0.2, 0.3]
    assert set(trajectory["signals"]) == {"total", "partner_rank", "edge", "epitope_bce"}
