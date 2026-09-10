"""The transition criterion for gradient alignment over training.

Two measurements of one cosine disagree in sign. The rule that decides whether
training stage explains that is applied once, so it is exercised here on
trajectories whose intended answer is not in doubt.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

from aggregate_gradient_trajectory import (  # noqa: E402
    COSINE_KEY,
    classify,
    read_trajectory,
    summarise,
    thirds,
)


def rising(seed_count=5, start=-0.06, end=0.60, epochs=12):
    """Trajectories that sit at `start` early, then climb to `end`.

    The early window is the first third, so a straight line from a small
    negative value to a large positive one is already positive when averaged
    over it. A transition has to be negative *there*, which is what the shape
    below produces and what the criterion is meant to detect.
    """
    head = max(1, epochs // 3)
    climb = epochs - head
    return {
        f"seed{index}": (
            [start + 0.001 * index] * head
            + [start + (end - start) * (step + 1) / climb + 0.001 * index
               for step in range(climb)]
        )
        for index in range(seed_count)
    }


def flat(seed_count=5, value=-0.06):
    return {f"seed{index}": [value + 0.001 * index] * 12 for index in range(seed_count)}


def test_a_clear_climb_across_zero_is_a_transition():
    summary = summarise(rising())
    assert summary["starts_negative"]
    assert summary["ends_positive"]
    assert summary["rose"]
    assert summary["transition"]


def test_a_climb_that_never_reaches_zero_is_a_rise_not_a_transition():
    """The distinction the protocol insists on: rising is not crossing."""
    summary = summarise(rising(start=-0.30, end=-0.05))
    assert summary["rose"]
    assert not summary["ends_positive"]
    assert not summary["transition"]
    assert summary["rise_without_transition"]


def test_a_flat_negative_trajectory_is_neither():
    summary = summarise(flat())
    assert summary["starts_negative"]
    assert not summary["rose"]
    assert not summary["transition"]


def test_a_trajectory_already_positive_throughout_does_not_transition():
    """Ending positive is not enough; it has to have started negative."""
    summary = summarise(flat(value=0.70))
    assert summary["ends_positive"]
    assert not summary["starts_negative"]
    assert not summary["transition"]


def test_windows_never_overlap_however_short_the_run():
    """Otherwise the same epochs would be read as both early and late."""
    for count in range(2, 15):
        head, tail = thirds(count)
        assert head >= 1 and tail >= 1
        assert head + tail <= count, f"windows overlap at {count} epochs"


def test_both_cohorts_transitioning_is_called_stage():
    cohorts = {"sabdab2": summarise(rising()), "asep": summarise(rising())}
    assert classify(cohorts)["verdict"] == "stage"


def test_only_sabdab2_transitioning_is_called_cohort():
    cohorts = {"sabdab2": summarise(rising()), "asep": summarise(flat())}
    assert classify(cohorts)["verdict"] == "cohort"


def test_neither_transitioning_indicts_the_instruments():
    cohorts = {"sabdab2": summarise(flat()), "asep": summarise(flat())}
    verdict = classify(cohorts)
    assert verdict["verdict"] == "neither"
    assert "instruments" in verdict["reason"]


def test_the_unregistered_direction_is_reported_rather_than_forced():
    """AsEP transitioning while sabdab2 does not matches no registered branch."""
    cohorts = {"sabdab2": summarise(flat()), "asep": summarise(rising())}
    assert classify(cohorts)["verdict"] == "stage and cohort"


def test_the_sabdab2_endpoint_is_compared_to_the_checkpoint_measurement():
    cohorts = {"sabdab2": summarise(flat(value=0.70)), "asep": summarise(flat())}
    assert abs(classify(cohorts)["sabdab2_final_vs_checkpoint"]) < 0.01


def test_one_cohort_alone_cannot_separate_stage_from_cohort():
    assert classify({"sabdab2": summarise(rising())})["verdict"] == "not classifiable"


def write_history(path: Path, series, key=COSINE_KEY):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"history": [
        {"epoch": index + 1, "train": {key: value}} for index, value in enumerate(series)
    ]}), encoding="utf-8")


def test_a_run_without_the_cosine_is_skipped_not_read_as_zero(tmp_path):
    """Runs from before the measure arm existed must drop out of the average."""
    history = tmp_path / "history.json"
    write_history(history, [-0.05, 0.1, 0.3], key="something_else")
    assert read_trajectory(history) is None


def test_a_single_epoch_cannot_support_an_early_and_a_late_window(tmp_path):
    history = tmp_path / "history.json"
    write_history(history, [-0.05])
    assert read_trajectory(history) is None


def test_a_usable_run_is_read_in_epoch_order(tmp_path):
    history = tmp_path / "history.json"
    write_history(history, [-0.05, 0.10, 0.30, 0.55])
    assert read_trajectory(history) == [-0.05, 0.10, 0.30, 0.55]
