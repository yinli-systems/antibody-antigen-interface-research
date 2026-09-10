"""The one reading of the epitope_ratio test split.

This split is read once. Everything the reading depends on is fixed beforehand,
so the only way it can go wrong is arithmetic or a threshold leaking from the
split it is applied to. Both are pinned here, before the reading happens.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from ratio_test_confirmation import (  # noqa: E402
    ROLE,
    ensemble_frame,
    load_node_frame,
    metrics_at,
)


def frame(rows):
    pandas = pytest.importorskip("pandas")
    return pandas.DataFrame(
        [{"instance_id": name, "role": ROLE, "res_idx": index,
          "label": label, "score": score}
         for name, entries in rows.items()
         for index, (label, score) in enumerate(entries)]
    )


def test_a_perfect_separation_at_the_threshold_scores_one():
    rows = {name: [(True, 0.9), (True, 0.8), (False, 0.2), (False, 0.1)]
            for name in ("a", "b")}
    result = metrics_at(frame(rows), 0.5)
    assert result["mcc"] == pytest.approx(1.0)
    assert result["precision"] == pytest.approx(1.0)
    assert result["recall"] == pytest.approx(1.0)
    assert result["f1"] == pytest.approx(1.0)


def test_the_threshold_actually_moves_the_answer():
    """A threshold above every score predicts nothing and cannot correlate."""
    rows = {"a": [(True, 0.6), (False, 0.4)]}
    assert metrics_at(frame(rows), 0.5)["mcc"] == pytest.approx(1.0)
    assert metrics_at(frame(rows), 0.99)["mcc"] == pytest.approx(0.0)


def test_complexes_are_averaged_not_pooled():
    """The registered figure is per-complex; a large complex must not dominate."""
    perfect = [(index < 30, 0.9 if index < 30 else 0.1) for index in range(60)]
    inverted = [(index < 2, 0.1 if index < 2 else 0.9) for index in range(4)]
    result = metrics_at(frame({"big": perfect, "small": inverted}), 0.5)
    assert result["mcc"] == pytest.approx(0.0, abs=0.02), (
        "one perfect and one inverted complex average to about zero"
    )
    assert result["complex_count"] == 2.0


def test_threshold_free_metrics_ignore_the_threshold():
    # Asymmetric on purpose: a threshold either side of a symmetric score set
    # gives the same MCC by coincidence, which would let the last assertion pass
    # while proving nothing.
    rows = {"a": [(True, 0.9), (True, 0.7), (True, 0.6), (False, 0.3), (False, 0.1)]}
    low = metrics_at(frame(rows), 0.2)
    high = metrics_at(frame(rows), 0.8)
    assert low["auprc"] == pytest.approx(high["auprc"])
    assert low["auroc"] == pytest.approx(high["auroc"])
    assert low["mcc"] != pytest.approx(high["mcc"])


def test_a_complex_where_nothing_is_predicted_scores_zero_not_nan():
    """Per-complex averaging means one NaN would take the whole figure with it."""
    result = metrics_at(frame({"a": [(True, 0.1), (False, 0.05)]}), 0.5)
    assert result["mcc"] == pytest.approx(0.0)
    assert result["precision"] == pytest.approx(0.0)
    assert result["f1"] == pytest.approx(0.0)


def test_the_ensemble_averages_scores_per_residue():
    first = frame({"a": [(True, 0.9), (False, 0.1)]})
    second = frame({"a": [(True, 0.5), (False, 0.3)]})
    merged = ensemble_frame([first, second])
    assert sorted(merged["score"].tolist()) == pytest.approx([0.2, 0.7])
    assert merged["label"].tolist() == [True, False]


def test_the_ensemble_refuses_seeds_that_do_not_share_residues():
    """Silently inner-joining mismatched runs would average different residues."""
    first = frame({"a": [(True, 0.9), (False, 0.1)]})
    second = frame({"b": [(True, 0.5), (False, 0.3)]})
    with pytest.raises(ValueError, match="refusing to average"):
        ensemble_frame([first, second])


def test_the_ensemble_refuses_a_seed_missing_only_some_residues():
    """The dangerous case is partial overlap, which an inner join hides.

    Disjoint runs produce an empty frame that almost anything would catch. A run
    short by one complex still merges, still yields one score column per seed,
    and silently averages over the intersection.
    """
    first = frame({"a": [(True, 0.9), (False, 0.1)], "b": [(True, 0.8), (False, 0.2)]})
    second = frame({"a": [(True, 0.5), (False, 0.3)]})
    with pytest.raises(ValueError, match="refusing to average"):
        ensemble_frame([first, second])


def test_only_epitope_rows_are_read(tmp_path):
    pandas = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    directory = tmp_path / "test"
    directory.mkdir()
    pandas.DataFrame([
        {"instance_id": "a", "role": "epitope", "res_idx": 0, "label": True, "score": 0.9},
        {"instance_id": "a", "role": "paratope", "res_idx": 0, "label": True, "score": 0.1},
    ]).to_parquet(directory / "node_predictions.parquet")
    loaded = load_node_frame(directory)
    assert len(loaded) == 1
    assert set(loaded["role"]) == {ROLE}


def test_a_missing_split_is_reported_rather_than_guessed(tmp_path):
    assert load_node_frame(tmp_path / "absent") is None


def test_the_ensemble_names_which_seed_disagrees_and_by_how_much():
    """A row-count mismatch does not say which seed is wrong, or about what."""
    first = frame({"a": [(True, 0.9), (False, 0.1)], "b": [(True, 0.8), (False, 0.2)]})
    second = frame({"a": [(True, 0.5), (False, 0.3)], "c": [(True, 0.4), (False, 0.6)]})
    with pytest.raises(ValueError, match="position 1 is missing 2 .* adds 2"):
        ensemble_frame([first, second])


def test_seeds_agreeing_on_residues_but_not_on_scores_still_average():
    """Only the residue index has to match; disagreeing is what averaging is for."""
    first = frame({"a": [(True, 1.0), (False, 0.0)]})
    second = frame({"a": [(True, 0.0), (False, 1.0)]})
    merged = ensemble_frame([first, second])
    assert merged["score"].tolist() == pytest.approx([0.5, 0.5])
