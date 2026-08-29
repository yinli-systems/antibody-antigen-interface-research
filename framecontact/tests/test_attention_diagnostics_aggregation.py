from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "aggregate_attention_diagnostics.py"
)
SPEC = importlib.util.spec_from_file_location("aggregate_attention_diagnostics", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_head_summary_preserves_seed_by_head_values_and_dispersion() -> None:
    summary = MODULE._summarize_head_vectors([[1.0, 3.0], [2.0, 6.0]])

    assert summary["values_by_seed_by_head"] == [[1.0, 3.0], [2.0, 6.0]]
    assert summary["per_seed_head_mean"] == [2.0, 4.0]
    assert summary["per_seed_head_range"] == [2.0, 4.0]
    assert summary["mean_of_seed_head_means"] == 3.0
    assert summary["pooled_seed_head_mean"] == 3.0
    assert summary["head_count"] == 2
    assert summary["seed_count"] == 2


@pytest.mark.parametrize(
    "vectors",
    (
        [],
        [[]],
        [[1.0, float("nan")]],
        [[1.0], [1.0, 2.0]],
    ),
)
def test_head_summary_rejects_invalid_matrices(vectors: list[list[float]]) -> None:
    with pytest.raises(ValueError):
        MODULE._summarize_head_vectors(vectors)
