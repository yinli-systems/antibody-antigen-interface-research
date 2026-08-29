"""Focused checks for preregistered STRING supplemental materialization."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "materialize_string_v12_supplemental_selection.py"


def _module():
    specification = importlib.util.spec_from_file_location("materialize_string", SCRIPT)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _row(protein1: str, protein2: str, *, experiments: int, database: int, combined: int) -> list[str]:
    return [
        protein1,
        protein2,
        "0",
        str(experiments),
        "0",
        str(database),
        "0",
        "0",
        "0",
        str(combined),
    ]


def test_selection_applies_frozen_threshold_and_keeps_strongest_edge() -> None:
    module = _module()
    candidate = {
        "combined_minimum_inclusive": 600,
        "combined_maximum_exclusive": 700,
        "direct_channel_maximum_minimum_inclusive": 300,
    }
    rows = [
        _row("a", "b", experiments=300, database=0, combined=610),
        _row("a2", "b", experiments=500, database=0, combined=690),
        _row("c", "d", experiments=299, database=0, combined=699),
    ]
    selected, counts = module.select_supplemental_rows(
        rows,
        {"a": "A", "a2": "A", "b": "B", "c": "C", "d": "D"},
        set(),
        set(),
        candidate,
    )
    assert set(selected) == {("A", "B")}
    assert selected[("A", "B")][1][0] == "a2"
    assert counts["eligible_candidate_edges"] == 2
    assert counts["duplicate_candidate_cluster_pair_edges"] == 1


def test_selection_excludes_development_tier1_self_and_missing_pairs() -> None:
    module = _module()
    candidate = {
        "combined_minimum_inclusive": 600,
        "combined_maximum_exclusive": 700,
        "direct_channel_maximum_minimum_inclusive": 1,
    }
    rows = [
        _row("a", "b", experiments=1, database=0, combined=600),
        _row("c", "d", experiments=1, database=0, combined=600),
        _row("e", "f", experiments=1, database=0, combined=600),
        _row("missing", "f", experiments=1, database=0, combined=600),
    ]
    selected, counts = module.select_supplemental_rows(
        rows,
        {"a": "A", "b": "B", "c": "C", "d": "C", "e": "E", "f": "F"},
        {("E", "F")},
        {"A"},
        candidate,
    )
    assert selected == {}
    assert counts["development_excluded_edges"] == 1
    assert counts["self_cluster_edges"] == 1
    assert counts["tier1_overlap_edges"] == 1
    assert counts["missing_endpoint_edges"] == 1
