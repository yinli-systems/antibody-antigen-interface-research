from __future__ import annotations

import sys
from pathlib import Path


SOURCE = (
    Path(__file__).resolve().parents[1]
    / "remote_tools"
    / "abf_paircontact_v1"
    / "pinder_scale_v1"
)
sys.path.insert(0, str(SOURCE))

import diagnose_pinder_antibody_s2_full_closure_infeasibility_v1 as diagnostic


def test_subset_sum_summary_finds_feasible_interval() -> None:
    roots = {"a": "x", "b": "y", "c": "z"}
    pairs = {("a", "g1"), ("a", "g2"), ("b", "g3"), ("c", "g4")}
    result = diagnostic.subset_sum_summary(
        roots, pairs, "antibody", minimum=1, maximum=1, fix_largest_to_train=True
    )
    assert result["feasible_subset_sum_exists"] is True
    assert result["minimum_feasible_subset_sum"] == 1


def test_subset_sum_summary_proves_gap() -> None:
    roots = {"a": "x", "b": "x", "c": "y"}
    pairs = {
        ("a", "g1"),
        ("a", "g2"),
        ("b", "g3"),
        ("c", "g4"),
    }
    result = diagnostic.subset_sum_summary(
        roots, pairs, "antibody", minimum=2, maximum=2, fix_largest_to_train=True
    )
    assert result["feasible_subset_sum_exists"] is False
    assert result["nearest_reachable_below_minimum"] == 1
    assert result["nearest_reachable_above_maximum"] is None
