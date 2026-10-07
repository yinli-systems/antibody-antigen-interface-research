from __future__ import annotations

import importlib.util
from pathlib import Path


SOURCE = (
    Path(__file__).parents[1]
    / "remote_tools"
    / "abf_paircontact_v1"
    / "pinder_scale_v1"
    / "diagnose_pinder_antibody_s40_components_v1.py"
)
SPEC = importlib.util.spec_from_file_location("diagnose_pinder_s40", SOURCE)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_disjoint_set_merges_transitively() -> None:
    disjoint = MODULE.DisjointSet(("a", "b", "c"))
    disjoint.union("a", "b")
    disjoint.union("b", "c")
    assert len({disjoint.find(value) for value in ("a", "b", "c")}) == 1


def test_subset_sum_reports_infeasible_when_largest_is_fixed_train() -> None:
    result = MODULE.subset_sum_feasibility(
        [9, 1], 3, 4, fix_largest_to_train=True
    )
    assert result["feasible_subset_sum_exists"] is False
    assert result["nearest_reachable_below_minimum"] == 1


def test_subset_sum_reports_feasible_interval() -> None:
    result = MODULE.subset_sum_feasibility(
        [5, 3, 2], 3, 4, fix_largest_to_train=True
    )
    assert result["feasible_subset_sum_exists"] is True
    assert result["minimum_feasible_subset_sum"] == 3
