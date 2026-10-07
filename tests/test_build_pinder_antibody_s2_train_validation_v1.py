from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


SOURCE_DIR = (
    Path(__file__).parents[1]
    / "remote_tools"
    / "abf_paircontact_v1"
    / "pinder_scale_v1"
)
for name in (
    "audit_pinder_antibody_s2_exact_four_view_v1",
    "build_pinder_antibody_s2_folds_v1",
    "prove_pinder_antibody_s2_three_fold_infeasibility_v1",
):
    source = SOURCE_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

SOURCE = SOURCE_DIR / "build_pinder_antibody_s2_train_validation_v1.py"
SPEC = importlib.util.spec_from_file_location("pinder_s2_train_validation", SOURCE)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
BASE = sys.modules["build_pinder_antibody_s2_folds_v1"]


def test_binary_solver_keeps_largest_antibody_component_in_train():
    rows = [
        BASE.SourceRow(f"s{i}", f"p{i}", "fab", f"a{i}", f"g{i}")
        for i in range(8)
    ]
    antibody_roots = {
        **{f"a{i}": "large" for i in range(5)},
        "a5": "small5",
        "a6": "small6",
        "a7": "small7",
    }
    antigen_roots = {f"g{i}": f"ag{i}" for i in range(8)}
    components, _, _ = BASE.interaction_components(
        rows, antibody_roots, antigen_roots
    )
    solution, result, antibody_clusters, antigen_clusters = MODULE.solve_binary(
        rows,
        antibody_roots,
        antigen_roots,
        components,
        0.125,
        0.25,
        30.0,
    )
    index = MODULE.VariableIndex(
        len(antibody_clusters), len(antigen_clusters), len(components)
    )
    large_index = antibody_clusters.index("large")
    assert solution[index.antibody_validation(large_index)] < 0.5
    assert result["mip_gap"] == 0.0
    assert result["optimal_retained_unique_exact_pair_count"] == 8


def test_component_train_and_validation_variables_are_disjoint():
    index = MODULE.VariableIndex(3, 4, 5)
    assert index.count == 17
    assert {
        index.component_train(i) for i in range(5)
    }.isdisjoint({index.component_validation(i) for i in range(5)})
    assert np.all(np.asarray([index.antibody_validation(i) for i in range(3)]) < 3)
