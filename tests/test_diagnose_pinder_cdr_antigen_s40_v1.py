from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


PATH = (
    Path(__file__).parents[1]
    / "remote_tools/abf_paircontact_v1/pinder_scale_v1/diagnose_pinder_cdr_antigen_s40_v1.py"
)
SPEC = spec_from_file_location("diagnose_pinder_cdr_antigen_s40_v1", PATH)
MODULE = module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_subset_witness_hits_nearest_target_deterministically():
    components = [("a", 10), ("b", 7), ("c", 5), ("d", 3)]
    selected = MODULE.choose_validation_components(components, 9, 12, 10)
    repeated = MODULE.choose_validation_components(components, 9, 12, 10)
    weights = dict(components)
    assert selected == repeated
    assert sum(weights[value] for value in selected) == 10


def test_subset_witness_reports_infeasible_interval():
    assert MODULE.choose_validation_components([("a", 8), ("b", 8)], 5, 7, 6) is None


def test_disjoint_set_is_stable_under_repeated_union():
    dsu = MODULE.DisjointSet(["a", "b", "c"])
    dsu.union("a", "b")
    dsu.union("b", "a")
    assert dsu.find("a") == dsu.find("b")
    assert dsu.find("a") != dsu.find("c")
