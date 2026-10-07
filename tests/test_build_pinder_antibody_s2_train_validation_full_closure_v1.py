from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


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
    "build_pinder_antibody_s2_train_validation_v1",
    "audit_pinder_antibody_s2_train_validation_exact_v1",
    "build_pinder_antibody_s2_train_validation_iteration_v2",
    "audit_pinder_antibody_s2_full_four_view_relation_v1",
):
    source = SOURCE_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

SOURCE = SOURCE_DIR / "build_pinder_antibody_s2_train_validation_full_closure_v1.py"
SPEC = importlib.util.spec_from_file_location("pinder_s2_full_closure", SOURCE)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def full_relation_payload():
    views = MODULE.REQUIRED_VIEWS
    return {
        "format_version": MODULE.full_relation.FORMAT_VERSION,
        "status": "PASS_COMPLETE_EXACT_FOUR_VIEW_S2_80_RELATION_MATERIALIZED",
        "authorized_for_training": False,
        "authorized_for_model_selection": False,
        "inputs": {"canonical_manifest": {"sha256": "canonical"}},
        "checks": {
            "all_four_views_complete": True,
            "all_unordered_unique_sequence_pairs_covered": True,
            "canonical_edge_orientation_is_digest_ascending": True,
        },
        "crossing_artifacts": {
            view: {"sha256": view, "edge_count": 1} for view in views
        },
        "results": {
            view: {
                "all_unordered_pairs_covered": True,
                "qualifying_unordered_edge_count": 1,
            }
            for view in views
        },
    }


def test_accepts_complete_relation_bound_to_canonical_manifest():
    contracts = MODULE.validate_full_relation_binding(
        full_relation_payload(), "canonical"
    )
    assert set(contracts) == set(MODULE.REQUIRED_VIEWS)


def test_rejects_relation_with_incomplete_view():
    payload = full_relation_payload()
    payload["results"]["vh"]["all_unordered_pairs_covered"] = False
    with pytest.raises(MODULE.FullClosureError, match="vh full relation is incomplete"):
        MODULE.validate_full_relation_binding(payload, "canonical")
