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
):
    source = SOURCE_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

SOURCE = SOURCE_DIR / "build_pinder_antibody_s2_train_validation_iteration_v2.py"
SPEC = importlib.util.spec_from_file_location("pinder_s2_tv_iteration", SOURCE)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def previous_assignment():
    return {
        "entity_assignments": {
            "antibody": [
                {
                    "sequence_sha256": "a1",
                    "component_root_sha256": "aroot",
                    "bucket": "train",
                },
                {
                    "sequence_sha256": "a2",
                    "component_root_sha256": "aroot",
                    "bucket": "train",
                },
            ],
            "antigen": [
                {
                    "sequence_sha256": "g1",
                    "component_root_sha256": "g1",
                    "bucket": "train",
                }
            ],
        }
    }


def test_loads_previous_component_roots_without_using_bucket_as_component():
    antibody, antigen = MODULE.load_previous_component_roots(
        previous_assignment(), {"a1", "a2"}, {"g1"}
    )
    assert antibody == {"a1": "aroot", "a2": "aroot"}
    assert antigen == {"g1": "g1"}


def test_rejects_previous_component_inventory_mismatch():
    with pytest.raises(MODULE.IterationError, match="inventory mismatch"):
        MODULE.load_previous_component_roots(
            previous_assignment(), {"a1", "a2", "a3"}, {"g1"}
        )


def test_failed_audit_binding_requires_exact_previous_assignment():
    payload = {
        "status": "FAIL_EXACT_FOUR_VIEW_TRAIN_VALIDATION_S2_80_CROSSINGS_DETECTED",
        "approval_status": "NOT_APPROVED_REBUILD_WITH_NEW_EXACT_EDGES",
        "authorized_for_training": False,
        "authorized_for_model_selection": False,
        "assignment_scientific_payload_sha256": "sci",
        "inputs": {
            "assignment": {"sha256": "assignment"},
            "canonical_manifest": {"sha256": "canonical"},
        },
        "crossing_artifacts": {
            view: {"sha256": view, "edge_count": int(view == "vh")}
            for view in MODULE.REQUIRED_VIEWS
        },
        "exhaustive_accounting": {"total_threshold_crossing_edge_count": 1},
    }
    contracts = MODULE.validate_failed_audit_binding(
        payload, "assignment", "sci", "canonical"
    )
    assert contracts["vh"]["edge_count"] == 1
    with pytest.raises(MODULE.IterationError, match="assignment SHA mismatch"):
        MODULE.validate_failed_audit_binding(payload, "other", "sci", "canonical")
