"""Tests for the registered V6-A canonical dry-run preflight."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "run_v6_canonical_dry_run_preflight.py"
SPEC = importlib.util.spec_from_file_location("v6_dry_run_preflight", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
SCRIPT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SCRIPT
SPEC.loader.exec_module(SCRIPT)


def _identifier(axis: str, index: int) -> str:
    return hashlib.sha256(f"{axis}:{index:05d}".encode("ascii")).hexdigest()


def _fixture(antibodies: int, antigens: int) -> tuple[dict, dict]:
    authority = {
        "format_version": "score_free_authority_v2",
        "counts": {
            "source_rows": 1,
            "exact_pairs": 1,
            "antibody_entities": antibodies,
            "antigen_entities": antigens,
        },
        "antibodies": [
            {"entity_id": value}
            for value in sorted(_identifier("a", index) for index in range(antibodies))
        ],
        "antigens": [
            {"entity_id": value}
            for value in sorted(_identifier("g", index) for index in range(antigens))
        ],
    }
    registration = {
        "format_version": "v6_a_canonical_dry_run_registration_v1",
        "single_registered_run": True,
        "registered_run_id": "synthetic_once",
        "selection": {
            "axis_scope": "independently_per_antibody_and_antigen_axis",
            "fraction_denominator": 10,
            "fraction_numerator": 1,
            "graph_or_pair_incidence_may_be_inspected_before_selection": False,
            "order": "lowercase_sha256_entity_id_lexicographic",
            "rounding": "ceiling",
        },
        "necessary_condition": {
            "minimum_validation_endpoints_per_axis_per_fold": 60,
            "outer_fold_count": 3,
            "outer_holdout_entities_are_disjoint": True,
        },
        "authority_inventory": authority["counts"],
    }
    return authority, registration


def test_registered_ten_percent_proves_static_infeasibility_before_graph_access() -> None:
    authority, registration = _fixture(1383, 1172)
    receipt = SCRIPT.evaluate_preflight(authority, registration)
    assert receipt["selected_antibody_count"] == 139
    assert receipt["selected_antigen_count"] == 118
    assert receipt["pre_graph_infeasibility_proven"] is True
    assert receipt["graph_or_pair_incidence_accessed"] is False
    assert receipt["cp_sat_invoked"] is False
    assert receipt["highs_invoked"] is False
    assert receipt["v6_a_must_stop"] is True
    assert {failure["axis"] for failure in receipt["static_failures"]} == {
        "antibody",
        "antigen",
    }


def test_static_bound_uses_exact_equality_and_canonical_selection() -> None:
    authority, registration = _fixture(1800, 1800)
    first = SCRIPT.evaluate_preflight(authority, registration)
    second = SCRIPT.evaluate_preflight(authority, registration)
    assert first == second
    assert first["selected_antibody_count"] == 180
    assert first["selected_antigen_count"] == 180
    assert first["static_preflight_passed"] is True
    assert first["pre_graph_infeasibility_proven"] is False
    assert first["dry_run_accepted"] is False
    assert first["training_authorized"] is False
