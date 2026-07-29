"""Tests for isolated direct-HiGHS solving from sparse V6 bundles."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from interaction_first.v6_highs_bundle_backend import solve_highs_projection_bundle
from interaction_first.v6_highs_bundle_backend import HighsBundleAuditError
from interaction_first.v6_joint_ir import CanonicalJointIR, IRLinearConstraint
from interaction_first.v6_linear_projection_bundle import export_linear_projection_bundle


HASH = "a" * 64


def _infeasible_ir() -> CanonicalJointIR:
    key = ("x",)
    return CanonicalJointIR(
        variable_keys=(key,),
        linear_constraints=(
            IRLinearConstraint("force_one", ((key, 1),), 1, 1),
            IRLinearConstraint("force_zero", ((key, 1),), 0, 0),
        ),
        separation_gates=(),
        source_bindings=(("source", HASH),),
        source_linear_projection_sha256=HASH,
        construction_scope="test",
        independent_raw_authority_reconstruction=True,
    )


def _feasible_ir() -> CanonicalJointIR:
    key = ("x",)
    return CanonicalJointIR(
        variable_keys=(key,),
        linear_constraints=(IRLinearConstraint("force_one", ((key, 1),), 1, 1),),
        separation_gates=(),
        source_bindings=(("source", HASH),),
        source_linear_projection_sha256=HASH,
        construction_scope="test",
        independent_raw_authority_reconstruction=True,
    )


def _isolate_from_loaded_ortools(monkeypatch) -> None:
    """Model the dedicated highspy interpreter used by the registered audit."""
    for name in tuple(sys.modules):
        if name == "ortools" or name.startswith("ortools."):
            monkeypatch.delitem(sys.modules, name)


def test_highs_bundle_backend_certifies_only_linear_infeasibility(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _isolate_from_loaded_ortools(monkeypatch)
    bundle_path = tmp_path / "bundle"
    output = tmp_path / "highs"
    try:
        bundle = export_linear_projection_bundle(_infeasible_ir(), bundle_path)
        receipt = solve_highs_projection_bundle(
            bundle_directory=bundle_path,
            expected_bundle_sha256=bundle["bundle_sha256"],
            output_directory=output,
            maximum_time_seconds=30,
        )
        assert receipt["linear_projection_infeasible"] is True
        assert receipt["linear_projection_certified"] is True
        assert receipt["exact_integer_row_audit_passed"] is False
        assert receipt["complete_score_free_feasible_region_certified"] is False
        assert receipt["cp_sat_or_ortools_imported"] is False
        assert receipt["user_solution_callback_used"] is False
        assert receipt["mip_logging_callback_used"] is True
        assert receipt["training_authorized"] is False
        assert (output / "original_model.mps").is_file()
        assert (output / "presolved_model.mps").is_file()
    finally:
        for root in (bundle_path, output):
            if root.exists():
                os.chmod(root, 0o700)
                for path in root.iterdir():
                    os.chmod(path, 0o600)


def test_highs_bundle_backend_exactly_audits_an_optimal_solution(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _isolate_from_loaded_ortools(monkeypatch)
    bundle_path = tmp_path / "bundle"
    output = tmp_path / "highs"
    try:
        bundle = export_linear_projection_bundle(_feasible_ir(), bundle_path)
        receipt = solve_highs_projection_bundle(
            bundle_directory=bundle_path,
            expected_bundle_sha256=bundle["bundle_sha256"],
            output_directory=output,
            maximum_time_seconds=30,
        )
        assert receipt["linear_projection_optimal"] is True
        assert receipt["linear_projection_infeasible"] is False
        assert receipt["exact_integer_row_audit_passed"] is True
        assert receipt["mip_gap_is_finite_zero"] is True
        assert receipt["linear_projection_certified"] is True
        assert receipt["complete_score_free_feasible_region_certified"] is False
    finally:
        for root in (bundle_path, output):
            if root.exists():
                os.chmod(root, 0o700)
                for path in root.iterdir():
                    os.chmod(path, 0o600)


def test_highs_bundle_backend_rejects_loaded_ortools(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setitem(sys.modules, "ortools.fake", object())
    with pytest.raises(HighsBundleAuditError, match="OR-Tools is loaded"):
        solve_highs_projection_bundle(
            bundle_directory=tmp_path / "absent",
            expected_bundle_sha256=HASH,
            output_directory=tmp_path / "output",
            maximum_time_seconds=30,
        )
