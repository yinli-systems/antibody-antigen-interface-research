"""Tests for the solver-neutral sparse V6 linear projection bundle."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from interaction_first.v6_joint_ir import (
    CanonicalJointIR,
    ExactSeparationGate,
    IRLinearConstraint,
)
from interaction_first.v6_linear_projection_bundle import (
    LinearProjectionBundleError,
    export_linear_projection_bundle,
    load_linear_projection_bundle,
)


HASH = "a" * 64


def _ir() -> CanonicalJointIR:
    variables = (("x", "0"), ("x", "1"))
    return CanonicalJointIR(
        variable_keys=variables,
        linear_constraints=(
            IRLinearConstraint("r0", ((variables[0], 2),), 1, 2),
            IRLinearConstraint("r1", ((variables[0], -1), (variables[1], 3)), None, 1),
        ),
        separation_gates=(
            ExactSeparationGate("g", "outer_0", "antibody", "role", "fraction", HASH),
        ),
        source_bindings=(("source", HASH),),
        source_linear_projection_sha256=HASH,
        construction_scope="test",
        independent_raw_authority_reconstruction=True,
    )


def test_sparse_bundle_round_trip_is_content_addressed_and_read_only(
    tmp_path: Path,
) -> None:
    output = tmp_path / "bundle"
    try:
        metadata = export_linear_projection_bundle(_ir(), output)
        loaded = load_linear_projection_bundle(
            output,
            expected_bundle_sha256=metadata["bundle_sha256"],
        )
        assert loaded.metadata["ir_sha256"] == _ir().ir_sha256
        assert loaded.metadata["dimensions"] == {
            "columns": 2,
            "rows": 2,
            "nonzeros": 3,
        }
        assert loaded.metadata["numeric_envelope"] == {
            "all_integers_exactly_representable_as_float64": True,
            "maximum_abs_bound": 2,
            "maximum_abs_coefficient": 3,
            "maximum_exact_float64_integer": 1 << 53,
        }
        assert loaded.arrays["row_starts"].tolist() == [0, 1, 3]
        assert loaded.arrays["column_indices"].tolist() == [0, 0, 1]
        assert loaded.arrays["coefficients"].tolist() == [2, -1, 3]
        assert loaded.arrays["row_lower_present"].tolist() == [1, 0]
        assert loaded.arrays["row_upper_present"].tolist() == [1, 1]
        assert os.stat(output).st_mode & 0o222 == 0
        assert all(os.stat(path).st_mode & 0o222 == 0 for path in output.iterdir())
    finally:
        if output.exists():
            os.chmod(output, 0o700)
            for path in output.iterdir():
                os.chmod(path, 0o600)


def test_sparse_bundle_rejects_wrong_digest_and_tamper(tmp_path: Path) -> None:
    output = tmp_path / "bundle"
    metadata = export_linear_projection_bundle(_ir(), output)
    try:
        with pytest.raises(LinearProjectionBundleError, match="metadata digest"):
            load_linear_projection_bundle(output, expected_bundle_sha256="b" * 64)
        symbolic = tmp_path / "symbolic"
        symbolic.symlink_to(output, target_is_directory=True)
        with pytest.raises(LinearProjectionBundleError, match="symbolic"):
            load_linear_projection_bundle(
                symbolic,
                expected_bundle_sha256=metadata["bundle_sha256"],
            )
        artifact = output / "coefficients.npy"
        os.chmod(artifact, 0o600)
        data = bytearray(artifact.read_bytes())
        data[-1] ^= 1
        artifact.write_bytes(data)
        with pytest.raises(LinearProjectionBundleError, match="artifact digest"):
            load_linear_projection_bundle(
                output,
                expected_bundle_sha256=metadata["bundle_sha256"],
            )
    finally:
        os.chmod(output, 0o700)
        for path in output.iterdir():
            os.chmod(path, 0o600)


def test_bundle_dictionaries_preserve_canonical_indices(tmp_path: Path) -> None:
    output = tmp_path / "bundle"
    export_linear_projection_bundle(_ir(), output)
    variables = [
        json.loads(line)
        for line in (output / "variable_dictionary.jsonl").read_text("ascii").splitlines()
    ]
    rows = [
        json.loads(line)
        for line in (output / "row_dictionary.jsonl").read_text("ascii").splitlines()
    ]
    assert variables == [
        {"index": 0, "key": ["x", "0"]},
        {"index": 1, "key": ["x", "1"]},
    ]
    assert rows == [{"index": 0, "name": "r0"}, {"index": 1, "name": "r1"}]
    os.chmod(output, 0o700)
    for path in output.iterdir():
        os.chmod(path, 0o600)


def test_bundle_rejects_integer_outside_exact_float64_envelope(
    tmp_path: Path,
) -> None:
    key = ("x",)
    ir = CanonicalJointIR(
        variable_keys=(key,),
        linear_constraints=(
            IRLinearConstraint("r", ((key, (1 << 53) + 1),), 0, 1),
        ),
        separation_gates=(),
        source_bindings=(("source", HASH),),
        source_linear_projection_sha256=HASH,
        construction_scope="test",
        independent_raw_authority_reconstruction=True,
    )
    output = tmp_path / "bundle"
    try:
        with pytest.raises(LinearProjectionBundleError, match="exact float64"):
            export_linear_projection_bundle(ir, output)
    finally:
        if output.exists():
            os.chmod(output, 0o700)
            for path in output.iterdir():
                os.chmod(path, 0o600)
