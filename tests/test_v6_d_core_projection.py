"""Tests for exact V6-D family-core row selection and export."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from interaction_first.v6_cp_sat_formulation import DirectCPModel
from interaction_first.v6_d_core_projection import (
    V6DCoreProjectionError,
    export_exact_family_core_bundle,
    select_exact_family_core,
)
from interaction_first.v6_linear_projection_bundle import (
    load_linear_projection_bundle,
)


HASH = "a" * 64


def _direct_model() -> DirectCPModel:
    cp_model = pytest.importorskip("ortools.sat.python.cp_model")
    model = cp_model.CpModel()
    value = model.new_bool_var("value")
    model.add(value >= 0)
    model.add(value == 0)
    model.add(value == 1)
    families = ("domain", "zero", "one")
    return DirectCPModel(
        model=model,
        variables={("value",): value},
        authority_payload_sha256=HASH,
        constraint_family_counts=(("domain", 1), ("one", 1), ("zero", 1)),
        constraint_families=families,
        formulation_sha256="b" * 64,
    )


def _make_writable(root: Path) -> None:
    if not root.exists():
        return
    os.chmod(root, 0o700)
    for path in root.rglob("*"):
        if path.is_dir():
            os.chmod(path, 0o700)
        elif not path.is_symlink():
            os.chmod(path, 0o600)


def test_exact_family_selection_preserves_source_rows_and_domains() -> None:
    source = _direct_model()
    source_text = str(source.model.Proto())
    selected = select_exact_family_core(source, ["one", "zero"])
    assert str(source.model.Proto()) == source_text
    assert selected.active_families == ("one", "zero")
    assert selected.source_row_indices == (1, 2)
    assert selected.full_dimensions == {"columns": 1, "rows": 3, "nonzeros": 3}
    assert selected.reduced_dimensions == {"columns": 1, "rows": 2, "nonzeros": 2}
    assert selected.row_subset_containment_passed is True
    assert selected.direct_model.constraint_families == ("zero", "one")
    assert tuple(selected.direct_model.model.Proto().variables[0].domain) == (0, 1)


def test_exact_core_bundle_round_trip_binds_source_row_indices(
    tmp_path: Path,
) -> None:
    output = tmp_path / "core"
    try:
        receipt = export_exact_family_core_bundle(
            _direct_model(),
            ["one", "zero"],
            solver_policy_sha256=HASH,
            direct_formulation_source_sha256="b" * 64,
            exact_oracle_source_sha256="c" * 64,
            diagnostic_receipt_sha256="d" * 64,
            output_directory=output,
        )
        assert receipt["row_subset_containment_passed"] is True
        assert receipt["all_source_variables_and_domains_preserved"] is True
        assert receipt["reduced_dimensions"] == {
            "columns": 1,
            "rows": 2,
            "nonzeros": 2,
        }
        assert all(
            receipt[name] is False
            for name in (
                "full_protocol_infeasibility_claim_authorized",
                "split_gate0_authorized",
                "training_authorized",
                "model_selection_authorized",
                "outer_evaluation_authorized",
                "test_access_authorized",
            )
        )
        rows = [
            json.loads(line)
            for line in (output / "source_row_indices.jsonl")
            .read_text("ascii")
            .splitlines()
        ]
        assert rows == [
            {"reduced_row_index": 0, "source_row_index": 1},
            {"reduced_row_index": 1, "source_row_index": 2},
        ]
        loaded = load_linear_projection_bundle(
            output / "bundle",
            expected_bundle_sha256=str(receipt["bundle_sha256"]),
        )
        assert loaded.metadata["dimensions"] == {
            "columns": 1,
            "rows": 2,
            "nonzeros": 2,
        }
        assert os.stat(output).st_mode & 0o222 == 0
    finally:
        _make_writable(output)


@pytest.mark.parametrize(
    ("families", "message"),
    [
        ([], "non-empty"),
        (["one", "one"], "unique"),
        (["absent"], "unknown"),
        ([" one"], "malformed"),
    ],
)
def test_exact_family_selection_fails_closed(
    families: list[str],
    message: str,
) -> None:
    with pytest.raises(V6DCoreProjectionError, match=message):
        select_exact_family_core(_direct_model(), families)
