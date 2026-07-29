"""Tests for independent V6-B direct-proto inner-core replay."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest
from ortools.sat.python import cp_model

from interaction_first.v6_b_independent_core_auditor import (
    IndependentCoreAuditError,
    replay_direct_inner_support,
)


def _direct_model() -> object:
    model = cp_model.CpModel()
    variables = {}
    for group in range(3):
        x_key = ("outer", str(group), "training_pair", f"p{group}")
        y_key = ("inner", str(group), "0", "training_pair", f"p{group}")
        x = model.new_bool_var(f"x{group}")
        y = model.new_bool_var(f"y{group}")
        variables[x_key] = x
        variables[y_key] = y
        model.add(y >= x)
        model.add(y <= 0)
    return SimpleNamespace(
        model=model,
        variables=variables,
        formulation_sha256="1" * 64,
        authority_payload_sha256="2" * 64,
    )


def test_independent_replay_proves_the_same_signed_support_infeasible() -> None:
    receipt = replay_direct_inner_support(
        _direct_model(),
        outer_fold=0,
        signed_support=[[["outer", "0", "training_pair", "p0"], 1]],
    )
    assert receipt["status"] == "INFEASIBLE"
    assert receipt["replay_infeasible"] is True
    assert receipt["selected_row_count"] == 2
    assert receipt["v6_b_decomposition_imported"] is False
    assert receipt["historical_milp_or_highs_artifact_consumed"] is False
    assert receipt["training_authorized"] is False


def test_independent_replay_detects_a_support_that_is_not_infeasible() -> None:
    receipt = replay_direct_inner_support(
        _direct_model(),
        outer_fold=0,
        signed_support=[[["outer", "0", "training_pair", "p0"], 0]],
    )
    assert receipt["status"] == "OPTIMAL"
    assert receipt["replay_infeasible"] is False


def test_independent_replay_rejects_foreign_or_duplicate_support() -> None:
    direct = _direct_model()
    with pytest.raises(IndependentCoreAuditError, match="target boundary"):
        replay_direct_inner_support(
            direct,
            outer_fold=0,
            signed_support=[[["outer", "1", "training_pair", "p1"], 1]],
        )
    with pytest.raises(IndependentCoreAuditError, match="duplicated"):
        replay_direct_inner_support(
            direct,
            outer_fold=0,
            signed_support=[
                [["outer", "0", "training_pair", "p0"], 1],
                [["outer", "0", "training_pair", "p0"], 1],
            ],
        )


def _exact_cut() -> dict[str, object]:
    payload: dict[str, object] = {
        "scope": "inner",
        "outer_fold": 0,
        "gate_signature": ["gate", "outer_0_inner_0", "both", "fold"],
        "support": [
            [["inner", "0", "0", "training_pair", "p0"], 0],
            [["outer", "0", "training_pair", "p0"], 0],
        ],
        "primary_statistics_receipt_sha256": "3" * 64,
        "independent_audit_receipt_sha256": "4" * 64,
        "failure_payload_sha256": "5" * 64,
    }
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii") + b"\n"
    payload["cut_sha256"] = hashlib.sha256(raw).hexdigest()
    return payload


def test_independent_replay_reconstructs_exact_no_goods() -> None:
    receipt = replay_direct_inner_support(
        _direct_model(),
        outer_fold=0,
        signed_support=[[["outer", "0", "training_pair", "p0"], 0]],
        exact_no_goods=(_exact_cut(),),
    )
    assert receipt["status"] == "INFEASIBLE"
    assert receipt["replay_infeasible"] is True
    assert receipt["exact_cut_sha256"] == [_exact_cut()["cut_sha256"]]


def test_independent_replay_rejects_a_tampered_exact_no_good() -> None:
    cut = _exact_cut()
    cut["failure_payload_sha256"] = "6" * 64
    with pytest.raises(IndependentCoreAuditError, match="self hash differs"):
        replay_direct_inner_support(
            _direct_model(),
            outer_fold=0,
            signed_support=[[["outer", "0", "training_pair", "p0"], 0]],
            exact_no_goods=(cut,),
        )
