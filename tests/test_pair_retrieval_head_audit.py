from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "pretraining" / "abpairfm_pair_retrieval_head_audit_v1.json"


def module():
    spec = importlib.util.spec_from_file_location(
        "audit_pair_retrieval_heads", ROOT / "scripts/audit_pair_retrieval_heads.py"
    )
    assert spec is not None and spec.loader is not None
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_expected_head_contracts_match_official_dimensions() -> None:
    value = module()
    assert value.mint_expected_shapes()["project.0.weight"] == (2560, 2560)
    assert value.pplm_expected_shapes()["mlp.0.weight"] == (1024, 3300)
    assert value.pplm_expected_shapes()["linear_embed.weight"] == (660, 1280)


def test_shape_audit_fails_closed_on_dtype_and_nonfinite_values() -> None:
    value = module()
    expected = {"weight": (2, 2), "bias": (2,)}
    good = {"weight": torch.zeros(2, 2), "bias": torch.ones(2)}
    assert value.audit_state_shapes(good, expected)["compatible"] is True
    bad = {"weight": torch.zeros(2, 2, dtype=torch.float64), "bias": torch.tensor([1.0, float("nan")])}
    audit = value.audit_state_shapes(bad, expected)
    assert audit["compatible"] is False
    assert audit["wrong_dtype_keys"] == ["weight"]
    assert audit["nonfinite_keys"] == ["bias"]


def test_frozen_head_audit_summary_keeps_contact_and_test_closed() -> None:
    import json

    result = json.loads(RESULT.read_text(encoding="utf-8"))
    assert result["status"] == "PAIR_RETRIEVAL_HEADS_COMPATIBLE"
    assert result["authorization"]["frozen_development_pair_retrieval"] is True
    assert result["authorization"]["frozen_development_contact"] is False
    assert result["test_access"] is False
    assert result["pplm"]["ensemble_member_count"] == 10
