"""Focused compatibility checks without materializing the 3B checkpoint."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")

from antibody_pretraining.pair_aware_esm2 import (  # noqa: E402
    PairAwareESM2,
    PairAwareESM2Config,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/audit_abpairfm_esm2_checkpoint.py"


def _module():
    spec = importlib.util.spec_from_file_location("esm2_checkpoint_audit", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _config() -> PairAwareESM2Config:
    return PairAwareESM2Config(
        layer_count=2,
        embed_dimension=32,
        head_count=4,
        ffn_dimension=128,
        maximum_tokens=16,
    )


def _official_like_state(model: PairAwareESM2):
    return {
        f"encoder.sentence_encoder.{name}": tensor.detach().clone()
        for name, tensor in model.state_dict().items()
        if not name.endswith("cross_scale")
    }


def test_state_receipt_proves_only_cross_scales_are_new() -> None:
    model = PairAwareESM2(_config())
    receipt = _module().audit_state_compatibility(
        model,
        _official_like_state(model),
        checkpoint_dimensions={
            "encoder_attention_heads": 4,
            "encoder_embed_dim": 32,
            "encoder_layers": 2,
            "token_dropout": True,
        },
    )
    assert receipt["allowed_initialized_key_count"] == 2
    assert receipt["initialized_pair_aware_parameter_count"] == 8
    assert receipt["shape_mismatch_count"] == 0


def test_state_receipt_rejects_shape_drift() -> None:
    model = PairAwareESM2(_config())
    state = _official_like_state(model)
    state["encoder.sentence_encoder.layers.0.fc1.bias"] = torch.zeros(129)
    with pytest.raises(RuntimeError, match="shape mismatches"):
        _module().audit_state_compatibility(
            model,
            state,
            checkpoint_dimensions={
                "encoder_attention_heads": 4,
                "encoder_embed_dim": 32,
                "encoder_layers": 2,
                "token_dropout": True,
            },
        )
