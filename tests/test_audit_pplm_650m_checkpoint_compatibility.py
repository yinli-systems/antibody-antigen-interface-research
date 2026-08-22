import importlib.util
from pathlib import Path

import torch

from antibody_pretraining.pair_aware_esm2 import PairAwareESM2, PairAwareESM2Config


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "audit_pplm_650m_checkpoint_compatibility_v2.py"
)
SPEC = importlib.util.spec_from_file_location("pplm_compatibility", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _pplm_shaped_state(config: PairAwareESM2Config) -> dict[str, torch.Tensor]:
    model = PairAwareESM2(config)
    state = dict(model.state_dict())
    for layer in range(config.layer_count):
        state[
            f"layers.{layer}.self_attn.inter_attn_weight"
        ] = state.pop(f"layers.{layer}.self_attn.cross_scale")
    state["lm_head.weight"] = state["embed_tokens.weight"].clone()
    return state


def test_normalize_pplm_state_is_exactly_compatible() -> None:
    config = PairAwareESM2Config(
        layer_count=2,
        embed_dimension=32,
        head_count=4,
        ffn_dimension=64,
        maximum_tokens=64,
    )
    normalized, audit = MODULE.normalize_pplm_state(
        _pplm_shaped_state(config), config
    )
    assert audit["encoder_mlm_compatible"] is True
    assert audit["end_to_end_contact_compatible"] is False
    assert audit["pplm_inter_attention"]["renamed_layer_count"] == 2
    assert "lm_head.weight" not in normalized


def test_normalize_pplm_state_fails_closed_on_shape_mismatch() -> None:
    config = PairAwareESM2Config(
        layer_count=1,
        embed_dimension=32,
        head_count=4,
        ffn_dimension=64,
        maximum_tokens=64,
    )
    state = _pplm_shaped_state(config)
    state["layers.0.self_attn.inter_attn_weight"] = torch.ones(3)
    _, audit = MODULE.normalize_pplm_state(state, config)
    assert audit["encoder_mlm_compatible"] is False
    assert "layers.0.self_attn.cross_scale" in audit["shape_mismatches"]
