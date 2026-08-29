from __future__ import annotations

import pytest
import torch

from scripts.convert_hf_esm2_safetensors_to_abpairfm import (
    convert_state,
    map_hf_esm2_key,
)


def test_maps_hf_esm2_layer_and_global_keys() -> None:
    assert (
        map_hf_esm2_key("esm.encoder.layer.12.attention.self.query.weight")
        == "layers.12.self_attn.q_proj.weight"
    )
    assert (
        map_hf_esm2_key("esm.encoder.layer.2.LayerNorm.bias")
        == "layers.2.final_layer_norm.bias"
    )
    assert (
        map_hf_esm2_key("esm.embeddings.word_embeddings.weight")
        == "embed_tokens.weight"
    )
    assert map_hf_esm2_key(
        "esm.encoder.layer.0.attention.self.rotary_embeddings.inv_freq"
    ) == "layers.0.self_attn.rot_emb.inv_freq"


def test_conversion_recreates_exact_tied_weight() -> None:
    embedding = torch.arange(12, dtype=torch.float32).view(3, 4)
    converted, ignored = convert_state(
        {
            "esm.embeddings.word_embeddings.weight": embedding,
            "lm_head.bias": torch.zeros(3),
            "esm.embeddings.position_ids": torch.arange(5).view(1, 5),
        }
    )
    assert ignored == ["esm.embeddings.position_ids"]
    assert converted["lm_head.weight"] is converted["embed_tokens.weight"]
    torch.testing.assert_close(converted["lm_head.weight"], embedding)


def test_unknown_tensor_fails_closed() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        map_hf_esm2_key("esm.encoder.layer.0.not_a_real_parameter")
