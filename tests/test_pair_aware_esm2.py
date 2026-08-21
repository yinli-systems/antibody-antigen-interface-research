"""Architecture, loss, and checkpoint-compatibility tests for pair-aware ESM2."""

from __future__ import annotations

from dataclasses import replace

import pytest


torch = pytest.importorskip("torch")

from antibody_pretraining.pair_aware_esm2 import (  # noqa: E402
    ESM2_ALPHABET,
    PairAwareESM2,
    PairAwareESM2Config,
    load_compatible_esm2_state,
    parameter_count,
    upgrade_esm2_state_dict,
)


def _small_config(**changes):
    base = PairAwareESM2Config(
        layer_count=2,
        embed_dimension=32,
        head_count=4,
        ffn_dimension=128,
        maximum_tokens=16,
    )
    return replace(base, **changes)


def test_official_esm2_alphabet_and_3b_meta_parameter_count() -> None:
    assert len(ESM2_ALPHABET) == 33
    assert ESM2_ALPHABET[0] == "<cls>"
    assert ESM2_ALPHABET[1] == "<pad>"
    assert ESM2_ALPHABET[2] == "<eos>"
    assert ESM2_ALPHABET[32] == "<mask>"
    with torch.device("meta"):
        model = PairAwareESM2()
    assert 2_830_000_000 <= parameter_count(model) <= 2_850_000_000


def test_forward_uses_chain_balanced_loss_and_exports_last_attention() -> None:
    torch.manual_seed(19)
    model = PairAwareESM2(_small_config())
    tokens = torch.tensor([[0, 5, 32, 2, 0, 7, 8, 32, 2, 1]])
    chain_ids = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1, 1, -1]])
    labels = torch.full_like(tokens, -100)
    labels[0, 2] = 6
    labels[0, 7] = 9
    output = model(tokens, chain_ids, labels=labels, return_last_attention=True)
    assert output.logits.shape == (1, 10, 33)
    assert output.contributing_chain_count.item() == 2
    assert output.balanced_chain_loss_sum is not None
    assert torch.isfinite(output.balanced_chain_loss_sum)
    assert output.last_attention is not None
    assert output.last_attention.shape == (1, 4, 10, 10)
    output.balanced_chain_loss_sum.backward()
    assert model.layers[0].self_attn.cross_scale.grad is not None


def test_compatible_loader_allows_only_new_cross_scales() -> None:
    source = PairAwareESM2(_small_config())
    target = PairAwareESM2(_small_config())
    state = {
        f"encoder.sentence_encoder.{name}": tensor.detach().clone()
        for name, tensor in source.state_dict().items()
        if not name.endswith("cross_scale")
    }
    receipt = load_compatible_esm2_state(target, state)
    assert receipt["loaded_state_key_count"] == len(state)
    assert receipt["unexpected_keys"] == []
    assert len(receipt["allowed_initialized_keys"]) == 2
    for layer in target.layers:
        torch.testing.assert_close(layer.self_attn.cross_scale, torch.ones(4))
    torch.testing.assert_close(target.embed_tokens.weight, source.embed_tokens.weight)


def test_state_upgrade_rejects_colliding_prefixes() -> None:
    tensor = torch.zeros(1)
    with pytest.raises(ValueError, match="Duplicate"):
        upgrade_esm2_state_dict(
            {"encoder.embed_tokens.weight": tensor, "embed_tokens.weight": tensor}
        )
