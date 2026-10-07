"""Architecture, loss, and checkpoint-compatibility tests for pair-aware ESM2."""

from __future__ import annotations

from dataclasses import replace

import pytest


torch = pytest.importorskip("torch")

from antibody_pretraining.multimodal_v1.pair_aware_esm2 import (  # noqa: E402
    ESM2_ALPHABET,
    PairAwareESM2,
    PairAwareESM2Config,
    canonicalize_tied_esm2_state,
    load_compatible_esm2_state,
    parameter_count,
    reset_selective_attention_checkpoint_policy_counts,
    selective_attention_checkpoint_policy_counts,
    selective_attention_checkpoint_policy,
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
    assert "lm_head.weight" not in model.state_dict()


def test_forward_uses_chain_balanced_loss_and_exports_last_attention() -> None:
    torch.manual_seed(19)
    model = PairAwareESM2(_small_config())
    tokens = torch.tensor([[0, 5, 32, 2, 0, 7, 8, 32, 2, 1]])
    chain_ids = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1, 1, -1]])
    labels = torch.full_like(tokens, -100)
    labels[0, 2] = 6
    labels[0, 7] = 9
    output = model(tokens, chain_ids, labels=labels, return_last_attention=True)
    assert output.logits is not None
    assert output.logits.shape == (1, 10, 33)
    assert output.contributing_chain_count.item() == 2
    assert output.balanced_chain_loss_sum is not None
    assert torch.isfinite(output.balanced_chain_loss_sum)
    assert output.last_attention is not None
    assert output.last_attention.shape == (1, 4, 10, 10)
    output.balanced_chain_loss_sum.backward()
    assert model.layers[0].self_attn.cross_scale.grad is not None


def test_chain_independent_model_view_is_invariant_to_partner_tokens() -> None:
    torch.manual_seed(21)
    model = PairAwareESM2(_small_config()).eval()
    first = torch.tensor([[0, 5, 6, 2, 0, 7, 8, 2, 1]])
    second = torch.tensor([[0, 5, 6, 2, 0, 11, 12, 2, 1]])
    chain_ids = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1, -1]])

    first_output = model(
        first, chain_ids, allow_cross_chain_attention=False
    )
    second_output = model(
        second, chain_ids, allow_cross_chain_attention=False
    )
    assert first_output.logits is not None and second_output.logits is not None
    first_independent = first_output.logits
    second_independent = second_output.logits
    torch.testing.assert_close(
        first_independent[:, :4], second_independent[:, :4], rtol=0.0, atol=0.0
    )

    first_joint_output = model(first, chain_ids)
    second_joint_output = model(second, chain_ids)
    assert first_joint_output.logits is not None
    assert second_joint_output.logits is not None
    first_joint = first_joint_output.logits
    second_joint = second_joint_output.logits
    assert not torch.allclose(first_joint[:, :4], second_joint[:, :4])


def test_hidden_only_forward_skips_lm_head() -> None:
    model = PairAwareESM2(_small_config()).eval()
    tokens = torch.tensor([[0, 5, 6, 2, 0, 7, 8, 2, 1]])
    chain_ids = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1, -1]])

    def forbidden_head(*_args, **_kwargs):
        raise AssertionError("hidden-only encoding must not run the MLM head")

    model.lm_head.forward = forbidden_head  # type: ignore[method-assign]
    output = model(
        tokens,
        chain_ids,
        allow_cross_chain_attention=False,
        return_hidden_states=True,
        compute_logits=False,
    )
    assert output.logits is None
    assert output.hidden_states is not None
    assert output.hidden_states.shape == (1, 9, 32)

    with pytest.raises(ValueError, match="MLM labels require"):
        model(tokens, chain_ids, labels=tokens, compute_logits=False)


def test_optional_embedding_delta_preserves_default_and_validates_padding() -> None:
    torch.manual_seed(31)
    model = PairAwareESM2(_small_config()).eval()
    tokens = torch.tensor([[0, 5, 6, 2, 0, 7, 8, 2, 1]])
    chain_ids = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1, -1]])
    zero_delta = torch.zeros(1, 9, 32)

    baseline = model(tokens, chain_ids).logits
    unchanged = model(
        tokens,
        chain_ids,
        input_embedding_delta=zero_delta,
    ).logits
    assert baseline is not None and unchanged is not None
    torch.testing.assert_close(unchanged, baseline, rtol=0.0, atol=0.0)

    nonzero = zero_delta.clone()
    nonzero[:, 1, 0] = 0.25
    changed = model(tokens, chain_ids, input_embedding_delta=nonzero).logits
    assert changed is not None
    assert not torch.equal(changed, baseline)

    bad_padding = zero_delta.clone()
    bad_padding[:, -1, 0] = 1.0
    with pytest.raises(ValueError, match="Padding positions"):
        model(tokens, chain_ids, input_embedding_delta=bad_padding)

    nonfinite = zero_delta.clone()
    nonfinite[:, 1, 0] = float("nan")
    with pytest.raises(ValueError, match="must be finite"):
        model(tokens, chain_ids, input_embedding_delta=nonfinite)


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


def test_checkpoint_tied_duplicate_is_validated_then_removed() -> None:
    tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    canonical = canonicalize_tied_esm2_state(
        {
            "embed_tokens.weight": tensor,
            "lm_head.weight": tensor.clone(),
        },
        require_lm_head_duplicate=True,
    )
    assert set(canonical) == {"embed_tokens.weight"}
    with pytest.raises(ValueError, match="values differ"):
        canonicalize_tied_esm2_state(
            {
                "embed_tokens.weight": tensor,
                "lm_head.weight": tensor + 1,
            }
        )


def test_selective_checkpoint_policy_only_caches_flash_attention() -> None:
    from torch.utils.checkpoint import CheckpointPolicy

    reset_selective_attention_checkpoint_policy_counts()
    flash = torch.ops.aten._scaled_dot_product_flash_attention.default
    assert (
        selective_attention_checkpoint_policy(None, flash)
        is CheckpointPolicy.MUST_SAVE
    )
    assert (
        selective_attention_checkpoint_policy(None, torch.ops.aten.add.Tensor)
        is CheckpointPolicy.PREFER_RECOMPUTE
    )
    assert selective_attention_checkpoint_policy_counts() == {
        "flash_must_save": 1,
        "prefer_recompute": 1,
    }


def test_selective_checkpoint_matches_full_checkpoint_on_cpu() -> None:
    torch.manual_seed(23)
    full = PairAwareESM2(
        _small_config(activation_checkpointing=True, activation_checkpoint_policy="full")
    ).train()
    selective = PairAwareESM2(
        _small_config(
            activation_checkpointing=True,
            activation_checkpoint_policy="selective_attention",
        )
    ).train()
    selective.load_state_dict(full.state_dict())
    tokens = torch.tensor([[0, 5, 32, 2, 0, 7, 8, 32, 2, 1]])
    chain_ids = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1, 1, -1]])
    labels = torch.full_like(tokens, -100)
    labels[0, 2] = 6
    labels[0, 7] = 9

    full_loss = full(tokens, chain_ids, labels=labels).balanced_chain_loss_sum
    selective_loss = selective(tokens, chain_ids, labels=labels).balanced_chain_loss_sum
    assert full_loss is not None and selective_loss is not None
    full_loss.backward()
    selective_loss.backward()
    torch.testing.assert_close(selective_loss, full_loss)
    for full_parameter, selective_parameter in zip(
        full.parameters(), selective.parameters(), strict=True
    ):
        torch.testing.assert_close(selective_parameter.grad, full_parameter.grad)


def test_checkpointed_independent_view_keeps_cross_scale_in_graph() -> None:
    torch.manual_seed(27)
    model = PairAwareESM2(
        _small_config(activation_checkpointing=True)
    ).train()
    tokens = torch.tensor([[0, 5, 32, 2, 0, 7, 8, 32, 2, 1]])
    chain_ids = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1, 1, -1]])
    labels = torch.full_like(tokens, -100)
    labels[0, 2] = 6
    labels[0, 7] = 9

    loss = model(
        tokens,
        chain_ids,
        labels=labels,
        allow_cross_chain_attention=False,
    ).balanced_chain_loss_sum
    assert loss is not None
    loss.backward()
    for layer in model.layers:
        gradient = layer.self_attn.cross_scale.grad
        assert gradient is not None
        assert torch.count_nonzero(gradient).item() == 0


def test_selective_policy_requires_checkpointing() -> None:
    with pytest.raises(ValueError, match="requires activation_checkpointing"):
        PairAwareESM2(_small_config(activation_checkpoint_policy="selective_attention"))


def test_checkpoint_without_rng_preservation_matches_default_on_cpu() -> None:
    torch.manual_seed(29)
    baseline = PairAwareESM2(
        _small_config(
            activation_checkpointing=True,
            activation_checkpoint_preserve_rng_state=True,
        )
    ).train()
    candidate = PairAwareESM2(
        _small_config(
            activation_checkpointing=True,
            activation_checkpoint_preserve_rng_state=False,
        )
    ).train()
    candidate.load_state_dict(baseline.state_dict())
    tokens = torch.tensor([[0, 5, 32, 2, 0, 7, 8, 32, 2, 1]])
    chain_ids = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1, 1, -1]])
    labels = torch.full_like(tokens, -100)
    labels[0, 2] = 6
    labels[0, 7] = 9

    baseline_loss = baseline(tokens, chain_ids, labels=labels).balanced_chain_loss_sum
    candidate_loss = candidate(tokens, chain_ids, labels=labels).balanced_chain_loss_sum
    assert baseline_loss is not None and candidate_loss is not None
    baseline_loss.backward()
    candidate_loss.backward()
    torch.testing.assert_close(candidate_loss, baseline_loss)
    for baseline_parameter, candidate_parameter in zip(
        baseline.parameters(), candidate.parameters(), strict=True
    ):
        torch.testing.assert_close(candidate_parameter.grad, baseline_parameter.grad)


def test_checkpoint_rng_preservation_requires_boolean() -> None:
    with pytest.raises(ValueError, match="preserve_rng_state"):
        PairAwareESM2(
            _small_config(activation_checkpoint_preserve_rng_state="false")  # type: ignore[arg-type]
        )


def test_disabling_checkpoint_rng_preservation_requires_checkpointing() -> None:
    with pytest.raises(ValueError, match="requires activation_checkpointing"):
        PairAwareESM2(
            _small_config(activation_checkpoint_preserve_rng_state=False)
        )
