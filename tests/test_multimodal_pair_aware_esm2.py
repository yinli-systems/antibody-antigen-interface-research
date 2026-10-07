"""Correctness tests for predicted-structure conditioning."""

from __future__ import annotations

from dataclasses import replace

import pytest


torch = pytest.importorskip("torch")

from antibody_pretraining.multimodal_pair_aware_esm2 import (  # noqa: E402
    StructureConditionedPairAwareESM2,
    StructureConditioningConfig,
)
from antibody_pretraining.pair_aware_esm2 import PairAwareESM2Config  # noqa: E402


def small_model() -> StructureConditionedPairAwareESM2:
    backbone = PairAwareESM2Config(
        layer_count=2,
        embed_dimension=32,
        head_count=4,
        ffn_dimension=128,
        maximum_tokens=16,
    )
    structure = replace(
        StructureConditioningConfig(),
        feature_dimension=5,
        projection_dimension=16,
    )
    return StructureConditionedPairAwareESM2(backbone, structure)


def inputs():
    tokens = torch.tensor([[0, 5, 6, 2, 0, 7, 8, 2, 1]])
    chain_ids = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1, -1]])
    structure_tokens = torch.tensor([[0, 3, 1, 0, 0, 4, 5, 2, 0]])
    feature_mask = torch.tensor(
        [[False, True, False, False, False, True, True, False, False]]
    )
    features = torch.zeros(1, 9, 5)
    features[feature_mask] = torch.tensor(
        [[0.1, 0.2, 0.3, 0.4, 0.5]] * 3
    )
    return tokens, chain_ids, structure_tokens, features, feature_mask


def test_zero_gate_is_exactly_sequence_compatible_then_conditions() -> None:
    torch.manual_seed(41)
    model = small_model().eval()
    tokens, chain_ids, structure_tokens, features, feature_mask = inputs()
    baseline = model(tokens, chain_ids).logits
    gated = model(
        tokens,
        chain_ids,
        structure_tokens=structure_tokens,
        structure_features=features,
        structure_feature_mask=feature_mask,
    ).logits
    assert baseline is not None and gated is not None
    torch.testing.assert_close(gated, baseline, rtol=0.0, atol=0.0)

    with torch.no_grad():
        model.structure_residual_gate.fill_(1.0)
    conditioned = model(
        tokens,
        chain_ids,
        structure_tokens=structure_tokens,
        structure_features=features,
        structure_feature_mask=feature_mask,
    ).logits
    assert conditioned is not None
    assert not torch.equal(conditioned, baseline)


def test_structure_conditioning_rejects_partial_or_leaking_inputs() -> None:
    model = small_model().eval()
    tokens, chain_ids, structure_tokens, features, feature_mask = inputs()
    with pytest.raises(ValueError, match="supplied together"):
        model(tokens, chain_ids, structure_tokens=structure_tokens)

    bad_features = features.clone()
    bad_features[:, 2, 0] = 1.0
    with pytest.raises(ValueError, match="must be zero"):
        model(
            tokens,
            chain_ids,
            structure_tokens=structure_tokens,
            structure_features=bad_features,
            structure_feature_mask=feature_mask,
        )

    bad_tokens = structure_tokens.clone()
    bad_tokens[:, -1] = 3
    with pytest.raises(ValueError, match="Padding positions"):
        model(
            tokens,
            chain_ids,
            structure_tokens=bad_tokens,
            structure_features=features,
            structure_feature_mask=feature_mask,
        )


def test_structure_reconstruction_is_chain_balanced_and_finite() -> None:
    torch.manual_seed(43)
    model = small_model().train()
    tokens, chain_ids, structure_tokens, features, feature_mask = inputs()
    output = model(
        tokens,
        chain_ids,
        structure_tokens=structure_tokens,
        structure_features=features,
        structure_feature_mask=feature_mask,
        return_hidden_states=True,
        compute_logits=False,
    )
    assert output.hidden_states is not None
    token_labels = torch.full_like(tokens, -100)
    token_labels[:, 2] = 6
    token_labels[:, 5] = 7
    feature_targets = torch.zeros_like(features)
    feature_target_mask = torch.zeros_like(feature_mask)
    feature_target_mask[:, 2] = True
    feature_target_mask[:, 5] = True
    feature_targets[:, 2] = 0.25
    feature_targets[:, 5] = -0.25

    losses = model.structure_reconstruction_loss(
        output.hidden_states,
        chain_ids,
        tokens.ne(model.config.padding_index),
        token_labels,
        feature_targets,
        feature_target_mask,
    )
    assert losses.token_chain_count.item() == 2
    assert losses.feature_chain_count.item() == 2
    assert losses.token_target_count == 2
    assert losses.feature_target_count == 2
    total = losses.token_chain_loss_sum + losses.feature_chain_loss_sum
    assert torch.isfinite(total)
    total.backward()
    assert model.structure_token_embedding.weight.grad is not None
    assert model.structure_feature_decoder.weight.grad is not None
