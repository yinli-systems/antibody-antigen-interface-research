"""Numerical contracts for the clean-room pair-aware attention core."""

from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from antibody_pretraining.pair_aware_attention import (  # noqa: E402
    ChainLocalRotaryEmbedding,
    PairAwareSelfAttention,
    chain_local_positions,
    dense_hybrid_attention,
    lifted_sdpa_hybrid_attention,
)


def test_chain_local_positions_reset_and_reject_bad_padding() -> None:
    chain_ids = torch.tensor([[0, 0, 1, 0, 1, -1], [3, 4, 4, -1, -1, -1]])
    valid = chain_ids.ne(-1)
    positions = chain_local_positions(chain_ids, valid)
    assert positions.tolist() == [[0, 1, 0, 2, 1, 0], [0, 0, 1, 0, 0, 0]]

    invalid = chain_ids.clone()
    invalid[0, -1] = 0
    with pytest.raises(ValueError, match="Padding tokens require"):
        chain_local_positions(invalid, valid)


def test_lifted_sdpa_matches_dense_hybrid_outputs_and_gradients() -> None:
    torch.manual_seed(7)
    shape = (2, 3, 7, 4)
    query = torch.randn(shape, dtype=torch.float64, requires_grad=True)
    key = torch.randn(shape, dtype=torch.float64, requires_grad=True)
    value = torch.randn(shape, dtype=torch.float64, requires_grad=True)
    chain_ids = torch.tensor([[0, 0, 1, 1, 2, -1, -1], [4, 5, 5, 6, 6, 6, -1]])
    valid = chain_ids.ne(-1)
    positions = chain_local_positions(chain_ids, valid)
    rotary = ChainLocalRotaryEmbedding(shape[-1]).to(dtype=torch.float64)
    rotary_query = rotary(query, positions)
    rotary_key = rotary(key, positions)
    cross_scale = torch.tensor([-0.35, 0.8, 1.7], dtype=torch.float64, requires_grad=True)

    dense, _ = dense_hybrid_attention(
        query,
        key,
        value,
        rotary_query,
        rotary_key,
        chain_ids,
        valid,
        cross_scale,
    )
    dense_loss = dense.square().sum()
    dense_gradients = torch.autograd.grad(
        dense_loss, (query, key, value, cross_scale), retain_graph=True
    )
    lifted = lifted_sdpa_hybrid_attention(
        query,
        key,
        value,
        rotary_query,
        rotary_key,
        chain_ids,
        valid,
        cross_scale,
    )
    lifted_gradients = torch.autograd.grad(
        lifted.square().sum(), (query, key, value, cross_scale)
    )

    torch.testing.assert_close(lifted, dense, rtol=1e-10, atol=1e-10)
    for observed, expected in zip(lifted_gradients, dense_gradients, strict=True):
        torch.testing.assert_close(observed, expected, rtol=1e-9, atol=1e-9)


def test_module_training_path_matches_dense_export_path() -> None:
    torch.manual_seed(11)
    lifted = PairAwareSelfAttention(32, 4, implementation="lifted_sdpa").double()
    dense = PairAwareSelfAttention(32, 4, implementation="dense").double()
    dense.load_state_dict(lifted.state_dict())
    hidden = torch.randn(2, 8, 32, dtype=torch.float64)
    chain_ids = torch.tensor(
        [[0, 0, 0, 1, 1, 2, -1, -1], [0, 1, 1, 1, 2, 2, 2, -1]]
    )
    valid = chain_ids.ne(-1)

    lifted_output, no_weights = lifted(hidden, chain_ids, valid)
    dense_output, weights = dense(hidden, chain_ids, valid, return_attention=True)

    assert no_weights is None
    assert weights is not None and weights.shape == (2, 4, 8, 8)
    torch.testing.assert_close(lifted_output, dense_output, rtol=1e-10, atol=1e-10)
    valid_row_sums = weights.sum(dim=-1)[valid[:, None, :].expand(-1, 4, -1)]
    torch.testing.assert_close(valid_row_sums, torch.ones_like(valid_row_sums))
    assert torch.count_nonzero(weights[:, :, :, -1]).item() == 0
