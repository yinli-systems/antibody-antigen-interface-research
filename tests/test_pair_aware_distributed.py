"""CPU-testable contracts for pair-aware FSDP2 initialization."""

from __future__ import annotations

from dataclasses import replace

import pytest


torch = pytest.importorskip("torch")

from antibody_pretraining.pair_aware_distributed import (  # noqa: E402
    PairAwareDistributedError,
    complete_esm2_initial_state,
    fully_shard_pair_aware_esm2,
    reshard_pair_aware_esm2,
)
from antibody_pretraining.pair_aware_esm2 import (  # noqa: E402
    PairAwareESM2,
    PairAwareESM2Config,
)


def _config(**changes) -> PairAwareESM2Config:
    return replace(
        PairAwareESM2Config(
            layer_count=2,
            embed_dimension=32,
            head_count=4,
            ffn_dimension=128,
            maximum_tokens=32,
        ),
        **changes,
    )


def test_complete_initial_state_adds_only_cross_scales() -> None:
    model = PairAwareESM2(_config())
    source = {
        f"encoder.sentence_encoder.{name}": tensor.detach().clone()
        for name, tensor in model.state_dict().items()
        if not name.endswith("cross_scale")
    }
    complete = complete_esm2_initial_state(model, source)
    assert set(complete) == set(model.state_dict())
    assert {
        tensor.dtype for tensor in complete.values() if tensor.is_floating_point()
    } == {torch.float32}
    for index in range(2):
        scale = complete[f"layers.{index}.self_attn.cross_scale"]
        torch.testing.assert_close(scale, torch.ones(4))


def test_complete_initial_state_upcasts_half_checkpoint() -> None:
    model = PairAwareESM2(_config())
    source = {
        name: tensor.detach().half()
        for name, tensor in model.state_dict().items()
        if not name.endswith("cross_scale")
    }
    complete = complete_esm2_initial_state(model, source)
    assert {
        tensor.dtype for tensor in complete.values() if tensor.is_floating_point()
    } == {torch.float32}


def test_complete_initial_state_rejects_unexpected_state() -> None:
    model = PairAwareESM2(_config())
    state = {
        name: tensor.detach().clone()
        for name, tensor in model.state_dict().items()
        if not name.endswith("cross_scale")
    }
    state["untrusted.weight"] = torch.zeros(1)
    with pytest.raises(PairAwareDistributedError, match="Unexpected checkpoint"):
        complete_esm2_initial_state(model, state)


def test_fully_shard_requires_initialized_process_group() -> None:
    model = PairAwareESM2(_config())
    with pytest.raises(PairAwareDistributedError, match="process group"):
        fully_shard_pair_aware_esm2(model)


def test_reshard_rejects_an_unsharded_model() -> None:
    model = PairAwareESM2(_config())
    with pytest.raises(PairAwareDistributedError, match="not fully sharded"):
        reshard_pair_aware_esm2(model)


def test_activation_checkpointing_matches_eager_gradients() -> None:
    torch.manual_seed(23)
    eager = PairAwareESM2(_config(activation_checkpointing=False)).double()
    recompute = PairAwareESM2(_config(activation_checkpointing=True)).double()
    recompute.load_state_dict(eager.state_dict())
    eager.train()
    recompute.train()
    tokens = torch.tensor([[0, 4, 32, 2, 0, 5, 32, 2]])
    chain_ids = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]])
    labels = torch.full_like(tokens, -100)
    labels[0, 2] = 6
    labels[0, 6] = 7
    eager_loss = eager(tokens, chain_ids, labels=labels).balanced_chain_loss_sum
    recompute_loss = recompute(tokens, chain_ids, labels=labels).balanced_chain_loss_sum
    assert eager_loss is not None and recompute_loss is not None
    eager_loss.backward()
    recompute_loss.backward()
    torch.testing.assert_close(recompute_loss, eager_loss, rtol=1e-10, atol=1e-10)
    for eager_parameter, recompute_parameter in zip(
        eager.parameters(), recompute.parameters(), strict=True
    ):
        torch.testing.assert_close(
            recompute_parameter.grad,
            eager_parameter.grad,
            rtol=1e-9,
            atol=1e-9,
        )
