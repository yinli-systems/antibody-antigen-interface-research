"""FSDP2 initialization and checkpoint loading for pair-aware ESM2.

The pretrained checkpoint path follows the public PyTorch FSDP2 contract:
construct on ``meta``, shard transformer layers bottom-up, and let rank zero
mmap the authenticated full state before DCP distributes parameter shards.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from .pair_aware_esm2 import (
    PairAwareESM2,
    authenticated_torch_checkpoint,
    canonicalize_tied_esm2_state,
    upgrade_esm2_state_dict,
)


class PairAwareDistributedError(RuntimeError):
    """Report a fail-closed distributed initialization violation."""


def fully_shard_pair_aware_esm2(
    model: PairAwareESM2,
    *,
    mesh: Any = None,
    precision: str = "bfloat16",
    reshard_after_forward: bool = True,
) -> PairAwareESM2:
    """Shard each transformer layer and then all remaining root parameters."""
    if not isinstance(model, PairAwareESM2):
        raise PairAwareDistributedError("model must be PairAwareESM2")
    if precision not in {"bfloat16", "float32"}:
        raise PairAwareDistributedError("precision must be bfloat16 or float32")
    if not isinstance(reshard_after_forward, bool):
        raise PairAwareDistributedError("reshard_after_forward must be boolean")
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        raise PairAwareDistributedError("FSDP2 requires an initialized process group")
    try:
        from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    except (ImportError, AttributeError) as error:
        raise PairAwareDistributedError("Public FSDP2 is unavailable") from error

    dtype = torch.bfloat16 if precision == "bfloat16" else None
    policy = MixedPrecisionPolicy(
        param_dtype=dtype,
        reduce_dtype=dtype,
        output_dtype=None,
        cast_forward_inputs=True,
    )
    for layer in model.layers:
        fully_shard(
            layer,
            mesh=mesh,
            reshard_after_forward=reshard_after_forward,
            mp_policy=policy,
        )
    fully_shard(
        model,
        mesh=mesh,
        reshard_after_forward=reshard_after_forward,
        mp_policy=policy,
    )
    return model


def reshard_pair_aware_esm2(model: PairAwareESM2) -> None:
    """Restore every FSDP2 parameter group to its optimizer-owned DTensor shard."""
    if not isinstance(model, PairAwareESM2):
        raise PairAwareDistributedError("model must be PairAwareESM2")
    modules = [*model.layers, model]
    for module in modules:
        reshard = getattr(module, "reshard", None)
        if not callable(reshard):
            raise PairAwareDistributedError("model is not fully sharded with FSDP2")
        reshard()


def complete_esm2_initial_state(
    model: PairAwareESM2,
    state_dict: Mapping[str, Tensor],
    *,
    training_dtype: torch.dtype = torch.float32,
) -> dict[str, Tensor]:
    """Validate and upcast the official state for stable mixed-precision training."""
    if training_dtype not in {torch.float32, torch.float64}:
        raise PairAwareDistributedError(
            "training_dtype must preserve full-precision optimizer parameters"
        )
    upgraded = canonicalize_tied_esm2_state(upgrade_esm2_state_dict(state_dict))
    expected = set(model.state_dict())
    found = set(upgraded)
    initialized = {
        f"layers.{index}.self_attn.cross_scale"
        for index in range(model.config.layer_count)
    }
    if expected - found != initialized:
        raise PairAwareDistributedError(
            f"Unexpected missing checkpoint keys: {sorted(expected - found)}"
        )
    if found - expected:
        raise PairAwareDistributedError(
            f"Unexpected checkpoint keys: {sorted(found - expected)}"
        )
    result = {
        name: tensor.to(dtype=training_dtype) if tensor.is_floating_point() else tensor
        for name, tensor in upgraded.items()
    }
    for name in initialized:
        result[name] = torch.ones(
            model.config.head_count,
            dtype=training_dtype,
            device="cpu",
        )
    return result


def load_authenticated_esm2_into_fsdp2(
    model: PairAwareESM2,
    checkpoint_path: Path,
    *,
    expected_size: int,
    expected_sha256: str,
    process_group: Any = None,
) -> dict[str, Any]:
    """Collectively load an authenticated full checkpoint into FSDP2 shards."""
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        raise PairAwareDistributedError("Checkpoint distribution requires a process group")
    distributed = torch.distributed
    rank = distributed.get_rank(group=process_group)
    full_state: dict[str, Tensor] = {}
    status: list[dict[str, Any]] = [{"error": None, "state_key_count": None}]
    if rank == 0:
        try:
            payload = authenticated_torch_checkpoint(
                checkpoint_path,
                expected_size=expected_size,
                expected_sha256=expected_sha256,
            )
            full_state = complete_esm2_initial_state(model, payload["model"])
            status[0]["state_key_count"] = len(full_state)
        except BaseException as error:
            status[0]["error"] = f"{type(error).__name__}: {error}"
    distributed.broadcast_object_list(status, src=0, group=process_group)
    if status[0]["error"] is not None:
        raise PairAwareDistributedError(
            f"Rank-zero checkpoint authentication failed: {status[0]['error']}"
        )
    try:
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            set_model_state_dict,
        )
    except (ImportError, AttributeError) as error:
        raise PairAwareDistributedError("DCP state-dict APIs are unavailable") from error
    incompatible = set_model_state_dict(
        model=model,
        model_state_dict=full_state,
        options=StateDictOptions(
            full_state_dict=True,
            strict=True,
            broadcast_from_rank0=True,
        ),
    )
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise PairAwareDistributedError(
            "FSDP2 checkpoint load returned incompatible keys: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    distributed.barrier(group=process_group)
    return {
        "checkpoint_sha256": expected_sha256,
        "checkpoint_size_bytes": expected_size,
        "format_version": "abpairfm_esm2_fsdp2_initialization_v1",
        "initialized_pair_aware_tensor_count": model.config.layer_count,
        "loaded_state_key_count": status[0]["state_key_count"],
        "source_rank": 0,
        "status": "LOADED",
        "training_parameter_dtype": "torch.float32",
    }


def optimizer_after_fsdp2(
    model: nn.Module,
    *,
    learning_rate: float,
    weight_decay: float,
    betas: tuple[float, float] = (0.9, 0.95),
) -> torch.optim.AdamW:
    """Construct AdamW only after parameters have become FSDP2 DTensors."""
    if learning_rate <= 0.0:
        raise PairAwareDistributedError("learning_rate must be positive")
    if weight_decay < 0.0:
        raise PairAwareDistributedError("weight_decay cannot be negative")
    return torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=betas,
        weight_decay=weight_decay,
        fused=True,
    )
