"""Checkpoint-compatible ESM2 encoder with clean-room pair-aware attention.

The module names, layer normalization, GELU, token-dropout, and tied MLM head
match Meta's MIT-licensed ESM2 implementation.  The only architectural addition
inside a pretrained transformer layer is ``self_attn.cross_scale``; attention
score semantics are implemented in :mod:`pair_aware_attention` from the
CC BY 4.0 PPLM paper equation, not from the separately licensed PPLM source.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import re
from typing import Any, Literal, Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .pair_aware_attention import AttentionImplementation, PairAwareSelfAttention


ESM2_ALPHABET = (
    "<cls>",
    "<pad>",
    "<eos>",
    "<unk>",
    "L",
    "A",
    "G",
    "V",
    "S",
    "E",
    "R",
    "T",
    "I",
    "D",
    "P",
    "K",
    "Q",
    "N",
    "F",
    "Y",
    "M",
    "H",
    "W",
    "C",
    "X",
    "B",
    "U",
    "Z",
    "O",
    ".",
    "-",
    "<null_1>",
    "<mask>",
)

_SELECTIVE_ATTENTION_CHECKPOINT_COUNTS = {
    "flash_must_save": 0,
    "prefer_recompute": 0,
}


@dataclass(frozen=True)
class PairAwareESM2Config:
    """Freeze the ESM2 dimensions and pair-aware runtime choice."""

    layer_count: int = 36
    embed_dimension: int = 2560
    head_count: int = 40
    ffn_dimension: int = 10240
    vocabulary_size: int = len(ESM2_ALPHABET)
    padding_index: int = 1
    mask_index: int = 32
    token_dropout: bool = True
    attention_implementation: AttentionImplementation = "lifted_sdpa"
    cross_logit_bias: float = 1.0
    maximum_tokens: int = 1024
    activation_checkpointing: bool = False
    activation_checkpoint_policy: Literal["full", "selective_attention"] = "full"
    activation_checkpoint_preserve_rng_state: bool = True

    def validate(self) -> None:
        for name in (
            "layer_count",
            "embed_dimension",
            "head_count",
            "ffn_dimension",
            "vocabulary_size",
            "maximum_tokens",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.embed_dimension % self.head_count:
            raise ValueError("embed_dimension must be divisible by head_count")
        if (self.embed_dimension // self.head_count) % 2:
            raise ValueError("Rotary head dimension must be even")
        if not 0 <= self.padding_index < self.vocabulary_size:
            raise ValueError("padding_index is outside the vocabulary")
        if not 0 <= self.mask_index < self.vocabulary_size:
            raise ValueError("mask_index is outside the vocabulary")
        if self.attention_implementation not in {"lifted_sdpa", "dense"}:
            raise ValueError("Unknown attention implementation")
        if not isinstance(self.activation_checkpointing, bool):
            raise ValueError("activation_checkpointing must be boolean")
        if self.activation_checkpoint_policy not in {"full", "selective_attention"}:
            raise ValueError("Unknown activation checkpoint policy")
        if not isinstance(self.activation_checkpoint_preserve_rng_state, bool):
            raise ValueError("activation_checkpoint_preserve_rng_state must be boolean")
        if not self.activation_checkpointing and self.activation_checkpoint_policy != "full":
            raise ValueError(
                "selective activation checkpointing requires activation_checkpointing"
            )
        if (
            not self.activation_checkpointing
            and not self.activation_checkpoint_preserve_rng_state
        ):
            raise ValueError(
                "disabling checkpoint RNG preservation requires activation_checkpointing"
            )


def selective_attention_checkpoint_policy(
    context: object,
    operation: object,
    *args: object,
    **kwargs: object,
):
    """Cache only the fused attention op while recomputing cheaper layer work."""
    del context, args, kwargs
    from torch.utils.checkpoint import CheckpointPolicy

    flash_attention = torch.ops.aten._scaled_dot_product_flash_attention.default
    if operation is flash_attention:
        _SELECTIVE_ATTENTION_CHECKPOINT_COUNTS["flash_must_save"] += 1
        return CheckpointPolicy.MUST_SAVE
    _SELECTIVE_ATTENTION_CHECKPOINT_COUNTS["prefer_recompute"] += 1
    return CheckpointPolicy.PREFER_RECOMPUTE


def selective_attention_checkpoint_contexts():
    """Build fresh selective-checkpoint modes for one transformer invocation."""
    from torch.utils.checkpoint import create_selective_checkpoint_contexts

    return create_selective_checkpoint_contexts(selective_attention_checkpoint_policy)


def reset_selective_attention_checkpoint_policy_counts() -> None:
    """Reset per-process dispatch evidence before a systems measurement."""
    for name in _SELECTIVE_ATTENTION_CHECKPOINT_COUNTS:
        _SELECTIVE_ATTENTION_CHECKPOINT_COUNTS[name] = 0


def selective_attention_checkpoint_policy_counts() -> dict[str, int]:
    """Return per-process evidence that the fused attention policy was exercised."""
    return dict(_SELECTIVE_ATTENTION_CHECKPOINT_COUNTS)


@dataclass(frozen=True)
class PairAwareMLMOutput:
    """Return logits plus a chain-balanced distributed loss numerator."""

    logits: Tensor | None
    balanced_chain_loss_sum: Tensor | None
    contributing_chain_count: Tensor
    last_attention: Tensor | None
    hidden_states: Tensor | None


def esm_gelu(value: Tensor) -> Tensor:
    """Use the exact GELU expression in the public ESM2 implementation."""
    return value * 0.5 * (1.0 + torch.erf(value / math.sqrt(2.0)))


class PairAwareESM2Layer(nn.Module):
    """One pre-normalized ESM2 block with hybrid multi-chain attention."""

    def __init__(self, config: PairAwareESM2Config) -> None:
        super().__init__()
        self.self_attn = PairAwareSelfAttention(
            config.embed_dimension,
            config.head_count,
            cross_logit_bias=config.cross_logit_bias,
            implementation=config.attention_implementation,
        )
        self.self_attn_layer_norm = nn.LayerNorm(config.embed_dimension)
        self.fc1 = nn.Linear(config.embed_dimension, config.ffn_dimension)
        self.fc2 = nn.Linear(config.ffn_dimension, config.embed_dimension)
        self.final_layer_norm = nn.LayerNorm(config.embed_dimension)

    def forward(
        self,
        hidden: Tensor,
        chain_ids: Tensor,
        valid_mask: Tensor,
        *,
        return_attention: bool = False,
        allow_cross_chain_attention: bool = True,
    ) -> tuple[Tensor, Tensor | None]:
        residual = hidden
        normalized = self.self_attn_layer_norm(hidden)
        attended, weights = self.self_attn(
            normalized,
            chain_ids,
            valid_mask,
            return_attention=return_attention,
            allow_cross_chain=allow_cross_chain_attention,
        )
        hidden = residual + attended
        residual = hidden
        hidden = self.fc2(esm_gelu(self.fc1(self.final_layer_norm(hidden))))
        return residual + hidden, weights


class ESM2RobertaLMHead(nn.Module):
    """Reproduce the ESM2 head without registering its tied weight twice."""

    def __init__(self, embed_dimension: int, vocabulary_size: int) -> None:
        super().__init__()
        self.dense = nn.Linear(embed_dimension, embed_dimension)
        self.layer_norm = nn.LayerNorm(embed_dimension)
        self.bias = nn.Parameter(torch.zeros(vocabulary_size))

    def forward(self, hidden: Tensor, tied_weight: Tensor) -> Tensor:
        hidden = self.layer_norm(esm_gelu(self.dense(hidden)))
        return F.linear(hidden, tied_weight) + self.bias


class PairAwareESM2(nn.Module):
    """Jointly encode physical chains with ESM2-compatible parameters."""

    def __init__(self, config: PairAwareESM2Config = PairAwareESM2Config()) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.embed_tokens = nn.Embedding(
            config.vocabulary_size,
            config.embed_dimension,
            padding_idx=config.padding_index,
        )
        self.layers = nn.ModuleList(PairAwareESM2Layer(config) for _ in range(config.layer_count))
        self.emb_layer_norm_after = nn.LayerNorm(config.embed_dimension)
        self.lm_head = ESM2RobertaLMHead(
            config.embed_dimension,
            config.vocabulary_size,
        )

    def _embed(self, tokens: Tensor, valid_mask: Tensor) -> Tensor:
        hidden = self.embed_tokens(tokens)
        if self.config.token_dropout:
            masked = tokens.eq(self.config.mask_index)
            hidden = hidden.masked_fill(masked.unsqueeze(-1), 0.0)
            expected_mask_ratio = 0.15 * 0.8
            lengths = valid_mask.sum(dim=-1)
            if bool(lengths.eq(0).any()):
                raise ValueError("Every model input must contain at least one valid token")
            observed_ratio = masked.sum(dim=-1).to(hidden.dtype) / lengths.to(hidden.dtype)
            if bool(observed_ratio.ge(1.0).any()):
                raise ValueError("Token dropout cannot receive an all-mask sequence")
            hidden = hidden * (
                (1.0 - expected_mask_ratio) / (1.0 - observed_ratio)
            )[:, None, None]
        return hidden * valid_mask.unsqueeze(-1).to(hidden.dtype)

    @staticmethod
    def _balanced_chain_loss(
        logits: Tensor,
        labels: Tensor,
        chain_ids: Tensor,
        valid_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if labels.shape != chain_ids.shape:
            raise ValueError("labels shape differs from chain metadata")
        if bool((labels[~valid_mask] != -100).any()):
            raise ValueError("Padding positions cannot be MLM targets")
        token_losses = F.cross_entropy(
            logits.flatten(0, 1), labels.flatten(), ignore_index=-100, reduction="none"
        ).view_as(labels)
        selected = labels.ne(-100)
        loss_sum = logits.sum() * 0.0
        chain_count = torch.zeros((), dtype=torch.long, device=logits.device)
        for batch_index in range(chain_ids.shape[0]):
            identifiers = torch.unique(chain_ids[batch_index, valid_mask[batch_index]])
            for identifier in identifiers.tolist():
                chain_selected = selected[batch_index] & chain_ids[batch_index].eq(identifier)
                target_count = chain_selected.sum()
                if not bool(target_count):
                    raise ValueError("Every valid chain requires at least one MLM target")
                loss_sum = loss_sum + token_losses[batch_index, chain_selected].sum() / target_count
                chain_count = chain_count + 1
        return loss_sum, chain_count

    def forward(
        self,
        tokens: Tensor,
        chain_ids: Tensor,
        *,
        labels: Tensor | None = None,
        return_last_attention: bool = False,
        allow_cross_chain_attention: bool = True,
        return_hidden_states: bool = False,
        compute_logits: bool = True,
    ) -> PairAwareMLMOutput:
        if tokens.ndim != 2 or chain_ids.shape != tokens.shape:
            raise ValueError("tokens and chain_ids must be equal rank-two tensors")
        if tokens.shape[1] > self.config.maximum_tokens:
            raise ValueError("Input exceeds the frozen maximum token count")
        if not isinstance(allow_cross_chain_attention, bool):
            raise ValueError("allow_cross_chain_attention must be boolean")
        if not isinstance(return_hidden_states, bool):
            raise ValueError("return_hidden_states must be boolean")
        if not isinstance(compute_logits, bool):
            raise ValueError("compute_logits must be boolean")
        if labels is not None and not compute_logits:
            raise ValueError("MLM labels require compute_logits")
        valid_mask = tokens.ne(self.config.padding_index)
        if bool((chain_ids[valid_mask] < 0).any()) or bool((chain_ids[~valid_mask] != -1).any()):
            raise ValueError("chain_ids must be non-negative for tokens and -1 for padding")
        hidden = self._embed(tokens, valid_mask)
        last_attention = None
        for layer_index, layer in enumerate(self.layers):
            if os.environ.get("ABPAIRFM_TRACE_LAYERS") == "1":
                print(
                    f"ABPAIRFM_LAYER rank={os.environ.get('RANK', '0')} "
                    f"layer={layer_index} phase=forward_start",
                    flush=True,
                )
            export_attention = (
                return_last_attention and layer_index == len(self.layers) - 1
            )
            if self.training and self.config.activation_checkpointing and not export_attention:
                checkpoint_arguments: dict[str, Any] = {
                    "preserve_rng_state": (
                        self.config.activation_checkpoint_preserve_rng_state
                    ),
                    "use_reentrant": False,
                }
                if self.config.activation_checkpoint_policy == "selective_attention":
                    checkpoint_arguments["context_fn"] = (
                        selective_attention_checkpoint_contexts
                    )
                hidden = checkpoint(
                    lambda state,
                    chains,
                    valid,
                    module=layer,
                    allow_cross=allow_cross_chain_attention: module(
                        state,
                        chains,
                        valid,
                        return_attention=False,
                        allow_cross_chain_attention=allow_cross,
                    )[0],
                    hidden,
                    chain_ids,
                    valid_mask,
                    **checkpoint_arguments,
                )
                attention = None
            else:
                hidden, attention = layer(
                    hidden,
                    chain_ids,
                    valid_mask,
                    return_attention=export_attention,
                    allow_cross_chain_attention=allow_cross_chain_attention,
                )
            if attention is not None:
                last_attention = attention
            if os.environ.get("ABPAIRFM_TRACE_LAYERS") == "1":
                print(
                    f"ABPAIRFM_LAYER rank={os.environ.get('RANK', '0')} "
                    f"layer={layer_index} phase=forward_done",
                    flush=True,
                )
        hidden = self.emb_layer_norm_after(hidden)
        logits = (
            self.lm_head(hidden, self.embed_tokens.weight)
            if compute_logits
            else None
        )
        loss_sum = None
        chain_count = torch.zeros((), dtype=torch.long, device=tokens.device)
        if labels is not None:
            assert logits is not None
            loss_sum, chain_count = self._balanced_chain_loss(
                logits, labels, chain_ids, valid_mask
            )
        return PairAwareMLMOutput(
            logits=logits,
            balanced_chain_loss_sum=loss_sum,
            contributing_chain_count=chain_count,
            last_attention=last_attention,
            hidden_states=hidden if return_hidden_states else None,
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def upgrade_esm2_state_dict(state_dict: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Remove only the two prefixes used by official ESM2 checkpoints."""
    pattern = re.compile(r"^(encoder\.sentence_encoder\.|encoder\.)")
    upgraded: dict[str, Tensor] = {}
    for name, tensor in state_dict.items():
        normalized = pattern.sub("", name)
        if normalized in upgraded:
            raise ValueError(f"Duplicate upgraded ESM2 state key: {normalized}")
        upgraded[normalized] = tensor
    return upgraded


def canonicalize_tied_esm2_state(
    state_dict: Mapping[str, Tensor],
    *,
    require_lm_head_duplicate: bool = False,
) -> dict[str, Tensor]:
    """Validate and remove the checkpoint-only duplicate tied LM-head weight."""
    canonical = dict(state_dict)
    duplicate = canonical.pop("lm_head.weight", None)
    embedding = canonical.get("embed_tokens.weight")
    if duplicate is None:
        if require_lm_head_duplicate:
            raise ValueError("Checkpoint is missing its tied LM-head duplicate")
        return canonical
    if embedding is None or not torch.equal(duplicate, embedding):
        raise ValueError("Checkpoint tied token and LM-head values differ")
    return canonical


def load_compatible_esm2_state(
    model: PairAwareESM2, state_dict: Mapping[str, Tensor]
) -> dict[str, Any]:
    """Load every compatible ESM2 tensor and only initialize cross scales."""
    upgraded = canonicalize_tied_esm2_state(upgrade_esm2_state_dict(state_dict))
    expected = set(model.state_dict())
    found = set(upgraded)
    allowed_missing = {
        f"layers.{index}.self_attn.cross_scale" for index in range(model.config.layer_count)
    }
    missing = expected - found
    unexpected = found - expected
    if missing != allowed_missing:
        raise RuntimeError(f"Unexpected missing ESM2 keys: {sorted(missing)}")
    if unexpected:
        raise RuntimeError(f"Unexpected ESM2 checkpoint keys: {sorted(unexpected)}")
    result = model.load_state_dict(upgraded, strict=False)
    if set(result.missing_keys) != allowed_missing or result.unexpected_keys:
        raise RuntimeError("PyTorch state loading disagrees with the compatibility audit")
    with torch.no_grad():
        for layer in model.layers:
            layer.self_attn.cross_scale.fill_(1.0)
    loaded_parameter_count = sum(upgraded[name].numel() for name in upgraded)
    return {
        "allowed_initialized_keys": sorted(allowed_missing),
        "format_version": "abpairfm_esm2_compatible_load_v1",
        "loaded_state_key_count": len(upgraded),
        "loaded_tensor_element_count": loaded_parameter_count,
        "unexpected_keys": [],
    }


def authenticated_torch_checkpoint(
    path: Path, *, expected_size: int, expected_sha256: str
) -> Mapping[str, Any]:
    """Deserialize a trusted official checkpoint only after byte authentication."""
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("Checkpoint must be a regular file")
    if path.stat().st_size != expected_size:
        raise RuntimeError("Checkpoint byte size differs from the frozen source anchor")
    if _sha256(path) != expected_sha256:
        raise RuntimeError("Checkpoint SHA-256 differs from the frozen source anchor")
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("model"), Mapping):
        raise RuntimeError("Official checkpoint does not contain a model state mapping")
    return payload


def parameter_count(model: nn.Module) -> int:
    """Count unique parameters despite the tied token/LM-head weight."""
    unique: dict[int, nn.Parameter] = {}
    for parameter in model.parameters():
        unique[id(parameter)] = parameter
    return sum(parameter.numel() for parameter in unique.values())
