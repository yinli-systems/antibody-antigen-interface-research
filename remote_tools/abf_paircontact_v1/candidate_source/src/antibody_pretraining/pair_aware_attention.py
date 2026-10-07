"""ESM2-compatible hybrid attention for jointly encoded protein chains.

The training path implements the hybrid score from the PPLM paper without
copying its source code.  Within-chain query/key pairs use rotary position
encoding.  Cross-chain pairs use position-free query/key scores, a learned
per-head scale, and the paper's fixed +1 cross-chain logit bias.

The default implementation lifts both score families into one SDPA call per
query chain.  This keeps the exact joint softmax over all chains while avoiding
materialization of a full attention matrix during pretraining.  Callers can
also disable cross-chain keys to obtain a partner-independent retrieval view.
The dense path exists for numerical auditing and frozen interaction-matrix
export only.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor, nn
import torch.nn.functional as F


AttentionImplementation = Literal["lifted_sdpa", "dense"]


def _next_multiple(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def chain_local_positions(chain_ids: Tensor, valid_mask: Tensor) -> Tensor:
    """Return zero-based positions reset independently for every chain.

    ``chain_ids`` and ``valid_mask`` have shape ``[batch, tokens]``.  Valid
    tokens require non-negative chain IDs; padding requires ``-1``.  Chains do
    not need to be contiguous, which makes the invariant explicit instead of
    relying on one particular tokenizer layout.
    """
    if chain_ids.ndim != 2 or valid_mask.shape != chain_ids.shape:
        raise ValueError("chain_ids and valid_mask must be equal rank-two tensors")
    if valid_mask.dtype is not torch.bool:
        raise ValueError("valid_mask must be boolean")
    if chain_ids.dtype is torch.bool or chain_ids.is_floating_point():
        raise ValueError("chain_ids must use an integer dtype")
    if bool((chain_ids[valid_mask] < 0).any()):
        raise ValueError("Valid tokens require non-negative chain IDs")
    if bool((chain_ids[~valid_mask] != -1).any()):
        raise ValueError("Padding tokens require chain ID -1")

    positions = torch.zeros_like(chain_ids)
    for batch_index in range(chain_ids.shape[0]):
        identifiers = torch.unique(chain_ids[batch_index, valid_mask[batch_index]])
        for identifier in identifiers.tolist():
            selected = valid_mask[batch_index] & chain_ids[batch_index].eq(identifier)
            positions[batch_index, selected] = torch.arange(
                int(selected.sum()), device=chain_ids.device, dtype=chain_ids.dtype
            )
    return positions


class ChainLocalRotaryEmbedding(nn.Module):
    """Apply the rotary convention used by the public ESM2 implementation."""

    def __init__(self, head_dimension: int, theta: float = 10_000.0) -> None:
        super().__init__()
        if head_dimension <= 0 or head_dimension % 2:
            raise ValueError("Rotary head_dimension must be positive and even")
        if theta <= 1.0:
            raise ValueError("Rotary theta must be greater than one")
        inverse_frequency = 1.0 / (
            theta
            ** (
                torch.arange(0, head_dimension, 2, dtype=torch.float32)
                / head_dimension
            )
        )
        self.register_buffer("inv_freq", inverse_frequency)

    @staticmethod
    def _rotate_half(value: Tensor) -> Tensor:
        first, second = value.chunk(2, dim=-1)
        return torch.cat((-second, first), dim=-1)

    def forward(self, value: Tensor, position_ids: Tensor) -> Tensor:
        if value.ndim != 4:
            raise ValueError("Rotary input must have shape [batch, heads, tokens, head]")
        if position_ids.shape != (value.shape[0], value.shape[2]):
            raise ValueError("position_ids shape differs from rotary input")
        frequencies = torch.einsum(
            "bt,d->btd", position_ids.to(self.inv_freq.dtype), self.inv_freq
        )
        angles = torch.cat((frequencies, frequencies), dim=-1)[:, None]
        cosine = angles.cos().to(value.dtype)
        sine = angles.sin().to(value.dtype)
        return value * cosine + self._rotate_half(value) * sine


def dense_hybrid_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    rotary_query: Tensor,
    rotary_key: Tensor,
    chain_ids: Tensor,
    valid_mask: Tensor,
    cross_scale: Tensor,
    *,
    cross_logit_bias: float = 1.0,
    dropout_p: float = 0.0,
    training: bool = False,
    return_weights: bool = False,
    allow_cross_chain: bool = True,
) -> tuple[Tensor, Tensor | None]:
    """Evaluate the hybrid attention formula with an explicit score matrix."""
    if query.shape != key.shape or query.shape != value.shape:
        raise ValueError("query, key, and value shapes must match")
    if rotary_query.shape != query.shape or rotary_key.shape != key.shape:
        raise ValueError("Rotary and raw query/key shapes must match")
    if query.ndim != 4:
        raise ValueError("Attention tensors must have shape [batch, heads, tokens, head]")
    batch_size, head_count, token_count, head_dimension = query.shape
    if chain_ids.shape != (batch_size, token_count) or valid_mask.shape != chain_ids.shape:
        raise ValueError("Chain metadata shape differs from attention tensors")
    if cross_scale.shape != (head_count,):
        raise ValueError("cross_scale must contain one value per attention head")
    if not 0.0 <= dropout_p < 1.0:
        raise ValueError("dropout_p must be in [0, 1)")
    if not isinstance(allow_cross_chain, bool):
        raise ValueError("allow_cross_chain must be boolean")

    inverse_sqrt_head = head_dimension**-0.5
    raw_scores = torch.matmul(query, key.transpose(-1, -2)) * inverse_sqrt_head
    rotary_scores = (
        torch.matmul(rotary_query, rotary_key.transpose(-1, -2))
        * inverse_sqrt_head
    )
    same_chain = chain_ids[:, None, :, None].eq(chain_ids[:, None, None, :])
    pair_valid = valid_mask[:, None, :, None] & valid_mask[:, None, None, :]
    same_chain &= pair_valid
    cross_scores = raw_scores * cross_scale[None, :, None, None]
    cross_scores = cross_scores + cross_logit_bias
    scores = torch.where(same_chain, rotary_scores, cross_scores)
    if not allow_cross_chain:
        scores = scores.masked_fill(pair_valid & ~same_chain, -torch.inf)
    scores = scores.masked_fill(~valid_mask[:, None, None, :], -torch.inf)

    # Avoid NaNs for padded query rows without changing any valid distribution.
    scores = torch.where(valid_mask[:, None, :, None], scores, torch.zeros_like(scores))
    softmax_dtype = (
        scores.dtype if scores.dtype is torch.float64 else torch.float32
    )
    weights = F.softmax(scores, dim=-1, dtype=softmax_dtype).to(value.dtype)
    if dropout_p and training:
        weights = F.dropout(weights, p=dropout_p, training=True)
    attended = torch.matmul(weights, value)
    attended = attended * valid_mask[:, None, :, None].to(attended.dtype)
    if return_weights:
        weights = weights * pair_valid.to(weights.dtype)
        return attended, weights
    return attended, None


def lifted_sdpa_hybrid_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    rotary_query: Tensor,
    rotary_key: Tensor,
    chain_ids: Tensor,
    valid_mask: Tensor,
    cross_scale: Tensor,
    *,
    cross_logit_bias: float = 1.0,
    dropout_p: float = 0.0,
    training: bool = False,
    allow_cross_chain: bool = True,
) -> Tensor:
    """Run exact hybrid attention without storing a full attention matrix.

    For one query chain, each lifted query is ``[q_rope, W*q_raw, sqrt(d)]``.
    Same-chain keys are ``[k_rope, 0, 0]`` and cross-chain keys are
    ``[0, k_raw, bias]``.  SDPA's scale is fixed to ``1/sqrt(d)``, so the dot
    product is exactly the dense hybrid score.  Padding to a multiple of eight
    keeps the lifted head dimension friendly to fused CUDA SDPA kernels.
    """
    if query.shape != key.shape or query.shape != value.shape:
        raise ValueError("query, key, and value shapes must match")
    if rotary_query.shape != query.shape or rotary_key.shape != key.shape:
        raise ValueError("Rotary and raw query/key shapes must match")
    if query.ndim != 4:
        raise ValueError("Attention tensors must have shape [batch, heads, tokens, head]")
    batch_size, head_count, token_count, head_dimension = query.shape
    if chain_ids.shape != (batch_size, token_count) or valid_mask.shape != chain_ids.shape:
        raise ValueError("Chain metadata shape differs from attention tensors")
    if cross_scale.shape != (head_count,):
        raise ValueError("cross_scale must contain one value per attention head")
    if not 0.0 <= dropout_p < 1.0:
        raise ValueError("dropout_p must be in [0, 1)")
    if not isinstance(allow_cross_chain, bool):
        raise ValueError("allow_cross_chain must be boolean")

    lifted_dimension = _next_multiple(2 * head_dimension + 1, 8)
    attended = torch.zeros_like(value)
    scale = head_dimension**-0.5
    effective_dropout = dropout_p if training else 0.0
    bias_query = math.sqrt(head_dimension) * cross_logit_bias

    for batch_index in range(batch_size):
        valid = valid_mask[batch_index]
        if not bool(valid.any()):
            continue
        valid_indices = torch.nonzero(valid, as_tuple=False).flatten()
        valid_chain_ids = chain_ids[batch_index, valid_indices]

        for identifier in torch.unique(valid_chain_ids).tolist():
            query_indices = torch.nonzero(
                valid & chain_ids[batch_index].eq(identifier), as_tuple=False
            ).flatten()
            key_indices = valid_indices if allow_cross_chain else query_indices
            key_chain_ids = chain_ids[batch_index, key_indices]
            raw_keys = key[batch_index : batch_index + 1, :, key_indices]
            rope_keys = rotary_key[batch_index : batch_index + 1, :, key_indices]
            values = value[batch_index : batch_index + 1, :, key_indices]
            raw_queries = query[batch_index : batch_index + 1, :, query_indices]
            rope_queries = rotary_query[batch_index : batch_index + 1, :, query_indices]
            if not allow_cross_chain:
                chain_output = F.scaled_dot_product_attention(
                    rope_queries,
                    rope_keys,
                    values,
                    dropout_p=effective_dropout,
                    is_causal=False,
                    scale=scale,
                )
                # Keep the joint-view-only scale in the distributed autograd
                # graph so DDP/FSDP can use one static parameter set.
                chain_output = chain_output + cross_scale.sum() * 0.0
                attended[batch_index : batch_index + 1, :, query_indices] = (
                    chain_output
                )
                continue
            query_lift = query.new_zeros(
                (1, head_count, query_indices.numel(), lifted_dimension)
            )
            key_lift = key.new_zeros(
                (1, head_count, key_indices.numel(), lifted_dimension)
            )
            value_lift = value.new_zeros(
                (1, head_count, key_indices.numel(), lifted_dimension)
            )
            query_lift[..., :head_dimension] = rope_queries
            query_lift[..., head_dimension : 2 * head_dimension] = (
                raw_queries * cross_scale[None, :, None, None]
            )
            query_lift[..., 2 * head_dimension] = bias_query

            same = key_chain_ids.eq(identifier)[None, None, :, None]
            key_lift[..., :head_dimension] = torch.where(
                same, rope_keys, torch.zeros_like(rope_keys)
            )
            key_lift[..., head_dimension : 2 * head_dimension] = torch.where(
                same, torch.zeros_like(raw_keys), raw_keys
            )
            key_lift[..., 2 * head_dimension] = (~same[..., 0]).to(key.dtype)
            # Flash SDPA requires Q/K/V to share one head dimension.  Appended
            # value coordinates are exactly zero, so slicing the output back to
            # ``head_dimension`` preserves the dense hybrid-attention result.
            value_lift[..., :head_dimension] = values
            chain_output = F.scaled_dot_product_attention(
                query_lift,
                key_lift,
                value_lift,
                dropout_p=effective_dropout,
                is_causal=False,
                scale=scale,
            )
            attended[batch_index : batch_index + 1, :, query_indices] = (
                chain_output[..., :head_dimension]
            )
    return attended


class PairAwareSelfAttention(nn.Module):
    """Drop-in ESM2-shaped attention projections with hybrid chain scores."""

    def __init__(
        self,
        embed_dimension: int,
        head_count: int,
        *,
        dropout_p: float = 0.0,
        cross_logit_bias: float = 1.0,
        implementation: AttentionImplementation = "lifted_sdpa",
    ) -> None:
        super().__init__()
        if embed_dimension <= 0 or head_count <= 0 or embed_dimension % head_count:
            raise ValueError("embed_dimension must be divisible by head_count")
        head_dimension = embed_dimension // head_count
        if head_dimension % 2:
            raise ValueError("Rotary head dimension must be even")
        if implementation not in {"lifted_sdpa", "dense"}:
            raise ValueError("Unknown attention implementation")
        self.embed_dimension = embed_dimension
        self.head_count = head_count
        self.head_dimension = head_dimension
        self.dropout_p = dropout_p
        self.cross_logit_bias = float(cross_logit_bias)
        self.implementation = implementation
        self.q_proj = nn.Linear(embed_dimension, embed_dimension, bias=True)
        self.k_proj = nn.Linear(embed_dimension, embed_dimension, bias=True)
        self.v_proj = nn.Linear(embed_dimension, embed_dimension, bias=True)
        self.out_proj = nn.Linear(embed_dimension, embed_dimension, bias=True)
        self.cross_scale = nn.Parameter(torch.ones(head_count))
        self.rot_emb = ChainLocalRotaryEmbedding(head_dimension)

    def forward(
        self,
        hidden: Tensor,
        chain_ids: Tensor,
        valid_mask: Tensor,
        *,
        return_attention: bool = False,
        allow_cross_chain: bool = True,
    ) -> tuple[Tensor, Tensor | None]:
        if hidden.ndim != 3:
            raise ValueError("hidden must have shape [batch, tokens, embedding]")
        batch_size, token_count, dimension = hidden.shape
        if dimension != self.embed_dimension:
            raise ValueError("hidden embedding dimension differs from the module")
        if chain_ids.shape != (batch_size, token_count) or valid_mask.shape != chain_ids.shape:
            raise ValueError("Chain metadata shape differs from hidden")
        if not isinstance(allow_cross_chain, bool):
            raise ValueError("allow_cross_chain must be boolean")

        shape = (batch_size, token_count, self.head_count, self.head_dimension)
        query = self.q_proj(hidden).view(shape).transpose(1, 2)
        key = self.k_proj(hidden).view(shape).transpose(1, 2)
        value = self.v_proj(hidden).view(shape).transpose(1, 2)
        positions = chain_local_positions(chain_ids, valid_mask)
        rotary_query = self.rot_emb(query, positions)
        rotary_key = self.rot_emb(key, positions)

        if self.implementation == "dense" or return_attention:
            attended, weights = dense_hybrid_attention(
                query,
                key,
                value,
                rotary_query,
                rotary_key,
                chain_ids,
                valid_mask,
                self.cross_scale,
                cross_logit_bias=self.cross_logit_bias,
                dropout_p=self.dropout_p,
                training=self.training,
                return_weights=return_attention,
                allow_cross_chain=allow_cross_chain,
            )
        else:
            attended = lifted_sdpa_hybrid_attention(
                query,
                key,
                value,
                rotary_query,
                rotary_key,
                chain_ids,
                valid_mask,
                self.cross_scale,
                cross_logit_bias=self.cross_logit_bias,
                dropout_p=self.dropout_p,
                training=self.training,
                allow_cross_chain=allow_cross_chain,
            )
            weights = None
        merged = attended.transpose(1, 2).contiguous().view_as(hidden)
        return self.out_proj(merged), weights
