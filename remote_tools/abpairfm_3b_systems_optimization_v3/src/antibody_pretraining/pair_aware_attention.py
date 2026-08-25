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

from dataclasses import dataclass
import math
from typing import Literal

import torch
from torch import Tensor, nn
import torch.nn.functional as F


AttentionImplementation = Literal["lifted_sdpa", "dense"]


@dataclass(frozen=True)
class PairAttentionLayout:
    """Reusable tensor layout for one padded batch of two-chain pairs."""

    position_ids: Tensor
    chain_masks: Tensor
    query_indices: Tensor
    query_valid: Tensor


def _next_multiple(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _cumulative_sequence_lengths(mask: Tensor) -> Tensor:
    """Return FlashAttention cumulative lengths with the required int32 dtype."""
    lengths = mask.sum(dim=-1, dtype=torch.int32)
    return F.pad(lengths.cumsum(dim=0, dtype=torch.int32), (1, 0))


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

    token_count = chain_ids.shape[1]
    token_positions = torch.arange(token_count, device=chain_ids.device)
    invalid_identifier = torch.iinfo(chain_ids.dtype).max
    sortable_ids = chain_ids.masked_fill(~valid_mask, invalid_identifier)
    order = torch.argsort(sortable_ids, dim=1, stable=True)
    sorted_ids = sortable_ids.gather(1, order)
    sorted_valid = valid_mask.gather(1, order)
    boundary = sorted_valid.clone()
    if token_count > 1:
        boundary[:, 1:] &= (
            ~sorted_valid[:, :-1] | sorted_ids[:, 1:].ne(sorted_ids[:, :-1])
        )
    starts = torch.where(
        boundary,
        token_positions[None].expand_as(chain_ids),
        torch.zeros_like(chain_ids),
    ).cummax(dim=1).values
    sorted_positions = (
        token_positions[None].to(chain_ids.dtype) - starts.to(chain_ids.dtype)
    ).masked_fill(~sorted_valid, 0)
    positions = torch.zeros_like(chain_ids)
    return positions.scatter(1, order, sorted_positions)


def build_pair_attention_layout(
    chain_ids: Tensor,
    valid_mask: Tensor,
    *,
    query_capacity: int | None = None,
) -> PairAttentionLayout:
    """Build one vectorized two-chain layout reused by every transformer layer."""
    if chain_ids.ndim != 2 or valid_mask.shape != chain_ids.shape:
        raise ValueError("chain_ids and valid_mask must be equal rank-two tensors")
    batch_size, token_count = chain_ids.shape
    if query_capacity is None:
        query_capacity = token_count
    if isinstance(query_capacity, bool) or not isinstance(query_capacity, int):
        raise ValueError("query_capacity must be an integer")
    if not 1 <= query_capacity <= token_count:
        raise ValueError("query_capacity must be within the padded token length")

    chain_slots = torch.arange(2, device=chain_ids.device, dtype=chain_ids.dtype)
    chain_masks = (
        chain_ids[:, None, :].eq(chain_slots[None, :, None])
        & valid_mask[:, None, :]
    )
    chain_counts = chain_masks.sum(dim=-1)
    if bool(chain_masks.sum(dim=1).ne(valid_mask).any()):
        raise ValueError("Pair attention requires chain IDs 0 and 1 only")
    if bool(chain_counts.eq(0).any()):
        raise ValueError("Every pair requires two non-empty chains")
    if bool(chain_counts.gt(query_capacity).any()):
        raise ValueError("query_capacity is smaller than a chain in the batch")

    token_positions = torch.arange(token_count, device=chain_ids.device)
    sort_keys = torch.where(
        chain_masks,
        token_positions[None, None, :],
        token_positions[None, None, :] + token_count,
    )
    query_indices = torch.argsort(sort_keys, dim=-1, stable=True)[
        :, :, :query_capacity
    ]
    query_valid = (
        torch.arange(query_capacity, device=chain_ids.device)[None, None, :]
        < chain_counts[:, :, None]
    )
    return PairAttentionLayout(
        position_ids=chain_local_positions(chain_ids, valid_mask),
        chain_masks=chain_masks,
        query_indices=query_indices,
        query_valid=query_valid,
    )


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
    layout: PairAttentionLayout | None = None,
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

    if layout is None:
        layout = build_pair_attention_layout(chain_ids, valid_mask)
    if layout.position_ids.shape != chain_ids.shape:
        raise ValueError("Pair attention layout shape differs from chain metadata")
    if layout.chain_masks.shape != (batch_size, 2, token_count):
        raise ValueError("Pair attention layout must contain exactly two chain masks")

    query_capacity = layout.query_indices.shape[-1]
    scale = head_dimension**-0.5
    effective_dropout = dropout_p if training else 0.0
    bias_query = math.sqrt(head_dimension) * cross_logit_bias

    def cuda_varlen_attention(
        packed_query: Tensor,
        packed_key: Tensor,
        packed_value: Tensor,
        query_mask: Tensor,
        key_mask: Tensor,
        maximum_key_length: int,
    ) -> Tensor:
        """Apply exact native variable-length attention and restore queries.

        PyTorch 2.7 does not expose ``torch.nn.attention.varlen`` yet, but its
        native FlashAttention operator already accepts cumulative sequence
        lengths.  Inputs are packed before lifting so neither padding nor
        padded lifted tensors enter the CUDA kernel.
        """
        flash_forward = getattr(torch.ops.aten, "_flash_attention_forward", None)
        if flash_forward is None:
            raise RuntimeError(
                "The pinned CUDA runtime lacks native variable-length FlashAttention"
            )
        cumulative_query = _cumulative_sequence_lengths(query_mask)
        cumulative_key = _cumulative_sequence_lengths(key_mask)
        packed_output = flash_forward.default(
            packed_query.contiguous(),
            packed_key.contiguous(),
            packed_value.contiguous(),
            cumulative_query,
            cumulative_key,
            query_capacity,
            maximum_key_length,
            effective_dropout,
            False,
            False,
            scale=scale,
            window_size_left=-1,
            window_size_right=-1,
        )[0]
        packed_positions = query_mask.flatten().cumsum(dim=0) - 1
        restored = packed_output.index_select(
            0, packed_positions.clamp_min(0)
        ).view(batch_size, query_capacity, head_count, -1)
        restored = restored * query_mask[:, :, None, None].to(restored.dtype)
        return restored.permute(0, 2, 1, 3)

    def attend_chain(chain_slot: int) -> Tensor:
        query_indices = layout.query_indices[:, chain_slot]
        gather_indices = query_indices[:, None, :, None].expand(
            batch_size, head_count, query_capacity, head_dimension
        )
        raw_queries = query.gather(2, gather_indices)
        rope_queries = rotary_query.gather(2, gather_indices)
        query_valid = layout.query_valid[:, chain_slot]
        same_chain_mask = layout.chain_masks[:, chain_slot]
        key_valid = (
            valid_mask
            if allow_cross_chain
            else same_chain_mask
        )

        if not query.is_cuda:
            if not allow_cross_chain:
                independent = F.scaled_dot_product_attention(
                    rope_queries,
                    rotary_key,
                    value,
                    attn_mask=key_valid[:, None, None, :],
                    dropout_p=effective_dropout,
                    is_causal=False,
                    scale=scale,
                )
                return independent + cross_scale.sum() * 0.0
            same_chain_key = same_chain_mask[:, None, :, None]
            lifted_dimension = _next_multiple(2 * head_dimension + 1, 8)
            query_lift = torch.cat(
                (
                    rope_queries,
                    raw_queries * cross_scale[None, :, None, None],
                    query.new_full(
                        (batch_size, head_count, query_capacity, 1), bias_query
                    ),
                ),
                dim=-1,
            )
            key_lift = torch.cat(
                (
                    torch.where(
                        same_chain_key, rotary_key, torch.zeros_like(rotary_key)
                    ),
                    torch.where(same_chain_key, torch.zeros_like(key), key),
                    (~same_chain_key).expand(
                        batch_size, head_count, token_count, 1
                    ).to(key.dtype),
                ),
                dim=-1,
            )
            query_lift = F.pad(
                query_lift, (0, lifted_dimension - query_lift.shape[-1])
            )
            key_lift = F.pad(
                key_lift, (0, lifted_dimension - key_lift.shape[-1])
            )
            value_lift = F.pad(value, (0, lifted_dimension - head_dimension))
            return F.scaled_dot_product_attention(
                query_lift,
                key_lift,
                value_lift,
                attn_mask=key_valid[:, None, None, :],
                dropout_p=effective_dropout,
                is_causal=False,
                scale=scale,
            )[..., :head_dimension]

        packed_rope_queries = rope_queries.permute(0, 2, 1, 3)[query_valid]
        if not allow_cross_chain:
            packed_rope_key = rotary_key.permute(0, 2, 1, 3)[key_valid]
            packed_value = value.permute(0, 2, 1, 3)[key_valid]
            independent = cuda_varlen_attention(
                packed_rope_queries,
                packed_rope_key,
                packed_value,
                query_valid,
                key_valid,
                query_capacity,
            )
            return independent + cross_scale.sum() * 0.0

        lifted_dimension = _next_multiple(2 * head_dimension + 1, 8)
        packed_raw_queries = raw_queries.permute(0, 2, 1, 3)[query_valid]
        query_lift = torch.cat(
            (
                packed_rope_queries,
                packed_raw_queries * cross_scale[None, :, None],
                query.new_full(
                    (packed_raw_queries.shape[0], head_count, 1), bias_query
                ),
            ),
            dim=-1,
        )
        packed_rotary_key = rotary_key.permute(0, 2, 1, 3)[key_valid]
        packed_raw_key = key.permute(0, 2, 1, 3)[key_valid]
        packed_value = value.permute(0, 2, 1, 3)[key_valid]
        packed_same_chain = same_chain_mask[key_valid][:, None, None]
        key_lift = torch.cat(
            (
                torch.where(
                    packed_same_chain,
                    packed_rotary_key,
                    torch.zeros_like(packed_rotary_key),
                ),
                torch.where(
                    packed_same_chain,
                    torch.zeros_like(packed_raw_key),
                    packed_raw_key,
                ),
                (~packed_same_chain).expand(-1, head_count, 1).to(key.dtype),
            ),
            dim=-1,
        )
        query_lift = F.pad(
            query_lift, (0, lifted_dimension - query_lift.shape[-1])
        )
        key_lift = F.pad(key_lift, (0, lifted_dimension - key_lift.shape[-1]))
        value_lift = F.pad(
            packed_value, (0, lifted_dimension - head_dimension)
        )
        return cuda_varlen_attention(
            query_lift,
            key_lift,
            value_lift,
            query_valid,
            key_valid,
            token_count,
        )[..., :head_dimension]

    # Two static batch-vectorized calls bound temporary memory while reducing
    # the old per-sample launch count from ``2 * batch_size`` to exactly two.
    chain_output = torch.stack((attend_chain(0), attend_chain(1)), dim=1)

    chain_output = chain_output * layout.query_valid[:, :, None, :, None].to(
        chain_output.dtype
    )
    scatter_indices = layout.query_indices[:, None, :, :, None].expand(
        batch_size, head_count, 2, query_capacity, head_dimension
    )
    scatter_values = chain_output.permute(0, 2, 1, 3, 4)
    attended = torch.zeros_like(value)
    return attended.scatter_add(
        2,
        scatter_indices.flatten(2, 3),
        scatter_values.flatten(2, 3),
    )


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
        attention_layout: PairAttentionLayout | None = None,
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
        if attention_layout is None:
            attention_layout = build_pair_attention_layout(chain_ids, valid_mask)
        positions = attention_layout.position_ids
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
                layout=attention_layout,
            )
            weights = None
        merged = attended.transpose(1, 2).contiguous().view_as(hidden)
        return self.out_proj(merged), weights
