#!/usr/bin/env python3
"""Dependency-minimal, state-dict-compatible ESM-C 600M inference runtime.

The module mirrors the sequence-only path of esm==3.1.4.  It intentionally
omits ESM3 structure/function APIs that are irrelevant to residue embeddings.
Module names and tensor operations match the published ESM-C implementation so
the original and AbCDR checkpoints can be loaded with strict=True.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


SEQUENCE_VOCAB = (
    "<cls>", "<pad>", "<eos>", "<unk>", "L", "A", "G", "V", "S", "E",
    "R", "T", "I", "D", "P", "K", "Q", "N", "F", "Y", "M", "H", "W",
    "C", "X", "B", "U", "Z", "O", ".", "-", "|", "<mask>",
)
TOKEN_TO_ID = {token: index for index, token in enumerate(SEQUENCE_VOCAB)}
PAD_TOKEN_ID = TOKEN_TO_ID["<pad>"]


def tokenize_sequence(sequence: str) -> torch.Tensor:
    ids = [TOKEN_TO_ID["<cls>"]]
    ids.extend(TOKEN_TO_ID.get(residue, TOKEN_TO_ID["<unk>"]) for residue in sequence)
    ids.append(TOKEN_TO_ID["<eos>"])
    return torch.tensor(ids, dtype=torch.int64)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1 / (
            base
            ** (
                torch.arange(0, dim, 2, dtype=torch.float32)
                / dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: torch.Tensor | None = None
        self._sin_cached: torch.Tensor | None = None

    def materialize_inv_freq(self, device: torch.device) -> None:
        """Recreate the sole non-persistent buffer after meta initialization."""
        self.inv_freq = 1 / (
            self.base
            ** (
                torch.arange(0, self.dim, 2, dtype=torch.float32, device=device)
                / self.dim
            )
        )

    def _cache(self, length: int, device: torch.device, dtype: torch.dtype) -> None:
        if (
            length > self._seq_len_cached
            or self._cos_cached is None
            or self._cos_cached.device != device
            or self._cos_cached.dtype != dtype
        ):
            self._seq_len_cached = length
            positions = torch.arange(length, device=device, dtype=torch.float32)
            frequencies = torch.outer(positions, self.inv_freq.to(device=device))
            duplicated = torch.cat((frequencies, frequencies), dim=-1)
            self._cos_cached = duplicated.cos().to(dtype)
            self._sin_cached = duplicated.sin().to(dtype)

    def forward(
        self, query: torch.Tensor, key: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._cache(query.shape[1], query.device, query.dtype)
        assert self._cos_cached is not None and self._sin_cached is not None
        cosine = self._cos_cached[: query.shape[1]][None, :, None, :]
        sine = self._sin_cached[: query.shape[1]][None, :, None, :]
        return (
            query * cosine + rotate_half(query) * sine,
            key * cosine + rotate_half(key) * sine,
        )


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.layernorm_qkv = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model * 3, bias=False)
        )
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.q_ln = nn.LayerNorm(d_model, bias=False)
        self.k_ln = nn.LayerNorm(d_model, bias=False)
        self.rotary = RotaryEmbedding(self.d_head)

    def forward(self, x: torch.Tensor, sequence_id: torch.Tensor | None) -> torch.Tensor:
        query, key, value = self.layernorm_qkv(x).chunk(3, dim=-1)
        query = self.q_ln(query).to(query.dtype)
        key = self.k_ln(key).to(query.dtype)
        batch, length, _ = query.shape
        query = query.view(batch, length, self.n_heads, self.d_head)
        key = key.view(batch, length, self.n_heads, self.d_head)
        query, key = self.rotary(query, key)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.view(batch, length, self.n_heads, self.d_head).transpose(1, 2)
        mask = None
        if sequence_id is not None:
            mask = (sequence_id[:, None, :, None] == sequence_id[:, None, None, :])
        context = F.scaled_dot_product_attention(query, key, value, attn_mask=mask)
        context = context.transpose(1, 2).contiguous().view(batch, length, self.d_model)
        return self.out_proj(context)


def swiglu_correction_fn(expansion_ratio: float, d_model: int) -> int:
    return int(((expansion_ratio * d_model) + 255) // 256 * 256)


class SwiGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first, second = x.chunk(2, dim=-1)
        return F.silu(first) * second


class UnifiedTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, expansion_ratio: float):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        hidden = swiglu_correction_fn(expansion_ratio, d_model)
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden * 2, bias=False),
            SwiGLU(),
            nn.Linear(hidden, d_model, bias=False),
        )

    def forward(self, x: torch.Tensor, sequence_id: torch.Tensor | None) -> torch.Tensor:
        x = x + self.attn(x, sequence_id)
        return x + self.ffn(x)


class TransformerStack(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_layers: int):
        super().__init__()
        expansion_ratio = 8 / 3
        self.blocks = nn.ModuleList(
            UnifiedTransformerBlock(d_model, n_heads, expansion_ratio)
            for _ in range(n_layers)
        )
        self.norm = nn.LayerNorm(d_model, bias=False)

    def forward(self, x: torch.Tensor, sequence_id: torch.Tensor | None) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, sequence_id)
        return self.norm(x)


def regression_head(d_model: int, output_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(d_model, d_model),
        nn.GELU(),
        nn.LayerNorm(d_model),
        nn.Linear(d_model, output_dim),
    )


@dataclass
class ESMCOutput:
    embeddings: torch.Tensor


class MinimalESMC600M(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(64, 1152)
        self.transformer = TransformerStack(1152, 18, 36)

    def forward(self, sequence_tokens: torch.Tensor) -> ESMCOutput:
        sequence_id = sequence_tokens != PAD_TOKEN_ID
        embeddings = self.transformer(self.embed(sequence_tokens), sequence_id)
        return ESMCOutput(embeddings)

    def materialize_nonpersistent_buffers(self, device: torch.device) -> None:
        for block in self.transformer.blocks:
            block.attn.rotary.materialize_inv_freq(device)


def extract_paired_residue_embeddings(
    embeddings: torch.Tensor, heavy_length: int, light_length: int
) -> torch.Tensor:
    if embeddings.ndim != 3 or embeddings.shape[0] != 1:
        raise ValueError("expected one batched paired embedding")
    expected = heavy_length + light_length + 3  # BOS, separator, EOS
    if embeddings.shape[1] != expected:
        raise ValueError("paired token length differs")
    heavy = embeddings[0, 1 : 1 + heavy_length]
    light = embeddings[0, 2 + heavy_length : 2 + heavy_length + light_length]
    result = torch.cat((heavy, light), dim=0)
    if result.shape != (heavy_length + light_length, 1152):
        raise ValueError("paired residue export differs")
    return result
