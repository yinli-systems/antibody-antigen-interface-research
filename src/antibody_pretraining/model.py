"""Bidirectional rotary Transformer for antibody foundation pretraining."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .tokenizer import CHAIN_TYPE_TO_ID, VOCABULARY


@dataclass(frozen=True)
class AntibodyModelConfig:
    """Define one exact architecture without downstream-dependent fields."""

    d_model: int
    ffn_hidden_size: int
    maximum_sequence_tokens: int
    n_heads: int
    n_layers: int
    qk_normalization: bool = True
    rope_theta: float = 10_000.0
    vocabulary_size: int = len(VOCABULARY)
    chain_type_count: int = len(CHAIN_TYPE_TO_ID)
    normalization_epsilon: float = 1e-6

    def validate(self) -> None:
        """Reject invalid dimensions before allocating parameters."""
        integer_fields = (
            "d_model",
            "ffn_hidden_size",
            "maximum_sequence_tokens",
            "n_heads",
            "n_layers",
            "vocabulary_size",
            "chain_type_count",
        )
        for field in integer_fields:
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field} must be a positive integer")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if (self.d_model // self.n_heads) % 2:
            raise ValueError("Rotary head dimension must be even")
        if self.rope_theta <= 1.0 or self.normalization_epsilon <= 0.0:
            raise ValueError("RoPE theta and normalization epsilon must be positive")


@dataclass(frozen=True)
class MLMOutput:
    """Return logits and an unnormalized selected-target loss."""

    logits: Tensor
    loss_sum: Tensor | None
    target_count: Tensor


class RMSNorm(nn.Module):
    """Apply parameterized root-mean-square normalization in FP32."""

    def __init__(self, dimension: int, epsilon: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dimension))
        self.epsilon = epsilon

    def forward(self, value: Tensor) -> Tensor:
        source_dtype = value.dtype
        normalized = value.float() * torch.rsqrt(
            value.float().pow(2).mean(dim=-1, keepdim=True) + self.epsilon
        )
        return normalized.to(source_dtype) * self.weight


class RotaryEmbedding(nn.Module):
    """Generate per-position rotary sine and cosine tables."""

    def __init__(self, head_dimension: int, maximum_length: int, theta: float) -> None:
        super().__init__()
        inverse_frequency = 1.0 / (
            theta
            ** (
                torch.arange(0, head_dimension, 2, dtype=torch.float32)
                / head_dimension
            )
        )
        positions = torch.arange(maximum_length, dtype=torch.float32)
        angles = torch.outer(positions, inverse_frequency)
        self.register_buffer("cosine", angles.cos(), persistent=False)
        self.register_buffer("sine", angles.sin(), persistent=False)

    def forward(self, query: Tensor, key: Tensor) -> tuple[Tensor, Tensor]:
        sequence_length = query.shape[-2]
        cosine = self.cosine[:sequence_length].to(query.dtype)[None, None, :, :]
        sine = self.sine[:sequence_length].to(query.dtype)[None, None, :, :]
        return _apply_rotary(query, cosine, sine), _apply_rotary(key, cosine, sine)


def _apply_rotary(value: Tensor, cosine: Tensor, sine: Tensor) -> Tensor:
    """Rotate adjacent head-dimension pairs."""
    even = value[..., 0::2]
    odd = value[..., 1::2]
    rotated_even = even * cosine - odd * sine
    rotated_odd = even * sine + odd * cosine
    return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)


def _headwise_rms(value: Tensor, epsilon: float) -> Tensor:
    """Normalize query or key vectors without learned parameters."""
    return value * torch.rsqrt(value.float().pow(2).mean(dim=-1, keepdim=True) + epsilon).to(value.dtype)


class SelfAttention(nn.Module):
    """Compute exact non-causal multi-head attention through PyTorch SDPA."""

    def __init__(self, config: AntibodyModelConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dimension = config.d_model // config.n_heads
        self.epsilon = config.normalization_epsilon
        self.qk_normalization = config.qk_normalization
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.output = nn.Linear(config.d_model, config.d_model, bias=False)
        self.rotary = RotaryEmbedding(
            self.head_dimension,
            config.maximum_sequence_tokens,
            config.rope_theta,
        )

    def forward(self, hidden: Tensor, attention_mask: Tensor | None) -> Tensor:
        batch_size, sequence_length, dimension = hidden.shape
        query, key, value = self.qkv(hidden).chunk(3, dim=-1)
        shape = (batch_size, sequence_length, self.n_heads, self.head_dimension)
        query = query.view(shape).transpose(1, 2)
        key = key.view(shape).transpose(1, 2)
        value = value.view(shape).transpose(1, 2)
        if self.qk_normalization:
            query = _headwise_rms(query, self.epsilon)
            key = _headwise_rms(key, self.epsilon)
        query, key = self.rotary(query, key)
        sdpa_mask = None
        if attention_mask is not None:
            if attention_mask.shape != (batch_size, sequence_length):
                raise ValueError("attention_mask shape differs from input")
            sdpa_mask = attention_mask[:, None, None, :].to(torch.bool)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=sdpa_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        return self.output(attended.transpose(1, 2).contiguous().view_as(hidden))


class SwiGLU(nn.Module):
    """Apply a fused gate/up projection followed by a down projection."""

    def __init__(self, config: AntibodyModelConfig) -> None:
        super().__init__()
        self.gate_up = nn.Linear(
            config.d_model,
            2 * config.ffn_hidden_size,
            bias=False,
        )
        self.down = nn.Linear(config.ffn_hidden_size, config.d_model, bias=False)

    def forward(self, hidden: Tensor) -> Tensor:
        gate, value = self.gate_up(hidden).chunk(2, dim=-1)
        return self.down(F.silu(gate) * value)


class TransformerBlock(nn.Module):
    """Apply pre-normalized attention and SwiGLU residual branches."""

    def __init__(self, config: AntibodyModelConfig) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.d_model, config.normalization_epsilon)
        self.attention = SelfAttention(config)
        self.ffn_norm = RMSNorm(config.d_model, config.normalization_epsilon)
        self.ffn = SwiGLU(config)

    def forward(self, hidden: Tensor, attention_mask: Tensor | None) -> Tensor:
        hidden = hidden + self.attention(self.attention_norm(hidden), attention_mask)
        return hidden + self.ffn(self.ffn_norm(hidden))


class AntibodyMaskedLanguageModel(nn.Module):
    """Encode antibody residues and score tied-vocabulary MLM targets."""

    def __init__(self, config: AntibodyModelConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocabulary_size, config.d_model)
        self.type_embedding = nn.Embedding(config.chain_type_count, config.d_model)
        self.blocks = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.n_layers)
        )
        self.final_norm = RMSNorm(config.d_model, config.normalization_epsilon)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize embeddings and projections with depth-scaled residuals."""
        base_standard_deviation = 0.02
        residual_standard_deviation = base_standard_deviation / math.sqrt(
            2 * self.config.n_layers
        )
        nn.init.normal_(self.token_embedding.weight, std=base_standard_deviation)
        nn.init.normal_(self.type_embedding.weight, std=base_standard_deviation)
        for block in self.blocks:
            nn.init.normal_(block.attention.qkv.weight, std=base_standard_deviation)
            nn.init.normal_(
                block.attention.output.weight,
                std=residual_standard_deviation,
            )
            nn.init.normal_(block.ffn.gate_up.weight, std=base_standard_deviation)
            nn.init.normal_(block.ffn.down.weight, std=residual_standard_deviation)

    def forward(
        self,
        input_ids: Tensor,
        type_ids: Tensor,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
    ) -> MLMOutput:
        if input_ids.shape != type_ids.shape or input_ids.ndim != 2:
            raise ValueError("input_ids and type_ids must be equal rank-two tensors")
        if input_ids.shape[1] > self.config.maximum_sequence_tokens:
            raise ValueError("Input exceeds the maximum sequence-token boundary")
        hidden = self.token_embedding(input_ids) + self.type_embedding(type_ids)
        for block in self.blocks:
            hidden = block(hidden, attention_mask)
        hidden = self.final_norm(hidden)
        logits = F.linear(hidden, self.token_embedding.weight)
        target_count = torch.zeros((), dtype=torch.long, device=input_ids.device)
        loss_sum = None
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError("labels shape differs from input_ids")
            selected = labels.ne(-100)
            target_count = selected.sum()
            if not bool(target_count):
                loss_sum = logits.sum() * 0.0
            else:
                loss_sum = F.cross_entropy(
                    logits.flatten(0, 1),
                    labels.flatten(),
                    ignore_index=-100,
                    reduction="sum",
                )
        return MLMOutput(logits=logits, loss_sum=loss_sum, target_count=target_count)


def load_model_family(path: Path) -> dict[str, Any]:
    """Load the frozen ASCII model-family configuration."""
    payload = path.read_bytes()
    try:
        family = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Model-family configuration must be ASCII JSON") from error
    if family.get("format_version") != "antibody_model_family_v1":
        raise ValueError("Unexpected model-family configuration version")
    return family


def config_from_preset(family: dict[str, Any], preset: str) -> AntibodyModelConfig:
    """Construct one model config from a frozen named preset."""
    if preset not in family["presets"]:
        raise ValueError(f"Unknown model preset: {preset}")
    values = dict(family["presets"][preset])
    values.update(
        {
            "maximum_sequence_tokens": family["maximum_sequence_tokens"],
            "qk_normalization": family["architecture"]["qk_normalization"],
            "rope_theta": family["architecture"]["rope_theta"],
        }
    )
    return AntibodyModelConfig(**values)


def parameter_receipt(
    model: AntibodyMaskedLanguageModel,
    *,
    preset: str,
    family_file_sha256: str,
) -> dict[str, Any]:
    """Return exact unique parameter counts and an architecture digest."""
    named = list(model.named_parameters())
    unique_parameter_ids = {id(parameter) for _, parameter in named}
    if len(unique_parameter_ids) != len(named):
        raise ValueError("named_parameters unexpectedly contains duplicate tensors")
    by_module = {
        "embeddings": sum(
            parameter.numel()
            for name, parameter in named
            if name.startswith(("token_embedding.", "type_embedding."))
        ),
        "transformer_blocks": sum(
            parameter.numel() for name, parameter in named if name.startswith("blocks.")
        ),
        "final_norm": sum(
            parameter.numel()
            for name, parameter in named
            if name.startswith("final_norm.")
        ),
    }
    receipt = {
        "architecture": asdict(model.config),
        "family_file_sha256": family_file_sha256,
        "format_version": "antibody_model_parameter_receipt_v1",
        "parameter_count_by_module": by_module,
        "preset": preset,
        "test_access": {
            "downstream_labels_used": False,
            "sabdab2_heldout_accessed": False,
        },
        "total_parameter_count": sum(parameter.numel() for _, parameter in named),
        "trainable_parameter_count": sum(
            parameter.numel() for _, parameter in named if parameter.requires_grad
        ),
    }
    canonical = (
        json.dumps(receipt, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("ascii")
    receipt["receipt_sha256"] = hashlib.sha256(canonical).hexdigest()
    return receipt
