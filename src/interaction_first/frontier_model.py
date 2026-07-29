"""Prospective D0/R1/R2 models for leakage-controlled partner retrieval."""

from __future__ import annotations

import math
from typing import Any

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError:
    torch = None
    nn = None
    F = None

from .asymmetric_retrieval import AsymmetricPartnerRetrievalModel


_ModuleBase = nn.Module if nn is not None else object


def require_torch() -> None:
    """Fail with a focused message when PyTorch is unavailable."""
    if torch is None or nn is None or F is None:
        raise RuntimeError("PyTorch is required for frontier model operations")


def _validate_residue_batch(
    features: Any,
    mask: Any,
    expected_dim: int,
    name: str,
) -> None:
    if not torch.is_tensor(features) or features.ndim != 3:
        raise ValueError(
            f"{name} features must have shape [batch, residues, channels]"
        )
    if not torch.is_floating_point(features):
        raise ValueError(f"{name} features must use a floating-point dtype")
    if int(features.shape[-1]) != expected_dim:
        raise ValueError(
            f"{name} features must have dimension {expected_dim}, "
            f"received {int(features.shape[-1])}"
        )
    if any(int(size) == 0 for size in features.shape):
        raise ValueError(f"{name} features cannot contain an empty dimension")
    if not torch.is_tensor(mask) or mask.dtype != torch.bool or mask.ndim != 2:
        raise ValueError(f"{name} mask must be a two-dimensional boolean tensor")
    if tuple(mask.shape) != tuple(features.shape[:2]):
        raise ValueError(f"{name} mask must align with the residue tensor")
    if mask.device != features.device:
        raise ValueError(f"{name} mask and features must use the same device")
    if bool((mask.sum(dim=1) == 0).any()):
        raise ValueError(f"Every {name} item must contain a valid residue")
    valid = features.masked_select(mask.unsqueeze(-1))
    if not bool(torch.isfinite(valid).all()):
        raise ValueError(f"Valid {name} features must be finite")


def _masked_mean(values: Any, mask: Any) -> Any:
    weights = mask.unsqueeze(-1).to(dtype=values.dtype)
    return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


class FixedTemperature(_ModuleBase):
    """Expose a protocol-locked positive temperature as immutable model state."""

    def __init__(self, value: float = 0.07) -> None:
        require_torch()
        super().__init__()
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError("Fixed temperature must be finite and positive")
        self.register_buffer(
            "value",
            torch.tensor(float(value), dtype=torch.float32),
            persistent=True,
        )

    def forward(self) -> Any:
        """Return the immutable scalar temperature."""
        return self.value


class ProtocolD0Retriever(AsymmetricPartnerRetrievalModel):
    """Locked architecture for the protocol-defined coarse dual encoder."""

    def __init__(
        self,
        antibody_input_dim: int = 512,
        antigen_input_dim: int = 1280,
        embedding_dim: int = 128,
        projection_hidden_dim: int = 256,
        dropout: float = 0.0,
    ) -> None:
        require_torch()
        super().__init__(
            antibody_input_dim=antibody_input_dim,
            antigen_input_dim=antigen_input_dim,
            embedding_dim=embedding_dim,
            antibody_hidden_dim=projection_hidden_dim,
            antigen_hidden_dim=projection_hidden_dim,
            dropout=dropout,
            chain_balanced_antibody_pooling=False,
            antibody_pooling="mean",
            antigen_pooling="mean",
            initial_temperature=0.07,
            minimum_temperature=0.01,
            maximum_temperature=1.0,
        )
        self.antibody_chain_type = nn.Embedding(2, antibody_input_dim)
        self.temperature_head = FixedTemperature(0.07)

    def encode_antibody(
        self,
        features: Any,
        mask: Any,
        chain_ids: Any,
    ) -> tuple[Any, Any]:
        """Add learned heavy/light type vectors before projected length pooling."""
        _validate_residue_batch(
            features,
            mask,
            self.antibody_pooler.input_dim,
            "antibody",
        )
        if not torch.is_tensor(chain_ids) or chain_ids.dtype != torch.long:
            raise ValueError("antibody chain IDs must be an int64 tensor")
        if tuple(chain_ids.shape) != tuple(mask.shape) or chain_ids.device != mask.device:
            raise ValueError("antibody chain IDs must align with the residue mask")
        if bool(((chain_ids[mask] < 0) | (chain_ids[mask] > 1)).any()):
            raise ValueError("Valid antibody residues must use chain ID 0 or 1")
        if bool((chain_ids[~mask] != -1).any()):
            raise ValueError("Padded antibody residues must use chain ID -1")
        safe_chain_ids = chain_ids.clamp_min(0)
        typed_features = features + self.antibody_chain_type(safe_chain_ids).to(
            dtype=features.dtype
        )
        typed_features = typed_features.masked_fill(~mask.unsqueeze(-1), 0.0)
        return self.antibody_pooler(typed_features, mask, None)

    def paired_scores(
        self,
        antibody_features: Any,
        antigen_features: Any,
        antibody_mask: Any,
        antigen_mask: Any,
        antibody_chain_ids: Any,
    ) -> dict[str, Any]:
        """Return aligned-pair scores without constructing a quadratic matrix."""
        antibody_embeddings, antibody_weights = self.encode_antibody(
            antibody_features,
            antibody_mask,
            antibody_chain_ids,
        )
        antigen_embeddings, antigen_weights = self.encode_antigen(
            antigen_features,
            antigen_mask,
        )
        if int(antibody_embeddings.shape[0]) != int(antigen_embeddings.shape[0]):
            raise ValueError("Paired scoring requires equal antibody and antigen batches")
        scores = (antibody_embeddings * antigen_embeddings).sum(dim=-1)
        return {
            "antibody_embeddings": antibody_embeddings,
            "antigen_embeddings": antigen_embeddings,
            "antibody_weights": antibody_weights,
            "antigen_weights": antigen_weights,
            "coarse_score": scores,
            "temperature": self.temperature_head(),
        }


class PreNormBidirectionalCrossAttention(_ModuleBase):
    """Apply independent two-way cross-attention and pre-normalized feed-forward updates."""

    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        ffn_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        require_torch()
        super().__init__()
        if d_model <= 0 or num_heads <= 0 or d_model % num_heads != 0:
            raise ValueError("d_model must be positive and divisible by num_heads")
        if ffn_dim <= 0:
            raise ValueError("ffn_dim must be positive")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.antibody_attention_norm = nn.LayerNorm(d_model)
        self.antigen_attention_norm = nn.LayerNorm(d_model)
        self.antibody_to_antigen = nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.antigen_to_antibody = nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.antibody_ffn_norm = nn.LayerNorm(d_model)
        self.antigen_ffn_norm = nn.LayerNorm(d_model)
        self.antibody_ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )
        self.antigen_ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )
        self.residual_dropout = nn.Dropout(dropout)

    def forward(
        self,
        antibody: Any,
        antigen: Any,
        antibody_mask: Any,
        antigen_mask: Any,
    ) -> tuple[Any, Any]:
        """Update both residue streams while excluding all padded keys and outputs."""
        antibody_normalized = self.antibody_attention_norm(antibody)
        antigen_normalized = self.antigen_attention_norm(antigen)
        antibody_context, _ = self.antibody_to_antigen(
            antibody_normalized,
            antigen_normalized,
            antigen_normalized,
            key_padding_mask=~antigen_mask,
            need_weights=False,
        )
        antigen_context, _ = self.antigen_to_antibody(
            antigen_normalized,
            antibody_normalized,
            antibody_normalized,
            key_padding_mask=~antibody_mask,
            need_weights=False,
        )
        antibody = antibody + self.residual_dropout(antibody_context)
        antigen = antigen + self.residual_dropout(antigen_context)
        antibody = antibody + self.residual_dropout(
            self.antibody_ffn(self.antibody_ffn_norm(antibody))
        )
        antigen = antigen + self.residual_dropout(
            self.antigen_ffn(self.antigen_ffn_norm(antigen))
        )
        antibody = antibody.masked_fill(~antibody_mask.unsqueeze(-1), 0.0)
        antigen = antigen.masked_fill(~antigen_mask.unsqueeze(-1), 0.0)
        return antibody, antigen


class NativeContactHead(_ModuleBase):
    """Decode independent sigmoid logits in antigen-by-antibody orientation."""

    def __init__(
        self,
        d_model: int = 256,
        pair_dim: int = 64,
        initial_bias: float = -4.0,
    ) -> None:
        require_torch()
        super().__init__()
        if d_model <= 0 or pair_dim <= 0:
            raise ValueError("Contact dimensions must be positive")
        if not math.isfinite(float(initial_bias)):
            raise ValueError("initial_bias must be finite")
        self.antibody_projection = nn.Linear(d_model, pair_dim, bias=False)
        self.antigen_projection = nn.Linear(d_model, pair_dim, bias=False)
        self.bias = nn.Parameter(torch.tensor(float(initial_bias)))
        self.scale = pair_dim**-0.5

    def forward(
        self,
        antibody: Any,
        antigen: Any,
        antibody_mask: Any,
        antigen_mask: Any,
    ) -> tuple[Any, Any]:
        """Return logits and a boolean mask in antigen-by-antibody orientation."""
        logits = self.scale * torch.matmul(
            self.antigen_projection(antigen),
            self.antibody_projection(antibody).transpose(-2, -1),
        )
        logits = logits + self.bias
        pair_mask = antigen_mask.unsqueeze(-1) & antibody_mask.unsqueeze(-2)
        logits = logits.masked_fill(~pair_mask, 0.0)
        return logits, pair_mask


class ProtocolReranker(_ModuleBase):
    """Implement protocol R1 and optional R2 contact supervision over a frozen D0."""

    def __init__(
        self,
        coarse_retriever: ProtocolD0Retriever,
        antibody_input_dim: int = 512,
        antigen_input_dim: int = 1280,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        ffn_dim: int = 512,
        dropout: float = 0.1,
        contact_pair_dim: int | None = None,
        contact_initial_bias: float = -4.0,
    ) -> None:
        require_torch()
        super().__init__()
        if not isinstance(coarse_retriever, ProtocolD0Retriever):
            raise TypeError("coarse_retriever must be a ProtocolD0Retriever")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        self.antibody_input_dim = int(antibody_input_dim)
        self.antigen_input_dim = int(antigen_input_dim)
        if coarse_retriever.antibody_pooler.input_dim != self.antibody_input_dim:
            raise ValueError("Coarse and reranker antibody input dimensions must match")
        if coarse_retriever.antigen_pooler.input_dim != self.antigen_input_dim:
            raise ValueError("Coarse and reranker antigen input dimensions must match")
        self.d_model = int(d_model)
        self.coarse_retriever = coarse_retriever
        self.coarse_retriever.requires_grad_(False)
        self.coarse_retriever.eval()
        self.antibody_projection = nn.Sequential(
            nn.LayerNorm(self.antibody_input_dim),
            nn.Linear(self.antibody_input_dim, self.d_model),
        )
        self.antigen_projection = nn.Sequential(
            nn.LayerNorm(self.antigen_input_dim),
            nn.Linear(self.antigen_input_dim, self.d_model),
        )
        self.blocks = nn.ModuleList(
            PreNormBidirectionalCrossAttention(
                d_model=self.d_model,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        )
        self.antibody_final_norm = nn.LayerNorm(self.d_model)
        self.antigen_final_norm = nn.LayerNorm(self.d_model)
        self.rerank_head = nn.Sequential(
            nn.Linear(4 * self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, 1),
        )
        self.contact_head = (
            None
            if contact_pair_dim is None
            else NativeContactHead(
                d_model=self.d_model,
                pair_dim=contact_pair_dim,
                initial_bias=contact_initial_bias,
            )
        )

    @property
    def stage(self) -> str:
        """Return the protocol candidate represented by this module."""
        return "R1" if self.contact_head is None else "R2"

    def train(self, mode: bool = True) -> "ProtocolReranker":
        """Keep the reused D0 checkpoint in evaluation mode during reranker fitting."""
        super().train(mode)
        self.coarse_retriever.eval()
        return self

    def forward(
        self,
        antibody_features: Any,
        antigen_features: Any,
        antibody_mask: Any,
        antigen_mask: Any,
        antibody_chain_ids: Any,
    ) -> dict[str, Any]:
        """Return coarse, residual, final retrieval, and optional contact outputs."""
        _validate_residue_batch(
            antibody_features,
            antibody_mask,
            self.antibody_input_dim,
            "antibody",
        )
        _validate_residue_batch(
            antigen_features,
            antigen_mask,
            self.antigen_input_dim,
            "antigen",
        )
        if int(antibody_features.shape[0]) != int(antigen_features.shape[0]):
            raise ValueError("Reranking requires aligned antibody-antigen pairs")
        with torch.no_grad():
            coarse = self.coarse_retriever.paired_scores(
                antibody_features,
                antigen_features,
                antibody_mask,
                antigen_mask,
                antibody_chain_ids,
            )
        antibody = self.antibody_projection(
            antibody_features.masked_fill(~antibody_mask.unsqueeze(-1), 0.0)
        )
        antigen = self.antigen_projection(
            antigen_features.masked_fill(~antigen_mask.unsqueeze(-1), 0.0)
        )
        antibody = antibody.masked_fill(~antibody_mask.unsqueeze(-1), 0.0)
        antigen = antigen.masked_fill(~antigen_mask.unsqueeze(-1), 0.0)
        for block in self.blocks:
            antibody, antigen = block(
                antibody,
                antigen,
                antibody_mask,
                antigen_mask,
            )
        antibody = self.antibody_final_norm(antibody)
        antigen = self.antigen_final_norm(antigen)
        antibody = antibody.masked_fill(~antibody_mask.unsqueeze(-1), 0.0)
        antigen = antigen.masked_fill(~antigen_mask.unsqueeze(-1), 0.0)
        antibody_mean = _masked_mean(antibody, antibody_mask)
        antigen_mean = _masked_mean(antigen, antigen_mask)
        joint = torch.cat(
            (
                antibody_mean,
                antigen_mean,
                antibody_mean * antigen_mean,
                torch.abs(antibody_mean - antigen_mean),
            ),
            dim=-1,
        )
        residual = self.rerank_head(joint).squeeze(-1)
        final_score = coarse["coarse_score"] + 0.5 * torch.tanh(residual)
        result = {
            **coarse,
            "stage": self.stage,
            "rerank_residual": residual,
            "retrieval_score": final_score,
            "antibody_residues": antibody,
            "antigen_residues": antigen,
        }
        if self.contact_head is not None:
            contact_logits, pair_mask = self.contact_head(
                antibody,
                antigen,
                antibody_mask,
                antigen_mask,
            )
            result["contact_logits"] = contact_logits
            result["contact_probabilities"] = torch.sigmoid(contact_logits).masked_fill(
                ~pair_mask,
                0.0,
            )
            result["contact_pair_mask"] = pair_mask
        return result

    def score_indexed_pairs(
        self,
        antibody_features: Any,
        antigen_features: Any,
        antibody_mask: Any,
        antigen_mask: Any,
        antibody_chain_ids: Any,
        pair_indices: Any,
    ) -> dict[str, Any]:
        """Score an explicit list of library edges without labeling absent pairs."""
        if not torch.is_tensor(pair_indices) or pair_indices.dtype != torch.long:
            raise ValueError("pair_indices must be an int64 tensor")
        if pair_indices.ndim != 2 or int(pair_indices.shape[0]) != 2:
            raise ValueError("pair_indices must have shape [2, pair_count]")
        if int(pair_indices.shape[1]) == 0:
            raise ValueError("pair_indices cannot be empty")
        if pair_indices.device != antibody_features.device:
            raise ValueError("pair_indices and model inputs must use the same device")
        if antibody_features.device != antigen_features.device:
            raise ValueError("Antibody and antigen libraries must use the same device")
        antibody_indices = pair_indices[0]
        antigen_indices = pair_indices[1]
        if bool((antibody_indices < 0).any()) or bool(
            (antibody_indices >= int(antibody_features.shape[0])).any()
        ):
            raise IndexError("pair_indices contains an out-of-range antibody index")
        if bool((antigen_indices < 0).any()) or bool(
            (antigen_indices >= int(antigen_features.shape[0])).any()
        ):
            raise IndexError("pair_indices contains an out-of-range antigen index")
        return self.forward(
            antibody_features.index_select(0, antibody_indices),
            antigen_features.index_select(0, antigen_indices),
            antibody_mask.index_select(0, antibody_indices),
            antigen_mask.index_select(0, antigen_indices),
            antibody_chain_ids.index_select(0, antibody_indices),
        )


def balanced_contact_bce(
    logits: Any,
    targets: Any,
    resolved_mask: Any,
    *,
    return_details: bool = False,
) -> Any:
    """Average per-complex class-balanced BCE over structurally resolved pairs."""
    require_torch()
    if not torch.is_tensor(logits) or logits.ndim != 3:
        raise ValueError("logits must have shape [batch, antigen, antibody]")
    if not torch.is_floating_point(logits) or not bool(torch.isfinite(logits).all()):
        raise ValueError("logits must be finite floating-point values")
    if not torch.is_tensor(targets) or tuple(targets.shape) != tuple(logits.shape):
        raise ValueError("targets must align with logits")
    if (
        targets.layout != torch.strided
        or not torch.is_floating_point(targets)
        or targets.is_complex()
    ):
        raise ValueError("targets must use a floating-point dtype")
    if not torch.is_tensor(resolved_mask) or resolved_mask.dtype != torch.bool:
        raise ValueError("resolved_mask must be boolean")
    if tuple(resolved_mask.shape) != tuple(logits.shape):
        raise ValueError("resolved_mask must align with logits")
    if logits.device != targets.device or logits.device != resolved_mask.device:
        raise ValueError("Contact tensors must use the same device")
    if not bool(torch.isfinite(targets).all()):
        raise ValueError("Contact targets must be finite")
    if bool(((targets != 0.0) & (targets != 1.0)).any()):
        raise ValueError("Contact targets must be exactly binary")
    if bool((~resolved_mask & (targets != 0.0)).any()):
        raise ValueError("Unresolved contact targets must use zero placeholders")
    elementwise = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
    )
    positive = resolved_mask & (targets == 1.0)
    negative = resolved_mask & (targets == 0.0)
    reduction_axes = tuple(range(1, logits.ndim))
    positive_count = positive.sum(dim=reduction_axes)
    negative_count = negative.sum(dim=reduction_axes)
    valid_complex = (positive_count > 0) & (negative_count > 0)
    positive_mean = (
        (elementwise * positive.to(dtype=elementwise.dtype)).sum(
            dim=reduction_axes
        )
        / positive_count.clamp_min(1).to(dtype=elementwise.dtype)
    )
    negative_mean = (
        (elementwise * negative.to(dtype=elementwise.dtype)).sum(
            dim=reduction_axes
        )
        / negative_count.clamp_min(1).to(dtype=elementwise.dtype)
    )
    per_complex = 0.5 * (positive_mean + negative_mean)
    valid_complex_count = int(valid_complex.sum().item())
    if valid_complex_count == 0:
        loss_sum = logits.sum() * 0.0
        loss = loss_sum
    else:
        loss_sum = per_complex[valid_complex].sum()
        loss = loss_sum / valid_complex_count
    if return_details:
        return {
            "loss": loss,
            "loss_sum": loss_sum,
            "valid_complex_count": valid_complex_count,
            "observed_complex_count": int(logits.shape[0]),
            "reduction": "equal_per_dual_class_complex",
        }
    return loss
