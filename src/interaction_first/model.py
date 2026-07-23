"""Partner-conditioned interaction-map model built on residue embeddings."""

from __future__ import annotations

import math
from typing import Any, Sequence

from .scoring import normalized_partner_score, pool_pair_marginals

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:
    torch = None
    nn = None


_ModuleBase = nn.Module if nn is not None else object


def require_torch() -> None:
    """Fail with a focused message when model code is used without PyTorch."""
    if torch is None or nn is None:
        raise RuntimeError("PyTorch is required for interaction-first model operations")


def _validate_mask(mask: Any, features: Any, name: str) -> None:
    if mask.dtype != torch.bool or tuple(mask.shape) != tuple(features.shape[:2]):
        raise ValueError(f"{name} must be a boolean tensor aligned to the residue axes")
    if torch.any(mask.sum(dim=1) == 0):
        raise ValueError(f"Every item must contain at least one valid {name} residue")


def merge_chunk_embeddings(
    chunks: Sequence[Any],
    starts: Sequence[int],
    full_length: int,
    device: Any,
) -> Any:
    """Average overlapping frozen-ESM chunks into one full residue embedding."""
    require_torch()
    if not chunks or len(chunks) != len(starts):
        raise ValueError("Embedding chunks and starts must be non-empty and aligned")
    if full_length <= 0:
        raise ValueError("full_length must be positive")
    embedding_dim = int(chunks[0].shape[-1])
    output = torch.zeros(full_length, embedding_dim, dtype=torch.float32, device=device)
    counts = torch.zeros(full_length, 1, dtype=torch.float32, device=device)
    for chunk, raw_start in zip(chunks, starts):
        if chunk.ndim != 2 or int(chunk.shape[-1]) != embedding_dim:
            raise ValueError("Embedding chunks must share a two-dimensional shape")
        start = int(raw_start)
        end = start + int(chunk.shape[0])
        if start < 0 or end > full_length:
            raise ValueError(f"Embedding interval [{start}, {end}) exceeds the sequence")
        output[start:end] += chunk.to(device=device, dtype=torch.float32)
        counts[start:end] += 1.0
    if torch.any(counts == 0):
        raise ValueError("Embedding chunks do not cover the complete sequence")
    return output / counts


def _masked_mean(values: Any, mask: Any, dimension: int) -> Any:
    weights = mask.to(dtype=values.dtype)
    denominator = weights.sum(dim=dimension).clamp_min(1.0)
    return (values * weights).sum(dim=dimension) / denominator


def _masked_topk_mean(values: Any, mask: Any, top_k: int) -> Any:
    """Return a per-batch top-k mean without allowing padded values to enter."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    flattened_values = values.reshape(values.shape[0], -1)
    flattened_mask = mask.reshape(mask.shape[0], -1)
    scores = []
    for batch_index in range(values.shape[0]):
        valid = flattened_values[batch_index][flattened_mask[batch_index]]
        if valid.numel() == 0:
            raise ValueError("Every item must contain at least one valid residue pair")
        k = min(top_k, int(valid.numel()))
        scores.append(torch.topk(valid, k=k, sorted=False).values.mean())
    return torch.stack(scores)


def _masked_topk_reduce(values: Any, mask: Any, dimension: int, top_k: int) -> Any:
    """Reduce one residue-pair axis while returning zero for padded residues."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    k = min(top_k, int(values.shape[dimension]))
    masked_values = values.masked_fill(~mask, -torch.inf)
    top_values, top_indices = torch.topk(masked_values, k=k, dim=dimension, sorted=False)
    selected_mask = torch.gather(mask, dimension, top_indices)
    finite_values = torch.where(selected_mask, top_values, torch.zeros_like(top_values))
    denominator = selected_mask.sum(dim=dimension).clamp_min(1).to(dtype=values.dtype)
    return finite_values.sum(dim=dimension) / denominator


class BidirectionalCrossAttentionBlock(_ModuleBase):
    """Interleave two-way partner communication with residue-wise feed-forward layers."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float,
        expansion: int = 4,
        gate_initial_logit: float = -2.0,
    ) -> None:
        require_torch()
        super().__init__()
        if d_model <= 0 or d_model % num_heads != 0:
            raise ValueError("d_model must be positive and divisible by num_heads")
        self.ag_norm = nn.LayerNorm(d_model)
        self.ab_norm = nn.LayerNorm(d_model)
        self.ag_attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.ab_attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        if not math.isfinite(float(gate_initial_logit)):
            raise ValueError("gate_initial_logit must be finite")
        self.ag_gate = nn.Parameter(torch.tensor(float(gate_initial_logit)))
        self.ab_gate = nn.Parameter(torch.tensor(float(gate_initial_logit)))
        hidden = expansion * d_model
        self.ag_ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
        )
        self.ab_ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        antigen: Any,
        antibody: Any,
        antigen_mask: Any,
        antibody_mask: Any,
    ) -> tuple[Any, Any]:
        """Update both residue streams using only valid residues from the partner."""
        ag_query = self.ag_norm(antigen)
        ab_query = self.ab_norm(antibody)
        ag_context, _ = self.ag_attention(
            ag_query,
            ab_query,
            ab_query,
            key_padding_mask=~antibody_mask,
            need_weights=False,
        )
        ab_context, _ = self.ab_attention(
            ab_query,
            ag_query,
            ag_query,
            key_padding_mask=~antigen_mask,
            need_weights=False,
        )
        antigen = antigen + torch.sigmoid(self.ag_gate) * self.dropout(ag_context)
        antibody = antibody + torch.sigmoid(self.ab_gate) * self.dropout(ab_context)
        antigen = antigen + self.dropout(self.ag_ffn(antigen))
        antibody = antibody + self.dropout(self.ab_ffn(antibody))
        antigen = antigen.masked_fill(~antigen_mask.unsqueeze(-1), 0.0)
        antibody = antibody.masked_fill(~antibody_mask.unsqueeze(-1), 0.0)
        return antigen, antibody


class BidirectionalBilinearDecoder(_ModuleBase):
    """Produce supervised pair logits from two directional low-rank affinities."""

    def __init__(self, d_model: int, pair_dim: int) -> None:
        require_torch()
        super().__init__()
        if d_model <= 0 or pair_dim <= 0:
            raise ValueError("Decoder dimensions must be positive")
        self.ag_query = nn.Linear(d_model, pair_dim, bias=False)
        self.ab_key = nn.Linear(d_model, pair_dim, bias=False)
        self.ab_query = nn.Linear(d_model, pair_dim, bias=False)
        self.ag_key = nn.Linear(d_model, pair_dim, bias=False)
        self.mixing_logits = nn.Parameter(torch.zeros(2))
        self.ag_bias = nn.Linear(d_model, 1)
        self.ab_bias = nn.Linear(d_model, 1)
        self.global_bias = nn.Parameter(torch.tensor(-4.0))
        self.scale = pair_dim**-0.5

    def components(self, antigen: Any, antibody: Any) -> dict[str, Any]:
        """Return pair-specific energy and separately identifiable unary terms."""
        ag_to_ab = torch.matmul(
            self.ag_query(antigen), self.ab_key(antibody).transpose(-2, -1)
        )
        ab_to_ag = torch.matmul(
            self.ag_key(antigen), self.ab_query(antibody).transpose(-2, -1)
        )
        weights = torch.softmax(self.mixing_logits, dim=0)
        interaction_logits = self.scale * (
            weights[0] * ag_to_ab + weights[1] * ab_to_ag
        )
        antigen_bias = self.ag_bias(antigen)
        antibody_bias = self.ab_bias(antibody).transpose(-2, -1)
        pair_logits = interaction_logits + antigen_bias + antibody_bias + self.global_bias
        return {
            "pair_logits": pair_logits,
            "interaction_logits": interaction_logits,
            "antigen_bias": antigen_bias,
            "antibody_bias": antibody_bias,
            "global_bias": self.global_bias,
        }

    def forward(self, antigen: Any, antibody: Any) -> Any:
        """Return calibrated-task logits with shape ``[batch, antigen, antibody]``."""
        return self.components(antigen, antibody)["pair_logits"]


class PartnerSpecificInteractionModel(_ModuleBase):
    """Predict a contact map whose marginals are antibody- and antigen-conditioned."""

    def __init__(
        self,
        antigen_input_dim: int = 1280,
        antibody_input_dim: int = 1280,
        d_model: int = 256,
        pair_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.10,
        marginal_pooling: str = "logmeanexp",
        marginal_top_k: int = 3,
        marginal_temperature: float = 1.0,
        score_top_k: int = 16,
        score_pooling: str = "logmeanexp",
        score_temperature: float = 1.0,
        score_top_fraction: float = 0.01,
        center_partner_score: bool = True,
        cross_attention_gate_initial_logit: float = -2.0,
    ) -> None:
        require_torch()
        super().__init__()
        if num_layers < 0:
            raise ValueError("num_layers cannot be negative")
        if marginal_pooling not in {"mean", "topk", "noisy_or", "logmeanexp"}:
            raise ValueError(
                "marginal_pooling must be 'mean', 'topk', 'noisy_or', or 'logmeanexp'"
            )
        if score_pooling not in {"logmeanexp", "top_fraction"}:
            raise ValueError("score_pooling must be 'logmeanexp' or 'top_fraction'")
        if marginal_temperature <= 0.0 or score_temperature <= 0.0:
            raise ValueError("Pooling temperatures must be positive")
        if not 0.0 < score_top_fraction <= 1.0:
            raise ValueError("score_top_fraction must be in (0, 1]")
        self.antigen_projection = nn.Sequential(
            nn.LayerNorm(antigen_input_dim),
            nn.Linear(antigen_input_dim, d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.antibody_projection = nn.Sequential(
            nn.LayerNorm(antibody_input_dim),
            nn.Linear(antibody_input_dim, d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList(
            BidirectionalCrossAttentionBlock(
                d_model,
                num_heads,
                dropout,
                gate_initial_logit=cross_attention_gate_initial_logit,
            )
            for _ in range(num_layers)
        )
        self.final_antigen_norm = nn.LayerNorm(d_model)
        self.final_antibody_norm = nn.LayerNorm(d_model)
        self.decoder = BidirectionalBilinearDecoder(d_model, pair_dim)
        self.epitope_head = nn.Linear(d_model, 1)
        self.paratope_head = nn.Linear(d_model, 1)
        nn.init.constant_(self.epitope_head.bias, -2.5)
        nn.init.constant_(self.paratope_head.bias, -2.5)
        self.marginal_pooling = marginal_pooling
        self.marginal_top_k = int(marginal_top_k)
        self.marginal_temperature = float(marginal_temperature)
        self.score_top_k = int(score_top_k)
        self.score_pooling = score_pooling
        self.score_temperature = float(score_temperature)
        self.score_top_fraction = float(score_top_fraction)
        self.center_partner_score = bool(center_partner_score)

    def _marginals(
        self,
        pair_logits: Any,
        pair_probabilities: Any,
        pair_mask: Any,
    ) -> tuple[Any, Any]:
        if self.marginal_pooling == "mean":
            epitope = _masked_mean(pair_probabilities, pair_mask, dimension=2)
            paratope = _masked_mean(pair_probabilities, pair_mask, dimension=1)
            return epitope, paratope
        return pool_pair_marginals(
            pair_logits,
            pair_mask,
            pooling=self.marginal_pooling,
            top_k=self.marginal_top_k,
            temperature=self.marginal_temperature,
        )

    def forward(
        self,
        antigen_features: Any,
        antibody_features: Any,
        antigen_mask: Any | None = None,
        antibody_mask: Any | None = None,
        candidate_pair_mask: Any | None = None,
    ) -> dict[str, Any]:
        """Encode a batch and return pair logits, marginals, and a ranking score."""
        if antigen_features.ndim != 3 or antibody_features.ndim != 3:
            raise ValueError("Residue features must have shape [batch, length, channels]")
        if antigen_features.shape[0] != antibody_features.shape[0]:
            raise ValueError("Antigen and antibody batches have different sizes")
        if antigen_mask is None:
            antigen_mask = torch.ones(
                antigen_features.shape[:2], dtype=torch.bool, device=antigen_features.device
            )
        if antibody_mask is None:
            antibody_mask = torch.ones(
                antibody_features.shape[:2], dtype=torch.bool, device=antibody_features.device
            )
        _validate_mask(antigen_mask, antigen_features, "antigen_mask")
        _validate_mask(antibody_mask, antibody_features, "antibody_mask")

        antigen = self.antigen_projection(antigen_features)
        antibody = self.antibody_projection(antibody_features)
        for block in self.blocks:
            antigen, antibody = block(antigen, antibody, antigen_mask, antibody_mask)
        antigen = self.final_antigen_norm(antigen)
        antibody = self.final_antibody_norm(antibody)
        decoder = self.decoder.components(antigen, antibody)
        pair_logits = decoder["pair_logits"]
        interaction_logits = decoder["interaction_logits"]
        axis_mask = antigen_mask.unsqueeze(2) & antibody_mask.unsqueeze(1)
        if candidate_pair_mask is None:
            edge_supervision_mask = axis_mask
        else:
            if candidate_pair_mask.dtype != torch.bool or candidate_pair_mask.shape != pair_logits.shape:
                raise ValueError("candidate_pair_mask must be boolean and aligned to pair logits")
            edge_supervision_mask = axis_mask & candidate_pair_mask
        if torch.any(edge_supervision_mask.reshape(edge_supervision_mask.shape[0], -1).sum(dim=1) == 0):
            raise ValueError("Every item must contain at least one valid candidate pair")

        pair_probabilities = torch.sigmoid(pair_logits)
        pair_epitope_probability, pair_paratope_probability = self._marginals(
            pair_logits, pair_probabilities, axis_mask
        )
        epitope_logits = self.epitope_head(antigen).squeeze(-1)
        paratope_logits = self.paratope_head(antibody).squeeze(-1)
        epitope_probability = torch.sigmoid(epitope_logits).masked_fill(
            ~antigen_mask, 0.0
        )
        paratope_probability = torch.sigmoid(paratope_logits).masked_fill(
            ~antibody_mask, 0.0
        )
        complex_score = normalized_partner_score(
            interaction_logits,
            axis_mask,
            pooling=self.score_pooling,
            temperature=self.score_temperature,
            top_fraction=self.score_top_fraction,
            center=self.center_partner_score,
            centering_iterations=1,
        )
        return {
            "pair_logits": pair_logits,
            "interaction_logits": interaction_logits,
            "pair_probabilities": pair_probabilities,
            "pair_mask": axis_mask,
            "edge_supervision_mask": edge_supervision_mask,
            "epitope_logits": epitope_logits,
            "paratope_logits": paratope_logits,
            "epitope_probability": epitope_probability,
            "paratope_probability": paratope_probability,
            "pair_epitope_probability": pair_epitope_probability,
            "pair_paratope_probability": pair_paratope_probability,
            "complex_score": complex_score,
            "antigen_embeddings": antigen,
            "antibody_embeddings": antibody,
        }


def model_parameter_count(model: Any) -> int:
    """Return the number of trainable scalar parameters."""
    require_torch()
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
