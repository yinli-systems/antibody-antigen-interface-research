"""FrameContact: geometry-aware, partner-specific antibody-antigen modeling.

The model deliberately separates three sources of evidence:

* independently parameterized antibody and antigen intra-chain encoders;
* an explicit residue-pair contact predictor that uses content, not raw
  cross-chain coordinates; and
* bidirectional cross-attention gated by the predicted contact distribution.

The geometry and cross-attention residuals are zero initialized.  A freshly
constructed model therefore has an exact dense self-attention residue path,
while the contact head can already receive direct pair-supervision gradients.
This makes every added information path ablatable without silently changing
the sequence baseline.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional
except ModuleNotFoundError:  # pragma: no cover - optional import boundary
    torch = None
    nn = None
    functional = None

from .intra_chain_geometry import IntraChainGeometryEncoder, build_backbone_frames
from .sabdab2_structure import StructureCacheError


_ModuleBase = nn.Module if nn is not None else object


def _require_torch() -> Any:
    if torch is None or nn is None or functional is None:
        raise RuntimeError("PyTorch is required for FrameContact")
    return torch


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_residue_stream(
    features: Any,
    residue_mask: Any,
    backbone: Any,
    chain_ids: Any,
    *,
    name: str,
    d_model: int,
) -> None:
    torch_module = _require_torch()
    if not torch_module.is_tensor(features) or features.ndim != 3:
        raise ValueError(f"{name}_features must have shape [batch, residue, channel]")
    if int(features.shape[-1]) != d_model:
        raise ValueError(f"{name}_features have the wrong channel dimension")
    expected = tuple(features.shape[:2])
    if (
        not torch_module.is_tensor(residue_mask)
        or residue_mask.dtype != torch_module.bool
        or tuple(residue_mask.shape) != expected
    ):
        raise ValueError(f"{name}_mask must be a boolean [batch, residue] tensor")
    if not bool(residue_mask.any(dim=-1).all()):
        raise ValueError(f"every {name} example must contain at least one residue")
    if not torch_module.is_tensor(backbone) or tuple(backbone.shape) != (*expected, 3, 3):
        raise ValueError(f"{name}_backbone must have shape [batch, residue, 3, 3]")
    if not torch_module.is_tensor(chain_ids) or tuple(chain_ids.shape) != expected:
        raise ValueError(f"{name}_chain_ids must align with the residue mask")


class FactorizedIntraChainEncoder(_ModuleBase):
    """Fuse a dense global path with a zero-gated spatial geometry path."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        *,
        geometry_feature_mode: str = "full_pose",
        geometry_bias_mode: str = "pair_mlp",
        ipa_num_points: int = 4,
        geometry_attention_graph_mode: str = "spatial",
        num_rbf: int = 16,
        neighbor_k: int | None = 32,
        neighbor_radius: float | None = 16.0,
        sequence_window: int = 32,
        dropout: float = 0.0,
    ) -> None:
        _require_torch()
        super().__init__()
        self.d_model = _positive_integer(d_model, "d_model")
        self.dense_encoder = IntraChainGeometryEncoder(
            d_model,
            num_heads,
            num_layers,
            geometry_feature_mode="none",
            attention_graph_mode="dense",
            num_rbf=num_rbf,
            neighbor_k=None,
            neighbor_radius=None,
            sequence_window=sequence_window,
            dropout=dropout,
        )
        self.geometry_encoder = IntraChainGeometryEncoder(
            d_model,
            num_heads,
            num_layers,
            geometry_feature_mode=geometry_feature_mode,
            geometry_bias_mode=geometry_bias_mode,
            ipa_num_points=ipa_num_points,
            attention_graph_mode=geometry_attention_graph_mode,
            num_rbf=num_rbf,
            neighbor_k=neighbor_k,
            neighbor_radius=neighbor_radius,
            sequence_window=sequence_window,
            dropout=dropout,
        )
        self.geometry_residual_gate = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        features: Any,
        residue_mask: Any,
        backbone: Any,
        *,
        chain_ids: Any,
        positions: Any | None = None,
        neighbor_mask_override: Any | None = None,
        return_attention: bool = False,
    ) -> tuple[Any, dict[str, Any] | None]:
        dense, dense_details = self.dense_encoder(
            features,
            residue_mask,
            backbone,
            chain_ids=chain_ids,
            positions=positions,
            return_attention=return_attention,
        )
        geometry, geometry_details = self.geometry_encoder(
            features,
            residue_mask,
            backbone,
            chain_ids=chain_ids,
            positions=positions,
            neighbor_mask_override=neighbor_mask_override,
            return_attention=return_attention,
        )
        gate = torch.tanh(self.geometry_residual_gate)
        output = dense + gate * (geometry - features)
        output = output.masked_fill(~residue_mask[..., None], 0.0)
        if not return_attention:
            return output, None
        return output, {
            "dense": dense_details,
            "geometry": geometry_details,
            "geometry_residual_gate": gate,
        }


class DualFactorizedIntraChainEncoder(_ModuleBase):
    """Two fully independent factorized streams; no antibody/antigen sharing."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        _require_torch()
        super().__init__()
        self.antibody_encoder = FactorizedIntraChainEncoder(*args, **kwargs)
        self.antigen_encoder = FactorizedIntraChainEncoder(*args, **kwargs)

    def forward(
        self,
        antibody_features: Any,
        antigen_features: Any,
        antibody_mask: Any,
        antigen_mask: Any,
        antibody_backbone: Any,
        antigen_backbone: Any,
        *,
        antibody_chain_ids: Any,
        antigen_chain_ids: Any,
        antibody_neighbor_mask_override: Any | None = None,
        antigen_neighbor_mask_override: Any | None = None,
        return_attention: bool = False,
    ) -> tuple[Any, Any, dict[str, Any] | None]:
        antibody, antibody_details = self.antibody_encoder(
            antibody_features,
            antibody_mask,
            antibody_backbone,
            chain_ids=antibody_chain_ids,
            neighbor_mask_override=antibody_neighbor_mask_override,
            return_attention=return_attention,
        )
        antigen, antigen_details = self.antigen_encoder(
            antigen_features,
            antigen_mask,
            antigen_backbone,
            chain_ids=antigen_chain_ids,
            neighbor_mask_override=antigen_neighbor_mask_override,
            return_attention=return_attention,
        )
        if not return_attention:
            return antibody, antigen, None
        return antibody, antigen, {
            "antibody": antibody_details,
            "antigen": antigen_details,
        }


class PairContactHead(_ModuleBase):
    """Low-rank explicit contact logits without a materialized pair embedding."""

    def __init__(self, d_model: int, pair_dim: int | None = None) -> None:
        _require_torch()
        super().__init__()
        d_model = _positive_integer(d_model, "d_model")
        pair_dim = d_model if pair_dim is None else _positive_integer(pair_dim, "pair_dim")
        self.pair_dim = pair_dim
        self.antibody_norm = nn.LayerNorm(d_model)
        self.antigen_norm = nn.LayerNorm(d_model)
        self.antibody_query = nn.Linear(d_model, pair_dim, bias=False)
        self.antigen_key = nn.Linear(d_model, pair_dim, bias=False)
        self.antibody_unary = nn.Linear(d_model, 1)
        self.antigen_unary = nn.Linear(d_model, 1)
        self.contact_bias = nn.Parameter(torch.tensor(-4.0))

    def components(
        self,
        antibody: Any,
        antigen: Any,
        pair_mask: Any,
    ) -> dict[str, Any]:
        """Return pair synergy separately from single-chain propensities.

        ``interaction_logits`` is the only term that depends jointly on an
        antibody residue and an antigen residue.  The two unary terms are
        useful for calibrated cognate contact prediction, but must not be
        allowed to establish partner specificity by themselves.
        """
        query = self.antibody_query(self.antibody_norm(antibody))
        key = self.antigen_key(self.antigen_norm(antigen))
        interaction_logits = torch.einsum("bid,bjd->bij", query, key)
        interaction_logits = interaction_logits / math.sqrt(self.pair_dim)
        antibody_unary_logits = self.antibody_unary(antibody)
        antigen_unary_logits = self.antigen_unary(antigen).transpose(1, 2)
        logits = (
            interaction_logits
            + antibody_unary_logits
            + antigen_unary_logits
            + self.contact_bias
        )
        return {
            "logits": logits.masked_fill(~pair_mask, 0.0),
            "interaction_logits": interaction_logits.masked_fill(~pair_mask, 0.0),
            "antibody_unary_logits": antibody_unary_logits,
            "antigen_unary_logits": antigen_unary_logits,
            "contact_bias": self.contact_bias,
        }

    def forward(
        self,
        antibody: Any,
        antigen: Any,
        pair_mask: Any,
    ) -> Any:
        return self.components(antibody, antigen, pair_mask)["logits"]


class BidirectionalPairContactHead(_ModuleBase):
    """Average two explicitly directional pair decoders on one candidate grid."""

    def __init__(self, d_model: int, pair_dim: int | None = None) -> None:
        _require_torch()
        super().__init__()
        self.directional = PairContactHead(d_model, pair_dim)

    def components(
        self,
        antibody: Any,
        antigen: Any,
        pair_mask: Any,
    ) -> dict[str, Any]:
        forward = self.directional.components(antibody, antigen, pair_mask)
        reverse_raw = self.directional.components(
            antigen, antibody, pair_mask.transpose(1, 2)
        )
        reverse_logits = reverse_raw["logits"].transpose(1, 2)
        reverse_interaction = reverse_raw["interaction_logits"].transpose(1, 2)
        reverse_antibody_unary = reverse_raw["antigen_unary_logits"].transpose(1, 2)
        reverse_antigen_unary = reverse_raw["antibody_unary_logits"].transpose(1, 2)
        return {
            "logits": 0.5 * (forward["logits"] + reverse_logits),
            "interaction_logits": 0.5
            * (forward["interaction_logits"] + reverse_interaction),
            "antibody_unary_logits": 0.5
            * (forward["antibody_unary_logits"] + reverse_antibody_unary),
            "antigen_unary_logits": 0.5
            * (forward["antigen_unary_logits"] + reverse_antigen_unary),
            "contact_bias": 0.5
            * (forward["contact_bias"] + reverse_raw["contact_bias"]),
        }

    def forward(
        self,
        antibody: Any,
        antigen: Any,
        pair_mask: Any,
    ) -> Any:
        return self.components(antibody, antigen, pair_mask)["logits"]


class PairDistanceBinHead(_ModuleBase):
    """Predict cross-chain distance classes from residue representations only."""

    def __init__(
        self,
        d_model: int,
        *,
        num_bins: int = 5,
        pair_dim: int | None = None,
    ) -> None:
        _require_torch()
        super().__init__()
        d_model = _positive_integer(d_model, "d_model")
        self.num_bins = _positive_integer(num_bins, "num_bins")
        pair_dim = d_model if pair_dim is None else _positive_integer(pair_dim, "pair_dim")
        self.pair_dim = pair_dim
        self.antibody_norm = nn.LayerNorm(d_model)
        self.antigen_norm = nn.LayerNorm(d_model)
        self.antibody_query = nn.Linear(d_model, self.num_bins * pair_dim, bias=False)
        self.antigen_key = nn.Linear(d_model, self.num_bins * pair_dim, bias=False)
        self.antibody_unary = nn.Linear(d_model, self.num_bins)
        self.antigen_unary = nn.Linear(d_model, self.num_bins)
        self.bin_bias = nn.Parameter(torch.zeros(self.num_bins))

    def forward(
        self,
        antibody: Any,
        antigen: Any,
        pair_mask: Any,
    ) -> Any:
        batch_size, antibody_count, _ = antibody.shape
        antigen_count = int(antigen.shape[1])
        query = self.antibody_query(self.antibody_norm(antibody)).view(
            batch_size, antibody_count, self.num_bins, self.pair_dim
        )
        key = self.antigen_key(self.antigen_norm(antigen)).view(
            batch_size, antigen_count, self.num_bins, self.pair_dim
        )
        logits = torch.einsum("bikd,bjkd->bijk", query, key)
        logits = logits / math.sqrt(self.pair_dim)
        logits = (
            logits
            + self.antibody_unary(antibody)[:, :, None, :]
            + self.antigen_unary(antigen)[:, None, :, :]
            + self.bin_bias
        )
        return logits.masked_fill(~pair_mask[..., None], 0.0)


class ContactGatedCrossAttention(_ModuleBase):
    """One directional cross-attention update controlled by contact logits."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        *,
        dropout: float = 0.0,
        contact_gate_mode: str = "learned",
    ) -> None:
        _require_torch()
        super().__init__()
        d_model = _positive_integer(d_model, "d_model")
        num_heads = _positive_integer(num_heads, "num_heads")
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        if contact_gate_mode not in {"learned", "content_only"}:
            raise ValueError("contact_gate_mode must be learned or content_only")
        self.contact_gate_mode = contact_gate_mode
        self.head_dim = d_model // num_heads
        self.query_norm = nn.LayerNorm(d_model)
        self.partner_norm = nn.LayerNorm(d_model)
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.contact_scale_unconstrained = nn.Parameter(
            torch.full((num_heads,), math.log(math.expm1(1.0)))
        )
        self.residual_gate = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        receiver: Any,
        partner: Any,
        receiver_mask: Any,
        partner_mask: Any,
        contact_logits: Any,
        pair_mask: Any,
        *,
        return_attention: bool = False,
    ) -> tuple[Any, dict[str, Any] | None]:
        batch_size, receiver_count, _ = receiver.shape
        partner_count = int(partner.shape[1])
        query = self.query(self.query_norm(receiver)).view(
            batch_size, receiver_count, self.num_heads, self.head_dim
        ).transpose(1, 2)
        normalized_partner = self.partner_norm(partner)
        key = self.key(normalized_partner).view(
            batch_size, partner_count, self.num_heads, self.head_dim
        ).transpose(1, 2)
        value = self.value(normalized_partner).view(
            batch_size, partner_count, self.num_heads, self.head_dim
        ).transpose(1, 2)
        content_logits = torch.einsum("bhid,bhjd->bhij", query, key)
        content_logits = content_logits / math.sqrt(self.head_dim)
        contact_log_probability = functional.logsigmoid(contact_logits)
        contact_scale = functional.softplus(self.contact_scale_unconstrained)
        logits = content_logits
        if self.contact_gate_mode == "learned":
            logits = (
                logits
                + contact_scale[None, :, None, None]
                * contact_log_probability[:, None]
            )
        else:
            contact_scale = torch.zeros_like(contact_scale)
        attention_mask = pair_mask[:, None]
        logits = logits.masked_fill(~attention_mask, torch.finfo(logits.dtype).min)
        attention = torch.softmax(logits, dim=-1)
        query_has_candidate = pair_mask.any(dim=-1)
        attention = attention * (
            receiver_mask & query_has_candidate
        )[:, None, :, None].to(attention.dtype)
        content_only_logits = content_logits.masked_fill(
            ~attention_mask, torch.finfo(content_logits.dtype).min
        )
        content_only_attention = torch.softmax(content_only_logits, dim=-1)
        content_only_attention = content_only_attention * (
            receiver_mask & query_has_candidate
        )[:, None, :, None].to(content_only_attention.dtype)
        context = torch.einsum("bhij,bhjd->bhid", self.dropout(attention), value)
        context = context.transpose(1, 2).reshape(batch_size, receiver_count, self.d_model)
        gate = torch.tanh(self.residual_gate)
        updated = receiver + gate * self.dropout(self.output(context))
        updated = updated.masked_fill(~receiver_mask[..., None], 0.0)
        if not return_attention:
            return updated, None
        return updated, {
            "weights": attention,
            "content_only_weights": content_only_attention,
            "contact_log_probability": contact_log_probability,
            "contact_scale_per_head": contact_scale,
            "residual_gate": gate,
            "pair_mask": pair_mask,
            "partner_mask": partner_mask,
        }


class BidirectionalContactGatedCrossAttention(_ModuleBase):
    """Distinct antibody<-antigen and antigen<-antibody cross-attention."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        *,
        dropout: float = 0.0,
        contact_gate_mode: str = "learned",
    ) -> None:
        _require_torch()
        super().__init__()
        self.antibody_from_antigen = ContactGatedCrossAttention(
            d_model,
            num_heads,
            dropout=dropout,
            contact_gate_mode=contact_gate_mode,
        )
        self.antigen_from_antibody = ContactGatedCrossAttention(
            d_model,
            num_heads,
            dropout=dropout,
            contact_gate_mode=contact_gate_mode,
        )

    def forward(
        self,
        antibody: Any,
        antigen: Any,
        antibody_mask: Any,
        antigen_mask: Any,
        contact_logits: Any,
        pair_mask: Any,
        *,
        return_attention: bool = False,
    ) -> tuple[Any, Any, dict[str, Any] | None]:
        antibody_updated, antibody_details = self.antibody_from_antigen(
            antibody,
            antigen,
            antibody_mask,
            antigen_mask,
            contact_logits,
            pair_mask,
            return_attention=return_attention,
        )
        antigen_updated, antigen_details = self.antigen_from_antibody(
            antigen,
            antibody,
            antigen_mask,
            antibody_mask,
            contact_logits.transpose(1, 2),
            pair_mask.transpose(1, 2),
            return_attention=return_attention,
        )
        if not return_attention:
            return antibody_updated, antigen_updated, None
        return antibody_updated, antigen_updated, {
            "antibody_from_antigen": antibody_details,
            "antigen_from_antibody": antigen_details,
        }


def masked_logmeanexp(logits: Any, mask: Any, *, dim: int) -> Any:
    """Length-normalized contact marginal on an explicitly observed pair set."""
    torch_module = _require_torch()
    if tuple(logits.shape) != tuple(mask.shape) or mask.dtype != torch_module.bool:
        raise ValueError("logits and boolean mask must have identical shapes")
    observed_count = mask.sum(dim=dim)
    count = observed_count.clamp_min(1)
    masked = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    result = torch.logsumexp(masked, dim=dim) - count.to(logits.dtype).log()
    return result.masked_fill(observed_count == 0, 0.0)


def masked_mean_probability_logit(logits: Any, mask: Any, *, dim: int) -> Any:
    """EpiFormer Eq. 9 mean-edge probability represented as a stable logit."""
    torch_module = _require_torch()
    if tuple(logits.shape) != tuple(mask.shape) or mask.dtype != torch_module.bool:
        raise ValueError("logits and boolean mask must have identical shapes")
    observed_count = mask.sum(dim=dim)
    count = observed_count.clamp_min(1).to(torch_module.float32)
    probabilities = torch_module.sigmoid(logits.to(torch_module.float32))
    mean_probability = probabilities.masked_fill(~mask, 0.0).sum(dim=dim) / count
    epsilon = torch_module.finfo(torch_module.float32).eps
    result = torch_module.logit(mean_probability.clamp(epsilon, 1.0 - epsilon))
    return result.masked_fill(observed_count == 0, 0.0)


def masked_topk_mean_probability_logit(
    logits: Any, mask: Any, *, dim: int, k: int = 2
) -> Any:
    """Masked EpiFormer top-k mean edge probability represented as a logit."""
    torch_module = _require_torch()
    if tuple(logits.shape) != tuple(mask.shape) or mask.dtype != torch_module.bool:
        raise ValueError("logits and boolean mask must have identical shapes")
    if k < 1:
        raise ValueError("k must be positive")
    observed_count = mask.sum(dim=dim)
    effective_k = min(k, logits.shape[dim])
    probabilities = torch_module.sigmoid(logits.to(torch_module.float32))
    masked = probabilities.masked_fill(~mask, -1.0)
    top_values = torch_module.topk(masked, k=effective_k, dim=dim).values
    selected = top_values.clamp_min(0.0).sum(dim=dim)
    count = observed_count.clamp(min=1, max=effective_k).to(torch_module.float32)
    mean_probability = selected / count
    epsilon = torch_module.finfo(torch_module.float32).eps
    result = torch_module.logit(mean_probability.clamp(epsilon, 1.0 - epsilon))
    return result.masked_fill(observed_count == 0, 0.0)


def masked_noisy_or_probability_logit(logits: Any, mask: Any, *, dim: int) -> Any:
    """Stable noisy-OR over observed pair probabilities, returned as a logit."""
    torch_module = _require_torch()
    if tuple(logits.shape) != tuple(mask.shape) or mask.dtype != torch_module.bool:
        raise ValueError("logits and boolean mask must have identical shapes")
    observed_count = mask.sum(dim=dim)
    probabilities = torch_module.sigmoid(logits.to(torch_module.float32))
    epsilon = torch_module.finfo(torch_module.float32).eps
    log_complement = torch_module.log1p(
        -probabilities.clamp(max=1.0 - epsilon)
    ).masked_fill(~mask, 0.0)
    pooled = -torch_module.expm1(log_complement.sum(dim=dim))
    result = torch_module.logit(pooled.clamp(epsilon, 1.0 - epsilon))
    return result.masked_fill(observed_count == 0, 0.0)


def masked_learned_softmax_probability_logit(
    logits: Any, mask: Any, *, dim: int, log_temperature: Any
) -> Any:
    """Learned-temperature softmax pooling over observed pair probabilities."""
    torch_module = _require_torch()
    if tuple(logits.shape) != tuple(mask.shape) or mask.dtype != torch_module.bool:
        raise ValueError("logits and boolean mask must have identical shapes")
    observed_count = mask.sum(dim=dim)
    temperature = log_temperature.exp().clamp(0.05, 20.0).to(torch_module.float32)
    scores = logits.to(torch_module.float32) / temperature
    scores = scores.masked_fill(~mask, torch_module.finfo(scores.dtype).min)
    weights = torch_module.softmax(scores, dim=dim).masked_fill(~mask, 0.0)
    weights = weights / weights.sum(dim=dim, keepdim=True).clamp_min(
        torch_module.finfo(weights.dtype).eps
    )
    probabilities = torch_module.sigmoid(logits.to(torch_module.float32))
    pooled = (weights * probabilities).sum(dim=dim)
    epsilon = torch_module.finfo(torch_module.float32).eps
    result = torch_module.logit(pooled.clamp(epsilon, 1.0 - epsilon))
    return result.masked_fill(observed_count == 0, 0.0)


class FrameContactModel(_ModuleBase):
    """End-to-end self geometry, explicit contacts, and gated interaction model.

    ``fusion_mode="late"`` is the capacity-minimal confirmatory model: all
    intra-chain layers run before one bidirectional contact-gated update.
    ``fusion_mode="interleaved"`` applies a separately parameterized
    bidirectional contact-gated update after every intra-chain layer.  The two
    modes are intentionally explicit so an interleaved model can never be
    reported when only a late decoder was executed.
    """

    def __init__(
        self,
        *,
        antibody_input_dim: int = 1280,
        antigen_input_dim: int = 1280,
        d_model: int = 128,
        num_heads: int = 8,
        num_self_layers: int = 2,
        geometry_feature_mode: str = "full_pose",
        geometry_bias_mode: str = "pair_mlp",
        ipa_num_points: int = 4,
        geometry_attention_graph_mode: str = "spatial",
        num_rbf: int = 16,
        neighbor_k: int | None = 32,
        neighbor_radius: float | None = 16.0,
        sequence_window: int = 32,
        pair_dim: int | None = None,
        dropout: float = 0.1,
        fusion_mode: str = "late",
        cross_attention_contact_gate_mode: str = "learned",
        adapter_mode: str = "framecontact",
        epitope_pooling: str = "topk_mean",
        epitope_pooling_k: int = 2,
        paratope_pooling: str = "logmeanexp",
        paratope_pooling_k: int = 2,
        contact_readout: str = "pair",
        distance_bin_count: int = 0,
        contrastive_dim: int = 64,
    ) -> None:
        _require_torch()
        super().__init__()
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = _positive_integer(d_model, "d_model")
        num_self_layers = _positive_integer(num_self_layers, "num_self_layers")
        if fusion_mode not in {"late", "interleaved"}:
            raise ValueError("fusion_mode must be 'late' or 'interleaved'")
        if adapter_mode not in {
            "framecontact",
            "epiformer_style_same_protocol",
            "mipe_style_same_protocol",
        }:
            raise ValueError("unknown same-protocol adapter mode")
        self.fusion_mode = fusion_mode
        self.adapter_mode = adapter_mode
        if epitope_pooling not in {
            "topk_mean",
            "noisy_or",
            "logmeanexp",
            "learned_softmax",
        }:
            raise ValueError("unknown epitope pooling rule")
        self.epitope_pooling = epitope_pooling
        self.epitope_pooling_k = _positive_integer(
            epitope_pooling_k, "epitope_pooling_k"
        )
        if paratope_pooling not in {"logmeanexp", "topk_mean"}:
            raise ValueError("unknown paratope pooling rule")
        if contact_readout not in {"pair", "topk_context_equal_logit"}:
            raise ValueError("unknown contact readout rule")
        self.paratope_pooling = paratope_pooling
        self.paratope_pooling_k = _positive_integer(
            paratope_pooling_k, "paratope_pooling_k"
        )
        self.contact_readout = contact_readout
        if epitope_pooling == "learned_softmax":
            self.epitope_pooling_log_temperature = nn.Parameter(torch.zeros(()))
        else:
            self.register_parameter("epitope_pooling_log_temperature", None)
        self.antibody_projection = nn.Sequential(
            nn.LayerNorm(antibody_input_dim), nn.Linear(antibody_input_dim, d_model)
        )
        self.antigen_projection = nn.Sequential(
            nn.LayerNorm(antigen_input_dim), nn.Linear(antigen_input_dim, d_model)
        )
        encoder_arguments = {
            "geometry_feature_mode": geometry_feature_mode,
            "geometry_bias_mode": geometry_bias_mode,
            "ipa_num_points": ipa_num_points,
            "geometry_attention_graph_mode": geometry_attention_graph_mode,
            "num_rbf": num_rbf,
            "neighbor_k": neighbor_k,
            "neighbor_radius": neighbor_radius,
            "sequence_window": sequence_window,
            "dropout": dropout,
        }
        if fusion_mode == "late":
            self.self_encoder = DualFactorizedIntraChainEncoder(
                d_model,
                num_heads,
                num_self_layers,
                **encoder_arguments,
            )
            self.interleaved_self_encoders = None
            self.interleaved_cross_attentions = None
        else:
            self.self_encoder = None
            self.interleaved_self_encoders = nn.ModuleList(
                DualFactorizedIntraChainEncoder(
                    d_model,
                    num_heads,
                    1,
                    **encoder_arguments,
                )
                for _ in range(num_self_layers)
            )
            self.interleaved_cross_attentions = nn.ModuleList(
                BidirectionalContactGatedCrossAttention(
                    d_model,
                    num_heads,
                    dropout=dropout,
                    contact_gate_mode=cross_attention_contact_gate_mode,
                )
                for _ in range(num_self_layers)
            )
        self.contact_head = (
            BidirectionalPairContactHead(d_model, pair_dim)
            if adapter_mode == "epiformer_style_same_protocol"
            else PairContactHead(d_model, pair_dim)
        )
        self.distance_bin_head = (
            PairDistanceBinHead(
                d_model,
                num_bins=_positive_integer(distance_bin_count, "distance_bin_count"),
                pair_dim=pair_dim,
            )
            if distance_bin_count
            else None
        )
        if adapter_mode == "mipe_style_same_protocol":
            contrastive_dim = _positive_integer(contrastive_dim, "contrastive_dim")
            self.antibody_sequence_contrastive = nn.Linear(d_model, contrastive_dim)
            self.antibody_geometry_contrastive = nn.Linear(d_model, contrastive_dim)
            self.antigen_sequence_contrastive = nn.Linear(d_model, contrastive_dim)
            self.antigen_geometry_contrastive = nn.Linear(d_model, contrastive_dim)
        else:
            self.antibody_sequence_contrastive = None
            self.antibody_geometry_contrastive = None
            self.antigen_sequence_contrastive = None
            self.antigen_geometry_contrastive = None
        self.cross_attention = (
            BidirectionalContactGatedCrossAttention(
                d_model,
                num_heads,
                dropout=dropout,
                contact_gate_mode=cross_attention_contact_gate_mode,
            )
            if fusion_mode == "late"
            else None
        )
        self.antibody_norm = nn.LayerNorm(d_model)
        self.antigen_norm = nn.LayerNorm(d_model)
        self.paratope_head = nn.Linear(d_model, 1)
        self.epitope_head = nn.Linear(d_model, 1)
        self.paratope_contact_gate = nn.Parameter(torch.zeros(()))
        self.epitope_contact_gate = nn.Parameter(torch.zeros(()))
        nn.init.constant_(self.paratope_head.bias, -2.5)
        nn.init.constant_(self.epitope_head.bias, -2.5)

    @staticmethod
    def _validate_frame_partition(batch: Mapping[str, Any], partner: str) -> None:
        residue_mask = batch[f"{partner}_mask"]
        valid = batch[f"{partner}_frame_valid_mask"]
        missing = batch[f"{partner}_missing_frame_mask"]
        if (
            valid.dtype != torch.bool
            or missing.dtype != torch.bool
            or not torch.equal(valid | missing, residue_mask)
            or bool((valid & missing).any())
        ):
            raise StructureCacheError(
                f"{partner} valid/missing masks do not partition real residues"
            )
        derived = build_backbone_frames(
            batch[f"{partner}_backbone"], residue_mask
        ).valid_mask
        if not torch.equal(derived, valid):
            raise StructureCacheError(
                f"{partner} frame mask differs from the N/CA/C geometry"
            )

    def forward(
        self,
        batch: Mapping[str, Any],
        *,
        return_attention: bool = False,
    ) -> dict[str, Any]:
        antibody_mask = batch["antibody_mask"]
        antigen_mask = batch["antigen_mask"]
        antibody = self.antibody_projection(batch["antibody_features"])
        antigen = self.antigen_projection(batch["antigen_features"])
        antibody_sequence_view = antibody
        antigen_sequence_view = antigen
        _validate_residue_stream(
            antibody,
            antibody_mask,
            batch["antibody_backbone"],
            batch["antibody_chain_ids"],
            name="antibody",
            d_model=self.d_model,
        )
        _validate_residue_stream(
            antigen,
            antigen_mask,
            batch["antigen_backbone"],
            batch["antigen_chain_ids"],
            name="antigen",
            d_model=self.d_model,
        )
        self._validate_frame_partition(batch, "antibody")
        self._validate_frame_partition(batch, "antigen")
        pair_mask = antibody_mask[:, :, None] & antigen_mask[:, None, :]
        if "pair_candidate_mask" in batch:
            candidate_mask = batch["pair_candidate_mask"]
            if (
                candidate_mask.dtype != torch.bool
                or tuple(candidate_mask.shape) != tuple(pair_mask.shape)
                or bool((candidate_mask & ~pair_mask).any())
            ):
                raise ValueError("pair_candidate_mask must be a subset of real residue pairs")
            pair_mask = pair_mask & candidate_mask
        if not bool(pair_mask.flatten(start_dim=1).any(dim=-1).all()):
            raise ValueError("every example must contain at least one candidate residue pair")
        encoder_keywords = {
            "antibody_chain_ids": batch["antibody_chain_ids"],
            "antigen_chain_ids": batch["antigen_chain_ids"],
            "antibody_neighbor_mask_override": batch.get(
                "antibody_neighbor_mask_override"
            ),
            "antigen_neighbor_mask_override": batch.get(
                "antigen_neighbor_mask_override"
            ),
            "return_attention": return_attention,
        }
        if self.fusion_mode == "late":
            assert self.self_encoder is not None and self.cross_attention is not None
            antibody, antigen, self_details = self.self_encoder(
                antibody,
                antigen,
                antibody_mask,
                antigen_mask,
                batch["antibody_backbone"],
                batch["antigen_backbone"],
                **encoder_keywords,
            )
            initial_contact_components = self.contact_head.components(
                antibody, antigen, pair_mask
            )
            initial_contact_logits = initial_contact_components["logits"]
            antibody_fused, antigen_fused, cross_details = self.cross_attention(
                antibody,
                antigen,
                antibody_mask,
                antigen_mask,
                initial_contact_logits,
                pair_mask,
                return_attention=return_attention,
            )
            intermediate_contact_logits = (initial_contact_logits,)
        else:
            assert self.interleaved_self_encoders is not None
            assert self.interleaved_cross_attentions is not None
            self_layer_details = []
            cross_layer_details = []
            intermediate_contact_logits_list = []
            initial_contact_components = None
            for self_layer, cross_layer in zip(
                self.interleaved_self_encoders,
                self.interleaved_cross_attentions,
                strict=True,
            ):
                antibody, antigen, layer_self_details = self_layer(
                    antibody,
                    antigen,
                    antibody_mask,
                    antigen_mask,
                    batch["antibody_backbone"],
                    batch["antigen_backbone"],
                    **encoder_keywords,
                )
                stage_contact_components = self.contact_head.components(
                    antibody, antigen, pair_mask
                )
                if initial_contact_components is None:
                    initial_contact_components = stage_contact_components
                stage_contact_logits = stage_contact_components["logits"]
                antibody, antigen, layer_cross_details = cross_layer(
                    antibody,
                    antigen,
                    antibody_mask,
                    antigen_mask,
                    stage_contact_logits,
                    pair_mask,
                    return_attention=return_attention,
                )
                intermediate_contact_logits_list.append(stage_contact_logits)
                if return_attention:
                    self_layer_details.append(layer_self_details)
                    cross_layer_details.append(layer_cross_details)
            intermediate_contact_logits = tuple(intermediate_contact_logits_list)
            initial_contact_logits = intermediate_contact_logits[0]
            assert initial_contact_components is not None
            antibody_fused, antigen_fused = antibody, antigen
            if return_attention:
                def merge_stream_details(partner: str, path: str) -> dict[str, Any]:
                    stage_details = [
                        details[partner][path] for details in self_layer_details
                    ]
                    merged = {
                        key: value
                        for key, value in stage_details[0].items()
                        if key != "layers"
                    }
                    merged["layers"] = tuple(
                        layer
                        for details in stage_details
                        for layer in details["layers"]
                    )
                    return merged

                self_details = {
                    "fusion_mode": "interleaved",
                    "antibody": {
                        "dense": merge_stream_details("antibody", "dense"),
                        "geometry": merge_stream_details("antibody", "geometry"),
                    },
                    "antigen": {
                        "dense": merge_stream_details("antigen", "dense"),
                        "geometry": merge_stream_details("antigen", "geometry"),
                    },
                }
            else:
                self_details = None
            cross_details = (
                {"fusion_mode": "interleaved", "layers": tuple(cross_layer_details)}
                if return_attention
                else None
            )
        # Reuse the exact same head after interaction. With zero-initialized cross
        # residual gates, fused and self features are identical, so this is an
        # exact baseline-preserving refinement rather than added pair capacity.
        contact_components = self.contact_head.components(
            antibody_fused, antigen_fused, pair_mask
        )
        raw_contact_logits = contact_components["logits"]
        distance_bin_logits = (
            None
            if self.distance_bin_head is None
            else self.distance_bin_head(antibody_fused, antigen_fused, pair_mask)
        )
        antibody_geometry_view = antibody_fused
        antigen_geometry_view = antigen_fused
        antibody_fused = self.antibody_norm(antibody_fused).masked_fill(
            ~antibody_mask[..., None], 0.0
        )
        antigen_fused = self.antigen_norm(antigen_fused).masked_fill(
            ~antigen_mask[..., None], 0.0
        )
        if self.paratope_pooling == "topk_mean":
            paratope_contact = masked_topk_mean_probability_logit(
                raw_contact_logits,
                pair_mask,
                dim=2,
                k=self.paratope_pooling_k,
            )
        else:
            paratope_contact = masked_logmeanexp(
                raw_contact_logits, pair_mask, dim=2
            )
        if self.adapter_mode == "epiformer_style_same_protocol":
            if self.epitope_pooling == "topk_mean":
                epitope_contact = masked_topk_mean_probability_logit(
                    raw_contact_logits,
                    pair_mask,
                    dim=1,
                    k=self.epitope_pooling_k,
                )
            elif self.epitope_pooling == "noisy_or":
                epitope_contact = masked_noisy_or_probability_logit(
                    raw_contact_logits, pair_mask, dim=1
                )
            elif self.epitope_pooling == "logmeanexp":
                epitope_contact = masked_logmeanexp(
                    raw_contact_logits, pair_mask, dim=1
                )
            else:
                epitope_contact = masked_learned_softmax_probability_logit(
                    raw_contact_logits,
                    pair_mask,
                    dim=1,
                    log_temperature=self.epitope_pooling_log_temperature,
                )
        else:
            epitope_contact = masked_logmeanexp(
                raw_contact_logits, pair_mask, dim=1
            )
        if self.contact_readout == "topk_context_equal_logit":
            contact_row_topk = masked_topk_mean_probability_logit(
                raw_contact_logits, pair_mask, dim=2, k=2
            )
            contact_column_topk = masked_topk_mean_probability_logit(
                raw_contact_logits, pair_mask, dim=1, k=2
            )
            contact_logits = (
                raw_contact_logits
                + contact_row_topk[:, :, None]
                + contact_column_topk[:, None, :]
            ) / 3.0
            contact_logits = contact_logits.masked_fill(~pair_mask, 0.0)
        else:
            contact_logits = raw_contact_logits
        paratope_logits = self.paratope_head(antibody_fused).squeeze(-1)
        epitope_logits = self.epitope_head(antigen_fused).squeeze(-1)
        if self.adapter_mode == "epiformer_style_same_protocol":
            paratope_logits = paratope_logits + paratope_contact
            # Force node supervision through the partner-specific edge map.
            epitope_logits = epitope_contact
        else:
            paratope_logits = paratope_logits + torch.tanh(
                self.paratope_contact_gate
            ) * paratope_contact
            epitope_logits = epitope_logits + torch.tanh(
                self.epitope_contact_gate
            ) * epitope_contact
        paratope_logits = paratope_logits.masked_fill(~antibody_mask, 0.0)
        epitope_logits = epitope_logits.masked_fill(~antigen_mask, 0.0)
        return {
            "initial_contact_logits": initial_contact_logits,
            "initial_contact_interaction_logits": initial_contact_components[
                "interaction_logits"
            ],
            "intermediate_contact_logits": intermediate_contact_logits,
            "contact_logits": contact_logits,
            "distance_bin_logits": distance_bin_logits,
            "contact_interaction_logits": contact_components["interaction_logits"],
            "antibody_contact_propensity_logits": contact_components[
                "antibody_unary_logits"
            ].squeeze(-1),
            "antigen_contact_propensity_logits": contact_components[
                "antigen_unary_logits"
            ].squeeze(1),
            "pair_candidate_mask": pair_mask,
            "paratope_contact_logits": paratope_contact,
            "epitope_contact_logits": epitope_contact,
            "paratope_logits": paratope_logits,
            "epitope_logits": epitope_logits,
            "antibody_mask": antibody_mask,
            "antigen_mask": antigen_mask,
            "self_attention_details": self_details,
            "cross_attention_details": cross_details,
            "fusion_mode": self.fusion_mode,
            "adapter_mode": self.adapter_mode,
            "contrastive_views": (
                {
                    "antibody_sequence": self.antibody_sequence_contrastive(
                        antibody_sequence_view
                    ),
                    "antibody_geometry": self.antibody_geometry_contrastive(
                        antibody_geometry_view
                    ),
                    "antigen_sequence": self.antigen_sequence_contrastive(
                        antigen_sequence_view
                    ),
                    "antigen_geometry": self.antigen_geometry_contrastive(
                        antigen_geometry_view
                    ),
                }
                if self.adapter_mode == "mipe_style_same_protocol"
                else None
            ),
        }


__all__ = [
    "BidirectionalContactGatedCrossAttention",
    "BidirectionalPairContactHead",
    "ContactGatedCrossAttention",
    "DualFactorizedIntraChainEncoder",
    "FactorizedIntraChainEncoder",
    "FrameContactModel",
    "PairContactHead",
    "PairDistanceBinHead",
    "masked_logmeanexp",
    "masked_noisy_or_probability_logit",
    "masked_learned_softmax_probability_logit",
    "masked_topk_mean_probability_logit",
]
