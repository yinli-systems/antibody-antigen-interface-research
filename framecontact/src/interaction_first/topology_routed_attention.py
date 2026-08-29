"""Adaptive, development-only topology-routed intra-chain attention.

This module is deliberately separate from the hash-locked confirmatory
implementation.  It retains one dense content-attention stream and adds
SE(3)-invariant local-pose score increments at fixed cumulative spatial scales.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from .framecontact import FrameContactModel
from .intra_chain_geometry import (
    IntraChainRelativeGeometry,
    apply_sequence_rope,
    build_backbone_frames,
    build_intra_chain_relative_geometry,
)


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


class TopologyRoutedMultiScaleHeadBias(nn.Module):
    """Route radial, direction, and rotation evidence by radius and head."""

    COMPONENT_NAMES = ("radial", "direction", "rotation")
    RELATION_NAMES = ("radius_6a", "radius_8a", "radius_12a")

    def __init__(
        self,
        geometry_dim: int,
        num_heads: int,
        *,
        num_rbf: int,
        radii: tuple[float, ...] = (6.0, 8.0, 12.0),
    ) -> None:
        super().__init__()
        geometry_dim = _positive_integer(geometry_dim, "geometry_dim")
        self.num_heads = _positive_integer(num_heads, "num_heads")
        self.num_rbf = _positive_integer(num_rbf, "num_rbf")
        if geometry_dim < self.num_rbf + 12:
            raise ValueError(
                "geometry_dim must contain RBF, direction, and rotation channels"
            )
        if (
            len(radii) != len(self.RELATION_NAMES)
            or any(
                not math.isfinite(float(radius)) or float(radius) <= 0
                for radius in radii
            )
            or any(left >= right for left, right in zip(radii, radii[1:]))
        ):
            raise ValueError(
                "radii must contain three strictly increasing positive values"
            )
        self.radii = tuple(float(radius) for radius in radii)
        component_dims = (self.num_rbf, 3, 9)
        self.component_networks = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.LayerNorm(component_dim),
                    nn.Linear(component_dim, max(component_dim, self.num_heads)),
                    nn.SiLU(),
                    nn.Linear(max(component_dim, self.num_heads), self.num_heads),
                )
                for name, component_dim in zip(
                    self.COMPONENT_NAMES, component_dims, strict=True
                )
            }
        )
        self.gate_unconstrained = nn.Parameter(
            torch.zeros(
                len(self.RELATION_NAMES),
                len(self.COMPONENT_NAMES),
                self.num_heads,
            )
        )

    def _feature_slices(
        self, geometry_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        radial_end = self.num_rbf
        direction_end = radial_end + 3
        rotation_end = direction_end + 9
        return (
            geometry_features[..., :radial_end],
            geometry_features[..., radial_end:direction_end],
            geometry_features[..., direction_end:rotation_end],
        )

    def relation_component_biases(
        self,
        geometry_features: torch.Tensor,
        distance: torch.Tensor,
        geometry_valid_mask: torch.Tensor,
    ) -> dict[str, dict[str, torch.Tensor]]:
        pair_shape = tuple(geometry_features.shape[:-1])
        if tuple(distance.shape) != pair_shape:
            raise ValueError("distance must align with pair features")
        if (
            geometry_valid_mask.dtype != torch.bool
            or tuple(geometry_valid_mask.shape) != pair_shape
        ):
            raise ValueError("geometry_valid_mask must be boolean and align")
        gates = torch.tanh(self.gate_unconstrained)
        component_values = {
            name: self.component_networks[name](values)
            for name, values in zip(
                self.COMPONENT_NAMES,
                self._feature_slices(geometry_features),
                strict=True,
            )
        }
        output: dict[str, dict[str, torch.Tensor]] = {}
        for relation_index, (relation_name, radius) in enumerate(
            zip(self.RELATION_NAMES, self.radii, strict=True)
        ):
            relation_mask = geometry_valid_mask & (distance <= radius)
            valid = relation_mask[..., None].to(geometry_features.dtype)
            output[relation_name] = {
                component_name: (
                    component_values[component_name]
                    * gates[relation_index, component_index][None, None, None, :]
                    * valid
                ).movedim(-1, 1)
                for component_index, component_name in enumerate(
                    self.COMPONENT_NAMES
                )
            }
        return output

    def forward(
        self,
        geometry_features: torch.Tensor,
        distance: torch.Tensor,
        geometry_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        contributions = self.relation_component_biases(
            geometry_features,
            distance,
            geometry_valid_mask,
        )
        return sum(
            component
            for relation in contributions.values()
            for component in relation.values()
        )


class TopologyRoutedSelfAttentionBlock(nn.Module):
    """Pre-norm dense self-attention with zero-gated local-pose routing."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        *,
        num_rbf: int = 16,
        ffn_expansion: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = _positive_integer(d_model, "d_model")
        self.num_heads = _positive_integer(num_heads, "num_heads")
        if self.d_model % self.num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        self.head_dim = self.d_model // self.num_heads
        if self.head_dim % 2:
            raise ValueError("Per-head channels must be even for sequence RoPE")
        self.attention_norm = nn.LayerNorm(self.d_model)
        self.qkv = nn.Linear(self.d_model, 3 * self.d_model)
        self.output_projection = nn.Linear(self.d_model, self.d_model)
        self.geometry_bias = TopologyRoutedMultiScaleHeadBias(
            num_rbf + 3 + 9 + 2,
            self.num_heads,
            num_rbf=num_rbf,
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.residual_dropout = nn.Dropout(dropout)
        hidden_dim = _positive_integer(ffn_expansion, "ffn_expansion") * self.d_model
        self.ffn_norm = nn.LayerNorm(self.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(self.d_model, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.d_model),
        )

    def forward(
        self,
        features: torch.Tensor,
        geometry: IntraChainRelativeGeometry,
        residue_mask: torch.Tensor,
        *,
        positions: torch.Tensor | None = None,
        routing_mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any] | None]:
        batch_size, residue_count, _ = features.shape
        normalized = self.attention_norm(features)
        qkv = self.qkv(normalized).view(
            batch_size,
            residue_count,
            3,
            self.num_heads,
            self.head_dim,
        )
        query, key, value = (
            qkv[:, :, index].transpose(1, 2) for index in range(3)
        )
        query = apply_sequence_rope(query, positions)
        key = apply_sequence_rope(key, positions)
        content_logits = (
            torch.einsum("bhid,bhjd->bhij", query, key)
            / math.sqrt(self.head_dim)
        )
        valid_geometry = geometry.geometry_valid_mask
        if routing_mask is not None:
            valid_geometry = valid_geometry & routing_mask
        geometry_features = geometry.features.to(dtype=features.dtype)
        distance = geometry.distance.to(dtype=features.dtype)
        geometry_bias = self.geometry_bias(
            geometry_features,
            distance,
            valid_geometry,
        )
        attention_mask = geometry.neighbor_mask[:, None, :, :]
        logits = (content_logits + geometry_bias).masked_fill(
            ~attention_mask, torch.finfo(content_logits.dtype).min
        )
        attention = torch.softmax(logits, dim=-1)
        attention = attention * residue_mask[:, None, :, None].to(
            attention.dtype
        )
        context = torch.einsum(
            "bhij,bhjd->bhid", self.attention_dropout(attention), value
        )
        context = context.transpose(1, 2).reshape(
            batch_size, residue_count, self.d_model
        )
        features = features + self.residual_dropout(
            self.output_projection(context)
        )
        features = features + self.residual_dropout(
            self.ffn(self.ffn_norm(features))
        )
        features = features.masked_fill(~residue_mask[..., None], 0.0)
        if not return_attention:
            return features, None
        content_only_logits = content_logits.masked_fill(
            ~attention_mask, torch.finfo(content_logits.dtype).min
        )
        content_only_attention = torch.softmax(content_only_logits, dim=-1)
        content_only_attention = content_only_attention * residue_mask[
            :, None, :, None
        ].to(content_only_attention.dtype)
        return features, {
            "weights": attention,
            "content_only_weights": content_only_attention,
            "geometry_bias": geometry_bias,
            "geometry_bias_components": (
                self.geometry_bias.relation_component_biases(
                    geometry_features,
                    distance,
                    valid_geometry,
                )
            ),
            "neighbor_mask": geometry.neighbor_mask,
            "geometry_valid_mask": valid_geometry,
        }


class TopologyRoutedIntraChainEncoder(nn.Module):
    """Stack a single dense QKV path with topology-routed score increments."""

    geometry_fusion_mode = "shared_qkv_multiscale"
    dense_encoder = None
    geometry_residual_gate = None

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        *,
        num_rbf: int = 16,
        distance_max: float = 20.0,
        sequence_window: int = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = _positive_integer(d_model, "d_model")
        self.num_rbf = _positive_integer(num_rbf, "num_rbf")
        self.distance_max = float(distance_max)
        self.sequence_window = _positive_integer(
            sequence_window, "sequence_window"
        )
        self.blocks = nn.ModuleList(
            TopologyRoutedSelfAttentionBlock(
                self.d_model,
                num_heads,
                num_rbf=self.num_rbf,
                dropout=dropout,
            )
            for _ in range(_positive_integer(num_layers, "num_layers"))
        )

    def forward(
        self,
        features: torch.Tensor,
        residue_mask: torch.Tensor,
        backbone: torch.Tensor,
        *,
        chain_ids: torch.Tensor,
        positions: torch.Tensor | None = None,
        neighbor_mask_override: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any] | None]:
        if (
            features.ndim != 3
            or int(features.shape[-1]) != self.d_model
            or residue_mask.dtype != torch.bool
            or tuple(residue_mask.shape) != tuple(features.shape[:2])
            or tuple(backbone.shape) != (*features.shape[:2], 3, 3)
            or tuple(chain_ids.shape) != tuple(features.shape[:2])
        ):
            raise ValueError("Topology-routed residue stream shapes are invalid")
        frames = build_backbone_frames(backbone, residue_mask)
        geometry = build_intra_chain_relative_geometry(
            frames,
            residue_mask,
            chain_ids=chain_ids,
            geometry_feature_mode="full_pose",
            attention_graph_mode="dense",
            num_rbf=self.num_rbf,
            distance_max=self.distance_max,
            neighbor_k=None,
            neighbor_radius=None,
            sequence_window=self.sequence_window,
        )
        routing_mask = None
        if neighbor_mask_override is not None:
            if (
                neighbor_mask_override.dtype != torch.bool
                or tuple(neighbor_mask_override.shape)
                != tuple(geometry.neighbor_mask.shape)
            ):
                raise ValueError(
                    "neighbor_mask_override must be boolean and align with pairs"
                )
            if bool((neighbor_mask_override & ~geometry.neighbor_mask).any()):
                raise ValueError("routing override contains forbidden chain pairs")
            routing_mask = neighbor_mask_override
        layer_details = []
        for block in self.blocks:
            features, details = block(
                features,
                geometry,
                residue_mask,
                positions=positions,
                routing_mask=routing_mask,
                return_attention=return_attention,
            )
            if details is not None:
                layer_details.append(details)
        if not return_attention:
            return features, None
        details = {
            "layers": tuple(layer_details),
            "frame_valid_mask": frames.valid_mask,
            "neighbor_mask": geometry.neighbor_mask,
            "distance": geometry.distance,
            "attention_graph_mode": "dense",
            "geometry_fusion_mode": self.geometry_fusion_mode,
        }
        return features, {
            "dense": details,
            "geometry": details,
            "geometry_residual_gate": torch.ones(
                (), dtype=features.dtype, device=features.device
            ),
            "geometry_fusion_mode": self.geometry_fusion_mode,
        }


class DualTopologyRoutedIntraChainEncoder(nn.Module):
    """Two unshared topology-routed antibody and antigen encoders."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.antibody_encoder = TopologyRoutedIntraChainEncoder(*args, **kwargs)
        self.antigen_encoder = TopologyRoutedIntraChainEncoder(*args, **kwargs)

    def forward(
        self,
        antibody_features: torch.Tensor,
        antigen_features: torch.Tensor,
        antibody_mask: torch.Tensor,
        antigen_mask: torch.Tensor,
        antibody_backbone: torch.Tensor,
        antigen_backbone: torch.Tensor,
        *,
        antibody_chain_ids: torch.Tensor,
        antigen_chain_ids: torch.Tensor,
        antibody_neighbor_mask_override: torch.Tensor | None = None,
        antigen_neighbor_mask_override: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any] | None]:
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


class TopologyRoutedFrameContactModel(FrameContactModel):
    """FrameContact late-fusion model with an isolated routed self encoder."""

    def __init__(
        self,
        *,
        antibody_input_dim: int = 1280,
        antigen_input_dim: int = 1280,
        d_model: int = 128,
        num_heads: int = 8,
        num_self_layers: int = 2,
        geometry_feature_mode: str = "full_pose",
        geometry_bias_mode: str = "topology_routed_multiscale",
        ipa_num_points: int = 4,
        local_pose_rotary_channels: int | None = None,
        geometry_attention_graph_mode: str = "spatial",
        num_rbf: int = 16,
        neighbor_k: int | None = 32,
        neighbor_radius: float | None = 16.0,
        sequence_window: int = 32,
        pair_dim: int | None = None,
        dropout: float = 0.1,
        fusion_mode: str = "late",
    ) -> None:
        del ipa_num_points, local_pose_rotary_channels
        del geometry_attention_graph_mode, neighbor_k, neighbor_radius
        if geometry_feature_mode != "full_pose":
            raise ValueError("Topology routing requires full_pose features")
        if geometry_bias_mode != "topology_routed_multiscale":
            raise ValueError(
                "Topology routing requires topology_routed_multiscale bias"
            )
        if fusion_mode != "late":
            raise ValueError("Topology-routed pilot supports late fusion only")
        super().__init__(
            antibody_input_dim=antibody_input_dim,
            antigen_input_dim=antigen_input_dim,
            d_model=d_model,
            num_heads=num_heads,
            num_self_layers=num_self_layers,
            geometry_feature_mode="full_pose",
            geometry_bias_mode="pair_mlp",
            num_rbf=num_rbf,
            neighbor_k=32,
            neighbor_radius=16.0,
            sequence_window=sequence_window,
            pair_dim=pair_dim,
            dropout=dropout,
            fusion_mode="late",
        )
        self.self_encoder = DualTopologyRoutedIntraChainEncoder(
            d_model,
            num_heads,
            num_self_layers,
            num_rbf=num_rbf,
            sequence_window=sequence_window,
            dropout=dropout,
        )


__all__ = [
    "DualTopologyRoutedIntraChainEncoder",
    "TopologyRoutedFrameContactModel",
    "TopologyRoutedIntraChainEncoder",
    "TopologyRoutedMultiScaleHeadBias",
    "TopologyRoutedSelfAttentionBlock",
]
