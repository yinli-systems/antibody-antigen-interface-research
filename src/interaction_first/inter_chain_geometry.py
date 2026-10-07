"""Inter-chain geometry features for antibody-antigen interaction modeling.

This module computes geometric relationships across the antibody-antigen interface,
projecting them into pairwise feature grids and attention biases.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any
import sys

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:
    torch = None
    nn = None

_ModuleBase = nn.Module if nn is not None else object

try:
    from interaction_first.intra_chain_geometry import build_backbone_frames, gaussian_rbf
except ImportError:
    sys.path.append("/Users/alice/.gemini/antigravity/scratch/antibody-antigen-interface-research/src")
    from interaction_first.intra_chain_geometry import build_backbone_frames, gaussian_rbf


def require_torch() -> None:
    """Fail with a focused message when PyTorch is unavailable."""
    if torch is None or nn is None:
        raise RuntimeError("PyTorch is required for inter-chain geometry")


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _optional_positive_float(value: Any, name: str) -> float | None:
    if value is None:
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a positive finite scalar or None")
    if not math.isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{name} must be a positive finite scalar or None")
    return converted


def _safe_normalize(values: Any, eps: float = 1e-8) -> Any:
    denominator = torch.linalg.vector_norm(values, dim=-1, keepdim=True)
    return values / denominator.clamp_min(eps)


@dataclass(frozen=True)
class InterChainRelativeGeometry:
    """Dense antibody-by-antigen relative geometry features."""
    distance: Any
    displacement_local: Any
    relative_rotation: Any
    distance_rbf: Any
    unit_direction: Any
    valid_mask: Any


def build_inter_chain_geometry(
    antibody_backbone: Any,
    antigen_backbone: Any,
    antibody_mask: Any,
    antigen_mask: Any,
    *,
    num_rbf: int = 16,
    distance_max: float = 20.0,
) -> InterChainRelativeGeometry:
    """Construct invariant inter-chain geometry features.

    All returned tensors use an antibody-by-antigen orientation, i.e.,
    shape is [batch, antibody_residue, antigen_residue, ...].
    """
    require_torch()
    num_rbf = _positive_integer(num_rbf, "num_rbf")
    distance_max = _optional_positive_float(distance_max, "distance_max")
    assert distance_max is not None

    if not torch.is_tensor(antibody_backbone) or antibody_backbone.ndim != 4:
        raise ValueError("antibody_backbone must have shape [batch, residue, 3, 3]")
    if not torch.is_tensor(antigen_backbone) or antigen_backbone.ndim != 4:
        raise ValueError("antigen_backbone must have shape [batch, residue, 3, 3]")
    if not torch.is_tensor(antibody_mask) or antibody_mask.dtype != torch.bool:
        raise ValueError("antibody_mask must be boolean")
    if not torch.is_tensor(antigen_mask) or antigen_mask.dtype != torch.bool:
        raise ValueError("antigen_mask must be boolean")

    ab_frames = build_backbone_frames(antibody_backbone, antibody_mask)
    ag_frames = build_backbone_frames(antigen_backbone, antigen_mask)

    ab_origin = ab_frames.origin
    ag_origin = ag_frames.origin
    ab_rotation = ab_frames.rotation
    ag_rotation = ag_frames.rotation

    # displacement of antigen in antibody global frame
    delta_global = ag_origin[:, None, :, :] - ab_origin[:, :, None, :]
    
    # rotation transpose for local projection
    ab_rotation_transpose = ab_rotation.transpose(-1, -2)
    
    # displacement of antigen in antibody local frame
    displacement_local = torch.einsum(
        "bixy,bijy->bijx", ab_rotation_transpose, delta_global
    )
    
    # relative rotation of antigen in antibody frame
    relative_rotation = torch.einsum(
        "bixy,bjyz->bijxz", ab_rotation_transpose, ag_rotation
    )
    
    distance = torch.linalg.vector_norm(displacement_local, dim=-1)
    unit_direction = _safe_normalize(displacement_local)
    
    valid_mask = (
        ab_frames.valid_mask[:, :, None]
        & ag_frames.valid_mask[:, None, :]
        & antibody_mask[:, :, None]
        & antigen_mask[:, None, :]
    )
    
    distance_rbf = gaussian_rbf(
        distance,
        num_bins=num_rbf,
        distance_max=distance_max,
    )
    
    return InterChainRelativeGeometry(
        distance=distance * valid_mask.to(distance.dtype),
        displacement_local=displacement_local * valid_mask[..., None].to(displacement_local.dtype),
        relative_rotation=relative_rotation * valid_mask[..., None, None].to(relative_rotation.dtype),
        distance_rbf=distance_rbf * valid_mask[..., None].to(distance_rbf.dtype),
        unit_direction=unit_direction * valid_mask[..., None].to(unit_direction.dtype),
        valid_mask=valid_mask,
    )


class InterChainGeometryBias(_ModuleBase):
    """Map invariant inter-chain pair features to per-head attention biases.
    
    Architecture: LayerNorm -> Linear -> SiLU -> Linear(-> num_heads)
    """

    def __init__(self, geometry_dim: int, num_heads: int) -> None:
        require_torch()
        super().__init__()
        geometry_dim = _positive_integer(geometry_dim, "geometry_dim")
        num_heads = _positive_integer(num_heads, "num_heads")
        self.network = nn.Sequential(
            nn.LayerNorm(geometry_dim),
            nn.Linear(geometry_dim, geometry_dim),
            nn.SiLU(),
            nn.Linear(geometry_dim, num_heads),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, geometry_features: Any) -> Any:
        # [B, L_ab, L_ag, num_heads] -> [B, num_heads, L_ab, L_ag]
        return self.network(geometry_features).movedim(-1, 1)


class InterChainPairFeatureProjection(_ModuleBase):
    """Project inter-chain geometry features into the pair state dimension.
    
    Architecture: LayerNorm -> Linear -> GELU -> Linear(-> pair_dim)
    """

    def __init__(self, geometry_dim: int, pair_dim: int) -> None:
        require_torch()
        super().__init__()
        geometry_dim = _positive_integer(geometry_dim, "geometry_dim")
        pair_dim = _positive_integer(pair_dim, "pair_dim")
        self.network = nn.Sequential(
            nn.LayerNorm(geometry_dim),
            nn.Linear(geometry_dim, pair_dim),
            nn.GELU(),
            nn.Linear(pair_dim, pair_dim),
        )

    def forward(self, geometry_features: Any) -> Any:
        return self.network(geometry_features)


__all__ = [
    "InterChainGeometryBias",
    "InterChainPairFeatureProjection",
    "InterChainRelativeGeometry",
    "build_inter_chain_geometry",
]
