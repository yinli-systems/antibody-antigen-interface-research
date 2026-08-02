"""SE(3)-invariant intra-chain geometry for residue self-attention.

This module is the structure-aware *monomer* stage.  Antibody and antigen are
encoded by separate parameter sets, and every geometric quantity is computed
inside one residue stream.  Consequently, the two input structures may use
unrelated global coordinate frames.  Inter-chain geometric attention is
deliberately outside the scope of this module.

Backbone tensors use atom order ``(N, CA, C)`` and shape
``[batch, residue, atom, xyz]``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional
except ModuleNotFoundError:
    torch = None
    nn = None
    functional = None


_ModuleBase = nn.Module if nn is not None else object


def require_torch() -> None:
    """Fail with a focused message when PyTorch is unavailable."""
    if torch is None or nn is None:
        raise RuntimeError("PyTorch is required for intra-chain geometry attention")


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _optional_positive_integer(value: Any, name: str) -> int | None:
    if value is None:
        return None
    return _positive_integer(value, name)


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


def _dropout_probability(value: Any) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        raise ValueError("dropout must be a finite scalar in [0, 1)")
    if not math.isfinite(converted) or not 0.0 <= converted < 1.0:
        raise ValueError("dropout must be a finite scalar in [0, 1)")
    return converted


def _geometry_feature_dimension(mode: str, num_rbf: int) -> int:
    allowed = {"none", "distance", "distance_direction", "full_pose"}
    if mode not in allowed:
        raise ValueError(
            "geometry_feature_mode must be 'none', 'distance', "
            "'distance_direction', or 'full_pose'"
        )
    # Keep one fixed width across ablations so model parameter counts are equal.
    return num_rbf + 3 + 9 + 2


def _attention_graph_mode(value: str) -> str:
    allowed = {"dense", "sequence", "spatial"}
    if value not in allowed:
        raise ValueError(
            "attention_graph_mode must be 'dense', 'sequence', or 'spatial'"
        )
    return value


def _attention_kernel(value: str) -> str:
    allowed = {"reference", "sdpa"}
    if value not in allowed:
        raise ValueError("attention_kernel must be 'reference' or 'sdpa'")
    return value


def _geometry_bias_mode(value: str) -> str:
    allowed = {
        "pair_mlp",
        "invariant_points",
        "local_pose_multibasis",
        "local_pose_rotary",
    }
    if value not in allowed:
        raise ValueError(
            "geometry_bias_mode must be 'pair_mlp', 'invariant_points', "
            "'local_pose_multibasis', or 'local_pose_rotary'"
        )
    return value


def _safe_normalize(values: Any, eps: float = 1e-8) -> Any:
    denominator = torch.linalg.vector_norm(values, dim=-1, keepdim=True)
    return values / denominator.clamp_min(eps)


def _validate_stream_inputs(
    features: Any,
    residue_mask: Any,
    backbone: Any,
    chain_ids: Any | None,
    expected_channels: int,
) -> None:
    if not torch.is_tensor(features) or features.ndim != 3:
        raise ValueError("features must have shape [batch, residue, channels]")
    if not torch.is_floating_point(features):
        raise ValueError("features must use a floating-point dtype")
    if int(features.shape[-1]) != expected_channels:
        raise ValueError(f"features must have {expected_channels} channels")
    if not bool(torch.isfinite(features).all()):
        raise ValueError("features must contain only finite values")
    if (
        not torch.is_tensor(residue_mask)
        or residue_mask.dtype != torch.bool
        or tuple(residue_mask.shape) != tuple(features.shape[:2])
    ):
        raise ValueError("residue_mask must be boolean and align with features")
    if residue_mask.device != features.device:
        raise ValueError("residue_mask and features must share a device")
    if bool((residue_mask.sum(dim=1) == 0).any()):
        raise ValueError("Every item must contain at least one valid residue")
    expected_backbone_shape = (*features.shape[:2], 3, 3)
    if (
        not torch.is_tensor(backbone)
        or not torch.is_floating_point(backbone)
        or tuple(backbone.shape) != tuple(expected_backbone_shape)
    ):
        raise ValueError("backbone must have shape [batch, residue, 3, 3]")
    if backbone.device != features.device:
        raise ValueError("backbone and features must share a device")
    if chain_ids is not None:
        if not torch.is_tensor(chain_ids) or tuple(chain_ids.shape) != tuple(
            features.shape[:2]
        ):
            raise ValueError("chain_ids must align with the residue axes")
        if torch.is_floating_point(chain_ids) or chain_ids.device != features.device:
            raise ValueError("chain_ids must be an integer tensor on the feature device")


@dataclass(frozen=True)
class BackboneFrames:
    """Right-handed per-residue frames with ``CA`` as the origin."""

    rotation: Any
    origin: Any
    valid_mask: Any


@dataclass(frozen=True)
class IntraChainRelativeGeometry:
    """Dense ordered-pair geometry plus the permitted attention graph."""

    features: Any
    distance: Any
    displacement_local: Any
    relative_rotation: Any
    geometry_valid_mask: Any
    neighbor_mask: Any


def build_backbone_frames(backbone: Any, residue_mask: Any) -> BackboneFrames:
    """Build robust right-handed ``N-CA-C`` frames.

    Degenerate or non-finite residues receive an identity frame, zero origin,
    and ``valid_mask=False``.  Their sequence representation can still pass
    through a self edge, but they cannot contribute a learned geometry bias.
    """
    require_torch()
    if not torch.is_tensor(backbone) or backbone.ndim != 4:
        raise ValueError("backbone must have shape [batch, residue, 3, 3]")
    if tuple(backbone.shape[-2:]) != (3, 3):
        raise ValueError("backbone atom order must be N, CA, C with xyz coordinates")
    if (
        not torch.is_tensor(residue_mask)
        or residue_mask.dtype != torch.bool
        or tuple(residue_mask.shape) != tuple(backbone.shape[:2])
    ):
        raise ValueError("residue_mask must be boolean and align with backbone")

    finite = torch.isfinite(backbone).all(dim=-1).all(dim=-1)
    safe_backbone = torch.where(torch.isfinite(backbone), backbone, 0.0)
    n_atom, ca_atom, c_atom = safe_backbone.unbind(dim=-2)
    x_raw = c_atom - ca_atom
    n_raw = n_atom - ca_atom
    x_axis = _safe_normalize(x_raw)
    y_raw = n_raw - (n_raw * x_axis).sum(dim=-1, keepdim=True) * x_axis
    y_axis = _safe_normalize(y_raw)
    z_axis = _safe_normalize(torch.linalg.cross(x_axis, y_axis, dim=-1))
    y_axis = _safe_normalize(torch.linalg.cross(z_axis, x_axis, dim=-1))
    rotation = torch.stack((x_axis, y_axis, z_axis), dim=-1)
    valid = (
        residue_mask
        & finite
        & (torch.linalg.vector_norm(x_raw, dim=-1) > 1e-5)
        & (torch.linalg.vector_norm(y_raw, dim=-1) > 1e-5)
    )
    identity = torch.eye(3, dtype=backbone.dtype, device=backbone.device)
    rotation = torch.where(valid[..., None, None], rotation, identity)
    origin = torch.where(valid[..., None], ca_atom, torch.zeros_like(ca_atom))
    return BackboneFrames(rotation=rotation, origin=origin, valid_mask=valid)


def gaussian_rbf(
    distances: Any,
    *,
    num_bins: int = 16,
    distance_max: float = 20.0,
) -> Any:
    """Expand distances in Angstrom into Gaussian radial basis features."""
    require_torch()
    num_bins = _positive_integer(num_bins, "num_bins")
    distance_max = _optional_positive_float(distance_max, "distance_max")
    assert distance_max is not None
    centers = torch.linspace(
        0.0,
        distance_max,
        num_bins,
        dtype=distances.dtype,
        device=distances.device,
    )
    width = distance_max / max(num_bins - 1, 1)
    return torch.exp(-((distances[..., None] - centers) / width) ** 2)


def build_intra_chain_relative_geometry(
    frames: BackboneFrames,
    residue_mask: Any,
    *,
    chain_ids: Any | None = None,
    geometry_feature_mode: str = "full_pose",
    attention_graph_mode: str = "spatial",
    num_rbf: int = 16,
    distance_max: float = 20.0,
    neighbor_k: int | None = 32,
    neighbor_radius: float | None = None,
    sequence_window: int = 32,
) -> IntraChainRelativeGeometry:
    """Construct invariant pair features and an intra-chain spatial graph.

    The attention graph is an explicit ablation variable. ``dense`` permits all
    same-chain residue pairs and never uses coordinates for edge selection.
    ``sequence`` permits a same-chain sequence window. ``spatial`` applies the
    radius filter first and then retains up to ``neighbor_k`` nearest residues.
    A self edge is always present for every non-padding residue, including a
    residue with a missing/degenerate backbone frame.
    """
    require_torch()
    neighbor_k = _optional_positive_integer(neighbor_k, "neighbor_k")
    neighbor_radius = _optional_positive_float(neighbor_radius, "neighbor_radius")
    sequence_window = _positive_integer(sequence_window, "sequence_window")
    num_rbf = _positive_integer(num_rbf, "num_rbf")
    _geometry_feature_dimension(geometry_feature_mode, num_rbf)
    attention_graph_mode = _attention_graph_mode(attention_graph_mode)
    distance_max = _optional_positive_float(distance_max, "distance_max")
    assert distance_max is not None

    rotation = frames.rotation
    origin = frames.origin
    if rotation.ndim != 4 or tuple(rotation.shape[-2:]) != (3, 3):
        raise ValueError("frames.rotation must have shape [batch, residue, 3, 3]")
    if tuple(origin.shape) != (*rotation.shape[:2], 3):
        raise ValueError("frames.origin must align with frames.rotation")
    if tuple(residue_mask.shape) != tuple(rotation.shape[:2]):
        raise ValueError("residue_mask must align with frames")

    batch_size, residue_count = residue_mask.shape
    if chain_ids is None:
        same_chain = torch.ones(
            batch_size,
            residue_count,
            residue_count,
            dtype=torch.bool,
            device=origin.device,
        )
    else:
        if tuple(chain_ids.shape) != tuple(residue_mask.shape):
            raise ValueError("chain_ids must align with residue_mask")
        same_chain = chain_ids[:, :, None] == chain_ids[:, None, :]

    delta_global = origin[:, None, :, :] - origin[:, :, None, :]
    rotation_transpose = rotation.transpose(-1, -2)
    displacement_local = torch.einsum(
        "bixy,bijy->bijx", rotation_transpose, delta_global
    )
    relative_rotation = torch.einsum(
        "bixy,bjyz->bijxz", rotation_transpose, rotation
    )
    distance = torch.linalg.vector_norm(displacement_local, dim=-1)
    unit_direction = _safe_normalize(displacement_local)
    geometry_valid = (
        frames.valid_mask[:, :, None]
        & frames.valid_mask[:, None, :]
        & residue_mask[:, :, None]
        & residue_mask[:, None, :]
        & same_chain
    )

    positions = torch.arange(residue_count, dtype=origin.dtype, device=origin.device)
    sequence_delta = positions[None, :] - positions[:, None]
    sequence_signed = (sequence_delta / 32.0).clamp(-1.0, 1.0)
    sequence_absolute = (sequence_delta.abs() / 32.0).clamp(0.0, 1.0)
    sequence_signed = sequence_signed.expand(batch_size, -1, -1)
    sequence_absolute = sequence_absolute.expand(batch_size, -1, -1)
    distance_features = gaussian_rbf(
        distance,
        num_bins=num_rbf,
        distance_max=distance_max,
    )
    full_features = torch.cat(
        (
            distance_features,
            unit_direction,
            relative_rotation.flatten(start_dim=-2),
            sequence_signed[..., None],
            sequence_absolute[..., None],
        ),
        dim=-1,
    )
    if geometry_feature_mode == "none":
        features = torch.zeros_like(full_features)
    elif geometry_feature_mode == "distance":
        channel_mask = torch.zeros(
            full_features.shape[-1],
            dtype=full_features.dtype,
            device=full_features.device,
        )
        channel_mask[:num_rbf] = 1.0
        features = full_features * channel_mask
    elif geometry_feature_mode == "distance_direction":
        channel_mask = torch.zeros(
            full_features.shape[-1],
            dtype=full_features.dtype,
            device=full_features.device,
        )
        channel_mask[: num_rbf + 3] = 1.0
        features = full_features * channel_mask
    else:
        features = full_features
    features = features * geometry_valid[..., None].to(features.dtype)

    permitted_pairs = (
        residue_mask[:, :, None]
        & residue_mask[:, None, :]
        & same_chain
    )
    if attention_graph_mode == "dense":
        selected_neighbors = permitted_pairs
    elif attention_graph_mode == "sequence":
        selected_neighbors = permitted_pairs & (
            sequence_delta.abs()[None, :, :] <= sequence_window
        )
    else:
        spatial_candidates = geometry_valid
        if neighbor_radius is not None:
            spatial_candidates = spatial_candidates & (distance <= neighbor_radius)
        if neighbor_k is not None:
            selected_count = min(neighbor_k, residue_count)
            candidate_distances = distance.masked_fill(~spatial_candidates, torch.inf)
            nearest_distance, nearest_index = torch.topk(
                candidate_distances,
                k=selected_count,
                dim=-1,
                largest=False,
                sorted=False,
            )
            selected_neighbors = torch.zeros_like(spatial_candidates)
            selected_neighbors.scatter_(
                dim=-1,
                index=nearest_index,
                src=torch.isfinite(nearest_distance),
            )
        else:
            selected_neighbors = spatial_candidates

    identity = torch.eye(residue_count, dtype=torch.bool, device=origin.device)
    self_edges = identity.unsqueeze(0) & residue_mask[:, :, None]
    neighbor_mask = (selected_neighbors | self_edges) & same_chain
    neighbor_mask = (
        neighbor_mask
        & residue_mask[:, :, None]
        & residue_mask[:, None, :]
    )
    return IntraChainRelativeGeometry(
        features=features,
        distance=distance,
        displacement_local=displacement_local,
        relative_rotation=relative_rotation,
        geometry_valid_mask=geometry_valid,
        neighbor_mask=neighbor_mask,
    )


def apply_sequence_rope(values: Any, positions: Any | None = None) -> Any:
    """Apply ordinary 1D sequence RoPE to ``[B, H, L, D]`` values."""
    require_torch()
    if values.ndim != 4 or int(values.shape[-1]) % 2:
        raise ValueError("RoPE values must have shape [B, H, L, even_channels]")
    batch_size, _, residue_count, channels = values.shape
    if positions is None:
        positions = torch.arange(residue_count, device=values.device)
    if not torch.is_tensor(positions):
        raise TypeError("positions must be a PyTorch tensor")
    if positions.ndim == 1:
        if int(positions.shape[0]) != residue_count:
            raise ValueError("positions must align with the residue axis")
        positions = positions.unsqueeze(0).expand(batch_size, -1)
    elif tuple(positions.shape) != (batch_size, residue_count):
        raise ValueError("positions must have shape [residue] or [batch, residue]")
    if positions.device != values.device:
        raise ValueError("positions and values must share a device")

    frequency = 10_000.0 ** (
        -torch.arange(0, channels, 2, dtype=values.dtype, device=values.device)
        / channels
    )
    angle = positions.to(values.dtype)[:, None, :, None] * frequency
    real, imaginary = values[..., 0::2], values[..., 1::2]
    cosine, sine = angle.cos(), angle.sin()
    return torch.stack(
        (
            real * cosine - imaginary * sine,
            real * sine + imaginary * cosine,
        ),
        dim=-1,
    ).flatten(start_dim=-2)


class GeometryHeadBias(_ModuleBase):
    """Map invariant pair features to one zero-initialized bias per head."""

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
        return self.network(geometry_features).movedim(-1, 1)


class InvariantPointHeadBias(_ModuleBase):
    """IPA-style bias from learned points in fixed residue-local frames.

    This is a matched self-attention baseline inspired by invariant point
    attention, not a reproduction of AlphaFold's complete IPA block.
    """

    def __init__(
        self,
        num_heads: int,
        *,
        num_points: int = 4,
        distance_scale: float = 10.0,
    ) -> None:
        require_torch()
        super().__init__()
        num_heads = _positive_integer(num_heads, "num_heads")
        num_points = _positive_integer(num_points, "num_points")
        distance_scale = _optional_positive_float(distance_scale, "distance_scale")
        assert distance_scale is not None
        self.num_heads = num_heads
        self.num_points = num_points
        self.distance_scale = distance_scale
        self.query_points = nn.Parameter(torch.empty(num_heads, num_points, 3))
        self.key_points = nn.Parameter(torch.empty(num_heads, num_points, 3))
        self.head_weight_unconstrained = nn.Parameter(
            torch.full((num_heads,), math.log(math.expm1(1.0)))
        )
        nn.init.normal_(self.query_points, std=0.1)
        nn.init.normal_(self.key_points, std=0.1)

    def forward(self, geometry: IntraChainRelativeGeometry) -> Any:
        # Evaluate mean_p ||q_hp - (R_ij k_hp + d_ij)||^2 without
        # materializing [B, L, L, H, P, 3].  The expanded norm is algebraically
        # identical and keeps the largest intermediate at [B, L, L, H].
        # This matters for long antigens: the naive six-dimensional delta can
        # consume multiple GiB even at batch size one.
        query_mean = self.query_points.mean(dim=1)
        key_mean = self.key_points.mean(dim=1)
        query_norm = self.query_points.square().sum(dim=-1).mean(dim=-1)
        key_norm = self.key_points.square().sum(dim=-1).mean(dim=-1)
        displacement = geometry.displacement_local
        relative_rotation = geometry.relative_rotation
        displacement_norm = displacement.square().sum(dim=-1)
        displacement_dot_rotated_key = torch.einsum(
            "bijx,bijxy,hy->bijh",
            displacement,
            relative_rotation,
            key_mean,
        )
        query_dot_rotated_key = torch.einsum(
            "hpx,bijxy,hpy->bijh",
            self.query_points,
            relative_rotation,
            self.key_points,
        ) / self.num_points
        query_dot_displacement = torch.einsum(
            "hx,bijx->bijh", query_mean, displacement
        )
        mean_squared_distance = (
            query_norm[None, None, None, :]
            + key_norm[None, None, None, :]
            + displacement_norm[..., None]
            + 2.0 * displacement_dot_rotated_key
            - 2.0 * query_dot_rotated_key
            - 2.0 * query_dot_displacement
        ).clamp_min(0.0)
        head_weight = torch.nn.functional.softplus(
            self.head_weight_unconstrained
        )
        bias = (
            -0.5
            * mean_squared_distance
            * head_weight[None, None, None, :]
            / (self.distance_scale**2)
        )
        bias = bias * geometry.geometry_valid_mask[..., None].to(bias.dtype)
        return bias.movedim(-1, 1)


class LocalPoseMultiBasisHeadBias(_ModuleBase):
    """Interpretable radial, direction, and rotation head-bias mixture.

    Each invariant local-pose basis has a separate predictor and a bounded
    per-head gate.  Gates start at exactly zero, so adding this module preserves
    the content-only attention function while still giving the gates a direct
    first-step gradient.  Sequence-separation channels are deliberately not
    consumed: the module isolates the contribution of three-dimensional pose.
    """

    _COMPONENT_NAMES = ("radial", "direction", "rotation")

    def __init__(self, geometry_dim: int, num_heads: int, *, num_rbf: int) -> None:
        require_torch()
        super().__init__()
        geometry_dim = _positive_integer(geometry_dim, "geometry_dim")
        num_heads = _positive_integer(num_heads, "num_heads")
        num_rbf = _positive_integer(num_rbf, "num_rbf")
        if geometry_dim < num_rbf + 12:
            raise ValueError(
                "geometry_dim must contain RBF, local-direction, and rotation channels"
            )
        self.num_heads = num_heads
        self.num_rbf = num_rbf
        component_dims = (num_rbf, 3, 9)
        self.component_networks = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.LayerNorm(component_dim),
                    nn.Linear(component_dim, max(component_dim, num_heads)),
                    nn.SiLU(),
                    nn.Linear(max(component_dim, num_heads), num_heads),
                )
                for name, component_dim in zip(
                    self._COMPONENT_NAMES, component_dims, strict=True
                )
            }
        )
        self.gate_unconstrained = nn.Parameter(
            torch.zeros(len(self._COMPONENT_NAMES), num_heads)
        )

    def _feature_slices(self, geometry_features: Any) -> tuple[Any, Any, Any]:
        radial_end = self.num_rbf
        direction_end = radial_end + 3
        rotation_end = direction_end + 9
        return (
            geometry_features[..., :radial_end],
            geometry_features[..., radial_end:direction_end],
            geometry_features[..., direction_end:rotation_end],
        )

    def component_biases(
        self,
        geometry_features: Any,
        geometry_valid_mask: Any,
    ) -> dict[str, Any]:
        if tuple(geometry_valid_mask.shape) != tuple(geometry_features.shape[:-1]):
            raise ValueError("geometry_valid_mask must align with pair features")
        gates = torch.tanh(self.gate_unconstrained)
        valid = geometry_valid_mask[..., None].to(geometry_features.dtype)
        components: dict[str, Any] = {}
        for component_index, (name, values) in enumerate(
            zip(
                self._COMPONENT_NAMES,
                self._feature_slices(geometry_features),
                strict=True,
            )
        ):
            raw = self.component_networks[name](values)
            gated = raw * gates[component_index][None, None, None, :] * valid
            components[name] = gated.movedim(-1, 1)
        return components

    def forward(self, geometry_features: Any, geometry_valid_mask: Any) -> Any:
        components = self.component_biases(geometry_features, geometry_valid_mask)
        return sum(components.values())


class PairConditionedLocalPoseRotaryBias(_ModuleBase):
    """Pair-conditioned rotary score from SE(3)-invariant local pose.

    Unlike ordinary token-factorized RoPE, the phase is predicted for an
    ordered residue pair from its local-frame relative geometry.  It rotates a
    bounded subset of Q/K complex channel pairs and contributes an additive
    score.  A zero-initialized per-head gate makes the initial attention exactly
    content-only.  This is therefore described as a *local-pose rotary score*,
    not as a literal three-dimensional extension of standard RoPE.
    """

    def __init__(
        self,
        geometry_dim: int,
        num_heads: int,
        head_dim: int,
        *,
        rotary_channels: int | None = None,
    ) -> None:
        require_torch()
        super().__init__()
        geometry_dim = _positive_integer(geometry_dim, "geometry_dim")
        self.num_heads = _positive_integer(num_heads, "num_heads")
        head_dim = _positive_integer(head_dim, "head_dim")
        if head_dim % 2:
            raise ValueError("head_dim must be even")
        if rotary_channels is None:
            rotary_channels = min(16, head_dim)
        rotary_channels = _positive_integer(rotary_channels, "rotary_channels")
        if rotary_channels % 2 or rotary_channels > head_dim:
            raise ValueError(
                "rotary_channels must be even and no larger than head_dim"
            )
        self.rotary_channels = rotary_channels
        self.rotary_pairs = rotary_channels // 2
        hidden_dim = max(geometry_dim, self.num_heads * self.rotary_pairs)
        self.phase_network = nn.Sequential(
            nn.LayerNorm(geometry_dim),
            nn.Linear(geometry_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.num_heads * self.rotary_pairs),
        )
        self.gate_unconstrained = nn.Parameter(torch.zeros(self.num_heads))

    def forward(
        self,
        query: Any,
        key: Any,
        geometry_features: Any,
        geometry_valid_mask: Any,
    ) -> Any:
        if query.ndim != 4 or key.shape != query.shape:
            raise ValueError("query and key must share shape [batch, head, residue, channel]")
        batch_size, num_heads, residue_count, _ = query.shape
        if num_heads != self.num_heads:
            raise ValueError("query head count differs from configured num_heads")
        expected_pair_shape = (batch_size, residue_count, residue_count)
        if tuple(geometry_features.shape[:3]) != expected_pair_shape:
            raise ValueError("geometry_features must align with query residue pairs")
        if tuple(geometry_valid_mask.shape) != expected_pair_shape:
            raise ValueError("geometry_valid_mask must align with query residue pairs")

        phase = self.phase_network(geometry_features).view(
            batch_size,
            residue_count,
            residue_count,
            self.num_heads,
            self.rotary_pairs,
        )
        phase = math.pi * torch.tanh(phase).permute(0, 3, 1, 2, 4)
        query_rotary = query[..., : self.rotary_channels]
        key_rotary = key[..., : self.rotary_channels]
        query_real, query_imaginary = (
            query_rotary[..., 0::2],
            query_rotary[..., 1::2],
        )
        key_real, key_imaginary = (
            key_rotary[..., 0::2],
            key_rotary[..., 1::2],
        )
        real_product = (
            query_real[:, :, :, None, :] * key_real[:, :, None, :, :]
            + query_imaginary[:, :, :, None, :] * key_imaginary[:, :, None, :, :]
        )
        imaginary_product = (
            query_imaginary[:, :, :, None, :] * key_real[:, :, None, :, :]
            - query_real[:, :, :, None, :] * key_imaginary[:, :, None, :, :]
        )
        pair_score = (
            real_product * phase.cos() - imaginary_product * phase.sin()
        ).sum(dim=-1) / math.sqrt(self.rotary_pairs)
        gate = torch.tanh(self.gate_unconstrained)[None, :, None, None]
        valid = geometry_valid_mask[:, None, :, :].to(pair_score.dtype)
        return gate * pair_score * valid


class GeometryBiasedSelfAttentionBlock(_ModuleBase):
    """Pre-norm self-attention with sequence RoPE and geometric head biases."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        geometry_dim: int,
        *,
        geometry_bias_mode: str = "pair_mlp",
        ipa_num_points: int = 4,
        num_rbf: int = 16,
        local_pose_rotary_channels: int | None = None,
        ffn_expansion: int = 4,
        dropout: float = 0.0,
        attention_kernel: str = "reference",
    ) -> None:
        require_torch()
        super().__init__()
        d_model = _positive_integer(d_model, "d_model")
        num_heads = _positive_integer(num_heads, "num_heads")
        geometry_dim = _positive_integer(geometry_dim, "geometry_dim")
        ffn_expansion = _positive_integer(ffn_expansion, "ffn_expansion")
        dropout = _dropout_probability(dropout)
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.head_dim = d_model // num_heads
        if self.head_dim % 2:
            raise ValueError("Per-head channels must be even for sequence RoPE")
        self.d_model = d_model
        self.num_heads = num_heads
        self.attention_kernel = _attention_kernel(attention_kernel)
        self.attention_norm = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.output_projection = nn.Linear(d_model, d_model)
        self.geometry_bias_mode = _geometry_bias_mode(geometry_bias_mode)
        if self.geometry_bias_mode == "pair_mlp":
            self.geometry_bias = GeometryHeadBias(geometry_dim, num_heads)
        elif self.geometry_bias_mode == "invariant_points":
            self.geometry_bias = InvariantPointHeadBias(
                num_heads,
                num_points=ipa_num_points,
            )
        elif self.geometry_bias_mode == "local_pose_multibasis":
            self.geometry_bias = LocalPoseMultiBasisHeadBias(
                geometry_dim,
                num_heads,
                num_rbf=num_rbf,
            )
        else:
            self.geometry_bias = PairConditionedLocalPoseRotaryBias(
                geometry_dim,
                num_heads,
                self.head_dim,
                rotary_channels=local_pose_rotary_channels,
            )
        self.attention_dropout = nn.Dropout(dropout)
        self.residual_dropout = nn.Dropout(dropout)
        hidden_dim = ffn_expansion * d_model
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(
        self,
        features: Any,
        geometry: IntraChainRelativeGeometry,
        residue_mask: Any,
        *,
        positions: Any | None = None,
        return_attention: bool = False,
    ) -> tuple[Any, dict[str, Any] | None]:
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
        content_logits = torch.einsum("bhid,bhjd->bhij", query, key)
        content_logits = content_logits / math.sqrt(self.head_dim)
        geometry_components = None
        if self.geometry_bias_mode == "pair_mlp":
            geometry_features = geometry.features.to(dtype=features.dtype)
            geometry_bias = self.geometry_bias(geometry_features)
        elif self.geometry_bias_mode == "invariant_points":
            geometry_bias = self.geometry_bias(geometry).to(dtype=features.dtype)
        elif self.geometry_bias_mode == "local_pose_multibasis":
            geometry_features = geometry.features.to(dtype=features.dtype)
            geometry_bias = self.geometry_bias(
                geometry_features,
                geometry.geometry_valid_mask,
            )
            if return_attention:
                geometry_components = self.geometry_bias.component_biases(
                    geometry_features,
                    geometry.geometry_valid_mask,
                )
        else:
            geometry_features = geometry.features.to(dtype=features.dtype)
            geometry_bias = self.geometry_bias(
                query,
                key,
                geometry_features,
                geometry.geometry_valid_mask,
            ).to(dtype=features.dtype)
        attention_mask = geometry.neighbor_mask[:, None, :, :]
        attention = None
        if self.attention_kernel == "sdpa" and not return_attention:
            # A float mask lets SDPA fuse FrameContact's arbitrary per-head
            # geometry score with masking.  Padded query rows are all -inf;
            # the explicit query mask below makes their output exactly zero.
            attention_bias = geometry_bias.masked_fill(
                ~attention_mask, float("-inf")
            )
            context = functional.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_bias,
                dropout_p=(
                    self.attention_dropout.p if self.training else 0.0
                ),
                is_causal=False,
            )
            context = context * residue_mask[:, None, :, None].to(context.dtype)
        else:
            logits = (content_logits + geometry_bias).masked_fill(
                ~attention_mask, torch.finfo(content_logits.dtype).min
            )
            attention = torch.softmax(logits, dim=-1)
            attention = attention * residue_mask[:, None, :, None].to(
                attention.dtype
            )
            dropped_attention = self.attention_dropout(attention)
            context = torch.einsum("bhij,bhjd->bhid", dropped_attention, value)
        context = context.transpose(1, 2).reshape(batch_size, residue_count, self.d_model)
        features = features + self.residual_dropout(self.output_projection(context))
        features = features + self.residual_dropout(self.ffn(self.ffn_norm(features)))
        features = features.masked_fill(~residue_mask[..., None], 0.0)
        if not return_attention:
            return features, None
        assert attention is not None
        content_only_logits = content_logits.masked_fill(
            ~attention_mask, torch.finfo(content_logits.dtype).min
        )
        content_only_attention = torch.softmax(content_only_logits, dim=-1)
        content_only_attention = content_only_attention * residue_mask[
            :, None, :, None
        ].to(content_only_attention.dtype)
        details = {
            "weights": attention,
            "content_only_weights": content_only_attention,
            "geometry_bias": geometry_bias,
            "neighbor_mask": geometry.neighbor_mask,
            "geometry_valid_mask": geometry.geometry_valid_mask,
        }
        if geometry_components is not None:
            details["geometry_bias_components"] = geometry_components
        if self.geometry_bias_mode == "local_pose_rotary":
            details["geometry_rotary_gate"] = torch.tanh(
                self.geometry_bias.gate_unconstrained
            )
            details["geometry_rotary_channels"] = self.geometry_bias.rotary_channels
        return features, details


class IntraChainGeometryEncoder(_ModuleBase):
    """Stack geometry-aware self-attention blocks for one residue stream."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        *,
        geometry_feature_mode: str = "full_pose",
        geometry_bias_mode: str = "pair_mlp",
        ipa_num_points: int = 4,
        local_pose_rotary_channels: int | None = None,
        attention_graph_mode: str = "spatial",
        num_rbf: int = 16,
        distance_max: float = 20.0,
        neighbor_k: int | None = 32,
        neighbor_radius: float | None = None,
        sequence_window: int = 32,
        ffn_expansion: int = 4,
        dropout: float = 0.0,
        attention_kernel: str = "reference",
    ) -> None:
        require_torch()
        super().__init__()
        self.d_model = _positive_integer(d_model, "d_model")
        self.num_rbf = _positive_integer(num_rbf, "num_rbf")
        self.geometry_feature_mode = geometry_feature_mode
        self.geometry_bias_mode = _geometry_bias_mode(geometry_bias_mode)
        self.attention_graph_mode = _attention_graph_mode(attention_graph_mode)
        self.attention_kernel = _attention_kernel(attention_kernel)
        self.distance_max = _optional_positive_float(distance_max, "distance_max")
        assert self.distance_max is not None
        self.neighbor_k = _optional_positive_integer(neighbor_k, "neighbor_k")
        self.neighbor_radius = _optional_positive_float(
            neighbor_radius, "neighbor_radius"
        )
        self.sequence_window = _positive_integer(sequence_window, "sequence_window")
        num_layers = _positive_integer(num_layers, "num_layers")
        geometry_dim = _geometry_feature_dimension(
            self.geometry_feature_mode,
            self.num_rbf,
        )
        self.blocks = nn.ModuleList(
            GeometryBiasedSelfAttentionBlock(
                d_model,
                num_heads,
                geometry_dim,
                geometry_bias_mode=self.geometry_bias_mode,
                ipa_num_points=ipa_num_points,
                num_rbf=self.num_rbf,
                local_pose_rotary_channels=local_pose_rotary_channels,
                ffn_expansion=ffn_expansion,
                dropout=dropout,
                attention_kernel=self.attention_kernel,
            )
            for _ in range(num_layers)
        )

    def forward(
        self,
        features: Any,
        residue_mask: Any,
        backbone: Any,
        *,
        chain_ids: Any | None = None,
        positions: Any | None = None,
        neighbor_mask_override: Any | None = None,
        return_attention: bool = False,
    ) -> tuple[Any, dict[str, Any] | None]:
        _validate_stream_inputs(
            features,
            residue_mask,
            backbone,
            chain_ids,
            self.d_model,
        )
        frames = build_backbone_frames(backbone, residue_mask)
        geometry = build_intra_chain_relative_geometry(
            frames,
            residue_mask,
            chain_ids=chain_ids,
            geometry_feature_mode=self.geometry_feature_mode,
            attention_graph_mode=self.attention_graph_mode,
            num_rbf=self.num_rbf,
            distance_max=self.distance_max,
            neighbor_k=self.neighbor_k,
            neighbor_radius=self.neighbor_radius,
            sequence_window=self.sequence_window,
        )
        if neighbor_mask_override is not None:
            if (
                not torch.is_tensor(neighbor_mask_override)
                or neighbor_mask_override.dtype != torch.bool
                or tuple(neighbor_mask_override.shape)
                != tuple(geometry.neighbor_mask.shape)
                or neighbor_mask_override.device != features.device
            ):
                raise ValueError(
                    "neighbor_mask_override must be boolean and align with residue pairs"
                )
            permitted = residue_mask[:, :, None] & residue_mask[:, None, :]
            if chain_ids is not None:
                permitted = permitted & (
                    chain_ids[:, :, None] == chain_ids[:, None, :]
                )
            if bool((neighbor_mask_override & ~permitted).any()):
                raise ValueError("neighbor_mask_override contains forbidden residue pairs")
            identity = torch.eye(
                residue_mask.shape[1], dtype=torch.bool, device=features.device
            )[None]
            required_self_edges = identity & residue_mask[:, :, None]
            if not bool((neighbor_mask_override & required_self_edges).equal(required_self_edges)):
                raise ValueError("neighbor_mask_override must retain every real self edge")
            geometry = IntraChainRelativeGeometry(
                features=geometry.features,
                distance=geometry.distance,
                displacement_local=geometry.displacement_local,
                relative_rotation=geometry.relative_rotation,
                geometry_valid_mask=geometry.geometry_valid_mask,
                neighbor_mask=neighbor_mask_override,
            )
        layer_details = []
        for block in self.blocks:
            features, details = block(
                features,
                geometry,
                residue_mask,
                positions=positions,
                return_attention=return_attention,
            )
            if details is not None:
                layer_details.append(details)
        if not return_attention:
            return features, None
        return features, {
            "layers": tuple(layer_details),
            "frame_valid_mask": frames.valid_mask,
            "neighbor_mask": geometry.neighbor_mask,
            "distance": geometry.distance,
            "attention_graph_mode": self.attention_graph_mode,
        }


class DualIntraChainGeometryEncoder(_ModuleBase):
    """Independent antibody and antigen structure-aware self-attention stacks."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        require_torch()
        super().__init__()
        # Two separate constructor calls are intentional: no module or tensor is shared.
        self.antibody_encoder = IntraChainGeometryEncoder(*args, **kwargs)
        self.antigen_encoder = IntraChainGeometryEncoder(*args, **kwargs)

    def forward(
        self,
        antibody_features: Any,
        antigen_features: Any,
        antibody_mask: Any,
        antigen_mask: Any,
        antibody_backbone: Any,
        antigen_backbone: Any,
        *,
        antibody_chain_ids: Any | None = None,
        antigen_chain_ids: Any | None = None,
        antibody_positions: Any | None = None,
        antigen_positions: Any | None = None,
        antibody_neighbor_mask_override: Any | None = None,
        antigen_neighbor_mask_override: Any | None = None,
        return_attention: bool = False,
    ) -> tuple[Any, Any, dict[str, Any] | None]:
        antibody, antibody_details = self.antibody_encoder(
            antibody_features,
            antibody_mask,
            antibody_backbone,
            chain_ids=antibody_chain_ids,
            positions=antibody_positions,
            neighbor_mask_override=antibody_neighbor_mask_override,
            return_attention=return_attention,
        )
        antigen, antigen_details = self.antigen_encoder(
            antigen_features,
            antigen_mask,
            antigen_backbone,
            chain_ids=antigen_chain_ids,
            positions=antigen_positions,
            neighbor_mask_override=antigen_neighbor_mask_override,
            return_attention=return_attention,
        )
        if not return_attention:
            return antibody, antigen, None
        return antibody, antigen, {
            "antibody": antibody_details,
            "antigen": antigen_details,
        }


__all__ = [
    "BackboneFrames",
    "DualIntraChainGeometryEncoder",
    "GeometryBiasedSelfAttentionBlock",
    "GeometryHeadBias",
    "InvariantPointHeadBias",
    "IntraChainGeometryEncoder",
    "IntraChainRelativeGeometry",
    "LocalPoseMultiBasisHeadBias",
    "PairConditionedLocalPoseRotaryBias",
    "apply_sequence_rope",
    "build_backbone_frames",
    "build_intra_chain_relative_geometry",
    "gaussian_rbf",
]
