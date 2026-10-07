"""Prospective pair-geometry architecture for antibody-antigen research.

This module is intentionally separate from the frozen D0/R1/R2 protocol
implementations. It defines an untrained research candidate and does not confer
authorization for model selection, held-out evaluation, or performance claims.

All pair tensors use antibody-by-antigen orientation:
``[batch, antibody_residue, antigen_residue, ...]``.
"""

from __future__ import annotations

import math
from collections.abc import Collection
from typing import Any

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError:
    torch = None
    nn = None
    F = None

from .scoring import (
    masked_double_center,
    masked_logmeanexp_score,
    pool_pair_marginals,
)
from .intra_chain_geometry import DualIntraChainGeometryEncoder
from .inter_chain_geometry import (
    build_inter_chain_geometry,
    InterChainGeometryBias,
    InterChainPairFeatureProjection,
)
from .pair_geometry_schema import PAIR_ORIENTATION, PairGeometrySchema


_ModuleBase = nn.Module if nn is not None else object


def require_torch() -> None:
    """Fail with a focused message when PyTorch is unavailable."""
    if torch is None or nn is None or F is None:
        raise RuntimeError("PyTorch is required for pair-geometry operations")


def _positive_integer(value: Any, name: str) -> int:
    """Return a positive integer or reject the configuration."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_finite(value: Any, name: str) -> float:
    """Return a positive finite scalar or reject the configuration."""
    try:
        converted = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a positive finite scalar")
    if not math.isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{name} must be a positive finite scalar")
    return converted


def _dropout_probability(value: Any) -> float:
    """Validate a dropout probability."""
    try:
        converted = float(value)
    except (TypeError, ValueError):
        raise ValueError("dropout must be a finite scalar in [0, 1)")
    if not math.isfinite(converted) or not 0.0 <= converted < 1.0:
        raise ValueError("dropout must be a finite scalar in [0, 1)")
    return converted


def _validate_residue_tensor(
    features: Any,
    mask: Any,
    expected_channels: int,
    name: str,
) -> None:
    """Validate one padded residue stream without accepting ambiguous inputs."""
    if not torch.is_tensor(features):
        raise TypeError(f"{name}_features must be a PyTorch tensor")
    if features.ndim != 3:
        raise ValueError(
            f"{name}_features must have shape [batch, residues, channels]"
        )
    if not torch.is_floating_point(features):
        raise ValueError(f"{name}_features must use a floating-point dtype")
    if any(int(size) == 0 for size in features.shape):
        raise ValueError(f"{name}_features cannot contain an empty dimension")
    if int(features.shape[-1]) != expected_channels:
        raise ValueError(
            f"{name}_features must have {expected_channels} channels, "
            f"received {int(features.shape[-1])}"
        )
    if not bool(torch.isfinite(features).all()):
        raise ValueError(f"{name}_features must contain only finite values")
    if not torch.is_tensor(mask):
        raise TypeError(f"{name}_mask must be a PyTorch tensor")
    if mask.dtype != torch.bool or mask.ndim != 2:
        raise ValueError(f"{name}_mask must be a two-dimensional boolean tensor")
    if tuple(mask.shape) != tuple(features.shape[:2]):
        raise ValueError(f"{name}_mask must align with {name}_features")
    if mask.device != features.device:
        raise ValueError(f"{name}_mask and {name}_features must share a device")
    if bool((mask.sum(dim=1) == 0).any()):
        raise ValueError(f"Every item must contain a valid {name} residue")


def _validate_pair_mask(
    candidate_pair_mask: Any,
    antibody_mask: Any,
    antigen_mask: Any,
) -> None:
    """Require an explicit nonempty candidate set contained in valid axes."""
    if not torch.is_tensor(candidate_pair_mask):
        raise TypeError("candidate_pair_mask must be a PyTorch tensor")
    if candidate_pair_mask.dtype != torch.bool or candidate_pair_mask.ndim != 3:
        raise ValueError(
            "candidate_pair_mask must be a three-dimensional boolean tensor"
        )
    expected_shape = (
        int(antibody_mask.shape[0]),
        int(antibody_mask.shape[1]),
        int(antigen_mask.shape[1]),
    )
    if tuple(candidate_pair_mask.shape) != expected_shape:
        raise ValueError(
            "candidate_pair_mask must have shape "
            "[batch, antibody_residue, antigen_residue]"
        )
    if candidate_pair_mask.device != antibody_mask.device:
        raise ValueError("candidate_pair_mask and residue masks must share a device")
    valid_axes = antibody_mask.unsqueeze(2) & antigen_mask.unsqueeze(1)
    if bool((candidate_pair_mask & ~valid_axes).any()):
        raise ValueError("candidate_pair_mask cannot include a padded residue")
    candidate_counts = candidate_pair_mask.flatten(1).sum(dim=1)
    if bool((candidate_counts == 0).any()):
        raise ValueError("Every item must contain at least one candidate pair")


def _masked_probabilities(logits: Any, mask: Any) -> Any:
    """Apply a sigmoid and assign exactly zero to unsupported pairs."""
    return torch.sigmoid(logits).masked_fill(~mask, 0.0)


def double_centered_partner_score(
    pair_interactions: Any,
    candidate_pair_mask: Any,
    *,
    temperature: float = 1.0,
    centering_iterations: int = 16,
) -> tuple[Any, Any]:
    """Return double-centered interactions and a length-normalized score.

    Inputs and centered outputs use antibody-by-antigen orientation. The score
    is a masked log-mean-exp over the double-centered interaction matrix, so
    row-only, column-only, and global main effects cannot directly determine it.
    """
    require_torch()
    temperature = _positive_finite(temperature, "temperature")
    centering_iterations = _positive_integer(
        centering_iterations,
        "centering_iterations",
    )
    if not torch.is_tensor(pair_interactions) or pair_interactions.ndim != 3:
        raise ValueError(
            "pair_interactions must have shape "
            "[batch, antibody_residue, antigen_residue]"
        )
    if not torch.is_floating_point(pair_interactions):
        raise ValueError("pair_interactions must use a floating-point dtype")
    if not torch.is_tensor(candidate_pair_mask):
        raise TypeError("candidate_pair_mask must be a PyTorch tensor")
    if candidate_pair_mask.dtype != torch.bool:
        raise ValueError("candidate_pair_mask must be boolean")
    if tuple(candidate_pair_mask.shape) != tuple(pair_interactions.shape):
        raise ValueError("candidate_pair_mask must align with pair_interactions")
    if candidate_pair_mask.device != pair_interactions.device:
        raise ValueError("Pair interactions and their mask must share a device")
    valid_counts = candidate_pair_mask.flatten(1).sum(dim=1)
    if bool((valid_counts == 0).any()):
        raise ValueError("Every item must contain at least one candidate pair")
    if not bool(torch.isfinite(pair_interactions[candidate_pair_mask]).all()):
        raise ValueError("Candidate pair interactions must be finite")

    antigen_by_antibody = pair_interactions.transpose(1, 2)
    transposed_mask = candidate_pair_mask.transpose(1, 2)
    centered = masked_double_center(
        antigen_by_antibody,
        transposed_mask,
        iterations=centering_iterations,
    )
    score = masked_logmeanexp_score(
        centered,
        transposed_mask,
        temperature=temperature,
    )
    return centered.transpose(1, 2), score


class EarlyBidirectionalResidueFusion(_ModuleBase):
    """Exchange partner context before constructing explicit residue pairs."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
    ) -> None:
        require_torch()
        super().__init__()
        d_model = _positive_integer(d_model, "d_model")
        num_heads = _positive_integer(num_heads, "num_heads")
        ffn_dim = _positive_integer(ffn_dim, "ffn_dim")
        dropout = _dropout_probability(dropout)
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.antibody_attention_norm = nn.LayerNorm(d_model)
        self.antigen_attention_norm = nn.LayerNorm(d_model)
        self.antibody_queries_antigen = nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.antigen_queries_antibody = nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.antibody_gate = nn.Parameter(torch.tensor(-2.0))
        self.antigen_gate = nn.Parameter(torch.tensor(-2.0))
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
        """Fuse both streams while preventing padded keys from contributing."""
        normalized_antibody = self.antibody_attention_norm(antibody)
        normalized_antigen = self.antigen_attention_norm(antigen)
        antibody_context, _ = self.antibody_queries_antigen(
            normalized_antibody,
            normalized_antigen,
            normalized_antigen,
            key_padding_mask=~antigen_mask,
            need_weights=False,
        )
        antigen_context, _ = self.antigen_queries_antibody(
            normalized_antigen,
            normalized_antibody,
            normalized_antibody,
            key_padding_mask=~antibody_mask,
            need_weights=False,
        )
        antibody = antibody + torch.sigmoid(self.antibody_gate) * (
            self.residual_dropout(antibody_context)
        )
        antigen = antigen + torch.sigmoid(self.antigen_gate) * (
            self.residual_dropout(antigen_context)
        )
        antibody = antibody + self.residual_dropout(
            self.antibody_ffn(self.antibody_ffn_norm(antibody))
        )
        antigen = antigen + self.residual_dropout(
            self.antigen_ffn(self.antigen_ffn_norm(antigen))
        )
        antibody = antibody.masked_fill(~antibody_mask.unsqueeze(-1), 0.0)
        antigen = antigen.masked_fill(~antigen_mask.unsqueeze(-1), 0.0)
        return antibody, antigen


class BipartitePairAxialUpdateBlock(_ModuleBase):
    """Update explicit pair states along both residue axes without mask leakage."""

    def __init__(
        self,
        pair_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
    ) -> None:
        require_torch()
        super().__init__()
        pair_dim = _positive_integer(pair_dim, "pair_dim")
        num_heads = _positive_integer(num_heads, "num_heads")
        ffn_dim = _positive_integer(ffn_dim, "ffn_dim")
        dropout = _dropout_probability(dropout)
        if pair_dim % num_heads != 0:
            raise ValueError("pair_dim must be divisible by pair update num_heads")

        self.pair_dim = pair_dim
        self.row_norm = nn.LayerNorm(pair_dim)
        self.column_norm = nn.LayerNorm(pair_dim)
        self.row_attention = nn.MultiheadAttention(
            pair_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.column_attention = nn.MultiheadAttention(
            pair_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.row_gate = nn.Parameter(torch.tensor(-2.0))
        self.column_gate = nn.Parameter(torch.tensor(-2.0))
        self.ffn_norm = nn.LayerNorm(pair_dim)
        self.ffn = nn.Sequential(
            nn.Linear(pair_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, pair_dim),
        )
        self.residual_dropout = nn.Dropout(dropout)

    @staticmethod
    def _validate_inputs(pair_state: Any, pair_mask: Any, pair_dim: int) -> None:
        """Reject malformed or non-finite pair states before attention."""
        if not torch.is_tensor(pair_state):
            raise TypeError("pair_state must be a PyTorch tensor")
        if pair_state.ndim != 4 or int(pair_state.shape[-1]) != pair_dim:
            raise ValueError(
                "pair_state must have shape "
                "[batch, antibody_residue, antigen_residue, pair_dim]"
            )
        if not torch.is_floating_point(pair_state):
            raise ValueError("pair_state must use a floating-point dtype")
        if any(int(size) == 0 for size in pair_state.shape):
            raise ValueError("pair_state cannot contain an empty dimension")
        if not torch.is_tensor(pair_mask):
            raise TypeError("pair_mask must be a PyTorch tensor")
        if pair_mask.dtype != torch.bool or pair_mask.ndim != 3:
            raise ValueError("pair_mask must be a three-dimensional boolean tensor")
        if tuple(pair_mask.shape) != tuple(pair_state.shape[:3]):
            raise ValueError("pair_mask must align with pair_state")
        if pair_mask.device != pair_state.device:
            raise ValueError("pair_mask and pair_state must share a device")
        if bool((pair_mask.flatten(1).sum(dim=1) == 0).any()):
            raise ValueError("Every item must contain at least one candidate pair")
        if not bool(torch.isfinite(pair_state[pair_mask]).all()):
            raise ValueError("Candidate pair states must be finite")

    @staticmethod
    def _attend_nonempty_sequences(
        attention: Any,
        sequences: Any,
        sequence_mask: Any,
    ) -> Any:
        """Apply attention only to rows with at least one unmasked key."""
        valid_sequences = sequence_mask.any(dim=1)
        valid_indices = valid_sequences.nonzero(as_tuple=False).squeeze(-1)
        selected_sequences = sequences.index_select(0, valid_indices)
        selected_mask = sequence_mask.index_select(0, valid_indices)
        selected_output, _ = attention(
            selected_sequences,
            selected_sequences,
            selected_sequences,
            key_padding_mask=~selected_mask,
            need_weights=False,
        )
        selected_output = selected_output.masked_fill(
            ~selected_mask.unsqueeze(-1),
            0.0,
        )
        output = torch.zeros_like(sequences)
        return output.index_copy(0, valid_indices, selected_output)

    def forward(self, pair_state: Any, pair_mask: Any) -> Any:
        """Apply antigen-row attention, antibody-column attention, and an FFN."""
        self._validate_inputs(pair_state, pair_mask, self.pair_dim)
        batch_size, antibody_length, antigen_length, pair_dim = pair_state.shape
        pair_state = pair_state.masked_fill(~pair_mask.unsqueeze(-1), 0.0)

        row_sequences = self.row_norm(pair_state).reshape(
            batch_size * antibody_length,
            antigen_length,
            pair_dim,
        )
        row_mask = pair_mask.reshape(
            batch_size * antibody_length,
            antigen_length,
        )
        row_update = self._attend_nonempty_sequences(
            self.row_attention,
            row_sequences,
            row_mask,
        ).reshape_as(pair_state)
        pair_state = pair_state + torch.sigmoid(self.row_gate) * (
            self.residual_dropout(row_update)
        )
        pair_state = pair_state.masked_fill(~pair_mask.unsqueeze(-1), 0.0)

        column_state = pair_state.transpose(1, 2)
        column_sequences = self.column_norm(column_state).reshape(
            batch_size * antigen_length,
            antibody_length,
            pair_dim,
        )
        column_mask = pair_mask.transpose(1, 2).reshape(
            batch_size * antigen_length,
            antibody_length,
        )
        column_update = self._attend_nonempty_sequences(
            self.column_attention,
            column_sequences,
            column_mask,
        ).reshape_as(column_state).transpose(1, 2)
        pair_state = pair_state + torch.sigmoid(self.column_gate) * (
            self.residual_dropout(column_update)
        )
        pair_state = pair_state.masked_fill(~pair_mask.unsqueeze(-1), 0.0)

        pair_state = pair_state + self.residual_dropout(
            self.ffn(self.ffn_norm(pair_state))
        )
        return pair_state.masked_fill(~pair_mask.unsqueeze(-1), 0.0)


class ProspectivePairGeometryModel(_ModuleBase):
    """Predict candidate-masked contacts and geometry from explicit pair states.

    The implementation chunks the antibody axis during pair construction to
    limit peak intermediate memory. Returning pair representations remains
    optional because the complete tensor is quadratic in sequence length.
    Enabling axial pair updates necessarily materializes the complete pair
    state before decoding because column attention crosses antibody chunks.
    """

    def __init__(
        self,
        antibody_input_dim: int,
        antigen_input_dim: int,
        *,
        d_model: int = 256,
        pair_dim: int = 128,
        num_heads: int = 8,
        num_fusion_layers: int = 2,
        intra_chain_geometry_layers: int = 0,
        intra_chain_geometry_feature_mode: str = "full_pose",
        intra_chain_geometry_num_rbf: int = 16,
        intra_chain_geometry_distance_max: float = 20.0,
        intra_chain_geometry_neighbor_k: int | None = 32,
        intra_chain_geometry_neighbor_radius: float | None = None,
        ffn_dim: int = 512,
        geometry_schema: PairGeometrySchema | None = None,
        num_distance_bins: int | None = None,
        contact_distance_bin_count: int | None = None,
        num_orientation_axes: int | None = None,
        num_orientation_bins: int | None = None,
        num_pair_update_layers: int = 0,
        pair_update_num_heads: int | None = None,
        pair_update_ffn_dim: int | None = None,
        pair_chunk_size: int = 64,
        dropout: float = 0.1,
        marginal_pooling: str = "logmeanexp",
        marginal_top_k: int = 3,
        marginal_temperature: float = 1.0,
        partner_score_temperature: float = 1.0,
        centering_iterations: int = 16,
        contact_logit_mode: str = "distance_mass",
    ) -> None:
        require_torch()
        super().__init__()
        antibody_input_dim = _positive_integer(
            antibody_input_dim,
            "antibody_input_dim",
        )
        antigen_input_dim = _positive_integer(antigen_input_dim, "antigen_input_dim")
        d_model = _positive_integer(d_model, "d_model")
        pair_dim = _positive_integer(pair_dim, "pair_dim")
        num_heads = _positive_integer(num_heads, "num_heads")
        ffn_dim = _positive_integer(ffn_dim, "ffn_dim")
        if not isinstance(geometry_schema, PairGeometrySchema):
            raise TypeError(
                "geometry_schema must be an explicit PairGeometrySchema"
            )
        schema_dimensions = {
            "num_distance_bins": geometry_schema.num_distance_bins,
            "contact_distance_bin_count": (
                geometry_schema.contact_distance_bin_count
            ),
            "num_orientation_axes": geometry_schema.num_orientation_axes,
            "num_orientation_bins": geometry_schema.num_orientation_bins,
        }
        supplied_dimensions = {
            "num_distance_bins": num_distance_bins,
            "contact_distance_bin_count": contact_distance_bin_count,
            "num_orientation_axes": num_orientation_axes,
            "num_orientation_bins": num_orientation_bins,
        }
        for name, supplied in supplied_dimensions.items():
            if supplied is not None:
                supplied = _positive_integer(supplied, name)
                if supplied != schema_dimensions[name]:
                    raise ValueError(
                        f"{name}={supplied} does not match geometry_schema "
                        f"value {schema_dimensions[name]}"
                    )
        num_distance_bins = schema_dimensions["num_distance_bins"]
        contact_distance_bin_count = schema_dimensions[
            "contact_distance_bin_count"
        ]
        num_orientation_axes = schema_dimensions["num_orientation_axes"]
        num_orientation_bins = schema_dimensions["num_orientation_bins"]
        pair_chunk_size = _positive_integer(pair_chunk_size, "pair_chunk_size")
        centering_iterations = _positive_integer(
            centering_iterations,
            "centering_iterations",
        )
        dropout = _dropout_probability(dropout)
        if isinstance(num_fusion_layers, bool) or not isinstance(
            num_fusion_layers,
            int,
        ):
            raise ValueError("num_fusion_layers must be a nonnegative integer")
        if num_fusion_layers < 0:
            raise ValueError("num_fusion_layers must be a nonnegative integer")
        if (
            isinstance(num_pair_update_layers, bool)
            or not isinstance(num_pair_update_layers, int)
            or num_pair_update_layers < 0
        ):
            raise ValueError("num_pair_update_layers must be a nonnegative integer")
        if (
            isinstance(intra_chain_geometry_layers, bool)
            or not isinstance(intra_chain_geometry_layers, int)
            or intra_chain_geometry_layers < 0
        ):
            raise ValueError("intra_chain_geometry_layers must be a nonnegative integer")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if pair_update_num_heads is None:
            pair_update_num_heads = num_heads
        pair_update_num_heads = _positive_integer(
            pair_update_num_heads,
            "pair_update_num_heads",
        )
        if pair_update_ffn_dim is None:
            pair_update_ffn_dim = 2 * pair_dim
        pair_update_ffn_dim = _positive_integer(
            pair_update_ffn_dim,
            "pair_update_ffn_dim",
        )
        if num_pair_update_layers > 0 and pair_dim % pair_update_num_heads != 0:
            raise ValueError(
                "pair_dim must be divisible by pair_update_num_heads when "
                "pair updates are enabled"
            )
        if contact_distance_bin_count >= num_distance_bins:
            raise ValueError(
                "contact_distance_bin_count must be smaller than num_distance_bins"
            )
        if marginal_pooling not in {"noisy_or", "topk", "logmeanexp"}:
            raise ValueError(
                "marginal_pooling must be 'noisy_or', 'topk', or 'logmeanexp'"
            )
        marginal_top_k = _positive_integer(marginal_top_k, "marginal_top_k")
        marginal_temperature = _positive_finite(
            marginal_temperature,
            "marginal_temperature",
        )
        partner_score_temperature = _positive_finite(
            partner_score_temperature,
            "partner_score_temperature",
        )
        if contact_logit_mode not in {
            "distance_mass",
            "unary_plus_interaction_residual",
        }:
            raise ValueError(
                "contact_logit_mode must be 'distance_mass' or "
                "'unary_plus_interaction_residual'"
            )

        self.antibody_input_dim = antibody_input_dim
        self.antigen_input_dim = antigen_input_dim
        self.geometry_schema = geometry_schema
        self.num_distance_bins = num_distance_bins
        self.contact_distance_bin_count = contact_distance_bin_count
        self.num_orientation_axes = num_orientation_axes
        self.num_orientation_bins = num_orientation_bins
        self.num_pair_update_layers = num_pair_update_layers
        self.pair_update_num_heads = pair_update_num_heads
        self.pair_update_ffn_dim = pair_update_ffn_dim
        self.pair_chunk_size = pair_chunk_size
        self.marginal_pooling = marginal_pooling
        self.marginal_top_k = marginal_top_k
        self.marginal_temperature = marginal_temperature
        self.partner_score_temperature = partner_score_temperature
        self.centering_iterations = centering_iterations
        self.contact_logit_mode = contact_logit_mode

        self.antibody_projection = nn.Sequential(
            nn.LayerNorm(antibody_input_dim),
            nn.Linear(antibody_input_dim, d_model),
            nn.GELU(),
        )
        self.antigen_projection = nn.Sequential(
            nn.LayerNorm(antigen_input_dim),
            nn.Linear(antigen_input_dim, d_model),
            nn.GELU(),
        )
        self.intra_chain_geometry = (
            DualIntraChainGeometryEncoder(
                d_model,
                num_heads,
                intra_chain_geometry_layers,
                geometry_feature_mode=intra_chain_geometry_feature_mode,
                num_rbf=intra_chain_geometry_num_rbf,
                distance_max=intra_chain_geometry_distance_max,
                neighbor_k=intra_chain_geometry_neighbor_k,
                neighbor_radius=intra_chain_geometry_neighbor_radius,
                dropout=dropout,
            )
            if intra_chain_geometry_layers > 0
            else None
        )
        self.inter_chain_geometry_num_rbf = intra_chain_geometry_num_rbf
        self.inter_chain_geometry_distance_max = intra_chain_geometry_distance_max
        inter_chain_geometry_dim = intra_chain_geometry_num_rbf + 15
        self.inter_chain_projection = InterChainPairFeatureProjection(
            geometry_dim=inter_chain_geometry_dim,
            pair_dim=pair_dim,
        )
        self.fusion_layers = nn.ModuleList(
            EarlyBidirectionalResidueFusion(
                d_model=d_model,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
            )
            for _ in range(num_fusion_layers)
        )
        self.antibody_final_norm = nn.LayerNorm(d_model)
        self.antigen_final_norm = nn.LayerNorm(d_model)

        self.antibody_pair_add = nn.Linear(d_model, pair_dim, bias=False)
        self.antigen_pair_add = nn.Linear(d_model, pair_dim, bias=False)
        self.antibody_pair_gate = nn.Linear(d_model, pair_dim, bias=False)
        self.antigen_pair_gate = nn.Linear(d_model, pair_dim, bias=False)
        self.pair_refinement = nn.Sequential(
            nn.LayerNorm(pair_dim),
            nn.Linear(pair_dim, 2 * pair_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * pair_dim, pair_dim),
            nn.LayerNorm(pair_dim),
        )
        self.pair_update_layers = nn.ModuleList(
            BipartitePairAxialUpdateBlock(
                pair_dim=pair_dim,
                num_heads=pair_update_num_heads,
                ffn_dim=pair_update_ffn_dim,
                dropout=dropout,
            )
            for _ in range(num_pair_update_layers)
        )
        self.distance_head = nn.Linear(pair_dim, num_distance_bins)
        self.orientation_head = nn.Linear(
            pair_dim,
            num_orientation_axes * num_orientation_bins,
        )
        self.interaction_head = nn.Linear(pair_dim, 1)
        if contact_logit_mode == "unary_plus_interaction_residual":
            self.antibody_unary_head = nn.Linear(d_model, 1)
            self.antigen_unary_head = nn.Linear(d_model, 1)
            self.specificity_head = nn.Linear(pair_dim, 1)
        else:
            self.antibody_unary_head = None
            self.antigen_unary_head = None
            self.specificity_head = None

    def _validate_inputs(
        self,
        antibody_features: Any,
        antigen_features: Any,
        antibody_mask: Any,
        antigen_mask: Any,
        candidate_pair_mask: Any,
    ) -> None:
        """Validate all model inputs before performing any tensor operations."""
        _validate_residue_tensor(
            antibody_features,
            antibody_mask,
            self.antibody_input_dim,
            "antibody",
        )
        _validate_residue_tensor(
            antigen_features,
            antigen_mask,
            self.antigen_input_dim,
            "antigen",
        )
        if int(antibody_features.shape[0]) != int(antigen_features.shape[0]):
            raise ValueError("Antibody and antigen batches must have equal size")
        if antibody_features.device != antigen_features.device:
            raise ValueError("All model inputs must share a device")
        if antibody_features.dtype != antigen_features.dtype:
            raise ValueError("Antibody and antigen features must share a dtype")
        if antibody_mask.device != antigen_mask.device:
            raise ValueError("All residue masks must share a device")
        _validate_pair_mask(candidate_pair_mask, antibody_mask, antigen_mask)

    def _pair_state_chunk(
        self,
        antibody: Any,
        antigen_add: Any,
        antigen_gate: Any,
        chunk_mask: Any,
        inter_chain_features: Any | None = None,
    ) -> Any:
        """Construct one candidate-masked antibody-axis pair-state chunk."""
        antibody_add = self.antibody_pair_add(antibody).unsqueeze(2)
        antibody_gate = self.antibody_pair_gate(antibody).unsqueeze(2)
        pair_seed = (
            antibody_add
            + antigen_add.unsqueeze(1)
            + antibody_gate * antigen_gate.unsqueeze(1)
        )
        if inter_chain_features is not None:
            pair_seed = pair_seed + inter_chain_features
        pair_state = self.pair_refinement(pair_seed)
        return pair_state.masked_fill(~chunk_mask.unsqueeze(-1), 0.0)

    def _decode_pair_state(
        self,
        pair_state: Any,
        chunk_mask: Any,
        enabled_heads: frozenset[str],
    ) -> dict[str, Any]:
        """Execute only requested prediction heads on a prepared pair state."""
        result = {"pair_representations": pair_state}
        if "distance" in enabled_heads:
            result["distance_bin_logits"] = self.distance_head(
                pair_state
            ).masked_fill(
                ~chunk_mask.unsqueeze(-1),
                0.0,
            )
        if "orientation" in enabled_heads:
            result["orientation_bin_logits"] = self.orientation_head(
                pair_state
            ).reshape(
                int(pair_state.shape[0]),
                int(pair_state.shape[1]),
                int(pair_state.shape[2]),
                self.num_orientation_axes,
                self.num_orientation_bins,
            ).masked_fill(
                ~chunk_mask.unsqueeze(-1).unsqueeze(-1),
                0.0,
            )
        if "interaction" in enabled_heads:
            result["pair_interactions"] = self.interaction_head(
                pair_state
            ).squeeze(-1).masked_fill(~chunk_mask, 0.0)
            if self.specificity_head is not None:
                result["specificity_interaction_residual_logits"] = (
                    self.specificity_head(pair_state)
                    .squeeze(-1)
                    .masked_fill(~chunk_mask, 0.0)
                )
        return result

    def _pair_chunk(
        self,
        antibody: Any,
        antigen_add: Any,
        antigen_gate: Any,
        chunk_mask: Any,
        enabled_heads: frozenset[str],
        inter_chain_features: Any | None = None,
    ) -> dict[str, Any]:
        """Construct and decode one antibody-axis chunk."""
        pair_state = self._pair_state_chunk(
            antibody,
            antigen_add,
            antigen_gate,
            chunk_mask,
            inter_chain_features=inter_chain_features,
        )
        return self._decode_pair_state(pair_state, chunk_mask, enabled_heads)

    def _validate_enabled_heads(
        self,
        enabled_heads: Collection[str] | None,
    ) -> frozenset[str]:
        """Return a nonempty, dependency-closed set of decoder heads."""
        allowed = frozenset({"contact", "distance", "orientation", "interaction"})
        if enabled_heads is None:
            return allowed
        if isinstance(enabled_heads, (str, bytes)) or not isinstance(
            enabled_heads,
            Collection,
        ):
            raise TypeError("enabled_heads must be a collection of head names")
        normalized = frozenset(enabled_heads)
        if not normalized:
            raise ValueError("enabled_heads must contain at least one head")
        if any(not isinstance(name, str) for name in normalized):
            raise ValueError("enabled_heads must contain only strings")
        unknown = sorted(normalized - allowed)
        if unknown:
            raise ValueError(f"enabled_heads contains unknown heads: {unknown}")
        if "contact" in normalized:
            if (
                self.contact_logit_mode == "distance_mass"
                and "distance" not in normalized
            ):
                raise ValueError(
                    "The contact head requires the distance head because contact "
                    "probability is derived from distance-bin mass"
                )
            if (
                self.contact_logit_mode == "unary_plus_interaction_residual"
                and "interaction" not in normalized
            ):
                raise ValueError(
                    "The decomposed contact head requires the interaction head"
                )
        return normalized

    def forward(
        self,
        antibody_features: Any,
        antigen_features: Any,
        antibody_mask: Any,
        antigen_mask: Any,
        candidate_pair_mask: Any,
        *,
        return_pair_representations: bool = True,
        enabled_heads: Collection[str] | None = None,
        antibody_backbone: Any | None = None,
        antigen_backbone: Any | None = None,
        antibody_chain_ids: Any | None = None,
        antigen_chain_ids: Any | None = None,
        antibody_positions: Any | None = None,
        antigen_positions: Any | None = None,
        return_intra_chain_attention: bool = False,
    ) -> dict[str, Any]:
        """Return only requested candidate-masked pair-geometry predictions."""
        self._validate_inputs(
            antibody_features,
            antigen_features,
            antibody_mask,
            antigen_mask,
            candidate_pair_mask,
        )
        if not isinstance(return_pair_representations, bool):
            raise ValueError("return_pair_representations must be boolean")
        enabled_heads = self._validate_enabled_heads(enabled_heads)

        antibody = self.antibody_projection(
            antibody_features.masked_fill(~antibody_mask.unsqueeze(-1), 0.0)
        )
        antigen = self.antigen_projection(
            antigen_features.masked_fill(~antigen_mask.unsqueeze(-1), 0.0)
        )
        antibody = antibody.masked_fill(~antibody_mask.unsqueeze(-1), 0.0)
        antigen = antigen.masked_fill(~antigen_mask.unsqueeze(-1), 0.0)
        intra_chain_details = None
        if self.intra_chain_geometry is not None:
            if antibody_backbone is None or antigen_backbone is None:
                raise ValueError(
                    "antibody_backbone and antigen_backbone are required when "
                    "intra-chain geometry is enabled"
                )
            antibody, antigen, intra_chain_details = self.intra_chain_geometry(
                antibody,
                antigen,
                antibody_mask,
                antigen_mask,
                antibody_backbone,
                antigen_backbone,
                antibody_chain_ids=antibody_chain_ids,
                antigen_chain_ids=antigen_chain_ids,
                antibody_positions=antibody_positions,
                antigen_positions=antigen_positions,
                return_attention=return_intra_chain_attention,
            )
        elif any(
            value is not None
            for value in (
                antibody_backbone,
                antigen_backbone,
                antibody_chain_ids,
                antigen_chain_ids,
                antibody_positions,
                antigen_positions,
            )
        ):
            raise ValueError(
                "Set intra_chain_geometry_layers > 0 before supplying structure inputs"
            )
        antibody_monomer = None
        antigen_monomer = None
        antibody_unary_logits = None
        antigen_unary_logits = None
        if self.contact_logit_mode == "unary_plus_interaction_residual":
            if self.antibody_unary_head is None or self.antigen_unary_head is None:
                raise RuntimeError("Decomposed contact heads were not initialized")
            antibody_monomer = self.antibody_final_norm(antibody).masked_fill(
                ~antibody_mask.unsqueeze(-1),
                0.0,
            )
            antigen_monomer = self.antigen_final_norm(antigen).masked_fill(
                ~antigen_mask.unsqueeze(-1),
                0.0,
            )
            antibody_unary_logits = self.antibody_unary_head(
                antibody_monomer
            ).squeeze(-1).masked_fill(~antibody_mask, 0.0)
            antigen_unary_logits = self.antigen_unary_head(
                antigen_monomer
            ).squeeze(-1).masked_fill(~antigen_mask, 0.0)
        for fusion_layer in self.fusion_layers:
            antibody, antigen = fusion_layer(
                antibody,
                antigen,
                antibody_mask,
                antigen_mask,
            )
        antibody = self.antibody_final_norm(antibody).masked_fill(
            ~antibody_mask.unsqueeze(-1),
            0.0,
        )
        antigen = self.antigen_final_norm(antigen).masked_fill(
            ~antigen_mask.unsqueeze(-1),
            0.0,
        )

        antigen_add = self.antigen_pair_add(antigen)
        antigen_gate = self.antigen_pair_gate(antigen)
        
        inter_chain_features = None
        if antibody_backbone is not None and antigen_backbone is not None:
            inter_geom = build_inter_chain_geometry(
                antibody_backbone,
                antigen_backbone,
                antibody_mask,
                antigen_mask,
                num_rbf=self.inter_chain_geometry_num_rbf,
                distance_max=self.inter_chain_geometry_distance_max,
            )
            raw_features = torch.cat([
                inter_geom.distance_rbf,
                inter_geom.unit_direction,
                inter_geom.displacement_local,
                inter_geom.relative_rotation.flatten(-2, -1),
            ], dim=-1)
            inter_chain_features = self.inter_chain_projection(raw_features)

        pair_chunks: list[Any] = []
        head_chunks: dict[str, list[Any]] = {
            name: []
            for name in (
                "distance_bin_logits",
                "orientation_bin_logits",
                "pair_interactions",
                "specificity_interaction_residual_logits",
            )
        }
        antibody_length = int(antibody.shape[1])
        chunk_ranges = tuple(
            (
                start,
                min(start + self.pair_chunk_size, antibody_length),
            )
            for start in range(0, antibody_length, self.pair_chunk_size)
        )
        if self.pair_update_layers:
            initial_pair_chunks = [
                self._pair_state_chunk(
                    antibody[:, start:end],
                    antigen_add,
                    antigen_gate,
                    candidate_pair_mask[:, start:end],
                    inter_chain_features=inter_chain_features[:, start:end] if inter_chain_features is not None else None,
                )
                for start, end in chunk_ranges
            ]
            updated_pair_state = torch.cat(initial_pair_chunks, dim=1)
            for pair_update_layer in self.pair_update_layers:
                updated_pair_state = pair_update_layer(
                    updated_pair_state,
                    candidate_pair_mask,
                )
            for start, end in chunk_ranges:
                chunk = self._decode_pair_state(
                    updated_pair_state[:, start:end],
                    candidate_pair_mask[:, start:end],
                    enabled_heads,
                )
                for name in head_chunks:
                    if name in chunk:
                        head_chunks[name].append(chunk[name])
            pair_representations = (
                updated_pair_state if return_pair_representations else None
            )
        else:
            for start, end in chunk_ranges:
                chunk = self._pair_chunk(
                    antibody[:, start:end],
                    antigen_add,
                    antigen_gate,
                    candidate_pair_mask[:, start:end],
                    enabled_heads,
                    inter_chain_features=inter_chain_features[:, start:end] if inter_chain_features is not None else None,
                )
                if return_pair_representations:
                    pair_chunks.append(chunk["pair_representations"])
                for name in head_chunks:
                    if name in chunk:
                        head_chunks[name].append(chunk[name])
            pair_representations = (
                torch.cat(pair_chunks, dim=1)
                if return_pair_representations
                else None
            )
        output: dict[str, Any] = {
            "pair_orientation": PAIR_ORIENTATION,
            "executed_heads": tuple(
                name
                for name in ("contact", "distance", "orientation", "interaction")
                if name in enabled_heads
            ),
            "geometry_schema": self.geometry_schema,
            "geometry_schema_metadata": self.geometry_schema.canonical_metadata(),
            "geometry_schema_sha256": self.geometry_schema.sha256,
            "candidate_pair_mask": candidate_pair_mask,
            "pair_representations": pair_representations,
            "epitope_mask": candidate_pair_mask.any(dim=1),
            "paratope_mask": candidate_pair_mask.any(dim=2),
            "antibody_residue_representations": antibody,
            "antigen_residue_representations": antigen,
            "contact_logit_mode": self.contact_logit_mode,
            "num_pair_update_layers": self.num_pair_update_layers,
            "pair_update_order": (
                "antigen_axis_within_antibody_row",
                "antibody_axis_within_antigen_column",
                "feed_forward",
            ),
            "intra_chain_geometry_enabled": self.intra_chain_geometry is not None,
            "intra_chain_geometry": intra_chain_details,
        }
        if antibody_monomer is not None and antigen_monomer is not None:
            output["antibody_monomer_representations"] = antibody_monomer
            output["antigen_monomer_representations"] = antigen_monomer
        distance_logits = (
            torch.cat(head_chunks["distance_bin_logits"], dim=1)
            if head_chunks["distance_bin_logits"]
            else None
        )
        orientation_logits = (
            torch.cat(head_chunks["orientation_bin_logits"], dim=1)
            if head_chunks["orientation_bin_logits"]
            else None
        )
        pair_interactions = (
            torch.cat(head_chunks["pair_interactions"], dim=1)
            if head_chunks["pair_interactions"]
            else None
        )
        specificity_interactions = (
            torch.cat(
                head_chunks["specificity_interaction_residual_logits"],
                dim=1,
            )
            if head_chunks["specificity_interaction_residual_logits"]
            else None
        )
        distance_probabilities = None
        orientation_probabilities = None
        if distance_logits is not None:
            distance_probabilities = torch.softmax(
                distance_logits,
                dim=-1,
            ).masked_fill(~candidate_pair_mask.unsqueeze(-1), 0.0)
            output["distance_bin_logits"] = distance_logits
            output["distance_bin_probabilities"] = distance_probabilities
            eps = torch.finfo(distance_probabilities.dtype).eps
            output["distance_uncertainty"] = -(
                distance_probabilities
                * distance_probabilities.clamp_min(eps).log()
            ).sum(dim=-1).div(math.log(self.num_distance_bins)).masked_fill(
                ~candidate_pair_mask,
                0.0,
            )
        if orientation_logits is not None:
            orientation_probabilities = torch.softmax(
                orientation_logits,
                dim=-1,
            ).masked_fill(
                ~candidate_pair_mask.unsqueeze(-1).unsqueeze(-1),
                0.0,
            )
            output["orientation_bin_logits"] = orientation_logits
            output["orientation_bin_probabilities"] = orientation_probabilities
            eps = torch.finfo(orientation_probabilities.dtype).eps
            output["orientation_uncertainty"] = -(
                orientation_probabilities
                * orientation_probabilities.clamp_min(eps).log()
            ).sum(dim=-1).div(math.log(self.num_orientation_bins)).mean(
                dim=-1
            ).masked_fill(~candidate_pair_mask, 0.0)
        if "contact" in enabled_heads:
            if self.contact_logit_mode == "distance_mass":
                if distance_logits is None:
                    raise RuntimeError("Internal contact-to-distance dependency failed")
                contact_log_mass = torch.logsumexp(
                    distance_logits[..., : self.contact_distance_bin_count],
                    dim=-1,
                )
                noncontact_log_mass = torch.logsumexp(
                    distance_logits[..., self.contact_distance_bin_count :],
                    dim=-1,
                )
                contact_logits = (
                    contact_log_mass - noncontact_log_mass
                ).masked_fill(
                    ~candidate_pair_mask,
                    0.0,
                )
                output["contact_logit_source"] = "distance_bin_mass_ratio"
            else:
                if (
                    antibody_unary_logits is None
                    or antigen_unary_logits is None
                    or pair_interactions is None
                ):
                    raise RuntimeError("Internal contact decomposition failed")
                contact_logits = (
                    antibody_unary_logits.unsqueeze(2)
                    + antigen_unary_logits.unsqueeze(1)
                    + pair_interactions
                ).masked_fill(~candidate_pair_mask, 0.0)
                output["antibody_unary_logits"] = antibody_unary_logits
                output["antigen_unary_logits"] = antigen_unary_logits
                output["contact_interaction_residual_logits"] = pair_interactions
                output["contact_logit_source"] = (
                    "antibody_unary_plus_antigen_unary_plus_interaction_residual"
                )
            output["contact_logits"] = contact_logits
            output["contact_probabilities"] = _masked_probabilities(
                contact_logits,
                candidate_pair_mask,
            )
            antigen_by_antibody_logits = contact_logits.transpose(1, 2)
            antigen_by_antibody_mask = candidate_pair_mask.transpose(1, 2)
            epitope_probability, paratope_probability = pool_pair_marginals(
                antigen_by_antibody_logits,
                antigen_by_antibody_mask,
                pooling=self.marginal_pooling,
                top_k=self.marginal_top_k,
                temperature=self.marginal_temperature,
            )
            output["epitope_probability"] = epitope_probability
            output["paratope_probability"] = paratope_probability
        if (
            distance_probabilities is not None
            and orientation_probabilities is not None
        ):
            pair_uncertainty = (
                0.5
                * (
                    output["distance_uncertainty"]
                    + output["orientation_uncertainty"]
                )
            ).masked_fill(~candidate_pair_mask, 0.0)
            output["pair_uncertainty"] = pair_uncertainty
            output["pair_confidence"] = (1.0 - pair_uncertainty).masked_fill(
                ~candidate_pair_mask,
                0.0,
            )
        if pair_interactions is not None:
            centered_pair_interactions, pair_interaction_score = (
                double_centered_partner_score(
                    pair_interactions,
                    candidate_pair_mask,
                    temperature=self.partner_score_temperature,
                    centering_iterations=self.centering_iterations,
                )
            )
            output["pair_interactions"] = pair_interactions
            output["double_centered_pair_interactions"] = (
                centered_pair_interactions
            )
            output["pair_interaction_score"] = pair_interaction_score
        if specificity_interactions is not None:
            centered_specificity, partner_score = double_centered_partner_score(
                specificity_interactions,
                candidate_pair_mask,
                temperature=self.partner_score_temperature,
                centering_iterations=self.centering_iterations,
            )
            output["partner_score"] = partner_score
            output["specificity_interaction_residual_logits"] = (
                specificity_interactions
            )
            output[
                "double_centered_specificity_interaction_residual_logits"
            ] = centered_specificity
            output["partner_score_source"] = (
                "double_centered_independent_specificity_interaction_residual"
            )
        elif pair_interactions is not None:
            output["partner_score"] = output["pair_interaction_score"]
            output["partner_score_source"] = "double_centered_pair_interactions"
        return output
