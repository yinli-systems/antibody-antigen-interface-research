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

from .contact_gauge import gauge_fixed_pair_components
from .intra_chain_geometry import IntraChainGeometryEncoder, build_backbone_frames
from .paired_sequence_features import PPLMPairedConditioner
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
        local_pose_rotary_channels: int | None = None,
        geometry_attention_graph_mode: str = "spatial",
        num_rbf: int = 16,
        neighbor_k: int | None = 32,
        neighbor_radius: float | None = 16.0,
        sequence_window: int = 32,
        dropout: float = 0.0,
        attention_kernel: str = "reference",
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
            attention_kernel=attention_kernel,
        )
        self.geometry_encoder = IntraChainGeometryEncoder(
            d_model,
            num_heads,
            num_layers,
            geometry_feature_mode=geometry_feature_mode,
            geometry_bias_mode=geometry_bias_mode,
            ipa_num_points=ipa_num_points,
            local_pose_rotary_channels=local_pose_rotary_channels,
            attention_graph_mode=geometry_attention_graph_mode,
            num_rbf=num_rbf,
            neighbor_k=neighbor_k,
            neighbor_radius=neighbor_radius,
            sequence_window=sequence_window,
            dropout=dropout,
            attention_kernel=attention_kernel,
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


def _validated_distance_bins(
    boundaries: tuple[float, ...], contact_cutoff: float
) -> tuple[tuple[float, ...], int]:
    if not boundaries:
        raise ValueError("distance_bin_boundaries cannot be empty")
    if any(not math.isfinite(value) or value <= 0 for value in boundaries):
        raise ValueError("distance bin boundaries must be finite and positive")
    if any(right <= left for left, right in zip(boundaries, boundaries[1:])):
        raise ValueError("distance bin boundaries must be strictly increasing")
    if not math.isfinite(contact_cutoff) or contact_cutoff <= 0:
        raise ValueError("contact_distance_cutoff must be finite and positive")
    matching = [index for index, value in enumerate(boundaries) if value <= contact_cutoff]
    if not matching or not math.isclose(
        boundaries[matching[-1]], contact_cutoff, rel_tol=0.0, abs_tol=1e-6
    ):
        raise ValueError("contact_distance_cutoff must equal a distance bin boundary")
    return boundaries, matching[-1] + 1


def contact_logit_from_distogram(
    distogram_logits: Any,
    *,
    contact_bin_count: int,
) -> Any:
    """Collapse one categorical distance distribution into a binary contact logit."""
    _require_torch()
    if distogram_logits.ndim != 4 or int(distogram_logits.shape[-1]) < 2:
        raise ValueError("distogram_logits must have shape [batch, ab, ag, bin]")
    bin_count = int(distogram_logits.shape[-1])
    if (
        isinstance(contact_bin_count, bool)
        or not isinstance(contact_bin_count, int)
        or not 0 < contact_bin_count < bin_count
    ):
        raise ValueError("contact_bin_count must split the distance bins")
    contact = torch.logsumexp(
        distogram_logits[..., :contact_bin_count], dim=-1
    )
    non_contact = torch.logsumexp(
        distogram_logits[..., contact_bin_count:], dim=-1
    )
    return contact - non_contact


def inject_pair_contact_prior(
    components: Mapping[str, Any],
    pair_prior_logits: Any | None,
    pair_mask: Any,
) -> dict[str, Any]:
    """Inject a joint partner prior without breaking the distogram contract.

    For a binary head, the prior is simply added to both total and
    interaction-only contact logits.  For a distogram head, adding ``p/2`` to
    every contact bin and ``-p/2`` to every non-contact bin shifts the exact
    near-vs-far log-odds by ``p`` while preserving the relative distribution
    within each group.  Contact therefore remains a deterministic projection
    of one distance head.
    """

    torch_module = _require_torch()
    updated = dict(components)
    if pair_prior_logits is None:
        return updated
    if (
        not torch_module.is_tensor(pair_prior_logits)
        or tuple(pair_prior_logits.shape) != tuple(pair_mask.shape)
        or pair_mask.dtype != torch_module.bool
        or not bool(torch_module.isfinite(pair_prior_logits).all())
    ):
        raise ValueError("pair contact prior must be finite and align with pair_mask")
    prior = pair_prior_logits.masked_fill(~pair_mask, 0.0)
    distogram_logits = components.get("distogram_logits")
    if distogram_logits is None:
        updated["logits"] = (components["logits"] + prior).masked_fill(
            ~pair_mask, 0.0
        )
        updated["interaction_logits"] = (
            components["interaction_logits"] + prior
        ).masked_fill(~pair_mask, 0.0)
        return updated

    contact_bin_count = components.get("contact_bin_count")
    if (
        isinstance(contact_bin_count, bool)
        or not isinstance(contact_bin_count, int)
        or not 0 < contact_bin_count < int(distogram_logits.shape[-1])
    ):
        raise ValueError("distogram components lack a valid contact bin split")
    half = 0.5 * prior[..., None]
    shift = torch_module.cat(
        (
            half.expand(*half.shape[:-1], contact_bin_count),
            (-half).expand(
                *half.shape[:-1], int(distogram_logits.shape[-1]) - contact_bin_count
            ),
        ),
        dim=-1,
    )
    distance_logits = (distogram_logits + shift).masked_fill(
        ~pair_mask[..., None], 0.0
    )
    interaction_distance_logits = (
        components["interaction_distogram_logits"] + shift
    ).masked_fill(~pair_mask[..., None], 0.0)
    updated["distogram_logits"] = distance_logits
    updated["interaction_distogram_logits"] = interaction_distance_logits
    distogram_contact_logits = contact_logit_from_distogram(
        distance_logits, contact_bin_count=contact_bin_count
    ).masked_fill(~pair_mask, 0.0)
    distogram_interaction_logits = contact_logit_from_distogram(
        interaction_distance_logits, contact_bin_count=contact_bin_count
    ).masked_fill(~pair_mask, 0.0)
    updated["distogram_contact_logits"] = distogram_contact_logits
    updated["distogram_interaction_logits"] = distogram_interaction_logits
    if components.get("direct_contact_logits") is not None:
        updated["direct_contact_logits"] = (
            components["direct_contact_logits"] + prior
        ).masked_fill(~pair_mask, 0.0)
        updated["direct_interaction_logits"] = (
            components["direct_interaction_logits"] + prior
        ).masked_fill(~pair_mask, 0.0)
        updated["logits"] = updated["direct_contact_logits"]
        updated["interaction_logits"] = updated["direct_interaction_logits"]
    else:
        updated["logits"] = distogram_contact_logits
        updated["interaction_logits"] = distogram_interaction_logits
    return updated


class PairDistogramHead(_ModuleBase):
    """Low-rank inter-chain distogram whose near bins define contact probability.

    The head never receives a cross-chain coordinate difference.  It predicts a
    categorical distance distribution solely from the two independently
    encoded monomer streams.  Binary contact is therefore a deterministic
    projection of the same logits, rather than a second potentially
    contradictory head.
    """

    def __init__(
        self,
        d_model: int,
        pair_dim: int | None = None,
        *,
        distance_bin_boundaries: tuple[float, ...] = (
            2.0,
            3.0,
            4.0,
            4.5,
            5.0,
            6.0,
            8.0,
            10.0,
            12.0,
            16.0,
            20.0,
        ),
        contact_distance_cutoff: float = 4.5,
    ) -> None:
        _require_torch()
        super().__init__()
        d_model = _positive_integer(d_model, "d_model")
        pair_dim = d_model if pair_dim is None else _positive_integer(pair_dim, "pair_dim")
        boundaries, contact_bin_count = _validated_distance_bins(
            tuple(float(value) for value in distance_bin_boundaries),
            float(contact_distance_cutoff),
        )
        self.pair_dim = pair_dim
        self.bin_count = len(boundaries) + 1
        self.contact_bin_count = contact_bin_count
        self.antibody_norm = nn.LayerNorm(d_model)
        self.antigen_norm = nn.LayerNorm(d_model)
        self.antibody_query = nn.Linear(
            d_model, self.bin_count * pair_dim, bias=False
        )
        self.antigen_key = nn.Linear(
            d_model, self.bin_count * pair_dim, bias=False
        )
        self.antibody_unary = nn.Linear(d_model, self.bin_count)
        self.antigen_unary = nn.Linear(d_model, self.bin_count)
        self.distance_bias = nn.Parameter(torch.zeros(self.bin_count))
        self.register_buffer(
            "distance_bin_boundaries",
            torch.tensor(boundaries, dtype=torch.float32),
            persistent=True,
        )

    @property
    def contact_distance_cutoff(self) -> float:
        return float(
            self.distance_bin_boundaries[self.contact_bin_count - 1].item()
        )

    def components(
        self,
        antibody: Any,
        antigen: Any,
        pair_mask: Any,
    ) -> dict[str, Any]:
        batch_size, antibody_count, _ = antibody.shape
        antigen_count = int(antigen.shape[1])
        query = self.antibody_query(self.antibody_norm(antibody)).view(
            batch_size, antibody_count, self.bin_count, self.pair_dim
        )
        key = self.antigen_key(self.antigen_norm(antigen)).view(
            batch_size, antigen_count, self.bin_count, self.pair_dim
        )
        interaction_distance_logits = torch.einsum(
            "birk,bjrk->bijr", query, key
        ) / math.sqrt(self.pair_dim)
        antibody_unary = self.antibody_unary(antibody)[:, :, None, :]
        antigen_unary = self.antigen_unary(antigen)[:, None, :, :]
        distance_logits = (
            interaction_distance_logits
            + antibody_unary
            + antigen_unary
            + self.distance_bias
        )
        distance_logits = distance_logits.masked_fill(~pair_mask[..., None], 0.0)
        interaction_distance_logits = interaction_distance_logits.masked_fill(
            ~pair_mask[..., None], 0.0
        )
        return {
            "logits": contact_logit_from_distogram(
                distance_logits, contact_bin_count=self.contact_bin_count
            ).masked_fill(~pair_mask, 0.0),
            "interaction_logits": contact_logit_from_distogram(
                interaction_distance_logits,
                contact_bin_count=self.contact_bin_count,
            ).masked_fill(~pair_mask, 0.0),
            "distogram_logits": distance_logits,
            "interaction_distogram_logits": interaction_distance_logits,
            "antibody_unary_logits": contact_logit_from_distogram(
                antibody_unary, contact_bin_count=self.contact_bin_count
            ),
            "antigen_unary_logits": contact_logit_from_distogram(
                antigen_unary, contact_bin_count=self.contact_bin_count
            ),
            "antibody_unary_distogram_logits": antibody_unary,
            "antigen_unary_distogram_logits": antigen_unary,
            "distance_bias": self.distance_bias,
            "distance_bin_boundaries": self.distance_bin_boundaries,
            "contact_bin_count": self.contact_bin_count,
        }

    def forward(self, antibody: Any, antigen: Any, pair_mask: Any) -> Any:
        return self.components(antibody, antigen, pair_mask)["logits"]


class JointPairContactDistogramHead(_ModuleBase):
    """Shared low-rank pair factors with direct-contact and distance outputs.

    The direct Bernoulli head remains the primary sparse-interface predictor.
    A categorical distogram is decoded from the *same* normalized residue
    factors, allowing an auxiliary distance objective and an explicit
    probability-consistency loss without pretending the two predictions are
    independent models.  No cross-chain coordinates are inputs to this head.
    """

    def __init__(
        self,
        d_model: int,
        pair_dim: int | None = None,
        *,
        distance_bin_boundaries: tuple[float, ...] = (
            2.0,
            3.0,
            4.0,
            4.5,
            5.0,
            6.0,
            8.0,
            10.0,
            12.0,
            16.0,
            20.0,
        ),
        contact_distance_cutoff: float = 4.5,
    ) -> None:
        _require_torch()
        super().__init__()
        d_model = _positive_integer(d_model, "d_model")
        pair_dim = d_model if pair_dim is None else _positive_integer(pair_dim, "pair_dim")
        boundaries, contact_bin_count = _validated_distance_bins(
            tuple(float(value) for value in distance_bin_boundaries),
            float(contact_distance_cutoff),
        )
        self.pair_dim = pair_dim
        self.bin_count = len(boundaries) + 1
        self.contact_bin_count = contact_bin_count
        self.antibody_norm = nn.LayerNorm(d_model)
        self.antigen_norm = nn.LayerNorm(d_model)
        self.antibody_factor = nn.Linear(d_model, pair_dim, bias=False)
        self.antigen_factor = nn.Linear(d_model, pair_dim, bias=False)
        self.distance_interaction_weight = nn.Parameter(
            torch.empty(self.bin_count, pair_dim)
        )
        nn.init.normal_(
            self.distance_interaction_weight,
            mean=0.0,
            std=1.0 / math.sqrt(pair_dim),
        )
        self.antibody_contact_unary = nn.Linear(d_model, 1)
        self.antigen_contact_unary = nn.Linear(d_model, 1)
        self.contact_bias = nn.Parameter(torch.tensor(-4.0))
        self.antibody_distance_unary = nn.Linear(d_model, self.bin_count)
        self.antigen_distance_unary = nn.Linear(d_model, self.bin_count)
        self.distance_bias = nn.Parameter(torch.zeros(self.bin_count))
        self.register_buffer(
            "distance_bin_boundaries",
            torch.tensor(boundaries, dtype=torch.float32),
            persistent=True,
        )

    def components(
        self,
        antibody: Any,
        antigen: Any,
        pair_mask: Any,
    ) -> dict[str, Any]:
        antibody_normalized = self.antibody_norm(antibody)
        antigen_normalized = self.antigen_norm(antigen)
        antibody_factor = self.antibody_factor(antibody_normalized)
        antigen_factor = self.antigen_factor(antigen_normalized)
        direct_interaction = torch.einsum(
            "bid,bjd->bij", antibody_factor, antigen_factor
        ) / math.sqrt(self.pair_dim)
        antibody_contact_unary = self.antibody_contact_unary(antibody)
        antigen_contact_unary = self.antigen_contact_unary(antigen).transpose(1, 2)
        direct_contact = (
            direct_interaction
            + antibody_contact_unary
            + antigen_contact_unary
            + self.contact_bias
        ).masked_fill(~pair_mask, 0.0)
        distance_interaction = torch.einsum(
            "bid,bjd,kd->bijk",
            antibody_factor,
            antigen_factor,
            self.distance_interaction_weight,
        ) / math.sqrt(self.pair_dim)
        antibody_distance_unary = self.antibody_distance_unary(antibody)[:, :, None, :]
        antigen_distance_unary = self.antigen_distance_unary(antigen)[:, None, :, :]
        distance_logits = (
            distance_interaction
            + antibody_distance_unary
            + antigen_distance_unary
            + self.distance_bias
        ).masked_fill(~pair_mask[..., None], 0.0)
        distogram_contact = contact_logit_from_distogram(
            distance_logits, contact_bin_count=self.contact_bin_count
        ).masked_fill(~pair_mask, 0.0)
        return {
            "logits": direct_contact,
            "interaction_logits": direct_interaction.masked_fill(~pair_mask, 0.0),
            "direct_contact_logits": direct_contact,
            "direct_interaction_logits": direct_interaction.masked_fill(
                ~pair_mask, 0.0
            ),
            "distogram_contact_logits": distogram_contact,
            "distogram_interaction_logits": contact_logit_from_distogram(
                distance_interaction,
                contact_bin_count=self.contact_bin_count,
            ).masked_fill(~pair_mask, 0.0),
            "distogram_logits": distance_logits,
            "interaction_distogram_logits": distance_interaction.masked_fill(
                ~pair_mask[..., None], 0.0
            ),
            "antibody_unary_logits": antibody_contact_unary,
            "antigen_unary_logits": antigen_contact_unary,
            "antibody_unary_distogram_logits": antibody_distance_unary,
            "antigen_unary_distogram_logits": antigen_distance_unary,
            "contact_bias": self.contact_bias,
            "distance_bias": self.distance_bias,
            "distance_bin_boundaries": self.distance_bin_boundaries,
            "contact_bin_count": self.contact_bin_count,
        }

    def forward(self, antibody: Any, antigen: Any, pair_mask: Any) -> Any:
        return self.components(antibody, antigen, pair_mask)["logits"]


class ContactGatedCrossAttention(_ModuleBase):
    """One directional cross-attention update controlled by contact logits."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        *,
        dropout: float = 0.0,
        attention_kernel: str = "reference",
    ) -> None:
        _require_torch()
        super().__init__()
        d_model = _positive_integer(d_model, "d_model")
        num_heads = _positive_integer(num_heads, "num_heads")
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        if attention_kernel not in {"reference", "sdpa"}:
            raise ValueError("attention_kernel must be 'reference' or 'sdpa'")
        self.attention_kernel = attention_kernel
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
        contact_gate_multiplier: Any = 1.0,
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
        attention_mask = pair_mask[:, None]
        query_has_candidate = pair_mask.any(dim=-1)
        query_valid = receiver_mask & query_has_candidate
        attention = None
        content_only_attention = None
        multiplier = torch.as_tensor(
            contact_gate_multiplier,
            device=contact_logits.device,
            dtype=contact_logits.dtype,
        )
        if multiplier.numel() != 1 or not bool(torch.isfinite(multiplier)) or bool(
            multiplier < 0
        ):
            raise ValueError("contact_gate_multiplier must be finite and nonnegative")
        contact_bias = (
            multiplier
            *
            contact_scale[None, :, None, None]
            * contact_log_probability[:, None]
        )
        if self.attention_kernel == "sdpa" and not return_attention:
            attention_bias = contact_bias.masked_fill(
                ~attention_mask, float("-inf")
            )
            context = functional.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_bias,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=False,
            )
            context = context * query_valid[:, None, :, None].to(context.dtype)
        else:
            logits = (content_logits + contact_bias).masked_fill(
                ~attention_mask, torch.finfo(content_logits.dtype).min
            )
            attention = torch.softmax(logits, dim=-1)
            attention = attention * query_valid[:, None, :, None].to(
                attention.dtype
            )
            context = torch.einsum(
                "bhij,bhjd->bhid", self.dropout(attention), value
            )
            if return_attention:
                content_only_logits = content_logits.masked_fill(
                    ~attention_mask, torch.finfo(content_logits.dtype).min
                )
                content_only_attention = torch.softmax(
                    content_only_logits, dim=-1
                )
                content_only_attention = content_only_attention * query_valid[
                    :, None, :, None
                ].to(content_only_attention.dtype)
        context = context.transpose(1, 2).reshape(batch_size, receiver_count, self.d_model)
        gate = torch.tanh(self.residual_gate)
        updated = receiver + gate * self.dropout(self.output(context))
        updated = updated.masked_fill(~receiver_mask[..., None], 0.0)
        if not return_attention:
            return updated, None
        assert attention is not None and content_only_attention is not None
        return updated, {
            "weights": attention,
            "content_only_weights": content_only_attention,
            "contact_log_probability": contact_log_probability,
            "contact_scale_per_head": contact_scale,
            "contact_gate_multiplier": multiplier,
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
        attention_kernel: str = "reference",
    ) -> None:
        _require_torch()
        super().__init__()
        self.antibody_from_antigen = ContactGatedCrossAttention(
            d_model,
            num_heads,
            dropout=dropout,
            attention_kernel=attention_kernel,
        )
        self.antigen_from_antibody = ContactGatedCrossAttention(
            d_model,
            num_heads,
            dropout=dropout,
            attention_kernel=attention_kernel,
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
        contact_gate_multiplier: Any = 1.0,
        return_attention: bool = False,
    ) -> tuple[Any, Any, dict[str, Any] | None]:
        antibody_updated, antibody_details = self.antibody_from_antigen(
            antibody,
            antigen,
            antibody_mask,
            antigen_mask,
            contact_logits,
            pair_mask,
            contact_gate_multiplier=contact_gate_multiplier,
            return_attention=return_attention,
        )
        antigen_updated, antigen_details = self.antigen_from_antibody(
            antigen,
            antibody,
            antigen_mask,
            antibody_mask,
            contact_logits.transpose(1, 2),
            pair_mask.transpose(1, 2),
            contact_gate_multiplier=contact_gate_multiplier,
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


def masked_mean_probability_logits(logits: Any, mask: Any, *, dim: int) -> Any:
    """Logit of the masked arithmetic mean of pair-contact probabilities.

    This is the strict deterministic residue score used by contact-first models:
    first interpret every pair logit as a Bernoulli contact probability, then
    average only over authenticated candidate partners.  It is intentionally
    distinct from ``masked_logmeanexp`` (mean odds in log space) and noisy-OR
    (probability of at least one independent contact).
    """

    torch_module = _require_torch()
    if tuple(logits.shape) != tuple(mask.shape) or mask.dtype != torch_module.bool:
        raise ValueError("logits and boolean mask must have identical shapes")
    observed_count = mask.sum(dim=dim)
    count = observed_count.clamp_min(1).to(torch.float32)
    probability = (
        torch_module.sigmoid(logits.float()).masked_fill(~mask, 0.0).sum(dim=dim)
        / count
    )
    probability = probability.clamp(min=1e-7, max=1.0 - 1e-7)
    result = torch_module.logit(probability).to(logits.dtype)
    return result.masked_fill(observed_count == 0, 0.0)


def soft_coarse_pair_log_prior(
    paratope_logits: Any,
    epitope_logits: Any,
    antibody_mask: Any,
    antigen_mask: Any,
    *,
    probability_floor: float = 0.05,
    temperature: float = 1.0,
    stop_gradient: bool = True,
) -> dict[str, Any]:
    """Build a non-destructive coarse paratope-by-epitope routing prior.

    The first-pass residue predictions are *not* turned into a hard crop.  Their
    outer product is mixed with a strictly positive floor, so every authenticated
    residue pair remains available to the second-pass contact predictor.  The
    default stop-gradient route prevents the model from improving its objective
    merely by sharpening its own crop; sharing the residue heads with the final
    supervised prediction still lets the coarse predictor improve over training.

    The returned logarithm is suitable as an additive cross-attention bias.  It
    is deliberately not added to the final contact logits and is never used as a
    pair-label mask.
    """

    torch_module = _require_torch()
    if paratope_logits.ndim != 2 or epitope_logits.ndim != 2:
        raise ValueError("coarse residue logits must be [batch, residue]")
    if (
        tuple(paratope_logits.shape) != tuple(antibody_mask.shape)
        or tuple(epitope_logits.shape) != tuple(antigen_mask.shape)
        or antibody_mask.dtype != torch_module.bool
        or antigen_mask.dtype != torch_module.bool
    ):
        raise ValueError("coarse residue logits and boolean masks must align")
    if not math.isfinite(probability_floor) or not 0.0 < probability_floor <= 1.0:
        raise ValueError("coarse probability floor must lie in (0, 1]")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("coarse temperature must be finite and positive")
    paratope_probability = torch_module.sigmoid(
        paratope_logits.float() / float(temperature)
    ).masked_fill(~antibody_mask, 0.0)
    epitope_probability = torch_module.sigmoid(
        epitope_logits.float() / float(temperature)
    ).masked_fill(~antigen_mask, 0.0)
    pair_mask = antibody_mask[:, :, None] & antigen_mask[:, None, :]
    coarse_probability = (
        paratope_probability[:, :, None] * epitope_probability[:, None, :]
    )
    routed_probability = float(probability_floor) + (
        1.0 - float(probability_floor)
    ) * coarse_probability
    routed_probability = routed_probability.masked_fill(~pair_mask, 1.0)
    log_prior = routed_probability.clamp_min(1e-7).log()
    if stop_gradient:
        paratope_probability = paratope_probability.detach()
        epitope_probability = epitope_probability.detach()
        coarse_probability = coarse_probability.detach()
        log_prior = log_prior.detach()
    return {
        "log_prior": log_prior.to(paratope_logits.dtype),
        "pair_probability": coarse_probability.to(paratope_logits.dtype),
        "paratope_probability": paratope_probability.to(paratope_logits.dtype),
        "epitope_probability": epitope_probability.to(epitope_logits.dtype),
        "pair_mask": pair_mask,
        "probability_floor": float(probability_floor),
        "temperature": float(temperature),
        "stop_gradient": bool(stop_gradient),
    }


def masked_noisy_or_logits(logits: Any, mask: Any, *, dim: int) -> Any:
    """Return the stable logit of ``P(any contact)`` along one partner axis.

    Residue interface labels are logical OR marginals of pair-contact labels.
    This pooling therefore has an explicit probabilistic interpretation.  It is
    kept separate from ``masked_logmeanexp`` because noisy-OR is sensitive to
    partner length and must be compared, not silently substituted.
    """
    torch_module = _require_torch()
    if tuple(logits.shape) != tuple(mask.shape) or mask.dtype != torch_module.bool:
        raise ValueError("logits and boolean mask must have identical shapes")
    observed_count = mask.sum(dim=dim)
    # Accumulate in fp32 under bf16 autocast. log(1-sigmoid(x)) is
    # logsigmoid(-x); the probability of no contact is their product.
    log_none = functional.logsigmoid(-logits.float()).masked_fill(~mask, 0.0).sum(
        dim=dim
    )
    log_none = log_none.clamp(min=-30.0, max=-1e-7)
    log_any = torch_module.log(-torch_module.expm1(log_none))
    result = log_any - log_none
    return result.masked_fill(observed_count == 0, 0.0)


def masked_length_normalized_noisy_or_logits(
    logits: Any, mask: Any, *, dim: int
) -> Any:
    """A partner-length-normalized complement-of-geometric-survival score.

    Ordinary noisy-OR is the correct logical-OR marginal only under calibrated
    independent pair probabilities, but it grows mechanically with the number
    of candidate partner residues.  This comparator averages log-survival
    before taking the complement.  For a constant pair logit it returns that
    same logit regardless of partner length.  It is therefore an explicit
    length-normalized ablation, not a replacement probabilistic OR claim.
    """

    torch_module = _require_torch()
    if tuple(logits.shape) != tuple(mask.shape) or mask.dtype != torch_module.bool:
        raise ValueError("logits and boolean mask must have identical shapes")
    observed_count = mask.sum(dim=dim)
    count = observed_count.clamp_min(1).to(torch.float32)
    mean_log_none = (
        functional.logsigmoid(-logits.float()).masked_fill(~mask, 0.0).sum(dim=dim)
        / count
    )
    mean_log_none = mean_log_none.clamp(min=-30.0, max=-1e-7)
    log_any = torch_module.log(-torch_module.expm1(mean_log_none))
    result = log_any - mean_log_none
    return result.masked_fill(observed_count == 0, 0.0)


class ContactBottleneckContrastiveHead(_ModuleBase):
    """Pool monomer features through counterfactual partner-contact lift.

    An interaction-only logit is not a calibrated contact log-odds because the
    contact bias and monomer unary terms have been removed.  Treating it as a
    Bernoulli probability would make noisy-OR saturate on long partners.  This
    head instead computes the positive probability lift caused by the pre-cross
    pair interaction,

    ``relu(sigmoid(total) - sigmoid(total - interaction))``.

    Noisy-OR is applied to this bounded counterfactual lift.  Pure unary
    propensity therefore produces zero pooling weight, while a positive
    pair-specific contribution can select residues without any true interface
    labels. ``stop_gradient`` and ``joint`` explicitly ablate whether retrieval
    loss may shape the contact predictor.
    """

    def __init__(
        self,
        d_model: int,
        projection_dim: int,
        *,
        gradient_mode: str = "stop_gradient",
    ) -> None:
        _require_torch()
        super().__init__()
        d_model = _positive_integer(d_model, "d_model")
        projection_dim = _positive_integer(projection_dim, "projection_dim")
        if gradient_mode not in {"stop_gradient", "joint"}:
            raise ValueError(
                "contact bottleneck gradient_mode must be stop_gradient or joint"
            )
        self.gradient_mode = gradient_mode
        self.antibody_projection = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, projection_dim, bias=False),
        )
        self.antigen_projection = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, projection_dim, bias=False),
        )

    @staticmethod
    def _counterfactual_contact_lift(
        total_logits: Any,
        interaction_logits: Any,
        pair_mask: Any,
    ) -> Any:
        torch_module = _require_torch()
        if (
            tuple(total_logits.shape) != tuple(interaction_logits.shape)
            or tuple(total_logits.shape) != tuple(pair_mask.shape)
            or pair_mask.dtype != torch_module.bool
        ):
            raise ValueError(
                "contact bottleneck total, interaction, and pair mask must align"
            )
        total = total_logits.float()
        without_interaction = total - interaction_logits.float()
        lift = (torch_module.sigmoid(total) - torch_module.sigmoid(without_interaction))
        return lift.clamp(min=0.0, max=1.0 - 1e-7).masked_fill(~pair_mask, 0.0)

    @staticmethod
    def _noisy_or_weights(lift: Any, mask: Any, *, dim: int) -> Any:
        torch_module = _require_torch()
        if tuple(lift.shape) != tuple(mask.shape) or mask.dtype != torch_module.bool:
            raise ValueError("contact lift and pair mask must align")
        candidate_count = mask.sum(dim=dim)
        # fp32 prevents bf16 accumulation from turning long survival products
        # into exact zero.  -expm1(log_none) is numerically stable near zero.
        log_none = torch_module.log1p(-lift.float()).masked_fill(~mask, 0.0).sum(
            dim=dim
        )
        log_none = log_none.clamp(min=-30.0, max=0.0)
        weights = -torch_module.expm1(log_none)
        return weights.masked_fill(candidate_count == 0, 0.0)

    @staticmethod
    def _weighted_pool(features: Any, weights: Any, residue_mask: Any) -> Any:
        torch_module = _require_torch()
        if (
            features.ndim != 3
            or tuple(features.shape[:2]) != tuple(weights.shape)
            or tuple(weights.shape) != tuple(residue_mask.shape)
            or residue_mask.dtype != torch_module.bool
        ):
            raise ValueError("contact bottleneck features, weights, and mask must align")
        weights = weights.masked_fill(~residue_mask, 0.0).to(features.dtype)
        denominator = weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return (features * weights[..., None]).sum(dim=1) / denominator

    def forward(
        self,
        antibody_self: Any,
        antigen_self: Any,
        antibody_mask: Any,
        antigen_mask: Any,
        total_contact_logits: Any,
        interaction_contact_logits: Any,
        pair_mask: Any,
    ) -> dict[str, Any]:
        total_for_weights = (
            total_contact_logits.detach()
            if self.gradient_mode == "stop_gradient"
            else total_contact_logits
        )
        interaction_for_weights = (
            interaction_contact_logits.detach()
            if self.gradient_mode == "stop_gradient"
            else interaction_contact_logits
        )
        pair_lift = self._counterfactual_contact_lift(
            total_for_weights,
            interaction_for_weights,
            pair_mask,
        )
        antibody_weights = self._noisy_or_weights(
            pair_lift, pair_mask, dim=2
        )
        antigen_weights = self._noisy_or_weights(
            pair_lift, pair_mask, dim=1
        )
        antibody_projected = self.antibody_projection(antibody_self)
        antigen_projected = self.antigen_projection(antigen_self)
        antibody_embedding = functional.normalize(
            self._weighted_pool(
                antibody_projected, antibody_weights, antibody_mask
            ).float(),
            dim=-1,
        )
        antigen_embedding = functional.normalize(
            self._weighted_pool(antigen_projected, antigen_weights, antigen_mask).float(),
            dim=-1,
        )
        compatibility = (antibody_embedding * antigen_embedding).sum(dim=-1)
        if not bool(torch.isfinite(compatibility).all()):
            raise ValueError("contact bottleneck compatibility is non-finite")
        return {
            "compatibility": compatibility,
            "pair_contact_lift": pair_lift,
            "antibody_weights": antibody_weights,
            "antigen_weights": antigen_weights,
            "antibody_embedding": antibody_embedding,
            "antigen_embedding": antigen_embedding,
            "gradient_mode": self.gradient_mode,
        }


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
        local_pose_rotary_channels: int | None = None,
        geometry_attention_graph_mode: str = "spatial",
        num_rbf: int = 16,
        neighbor_k: int | None = 32,
        neighbor_radius: float | None = 16.0,
        sequence_window: int = 32,
        pair_dim: int | None = None,
        dropout: float = 0.1,
        fusion_mode: str = "late",
        pair_output_mode: str = "binary_contact",
        residue_output_mode: str = "gated_hybrid",
        residue_unary_source: str = "post_cross",
        contact_marginalization: str = "logmeanexp",
        distance_bin_boundaries: tuple[float, ...] = (
            2.0,
            3.0,
            4.0,
            4.5,
            5.0,
            6.0,
            8.0,
            10.0,
            12.0,
            16.0,
            20.0,
        ),
        contact_distance_cutoff: float = 4.5,
        paired_sequence_embedding_dim: int | None = None,
        paired_sequence_attention_heads: int | None = None,
        contact_bottleneck_projection_dim: int | None = None,
        contact_bottleneck_gradient_mode: str = "stop_gradient",
        attention_kernel: str = "reference",
        pair_gauge_mode: str = "raw",
        pair_gauge_iterations: int = 1,
        contact_gate_source: str = "full_contact",
        coarse_contact_crop_mode: str = "none",
        coarse_contact_crop_floor: float = 0.05,
        coarse_contact_crop_temperature: float = 1.0,
        coarse_contact_crop_strength: float = 1.0,
        coarse_contact_crop_stop_gradient: bool = True,
    ) -> None:
        _require_torch()
        super().__init__()
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = _positive_integer(d_model, "d_model")
        num_self_layers = _positive_integer(num_self_layers, "num_self_layers")
        if fusion_mode not in {"late", "interleaved"}:
            raise ValueError("fusion_mode must be 'late' or 'interleaved'")
        if pair_output_mode not in {
            "binary_contact",
            "distogram",
            "joint_contact_distogram",
        }:
            raise ValueError(
                "pair_output_mode must be binary_contact, distogram, or "
                "joint_contact_distogram"
            )
        if residue_output_mode not in {
            "unary_only",
            "contact_marginal_only",
            "gated_hybrid",
            "gauge_marginal_only",
            "gauge_hybrid",
        }:
            raise ValueError(
                "residue_output_mode must be unary_only, contact_marginal_only, "
                "gated_hybrid, gauge_marginal_only, or gauge_hybrid"
            )
        if pair_gauge_mode not in {"raw", "doubly_centered"}:
            raise ValueError("pair_gauge_mode must be raw or doubly_centered")
        if (
            isinstance(pair_gauge_iterations, bool)
            or not isinstance(pair_gauge_iterations, int)
            or pair_gauge_iterations < 1
        ):
            raise ValueError("pair_gauge_iterations must be a positive integer")
        if residue_output_mode in {"gauge_marginal_only", "gauge_hybrid"} and (
            pair_gauge_mode != "doubly_centered"
        ):
            raise ValueError(
                "gauge-routed residue heads require pair_gauge_mode='doubly_centered' "
                "so the partner-specific channel is identifiable"
            )
        if contact_marginalization not in {
            "logmeanexp",
            "mean_probability",
            "noisy_or",
            "length_normalized_noisy_or",
        }:
            raise ValueError(
                "contact_marginalization must be logmeanexp, mean_probability, "
                "noisy_or, or length_normalized_noisy_or"
            )
        if contact_gate_source not in {
            "none",
            "full_contact",
            "interaction_only",
        }:
            raise ValueError(
                "contact_gate_source must be none, full_contact, or interaction_only"
            )
        if coarse_contact_crop_mode not in {"none", "soft_unary"}:
            raise ValueError(
                "coarse_contact_crop_mode must be none or soft_unary"
            )
        if coarse_contact_crop_mode != "none" and fusion_mode != "interleaved":
            raise ValueError(
                "coarse contact cropping requires interleaved cross-attention"
            )
        if (
            not math.isfinite(coarse_contact_crop_floor)
            or not 0.0 < coarse_contact_crop_floor <= 1.0
        ):
            raise ValueError("coarse contact crop floor must lie in (0, 1]")
        if (
            not math.isfinite(coarse_contact_crop_temperature)
            or coarse_contact_crop_temperature <= 0
        ):
            raise ValueError(
                "coarse contact crop temperature must be finite and positive"
            )
        if (
            not math.isfinite(coarse_contact_crop_strength)
            or coarse_contact_crop_strength < 0
        ):
            raise ValueError(
                "coarse contact crop strength must be finite and nonnegative"
            )
        if residue_unary_source not in {"intra_chain", "post_cross"}:
            raise ValueError("residue_unary_source must be intra_chain or post_cross")
        if fusion_mode == "interleaved" and residue_unary_source == "intra_chain":
            raise ValueError(
                "an intra_chain unary source is undefined for interleaved cross-attention"
            )
        if contact_bottleneck_projection_dim is not None:
            _positive_integer(
                contact_bottleneck_projection_dim,
                "contact_bottleneck_projection_dim",
            )
            if fusion_mode != "late":
                raise ValueError(
                    "contact bottleneck requires late fusion so monomer self features "
                    "remain partner-independent"
                )
        if contact_bottleneck_gradient_mode not in {"stop_gradient", "joint"}:
            raise ValueError(
                "contact_bottleneck_gradient_mode must be stop_gradient or joint"
            )
        self.fusion_mode = fusion_mode
        self.pair_output_mode = pair_output_mode
        self.residue_output_mode = residue_output_mode
        self.residue_unary_source = residue_unary_source
        self.contact_marginalization = contact_marginalization
        self.contact_bottleneck_gradient_mode = contact_bottleneck_gradient_mode
        self.pair_gauge_mode = pair_gauge_mode
        self.pair_gauge_iterations = int(pair_gauge_iterations)
        self.contact_gate_source = contact_gate_source
        self.coarse_contact_crop_mode = coarse_contact_crop_mode
        self.coarse_contact_crop_floor = float(coarse_contact_crop_floor)
        self.coarse_contact_crop_temperature = float(
            coarse_contact_crop_temperature
        )
        self.coarse_contact_crop_strength = float(coarse_contact_crop_strength)
        self.coarse_contact_crop_stop_gradient = bool(
            coarse_contact_crop_stop_gradient
        )
        self.register_buffer(
            "contact_gate_multiplier",
            torch.ones((), dtype=torch.float32),
            persistent=True,
        )
        if (paired_sequence_embedding_dim is None) != (
            paired_sequence_attention_heads is None
        ):
            raise ValueError(
                "paired sequence embedding and attention dimensions must be enabled together"
            )
        self.paired_sequence_conditioner = (
            None
            if paired_sequence_embedding_dim is None
            else PPLMPairedConditioner(
                antibody_base_dim=antibody_input_dim,
                antigen_base_dim=antigen_input_dim,
                paired_embedding_dim=paired_sequence_embedding_dim,
                attention_head_count=paired_sequence_attention_heads,
            )
        )
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
            "local_pose_rotary_channels": local_pose_rotary_channels,
            "geometry_attention_graph_mode": geometry_attention_graph_mode,
            "num_rbf": num_rbf,
            "neighbor_k": neighbor_k,
            "neighbor_radius": neighbor_radius,
            "sequence_window": sequence_window,
            "dropout": dropout,
            "attention_kernel": attention_kernel,
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
                    attention_kernel=attention_kernel,
                )
                for _ in range(num_self_layers)
            )
        if pair_output_mode == "binary_contact":
            self.contact_head = PairContactHead(d_model, pair_dim)
        elif pair_output_mode == "distogram":
            self.contact_head = PairDistogramHead(
                d_model,
                pair_dim,
                distance_bin_boundaries=distance_bin_boundaries,
                contact_distance_cutoff=contact_distance_cutoff,
            )
        else:
            self.contact_head = JointPairContactDistogramHead(
                d_model,
                pair_dim,
                distance_bin_boundaries=distance_bin_boundaries,
                contact_distance_cutoff=contact_distance_cutoff,
            )
        self.cross_attention = (
            BidirectionalContactGatedCrossAttention(
                d_model,
                num_heads,
                dropout=dropout,
                attention_kernel=attention_kernel,
            )
            if fusion_mode == "late"
            else None
        )
        self.contact_bottleneck_head = (
            None
            if contact_bottleneck_projection_dim is None
            else ContactBottleneckContrastiveHead(
                d_model,
                contact_bottleneck_projection_dim,
                gradient_mode=contact_bottleneck_gradient_mode,
            )
        )
        self.antibody_norm = nn.LayerNorm(d_model)
        self.antigen_norm = nn.LayerNorm(d_model)
        self.paratope_head = nn.Linear(d_model, 1)
        self.epitope_head = nn.Linear(d_model, 1)
        self.paratope_contact_gate = nn.Parameter(torch.zeros(()))
        self.epitope_contact_gate = nn.Parameter(torch.zeros(()))
        nn.init.constant_(self.paratope_head.bias, -2.5)
        nn.init.constant_(self.epitope_head.bias, -2.5)

    def _contact_gate_logits(
        self,
        components: Mapping[str, Any],
        pair_mask: Any,
    ) -> Any:
        """Select the pre-registered routing channel without changing pair output.

        ``none`` supplies one constant logit on every valid row, which cancels
        exactly inside softmax and therefore implements ordinary cross-attention.
        ``interaction_only`` excludes additive antibody/antigen propensities so
        partner-independent unary shortcuts cannot route cross-chain messages.
        """

        if self.contact_gate_source == "full_contact":
            return components["logits"]
        if self.contact_gate_source == "interaction_only":
            return components["interaction_logits"]
        return torch.zeros_like(components["logits"]).masked_fill(~pair_mask, 0.0)

    def set_contact_gate_multiplier(self, value: float) -> None:
        """Set a checkpointed routing curriculum scalar in ``[0, 1]``."""

        if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
            raise ValueError("contact gate multiplier must lie in [0, 1]")
        self.contact_gate_multiplier.fill_(float(value))

    def _coarse_contact_crop(
        self,
        antibody: Any,
        antigen: Any,
        antibody_mask: Any,
        antigen_mask: Any,
    ) -> dict[str, Any] | None:
        if self.coarse_contact_crop_mode == "none":
            return None
        coarse_paratope_logits = self.paratope_head(
            self.antibody_norm(antibody)
        ).squeeze(-1)
        coarse_epitope_logits = self.epitope_head(
            self.antigen_norm(antigen)
        ).squeeze(-1)
        prior = soft_coarse_pair_log_prior(
            coarse_paratope_logits,
            coarse_epitope_logits,
            antibody_mask,
            antigen_mask,
            probability_floor=self.coarse_contact_crop_floor,
            temperature=self.coarse_contact_crop_temperature,
            stop_gradient=self.coarse_contact_crop_stop_gradient,
        )
        prior["paratope_logits"] = coarse_paratope_logits.masked_fill(
            ~antibody_mask, 0.0
        )
        prior["epitope_logits"] = coarse_epitope_logits.masked_fill(
            ~antigen_mask, 0.0
        )
        return prior

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
        paired_details = None
        if self.paired_sequence_conditioner is None:
            antibody_input = batch["antibody_features"]
            antigen_input = batch["antigen_features"]
            pair_prior_logits = None
        else:
            required = (
                "antibody_paired_sequence_features",
                "antigen_paired_sequence_features",
                "paired_sequence_inter_attention",
            )
            if any(name not in batch for name in required):
                raise ValueError("paired-sequence FrameContact input is incomplete")
            paired_details = self.paired_sequence_conditioner(
                batch["antibody_features"],
                batch["antigen_features"],
                batch["antibody_paired_sequence_features"],
                batch["antigen_paired_sequence_features"],
                batch["paired_sequence_inter_attention"],
                antibody_mask,
                antigen_mask,
            )
            antibody_input = paired_details["antibody_features"]
            antigen_input = paired_details["antigen_features"]
            pair_prior_logits = paired_details["pair_prior_logits"]
        antibody = self.antibody_projection(antibody_input)
        antigen = self.antigen_projection(antigen_input)
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
        if pair_prior_logits is not None:
            pair_prior_logits = pair_prior_logits.masked_fill(~pair_mask, 0.0)
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
            antibody_self, antigen_self = antibody, antigen
            initial_contact_components = inject_pair_contact_prior(
                self.contact_head.components(antibody, antigen, pair_mask),
                pair_prior_logits,
                pair_mask,
            )
            initial_contact_logits = initial_contact_components["logits"]
            initial_contact_gate_logits = self._contact_gate_logits(
                initial_contact_components, pair_mask
            )
            antibody_fused, antigen_fused, cross_details = self.cross_attention(
                antibody,
                antigen,
                antibody_mask,
                antigen_mask,
                initial_contact_gate_logits,
                pair_mask,
                contact_gate_multiplier=self.contact_gate_multiplier,
                return_attention=return_attention,
            )
            intermediate_contact_logits = (initial_contact_logits,)
            intermediate_contact_gate_logits = (initial_contact_gate_logits,)
            intermediate_coarse_contact_crops: tuple[dict[str, Any], ...] = ()
        else:
            assert self.interleaved_self_encoders is not None
            assert self.interleaved_cross_attentions is not None
            self_layer_details = []
            cross_layer_details = []
            intermediate_contact_logits_list = []
            intermediate_contact_gate_logits_list = []
            intermediate_coarse_contact_crop_list = []
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
                stage_contact_components = inject_pair_contact_prior(
                    self.contact_head.components(antibody, antigen, pair_mask),
                    pair_prior_logits,
                    pair_mask,
                )
                if initial_contact_components is None:
                    initial_contact_components = stage_contact_components
                stage_contact_logits = stage_contact_components["logits"]
                stage_contact_gate_logits = self._contact_gate_logits(
                    stage_contact_components, pair_mask
                )
                coarse_contact_crop = self._coarse_contact_crop(
                    antibody,
                    antigen,
                    antibody_mask,
                    antigen_mask,
                )
                if coarse_contact_crop is not None:
                    stage_contact_gate_logits = (
                        stage_contact_gate_logits
                        + self.coarse_contact_crop_strength
                        * coarse_contact_crop["log_prior"]
                    ).masked_fill(~pair_mask, 0.0)
                    intermediate_coarse_contact_crop_list.append(
                        coarse_contact_crop
                    )
                antibody, antigen, layer_cross_details = cross_layer(
                    antibody,
                    antigen,
                    antibody_mask,
                    antigen_mask,
                    stage_contact_gate_logits,
                    pair_mask,
                    contact_gate_multiplier=self.contact_gate_multiplier,
                    return_attention=return_attention,
                )
                intermediate_contact_logits_list.append(stage_contact_logits)
                intermediate_contact_gate_logits_list.append(
                    stage_contact_gate_logits
                )
                if return_attention:
                    self_layer_details.append(layer_self_details)
                    cross_layer_details.append(layer_cross_details)
            intermediate_contact_logits = tuple(intermediate_contact_logits_list)
            intermediate_contact_gate_logits = tuple(
                intermediate_contact_gate_logits_list
            )
            intermediate_coarse_contact_crops = tuple(
                intermediate_coarse_contact_crop_list
            )
            initial_contact_logits = intermediate_contact_logits[0]
            initial_contact_gate_logits = intermediate_contact_gate_logits[0]
            assert initial_contact_components is not None
            antibody_fused, antigen_fused = antibody, antigen
            antibody_self, antigen_self = antibody, antigen
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
        contact_components = inject_pair_contact_prior(
            self.contact_head.components(antibody_fused, antigen_fused, pair_mask),
            pair_prior_logits,
            pair_mask,
        )
        contact_logits = contact_components["logits"]
        contact_bottleneck = (
            None
            if self.contact_bottleneck_head is None
            else self.contact_bottleneck_head(
                antibody_self,
                antigen_self,
                antibody_mask,
                antigen_mask,
                initial_contact_components["logits"],
                initial_contact_components["interaction_logits"],
                pair_mask,
            )
        )
        antibody_fused = self.antibody_norm(antibody_fused).masked_fill(
            ~antibody_mask[..., None], 0.0
        )
        antigen_fused = self.antigen_norm(antigen_fused).masked_fill(
            ~antigen_mask[..., None], 0.0
        )
        antibody_self = self.antibody_norm(antibody_self).masked_fill(
            ~antibody_mask[..., None], 0.0
        )
        antigen_self = self.antigen_norm(antigen_self).masked_fill(
            ~antigen_mask[..., None], 0.0
        )
        marginalizer = {
            "logmeanexp": masked_logmeanexp,
            "mean_probability": masked_mean_probability_logits,
            "noisy_or": masked_noisy_or_logits,
            "length_normalized_noisy_or": masked_length_normalized_noisy_or_logits,
        }[self.contact_marginalization]
        paratope_contact = marginalizer(contact_logits, pair_mask, dim=2)
        epitope_contact = marginalizer(contact_logits, pair_mask, dim=1)
        # The raw marginals above still carry the single-chain propensity terms
        # of the contact map, which cannot distinguish partners by construction.
        # The gauge isolates the jointly dependent channel before aggregating,
        # in fp32 because the centering subtracts nearly equal quantities.
        #
        # A centered interaction residual is *not* a Bernoulli contact logit:
        # zero means "no evidence beyond additive row/column propensities", not
        # a 50% contact probability.  Applying noisy-OR to hundreds of centered
        # zeros therefore saturates at probability one and destroys residue
        # ranking.  The gauge path instead uses log-mean-exp, which is exactly
        # zero for flat centered evidence, is length normalized, and increases
        # with partner-specific concentration.  Raw contact marginals retain
        # the user-selected probabilistic aggregation above.
        if self.pair_gauge_mode == "doubly_centered":
            gauge = gauge_fixed_pair_components(
                contact_logits.float(),
                pair_mask,
                iterations=self.pair_gauge_iterations,
            )
            gauge_interaction = gauge["interaction"].to(contact_logits.dtype)
            paratope_gauge = masked_logmeanexp(
                gauge_interaction, pair_mask, dim=2
            )
            epitope_gauge = masked_logmeanexp(gauge_interaction, pair_mask, dim=1)
        else:
            gauge = None
            gauge_interaction = None
            paratope_gauge = None
            epitope_gauge = None
        antibody_unary_source = (
            antibody_self
            if self.residue_unary_source == "intra_chain"
            else antibody_fused
        )
        antigen_unary_source = (
            antigen_self if self.residue_unary_source == "intra_chain" else antigen_fused
        )
        paratope_unary = self.paratope_head(antibody_unary_source).squeeze(-1)
        epitope_unary = self.epitope_head(antigen_unary_source).squeeze(-1)
        if self.residue_output_mode == "unary_only":
            paratope_logits = paratope_unary
            epitope_logits = epitope_unary
        elif self.residue_output_mode == "contact_marginal_only":
            paratope_logits = paratope_contact
            epitope_logits = epitope_contact
        elif self.residue_output_mode == "gauge_marginal_only":
            paratope_logits = paratope_gauge
            epitope_logits = epitope_gauge
        elif self.residue_output_mode == "gauge_hybrid":
            paratope_logits = paratope_unary + torch.tanh(
                self.paratope_contact_gate
            ) * paratope_gauge
            epitope_logits = epitope_unary + torch.tanh(
                self.epitope_contact_gate
            ) * epitope_gauge
        else:
            paratope_logits = paratope_unary + torch.tanh(
                self.paratope_contact_gate
            ) * paratope_contact
            epitope_logits = epitope_unary + torch.tanh(
                self.epitope_contact_gate
            ) * epitope_contact
        paratope_logits = paratope_logits.masked_fill(~antibody_mask, 0.0)
        epitope_logits = epitope_logits.masked_fill(~antigen_mask, 0.0)
        return {
            "initial_contact_logits": initial_contact_logits,
            "initial_contact_gate_logits": initial_contact_gate_logits,
            "initial_contact_interaction_logits": initial_contact_components[
                "interaction_logits"
            ],
            "intermediate_contact_logits": intermediate_contact_logits,
            "intermediate_contact_gate_logits": intermediate_contact_gate_logits,
            "intermediate_coarse_contact_crops": (
                intermediate_coarse_contact_crops
            ),
            "contact_logits": contact_logits,
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
            "contact_gauge_interaction_logits": gauge_interaction,
            "paratope_gauge_logits": paratope_gauge,
            "epitope_gauge_logits": epitope_gauge,
            "contact_gauge_antibody_effect": (
                None if gauge is None else gauge["antibody_effect"]
            ),
            "contact_gauge_antigen_effect": (
                None if gauge is None else gauge["antigen_effect"]
            ),
            "contact_gauge_grand": None if gauge is None else gauge["grand"],
            "contact_gauge_attribution": (
                None if gauge is None else gauge["attribution"]
            ),
            "contact_gauge_centering_residual": (
                None if gauge is None else gauge["centering_residual"]
            ),
            "pair_gauge_mode": self.pair_gauge_mode,
            "gauge_marginalization": (
                None if gauge is None else "centered_logmeanexp"
            ),
            "paratope_unary_logits": paratope_unary.masked_fill(
                ~antibody_mask, 0.0
            ),
            "epitope_unary_logits": epitope_unary.masked_fill(~antigen_mask, 0.0),
            "paratope_logits": paratope_logits,
            "epitope_logits": epitope_logits,
            "antibody_mask": antibody_mask,
            "antigen_mask": antigen_mask,
            "self_attention_details": self_details,
            "cross_attention_details": cross_details,
            "fusion_mode": self.fusion_mode,
            "pair_output_mode": self.pair_output_mode,
            "residue_output_mode": self.residue_output_mode,
            "residue_unary_source": self.residue_unary_source,
            "contact_marginalization": self.contact_marginalization,
            "contact_gate_source": self.contact_gate_source,
            "contact_gate_multiplier": self.contact_gate_multiplier,
            "coarse_contact_crop_mode": self.coarse_contact_crop_mode,
            "coarse_contact_crop_floor": self.coarse_contact_crop_floor,
            "coarse_contact_crop_temperature": (
                self.coarse_contact_crop_temperature
            ),
            "coarse_contact_crop_strength": self.coarse_contact_crop_strength,
            "coarse_contact_crop_stop_gradient": (
                self.coarse_contact_crop_stop_gradient
            ),
            "contact_bottleneck_compatibility": (
                None if contact_bottleneck is None else contact_bottleneck["compatibility"]
            ),
            "contact_bottleneck_pair_contact_lift": (
                None
                if contact_bottleneck is None
                else contact_bottleneck["pair_contact_lift"]
            ),
            "contact_bottleneck_antibody_weights": (
                None
                if contact_bottleneck is None
                else contact_bottleneck["antibody_weights"]
            ),
            "contact_bottleneck_antigen_weights": (
                None
                if contact_bottleneck is None
                else contact_bottleneck["antigen_weights"]
            ),
            "contact_bottleneck_antibody_embedding": (
                None
                if contact_bottleneck is None
                else contact_bottleneck["antibody_embedding"]
            ),
            "contact_bottleneck_antigen_embedding": (
                None
                if contact_bottleneck is None
                else contact_bottleneck["antigen_embedding"]
            ),
            "contact_bottleneck_gradient_mode": (
                None if contact_bottleneck is None else contact_bottleneck["gradient_mode"]
            ),
            "distogram_logits": contact_components.get("distogram_logits"),
            "distogram_contact_logits": contact_components.get(
                "distogram_contact_logits"
            ),
            "distogram_interaction_logits": contact_components.get(
                "distogram_interaction_logits"
            ),
            "direct_contact_logits": contact_components.get(
                "direct_contact_logits"
            ),
            "interaction_distogram_logits": contact_components.get(
                "interaction_distogram_logits"
            ),
            "distance_bin_boundaries": contact_components.get(
                "distance_bin_boundaries"
            ),
            "paired_sequence_conditioning": paired_details,
            "paired_sequence_pair_prior_logits": pair_prior_logits,
        }


__all__ = [
    "BidirectionalContactGatedCrossAttention",
    "ContactGatedCrossAttention",
    "DualFactorizedIntraChainEncoder",
    "FactorizedIntraChainEncoder",
    "FrameContactModel",
    "JointPairContactDistogramHead",
    "masked_mean_probability_logits",
    "PairContactHead",
    "PairDistogramHead",
    "contact_logit_from_distogram",
    "inject_pair_contact_prior",
    "masked_length_normalized_noisy_or_logits",
    "masked_logmeanexp",
    "masked_noisy_or_logits",
    "soft_coarse_pair_log_prior",
]
