"""Architecturally one-sided controls for PAIRScale contact prediction."""

from __future__ import annotations

from collections.abc import Collection
import math
from typing import Any, Literal

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:
    torch = None
    nn = None

from .development_contact_dataset import DevelopmentContactDataset
from .pair_geometry_schema import PAIR_ORIENTATION
from .pairscale_overfit import PairscaleOverfitConfig


CONTROL_SIDES = ("antibody_only", "antigen_only")
ControlSide = Literal["antibody_only", "antigen_only"]
_ModuleBase = nn.Module if nn is not None else object


def _require_torch() -> None:
    if torch is None or nn is None:
        raise RuntimeError("PyTorch is required for PAIRScale unary controls")


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _validate_side(side: str) -> ControlSide:
    if side not in CONTROL_SIDES:
        raise ValueError(f"control_side must be one of {CONTROL_SIDES}")
    return side  # type: ignore[return-value]


def _validate_stream_metadata(
    features: Any,
    mask: Any,
    *,
    expected_channels: int,
    name: str,
    inspect_values: bool,
) -> None:
    """Validate tensor metadata while leaving the forbidden stream unread."""

    if not torch.is_tensor(features) or features.ndim != 3:
        raise ValueError(f"{name}_features must be a rank-three tensor")
    if not torch.is_floating_point(features):
        raise ValueError(f"{name}_features must use a floating-point dtype")
    if int(features.shape[-1]) != expected_channels:
        raise ValueError(f"{name}_features channel count differs")
    if any(int(size) <= 0 for size in features.shape):
        raise ValueError(f"{name}_features cannot contain an empty dimension")
    if not torch.is_tensor(mask) or mask.dtype != torch.bool or mask.ndim != 2:
        raise ValueError(f"{name}_mask must be a rank-two boolean tensor")
    if tuple(mask.shape) != tuple(features.shape[:2]):
        raise ValueError(f"{name}_mask must align with {name}_features")
    if mask.device != features.device:
        raise ValueError(f"{name}_mask and {name}_features must share a device")
    if bool((mask.sum(dim=1) == 0).any()):
        raise ValueError(f"Every item must contain a valid {name} residue")
    if inspect_values and not bool(torch.isfinite(features).all()):
        raise ValueError(f"{name}_features must contain only finite values")


class _ResidualResidueBlock(_ModuleBase):
    """A deterministic pre-normalized residue-wise residual MLP."""

    def __init__(self, hidden_dim: int, ffn_dim: int) -> None:
        _require_torch()
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.input = nn.Linear(hidden_dim, ffn_dim)
        self.output = nn.Linear(ffn_dim, hidden_dim)

    def forward(self, residues: Any, mask: Any) -> Any:
        update = self.output(torch.nn.functional.gelu(self.input(self.norm(residues))))
        return (residues + update).masked_fill(~mask.unsqueeze(-1), 0.0)


class UnaryResidueContactModel(_ModuleBase):
    """Broadcast one endpoint's residue logits without reading its partner values.

    The forbidden stream is inspected only for tensor metadata needed to validate
    dimensions and masks. Its feature values never participate in the forward
    graph. The resulting contact matrix is therefore strictly row-only for the
    antibody control or column-only for the antigen control.
    """

    def __init__(
        self,
        *,
        control_side: ControlSide,
        antibody_input_dim: int,
        antigen_input_dim: int,
        hidden_dim: int = 192,
        ffn_dim: int = 384,
        residual_blocks: int = 2,
    ) -> None:
        _require_torch()
        super().__init__()
        self.control_side = _validate_side(control_side)
        self.antibody_input_dim = _positive_integer(
            antibody_input_dim,
            "antibody_input_dim",
        )
        self.antigen_input_dim = _positive_integer(
            antigen_input_dim,
            "antigen_input_dim",
        )
        hidden_dim = _positive_integer(hidden_dim, "hidden_dim")
        ffn_dim = _positive_integer(ffn_dim, "ffn_dim")
        residual_blocks = _positive_integer(residual_blocks, "residual_blocks")
        visible_dim = (
            self.antibody_input_dim
            if self.control_side == "antibody_only"
            else self.antigen_input_dim
        )
        self.input_norm = nn.LayerNorm(visible_dim)
        self.projection = nn.Linear(visible_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            _ResidualResidueBlock(hidden_dim, ffn_dim)
            for _ in range(residual_blocks)
        )
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.logit_head = nn.Linear(hidden_dim, 1)

    @staticmethod
    def _validate_requested_heads(enabled_heads: Collection[str] | None) -> None:
        if enabled_heads is None:
            return
        if isinstance(enabled_heads, (str, bytes)) or not isinstance(
            enabled_heads,
            Collection,
        ):
            raise TypeError("enabled_heads must be a collection")
        requested = frozenset(enabled_heads)
        if "contact" not in requested or requested - {"contact", "distance"}:
            raise ValueError("Unary controls execute only the contact head")

    def forward(
        self,
        antibody_features: Any,
        antigen_features: Any,
        antibody_mask: Any,
        antigen_mask: Any,
        candidate_pair_mask: Any,
        *,
        return_pair_representations: bool = False,
        enabled_heads: Collection[str] | None = None,
    ) -> dict[str, Any]:
        """Return a candidate-masked row-only or column-only contact matrix."""

        self._validate_requested_heads(enabled_heads)
        if return_pair_representations is not False:
            raise ValueError("Unary controls do not construct pair representations")
        antibody_visible = self.control_side == "antibody_only"
        _validate_stream_metadata(
            antibody_features,
            antibody_mask,
            expected_channels=self.antibody_input_dim,
            name="antibody",
            inspect_values=antibody_visible,
        )
        _validate_stream_metadata(
            antigen_features,
            antigen_mask,
            expected_channels=self.antigen_input_dim,
            name="antigen",
            inspect_values=not antibody_visible,
        )
        expected_pair_shape = (
            int(antibody_features.shape[0]),
            int(antibody_features.shape[1]),
            int(antigen_features.shape[1]),
        )
        if (
            not torch.is_tensor(candidate_pair_mask)
            or candidate_pair_mask.dtype != torch.bool
            or tuple(candidate_pair_mask.shape) != expected_pair_shape
        ):
            raise ValueError("candidate_pair_mask shape or dtype differs")
        if candidate_pair_mask.device != antibody_features.device:
            raise ValueError("All inputs must share a device")
        valid_axes = antibody_mask.unsqueeze(2) & antigen_mask.unsqueeze(1)
        if bool((candidate_pair_mask & ~valid_axes).any()):
            raise ValueError("candidate_pair_mask includes a padded residue")
        if bool((candidate_pair_mask.flatten(1).sum(dim=1) == 0).any()):
            raise ValueError("Every item must contain at least one candidate pair")

        visible_features = antibody_features if antibody_visible else antigen_features
        visible_mask = antibody_mask if antibody_visible else antigen_mask
        residues = self.projection(
            self.input_norm(
                visible_features.masked_fill(~visible_mask.unsqueeze(-1), 0.0)
            )
        ).masked_fill(~visible_mask.unsqueeze(-1), 0.0)
        for block in self.blocks:
            residues = block(residues, visible_mask)
        residue_logits = self.logit_head(self.final_norm(residues)).squeeze(-1)
        if antibody_visible:
            contact_logits = residue_logits.unsqueeze(2).expand(expected_pair_shape)
        else:
            contact_logits = residue_logits.unsqueeze(1).expand(expected_pair_shape)
        contact_logits = contact_logits.masked_fill(~candidate_pair_mask, 0.0)
        contact_probabilities = torch.sigmoid(contact_logits).masked_fill(
            ~candidate_pair_mask,
            0.0,
        )
        return {
            "pair_orientation": PAIR_ORIENTATION,
            "executed_heads": ("contact",),
            "candidate_pair_mask": candidate_pair_mask,
            "pair_representations": None,
            "contact_logits": contact_logits,
            "contact_probabilities": contact_probabilities,
            "unary_control_side": self.control_side,
            "forbidden_partner_values_read": False,
        }


def build_unary_control_model(
    dataset: DevelopmentContactDataset,
    config: PairscaleOverfitConfig,
    control_side: ControlSide,
) -> UnaryResidueContactModel:
    """Construct the protocol-locked one-sided residue control."""

    config.validate()
    if config.partner_ranking_weight != 0.0:
        raise ValueError("Unary controls require zero partner-ranking weight")
    if not math.isclose(config.dropout, 0.0, abs_tol=0.0):
        raise ValueError("Unary controls require zero dropout")
    return UnaryResidueContactModel(
        control_side=_validate_side(control_side),
        antibody_input_dim=dataset.antibody_input_dim,
        antigen_input_dim=dataset.antigen_input_dim,
        hidden_dim=config.d_model,
        ffn_dim=config.ffn_dim,
        residual_blocks=config.num_fusion_layers,
    )


__all__ = [
    "CONTROL_SIDES",
    "ControlSide",
    "UnaryResidueContactModel",
    "build_unary_control_model",
]
