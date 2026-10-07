"""Sequence/structure conditioning for pair-aware ESM2 pretraining.

The structure pathway consumes only independently predicted, chain-internal
geometry.  It never constructs cross-partner distances.  A zero-initialized
residual gate keeps initialization exactly compatible with an existing
sequence-only PairAwareESM2 checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .pair_aware_esm2 import (
    PairAwareESM2,
    PairAwareESM2Config,
    PairAwareMLMOutput,
)


@dataclass(frozen=True)
class StructureConditioningConfig:
    """Freeze the predicted-structure vocabulary and continuous feature size."""

    token_vocabulary_size: int = 23
    feature_dimension: int = 17
    projection_dimension: int = 256
    missing_token_index: int = 0
    mask_token_index: int = 1
    dropped_token_index: int = 2
    first_3di_token_index: int = 3

    def validate(self) -> None:
        for name in (
            "token_vocabulary_size",
            "feature_dimension",
            "projection_dimension",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        special = {
            self.missing_token_index,
            self.mask_token_index,
            self.dropped_token_index,
        }
        if len(special) != 3 or min(special) < 0:
            raise ValueError("Structure special-token indices must be distinct")
        if self.first_3di_token_index <= max(special):
            raise ValueError("3Di tokens must follow all structure special tokens")
        if self.first_3di_token_index >= self.token_vocabulary_size:
            raise ValueError("Structure vocabulary has no 3Di tokens")


@dataclass(frozen=True)
class StructureReconstructionLoss:
    """Chain-balanced loss numerators for distributed aggregation."""

    token_chain_loss_sum: Tensor
    token_chain_count: Tensor
    feature_chain_loss_sum: Tensor
    feature_chain_count: Tensor
    token_target_count: int
    feature_target_count: int


class PredictedStructureAdapter(nn.Module):
    """Small structure pathway that can wrap an already-sharded backbone."""

    def __init__(
        self,
        *,
        embed_dimension: int,
        padding_index: int,
        structure_config: StructureConditioningConfig = StructureConditioningConfig(),
    ) -> None:
        super().__init__()
        structure_config.validate()
        if isinstance(embed_dimension, bool) or not isinstance(embed_dimension, int) or embed_dimension <= 0:
            raise ValueError("embed_dimension must be a positive integer")
        if isinstance(padding_index, bool) or not isinstance(padding_index, int) or padding_index < 0:
            raise ValueError("padding_index must be a nonnegative integer")
        self.embed_dimension = embed_dimension
        self.padding_index = padding_index
        self.structure_config = structure_config
        self.structure_token_embedding = nn.Embedding(
            structure_config.token_vocabulary_size,
            embed_dimension,
            padding_idx=structure_config.missing_token_index,
        )
        self.structure_feature_norm = nn.LayerNorm(
            structure_config.feature_dimension
        )
        self.structure_feature_projection = nn.Sequential(
            nn.Linear(
                structure_config.feature_dimension,
                structure_config.projection_dimension,
            ),
            nn.SiLU(),
            nn.Linear(
                structure_config.projection_dimension,
                embed_dimension,
            ),
        )
        self.structure_residual_gate = nn.Parameter(torch.zeros(()))
        self.structure_decoder_norm = nn.LayerNorm(embed_dimension)
        self.structure_token_bias = nn.Parameter(
            torch.zeros(structure_config.token_vocabulary_size)
        )
        self.structure_feature_decoder = nn.Linear(
            embed_dimension,
            structure_config.feature_dimension,
        )

    def structure_embedding_delta(
        self,
        tokens: Tensor,
        structure_tokens: Tensor,
        structure_features: Tensor,
        structure_feature_mask: Tensor,
    ) -> Tensor:
        """Return a gated, padding-safe predicted-structure residual."""
        if tokens.ndim != 2 or structure_tokens.shape != tokens.shape:
            raise ValueError("structure_tokens must match rank-two sequence tokens")
        expected_features = (*tokens.shape, self.structure_config.feature_dimension)
        if structure_features.shape != expected_features:
            raise ValueError("structure_features have the wrong shape")
        if structure_feature_mask.shape != tokens.shape:
            raise ValueError("structure_feature_mask must match sequence tokens")
        if structure_tokens.dtype != torch.long:
            raise ValueError("structure_tokens must use torch.long")
        if structure_feature_mask.dtype != torch.bool:
            raise ValueError("structure_feature_mask must be boolean")
        if not structure_features.is_floating_point():
            raise ValueError("structure_features must be floating point")
        if not (
            tokens.device
            == structure_tokens.device
            == structure_features.device
            == structure_feature_mask.device
        ):
            raise ValueError("All sequence and structure tensors must share a device")
        if bool(
            (structure_tokens < 0).any()
            or (
                structure_tokens
                >= self.structure_config.token_vocabulary_size
            ).any()
        ):
            raise ValueError("structure token index is outside the vocabulary")
        if not bool(torch.isfinite(structure_features).all()):
            raise ValueError("structure_features must be finite")

        valid_mask = tokens.ne(self.padding_index)
        missing = structure_tokens.eq(self.structure_config.missing_token_index)
        dropped = structure_tokens.eq(self.structure_config.dropped_token_index)
        regular = structure_tokens.ge(self.structure_config.first_3di_token_index)
        if bool(structure_tokens[~valid_mask].ne(self.structure_config.missing_token_index).any()):
            raise ValueError("Padding positions require the missing-structure token")
        if bool(structure_feature_mask[~valid_mask].any()):
            raise ValueError("Padding positions cannot contain structure features")
        if bool((structure_feature_mask & ~regular).any()):
            raise ValueError("Continuous features require an observed 3Di token")
        if bool(structure_features[~structure_feature_mask].ne(0).any()):
            raise ValueError("Masked continuous structure features must be zero")

        token_delta = self.structure_token_embedding(structure_tokens)
        active_token = valid_mask & ~missing & ~dropped
        token_delta = token_delta * active_token.unsqueeze(-1).to(token_delta.dtype)
        feature_delta = self.structure_feature_projection(
            self.structure_feature_norm(structure_features)
        )
        feature_delta = feature_delta * structure_feature_mask.unsqueeze(-1).to(
            feature_delta.dtype
        )
        delta = token_delta + feature_delta
        return (
            torch.tanh(self.structure_residual_gate)
            * delta
            * valid_mask.unsqueeze(-1).to(delta.dtype)
        )

    def decode_structure(self, hidden_states: Tensor) -> tuple[Tensor, Tensor]:
        """Decode 3Di classes and continuous local-frame features."""
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.embed_dimension:
            raise ValueError("hidden_states have the wrong shape")
        normalized = self.structure_decoder_norm(hidden_states)
        token_logits = F.linear(
            normalized,
            self.structure_token_embedding.weight,
            self.structure_token_bias,
        )
        return token_logits, self.structure_feature_decoder(normalized)

    @staticmethod
    def _chain_balanced_sum(
        residue_losses: Tensor,
        target_mask: Tensor,
        chain_ids: Tensor,
        valid_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        chain_slots = torch.arange(2, device=chain_ids.device, dtype=chain_ids.dtype)
        chain_masks = (
            chain_ids[:, None, :].eq(chain_slots[None, :, None])
            & valid_mask[:, None, :]
        )
        if bool(chain_masks.sum(dim=1).ne(valid_mask).any()):
            raise ValueError("Structure loss requires chain IDs 0 and 1")
        selected = target_mask[:, None, :] & chain_masks
        counts = selected.sum(dim=-1)
        contributing = counts.gt(0)
        safe_counts = counts.clamp_min(1).to(residue_losses.dtype)
        chain_losses = (
            residue_losses[:, None, :] * selected.to(residue_losses.dtype)
        ).sum(dim=-1) / safe_counts
        return chain_losses[contributing].sum(), contributing.sum()

    def structure_reconstruction_loss(
        self,
        hidden_states: Tensor,
        chain_ids: Tensor,
        valid_mask: Tensor,
        structure_token_labels: Tensor,
        structure_feature_targets: Tensor,
        structure_feature_target_mask: Tensor,
    ) -> StructureReconstructionLoss:
        """Compute masked, chain-balanced discrete and continuous losses."""
        if chain_ids.shape != valid_mask.shape or chain_ids.shape != structure_token_labels.shape:
            raise ValueError("Structure loss residue axes differ")
        if valid_mask.dtype != torch.bool or structure_feature_target_mask.dtype != torch.bool:
            raise ValueError("Structure validity and target masks must be boolean")
        if structure_feature_target_mask.shape != valid_mask.shape:
            raise ValueError("Continuous structure target mask has the wrong shape")
        expected_features = (*valid_mask.shape, self.structure_config.feature_dimension)
        if structure_feature_targets.shape != expected_features:
            raise ValueError("Continuous structure targets have the wrong shape")
        if bool((structure_token_labels[~valid_mask] != -100).any()):
            raise ValueError("Padding positions cannot be structure-token targets")
        if bool(structure_feature_target_mask[~valid_mask].any()):
            raise ValueError("Padding positions cannot be continuous-structure targets")
        token_target_mask = structure_token_labels.ne(-100)
        if bool(
            (
                structure_token_labels[token_target_mask]
                < self.structure_config.first_3di_token_index
            ).any()
        ):
            raise ValueError("Structure-token labels must target observed 3Di classes")
        if bool(
            (
                structure_token_labels[token_target_mask]
                >= self.structure_config.token_vocabulary_size
            ).any()
        ):
            raise ValueError("Structure-token label is outside the vocabulary")
        if not bool(torch.isfinite(structure_feature_targets).all()):
            raise ValueError("Continuous structure targets must be finite")

        token_logits, feature_predictions = self.decode_structure(hidden_states)
        token_losses = F.cross_entropy(
            token_logits.flatten(0, 1),
            structure_token_labels.flatten(),
            ignore_index=-100,
            reduction="none",
        ).view_as(structure_token_labels)
        feature_losses = F.smooth_l1_loss(
            feature_predictions,
            structure_feature_targets,
            reduction="none",
        ).mean(dim=-1)
        token_sum, token_chains = self._chain_balanced_sum(
            token_losses,
            token_target_mask,
            chain_ids,
            valid_mask,
        )
        feature_sum, feature_chains = self._chain_balanced_sum(
            feature_losses,
            structure_feature_target_mask,
            chain_ids,
            valid_mask,
        )
        if int(token_chains) + int(feature_chains) == 0:
            raise ValueError("Structure reconstruction batch has no targets")
        return StructureReconstructionLoss(
            token_chain_loss_sum=token_sum,
            token_chain_count=token_chains,
            feature_chain_loss_sum=feature_sum,
            feature_chain_count=feature_chains,
            token_target_count=int(token_target_mask.sum()),
            feature_target_count=int(structure_feature_target_mask.sum()),
        )


class StructureConditionedPairAwareESM2(PairAwareESM2):
    """Add masked 3Di and continuous local geometry to PairAwareESM2."""

    def __init__(
        self,
        config: PairAwareESM2Config = PairAwareESM2Config(),
        structure_config: StructureConditioningConfig = StructureConditioningConfig(),
    ) -> None:
        super().__init__(config)
        self.structure_adapter = PredictedStructureAdapter(
            embed_dimension=config.embed_dimension,
            padding_index=config.padding_index,
            structure_config=structure_config,
        )
        self.structure_config = structure_config

    @property
    def structure_token_embedding(self) -> nn.Embedding:
        return self.structure_adapter.structure_token_embedding

    @property
    def structure_residual_gate(self) -> nn.Parameter:
        return self.structure_adapter.structure_residual_gate

    @property
    def structure_feature_decoder(self) -> nn.Linear:
        return self.structure_adapter.structure_feature_decoder

    def structure_embedding_delta(
        self,
        tokens: Tensor,
        structure_tokens: Tensor,
        structure_features: Tensor,
        structure_feature_mask: Tensor,
    ) -> Tensor:
        return self.structure_adapter.structure_embedding_delta(
            tokens,
            structure_tokens,
            structure_features,
            structure_feature_mask,
        )

    def forward(
        self,
        tokens: Tensor,
        chain_ids: Tensor,
        *,
        structure_tokens: Tensor | None = None,
        structure_features: Tensor | None = None,
        structure_feature_mask: Tensor | None = None,
        labels: Tensor | None = None,
        return_last_attention: bool = False,
        allow_cross_chain_attention: bool = True,
        return_hidden_states: bool = False,
        compute_logits: bool = True,
        attention_query_capacity: int | None = None,
    ) -> PairAwareMLMOutput:
        supplied = (
            structure_tokens is not None,
            structure_features is not None,
            structure_feature_mask is not None,
        )
        if any(supplied) and not all(supplied):
            raise ValueError("All predicted-structure inputs must be supplied together")
        delta = None
        if all(supplied):
            assert structure_tokens is not None
            assert structure_features is not None
            assert structure_feature_mask is not None
            delta = self.structure_embedding_delta(
                tokens,
                structure_tokens,
                structure_features,
                structure_feature_mask,
            )
        return super().forward(
            tokens,
            chain_ids,
            input_embedding_delta=delta,
            labels=labels,
            return_last_attention=return_last_attention,
            allow_cross_chain_attention=allow_cross_chain_attention,
            return_hidden_states=return_hidden_states,
            compute_logits=compute_logits,
            attention_query_capacity=attention_query_capacity,
        )

    def decode_structure(self, hidden_states: Tensor) -> tuple[Tensor, Tensor]:
        return self.structure_adapter.decode_structure(hidden_states)

    def structure_reconstruction_loss(
        self,
        hidden_states: Tensor,
        chain_ids: Tensor,
        valid_mask: Tensor,
        structure_token_labels: Tensor,
        structure_feature_targets: Tensor,
        structure_feature_target_mask: Tensor,
    ) -> StructureReconstructionLoss:
        return self.structure_adapter.structure_reconstruction_loss(
            hidden_states,
            chain_ids,
            valid_mask,
            structure_token_labels,
            structure_feature_targets,
            structure_feature_target_mask,
        )
