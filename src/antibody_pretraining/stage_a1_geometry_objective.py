"""Pair-feature contact and train-only distance-bin objectives for Stage-A1."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .stage_a1_objective import StageA1Heads


class StageA1GeometryHeads(StageA1Heads):
    """Predict contact probability as mass over learned distance bins."""

    def __init__(
        self,
        embedding_dimension: int = 2560,
        projection_dimension: int = 256,
        pair_dimension: int = 256,
        num_distance_bins: int = 8,
        contact_distance_bin_count: int = 2,
        temperature: float = 0.07,
    ) -> None:
        super().__init__(
            embedding_dimension=embedding_dimension,
            projection_dimension=projection_dimension,
            temperature=temperature,
        )
        if pair_dimension <= 0:
            raise ValueError("Stage-A1 pair dimension must be positive")
        if not 0 < contact_distance_bin_count < num_distance_bins:
            raise ValueError("Stage-A1 contact distance-bin prefix is invalid")
        self.num_distance_bins = int(num_distance_bins)
        self.contact_distance_bin_count = int(contact_distance_bin_count)
        pair_input_dimension = 4 * projection_dimension
        self.contact_pair_norm = nn.LayerNorm(pair_input_dimension)
        self.contact_pair_hidden = nn.Linear(pair_input_dimension, pair_dimension)
        self.contact_distance_head = nn.Linear(pair_dimension, num_distance_bins)

    def _pair_hidden(
        self,
        hidden: Tensor,
        pair_batch_indices: Tensor,
        pair_left_token_indices: Tensor,
        pair_right_token_indices: Tensor,
    ) -> Tensor:
        indices = (
            pair_batch_indices,
            pair_left_token_indices,
            pair_right_token_indices,
        )
        if any(value.ndim != 1 for value in indices):
            raise ValueError("Sparse contact indices must be one-dimensional")
        if len({len(value) for value in indices}) != 1 or len(pair_batch_indices) == 0:
            raise ValueError("Sparse contact indices are empty or misaligned")
        if (
            int(pair_batch_indices.min()) < 0
            or int(pair_batch_indices.max()) >= hidden.shape[0]
            or int(pair_left_token_indices.min()) < 0
            or int(pair_left_token_indices.max()) >= hidden.shape[1]
            or int(pair_right_token_indices.min()) < 0
            or int(pair_right_token_indices.max()) >= hidden.shape[1]
        ):
            raise ValueError("Sparse contact index is outside the hidden-state tensor")
        left = self.contact_left(hidden[pair_batch_indices, pair_left_token_indices])
        right = self.contact_right(hidden[pair_batch_indices, pair_right_token_indices])
        pair = torch.cat((left, right, left * right, (left - right).abs()), dim=-1)
        return F.gelu(self.contact_pair_hidden(self.contact_pair_norm(pair)))

    def distance_logits(
        self,
        hidden: Tensor,
        pair_batch_indices: Tensor,
        pair_left_token_indices: Tensor,
        pair_right_token_indices: Tensor,
    ) -> Tensor:
        pair = self._pair_hidden(
            hidden,
            pair_batch_indices,
            pair_left_token_indices,
            pair_right_token_indices,
        )
        return self.contact_distance_head(pair)

    def contact_logits(
        self,
        hidden: Tensor,
        pair_batch_indices: Tensor,
        pair_left_token_indices: Tensor,
        pair_right_token_indices: Tensor,
    ) -> Tensor:
        distance = self.distance_logits(
            hidden,
            pair_batch_indices,
            pair_left_token_indices,
            pair_right_token_indices,
        )
        contact_mass = torch.logsumexp(
            distance[:, : self.contact_distance_bin_count], dim=-1
        )
        noncontact_mass = torch.logsumexp(
            distance[:, self.contact_distance_bin_count :], dim=-1
        )
        return contact_mass - noncontact_mass + self.contact_bias

    def contact_and_distance_loss(
        self,
        *,
        hidden: Tensor,
        pair_batch_indices: Tensor,
        pair_left_token_indices: Tensor,
        pair_right_token_indices: Tensor,
        distance_bin_targets: Tensor,
    ) -> tuple[Tensor, Tensor, int, int]:
        distance = self.distance_logits(
            hidden,
            pair_batch_indices,
            pair_left_token_indices,
            pair_right_token_indices,
        )
        if distance_bin_targets.shape != distance.shape[:1] or distance_bin_targets.dtype != torch.long:
            raise ValueError("Stage-A1 distance-bin targets are misaligned")
        if bool((distance_bin_targets < 0).any()) or bool(
            (distance_bin_targets >= self.num_distance_bins).any()
        ):
            raise ValueError("Stage-A1 distance-bin target is out of range")
        labels = distance_bin_targets < self.contact_distance_bin_count
        positive_count = int(labels.sum())
        if positive_count == 0 or positive_count == len(labels):
            raise ValueError("Stage-A1 geometry batches require both contact classes")
        contact_mass = torch.logsumexp(
            distance[:, : self.contact_distance_bin_count], dim=-1
        )
        noncontact_mass = torch.logsumexp(
            distance[:, self.contact_distance_bin_count :], dim=-1
        )
        contact_logits = contact_mass - noncontact_mass + self.contact_bias
        contact_loss = F.binary_cross_entropy_with_logits(
            contact_logits,
            labels.to(dtype=contact_logits.dtype),
        )
        distance_loss = F.cross_entropy(distance, distance_bin_targets)
        if not bool(torch.isfinite(contact_loss) and torch.isfinite(distance_loss)):
            raise RuntimeError("Stage-A1 geometry loss is nonfinite")
        return contact_loss, distance_loss, len(labels), positive_count


def expected_random_contact_loss(negative_to_positive_ratio: int) -> float:
    """Return unweighted BCE at the sampled class-prior optimum."""

    if negative_to_positive_ratio < 1:
        raise ValueError("Negative-to-positive ratio must be positive")
    positive_probability = 1.0 / (negative_to_positive_ratio + 1.0)
    return -(
        positive_probability * math.log(positive_probability)
        + (1.0 - positive_probability) * math.log(1.0 - positive_probability)
    )
