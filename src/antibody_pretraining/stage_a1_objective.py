"""Explicit retrieval and contact heads for ABPairFM Stage-A1."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class StageA1Loss:
    total: Tensor
    mlm: Tensor
    retrieval: Tensor
    contact: Tensor
    retrieval_accuracy_left_to_right: Tensor
    retrieval_accuracy_right_to_left: Tensor
    contact_count: int
    contact_positive_count: int


def _masked_chain_mean(hidden: Tensor, chain_ids: Tensor, chain: int) -> Tensor:
    selected = chain_ids.eq(chain)
    if bool(selected.sum(dim=1).eq(0).any()):
        raise ValueError(f"Every example requires chain {chain}")
    return (hidden * selected.unsqueeze(-1)).sum(dim=1) / selected.sum(
        dim=1, keepdim=True
    )


def _masked_bidirectional_cross_entropy(
    left_logits: Tensor,
    right_logits: Tensor,
    allowed: Tensor,
    targets: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    if (
        left_logits.ndim != 2
        or right_logits.shape != left_logits.shape
        or targets.shape != (left_logits.shape[0],)
        or targets.dtype != torch.long
    ):
        raise ValueError("Retrieval logits or targets have an invalid shape")
    if allowed.shape != left_logits.shape or allowed.dtype != torch.bool:
        raise ValueError("Retrieval candidate mask differs from logits")
    if bool((targets < 0).any()) or bool((targets >= left_logits.shape[1]).any()):
        raise ValueError("Retrieval target is outside the candidate matrix")
    rows = torch.arange(left_logits.shape[0], device=left_logits.device)
    if not bool(allowed[rows, targets].all()):
        raise ValueError("Every cognate retrieval pair must remain allowed")
    if bool(allowed.sum(dim=1).lt(2).any()):
        raise ValueError("Every retrieval direction requires at least one negative")
    masked_left = left_logits.masked_fill(~allowed, -torch.inf)
    masked_right = right_logits.masked_fill(~allowed, -torch.inf)
    left_loss = F.cross_entropy(masked_left, targets)
    right_loss = F.cross_entropy(masked_right, targets)
    left_accuracy = masked_left.argmax(dim=1).eq(targets).float().mean()
    right_accuracy = masked_right.argmax(dim=1).eq(targets).float().mean()
    return (left_loss + right_loss) / 2.0, left_accuracy, right_accuracy


class StageA1Heads(nn.Module):
    """Pair heads over explicit retrieval and joint-contact backbone views."""

    def __init__(
        self,
        embedding_dimension: int = 2560,
        projection_dimension: int = 256,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        if embedding_dimension <= 0 or projection_dimension <= 0:
            raise ValueError("Stage-A1 head dimensions must be positive")
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("Stage-A1 temperature must be positive and finite")
        self.temperature = float(temperature)
        self.retrieval_left = nn.Linear(embedding_dimension, projection_dimension, bias=False)
        self.retrieval_right = nn.Linear(embedding_dimension, projection_dimension, bias=False)
        self.contact_left = nn.Linear(embedding_dimension, projection_dimension, bias=False)
        self.contact_right = nn.Linear(embedding_dimension, projection_dimension, bias=False)
        self.contact_bias = nn.Parameter(torch.zeros(()))

    def retrieval_embeddings(self, hidden: Tensor, chain_ids: Tensor) -> tuple[Tensor, Tensor]:
        if hidden.ndim != 3 or chain_ids.shape != hidden.shape[:2]:
            raise ValueError("Hidden states and chain IDs are not aligned")
        left = F.normalize(
            self.retrieval_left(_masked_chain_mean(hidden, chain_ids, 0)), dim=-1
        )
        right = F.normalize(
            self.retrieval_right(_masked_chain_mean(hidden, chain_ids, 1)), dim=-1
        )
        return left, right

    def retrieval_logits(self, hidden: Tensor, chain_ids: Tensor) -> Tensor:
        left, right = self.retrieval_embeddings(hidden, chain_ids)
        return left @ right.transpose(0, 1) / self.temperature

    def contact_logits(
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
        if not (
            len(pair_batch_indices)
            == len(pair_left_token_indices)
            == len(pair_right_token_indices)
        ):
            raise ValueError("Sparse contact index columns have different lengths")
        if len(pair_batch_indices) == 0:
            raise ValueError("Stage-A1 contact supervision cannot be empty")
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
        return (left * right).sum(dim=-1) / math.sqrt(left.shape[-1]) + self.contact_bias

    def retrieval_loss(
        self,
        *,
        hidden: Tensor,
        chain_ids: Tensor,
        retrieval_allowed: Tensor,
        retrieval_candidate_left: Tensor | None = None,
        retrieval_candidate_right: Tensor | None = None,
        retrieval_target_indices: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        retrieval_left, retrieval_right = self.retrieval_embeddings(hidden, chain_ids)
        if (retrieval_candidate_left is None) != (retrieval_candidate_right is None):
            raise ValueError("Both retrieval candidate directions are required")
        if retrieval_candidate_left is None:
            retrieval_candidate_left = retrieval_left
            retrieval_candidate_right = retrieval_right
        assert retrieval_candidate_right is not None
        if (
            retrieval_candidate_left.ndim != 2
            or retrieval_candidate_right.shape != retrieval_candidate_left.shape
            or retrieval_candidate_left.shape[1] != retrieval_left.shape[1]
        ):
            raise ValueError("Retrieval candidate embeddings have an invalid shape")
        if retrieval_target_indices is None:
            if retrieval_candidate_left.shape[0] != retrieval_left.shape[0]:
                raise ValueError("Global retrieval candidates require explicit targets")
            retrieval_target_indices = torch.arange(
                retrieval_left.shape[0], device=retrieval_left.device
            )
        left_logits = (
            retrieval_left @ retrieval_candidate_right.transpose(0, 1)
        ) / self.temperature
        right_logits = (
            retrieval_right @ retrieval_candidate_left.transpose(0, 1)
        ) / self.temperature
        retrieval_loss, left_accuracy, right_accuracy = (
            _masked_bidirectional_cross_entropy(
                left_logits,
                right_logits,
                retrieval_allowed,
                retrieval_target_indices,
            )
        )
        return retrieval_loss, left_accuracy, right_accuracy

    def contact_loss(
        self,
        *,
        hidden: Tensor,
        pair_batch_indices: Tensor,
        pair_left_token_indices: Tensor,
        pair_right_token_indices: Tensor,
        pair_labels: Tensor,
    ) -> tuple[Tensor, int, int]:
        contact_logits = self.contact_logits(
            hidden,
            pair_batch_indices,
            pair_left_token_indices,
            pair_right_token_indices,
        )
        if pair_labels.shape != contact_logits.shape:
            raise ValueError("Contact labels differ from sparse contact logits")
        labels = pair_labels.to(dtype=contact_logits.dtype)
        if not bool(((labels == 0) | (labels == 1)).all()):
            raise ValueError("Contact labels must be binary")
        positive_count = int(labels.sum().item())
        negative_count = len(labels) - positive_count
        if positive_count == 0 or negative_count == 0:
            raise ValueError("Stage-A1 contact batches require both classes")
        positive_weight = torch.tensor(
            negative_count / positive_count,
            device=contact_logits.device,
            dtype=contact_logits.dtype,
        )
        contact_loss = F.binary_cross_entropy_with_logits(
            contact_logits,
            labels,
            pos_weight=positive_weight,
        )
        return contact_loss, len(labels), positive_count

    def loss(
        self,
        *,
        hidden: Tensor,
        chain_ids: Tensor,
        retrieval_hidden: Tensor | None = None,
        mlm_loss: Tensor,
        retrieval_allowed: Tensor,
        pair_batch_indices: Tensor,
        pair_left_token_indices: Tensor,
        pair_right_token_indices: Tensor,
        pair_labels: Tensor,
        retrieval_candidate_left: Tensor | None = None,
        retrieval_candidate_right: Tensor | None = None,
        retrieval_target_indices: Tensor | None = None,
        mlm_weight: float = 1.0,
        retrieval_weight: float = 0.2,
        contact_weight: float = 0.5,
    ) -> StageA1Loss:
        weights = (mlm_weight, retrieval_weight, contact_weight)
        if any(not math.isfinite(value) or value < 0.0 for value in weights):
            raise ValueError("Stage-A1 loss weights must be finite and non-negative")
        if mlm_loss.ndim != 0 or not bool(torch.isfinite(mlm_loss)):
            raise ValueError("Stage-A1 requires one finite scalar MLM loss")
        if retrieval_hidden is None:
            retrieval_hidden = hidden
        if retrieval_hidden.shape != hidden.shape:
            raise ValueError("Retrieval and contact hidden states must have equal shapes")
        retrieval_loss, left_accuracy, right_accuracy = self.retrieval_loss(
            hidden=retrieval_hidden,
            chain_ids=chain_ids,
            retrieval_allowed=retrieval_allowed,
            retrieval_candidate_left=retrieval_candidate_left,
            retrieval_candidate_right=retrieval_candidate_right,
            retrieval_target_indices=retrieval_target_indices,
        )
        contact_loss, contact_count, positive_count = self.contact_loss(
            hidden=hidden,
            pair_batch_indices=pair_batch_indices,
            pair_left_token_indices=pair_left_token_indices,
            pair_right_token_indices=pair_right_token_indices,
            pair_labels=pair_labels,
        )
        total = (
            mlm_weight * mlm_loss
            + retrieval_weight * retrieval_loss
            + contact_weight * contact_loss
        )
        if not bool(torch.isfinite(total)):
            raise RuntimeError("Stage-A1 produced a nonfinite multitask loss")
        return StageA1Loss(
            total=total,
            mlm=mlm_loss,
            retrieval=retrieval_loss,
            contact=contact_loss,
            retrieval_accuracy_left_to_right=left_accuracy,
            retrieval_accuracy_right_to_left=right_accuracy,
            contact_count=contact_count,
            contact_positive_count=positive_count,
        )
