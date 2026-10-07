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
    contact_bce: Tensor
    contact_ranking: Tensor
    contact_centering: Tensor
    retrieval_accuracy_left_to_right: Tensor
    retrieval_accuracy_right_to_left: Tensor
    contact_count: int
    contact_positive_count: int
    contact_row_comparison_count: int
    contact_column_comparison_count: int


@dataclass(frozen=True)
class PairContactComponents:
    """Decomposed sparse pair logits for identifiable interaction learning."""

    full_logits: Tensor
    interaction_logits: Tensor
    raw_interaction_logits: Tensor
    left_unary_logits: Tensor
    right_unary_logits: Tensor
    centering_error: Tensor


@dataclass(frozen=True)
class PairContactLoss:
    """Contact-only objective terms for frozen-backbone diagnostics."""

    total: Tensor
    bce: Tensor
    ranking: Tensor
    centering: Tensor
    positive_count: int
    pair_count: int
    row_comparison_count: int
    column_comparison_count: int


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


def _group_means(values: Tensor, keys: Tensor) -> Tensor:
    if values.ndim != 1 or keys.ndim != 2 or keys.shape[0] != len(values):
        raise ValueError("Grouped values and keys are not aligned")
    _, inverse = torch.unique(keys, dim=0, return_inverse=True)
    group_count = int(inverse.max().item()) + 1
    sums = values.new_zeros(group_count)
    sums.index_add_(0, inverse, values)
    counts = torch.bincount(inverse, minlength=group_count).to(values.dtype)
    return sums / counts


def _center_unique_endpoint_vectors(values: Tensor, keys: Tensor) -> tuple[Tensor, Tensor]:
    """Center endpoint vectors once per unique residue within each complex.

    Sparse supervision repeats one endpoint for every sampled pair.  Weighting
    the mean by those repetitions would make the residual depend on the
    negative-sampling degree.  We therefore deduplicate ``(batch, residue)``
    keys first, average the unique endpoint vectors within each complex, and
    map the resulting zero-sum representation back to every sampled pair.
    """

    if values.ndim != 2 or keys.ndim != 2 or keys.shape != (len(values), 2):
        raise ValueError("Endpoint vectors and keys are not aligned")
    # Autocast supplies bfloat16 projection vectors in the distributed canary.
    # Mean subtraction in that dtype leaves an O(1e-3) endpoint mean even when
    # the algebra is exactly zero-sum.  The endpoint sets are tiny relative to
    # the backbone, so perform only this gauge-fixing calculation in float32.
    # Gradients still flow through the cast to the projection parameters.
    working_values = values.float()
    unique_keys, inverse = torch.unique(keys, dim=0, sorted=True, return_inverse=True)
    unique_count = len(unique_keys)
    repeated_sums = working_values.new_zeros((unique_count, values.shape[1]))
    repeated_sums.index_add_(0, inverse, working_values)
    repeated_counts = torch.bincount(inverse, minlength=unique_count).to(
        working_values.dtype
    )
    unique_values = repeated_sums / repeated_counts.unsqueeze(1)

    batch_keys = unique_keys[:, 0]
    batch_count = int(batch_keys.max().item()) + 1
    endpoint_counts = torch.bincount(batch_keys, minlength=batch_count)
    present_batches = torch.unique(batch_keys, sorted=True)
    if bool(endpoint_counts[present_batches].lt(2).any()):
        raise ValueError("Every contact complex requires at least two unique endpoints")
    batch_sums = working_values.new_zeros((batch_count, values.shape[1]))
    batch_sums.index_add_(0, batch_keys, unique_values)
    safe_endpoint_counts = (
        endpoint_counts.clamp_min(1).to(working_values.dtype).unsqueeze(1)
    )
    batch_means = batch_sums / safe_endpoint_counts
    unique_centered = unique_values - batch_means[batch_keys]

    centered_batch_sums = working_values.new_zeros((batch_count, values.shape[1]))
    centered_batch_sums.index_add_(0, batch_keys, unique_centered)
    centered_batch_means = centered_batch_sums / safe_endpoint_counts
    centering_error = centered_batch_means[present_batches].square().mean()
    return unique_centered[inverse], centering_error


def _interaction_centering_penalty(
    interaction_logits: Tensor,
    pair_batch_indices: Tensor,
    pair_left_token_indices: Tensor,
    pair_right_token_indices: Tensor,
) -> Tensor:
    row_keys = torch.stack((pair_batch_indices, pair_left_token_indices), dim=1)
    column_keys = torch.stack((pair_batch_indices, pair_right_token_indices), dim=1)
    row_means = _group_means(interaction_logits, row_keys)
    column_means = _group_means(interaction_logits, column_keys)
    return 0.5 * (row_means.square().mean() + column_means.square().mean())


def _matched_interaction_ranking_loss(
    interaction_logits: Tensor,
    labels: Tensor,
    pair_batch_indices: Tensor,
    pair_left_token_indices: Tensor,
    pair_right_token_indices: Tensor,
    *,
    margin: float,
) -> tuple[Tensor, int, int]:
    row_terms: list[Tensor] = []
    column_terms: list[Tensor] = []
    for batch_index in torch.unique(pair_batch_indices, sorted=True):
        selected = pair_batch_indices.eq(batch_index)
        positive = selected & labels.eq(1)
        negative = selected & labels.eq(0)
        positive_logits = interaction_logits[positive]
        negative_logits = interaction_logits[negative]
        if len(positive_logits) == 0 or len(negative_logits) == 0:
            raise ValueError("Every contact-ranking complex requires both classes")
        pairwise = F.softplus(
            margin - positive_logits[:, None] + negative_logits[None, :]
        )
        row_matched = pair_left_token_indices[positive, None].eq(
            pair_left_token_indices[None, negative]
        )
        column_matched = pair_right_token_indices[positive, None].eq(
            pair_right_token_indices[None, negative]
        )
        if bool(row_matched.any()):
            row_terms.append(pairwise[row_matched])
        if bool(column_matched.any()):
            column_terms.append(pairwise[column_matched])
    row_count = sum(term.numel() for term in row_terms)
    column_count = sum(term.numel() for term in column_terms)
    if row_count + column_count == 0:
        zero = interaction_logits.sum() * 0.0
        return zero, 0, 0
    return (
        torch.cat((*row_terms, *column_terms)).mean(),
        row_count,
        column_count,
    )


class StageA1Heads(nn.Module):
    """Pair heads over explicit retrieval and joint-contact backbone views."""

    def __init__(
        self,
        embedding_dimension: int = 2560,
        projection_dimension: int = 256,
        temperature: float = 0.07,
        interaction_residualization: str = "none",
    ) -> None:
        super().__init__()
        if embedding_dimension <= 0 or projection_dimension <= 0:
            raise ValueError("Stage-A1 head dimensions must be positive")
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("Stage-A1 temperature must be positive and finite")
        if interaction_residualization not in {"none", "endpoint_zero_sum"}:
            raise ValueError("Unsupported Stage-A1 interaction residualization")
        self.temperature = float(temperature)
        self.interaction_residualization = interaction_residualization
        self.retrieval_left = nn.Linear(embedding_dimension, projection_dimension, bias=False)
        self.retrieval_right = nn.Linear(embedding_dimension, projection_dimension, bias=False)
        self.contact_left = nn.Linear(embedding_dimension, projection_dimension, bias=False)
        self.contact_right = nn.Linear(embedding_dimension, projection_dimension, bias=False)
        self.contact_left_unary = nn.Linear(embedding_dimension, 1, bias=False)
        self.contact_right_unary = nn.Linear(embedding_dimension, 1, bias=False)
        nn.init.zeros_(self.contact_left_unary.weight)
        nn.init.zeros_(self.contact_right_unary.weight)
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

    def contact_components(
        self,
        hidden: Tensor,
        pair_batch_indices: Tensor,
        pair_left_token_indices: Tensor,
        pair_right_token_indices: Tensor,
    ) -> PairContactComponents:
        if hidden.ndim != 3:
            raise ValueError("Contact hidden states must have shape [batch, tokens, dim]")
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
        left_hidden = hidden[pair_batch_indices, pair_left_token_indices]
        right_hidden = hidden[pair_batch_indices, pair_right_token_indices]
        left = self.contact_left(left_hidden)
        right = self.contact_right(right_hidden)
        raw_interaction_logits = (left * right).sum(dim=-1) / math.sqrt(
            left.shape[-1]
        )
        if self.interaction_residualization == "endpoint_zero_sum":
            left, left_centering_error = _center_unique_endpoint_vectors(
                left,
                torch.stack((pair_batch_indices, pair_left_token_indices), dim=1),
            )
            right, right_centering_error = _center_unique_endpoint_vectors(
                right,
                torch.stack((pair_batch_indices, pair_right_token_indices), dim=1),
            )
            interaction_logits = (left * right).sum(dim=-1) / math.sqrt(
                left.shape[-1]
            )
            centering_error = 0.5 * (left_centering_error + right_centering_error)
        else:
            interaction_logits = raw_interaction_logits
            centering_error = _interaction_centering_penalty(
                interaction_logits,
                pair_batch_indices,
                pair_left_token_indices,
                pair_right_token_indices,
            )
        left_unary_logits = self.contact_left_unary(left_hidden).squeeze(-1)
        right_unary_logits = self.contact_right_unary(right_hidden).squeeze(-1)
        full_logits = (
            left_unary_logits
            + right_unary_logits
            + interaction_logits
            + self.contact_bias
        )
        return PairContactComponents(
            full_logits=full_logits,
            interaction_logits=interaction_logits,
            raw_interaction_logits=raw_interaction_logits,
            left_unary_logits=left_unary_logits,
            right_unary_logits=right_unary_logits,
            centering_error=centering_error,
        )

    def contact_logits(
        self,
        hidden: Tensor,
        pair_batch_indices: Tensor,
        pair_left_token_indices: Tensor,
        pair_right_token_indices: Tensor,
    ) -> Tensor:
        return self.contact_components(
            hidden,
            pair_batch_indices,
            pair_left_token_indices,
            pair_right_token_indices,
        ).full_logits

    def contact_objective(
        self,
        *,
        hidden: Tensor,
        pair_batch_indices: Tensor,
        pair_left_token_indices: Tensor,
        pair_right_token_indices: Tensor,
        pair_labels: Tensor,
        ranking_weight: float = 0.0,
        centering_weight: float = 0.0,
        ranking_margin: float = 0.5,
    ) -> PairContactLoss:
        """Return the contact-only objective without evaluating retrieval heads."""

        weights = (ranking_weight, centering_weight)
        if any(not math.isfinite(value) or value < 0.0 for value in weights):
            raise ValueError("Contact objective weights must be finite and non-negative")
        if not math.isfinite(ranking_margin) or ranking_margin <= 0.0:
            raise ValueError("Contact ranking margin must be positive and finite")
        components = self.contact_components(
            hidden,
            pair_batch_indices,
            pair_left_token_indices,
            pair_right_token_indices,
        )
        if pair_labels.shape != components.full_logits.shape:
            raise ValueError("Contact labels differ from sparse contact logits")
        labels = pair_labels.to(dtype=components.full_logits.dtype)
        if not bool(((labels == 0) | (labels == 1)).all()):
            raise ValueError("Contact labels must be binary")
        positive_count = int(labels.sum().item())
        negative_count = len(labels) - positive_count
        if positive_count == 0 or negative_count == 0:
            raise ValueError("Stage-A1 contact batches require both classes")
        positive_weight = torch.tensor(
            negative_count / positive_count,
            device=components.full_logits.device,
            dtype=components.full_logits.dtype,
        )
        bce = F.binary_cross_entropy_with_logits(
            components.full_logits,
            labels,
            pos_weight=positive_weight,
        )
        ranking, row_count, column_count = _matched_interaction_ranking_loss(
            components.interaction_logits,
            labels,
            pair_batch_indices,
            pair_left_token_indices,
            pair_right_token_indices,
            margin=ranking_margin,
        )
        if ranking_weight > 0.0 and row_count + column_count == 0:
            raise ValueError(
                "Contact ranking requires at least one row- or column-matched negative"
            )
        centering = components.centering_error
        total = bce + ranking_weight * ranking + centering_weight * centering
        if not bool(torch.isfinite(total)):
            raise RuntimeError("Stage-A1 contact objective is nonfinite")
        return PairContactLoss(
            total=total,
            bce=bce,
            ranking=ranking,
            centering=centering,
            positive_count=positive_count,
            pair_count=len(labels),
            row_comparison_count=row_count,
            column_comparison_count=column_count,
        )

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
        contact_ranking_weight: float = 0.0,
        contact_centering_weight: float = 0.0,
        contact_ranking_margin: float = 0.5,
    ) -> StageA1Loss:
        weights = (
            mlm_weight,
            retrieval_weight,
            contact_weight,
            contact_ranking_weight,
            contact_centering_weight,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in weights):
            raise ValueError("Stage-A1 loss weights must be finite and non-negative")
        if not math.isfinite(contact_ranking_margin) or contact_ranking_margin <= 0.0:
            raise ValueError("Contact ranking margin must be positive and finite")
        if mlm_loss.ndim != 0 or not bool(torch.isfinite(mlm_loss)):
            raise ValueError("Stage-A1 requires one finite scalar MLM loss")
        if retrieval_hidden is None:
            retrieval_hidden = hidden
        if retrieval_hidden.shape != hidden.shape:
            raise ValueError("Retrieval and contact hidden states must have equal shapes")
        retrieval_left, retrieval_right = self.retrieval_embeddings(
            retrieval_hidden, chain_ids
        )
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
        contact = self.contact_objective(
            hidden=hidden,
            pair_batch_indices=pair_batch_indices,
            pair_left_token_indices=pair_left_token_indices,
            pair_right_token_indices=pair_right_token_indices,
            pair_labels=pair_labels,
            ranking_weight=contact_ranking_weight,
            centering_weight=contact_centering_weight,
            ranking_margin=contact_ranking_margin,
        )
        total = (
            mlm_weight * mlm_loss
            + retrieval_weight * retrieval_loss
            + contact_weight * contact.total
        )
        if not bool(torch.isfinite(total)):
            raise RuntimeError("Stage-A1 produced a nonfinite multitask loss")
        return StageA1Loss(
            total=total,
            mlm=mlm_loss,
            retrieval=retrieval_loss,
            contact=contact.total,
            contact_bce=contact.bce,
            contact_ranking=contact.ranking,
            contact_centering=contact.centering,
            retrieval_accuracy_left_to_right=left_accuracy,
            retrieval_accuracy_right_to_left=right_accuracy,
            contact_count=contact.pair_count,
            contact_positive_count=contact.positive_count,
            contact_row_comparison_count=contact.row_comparison_count,
            contact_column_comparison_count=contact.column_comparison_count,
        )
