"""Loss and negative-routing primitives for pair-contact pretraining.

The functions in this module are intentionally independent of a particular
backbone.  A distributed trainer supplies one dense pair-logit matrix per
complex; this module balances systems, endpoints, and contact classes without
using evaluation data or experimental distances as model inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class PairContactPretrainingLoss:
    total: Tensor
    pair: Tensor
    paratope: Tensor
    epitope: Tensor
    partner_swap: Tensor
    positive_pairs: int
    negative_pairs: int
    valid_systems: int


@dataclass(frozen=True)
class PairContactRankingMetrics:
    average_precision: float
    auroc: float
    mean_reciprocal_rank: float
    recall_at_1: float
    recall_at_l_over_10: float
    precision_at_l_over_10: float
    prevalence: float


class SymmetricLowRankPairContactHead(nn.Module):
    """Score every cross-chain residue pair with an order-invariant low-rank head."""

    def __init__(self, embedding_dimension: int, pair_dimension: int = 128) -> None:
        super().__init__()
        if embedding_dimension < 1 or pair_dimension < 1:
            raise ValueError("Pair-contact head dimensions must be positive")
        self.pair_dimension = pair_dimension
        self.query = nn.Linear(embedding_dimension, pair_dimension, bias=False)
        self.key = nn.Linear(embedding_dimension, pair_dimension, bias=False)
        self.source_unary = nn.Linear(embedding_dimension, 1)
        self.target_unary = nn.Linear(embedding_dimension, 1)
        self.bias = nn.Parameter(torch.zeros(()))

    def _directed(self, source: Tensor, target: Tensor) -> Tensor:
        query = self.query(source)
        key = self.key(target)
        pair = query @ key.transpose(-1, -2) / math.sqrt(self.pair_dimension)
        return (
            pair
            + self.source_unary(source)
            + self.target_unary(target).transpose(-1, -2)
            + self.bias
        )

    def forward(self, left: Tensor, right: Tensor) -> Tensor:
        if left.ndim != 3 or right.ndim != 3 or left.shape[0] != right.shape[0]:
            raise ValueError("Pair-contact head expects [B,L,D] and [B,R,D]")
        if left.shape[-1] != right.shape[-1] or left.shape[-1] != self.query.in_features:
            raise ValueError("Pair-contact hidden dimensions differ")
        forward = self._directed(left, right)
        reverse = self._directed(right, left).transpose(-1, -2)
        return (forward + reverse) / 2.0


class BipartiteEdgeRefinementContactHead(nn.Module):
    """Refine explicit complete-bipartite edge states before contact scoring.

    The head keeps the inexpensive symmetric low-rank score as a residual and
    adds a compact edge state for every valid cross-chain residue pair.  Each
    refinement step pools edge messages along both incident-node axes, then
    combines the two endpoint contexts with symmetric statistics.  Shared
    projections and symmetric statistics make swapping the partner order
    equivalent to transposing the output contact map.

    This is deliberately a head-only module: it does not introduce coordinates,
    distances, or labels as model inputs, and it can be compared with
    :class:`SymmetricLowRankPairContactHead` without changing the backbone or
    objective.
    """

    def __init__(
        self,
        embedding_dimension: int,
        *,
        pair_dimension: int = 128,
        edge_dimension: int = 32,
        refinement_steps: int = 2,
    ) -> None:
        super().__init__()
        dimensions = (embedding_dimension, pair_dimension, edge_dimension)
        if any(value < 1 for value in dimensions) or refinement_steps < 1:
            raise ValueError("Bipartite contact-head dimensions must be positive")
        self.embedding_dimension = embedding_dimension
        self.edge_dimension = edge_dimension
        self.refinement_steps = refinement_steps
        self.low_rank = SymmetricLowRankPairContactHead(
            embedding_dimension,
            pair_dimension,
        )
        self.node_projection = nn.Sequential(
            nn.LayerNorm(embedding_dimension),
            nn.Linear(embedding_dimension, edge_dimension),
        )
        self.edge_input = nn.Sequential(
            nn.Linear(3 * edge_dimension, edge_dimension),
            nn.SiLU(),
            nn.Linear(edge_dimension, edge_dimension),
        )
        self.edge_norms = nn.ModuleList(
            nn.LayerNorm(edge_dimension) for _ in range(refinement_steps)
        )
        self.edge_attention = nn.ModuleList(
            nn.Linear(edge_dimension, 1, bias=False)
            for _ in range(refinement_steps)
        )
        self.edge_updates = nn.ModuleList(
            nn.Sequential(
                nn.Linear(4 * edge_dimension, 2 * edge_dimension),
                nn.SiLU(),
                nn.Linear(2 * edge_dimension, edge_dimension),
            )
            for _ in range(refinement_steps)
        )
        self.refinement_gain = nn.Parameter(torch.full((refinement_steps,), 0.25))
        self.output_norm = nn.LayerNorm(edge_dimension)
        self.output = nn.Linear(edge_dimension, 1)
        self.output_gain = nn.Parameter(torch.tensor(0.25))

    @staticmethod
    def _validate_inputs(
        left: Tensor,
        right: Tensor,
        pair_valid: Tensor | None,
        embedding_dimension: int,
    ) -> Tensor:
        if left.ndim != 3 or right.ndim != 3 or left.shape[0] != right.shape[0]:
            raise ValueError("Bipartite contact head expects [B,L,D] and [B,R,D]")
        if left.shape[-1] != embedding_dimension or right.shape[-1] != embedding_dimension:
            raise ValueError("Bipartite contact-head hidden dimensions differ")
        expected = (left.shape[0], left.shape[1], right.shape[1])
        if pair_valid is None:
            pair_valid = torch.ones(expected, dtype=torch.bool, device=left.device)
        if pair_valid.shape != expected or pair_valid.dtype != torch.bool:
            raise ValueError("pair_valid must be a boolean [B,L,R] tensor")
        if pair_valid.device != left.device:
            raise ValueError("pair_valid and hidden states must share a device")
        if bool(pair_valid.any(dim=2).logical_not().any()) or bool(
            pair_valid.any(dim=1).logical_not().any()
        ):
            raise ValueError("Every bipartite node requires at least one valid edge")
        return pair_valid

    @staticmethod
    def _symmetric_features(left: Tensor, right: Tensor) -> Tensor:
        return torch.cat((left + right, (left - right).abs(), left * right), dim=-1)

    def forward(
        self,
        left: Tensor,
        right: Tensor,
        pair_valid: Tensor | None = None,
    ) -> Tensor:
        pair_valid = self._validate_inputs(
            left,
            right,
            pair_valid,
            self.embedding_dimension,
        )
        left_nodes = self.node_projection(left)[:, :, None, :]
        right_nodes = self.node_projection(right)[:, None, :, :]
        state = self.edge_input(self._symmetric_features(left_nodes, right_nodes))

        for step in range(self.refinement_steps):
            normalized = self.edge_norms[step](state)
            scores = self.edge_attention[step](normalized).squeeze(-1)
            scores = scores.masked_fill(~pair_valid, -torch.inf)
            row_weights = torch.softmax(scores.float(), dim=2).to(state.dtype)
            column_weights = torch.softmax(scores.float(), dim=1).to(state.dtype)
            row_context = (row_weights[..., None] * state).sum(dim=2)[:, :, None, :]
            column_context = (column_weights[..., None] * state).sum(dim=1)[:, None, :, :]
            context = self._symmetric_features(row_context, column_context)
            update = self.edge_updates[step](torch.cat((normalized, context), dim=-1))
            state = state + torch.tanh(self.refinement_gain[step]) * update

        residual = self.output(self.output_norm(state)).squeeze(-1)
        return self.low_rank(left, right) + torch.tanh(self.output_gain) * residual


def paircontact_ranking_metrics(
    scores: Tensor,
    labels: Tensor,
    valid: Tensor | None = None,
) -> PairContactRankingMetrics:
    """Compute exact ranked pair metrics for one uncropped contact map."""

    if scores.ndim != 2 or labels.shape != scores.shape or labels.dtype != torch.bool:
        raise ValueError("Pair ranking expects scores and boolean labels with shape [L,R]")
    if valid is None:
        valid = torch.ones_like(labels)
    if valid.shape != scores.shape or valid.dtype != torch.bool:
        raise ValueError("Pair-ranking validity must be a boolean [L,R] tensor")
    selected_scores = scores[valid].detach().float().cpu()
    selected_labels = labels[valid].detach().bool().cpu()
    if len(selected_scores) == 0 or not bool(torch.isfinite(selected_scores).all()):
        raise ValueError("Pair-ranking scores must be finite and non-empty")
    positive_count = int(selected_labels.sum())
    negative_count = len(selected_labels) - positive_count
    if positive_count == 0 or negative_count == 0:
        raise ValueError("Pair ranking requires positive and negative edges")

    descending = torch.argsort(selected_scores, descending=True, stable=True)
    ranked = selected_labels[descending].to(torch.float64)
    ranks = torch.arange(1, len(ranked) + 1, dtype=torch.float64)
    precision = ranked.cumsum(0) / ranks
    average_precision = float((precision * ranked).sum() / positive_count)
    first_positive_rank = int(torch.nonzero(ranked, as_tuple=False)[0, 0]) + 1
    top_k = max(1, math.ceil(min(scores.shape) / 10))
    top_positive = float(ranked[:top_k].sum())

    ascending_scores, ascending_order = torch.sort(selected_scores, stable=True)
    ascending_labels = selected_labels[ascending_order]
    _, counts = torch.unique_consecutive(ascending_scores, return_counts=True)
    group_ends = counts.cumsum(0).to(torch.float64)
    average_ranks = group_ends - (counts.to(torch.float64) - 1.0) / 2.0
    tied_ranks = torch.repeat_interleave(average_ranks, counts)
    positive_rank_sum = tied_ranks[ascending_labels].sum()
    auroc = float(
        (
            positive_rank_sum
            - positive_count * (positive_count + 1) / 2.0
        )
        / (positive_count * negative_count)
    )
    return PairContactRankingMetrics(
        average_precision=average_precision,
        auroc=auroc,
        mean_reciprocal_rank=1.0 / first_positive_rank,
        recall_at_1=float(ranked[0]),
        recall_at_l_over_10=top_positive / positive_count,
        precision_at_l_over_10=top_positive / top_k,
        prevalence=positive_count / len(selected_labels),
    )


def build_partner_swap_permutation(allowed: Tensor, *, seed: int) -> Tensor:
    """Return a deterministic perfect matching over allowed partner swaps.

    ``allowed[i, j]`` must already exclude cognate pairs, known positives,
    exact endpoint reuse, and endpoint-cluster reuse.  The returned vector maps
    every left endpoint to one unique right endpoint.  Failure to find a
    perfect matching is explicit instead of silently admitting false negatives.
    """

    if allowed.ndim != 2 or allowed.shape[0] != allowed.shape[1]:
        raise ValueError("Partner-swap mask must be square")
    if allowed.dtype != torch.bool or allowed.device.type != "cpu":
        raise ValueError("Partner-swap matching requires a CPU boolean mask")
    size = allowed.shape[0]
    if size < 2 or bool(allowed.sum(dim=1).eq(0).any()):
        raise ValueError("Every endpoint requires at least one allowed swap")
    if not isinstance(seed, int):
        raise TypeError("Partner-swap seed must be an integer")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed % (2**63 - 1))
    row_order = torch.randperm(size, generator=generator).tolist()
    column_priorities = [
        torch.randperm(size, generator=generator).tolist() for _ in range(size)
    ]
    column_to_row = [-1] * size

    def augment(row: int, visited: list[bool]) -> bool:
        for column in column_priorities[row]:
            if not bool(allowed[row, column]) or visited[column]:
                continue
            visited[column] = True
            previous = column_to_row[column]
            if previous == -1 or augment(previous, visited):
                column_to_row[column] = row
                return True
        return False

    for row in row_order:
        if not augment(row, [False] * size):
            raise ValueError("Allowed partner swaps have no perfect matching")
    row_to_column = torch.empty(size, dtype=torch.long)
    for column, row in enumerate(column_to_row):
        row_to_column[row] = column
    if not bool(allowed[torch.arange(size), row_to_column].all()):
        raise RuntimeError("Partner-swap matching violated the allowed mask")
    return row_to_column


def adaptive_negative_weights(
    logits: Tensor,
    row_indices: Tensor,
    column_indices: Tensor,
    *,
    temperature: float,
) -> Tensor:
    """Combine detached confidence weighting with endpoint-degree balancing."""

    if logits.ndim != 1 or row_indices.shape != logits.shape or column_indices.shape != logits.shape:
        raise ValueError("Negative logits and endpoint indices must be aligned")
    if len(logits) == 0:
        raise ValueError("Adaptive negative weighting requires negatives")
    if row_indices.dtype != torch.long or column_indices.dtype != torch.long:
        raise ValueError("Endpoint indices must be integer tensors")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("Hard-negative temperature must be positive and finite")

    row_degree = torch.bincount(row_indices).clamp_min(1).to(logits.dtype)
    column_degree = torch.bincount(column_indices).clamp_min(1).to(logits.dtype)
    degree = 0.5 * (
        row_degree[row_indices].reciprocal()
        + column_degree[column_indices].reciprocal()
    )
    confidence = torch.softmax(logits.detach().float() / temperature, dim=0).to(
        logits.dtype
    )
    weights = degree * confidence
    return weights * (len(weights) / weights.sum().clamp_min(torch.finfo(weights.dtype).tiny))


def _balanced_binary_loss(logits: Tensor, labels: Tensor) -> Tensor:
    positive = labels
    negative = ~labels
    if not bool(positive.any()) or not bool(negative.any()):
        return F.binary_cross_entropy_with_logits(logits, labels.to(logits.dtype))
    return 0.5 * (
        F.softplus(-logits[positive]).mean()
        + F.softplus(logits[negative]).mean()
    )


def _balanced_focal_pair_loss(
    positive_logits: Tensor,
    negative_logits: Tensor,
    *,
    gamma: float,
) -> Tensor:
    if not math.isfinite(gamma) or gamma < 0.0:
        raise ValueError("Focal gamma must be finite and non-negative")
    positive_modulation = torch.sigmoid(-positive_logits).pow(gamma)
    negative_modulation = torch.sigmoid(negative_logits).pow(gamma)
    return 0.5 * (
        (positive_modulation * F.softplus(-positive_logits)).mean()
        + (negative_modulation * F.softplus(negative_logits)).mean()
    )


def _soft_endpoint_max(logits: Tensor, valid: Tensor, *, dim: int, temperature: float) -> Tensor:
    masked = logits.masked_fill(~valid, -torch.inf)
    return temperature * torch.logsumexp(masked / temperature, dim=dim)


def paircontact_pretraining_loss(
    pair_logits: Tensor,
    pair_labels: Tensor,
    pair_valid: Tensor,
    *,
    system_weights: Tensor | None = None,
    swapped_pair_logits: Tensor | None = None,
    swapped_pair_valid: Tensor | None = None,
    pair_weight: float = 1.0,
    paratope_weight: float = 0.2,
    epitope_weight: float = 0.2,
    partner_swap_weight: float = 0.2,
    pair_loss_mode: str = "adaptive_balanced",
    focal_gamma: float = 2.0,
    hard_negative_temperature: float = 0.5,
    marginal_temperature: float = 0.25,
) -> PairContactPretrainingLoss:
    """Compute complex-balanced contact and interface auxiliary losses.

    ``system_weights`` are frozen upstream weights (for example inverse
    exact-pair/PDB multiplicity), normalized here to mean one.  Partner swaps
    are all-negative examples produced only by
    :func:`build_partner_swap_permutation` after known-positive exclusions.
    """

    if pair_logits.ndim != 3 or pair_labels.shape != pair_logits.shape or pair_valid.shape != pair_logits.shape:
        raise ValueError("Pair logits, labels, and validity masks must have shape [B, L, R]")
    if pair_labels.dtype != torch.bool or pair_valid.dtype != torch.bool:
        raise ValueError("Pair labels and validity masks must be boolean")
    if not bool(torch.isfinite(pair_logits[pair_valid]).all()):
        raise ValueError("Valid pair logits must be finite")
    weights = (pair_weight, paratope_weight, epitope_weight, partner_swap_weight)
    if any(not math.isfinite(value) or value < 0.0 for value in weights):
        raise ValueError("Objective weights must be finite and non-negative")
    if not math.isfinite(marginal_temperature) or marginal_temperature <= 0.0:
        raise ValueError("Marginal temperature must be positive and finite")
    if pair_loss_mode not in {"adaptive_balanced", "balanced_focal"}:
        raise ValueError("Pair loss mode is not reviewed")

    batch_size = pair_logits.shape[0]
    if system_weights is None:
        system_weights = pair_logits.new_ones(batch_size)
    if system_weights.shape != (batch_size,) or not bool(
        torch.isfinite(system_weights).all() & system_weights.gt(0).all()
    ):
        raise ValueError("System weights must be finite positive values")
    system_weights = system_weights / system_weights.mean()

    pair_losses: list[Tensor] = []
    paratope_losses: list[Tensor] = []
    epitope_losses: list[Tensor] = []
    positive_count = 0
    negative_count = 0
    for batch_index in range(batch_size):
        valid = pair_valid[batch_index]
        labels = pair_labels[batch_index]
        if not bool(valid.any()) or bool((labels & ~valid).any()):
            raise ValueError("Every system requires valid, in-mask contact labels")
        positive = valid & labels
        negative = valid & ~labels
        if not bool(positive.any()) or not bool(negative.any()):
            raise ValueError("Every contact-supervised system requires both classes")

        positive_logits = pair_logits[batch_index][positive]
        negative_logits = pair_logits[batch_index][negative]
        if pair_loss_mode == "adaptive_balanced":
            negative_endpoints = negative.nonzero(as_tuple=False)
            negative_weights = adaptive_negative_weights(
                negative_logits,
                negative_endpoints[:, 0],
                negative_endpoints[:, 1],
                temperature=hard_negative_temperature,
            )
            pair_losses.append(
                0.5
                * (
                    F.softplus(-positive_logits).mean()
                    + (negative_weights * F.softplus(negative_logits)).mean()
                )
            )
        else:
            pair_losses.append(
                _balanced_focal_pair_loss(
                    positive_logits,
                    negative_logits,
                    gamma=focal_gamma,
                )
            )

        paratope_valid = valid.any(dim=1)
        epitope_valid = valid.any(dim=0)
        paratope_targets = positive.any(dim=1)
        epitope_targets = positive.any(dim=0)
        paratope_logits = _soft_endpoint_max(
            pair_logits[batch_index], valid, dim=1, temperature=marginal_temperature
        )
        epitope_logits = _soft_endpoint_max(
            pair_logits[batch_index], valid, dim=0, temperature=marginal_temperature
        )
        paratope_losses.append(
            _balanced_binary_loss(
                paratope_logits[paratope_valid], paratope_targets[paratope_valid]
            )
        )
        epitope_losses.append(
            _balanced_binary_loss(
                epitope_logits[epitope_valid], epitope_targets[epitope_valid]
            )
        )
        positive_count += int(positive.sum())
        negative_count += int(negative.sum())

    def weighted_mean(values: list[Tensor]) -> Tensor:
        stacked = torch.stack(values)
        return (stacked * system_weights).sum() / system_weights.sum()

    pair_loss = weighted_mean(pair_losses)
    paratope_loss = weighted_mean(paratope_losses)
    epitope_loss = weighted_mean(epitope_losses)

    if swapped_pair_logits is None:
        if partner_swap_weight > 0.0:
            raise ValueError("A positive partner-swap weight requires swapped logits")
        swap_loss = pair_logits.sum() * 0.0
    else:
        if swapped_pair_valid is None or swapped_pair_valid.shape != swapped_pair_logits.shape:
            raise ValueError("Swapped pair logits require an aligned validity mask")
        selected = swapped_pair_logits[swapped_pair_valid]
        if len(selected) == 0 or not bool(torch.isfinite(selected).all()):
            raise ValueError("Swapped pair supervision must be finite and non-empty")
        swap_weights = torch.softmax(
            selected.detach().float() / hard_negative_temperature, dim=0
        ).to(selected.dtype)
        swap_weights = swap_weights * len(selected)
        swap_loss = (swap_weights * F.softplus(selected)).mean()

    total = (
        pair_weight * pair_loss
        + paratope_weight * paratope_loss
        + epitope_weight * epitope_loss
        + partner_swap_weight * swap_loss
    )
    if not bool(torch.isfinite(total)):
        raise RuntimeError("Pair-contact pretraining objective is nonfinite")
    return PairContactPretrainingLoss(
        total=total,
        pair=pair_loss,
        paratope=paratope_loss,
        epitope=epitope_loss,
        partner_swap=swap_loss,
        positive_pairs=positive_count,
        negative_pairs=negative_count,
        valid_systems=batch_size,
    )
