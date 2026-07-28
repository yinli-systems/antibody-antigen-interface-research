"""Bidirectional score-free partner objectives for PAIRScale E1/E2."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Mapping, Sequence

from .development_contact_dataset import DevelopmentContactExample


DECOY_POLICY = "score_free_bidirectional_length_matched_incidence_excluded_v1"
PARTNER_SCORE_SOURCE = (
    "double_centered_independent_specificity_interaction_residual"
)
_TIE_DOMAIN = "pairscale-e1-e2-decoy-v1"


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_positive(value: Any, name: str) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be positive and finite") from error
    if not math.isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return converted


def _endpoint_length(example: DevelopmentContactExample, axis: str) -> int:
    if axis == "antibody":
        tensor = example.antibody_embeddings
    elif axis == "antigen":
        tensor = example.antigen_embeddings
    else:
        raise ValueError("axis must be antibody or antigen")
    shape = getattr(tensor, "shape", None)
    if shape is None or len(shape) != 2:
        raise ValueError(f"{axis} embeddings must be a two-dimensional tensor")
    return _positive_integer(int(shape[0]), f"{axis} length")


def _tie_digest(
    matched: DevelopmentContactExample,
    candidate: DevelopmentContactExample,
    *,
    seed: int,
    direction: str,
) -> str:
    payload = (
        f"{_TIE_DOMAIN}:{seed}:{direction}:"
        f"{matched.instance_id}:{candidate.instance_id}"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class BidirectionalDecoyPanel:
    """Store aligned score-free decoys for both retrieval directions."""

    antibody_to_antigen: tuple[tuple[int, ...], ...]
    antigen_to_antibody: tuple[tuple[int, ...], ...]
    count_per_direction: int
    seed: int

    def validate(self, example_count: int) -> None:
        """Fail closed when a panel is incomplete or misaligned."""
        _positive_integer(example_count, "example_count")
        _positive_integer(self.count_per_direction, "count_per_direction")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        for name, rows in (
            ("antibody_to_antigen", self.antibody_to_antigen),
            ("antigen_to_antibody", self.antigen_to_antibody),
        ):
            if len(rows) != example_count:
                raise ValueError(f"{name} must align with the example inventory")
            for row in rows:
                if len(row) != self.count_per_direction:
                    raise ValueError(f"{name} contains an incomplete decoy row")
                if len(set(row)) != len(row):
                    raise ValueError(f"{name} contains duplicate indices")
                if any(
                    isinstance(index, bool)
                    or not isinstance(index, int)
                    or not 0 <= index < example_count
                    for index in row
                ):
                    raise ValueError(f"{name} contains an invalid index")


def _select_direction(
    examples: Sequence[DevelopmentContactExample],
    *,
    observed_pairs: frozenset[tuple[str, str]],
    count: int,
    seed: int,
    direction: str,
) -> tuple[tuple[int, ...], ...]:
    if direction == "antibody_to_antigen":
        replaced_axis = "antigen"
    elif direction == "antigen_to_antibody":
        replaced_axis = "antibody"
    else:
        raise ValueError("Unsupported retrieval direction")
    rows: list[tuple[int, ...]] = []
    for matched_index, matched in enumerate(examples):
        matched_length = _endpoint_length(matched, replaced_axis)
        candidates: list[tuple[int, str, str, int]] = []
        for candidate_index, candidate in enumerate(examples):
            if candidate_index == matched_index:
                continue
            if direction == "antibody_to_antigen":
                replaced_id = candidate.antigen_id
                if replaced_id == matched.antigen_id:
                    continue
                cross_pair = (matched.antibody_id, replaced_id)
            else:
                replaced_id = candidate.antibody_id
                if replaced_id == matched.antibody_id:
                    continue
                cross_pair = (replaced_id, matched.antigen_id)
            if cross_pair in observed_pairs:
                continue
            candidates.append(
                (
                    abs(
                        _endpoint_length(candidate, replaced_axis)
                        - matched_length
                    ),
                    _tie_digest(
                        matched,
                        candidate,
                        seed=seed,
                        direction=direction,
                    ),
                    candidate.instance_id,
                    candidate_index,
                )
            )
        selected: list[int] = []
        selected_ids: set[str] = set()
        for _, _, _, candidate_index in sorted(candidates):
            candidate = examples[candidate_index]
            replaced_id = (
                candidate.antigen_id
                if direction == "antibody_to_antigen"
                else candidate.antibody_id
            )
            if replaced_id in selected_ids:
                continue
            selected.append(candidate_index)
            selected_ids.add(replaced_id)
            if len(selected) == count:
                break
        if len(selected) != count:
            raise ValueError(
                f"Insufficient incidence-excluded decoys for {direction} "
                f"at matched index {matched_index}"
            )
        rows.append(tuple(selected))
    return tuple(rows)


def select_score_free_bidirectional_decoys(
    examples: Sequence[DevelopmentContactExample],
    *,
    count: int = 4,
    seed: int = 17,
) -> BidirectionalDecoyPanel:
    """Select symmetric unknown cross-pairs without reading labels or values."""
    count = _positive_integer(count, "decoy count")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if len(examples) < count + 1:
        raise ValueError("The example inventory is too small for the decoy count")
    instance_ids = [example.instance_id for example in examples]
    if len(set(instance_ids)) != len(instance_ids):
        raise ValueError("Instance identifiers must be unique")
    observed_pairs = frozenset(
        (example.antibody_id, example.antigen_id) for example in examples
    )
    panel = BidirectionalDecoyPanel(
        antibody_to_antigen=_select_direction(
            examples,
            observed_pairs=observed_pairs,
            count=count,
            seed=seed,
            direction="antibody_to_antigen",
        ),
        antigen_to_antibody=_select_direction(
            examples,
            observed_pairs=observed_pairs,
            count=count,
            seed=seed,
            direction="antigen_to_antibody",
        ),
        count_per_direction=count,
        seed=seed,
    )
    panel.validate(len(examples))
    return panel


def _validate_score_matrix(scores: Any, positive_mask: Any) -> tuple[Any, Any]:
    import torch

    if not torch.is_tensor(scores) or scores.ndim != 2:
        raise ValueError("scores must have shape [anchors, candidates]")
    if not torch.is_floating_point(scores):
        raise ValueError("scores must use a floating-point dtype")
    if not torch.is_tensor(positive_mask):
        raise TypeError("positive_mask must be a PyTorch tensor")
    if positive_mask.dtype != torch.bool or tuple(positive_mask.shape) != tuple(
        scores.shape
    ):
        raise ValueError("positive_mask must be boolean and align with scores")
    if positive_mask.device != scores.device:
        raise ValueError("scores and positive_mask must share a device")
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("scores must be finite")
    positive_counts = positive_mask.sum(dim=1)
    negative_counts = (~positive_mask).sum(dim=1)
    if bool((positive_counts == 0).any()):
        raise ValueError("Every anchor must contain at least one positive")
    if bool((negative_counts == 0).any()):
        raise ValueError("Every anchor must contain at least one negative")
    return scores.float(), positive_mask


def multi_positive_retrieval_loss(
    scores: Any,
    positive_mask: Any,
    *,
    temperature: float,
) -> Any:
    """Maximize the total softmax mass assigned to all known positives."""
    import torch

    temperature = _finite_positive(temperature, "temperature")
    scores, positive_mask = _validate_score_matrix(scores, positive_mask)
    scaled = scores / temperature
    negative_infinity = torch.tensor(
        float("-inf"),
        device=scaled.device,
        dtype=scaled.dtype,
    )
    positive_scaled = torch.where(positive_mask, scaled, negative_infinity)
    log_positive_mass = torch.logsumexp(positive_scaled, dim=1)
    log_total_mass = torch.logsumexp(scaled, dim=1)
    return (-(log_positive_mass - log_total_mass)).mean()


def multi_positive_margin_loss(
    scores: Any,
    positive_mask: Any,
    *,
    margin: float,
    temperature: float,
) -> Any:
    """Apply a softplus margin to every positive-negative score pair."""
    import torch

    margin = _finite_positive(margin, "margin")
    temperature = _finite_positive(temperature, "temperature")
    scores, positive_mask = _validate_score_matrix(scores, positive_mask)
    losses: list[Any] = []
    for row, row_mask in zip(scores, positive_mask, strict=True):
        positives = row[row_mask]
        negatives = row[~row_mask]
        differences = (
            margin - positives.unsqueeze(1) + negatives.unsqueeze(0)
        ) / temperature
        losses.append(
            torch.nn.functional.softplus(differences).mean() * temperature
        )
    return torch.stack(losses).mean()


@dataclass(frozen=True)
class BidirectionalSpecificityLosses:
    """Expose every specificity loss term before weighted aggregation."""

    retrieval: Any
    margin: Any
    antibody_to_antigen_retrieval: Any
    antigen_to_antibody_retrieval: Any
    antibody_to_antigen_margin: Any
    antigen_to_antibody_margin: Any


def bidirectional_specificity_losses(
    *,
    antibody_to_antigen_scores: Any,
    antibody_to_antigen_positive_mask: Any,
    antigen_to_antibody_scores: Any,
    antigen_to_antibody_positive_mask: Any,
    retrieval_temperature: float,
    margin: float,
    margin_temperature: float,
) -> BidirectionalSpecificityLosses:
    """Compute equal-weight retrieval and margin terms in both directions."""
    import torch

    ab_retrieval = multi_positive_retrieval_loss(
        antibody_to_antigen_scores,
        antibody_to_antigen_positive_mask,
        temperature=retrieval_temperature,
    )
    ag_retrieval = multi_positive_retrieval_loss(
        antigen_to_antibody_scores,
        antigen_to_antibody_positive_mask,
        temperature=retrieval_temperature,
    )
    ab_margin = multi_positive_margin_loss(
        antibody_to_antigen_scores,
        antibody_to_antigen_positive_mask,
        margin=margin,
        temperature=margin_temperature,
    )
    ag_margin = multi_positive_margin_loss(
        antigen_to_antibody_scores,
        antigen_to_antibody_positive_mask,
        margin=margin,
        temperature=margin_temperature,
    )
    return BidirectionalSpecificityLosses(
        retrieval=torch.stack((ab_retrieval, ag_retrieval)).mean(),
        margin=torch.stack((ab_margin, ag_margin)).mean(),
        antibody_to_antigen_retrieval=ab_retrieval,
        antigen_to_antibody_retrieval=ag_retrieval,
        antibody_to_antigen_margin=ab_margin,
        antigen_to_antibody_margin=ag_margin,
    )


def require_specificity_partner_score(output: Mapping[str, Any]) -> Any:
    """Return a scalar-batch score only from the independent residual path."""
    import torch

    if output.get("partner_score_source") != PARTNER_SCORE_SOURCE:
        raise ValueError(
            "Partner score must come from the independent centered "
            "specificity interaction residual"
        )
    score = output.get("partner_score")
    if not torch.is_tensor(score) or score.ndim != 1:
        raise ValueError("partner_score must have shape [batch]")
    if not torch.is_floating_point(score) or not bool(torch.isfinite(score).all()):
        raise ValueError("partner_score must be finite and floating point")
    return score


def raw_specificity_partner_score(
    output: Mapping[str, Any],
    *,
    temperature: float = 1.0,
) -> Any:
    """Aggregate the uncentered independent residual for diagnostic controls."""
    import torch

    from .scoring import masked_logmeanexp_score

    temperature = _finite_positive(temperature, "temperature")
    if output.get("partner_score_source") != PARTNER_SCORE_SOURCE:
        raise ValueError(
            "Raw partner score requires the independent specificity residual"
        )
    residual = output.get("specificity_interaction_residual_logits")
    mask = output.get("candidate_pair_mask")
    if not torch.is_tensor(residual) or residual.ndim != 3:
        raise ValueError(
            "specificity residual must have shape [batch, antibody, antigen]"
        )
    if not torch.is_tensor(mask) or mask.dtype != torch.bool:
        raise ValueError("candidate_pair_mask must be boolean")
    if tuple(mask.shape) != tuple(residual.shape) or mask.device != residual.device:
        raise ValueError("specificity residual and candidate mask must align")
    return masked_logmeanexp_score(
        residual.float().transpose(1, 2),
        mask.transpose(1, 2),
        temperature=temperature,
    )


def build_single_anchor_score_row(
    matched_output: Mapping[str, Any],
    decoy_outputs: Sequence[Mapping[str, Any]],
) -> Any:
    """Build one positive-first retrieval row from audited residual scores."""
    import torch

    if not decoy_outputs:
        raise ValueError("At least one decoy output is required")
    scores = [require_specificity_partner_score(matched_output)]
    scores.extend(require_specificity_partner_score(item) for item in decoy_outputs)
    if any(int(score.numel()) != 1 for score in scores):
        raise ValueError("Single-anchor rows require one complex per output")
    return torch.stack([score[0] for score in scores]).unsqueeze(0)


__all__ = [
    "BidirectionalDecoyPanel",
    "BidirectionalSpecificityLosses",
    "DECOY_POLICY",
    "PARTNER_SCORE_SOURCE",
    "bidirectional_specificity_losses",
    "build_single_anchor_score_row",
    "multi_positive_margin_loss",
    "multi_positive_retrieval_loss",
    "raw_specificity_partner_score",
    "require_specificity_partner_score",
    "select_score_free_bidirectional_decoys",
]
