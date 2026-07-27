"""Score-free multi-decoy partner objective for PAIRScale repair experiments."""

from __future__ import annotations

import hashlib
import math
from typing import Any, Mapping, Sequence

from .development_contact_dataset import DevelopmentContactExample
from .pair_geometry import double_centered_partner_score
from .pair_geometry_losses import masked_contact_bce_with_logits
from .pairscale_no_centering_training import locked_no_centering_config
from .pairscale_overfit import (
    PairscaleOverfitConfig,
    _forward_contact,
    _forward_cross_pair,
    hard_negative_contact_ranking_loss,
)


DECOY_COUNT = 4
DECOY_POLICY = "score_free_length_matched_distinct_antigen_sha256_tie_break_v1"
PARTNER_OBJECTIVE = "double_centered_equal_mean_pairwise_softplus_v1"


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _antigen_length(example: DevelopmentContactExample) -> int:
    shape = getattr(example.antigen_embeddings, "shape", None)
    if shape is None or len(shape) != 2:
        raise ValueError("Antigen embeddings must be a two-dimensional tensor")
    return _positive_integer(int(shape[0]), "antigen length")


def _tie_digest(
    matched: DevelopmentContactExample,
    candidate: DevelopmentContactExample,
    seed: int,
) -> str:
    payload = f"{seed}:{matched.instance_id}:{candidate.instance_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def select_score_free_length_matched_decoys(
    examples: Sequence[DevelopmentContactExample],
    *,
    count: int = DECOY_COUNT,
    seed: int = 17,
) -> tuple[tuple[int, ...], ...]:
    """Select distinct-antigen decoys without reading labels or feature values."""

    count = _positive_integer(count, "decoy count")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if len(examples) < count + 1:
        raise ValueError("The example inventory is too small for the decoy count")
    result: list[tuple[int, ...]] = []
    for matched_index, matched in enumerate(examples):
        matched_length = _antigen_length(matched)
        candidates = sorted(
            (
                (
                    abs(_antigen_length(candidate) - matched_length),
                    _tie_digest(matched, candidate, seed),
                    candidate.instance_id,
                    candidate_index,
                )
                for candidate_index, candidate in enumerate(examples)
                if candidate_index != matched_index
                and candidate.antigen_id != matched.antigen_id
            )
        )
        selected: list[int] = []
        selected_antigens: set[str] = set()
        for _, _, _, candidate_index in candidates:
            antigen_id = examples[candidate_index].antigen_id
            if antigen_id in selected_antigens:
                continue
            selected.append(candidate_index)
            selected_antigens.add(antigen_id)
            if len(selected) == count:
                break
        if len(selected) != count:
            raise ValueError("Insufficient distinct non-cognate antigen identifiers")
        result.append(tuple(selected))
    return tuple(result)


def centered_multidecoy_partner_ranking_loss(
    matched_outputs: Mapping[str, Any],
    decoy_outputs: Sequence[Mapping[str, Any]],
    *,
    margin: float,
    temperature: float,
) -> Any:
    """Rank one matched pair above score-free decoys using centered scores only."""

    import torch

    if not math.isfinite(margin) or margin <= 0.0:
        raise ValueError("margin must be positive and finite")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be positive and finite")
    if not decoy_outputs:
        raise ValueError("At least one decoy output is required")
    matched_logits = matched_outputs["contact_logits"].float()
    _, matched_score = double_centered_partner_score(
        matched_logits,
        matched_outputs["candidate_pair_mask"],
        centering_iterations=32,
    )
    if int(matched_score.numel()) != 1:
        raise ValueError("The repair objective requires a single matched complex")
    losses: list[Any] = []
    for decoy in decoy_outputs:
        decoy_logits = decoy["contact_logits"].float()
        _, decoy_score = double_centered_partner_score(
            decoy_logits,
            decoy["candidate_pair_mask"],
            centering_iterations=32,
        )
        if int(decoy_score.numel()) != 1:
            raise ValueError("Every repair decoy must contain one complex")
        losses.append(
            torch.nn.functional.softplus(
                (margin - matched_score[0] + decoy_score[0]) / temperature
            )
            * temperature
        )
    return torch.stack(losses).mean()


def train_one_epoch_centered_multidecoy(
    *,
    model: Any,
    optimizer: Any,
    examples: Sequence[DevelopmentContactExample],
    device: Any,
    config: PairscaleOverfitConfig,
    epoch: int,
    decoy_indices: Sequence[Sequence[int]],
) -> tuple[Mapping[str, float], int]:
    """Train one epoch with the fixed centered-only multi-decoy objective."""

    import torch

    if len(decoy_indices) != len(examples):
        raise ValueError("Decoy assignments must align with examples")
    model.train()
    loss_sums = {
        "total": 0.0,
        "contact_bce": 0.0,
        "contact_ranking": 0.0,
        "partner_ranking_centered_multidecoy": 0.0,
    }
    step_count = 0
    use_bf16 = config.precision == "bf16" and device.type == "cuda"
    from .pairscale_overfit import deterministic_epoch_order

    for index in deterministic_epoch_order(len(examples), config.seed, epoch):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16 if use_bf16 else torch.float32,
            enabled=use_bf16,
        ):
            outputs, targets = _forward_contact(model, examples[index], device)
            contact_bce = masked_contact_bce_with_logits(
                outputs["contact_logits"],
                targets,
                outputs["candidate_pair_mask"],
                positive_weight=config.positive_weight,
            )
            contact_ranking = hard_negative_contact_ranking_loss(
                outputs["contact_logits"],
                targets,
                outputs["candidate_pair_mask"],
                negative_ratio=config.hard_negative_ratio,
                temperature=config.contact_ranking_temperature,
            )
            decoys = [
                _forward_cross_pair(
                    model,
                    examples[index],
                    examples[decoy_index],
                    device,
                )
                for decoy_index in decoy_indices[index]
            ]
            partner_ranking = centered_multidecoy_partner_ranking_loss(
                outputs,
                decoys,
                margin=config.partner_ranking_margin,
                temperature=config.partner_ranking_temperature,
            )
            loss = (
                contact_bce
                + config.contact_ranking_weight * contact_ranking
                + config.partner_ranking_weight * partner_ranking
            )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"Non-finite loss at epoch {epoch}, example {index}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config.gradient_clip_norm,
            error_if_nonfinite=True,
        )
        if not bool(torch.isfinite(gradient_norm)):
            raise RuntimeError("Non-finite gradient norm")
        optimizer.step()
        for name, value in (
            ("total", loss),
            ("contact_bce", contact_bce),
            ("contact_ranking", contact_ranking),
            ("partner_ranking_centered_multidecoy", partner_ranking),
        ):
            loss_sums[name] += float(value.detach().cpu())
        step_count += 1
    return {
        name: value / step_count for name, value in loss_sums.items()
    }, step_count


def locked_centered_multidecoy_config() -> PairscaleOverfitConfig:
    """Reuse every frozen architecture and optimization field from the control."""

    return locked_no_centering_config()


__all__ = [
    "DECOY_COUNT",
    "DECOY_POLICY",
    "PARTNER_OBJECTIVE",
    "centered_multidecoy_partner_ranking_loss",
    "locked_centered_multidecoy_config",
    "select_score_free_length_matched_decoys",
    "train_one_epoch_centered_multidecoy",
]
