"""Joint sparse-interface and partner-ranking objectives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError:
    torch = None
    F = None


def require_torch() -> None:
    """Fail with a focused message when loss code is used without PyTorch."""
    if torch is None or F is None:
        raise RuntimeError("PyTorch is required for interaction-first losses")


@dataclass(frozen=True)
class InteractionLossWeights:
    """Explicit weights for every supervised objective."""

    edge: float = 1.0
    epitope_bce: float = 0.25
    paratope_bce: float = 0.25
    epitope_dice: float = 0.25
    paratope_dice: float = 0.25
    edge_count: float = 0.0
    epitope_count: float = 0.0
    paratope_count: float = 0.0
    epitope_consistency: float = 0.0
    paratope_consistency: float = 0.0
    partner_rank: float = 0.25
    focal_gamma: float = 0.0
    margin_separation: float = 0.0
    margin_value: float = 2.0

    def validate(self) -> None:
        """Reject negative or entirely disabled loss configurations."""
        values = tuple(self.__dict__.values())
        if any(value < 0.0 for value in values):
            raise ValueError("Loss weights cannot be negative")
        if sum(values) <= 0.0:
            raise ValueError("At least one loss weight must be positive")


def masked_focal_loss_with_logits(
    logits: Any,
    targets: Any,
    mask: Any,
    positive_weight: float = 1.0,
    gamma: float = 2.0,
) -> Any:
    """Compute stable class-reweighted Focal Loss over valid entries only."""
    require_torch()
    if logits.shape != targets.shape or logits.shape != mask.shape:
        raise ValueError("Logits, targets, and mask must have identical shapes")
    if mask.dtype != torch.bool or not torch.any(mask):
        raise ValueError("Focal mask must contain at least one valid boolean entry")
    if positive_weight <= 0.0:
        raise ValueError("positive_weight must be positive")
    
    selected_logits = logits[mask]
    selected_targets = targets[mask].to(dtype=logits.dtype)
    pos_weight = logits.new_tensor(float(positive_weight))
    
    bce_loss = F.binary_cross_entropy_with_logits(
        selected_logits,
        selected_targets,
        pos_weight=pos_weight,
        reduction='none'
    )
    
    probs = torch.sigmoid(selected_logits)
    p_t = probs * selected_targets + (1 - probs) * (1 - selected_targets)
    focal_weight = (1 - p_t) ** gamma
    
    return (focal_weight * bce_loss).mean()

def masked_bce_with_logits(
    logits: Any,
    targets: Any,
    mask: Any,
    positive_weight: float = 1.0,
) -> Any:
    """Compute stable class-reweighted BCE over valid entries only."""
    require_torch()
    if logits.shape != targets.shape or logits.shape != mask.shape:
        raise ValueError("Logits, targets, and mask must have identical shapes")
    if mask.dtype != torch.bool or not torch.any(mask):
        raise ValueError("BCE mask must contain at least one valid boolean entry")
    if positive_weight <= 0.0:
        raise ValueError("positive_weight must be positive")
    selected_logits = logits[mask]
    selected_targets = targets[mask].to(dtype=logits.dtype)
    pos_weight = logits.new_tensor(float(positive_weight))
    return F.binary_cross_entropy_with_logits(
        selected_logits,
        selected_targets,
        pos_weight=pos_weight,
    )


def masked_probability_bce(
    probabilities: Any,
    targets: Any,
    mask: Any,
    positive_weight: float = 1.0,
) -> Any:
    """Compute weighted BCE for probabilities derived from a contact map."""
    require_torch()
    if probabilities.shape != targets.shape or probabilities.shape != mask.shape:
        raise ValueError("Probabilities, targets, and mask must have identical shapes")
    if mask.dtype != torch.bool or not torch.any(mask):
        raise ValueError("Probability BCE mask must contain valid entries")
    probabilities = probabilities[mask].clamp(1e-7, 1.0 - 1e-7)
    targets = targets[mask].to(dtype=probabilities.dtype)
    weights = torch.where(
        targets > 0.5,
        probabilities.new_tensor(float(positive_weight)),
        probabilities.new_tensor(1.0),
    )
    return F.binary_cross_entropy(probabilities, targets, weight=weights)


def soft_dice_loss(probabilities: Any, targets: Any, mask: Any, epsilon: float = 1.0) -> Any:
    """Compute mean per-complex Dice loss for a padded node batch."""
    require_torch()
    if probabilities.shape != targets.shape or probabilities.shape != mask.shape:
        raise ValueError("Dice inputs must have identical shapes")
    losses = []
    for batch_index in range(probabilities.shape[0]):
        valid = mask[batch_index]
        if not torch.any(valid):
            continue
        prediction = probabilities[batch_index][valid]
        truth = targets[batch_index][valid].to(dtype=prediction.dtype)
        numerator = 2.0 * (prediction * truth).sum() + epsilon
        denominator = prediction.sum() + truth.sum() + epsilon
        losses.append(1.0 - numerator / denominator)
    if not losses:
        raise ValueError("Dice loss received no valid complexes")
    return torch.stack(losses).mean()


def relative_count_loss(probabilities: Any, targets: Any, mask: Any) -> Any:
    """Match per-complex positive mass after normalizing by true interface size."""
    require_torch()
    if probabilities.shape != targets.shape or probabilities.shape != mask.shape:
        raise ValueError("Count-loss inputs must have identical shapes")
    losses = []
    for batch_index in range(probabilities.shape[0]):
        valid = mask[batch_index]
        if not torch.any(valid):
            continue
        predicted_count = probabilities[batch_index][valid].sum()
        true_count = targets[batch_index][valid].to(dtype=probabilities.dtype).sum()
        scale = true_count.detach().clamp_min(1.0)
        losses.append(F.smooth_l1_loss(predicted_count / scale, true_count / scale))
    if not losses:
        raise ValueError("Count loss received no valid complexes")
    return torch.stack(losses).mean()


def masked_consistency_loss(left: Any, right: Any, mask: Any) -> Any:
    """Align independent node heads with contact-matrix-derived marginals."""
    require_torch()
    if left.shape != right.shape or left.shape != mask.shape:
        raise ValueError("Consistency inputs must have identical shapes")
    if mask.dtype != torch.bool or not torch.any(mask):
        raise ValueError("Consistency mask must contain valid entries")
    return F.smooth_l1_loss(left[mask], right[mask])



def masked_margin_separation_loss(
    logits: Any,
    targets: Any,
    mask: Any,
    margin: float = 2.0,
) -> Any:
    """Hinge-based margin loss that pushes positive logits above +margin
    and negative logits below -margin.

    For positives:  loss = max(0, margin - logit)
    For negatives:  loss = max(0, margin + logit)

    This directly widens the score gap between contacts and non-contacts,
    producing a bimodal distribution that is robust to threshold choice.
    """
    require_torch()
    if logits.shape != targets.shape or logits.shape != mask.shape:
        raise ValueError("Margin loss inputs must have identical shapes")
    if mask.dtype != torch.bool or not torch.any(mask):
        raise ValueError("Margin mask must contain at least one valid entry")
    selected_logits = logits[mask]
    selected_targets = targets[mask].to(dtype=logits.dtype)
    # For positives (target=1): want logit > +margin → loss = max(0, margin - logit)
    # For negatives (target=0): want logit < -margin → loss = max(0, margin + logit)
    pos_loss = torch.clamp(margin - selected_logits, min=0.0) * selected_targets
    neg_loss = torch.clamp(margin + selected_logits, min=0.0) * (1.0 - selected_targets)
    return (pos_loss + neg_loss).mean()


def _validated_supervision_mask(
    outputs: Mapping[str, Any],
    key: str,
    residue_mask: Any,
) -> Any:
    """Validate a candidate-supported node mask against the residue padding mask."""
    if not torch.is_tensor(residue_mask) or residue_mask.dtype != torch.bool:
        raise ValueError("Residue supervision masks must be boolean tensors")
    if key not in outputs:
        raise KeyError(f"Model output is missing key: {key}")
    mask = outputs[key]
    if (
        not torch.is_tensor(mask)
        or mask.dtype != torch.bool
        or tuple(mask.shape) != tuple(residue_mask.shape)
    ):
        raise ValueError(f"{key} must be boolean and aligned to the residue mask")
    if mask.device != residue_mask.device:
        raise ValueError(f"{key} and the residue mask must use the same device")
    if bool(torch.any(mask & ~residue_mask)):
        raise ValueError(f"{key} cannot select padded residues")
    if not bool(torch.any(mask)):
        raise ValueError(f"{key} must contain at least one supervised residue")
    return mask


def partner_ranking_loss(
    positive_scores: Any,
    negative_scores: Any,
    temperature: float = 0.10,
) -> Any:
    """Apply complex-level InfoNCE without imposing all-zero negative contact maps."""
    require_torch()
    if positive_scores.ndim != 1 or negative_scores.ndim != 2:
        raise ValueError("Partner scores must have shapes [batch] and [batch, negatives]")
    if positive_scores.shape[0] != negative_scores.shape[0] or negative_scores.shape[1] == 0:
        raise ValueError("Partner ranking requires aligned non-empty negatives")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    logits = torch.cat((positive_scores.unsqueeze(1), negative_scores), dim=1) / temperature
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, labels)


def interaction_first_loss(
    outputs: Mapping[str, Any],
    pair_targets: Any,
    epitope_targets: Any,
    paratope_targets: Any,
    antigen_mask: Any,
    antibody_mask: Any,
    weights: InteractionLossWeights = InteractionLossWeights(),
    edge_positive_weight: float = 1.0,
    epitope_positive_weight: float = 1.0,
    paratope_positive_weight: float = 1.0,
    negative_scores: Any | None = None,
    partner_temperature: float = 0.10,
) -> tuple[Any, dict[str, Any]]:
    """Compute the complete matched-contact and wrong-partner objective."""
    require_torch()
    weights.validate()
    required = {
        "pair_logits",
        "pair_probabilities",
        "pair_mask",
        "edge_supervision_mask",
        "epitope_supervision_mask",
        "paratope_supervision_mask",
        "epitope_logits",
        "paratope_logits",
        "epitope_probability",
        "paratope_probability",
        "pair_epitope_probability",
        "pair_paratope_probability",
        "complex_score",
    }
    missing = required - set(outputs)
    if missing:
        raise KeyError(f"Model output is missing keys: {sorted(missing)}")
    pair_mask = outputs["edge_supervision_mask"]
    epitope_mask = _validated_supervision_mask(
        outputs,
        "epitope_supervision_mask",
        antigen_mask,
    )
    paratope_mask = _validated_supervision_mask(
        outputs,
        "paratope_supervision_mask",
        antibody_mask,
    )
    if weights.focal_gamma > 0.0:
        edge_loss = masked_focal_loss_with_logits(
            outputs["pair_logits"], pair_targets, pair_mask, edge_positive_weight, weights.focal_gamma
        )
        epitope_bce = masked_focal_loss_with_logits(
            outputs["epitope_logits"],
            epitope_targets,
            epitope_mask,
            epitope_positive_weight,
            weights.focal_gamma
        )
        paratope_bce = masked_focal_loss_with_logits(
            outputs["paratope_logits"],
            paratope_targets,
            paratope_mask,
            paratope_positive_weight,
            weights.focal_gamma
        )
    else:
        edge_loss = masked_bce_with_logits(
            outputs["pair_logits"], pair_targets, pair_mask, edge_positive_weight
        )
        epitope_bce = masked_bce_with_logits(
            outputs["epitope_logits"],
            epitope_targets,
            epitope_mask,
            epitope_positive_weight,
        )
        paratope_bce = masked_bce_with_logits(
            outputs["paratope_logits"],
            paratope_targets,
            paratope_mask,
            paratope_positive_weight,
        )
    epitope_dice = soft_dice_loss(
        outputs["epitope_probability"], epitope_targets, epitope_mask
    )
    paratope_dice = soft_dice_loss(
        outputs["paratope_probability"], paratope_targets, paratope_mask
    )
    zero = outputs["complex_score"].new_zeros(())
    edge_count = (
        relative_count_loss(outputs["pair_probabilities"], pair_targets, pair_mask)
        if weights.edge_count > 0.0
        else zero
    )
    epitope_count = (
        relative_count_loss(
            outputs["epitope_probability"],
            epitope_targets,
            epitope_mask,
        )
        if weights.epitope_count > 0.0
        else zero
    )
    paratope_count = (
        relative_count_loss(
            outputs["paratope_probability"],
            paratope_targets,
            paratope_mask,
        )
        if weights.paratope_count > 0.0
        else zero
    )
    epitope_consistency = (
        masked_consistency_loss(
            outputs["epitope_probability"],
            outputs["pair_epitope_probability"],
            epitope_mask,
        )
        if weights.epitope_consistency > 0.0
        else zero
    )
    paratope_consistency = (
        masked_consistency_loss(
            outputs["paratope_probability"],
            outputs["pair_paratope_probability"],
            paratope_mask,
        )
        if weights.paratope_consistency > 0.0
        else zero
    )
    if negative_scores is None:
        rank_loss = outputs["complex_score"].new_zeros(())
    else:
        rank_loss = partner_ranking_loss(
            outputs["complex_score"], negative_scores, partner_temperature
        )

    # Margin separation loss on pair logits
    if weights.margin_separation > 0.0:
        margin_sep = masked_margin_separation_loss(
            outputs["pair_logits"], pair_targets, pair_mask, weights.margin_value
        )
    else:
        margin_sep = outputs["complex_score"].new_zeros(())

    components = {
        "edge": edge_loss,
        "margin_separation": margin_sep,
        "epitope_bce": epitope_bce,
        "paratope_bce": paratope_bce,
        "epitope_dice": epitope_dice,
        "paratope_dice": paratope_dice,
        "edge_count": edge_count,
        "epitope_count": epitope_count,
        "paratope_count": paratope_count,
        "epitope_consistency": epitope_consistency,
        "paratope_consistency": paratope_consistency,
        "partner_rank": rank_loss,
    }
    total = sum(getattr(weights, name) * value for name, value in components.items())
    return total, components
