"""Directly supervised contact prediction head for MIPE."""

from __future__ import annotations

from typing import Any, Mapping

import math

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError:
    torch = None
    nn = None
    F = None

_ModuleBase = nn.Module if nn is not None else object

def require_torch() -> None:
    """Fail with a focused message when operations are used without PyTorch."""
    if torch is None or nn is None or F is None:
        raise RuntimeError("PyTorch is required for MIPE contact operations")

class MIPEContactHead(_ModuleBase):
    """Predict pairwise contacts from cross-attended antibody and antigen representations."""

    def __init__(self, d_model: int, pair_dim: int) -> None:
        require_torch()
        super().__init__()
        if d_model <= 0 or pair_dim <= 0:
            raise ValueError("Dimensions must be positive integers")

        self.ab_norm = nn.LayerNorm(d_model)
        self.ag_norm = nn.LayerNorm(d_model)

        self.ab_query = nn.Linear(d_model, pair_dim, bias=False)
        self.ag_key = nn.Linear(d_model, pair_dim, bias=False)

        self.ab_key = nn.Linear(d_model, pair_dim, bias=False)
        self.ag_query = nn.Linear(d_model, pair_dim, bias=False)

        self.mixing_logits = nn.Parameter(torch.zeros(2))

        self.ab_bias = nn.Linear(d_model, 1)
        self.ag_bias = nn.Linear(d_model, 1)
        self.global_bias = nn.Parameter(torch.tensor(-4.0))

        self.scale = pair_dim**-0.5

    def components(self, antibody: Any, antigen: Any) -> dict[str, Any]:
        """Compute the decomposed contact components."""
        if antibody.ndim != 3 or antigen.ndim != 3:
            raise ValueError("Inputs must have shape [batch, length, channels]")

        ab = self.ab_norm(antibody)
        ag = self.ag_norm(antigen)

        ab_to_ag = torch.matmul(self.ab_query(ab), self.ag_key(ag).transpose(-2, -1))
        ag_to_ab = torch.matmul(self.ab_key(ab), self.ag_query(ag).transpose(-2, -1))

        weights = torch.softmax(self.mixing_logits, dim=0)
        interaction_logits = self.scale * (weights[0] * ab_to_ag + weights[1] * ag_to_ab)

        ab_bias_term = self.ab_bias(ab)
        ag_bias_term = self.ag_bias(ag).transpose(-2, -1)

        pair_logits = interaction_logits + ab_bias_term + ag_bias_term + self.global_bias

        return {
            "pair_logits": pair_logits,
            "interaction_logits": interaction_logits,
            "antibody_bias": ab_bias_term,
            "antigen_bias": ag_bias_term,
            "global_bias": self.global_bias,
        }

    def forward(self, antibody: Any, antigen: Any) -> Any:
        """Return contact logits of shape [batch, antibody_residue, antigen_residue]."""
        return self.components(antibody, antigen)["pair_logits"]


def focal_loss(
    logits: Any,
    targets: Any,
    mask: Any,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> Any:
    """Compute Focal Loss for extremely imbalanced contact prediction."""
    require_torch()
    if logits.shape != targets.shape or logits.shape != mask.shape:
        raise ValueError("Logits, targets, and mask must have identical shapes")
    if mask.dtype != torch.bool or not torch.any(mask):
        raise ValueError("Mask must contain at least one valid boolean entry")

    selected_logits = logits[mask]
    selected_targets = targets[mask].to(dtype=logits.dtype)

    bce = F.binary_cross_entropy_with_logits(
        selected_logits, selected_targets, reduction="none"
    )

    p = torch.sigmoid(selected_logits)
    p_t = p * selected_targets + (1 - p) * (1 - selected_targets)
    alpha_t = alpha * selected_targets + (1 - alpha) * (1 - selected_targets)

    focal_weight = alpha_t * (1 - p_t) ** gamma
    return (focal_weight * bce).mean()


def soft_dice_loss(probabilities: Any, targets: Any, mask: Any, epsilon: float = 1.0) -> Any:
    """Compute mean per-complex Dice loss over valid entries."""
    require_torch()
    if probabilities.shape != targets.shape or probabilities.shape != mask.shape:
        raise ValueError("Dice inputs must have identical shapes")

    losses = []
    for i in range(probabilities.shape[0]):
        valid = mask[i]
        if not torch.any(valid):
            continue
        pred = probabilities[i][valid]
        truth = targets[i][valid].to(dtype=pred.dtype)
        numerator = 2.0 * (pred * truth).sum() + epsilon
        denominator = pred.sum() + truth.sum() + epsilon
        losses.append(1.0 - numerator / denominator)

    if not losses:
        raise ValueError("Dice loss received no valid complexes")
    return torch.stack(losses).mean()


def mipe_contact_loss(
    logits: Any,
    targets: Any,
    mask: Any,
    focal_weight: float = 1.0,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    dice_weight: float = 0.0,
    consistency_weight: float = 0.0,
    epitope_targets: Any | None = None,
    paratope_targets: Any | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Combined loss for MIPE contact prediction."""
    require_torch()

    fl = focal_loss(logits, targets, mask, alpha=focal_alpha, gamma=focal_gamma)
    total_loss = focal_weight * fl
    components = {"focal_loss": fl}

    probs = torch.sigmoid(logits)

    if dice_weight > 0.0:
        dl = soft_dice_loss(probs, targets, mask)
        total_loss = total_loss + dice_weight * dl
        components["dice_loss"] = dl
    else:
        components["dice_loss"] = logits.new_zeros(())

    if consistency_weight > 0.0:
        if epitope_targets is None or paratope_targets is None:
            raise ValueError("Epitope and paratope targets required for consistency loss")
        
        valid_probs = probs.masked_fill(~mask, 0.0)
        
        # probs is [B, L_ab, L_ag]
        # margin paratope (max over ag): [B, L_ab]
        pred_paratope = valid_probs.max(dim=2).values
        # margin epitope (max over ab): [B, L_ag]
        pred_epitope = valid_probs.max(dim=1).values
        
        # supervision masks
        ab_mask = mask.any(dim=2)
        ag_mask = mask.any(dim=1)
        
        if not torch.any(ab_mask) or not torch.any(ag_mask):
             raise ValueError("Consistency masks are empty")

        cl_para = F.smooth_l1_loss(pred_paratope[ab_mask], paratope_targets[ab_mask].to(dtype=logits.dtype))
        cl_epi = F.smooth_l1_loss(pred_epitope[ag_mask], epitope_targets[ag_mask].to(dtype=logits.dtype))
        cl = cl_para + cl_epi

        total_loss = total_loss + consistency_weight * cl
        components["consistency_loss"] = cl
    else:
        components["consistency_loss"] = logits.new_zeros(())

    return total_loss, components


class MIPEContactEvaluator:
    """Evaluates pair-level contact predictions."""

    @staticmethod
    def _numpy() -> Any:
        try:
            import numpy as np
        except ModuleNotFoundError as error:
            raise RuntimeError("NumPy is required for evaluation") from error
        return np

    @classmethod
    def average_precision(cls, labels: Any, scores: Any) -> float | None:
        """Compute tie-aware non-interpolated AUPRC."""
        np = cls._numpy()
        labels = np.asarray(labels, dtype=np.uint8).reshape(-1)
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        if labels.size == 0:
            raise ValueError("Labels and scores must be non-empty")
        pos_total = int(labels.sum())
        if pos_total == 0:
            return None
        order = np.argsort(-scores, kind="stable")
        labels = labels[order]
        scores = scores[order]
        tp = 0
        fp = 0
        result = 0.0
        start = 0
        while start < len(labels):
            end = start + 1
            while end < len(labels) and scores[end] == scores[start]:
                end += 1
            group_pos = int(labels[start:end].sum())
            group_neg = end - start - group_pos
            tp += group_pos
            fp += group_neg
            if group_pos:
                precision = tp / (tp + fp)
                result += (group_pos / pos_total) * precision
            start = end
        return float(result)

    @classmethod
    def roc_auc(cls, labels: Any, scores: Any) -> float | None:
        """Compute tie-aware AUROC."""
        np = cls._numpy()
        labels = np.asarray(labels, dtype=np.uint8).reshape(-1)
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        if labels.size == 0:
            raise ValueError("Labels and scores must be non-empty")
        pos_total = int(labels.sum())
        neg_total = len(labels) - pos_total
        if pos_total == 0 or neg_total == 0:
            return None
        order = np.argsort(-scores, kind="stable")
        labels = labels[order]
        scores = scores[order]
        tp = 0
        fp = 0
        prev_tpr = 0.0
        prev_fpr = 0.0
        area = 0.0
        start = 0
        while start < len(labels):
            end = start + 1
            while end < len(labels) and scores[end] == scores[start]:
                end += 1
            group_pos = int(labels[start:end].sum())
            group_neg = end - start - group_pos
            tp += group_pos
            fp += group_neg
            tpr = tp / pos_total
            fpr = fp / neg_total
            area += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0
            prev_tpr = tpr
            prev_fpr = fpr
            start = end
        return float(area)

    @classmethod
    def evaluate(cls, labels: Any, scores: Any, mask: Any) -> dict[str, float | None]:
        """Compute contact metrics for a batch."""
        np = cls._numpy()
        if not torch.is_tensor(labels) or not torch.is_tensor(scores) or not torch.is_tensor(mask):
            labels = np.asarray(labels)
            scores = np.asarray(scores)
            mask = np.asarray(mask, dtype=np.bool_)
        else:
            labels = labels.detach().cpu().numpy()
            scores = scores.detach().cpu().numpy()
            mask = mask.detach().cpu().numpy()

        if labels.shape != scores.shape or labels.shape != mask.shape:
            raise ValueError("Shapes must match")

        valid_labels = labels[mask]
        valid_scores = scores[mask]
        
        if valid_labels.size == 0:
            raise ValueError("No valid items in mask")

        metrics: dict[str, float | None] = {}
        metrics["auprc"] = cls.average_precision(valid_labels, valid_scores)
        metrics["auroc"] = cls.roc_auc(valid_labels, valid_scores)

        # Ranking metrics P@1, R@1, MRR
        # Treat each complex in the batch independently
        batch_size = labels.shape[0]
        p_at_1 = []
        r_at_1 = []
        mrr = []
        ious = []
        for i in range(batch_size):
            m = mask[i]
            if not np.any(m):
                continue
            l = valid_labels = labels[i][m]
            s = valid_scores = scores[i][m]
            pos_total = int(l.sum())
            if pos_total == 0:
                continue
            
            order = np.argsort(-s, kind="stable")
            l_sorted = l[order]
            s_sorted = s[order]

            # p@1
            top_score = s_sorted[0]
            ties = s_sorted == top_score
            tie_count = ties.sum()
            tp_at_boundary = l_sorted[ties].sum()
            p1 = float(tp_at_boundary / tie_count)
            p_at_1.append(p1)
            
            r1 = float(tp_at_boundary / pos_total)
            r_at_1.append(r1)

            # MRR
            pos_indices = np.where(l_sorted == 1)[0]
            first_pos_idx = pos_indices[0]
            # Handle ties for the first positive
            first_pos_score = s_sorted[first_pos_idx]
            # all candidates with >= first_pos_score
            better_or_equal = np.sum(s_sorted >= first_pos_score)
            strictly_better = np.sum(s_sorted > first_pos_score)
            # Rank is expectation over ties
            rank = strictly_better + 1.0 + (better_or_equal - strictly_better - 1.0) / 2.0
            mrr.append(1.0 / rank)

            # optimal threshold iou (assuming top POS_TOTAL predictions)
            boundary_score = s_sorted[pos_total - 1]
            strictly_above = s_sorted > boundary_score
            at_boundary = s_sorted == boundary_score
            
            tp = l_sorted[strictly_above].sum()
            boundary_budget = pos_total - strictly_above.sum()
            boundary_tp_rate = l_sorted[at_boundary].mean()
            tp += boundary_budget * boundary_tp_rate
            
            union = pos_total + pos_total - tp
            ious.append(float(tp / union) if union > 0 else 0.0)

        metrics["precision_at_1"] = float(np.mean(p_at_1)) if p_at_1 else None
        metrics["recall_at_1"] = float(np.mean(r_at_1)) if r_at_1 else None
        metrics["mrr"] = float(np.mean(mrr)) if mrr else None
        metrics["contact_iou"] = float(np.mean(ious)) if ious else None

        return metrics
