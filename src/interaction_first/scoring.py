"""Length-normalized scoring utilities for sparse residue-pair predictions."""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError:
    torch = None
    F = None


def require_torch() -> None:
    """Fail with a focused message when scoring code is used without PyTorch."""
    if torch is None or F is None:
        raise RuntimeError("PyTorch is required for interaction scoring")


def _validate_pair_inputs(pair_values: Any, pair_mask: Any) -> None:
    """Validate a padded batch of residue-pair values and its mask."""
    require_torch()
    if not torch.is_tensor(pair_values) or not torch.is_tensor(pair_mask):
        raise TypeError("pair_values and pair_mask must be PyTorch tensors")
    if pair_values.ndim != 3:
        raise ValueError("pair_values must have shape [batch, antigen, antibody]")
    if tuple(pair_values.shape) != tuple(pair_mask.shape):
        raise ValueError("pair_mask must have the same shape as pair_values")
    if pair_mask.dtype != torch.bool:
        raise ValueError("pair_mask must be a boolean tensor")
    if pair_values.device != pair_mask.device:
        raise ValueError("pair_values and pair_mask must be on the same device")
    if not torch.is_floating_point(pair_values):
        raise ValueError("pair_values must use a floating-point dtype")
    if pair_values.shape[0] == 0 or pair_values.shape[1] == 0 or pair_values.shape[2] == 0:
        raise ValueError("pair_values cannot contain an empty tensor dimension")
    valid_counts = pair_mask.reshape(pair_mask.shape[0], -1).sum(dim=1)
    if bool(torch.any(valid_counts == 0)):
        raise ValueError("Every batch item must contain at least one valid pair")
    if not bool(torch.all(torch.isfinite(pair_values[pair_mask]))):
        raise ValueError("Valid pair values must be finite")


def _positive_finite_scalar(value: Any, name: str) -> float:
    """Convert a scalar hyperparameter while rejecting invalid values."""
    try:
        converted = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a positive finite scalar")
    if not math.isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{name} must be a positive finite scalar")
    return converted


def _positive_integer(value: Any, name: str) -> int:
    """Validate a strictly positive integer hyperparameter."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _normalized_axis_bias(
    bias: Any,
    pair_values: Any,
    axis: str,
) -> Any:
    """Validate and reshape one residue-wise unary bias for broadcasting."""
    if not torch.is_tensor(bias):
        raise TypeError(f"{axis}_bias must be a PyTorch tensor")
    if bias.device != pair_values.device:
        raise ValueError(f"{axis}_bias must be on the same device as pair_values")
    if not torch.is_floating_point(bias) or not bool(torch.all(torch.isfinite(bias))):
        raise ValueError(f"{axis}_bias must contain finite floating-point values")

    batch_size, antigen_length, antibody_length = pair_values.shape
    if axis == "antigen":
        matrix_shape = (batch_size, antigen_length)
        broadcast_shape = (batch_size, antigen_length, 1)
    elif axis == "antibody":
        matrix_shape = (batch_size, antibody_length)
        broadcast_shape = (batch_size, 1, antibody_length)
    else:
        raise ValueError("axis must be 'antigen' or 'antibody'")

    if tuple(bias.shape) == matrix_shape:
        bias = bias.reshape(broadcast_shape)
    elif tuple(bias.shape) != broadcast_shape:
        raise ValueError(
            f"{axis}_bias must have shape {matrix_shape} or {broadcast_shape}"
        )
    return bias.to(dtype=pair_values.dtype)


def _normalized_global_bias(global_bias: Any, pair_values: Any) -> Any:
    """Validate and reshape a scalar or per-complex global bias."""
    if not torch.is_tensor(global_bias):
        raise TypeError("global_bias must be a PyTorch tensor")
    if global_bias.device != pair_values.device:
        raise ValueError("global_bias must be on the same device as pair_values")
    if not torch.is_floating_point(global_bias) or not bool(
        torch.all(torch.isfinite(global_bias))
    ):
        raise ValueError("global_bias must contain finite floating-point values")

    batch_size = int(pair_values.shape[0])
    if global_bias.ndim == 0:
        normalized = global_bias.reshape(1, 1, 1)
    elif tuple(global_bias.shape) == (batch_size,):
        normalized = global_bias.reshape(batch_size, 1, 1)
    elif tuple(global_bias.shape) == (batch_size, 1, 1):
        normalized = global_bias
    else:
        raise ValueError(
            "global_bias must be scalar or have shape [batch] or [batch, 1, 1]"
        )
    return normalized.to(dtype=pair_values.dtype)


def remove_unary_biases(
    pair_logits: Any,
    pair_mask: Any,
    antigen_bias: Optional[Any] = None,
    antibody_bias: Optional[Any] = None,
    global_bias: Optional[Any] = None,
) -> Any:
    """Subtract explicit decoder biases and zero all masked pair positions.

    Antigen biases may have shape ``[B, A]`` or ``[B, A, 1]``. Antibody
    biases may have shape ``[B, R]`` or ``[B, 1, R]``. A global bias may be
    scalar, per-complex, or explicitly broadcast with shape ``[B, 1, 1]``.
    """
    _validate_pair_inputs(pair_logits, pair_mask)
    residual = pair_logits
    if antigen_bias is not None:
        residual = residual - _normalized_axis_bias(
            antigen_bias, pair_logits, "antigen"
        )
    if antibody_bias is not None:
        residual = residual - _normalized_axis_bias(
            antibody_bias, pair_logits, "antibody"
        )
    if global_bias is not None:
        residual = residual - _normalized_global_bias(global_bias, pair_logits)
    return torch.where(pair_mask, residual, torch.zeros_like(residual))


def masked_double_center(
    pair_values: Any,
    pair_mask: Any,
    iterations: int = 16,
) -> Any:
    """Remove masked row and column main effects from each pair matrix.

    Alternating masked projections are exact after one pass for rectangular
    padding masks. Repeated passes also support sparse masks with internal holes.
    Rows and columns without a valid pair remain zero.
    """
    _validate_pair_inputs(pair_values, pair_mask)
    iterations = _positive_integer(iterations, "iterations")
    weights = pair_mask.to(dtype=pair_values.dtype)
    residual = torch.where(pair_mask, pair_values, torch.zeros_like(pair_values))

    row_counts = weights.sum(dim=2, keepdim=True)
    column_counts = weights.sum(dim=1, keepdim=True)
    for _ in range(iterations):
        row_means = residual.sum(dim=2, keepdim=True) / row_counts.clamp_min(1.0)
        residual = torch.where(pair_mask, residual - row_means, torch.zeros_like(residual))
        column_means = residual.sum(dim=1, keepdim=True) / column_counts.clamp_min(1.0)
        residual = torch.where(
            pair_mask, residual - column_means, torch.zeros_like(residual)
        )
    return residual


def interaction_residual(
    pair_logits: Any,
    pair_mask: Any,
    antigen_bias: Optional[Any] = None,
    antibody_bias: Optional[Any] = None,
    global_bias: Optional[Any] = None,
    center: bool = True,
    centering_iterations: int = 16,
) -> Any:
    """Isolate pairwise signal by explicit bias removal and optional centering."""
    if not isinstance(center, bool):
        raise ValueError("center must be boolean")
    residual = remove_unary_biases(
        pair_logits,
        pair_mask,
        antigen_bias=antigen_bias,
        antibody_bias=antibody_bias,
        global_bias=global_bias,
    )
    if center:
        residual = masked_double_center(
            residual, pair_mask, iterations=centering_iterations
        )
    return residual


def _masked_logmeanexp_along(
    values: Any,
    mask: Any,
    dimension: int,
    temperature: float,
) -> Any:
    """Compute normalized log-mean-exp along one masked dimension."""
    counts = mask.sum(dim=dimension)
    scaled = values / temperature
    masked_scaled = scaled.masked_fill(~mask, -torch.inf)
    log_total = torch.logsumexp(masked_scaled, dim=dimension)
    safe_counts = counts.clamp_min(1).to(dtype=values.dtype)
    pooled = temperature * (log_total - torch.log(safe_counts))
    return torch.where(counts > 0, pooled, torch.zeros_like(pooled))


def masked_logmeanexp_score(
    pair_values: Any,
    pair_mask: Any,
    temperature: float = 1.0,
) -> Any:
    """Return a stable length-normalized log-mean-exp score per complex."""
    _validate_pair_inputs(pair_values, pair_mask)
    temperature = _positive_finite_scalar(temperature, "temperature")
    flattened_values = pair_values.reshape(pair_values.shape[0], -1)
    flattened_mask = pair_mask.reshape(pair_mask.shape[0], -1)
    return _masked_logmeanexp_along(
        flattened_values, flattened_mask, dimension=1, temperature=temperature
    )


def masked_top_fraction_score(
    pair_values: Any,
    pair_mask: Any,
    fraction: float = 0.05,
    min_count: int = 1,
) -> Any:
    """Average the top fraction of valid pairs without a fixed-k length bias."""
    _validate_pair_inputs(pair_values, pair_mask)
    fraction = _positive_finite_scalar(fraction, "fraction")
    if fraction > 1.0:
        raise ValueError("fraction must not exceed 1.0")
    min_count = _positive_integer(min_count, "min_count")

    flattened_values = pair_values.reshape(pair_values.shape[0], -1)
    flattened_mask = pair_mask.reshape(pair_mask.shape[0], -1)
    scores = []
    for batch_index in range(int(pair_values.shape[0])):
        valid = flattened_values[batch_index][flattened_mask[batch_index]]
        count = int(valid.numel())
        selected_count = min(count, max(min_count, int(math.ceil(fraction * count))))
        selected = torch.topk(valid, k=selected_count, sorted=False).values
        scores.append(selected.mean())
    return torch.stack(scores)


def _masked_topk_probability_along(
    probabilities: Any,
    mask: Any,
    dimension: int,
    top_k: int,
) -> Any:
    """Pool the top valid probabilities along one residue-pair axis."""
    k = min(top_k, int(probabilities.shape[dimension]))
    masked = probabilities.masked_fill(~mask, -torch.inf)
    selected, indices = torch.topk(masked, k=k, dim=dimension, sorted=False)
    selected_mask = torch.gather(mask, dimension, indices)
    selected = torch.where(selected_mask, selected, torch.zeros_like(selected))
    counts = selected_mask.sum(dim=dimension)
    pooled = selected.sum(dim=dimension) / counts.clamp_min(1).to(
        dtype=probabilities.dtype
    )
    return torch.where(counts > 0, pooled, torch.zeros_like(pooled))


def _masked_noisy_or_along(
    pair_logits: Any,
    pair_mask: Any,
    dimension: int,
) -> Any:
    """Compute a stable probability that at least one valid pair is positive."""
    log_not_contact = F.logsigmoid(-pair_logits)
    log_none = torch.where(
        pair_mask, log_not_contact, torch.zeros_like(log_not_contact)
    ).sum(dim=dimension)
    counts = pair_mask.sum(dim=dimension)
    marginal = -torch.expm1(log_none)
    return torch.where(counts > 0, marginal, torch.zeros_like(marginal))


def pool_pair_marginals(
    pair_logits: Any,
    pair_mask: Any,
    pooling: str = "logmeanexp",
    top_k: int = 3,
    temperature: float = 1.0,
) -> Tuple[Any, Any]:
    """Pool pair logits into antigen and antibody marginal probabilities.

    ``noisy_or`` models the probability of at least one contact. ``topk``
    averages the strongest contact probabilities. ``logmeanexp`` provides a
    smooth, length-normalized maximum in logit space before applying sigmoid.
    """
    _validate_pair_inputs(pair_logits, pair_mask)
    if pooling not in {"noisy_or", "topk", "logmeanexp"}:
        raise ValueError("pooling must be 'noisy_or', 'topk', or 'logmeanexp'")

    if pooling == "noisy_or":
        antigen = _masked_noisy_or_along(pair_logits, pair_mask, dimension=2)
        antibody = _masked_noisy_or_along(pair_logits, pair_mask, dimension=1)
    elif pooling == "topk":
        top_k = _positive_integer(top_k, "top_k")
        probabilities = torch.sigmoid(pair_logits)
        antigen = _masked_topk_probability_along(
            probabilities, pair_mask, dimension=2, top_k=top_k
        )
        antibody = _masked_topk_probability_along(
            probabilities, pair_mask, dimension=1, top_k=top_k
        )
    else:
        temperature = _positive_finite_scalar(temperature, "temperature")
        antigen_logits = _masked_logmeanexp_along(
            pair_logits, pair_mask, dimension=2, temperature=temperature
        )
        antibody_logits = _masked_logmeanexp_along(
            pair_logits, pair_mask, dimension=1, temperature=temperature
        )
        antigen_valid = pair_mask.any(dim=2)
        antibody_valid = pair_mask.any(dim=1)
        antigen = torch.where(
            antigen_valid, torch.sigmoid(antigen_logits), torch.zeros_like(antigen_logits)
        )
        antibody = torch.where(
            antibody_valid,
            torch.sigmoid(antibody_logits),
            torch.zeros_like(antibody_logits),
        )
    return antigen, antibody


def normalized_partner_score(
    pair_logits: Any,
    pair_mask: Any,
    pooling: str = "logmeanexp",
    temperature: float = 1.0,
    top_fraction: float = 0.05,
    min_count: int = 1,
    antigen_bias: Optional[Any] = None,
    antibody_bias: Optional[Any] = None,
    global_bias: Optional[Any] = None,
    center: bool = True,
    centering_iterations: int = 16,
) -> Any:
    """Score partner compatibility from isolated, length-normalized interactions."""
    residual = interaction_residual(
        pair_logits,
        pair_mask,
        antigen_bias=antigen_bias,
        antibody_bias=antibody_bias,
        global_bias=global_bias,
        center=center,
        centering_iterations=centering_iterations,
    )
    if pooling == "logmeanexp":
        return masked_logmeanexp_score(
            residual, pair_mask, temperature=temperature
        )
    if pooling == "top_fraction":
        return masked_top_fraction_score(
            residual, pair_mask, fraction=top_fraction, min_count=min_count
        )
    raise ValueError("pooling must be 'logmeanexp' or 'top_fraction'")


__all__ = [
    "interaction_residual",
    "masked_double_center",
    "masked_logmeanexp_score",
    "masked_top_fraction_score",
    "normalized_partner_score",
    "pool_pair_marginals",
    "remove_unary_biases",
]
