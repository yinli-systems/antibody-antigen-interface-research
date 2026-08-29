"""Identifiable partner-specificity gauge for bipartite contact logit maps.

A bipartite contact logit map admits an additive decomposition

``x[i, j] = g + a[i] + b[j] + w[i, j]``

where ``a`` and ``b`` are single-chain propensities and ``w`` is the only term
that depends jointly on an antibody residue and an antigen residue.  That
decomposition is **not unique**: any row/column profile can be shifted between
``w`` and the unary terms without changing ``x``.  Every quantity derived from
an unconstrained ``w`` -- including a partner-specificity score -- is therefore
gauge dependent and not comparable across models.

This module fixes the gauge with the standard two-way additive-effects
(ANOVA) projection: ``w`` is constrained to have zero masked row means and zero
masked column means.  Under that constraint the decomposition is unique, the
reconstruction is exact, and for a rectangular mask the three components are
mutually orthogonal, so the observed variance of ``x`` splits exactly into a
partner-agnostic part and a partner-specific part.

Two consequences matter for FrameContact:

* The gauge is applied to the *final* contact logits, so it is head agnostic.
  A binary contact head and a distogram head produce comparable specificity
  channels even though the distogram collapses bins through a nonlinear
  ``logsumexp`` and has no additive interaction component of its own.
* The projection never changes ``x``.  Enabling the gauge re-attributes
  evidence between channels; it cannot alter the contact prediction itself.
"""

from __future__ import annotations

from typing import Any

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional import boundary
    torch = None


def _require_torch() -> Any:
    if torch is None:
        raise RuntimeError("PyTorch is required for the contact specificity gauge")
    return torch


def _validate_pair_map(values: Any, mask: Any) -> None:
    torch_module = _require_torch()
    if not torch_module.is_tensor(values) or values.ndim != 3:
        raise ValueError("values must have shape [batch, antibody, antigen]")
    if (
        not torch_module.is_tensor(mask)
        or mask.dtype != torch_module.bool
        or tuple(mask.shape) != tuple(values.shape)
    ):
        raise ValueError("mask must be a boolean tensor aligned with values")


def masked_two_way_decomposition(
    values: Any,
    mask: Any,
    *,
    iterations: int = 1,
) -> dict[str, Any]:
    """Split a masked bipartite logit map into identifiable additive channels.

    The returned components satisfy, exactly and for any mask,

    ``values == grand + antibody_effect + antigen_effect + interaction``

    on every observed entry.  ``interaction`` is the gauge-fixed
    partner-specific channel: for a rectangular mask its masked row means and
    column means are exactly zero after a single pass.  Ragged masks are
    refined by alternating projection; extra ``iterations`` tighten the
    centering without ever breaking the reconstruction identity.
    """

    torch_module = _require_torch()
    _validate_pair_map(values, mask)
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        raise ValueError("iterations must be a positive integer")

    weights = mask.to(values.dtype)
    observed = values * weights
    row_count = weights.sum(dim=2)
    column_count = weights.sum(dim=1)
    total_count = weights.sum(dim=(1, 2))
    if not bool((total_count > 0).all()):
        raise ValueError("every complex must contain at least one observed pair")

    row_valid = row_count > 0
    column_valid = column_count > 0

    grand = observed.sum(dim=(1, 2)) / total_count.clamp_min(1.0)
    antibody_effect = torch_module.zeros_like(row_count)
    antigen_effect = torch_module.zeros_like(column_count)
    interaction = (values - grand[:, None, None]) * weights

    for _ in range(iterations):
        row_mean = interaction.sum(dim=2) / row_count.clamp_min(1.0)
        row_mean = row_mean * row_valid
        antibody_effect = antibody_effect + row_mean
        interaction = (interaction - row_mean[:, :, None]) * weights

        column_mean = interaction.sum(dim=1) / column_count.clamp_min(1.0)
        column_mean = column_mean * column_valid
        antigen_effect = antigen_effect + column_mean
        interaction = (interaction - column_mean[:, None, :]) * weights

    return {
        "grand": grand,
        "antibody_effect": antibody_effect,
        "antigen_effect": antigen_effect,
        "interaction": interaction,
        "mask": mask,
    }


def reconstruct_from_decomposition(decomposition: dict[str, Any]) -> Any:
    """Rebuild the masked logit map from its gauge-fixed channels."""
    _require_torch()
    mask = decomposition["mask"]
    weights = mask.to(decomposition["interaction"].dtype)
    total = (
        decomposition["grand"][:, None, None]
        + decomposition["antibody_effect"][:, :, None]
        + decomposition["antigen_effect"][:, None, :]
        + decomposition["interaction"]
    )
    return total * weights


def gauge_variance_attribution(decomposition: dict[str, Any]) -> dict[str, Any]:
    """Report how much observed contact variance is genuinely partner specific.

    For a rectangular mask -- including the padded-rectangular masks produced by
    an antibody-mask/antigen-mask outer product -- the three channels are
    orthogonal, so the sums of squares add up exactly and the returned shares
    form a partition of one.  ``partner_specific_share`` is the fraction of
    contact-map variance that no pair of single-chain propensity profiles could
    have produced.

    Under a genuinely ragged mask (for example a CDR-by-patch candidate mask)
    the channels are only approximately orthogonal, so the channel sums of
    squares need not reproduce the observed variance.  ``partition_gap``
    reports that discrepancy as a fraction of the observed variance; treat the
    shares as exact only while it is negligible.
    """

    torch_module = _require_torch()
    mask = decomposition["mask"]
    weights = mask.to(decomposition["interaction"].dtype)
    row_count = weights.sum(dim=2)
    column_count = weights.sum(dim=1)
    total_count = weights.sum(dim=(1, 2)).clamp_min(1.0)

    antibody_sum_squares = (decomposition["antibody_effect"].pow(2) * row_count).sum(dim=1)
    antigen_sum_squares = (decomposition["antigen_effect"].pow(2) * column_count).sum(dim=1)
    interaction_sum_squares = (decomposition["interaction"].pow(2) * weights).sum(dim=(1, 2))
    additive_sum_squares = antibody_sum_squares + antigen_sum_squares
    total_sum_squares = additive_sum_squares + interaction_sum_squares
    tiny = torch_module.finfo(weights.dtype).tiny
    denominator = total_sum_squares.clamp_min(tiny)

    observed_sum_squares = (
        reconstruct_from_decomposition(decomposition) - decomposition["grand"][:, None, None]
    ).mul(weights).pow(2).sum(dim=(1, 2))
    partition_gap = (total_sum_squares - observed_sum_squares).abs() / (
        observed_sum_squares.clamp_min(tiny)
    )

    return {
        "antibody_propensity_share": antibody_sum_squares / denominator,
        "antigen_propensity_share": antigen_sum_squares / denominator,
        "partner_specific_share": interaction_sum_squares / denominator,
        "antibody_propensity_variance": antibody_sum_squares / total_count,
        "antigen_propensity_variance": antigen_sum_squares / total_count,
        "partner_specific_variance": interaction_sum_squares / total_count,
        "total_variance": total_sum_squares / total_count,
        "observed_variance": observed_sum_squares / total_count,
        "partition_gap": partition_gap,
    }


def centering_residual(decomposition: dict[str, Any]) -> dict[str, Any]:
    """Largest surviving masked row/column mean of the partner-specific channel.

    This is the numerical check that the gauge actually holds.  It is exactly
    zero for rectangular masks and is reported so ragged-mask runs cannot
    silently claim an identifiable decomposition they do not have.
    """

    torch_module = _require_torch()
    mask = decomposition["mask"]
    weights = mask.to(decomposition["interaction"].dtype)
    interaction = decomposition["interaction"]
    row_count = weights.sum(dim=2)
    column_count = weights.sum(dim=1)
    row_mean = interaction.sum(dim=2) / row_count.clamp_min(1.0)
    column_mean = interaction.sum(dim=1) / column_count.clamp_min(1.0)
    row_mean = row_mean * (row_count > 0)
    column_mean = column_mean * (column_count > 0)
    return {
        "maximum_absolute_row_mean": row_mean.abs().amax(dim=1),
        "maximum_absolute_column_mean": column_mean.abs().amax(dim=1),
        "maximum_absolute_mean": torch_module.maximum(
            row_mean.abs().amax(dim=1), column_mean.abs().amax(dim=1)
        ),
    }


def gauge_fixed_pair_components(
    contact_logits: Any,
    pair_mask: Any,
    *,
    iterations: int = 1,
) -> dict[str, Any]:
    """Full gauge report for one contact map: channels, shares, and residual."""
    decomposition = masked_two_way_decomposition(
        contact_logits, pair_mask, iterations=iterations
    )
    attribution = gauge_variance_attribution(decomposition)
    residual = centering_residual(decomposition)
    return {
        "grand": decomposition["grand"],
        "antibody_effect": decomposition["antibody_effect"],
        "antigen_effect": decomposition["antigen_effect"],
        "interaction": decomposition["interaction"],
        "mask": pair_mask,
        "attribution": attribution,
        "centering_residual": residual,
    }


__all__ = [
    "centering_residual",
    "gauge_fixed_pair_components",
    "gauge_variance_attribution",
    "masked_two_way_decomposition",
    "reconstruct_from_decomposition",
]
