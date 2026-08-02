"""Contract tests for the identifiable partner-specificity gauge."""

from __future__ import annotations

import importlib.util

import pytest


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="PyTorch is unavailable",
)


def _padded_mask(lengths):
    import torch

    antibody = max(length[0] for length in lengths)
    antigen = max(length[1] for length in lengths)
    mask = torch.zeros(len(lengths), antibody, antigen, dtype=torch.bool)
    for index, (antibody_length, antigen_length) in enumerate(lengths):
        mask[index, :antibody_length, :antigen_length] = True
    return mask


def test_decomposition_reconstructs_the_masked_map_exactly() -> None:
    import torch

    from interaction_first.contact_gauge import (
        masked_two_way_decomposition,
        reconstruct_from_decomposition,
    )

    torch.manual_seed(20260802)
    mask = _padded_mask([(7, 11), (5, 9), (3, 4)])
    values = torch.randn(3, 7, 11, dtype=torch.float64) * 3.0 - 1.0

    decomposition = masked_two_way_decomposition(values, mask)
    rebuilt = reconstruct_from_decomposition(decomposition)

    assert torch.allclose(rebuilt, values * mask, atol=1e-12)


def test_padded_rectangular_masks_are_centered_in_a_single_pass() -> None:
    import torch

    from interaction_first.contact_gauge import (
        centering_residual,
        masked_two_way_decomposition,
    )

    torch.manual_seed(20260802)
    mask = _padded_mask([(6, 8), (4, 5)])
    values = torch.randn(2, 6, 8, dtype=torch.float64)

    residual = centering_residual(masked_two_way_decomposition(values, mask))

    assert float(residual["maximum_absolute_mean"].max()) < 1e-12


def test_additively_separable_maps_carry_no_partner_specific_signal() -> None:
    """A contact map that is only propensity times propensity must read zero.

    This is the degenerate solution a partner-agnostic model converges to, and
    the whole point of the gauge is that it is detectable.
    """

    import torch

    from interaction_first.contact_gauge import gauge_fixed_pair_components

    torch.manual_seed(20260802)
    mask = _padded_mask([(6, 8), (4, 5)])
    antibody = torch.randn(2, 6, 1, dtype=torch.float64)
    antigen = torch.randn(2, 1, 8, dtype=torch.float64)
    separable = (antibody + antigen).expand(2, 6, 8).contiguous()

    report = gauge_fixed_pair_components(separable, mask)

    assert float(report["interaction"].abs().max()) < 1e-10
    assert float(report["attribution"]["partner_specific_share"].max()) < 1e-10


def test_variance_shares_partition_one() -> None:
    import torch

    from interaction_first.contact_gauge import gauge_fixed_pair_components

    torch.manual_seed(20260802)
    mask = _padded_mask([(7, 11), (5, 9)])
    values = torch.randn(2, 7, 11, dtype=torch.float64) * 2.0

    attribution = gauge_fixed_pair_components(values, mask)["attribution"]
    total = (
        attribution["antibody_propensity_share"]
        + attribution["antigen_propensity_share"]
        + attribution["partner_specific_share"]
    )

    assert torch.allclose(total, torch.ones_like(total), atol=1e-12)


def test_partition_gap_is_zero_for_padded_masks_and_exposed_for_ragged_ones() -> None:
    """The shares only partition observed variance while the channels stay orthogonal.

    Padded-rectangular masks -- the production case, an antibody/antigen mask
    outer product -- keep the channels orthogonal, so the reported channel
    variances reproduce the observed variance exactly.  A ragged candidate mask
    refined by alternating projection breaks that orthogonality, and the gap
    must be reported rather than silently folded into the shares.
    """

    import torch

    from interaction_first.contact_gauge import gauge_fixed_pair_components

    torch.manual_seed(20260802)
    values = torch.randn(1, 7, 7, dtype=torch.float64)

    padded = _padded_mask([(5, 6)])
    padded_gap = gauge_fixed_pair_components(values[:, :5, :6], padded)["attribution"][
        "partition_gap"
    ]
    assert float(padded_gap.max()) < 1e-12

    ragged = torch.rand(1, 7, 7) > 0.25
    refined_gap = gauge_fixed_pair_components(values, ragged, iterations=32)["attribution"][
        "partition_gap"
    ]
    assert float(refined_gap.max()) > 1e-6


def test_gauge_is_invariant_to_moving_rank_one_mass_into_the_interaction() -> None:
    """The gauge removes the ambiguity that makes raw interaction terms unusable.

    Shifting an arbitrary row/column profile between the unary and interaction
    channels leaves the contact map unchanged, so any honest specificity
    channel must be unchanged too.
    """

    import torch

    from interaction_first.contact_gauge import masked_two_way_decomposition

    torch.manual_seed(20260802)
    mask = _padded_mask([(6, 8), (5, 7)])
    values = torch.randn(2, 6, 8, dtype=torch.float64)
    drift = torch.randn(2, 6, 1, dtype=torch.float64) + torch.randn(
        2, 1, 8, dtype=torch.float64
    )

    baseline = masked_two_way_decomposition(values, mask)["interaction"]
    shifted = masked_two_way_decomposition(values + drift * mask, mask)["interaction"]

    assert not torch.allclose(values, values + drift * mask)
    assert torch.allclose(baseline, shifted, atol=1e-10)


def test_ragged_masks_converge_under_alternating_projection() -> None:
    import torch

    from interaction_first.contact_gauge import (
        centering_residual,
        masked_two_way_decomposition,
        reconstruct_from_decomposition,
    )

    torch.manual_seed(20260802)
    values = torch.randn(1, 6, 6, dtype=torch.float64)
    mask = torch.ones(1, 6, 6, dtype=torch.bool)
    mask[0, 0, 5] = False
    mask[0, 3, 1] = False
    mask[0, 4, 4] = False

    single = centering_residual(masked_two_way_decomposition(values, mask))
    refined_decomposition = masked_two_way_decomposition(values, mask, iterations=64)
    refined = centering_residual(refined_decomposition)

    assert float(single["maximum_absolute_mean"].max()) > 1e-9
    assert float(refined["maximum_absolute_mean"].max()) < 1e-9
    assert torch.allclose(
        reconstruct_from_decomposition(refined_decomposition), values * mask, atol=1e-12
    )


def test_decomposition_rejects_misshaped_input() -> None:
    import torch

    from interaction_first.contact_gauge import masked_two_way_decomposition

    values = torch.randn(2, 4, 5)
    mask = torch.ones(2, 4, 5, dtype=torch.bool)

    with pytest.raises(ValueError):
        masked_two_way_decomposition(values, mask.float())
    with pytest.raises(ValueError):
        masked_two_way_decomposition(values, torch.ones(2, 4, 6, dtype=torch.bool))
    with pytest.raises(ValueError):
        masked_two_way_decomposition(values, mask, iterations=0)
    with pytest.raises(ValueError):
        masked_two_way_decomposition(values, torch.zeros_like(mask))
