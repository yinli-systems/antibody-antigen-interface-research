"""Focused tests for architecture-level PAIRScale unary controls."""

from __future__ import annotations

from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from interaction_first.pairscale_overfit import PairscaleOverfitConfig
from interaction_first.pairscale_unary_controls import (
    UnaryResidueContactModel,
    build_unary_control_model,
)


def _inputs() -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(811)
    antibody = torch.randn((2, 4, 7), generator=generator)
    antigen = torch.randn((2, 5, 11), generator=generator)
    antibody_mask = torch.tensor(
        [[True, True, True, True], [True, True, True, False]]
    )
    antigen_mask = torch.tensor(
        [[True, True, True, True, True], [True, True, True, True, False]]
    )
    pair_mask = antibody_mask.unsqueeze(2) & antigen_mask.unsqueeze(1)
    return antibody, antigen, antibody_mask, antigen_mask, pair_mask


@pytest.mark.parametrize(
    ("side", "forbidden_index"),
    (("antibody_only", 1), ("antigen_only", 0)),
)
def test_forbidden_partner_values_are_bitwise_invariant(
    side: str,
    forbidden_index: int,
) -> None:
    torch.manual_seed(71)
    model = UnaryResidueContactModel(
        control_side=side,
        antibody_input_dim=7,
        antigen_input_dim=11,
    ).eval()
    inputs = list(_inputs())
    original = model(*inputs, enabled_heads={"contact", "distance"})
    corrupted = list(inputs)
    corrupted[forbidden_index] = torch.full_like(
        corrupted[forbidden_index],
        float("nan"),
    )
    repeated = model(*corrupted, enabled_heads={"contact", "distance"})
    assert torch.equal(original["contact_logits"], repeated["contact_logits"])
    assert torch.equal(
        original["contact_probabilities"],
        repeated["contact_probabilities"],
    )
    assert repeated["forbidden_partner_values_read"] is False


@pytest.mark.parametrize(
    ("side", "forbidden_index"),
    (("antibody_only", 1), ("antigen_only", 0)),
)
def test_forbidden_partner_has_no_gradient_path(
    side: str,
    forbidden_index: int,
) -> None:
    torch.manual_seed(72)
    model = UnaryResidueContactModel(
        control_side=side,
        antibody_input_dim=7,
        antigen_input_dim=11,
    )
    inputs = list(_inputs())
    inputs[0] = inputs[0].requires_grad_()
    inputs[1] = inputs[1].requires_grad_()
    output = model(*inputs, enabled_heads={"contact", "distance"})
    output["contact_logits"][output["candidate_pair_mask"]].sum().backward()
    visible_index = 1 - forbidden_index
    assert inputs[forbidden_index].grad is None
    assert inputs[visible_index].grad is not None
    assert bool(torch.isfinite(inputs[visible_index].grad).all())
    assert float(inputs[visible_index].grad.abs().sum()) > 0.0


def test_controls_are_strictly_row_only_or_column_only() -> None:
    inputs = _inputs()
    for side in ("antibody_only", "antigen_only"):
        torch.manual_seed(73)
        output = UnaryResidueContactModel(
            control_side=side,
            antibody_input_dim=7,
            antigen_input_dim=11,
        )(*inputs, enabled_heads={"contact", "distance"})
        logits = output["contact_logits"]
        mask = output["candidate_pair_mask"]
        if side == "antibody_only":
            for batch_index in range(2):
                for antibody_index in range(4):
                    values = logits[batch_index, antibody_index][
                        mask[batch_index, antibody_index]
                    ]
                    if values.numel():
                        assert torch.equal(values, values[0].expand_as(values))
        else:
            for batch_index in range(2):
                for antigen_index in range(5):
                    values = logits[batch_index, :, antigen_index][
                        mask[batch_index, :, antigen_index]
                    ]
                    if values.numel():
                        assert torch.equal(values, values[0].expand_as(values))


def test_builder_rejects_partner_ranking_loss() -> None:
    class _Dataset:
        antibody_input_dim = 7
        antigen_input_dim = 11

    base = PairscaleOverfitConfig(
        expected_complex_count=192,
        epochs=260,
        partner_ranking_weight=0.0,
    )
    model = build_unary_control_model(_Dataset(), base, "antibody_only")
    assert model.control_side == "antibody_only"
    with pytest.raises(ValueError, match="zero partner-ranking"):
        build_unary_control_model(
            _Dataset(),
            replace(base, partner_ranking_weight=0.25),
            "antibody_only",
        )


def test_model_rejects_invalid_masks_and_pair_representations() -> None:
    model = UnaryResidueContactModel(
        control_side="antibody_only",
        antibody_input_dim=7,
        antigen_input_dim=11,
    )
    inputs = list(_inputs())
    invalid_pair_mask = inputs[4].clone()
    invalid_pair_mask[1, 3, 0] = True
    inputs[4] = invalid_pair_mask
    with pytest.raises(ValueError, match="padded residue"):
        model(*inputs, enabled_heads={"contact", "distance"})
    with pytest.raises(ValueError, match="do not construct"):
        model(*_inputs(), return_pair_representations=True)
