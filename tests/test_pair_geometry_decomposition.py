"""Tests for auditable unary and interaction-residual contact semantics."""

from __future__ import annotations

import importlib.util

import pytest


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="PyTorch is unavailable",
)


def _schema():
    from interaction_first.pair_geometry_schema import PairGeometrySchema

    return PairGeometrySchema(
        distance_bin_edges_angstrom=(2.0, 4.5, 8.0, 16.0),
        contact_cutoff_angstrom=4.5,
        orientation_axis_names=("placeholder",),
        orientation_axis_is_cyclic=(False,),
        orientation_bin_edges_radians=((0.0,),),
        label_format_version="pairscale_e1_e2_test_targets_v1",
        orientation_schema_version="pairscale_e1_e2_test_orientation_v1",
        primary_contact_definition="test_contact_definition",
        distance_definition="test_distance_definition",
        orientation_definition="orientation_not_executed",
        orientation_bin_edge_convention=(
            "internal_right_closed_thresholds_num_bins_minus_one"
        ),
    )


def _model(*, decomposed: bool = True):
    from interaction_first.pair_geometry import ProspectivePairGeometryModel

    return ProspectivePairGeometryModel(
        antibody_input_dim=5,
        antigen_input_dim=7,
        d_model=12,
        pair_dim=8,
        num_heads=3,
        num_fusion_layers=2,
        ffn_dim=24,
        geometry_schema=_schema(),
        pair_chunk_size=2,
        dropout=0.0,
        centering_iterations=32,
        contact_logit_mode=(
            "unary_plus_interaction_residual" if decomposed else "distance_mass"
        ),
    )


def _inputs():
    import torch

    torch.manual_seed(20260728)
    antibody = torch.randn(2, 4, 5)
    antigen = torch.randn(2, 6, 7)
    antibody_mask = torch.tensor(
        [[True, True, True, True], [True, True, True, False]]
    )
    antigen_mask = torch.tensor(
        [
            [True, True, True, True, True, True],
            [True, True, True, True, False, False],
        ]
    )
    candidate_mask = antibody_mask.unsqueeze(2) & antigen_mask.unsqueeze(1)
    candidate_mask[0, 1, 4] = False
    candidate_mask[1, 0, 2] = False
    return antibody, antigen, antibody_mask, antigen_mask, candidate_mask


def test_contact_logits_exactly_reconstruct_masked_unary_plus_residual() -> None:
    import torch

    model = _model().eval()
    output = model(
        *_inputs(),
        return_pair_representations=False,
        enabled_heads={"contact", "interaction"},
    )
    mask = output["candidate_pair_mask"]
    reconstructed = (
        output["antibody_unary_logits"].unsqueeze(2)
        + output["antigen_unary_logits"].unsqueeze(1)
        + output["contact_interaction_residual_logits"]
    ).masked_fill(~mask, 0.0)
    torch.testing.assert_close(output["contact_logits"], reconstructed)
    assert output["contact_logit_source"] == (
        "antibody_unary_plus_antigen_unary_plus_interaction_residual"
    )
    assert output["partner_score_source"] == (
        "double_centered_independent_specificity_interaction_residual"
    )
    for key in (
        "contact_logits",
        "contact_interaction_residual_logits",
        "specificity_interaction_residual_logits",
        "double_centered_specificity_interaction_residual_logits",
    ):
        assert torch.equal(output[key][~mask], torch.zeros_like(output[key][~mask]))


def test_unary_logits_are_invariant_to_the_opposite_partner() -> None:
    import torch

    model = _model().eval()
    antibody, antigen, antibody_mask, antigen_mask, candidate_mask = _inputs()
    baseline = model(
        antibody,
        antigen,
        antibody_mask,
        antigen_mask,
        candidate_mask,
        enabled_heads={"contact", "interaction"},
    )
    changed_antigen = model(
        antibody,
        antigen + 9.0,
        antibody_mask,
        antigen_mask,
        candidate_mask,
        enabled_heads={"contact", "interaction"},
    )
    changed_antibody = model(
        antibody - 7.0,
        antigen,
        antibody_mask,
        antigen_mask,
        candidate_mask,
        enabled_heads={"contact", "interaction"},
    )
    torch.testing.assert_close(
        baseline["antibody_unary_logits"],
        changed_antigen["antibody_unary_logits"],
    )
    torch.testing.assert_close(
        baseline["antigen_unary_logits"],
        changed_antibody["antigen_unary_logits"],
    )


def test_specificity_score_cannot_read_or_update_unary_heads() -> None:
    import torch

    from interaction_first.pairscale_partner_specificity import (
        bidirectional_specificity_losses,
        build_single_anchor_score_row,
    )

    model = _model()
    antibody, antigen, antibody_mask, antigen_mask, candidate_mask = _inputs()
    matched = model(
        antibody[:1],
        antigen[:1],
        antibody_mask[:1],
        antigen_mask[:1],
        candidate_mask[:1],
        return_pair_representations=False,
        enabled_heads={"interaction"},
    )
    antigen_decoy = model(
        antibody[:1],
        antigen[:1] + 2.0,
        antibody_mask[:1],
        antigen_mask[:1],
        candidate_mask[:1],
        return_pair_representations=False,
        enabled_heads={"interaction"},
    )
    antibody_decoy = model(
        antibody[:1] - 3.0,
        antigen[:1],
        antibody_mask[:1],
        antigen_mask[:1],
        candidate_mask[:1],
        return_pair_representations=False,
        enabled_heads={"interaction"},
    )
    positive_mask = torch.tensor([[True, False]])
    losses = bidirectional_specificity_losses(
        antibody_to_antigen_scores=build_single_anchor_score_row(
            matched,
            [antigen_decoy],
        ),
        antibody_to_antigen_positive_mask=positive_mask,
        antigen_to_antibody_scores=build_single_anchor_score_row(
            matched,
            [antibody_decoy],
        ),
        antigen_to_antibody_positive_mask=positive_mask,
        retrieval_temperature=0.5,
        margin=0.2,
        margin_temperature=0.5,
    )
    (losses.retrieval + losses.margin).backward()
    assert model.specificity_head is not None
    assert model.specificity_head.weight.grad is not None
    assert bool(torch.isfinite(model.specificity_head.weight.grad).all())
    assert model.antibody_unary_head is not None
    assert model.antigen_unary_head is not None
    assert model.antibody_unary_head.weight.grad is None
    assert model.antibody_unary_head.bias.grad is None
    assert model.antigen_unary_head.weight.grad is None
    assert model.antigen_unary_head.bias.grad is None
    assert model.interaction_head.weight.grad is None


def test_changing_unary_heads_changes_contacts_not_partner_score() -> None:
    import torch

    model = _model().eval()
    inputs = _inputs()
    baseline = model(*inputs, enabled_heads={"contact", "interaction"})
    assert model.antibody_unary_head is not None
    assert model.antigen_unary_head is not None
    with torch.no_grad():
        model.antibody_unary_head.weight.add_(3.0)
        model.antibody_unary_head.bias.add_(2.0)
        model.antigen_unary_head.weight.sub_(4.0)
        model.antigen_unary_head.bias.sub_(1.0)
    altered = model(*inputs, enabled_heads={"contact", "interaction"})
    assert not torch.equal(baseline["contact_logits"], altered["contact_logits"])
    torch.testing.assert_close(baseline["partner_score"], altered["partner_score"])
    torch.testing.assert_close(
        baseline["specificity_interaction_residual_logits"],
        altered["specificity_interaction_residual_logits"],
    )


def test_decomposed_contact_requires_interaction_head() -> None:
    model = _model()
    with pytest.raises(ValueError, match="requires the interaction head"):
        model(*_inputs(), enabled_heads={"contact", "distance"})


def test_default_mode_retains_legacy_parameter_and_dependency_surface() -> None:
    model = _model(decomposed=False)
    state_names = set(model.state_dict())
    assert not any(name.startswith("antibody_unary_head") for name in state_names)
    assert not any(name.startswith("antigen_unary_head") for name in state_names)
    assert not any(name.startswith("specificity_head") for name in state_names)
    with pytest.raises(ValueError, match="contact head requires"):
        model(*_inputs(), enabled_heads={"contact"})
