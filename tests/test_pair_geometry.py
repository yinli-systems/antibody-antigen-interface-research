"""Focused tests for the prospective pair-geometry research module."""

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
        distance_bin_edges_angstrom=(1.5, 3.0, 4.5, 6.0, 8.0, 12.0, 20.0),
        contact_cutoff_angstrom=4.5,
        orientation_axis_names=("ab_azimuth", "ab_polar", "ag_azimuth"),
        orientation_axis_is_cyclic=(True, False, True),
        orientation_bin_edges_radians=(
            (-2.0, -1.0, 0.0, 1.0, 2.0),
            (0.4, 0.8, 1.2, 1.6, 2.0),
            (-2.0, -1.0, 0.0, 1.0, 2.0),
        ),
        label_format_version="pair_geometry_labels_v2",
        orientation_schema_version="test_orientation_schema_v1",
        primary_contact_definition="test_positive_or_complete_contact",
        distance_definition="test_complete_pair_distance",
        orientation_definition=(
            "right_handed_N_CA_C_backbone_frames_with_CA_to_CA_directions"
        ),
        orientation_bin_edge_convention=(
            "internal_right_closed_thresholds_num_bins_minus_one"
        ),
    )


def _model(*, pair_chunk_size: int = 2, num_pair_update_layers: int = 0):
    from interaction_first.pair_geometry import ProspectivePairGeometryModel

    return ProspectivePairGeometryModel(
        antibody_input_dim=7,
        antigen_input_dim=11,
        d_model=16,
        pair_dim=12,
        num_heads=4,
        num_fusion_layers=2,
        ffn_dim=32,
        geometry_schema=_schema(),
        num_distance_bins=8,
        contact_distance_bin_count=3,
        num_orientation_axes=3,
        num_orientation_bins=6,
        num_pair_update_layers=num_pair_update_layers,
        pair_chunk_size=pair_chunk_size,
        dropout=0.0,
        centering_iterations=32,
    )


def _inputs():
    import torch

    torch.manual_seed(20260725)
    antibody = torch.randn(2, 5, 7)
    antigen = torch.randn(2, 6, 11)
    antibody_mask = torch.tensor(
        [
            [True, True, True, True, True],
            [True, True, True, False, False],
        ]
    )
    antigen_mask = torch.tensor(
        [
            [True, True, True, True, True, True],
            [True, True, True, True, False, False],
        ]
    )
    candidate_mask = antibody_mask.unsqueeze(2) & antigen_mask.unsqueeze(1)
    candidate_mask[0, 1, 3] = False
    candidate_mask[0, 3, 2] = False
    candidate_mask[1, 0, 1] = False
    return antibody, antigen, antibody_mask, antigen_mask, candidate_mask


def test_outputs_are_explicit_pair_geometry_with_coupled_contact_probability() -> None:
    import torch

    model = _model().eval()
    inputs = _inputs()
    output = model(*inputs)
    candidate_mask = inputs[-1]

    assert output["pair_orientation"] == "antibody_by_antigen"
    assert output["geometry_schema"] == _schema()
    assert output["geometry_schema_sha256"] == _schema().sha256
    assert hash(output["geometry_schema"]) == hash(_schema())
    assert hash(output["geometry_schema_metadata"])
    with pytest.raises(TypeError):
        output["geometry_schema_metadata"]["num_distance_bins"] = 99
    assert output["executed_heads"] == (
        "contact",
        "distance",
        "orientation",
        "interaction",
    )
    assert output["pair_representations"].shape == (2, 5, 6, 12)
    assert output["contact_logits"].shape == (2, 5, 6)
    assert output["distance_bin_logits"].shape == (2, 5, 6, 8)
    assert output["orientation_bin_logits"].shape == (2, 5, 6, 3, 6)
    assert output["epitope_probability"].shape == (2, 6)
    assert output["paratope_probability"].shape == (2, 5)
    assert output["partner_score"].shape == (2,)

    distance_mass = output["distance_bin_probabilities"][..., :3].sum(dim=-1)
    torch.testing.assert_close(
        output["contact_probabilities"][candidate_mask],
        distance_mass[candidate_mask],
    )
    assert torch.equal(
        output["pair_representations"][~candidate_mask],
        torch.zeros_like(output["pair_representations"][~candidate_mask]),
    )
    for key in (
        "contact_probabilities",
        "pair_confidence",
        "pair_uncertainty",
        "pair_interactions",
        "double_centered_pair_interactions",
    ):
        assert torch.equal(
            output[key][~candidate_mask],
            torch.zeros_like(output[key][~candidate_mask]),
        )
    torch.testing.assert_close(
        (
            output["pair_confidence"][candidate_mask]
            + output["pair_uncertainty"][candidate_mask]
        ),
        torch.ones_like(output["pair_confidence"][candidate_mask]),
    )
    assert bool(
        (
            (output["pair_uncertainty"][candidate_mask] >= 0.0)
            & (output["pair_uncertainty"][candidate_mask] <= 1.0)
        ).all()
    )


def test_residue_marginals_are_derived_only_from_pair_contact_logits() -> None:
    from interaction_first.scoring import pool_pair_marginals

    model = _model().eval()
    output = model(*_inputs())
    expected = pool_pair_marginals(
        output["contact_logits"].transpose(1, 2),
        output["candidate_pair_mask"].transpose(1, 2),
        pooling=model.marginal_pooling,
        top_k=model.marginal_top_k,
        temperature=model.marginal_temperature,
    )
    import torch

    torch.testing.assert_close(output["epitope_probability"], expected[0])
    torch.testing.assert_close(output["paratope_probability"], expected[1])


def test_chunk_size_changes_peak_construction_not_numerical_results() -> None:
    import torch

    chunked = _model(pair_chunk_size=2).eval()
    unchunked = _model(pair_chunk_size=64).eval()
    unchunked.load_state_dict(chunked.state_dict())
    chunked_output = chunked(*_inputs())
    unchunked_output = unchunked(*_inputs())
    for key in (
        "pair_representations",
        "contact_logits",
        "distance_bin_logits",
        "orientation_bin_logits",
        "pair_confidence",
        "pair_uncertainty",
        "double_centered_pair_interactions",
        "partner_score",
        "epitope_probability",
        "paratope_probability",
    ):
        torch.testing.assert_close(chunked_output[key], unchunked_output[key])


def test_pair_axial_updates_are_chunk_invariant_and_execute_both_axes() -> None:
    import torch

    chunked = _model(
        pair_chunk_size=2,
        num_pair_update_layers=2,
    ).eval()
    unchunked = _model(
        pair_chunk_size=64,
        num_pair_update_layers=2,
    ).eval()
    unchunked.load_state_dict(chunked.state_dict())

    chunked_output = chunked(*_inputs())
    unchunked_output = unchunked(*_inputs())
    assert chunked_output["num_pair_update_layers"] == 2
    assert chunked_output["pair_update_order"] == (
        "antigen_axis_within_antibody_row",
        "antibody_axis_within_antigen_column",
        "feed_forward",
    )
    for key in (
        "pair_representations",
        "contact_logits",
        "distance_bin_logits",
        "orientation_bin_logits",
        "pair_interactions",
        "partner_score",
    ):
        torch.testing.assert_close(chunked_output[key], unchunked_output[key])

    omitted_output = chunked(*_inputs(), return_pair_representations=False)
    assert omitted_output["pair_representations"] is None
    torch.testing.assert_close(
        chunked_output["contact_logits"],
        omitted_output["contact_logits"],
    )

    chunked_output["contact_logits"][
        chunked_output["candidate_pair_mask"]
    ].square().mean().backward()
    for layer in chunked.pair_update_layers:
        assert layer.row_attention.in_proj_weight.grad is not None
        assert layer.column_attention.in_proj_weight.grad is not None
        assert bool(torch.isfinite(layer.row_attention.in_proj_weight.grad).all())
        assert bool(torch.isfinite(layer.column_attention.in_proj_weight.grad).all())


def test_pair_axial_block_is_mask_safe_with_empty_rows_and_columns() -> None:
    import torch

    from interaction_first.pair_geometry import BipartitePairAxialUpdateBlock

    torch.manual_seed(20260729)
    block = BipartitePairAxialUpdateBlock(
        pair_dim=12,
        num_heads=4,
        ffn_dim=24,
        dropout=0.0,
    ).eval()
    pair_state = torch.randn(2, 4, 5, 12, requires_grad=True)
    pair_mask = torch.zeros(2, 4, 5, dtype=torch.bool)
    pair_mask[0, 0, :3] = True
    pair_mask[0, 2, 1:5] = True
    pair_mask[1, 1, 0] = True
    pair_mask[1, 3, 2:4] = True

    output = block(pair_state, pair_mask)
    altered_state = pair_state.detach().clone()
    altered_state[~pair_mask] = 1e6
    altered_output = block(altered_state, pair_mask)

    assert bool(torch.isfinite(output).all())
    assert torch.equal(output[~pair_mask], torch.zeros_like(output[~pair_mask]))
    torch.testing.assert_close(output[pair_mask], altered_output[pair_mask])
    output[pair_mask].square().mean().backward()
    assert pair_state.grad is not None
    assert bool(torch.isfinite(pair_state.grad).all())


def test_pair_axial_configuration_fails_closed() -> None:
    from interaction_first.pair_geometry import (
        BipartitePairAxialUpdateBlock,
        ProspectivePairGeometryModel,
    )

    with pytest.raises(ValueError, match="nonnegative integer"):
        ProspectivePairGeometryModel(
            7,
            11,
            geometry_schema=_schema(),
            num_pair_update_layers=-1,
        )
    with pytest.raises(ValueError, match="nonnegative integer"):
        ProspectivePairGeometryModel(
            7,
            11,
            geometry_schema=_schema(),
            num_pair_update_layers=True,
        )
    with pytest.raises(ValueError, match="pair_dim must be divisible"):
        ProspectivePairGeometryModel(
            7,
            11,
            d_model=16,
            pair_dim=10,
            num_heads=4,
            geometry_schema=_schema(),
            num_pair_update_layers=1,
        )
    with pytest.raises(ValueError, match="pair_dim must be divisible"):
        BipartitePairAxialUpdateBlock(
            pair_dim=10,
            num_heads=4,
            ffn_dim=20,
            dropout=0.0,
        )


def test_pair_representations_can_be_omitted_for_lower_peak_retention() -> None:
    import torch

    model = _model().eval()
    retained = model(*_inputs())
    omitted = model(*_inputs(), return_pair_representations=False)
    assert omitted["pair_representations"] is None
    torch.testing.assert_close(retained["contact_logits"], omitted["contact_logits"])
    torch.testing.assert_close(retained["partner_score"], omitted["partner_score"])


def test_optional_intra_chain_geometry_runs_before_partner_fusion() -> None:
    import torch

    from interaction_first.pair_geometry import ProspectivePairGeometryModel

    antibody, antigen, antibody_mask, antigen_mask, candidate_mask = _inputs()
    model = ProspectivePairGeometryModel(
        antibody_input_dim=7,
        antigen_input_dim=11,
        d_model=16,
        pair_dim=12,
        num_heads=4,
        num_fusion_layers=0,
        intra_chain_geometry_layers=1,
        intra_chain_geometry_neighbor_k=3,
        ffn_dim=32,
        geometry_schema=_schema(),
        pair_chunk_size=2,
        dropout=0.0,
    )
    antibody_backbone = torch.randn(2, 5, 3, 3)
    antigen_backbone = torch.randn(2, 6, 3, 3)
    output = model(
        antibody,
        antigen,
        antibody_mask,
        antigen_mask,
        candidate_mask,
        antibody_backbone=antibody_backbone,
        antigen_backbone=antigen_backbone,
        return_intra_chain_attention=True,
    )
    assert output["intra_chain_geometry_enabled"] is True
    assert output["intra_chain_geometry"] is not None
    assert len(output["intra_chain_geometry"]["antibody"]["layers"]) == 1
    assert len(output["intra_chain_geometry"]["antigen"]["layers"]) == 1
    output["contact_logits"].square().mean().backward()
    assert (
        model.intra_chain_geometry
        .antibody_encoder.blocks[0]
        .geometry_bias.network[-1].weight.grad
        is not None
    )
    assert (
        model.intra_chain_geometry
        .antigen_encoder.blocks[0]
        .geometry_bias.network[-1].weight.grad
        is not None
    )


def test_enabled_heads_skip_disabled_decoders_and_enforce_dependencies() -> None:
    import torch

    model = _model()
    output = model(
        *_inputs(),
        return_pair_representations=False,
        enabled_heads={"distance"},
    )
    assert output["executed_heads"] == ("distance",)
    assert "distance_bin_logits" in output
    for key in (
        "contact_logits",
        "orientation_bin_logits",
        "pair_interactions",
        "partner_score",
    ):
        assert key not in output
    output["distance_bin_logits"].sum().backward()
    assert model.distance_head.weight.grad is not None
    assert model.orientation_head.weight.grad is None
    assert model.interaction_head.weight.grad is None

    with pytest.raises(ValueError, match="contact head requires"):
        model(*_inputs(), enabled_heads={"contact"})
    with pytest.raises(ValueError, match="at least one"):
        model(*_inputs(), enabled_heads=set())


def test_model_rejects_missing_or_dimension_mismatched_schema() -> None:
    from interaction_first.pair_geometry import ProspectivePairGeometryModel

    with pytest.raises(TypeError, match="explicit PairGeometrySchema"):
        ProspectivePairGeometryModel(7, 11)
    with pytest.raises(ValueError, match="does not match geometry_schema"):
        ProspectivePairGeometryModel(
            7,
            11,
            geometry_schema=_schema(),
            num_distance_bins=9,
        )


def test_partner_score_is_invariant_to_additive_row_and_column_effects() -> None:
    import torch

    from interaction_first.pair_geometry import double_centered_partner_score

    torch.manual_seed(43)
    interactions = torch.randn(2, 5, 7)
    mask = torch.ones_like(interactions, dtype=torch.bool)
    row_effect = torch.randn(2, 5, 1)
    column_effect = torch.randn(2, 1, 7)
    centered, score = double_centered_partner_score(
        interactions,
        mask,
        centering_iterations=32,
    )
    shifted_centered, shifted_score = double_centered_partner_score(
        interactions + row_effect + column_effect + 9.0,
        mask,
        centering_iterations=32,
    )
    torch.testing.assert_close(centered, shifted_centered, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(score, shifted_score, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(centered.mean(dim=1), torch.zeros(2, 7), atol=1e-6, rtol=0)
    torch.testing.assert_close(centered.mean(dim=2), torch.zeros(2, 5), atol=1e-6, rtol=0)


@pytest.mark.parametrize("num_pair_update_layers", [0, 2])
def test_padded_feature_values_cannot_change_candidate_outputs(
    num_pair_update_layers: int,
) -> None:
    import torch

    model = _model(num_pair_update_layers=num_pair_update_layers).eval()
    inputs = list(_inputs())
    original = model(*inputs)
    changed = [value.clone() for value in inputs]
    changed[0][~changed[2]] = 1e6
    changed[1][~changed[3]] = -1e6
    altered = model(*changed)
    candidate_mask = inputs[-1]
    for key in (
        "contact_logits",
        "distance_bin_logits",
        "orientation_bin_logits",
        "pair_interactions",
    ):
        expanded_mask = candidate_mask
        while expanded_mask.ndim < original[key].ndim:
            expanded_mask = expanded_mask.unsqueeze(-1)
        expanded_mask = expanded_mask.expand_as(original[key])
        torch.testing.assert_close(
            original[key][expanded_mask],
            altered[key][expanded_mask],
        )
    torch.testing.assert_close(original["partner_score"], altered["partner_score"])


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("candidate_none", "candidate_pair_mask"),
        ("candidate_float", "boolean"),
        ("candidate_padded", "padded residue"),
        ("candidate_empty", "at least one candidate pair"),
        ("antibody_nan", "finite"),
        ("batch_mismatch", "equal size"),
    ],
)
def test_inputs_fail_closed(mutation: str, message: str) -> None:
    import torch

    model = _model().eval()
    inputs = list(_inputs())
    if mutation == "candidate_none":
        inputs[4] = None
    elif mutation == "candidate_float":
        inputs[4] = inputs[4].float()
    elif mutation == "candidate_padded":
        inputs[4][1, 4, 5] = True
    elif mutation == "candidate_empty":
        inputs[4][0] = False
    elif mutation == "antibody_nan":
        inputs[0][0, 0, 0] = torch.nan
    elif mutation == "batch_mismatch":
        inputs[1] = inputs[1][:1]
        inputs[3] = inputs[3][:1]
    else:
        raise AssertionError(f"Unhandled mutation: {mutation}")
    with pytest.raises((TypeError, ValueError), match=message):
        model(*inputs)


def test_joint_geometry_objective_has_finite_gradients() -> None:
    import torch

    model = _model()
    output = model(*_inputs())
    mask = output["candidate_pair_mask"]
    loss = (
        output["contact_logits"][mask].square().mean()
        + output["distance_bin_logits"][mask].square().mean()
        + output["orientation_bin_logits"][mask].square().mean()
        + output["pair_uncertainty"][mask].mean()
        + output["partner_score"].square().mean()
    )
    loss.backward()
    assert bool(torch.isfinite(loss))
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients
    assert all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
