"""Contract tests for the interaction-first FrameContact model."""

from __future__ import annotations

import importlib.util

import pytest


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="PyTorch is unavailable",
)


def _backbone(batch_size: int, residue_count: int):
    import torch

    ca = torch.cumsum(torch.randn(batch_size, residue_count, 3), dim=1)
    n = ca + torch.tensor([-1.1, 0.7, 0.2])
    c = ca + torch.tensor([1.2, 0.1, -0.3])
    return torch.stack((n, ca, c), dim=-2)


def _batch(batch_size: int = 2):
    import torch

    from interaction_first.intra_chain_geometry import build_backbone_frames

    torch.manual_seed(20260730)
    antibody_mask = torch.tensor([[True] * 5, [True, True, True, True, False]])
    antigen_mask = torch.tensor(
        [[True] * 7, [True, True, True, True, True, True, False]]
    )
    if batch_size == 1:
        antibody_mask = antibody_mask[:1]
        antigen_mask = antigen_mask[:1]
    antibody_backbone = _backbone(batch_size, 5)
    antigen_backbone = _backbone(batch_size, 7)
    antibody_backbone = antibody_backbone.masked_fill(
        ~antibody_mask[..., None, None], float("nan")
    )
    antigen_backbone = antigen_backbone.masked_fill(
        ~antigen_mask[..., None, None], float("nan")
    )
    antibody_valid = build_backbone_frames(
        antibody_backbone, antibody_mask
    ).valid_mask
    antigen_valid = build_backbone_frames(antigen_backbone, antigen_mask).valid_mask
    return {
        "antibody_features": torch.randn(batch_size, 5, 12).masked_fill(
            ~antibody_mask[..., None], 0.0
        ),
        "antigen_features": torch.randn(batch_size, 7, 10).masked_fill(
            ~antigen_mask[..., None], 0.0
        ),
        "antibody_backbone": antibody_backbone,
        "antigen_backbone": antigen_backbone,
        "antibody_mask": antibody_mask,
        "antigen_mask": antigen_mask,
        "antibody_chain_ids": torch.tensor([[0, 0, 0, 1, 1]]).expand(
            batch_size, -1
        ).clone(),
        "antigen_chain_ids": torch.zeros(batch_size, 7, dtype=torch.long),
        "antibody_frame_valid_mask": antibody_valid,
        "antigen_frame_valid_mask": antigen_valid,
        "antibody_missing_frame_mask": antibody_mask & ~antibody_valid,
        "antigen_missing_frame_mask": antigen_mask & ~antigen_valid,
    }


def _model(attention_kernel: str = "reference"):
    from interaction_first.framecontact import FrameContactModel

    return FrameContactModel(
        antibody_input_dim=12,
        antigen_input_dim=10,
        d_model=16,
        num_heads=4,
        num_self_layers=1,
        neighbor_k=4,
        neighbor_radius=None,
        dropout=0.0,
        attention_kernel=attention_kernel,
    ).eval()


def _interleaved_model():
    from interaction_first.framecontact import FrameContactModel

    return FrameContactModel(
        antibody_input_dim=12,
        antigen_input_dim=10,
        d_model=16,
        num_heads=4,
        num_self_layers=2,
        neighbor_k=4,
        neighbor_radius=None,
        dropout=0.0,
        fusion_mode="interleaved",
    ).eval()


def _activate_geometry(model) -> None:
    import torch

    with torch.no_grad():
        for stream in (
            model.self_encoder.antibody_encoder,
            model.self_encoder.antigen_encoder,
        ):
            stream.geometry_residual_gate.fill_(0.7)
            for block in stream.geometry_encoder.blocks:
                block.geometry_bias.network[-1].weight.normal_(std=0.08)
        model.cross_attention.antibody_from_antigen.residual_gate.fill_(0.4)
        model.cross_attention.antigen_from_antibody.residual_gate.fill_(0.4)
        model.paratope_contact_gate.fill_(0.3)
        model.epitope_contact_gate.fill_(0.3)


def test_framecontact_shapes_masks_and_parameter_independence() -> None:
    import torch

    model = _model()
    batch = _batch()
    result = model(batch, return_attention=True)
    assert result["contact_logits"].shape == (2, 5, 7)
    assert result["contact_interaction_logits"].shape == (2, 5, 7)
    assert result["antibody_contact_propensity_logits"].shape == (2, 5)
    assert result["antigen_contact_propensity_logits"].shape == (2, 7)
    torch.testing.assert_close(
        result["initial_contact_logits"],
        result["contact_logits"],
        atol=0.0,
        rtol=0.0,
    )
    assert result["paratope_logits"].shape == (2, 5)
    assert result["epitope_logits"].shape == (2, 7)
    assert torch.isfinite(result["contact_logits"]).all()
    assert torch.equal(
        result["pair_candidate_mask"],
        batch["antibody_mask"][:, :, None] & batch["antigen_mask"][:, None, :],
    )
    assert torch.equal(
        result["contact_logits"][~result["pair_candidate_mask"]],
        torch.zeros_like(result["contact_logits"][~result["pair_candidate_mask"]]),
    )
    reconstructed = (
        result["contact_interaction_logits"]
        + result["antibody_contact_propensity_logits"][:, :, None]
        + result["antigen_contact_propensity_logits"][:, None, :]
        + model.contact_head.contact_bias
    )
    torch.testing.assert_close(
        result["contact_logits"][result["pair_candidate_mask"]],
        reconstructed[result["pair_candidate_mask"]],
    )
    antibody_storage = {
        parameter.data_ptr()
        for parameter in model.self_encoder.antibody_encoder.parameters()
    }
    antigen_storage = {
        parameter.data_ptr()
        for parameter in model.self_encoder.antigen_encoder.parameters()
    }
    assert antibody_storage.isdisjoint(antigen_storage)
    assert (
        model.cross_attention.antibody_from_antigen
        is not model.cross_attention.antigen_from_antibody
    )


def test_interleaved_model_runs_one_contact_gated_update_after_each_self_layer() -> None:
    import torch

    model = _interleaved_model()
    assert model.self_encoder is None
    assert model.cross_attention is None
    assert model.interleaved_self_encoders is not None
    assert model.interleaved_cross_attentions is not None
    assert len(model.interleaved_self_encoders) == 2
    assert len(model.interleaved_cross_attentions) == 2
    assert (
        model.interleaved_cross_attentions[0].antibody_from_antigen
        is not model.interleaved_cross_attentions[1].antibody_from_antigen
    )

    batch = _batch()
    result = model(batch, return_attention=True)
    assert result["fusion_mode"] == "interleaved"
    assert len(result["intermediate_contact_logits"]) == 2
    assert len(result["cross_attention_details"]["layers"]) == 2
    assert len(
        result["self_attention_details"]["antigen"]["geometry"]["layers"]
    ) == 2
    assert result["contact_logits"].shape == (2, 5, 7)
    assert torch.isfinite(result["contact_logits"]).all()


def test_unknown_fusion_mode_fails_closed() -> None:
    from interaction_first.framecontact import FrameContactModel

    with pytest.raises(ValueError, match="fusion_mode"):
        FrameContactModel(
            antibody_input_dim=12,
            antigen_input_dim=10,
            d_model=16,
            num_heads=4,
            num_self_layers=1,
            fusion_mode="decoder_magic",
        )


def test_framecontact_sdpa_matches_reference_end_to_end() -> None:
    import torch

    torch.manual_seed(20260801)
    reference = _model("reference")
    sdpa = _model("sdpa")
    _activate_geometry(reference)
    with torch.no_grad():
        sdpa.load_state_dict(reference.state_dict())
    reference.train()
    sdpa.train()
    reference_batch = _batch(batch_size=1)
    sdpa_batch = {
        key: value.detach().clone().requires_grad_(
            key in {"antibody_features", "antigen_features"}
        )
        if torch.is_tensor(value)
        else value
        for key, value in reference_batch.items()
    }
    reference_batch["antibody_features"].requires_grad_(True)
    reference_batch["antigen_features"].requires_grad_(True)
    reference_output = reference(reference_batch, return_attention=False)
    sdpa_output = sdpa(sdpa_batch, return_attention=False)
    for key in ("contact_logits", "paratope_logits", "epitope_logits"):
        torch.testing.assert_close(
            reference_output[key], sdpa_output[key], atol=3e-5, rtol=3e-5
        )
    reference_loss = sum(
        reference_output[key].float().square().mean()
        for key in ("contact_logits", "paratope_logits", "epitope_logits")
    )
    sdpa_loss = sum(
        sdpa_output[key].float().square().mean()
        for key in ("contact_logits", "paratope_logits", "epitope_logits")
    )
    reference_loss.backward()
    sdpa_loss.backward()
    for key in ("antibody_features", "antigen_features"):
        torch.testing.assert_close(
            reference_batch[key].grad,
            sdpa_batch[key].grad,
            atol=8e-5,
            rtol=8e-5,
        )
    for (reference_name, reference_parameter), (sdpa_name, sdpa_parameter) in zip(
        reference.named_parameters(), sdpa.named_parameters(), strict=True
    ):
        assert reference_name == sdpa_name
        if reference_parameter.grad is None or sdpa_parameter.grad is None:
            assert reference_parameter.grad is None and sdpa_parameter.grad is None
            continue
        torch.testing.assert_close(
            reference_parameter.grad,
            sdpa_parameter.grad,
            atol=2e-4,
            rtol=2e-4,
        )


def test_partner_conditioning_changes_epitope_for_a_fixed_antigen() -> None:
    import torch

    model = _model()
    batch = _batch()
    for key in tuple(batch):
        if key.startswith("antigen_"):
            batch[key][1] = batch[key][0]

    independent = model(batch)
    torch.testing.assert_close(
        independent["epitope_logits"][0],
        independent["epitope_logits"][1],
        atol=1e-6,
        rtol=1e-6,
    )

    _activate_geometry(model)
    conditioned = model(batch)
    assert not torch.allclose(
        conditioned["epitope_logits"][0],
        conditioned["epitope_logits"][1],
    )


def test_cross_attention_diagnostics_are_direction_and_layer_resolved() -> None:
    from interaction_first.attention_diagnostics import (
        CrossAttentionDiagnosticsAccumulator,
    )

    model = _interleaved_model()
    batch = _batch()
    output = model(batch, return_attention=True)
    accumulator = CrossAttentionDiagnosticsAccumulator()
    accumulator.update(
        output["cross_attention_details"],
        output["antibody_mask"],
        output["antigen_mask"],
    )
    result = accumulator.result()
    assert set(result["directions"]) == {
        "antibody_from_antigen",
        "antigen_from_antibody",
    }
    for direction in result["directions"].values():
        assert len(direction["layers"]) == 2
        for layer in direction["layers"]:
            assert len(layer["contact_scale_per_head"]) == 4
            assert len(
                layer["metrics"]["mean_contact_gate_total_variation_per_head"]
            ) == 4


def test_independent_monomer_se3_invariance_after_all_paths_are_active() -> None:
    import torch

    from interaction_first.framecontact_interventions import (
        independent_monomer_rigid_transform,
    )

    torch.manual_seed(31)
    model = _model()
    _activate_geometry(model)
    batch = _batch()
    before = model(batch, return_attention=True)
    generator = torch.Generator().manual_seed(37)
    transformed = independent_monomer_rigid_transform(
        batch, generator=generator
    )
    after = model(transformed, return_attention=True)
    for key in ("contact_logits", "paratope_logits", "epitope_logits"):
        torch.testing.assert_close(before[key], after[key], atol=3e-5, rtol=3e-5)
    before_cross = before["cross_attention_details"]["antibody_from_antigen"]
    after_cross = after["cross_attention_details"]["antibody_from_antigen"]
    torch.testing.assert_close(
        before_cross["weights"], after_cross["weights"], atol=3e-5, rtol=3e-5
    )


def test_contact_bottleneck_is_independently_monomer_se3_invariant() -> None:
    import torch

    from interaction_first.framecontact import FrameContactModel
    from interaction_first.framecontact_interventions import (
        independent_monomer_rigid_transform,
    )

    torch.manual_seed(20260801)
    model = FrameContactModel(
        antibody_input_dim=12,
        antigen_input_dim=10,
        d_model=16,
        num_heads=4,
        num_self_layers=1,
        neighbor_k=4,
        neighbor_radius=None,
        dropout=0.0,
        fusion_mode="late",
        contact_bottleneck_projection_dim=8,
    ).eval()
    _activate_geometry(model)
    batch = _batch()
    before = model(batch)
    transformed = independent_monomer_rigid_transform(
        batch, generator=torch.Generator().manual_seed(20260802)
    )
    after = model(transformed)
    torch.testing.assert_close(
        before["contact_bottleneck_compatibility"],
        after["contact_bottleneck_compatibility"],
        atol=3e-5,
        rtol=3e-5,
    )
    torch.testing.assert_close(
        before["contact_bottleneck_pair_contact_lift"],
        after["contact_bottleneck_pair_contact_lift"],
        atol=3e-5,
        rtol=3e-5,
    )


def test_contact_bottleneck_candidate_mask_and_gradient_modes() -> None:
    import torch

    from interaction_first.framecontact import ContactBottleneckContrastiveHead

    torch.manual_seed(20260803)
    antibody_mask = torch.tensor([[True, True, False]])
    antigen_mask = torch.tensor([[True, True, False, False]])
    pair_mask = torch.tensor(
        [
            [
                [True, False, False, False],
                [False, True, False, False],
                [False, False, False, False],
            ]
        ]
    )
    antibody = torch.randn(1, 3, 6, requires_grad=True)
    antigen = torch.randn(1, 4, 6, requires_grad=True)
    total_logits = torch.full((1, 3, 4), -2.0, requires_grad=True)
    interaction_logits = torch.full((1, 3, 4), 2.0, requires_grad=True)

    stop = ContactBottleneckContrastiveHead(
        6, 5, gradient_mode="stop_gradient"
    )
    first = stop(
        antibody,
        antigen,
        antibody_mask,
        antigen_mask,
        total_logits,
        interaction_logits,
        pair_mask,
    )
    changed_total = total_logits.detach().clone()
    changed_interaction = interaction_logits.detach().clone()
    changed_total[~pair_mask] = 100.0
    changed_interaction[~pair_mask] = -100.0
    second = stop(
        antibody,
        antigen,
        antibody_mask,
        antigen_mask,
        changed_total,
        changed_interaction,
        pair_mask,
    )
    torch.testing.assert_close(first["antibody_weights"], second["antibody_weights"])
    torch.testing.assert_close(first["antigen_weights"], second["antigen_weights"])
    assert first["antibody_weights"][0, 2].item() == 0.0
    assert torch.equal(
        first["antigen_weights"][0, 2:], torch.zeros_like(first["antigen_weights"][0, 2:])
    )
    first["compatibility"].sum().backward()
    assert total_logits.grad is None
    assert interaction_logits.grad is None
    assert antibody.grad is not None and float(antibody.grad.abs().sum()) > 0
    assert antigen.grad is not None and float(antigen.grad.abs().sum()) > 0

    joint_total = torch.full((1, 3, 4), -2.0, requires_grad=True)
    joint_interaction = torch.full((1, 3, 4), 2.0, requires_grad=True)
    joint = ContactBottleneckContrastiveHead(6, 5, gradient_mode="joint")
    joint_result = joint(
        antibody.detach(),
        antigen.detach(),
        antibody_mask,
        antigen_mask,
        joint_total,
        joint_interaction,
        pair_mask,
    )
    joint_result["compatibility"].sum().backward()
    for logits in (joint_total, joint_interaction):
        assert logits.grad is not None
        assert float(logits.grad[pair_mask].abs().sum()) > 0
        assert float(logits.grad[~pair_mask].abs().sum()) == 0.0


def test_contact_bottleneck_counterfactual_lift_rejects_unary_only_shortcut() -> None:
    import torch

    from interaction_first.framecontact import ContactBottleneckContrastiveHead

    pair_mask = torch.ones(1, 2, 3, dtype=torch.bool)
    unary_only = torch.tensor(
        [[[-3.0, -2.0, -1.0], [-2.5, -1.5, -0.5]]]
    )
    zero_interaction = torch.zeros_like(unary_only)
    lift = ContactBottleneckContrastiveHead._counterfactual_contact_lift(
        unary_only, zero_interaction, pair_mask
    )
    assert torch.equal(lift, torch.zeros_like(lift))

    positive_interaction = torch.full_like(unary_only, 1.25)
    total = unary_only + positive_interaction
    positive_lift = ContactBottleneckContrastiveHead._counterfactual_contact_lift(
        total, positive_interaction, pair_mask
    )
    expected = torch.sigmoid(total) - torch.sigmoid(unary_only)
    torch.testing.assert_close(positive_lift, expected)
    assert bool((positive_lift > 0).all())


def test_contact_bottleneck_is_default_off_and_rejects_interleaved_semantics() -> None:
    from interaction_first.framecontact import FrameContactModel

    default_model = _model()
    assert default_model.contact_bottleneck_head is None
    assert not any(
        name.startswith("contact_bottleneck_head.")
        for name in default_model.state_dict()
    )
    assert default_model(_batch())["contact_bottleneck_compatibility"] is None
    with pytest.raises(ValueError, match="requires late fusion"):
        FrameContactModel(
            antibody_input_dim=12,
            antigen_input_dim=10,
            d_model=16,
            num_heads=4,
            num_self_layers=1,
            fusion_mode="interleaved",
            contact_bottleneck_projection_dim=8,
        )


def test_zero_gates_preserve_coordinate_independent_dense_residue_path() -> None:
    import torch

    from interaction_first.framecontact_interventions import geometry_shuffle

    model = _model()
    batch = _batch()
    before = model(batch)
    shuffled = geometry_shuffle(
        batch, generator=torch.Generator().manual_seed(43)
    )
    after = model(shuffled)
    assert model.self_encoder.antibody_encoder.geometry_residual_gate.item() == 0.0
    assert model.cross_attention.antibody_from_antigen.residual_gate.item() == 0.0
    for key in ("contact_logits", "paratope_logits", "epitope_logits"):
        torch.testing.assert_close(before[key], after[key], atol=0.0, rtol=0.0)


def test_geometry_ablated_partner_aware_path_is_coordinate_independent() -> None:
    import torch

    from interaction_first.framecontact_interventions import geometry_shuffle

    model = _model()
    with torch.no_grad():
        for stream in (
            model.self_encoder.antibody_encoder,
            model.self_encoder.antigen_encoder,
        ):
            stream.geometry_residual_gate.zero_()
            stream.geometry_residual_gate.requires_grad_(False)
        model.cross_attention.antibody_from_antigen.residual_gate.fill_(0.4)
        model.cross_attention.antigen_from_antibody.residual_gate.fill_(0.4)
        model.paratope_contact_gate.fill_(0.3)
        model.epitope_contact_gate.fill_(0.3)
    batch = _batch()
    before = model(batch)
    shuffled = geometry_shuffle(
        batch, generator=torch.Generator().manual_seed(20260801)
    )
    after = model(shuffled)
    for key in (
        "contact_logits",
        "contact_interaction_logits",
        "paratope_logits",
        "epitope_logits",
    ):
        torch.testing.assert_close(before[key], after[key], atol=0.0, rtol=0.0)

    partner_swapped = {key: value for key, value in batch.items()}
    for key in (
        "antigen_features",
        "antigen_backbone",
        "antigen_mask",
        "antigen_chain_ids",
        "antigen_frame_valid_mask",
        "antigen_missing_frame_mask",
    ):
        partner_swapped[key] = batch[key].flip(0)
    swapped = model(partner_swapped)
    assert not torch.equal(before["contact_logits"], swapped["contact_logits"])
    assert not torch.equal(before["epitope_logits"], swapped["epitope_logits"])


def test_contact_logits_control_cross_attention_weights() -> None:
    import torch

    from interaction_first.framecontact import ContactGatedCrossAttention

    torch.manual_seed(47)
    layer = ContactGatedCrossAttention(16, 4, dropout=0.0).eval()
    receiver = torch.randn(1, 2, 16)
    partner = torch.randn(1, 3, 16)
    receiver_mask = torch.ones(1, 2, dtype=torch.bool)
    partner_mask = torch.ones(1, 3, dtype=torch.bool)
    pair_mask = torch.ones(1, 2, 3, dtype=torch.bool)
    neutral = torch.zeros(1, 2, 3)
    favored = neutral.clone()
    favored[:, :, 1] = 8.0
    _, neutral_details = layer(
        receiver,
        partner,
        receiver_mask,
        partner_mask,
        neutral,
        pair_mask,
        return_attention=True,
    )
    _, favored_details = layer(
        receiver,
        partner,
        receiver_mask,
        partner_mask,
        favored,
        pair_mask,
        return_attention=True,
    )
    assert neutral_details is not None and favored_details is not None
    assert bool(
        (
            favored_details["weights"][..., 1]
            > neutral_details["weights"][..., 1]
        ).all()
    )


def test_sdpa_cross_attention_matches_reference_outputs_and_gradients() -> None:
    import torch

    from interaction_first.framecontact import ContactGatedCrossAttention

    torch.manual_seed(61)
    reference = ContactGatedCrossAttention(
        16, 4, dropout=0.0, attention_kernel="reference"
    ).eval()
    sdpa = ContactGatedCrossAttention(
        16, 4, dropout=0.0, attention_kernel="sdpa"
    ).eval()
    with torch.no_grad():
        reference.residual_gate.fill_(0.6)
        reference.contact_scale_unconstrained.normal_(std=0.2)
        sdpa.load_state_dict(reference.state_dict())
    receiver_mask = torch.tensor([[True, True, False]])
    partner_mask = torch.tensor([[True, True, True, False]])
    pair_mask = receiver_mask[:, :, None] & partner_mask[:, None, :]
    contact_logits = torch.randn(1, 3, 4)
    reference_receiver = torch.randn(1, 3, 16, requires_grad=True)
    reference_partner = torch.randn(1, 4, 16, requires_grad=True)
    reference_contact = contact_logits.detach().clone().requires_grad_(True)
    sdpa_receiver = reference_receiver.detach().clone().requires_grad_(True)
    sdpa_partner = reference_partner.detach().clone().requires_grad_(True)
    sdpa_contact = contact_logits.detach().clone().requires_grad_(True)
    reference_output, _ = reference(
        reference_receiver,
        reference_partner,
        receiver_mask,
        partner_mask,
        reference_contact,
        pair_mask,
    )
    sdpa_output, _ = sdpa(
        sdpa_receiver,
        sdpa_partner,
        receiver_mask,
        partner_mask,
        sdpa_contact,
        pair_mask,
    )
    torch.testing.assert_close(reference_output, sdpa_output, atol=2e-5, rtol=2e-5)
    reference_output.square().sum().backward()
    sdpa_output.square().sum().backward()
    for reference_gradient, sdpa_gradient in (
        (reference_receiver.grad, sdpa_receiver.grad),
        (reference_partner.grad, sdpa_partner.grad),
        (reference_contact.grad, sdpa_contact.grad),
    ):
        torch.testing.assert_close(
            reference_gradient, sdpa_gradient, atol=4e-5, rtol=4e-5
        )
    for (reference_name, reference_parameter), (sdpa_name, sdpa_parameter) in zip(
        reference.named_parameters(), sdpa.named_parameters(), strict=True
    ):
        assert reference_name == sdpa_name
        assert reference_parameter.grad is not None and sdpa_parameter.grad is not None
        torch.testing.assert_close(
            reference_parameter.grad,
            sdpa_parameter.grad,
            atol=5e-5,
            rtol=5e-5,
        )

    diagnostic_output, diagnostic = sdpa(
        sdpa_receiver.detach(),
        sdpa_partner.detach(),
        receiver_mask,
        partner_mask,
        sdpa_contact.detach(),
        pair_mask,
        return_attention=True,
    )
    torch.testing.assert_close(reference_output.detach(), diagnostic_output)
    assert diagnostic is not None
    assert diagnostic["weights"].shape == (1, 4, 3, 4)


def test_unknown_cross_attention_kernel_fails_closed() -> None:
    from interaction_first.framecontact import ContactGatedCrossAttention

    with pytest.raises(ValueError, match="attention_kernel"):
        ContactGatedCrossAttention(16, 4, attention_kernel="opaque_extension")


def test_pair_loss_reaches_contact_head_at_zero_residual_gates() -> None:
    import torch

    model = _model().train()
    batch = _batch()
    output = model(batch)
    pair_mask = output["pair_candidate_mask"]
    labels = torch.zeros_like(output["contact_logits"])
    labels[:, 0, 0] = 1.0
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        output["contact_logits"][pair_mask], labels[pair_mask]
    )
    loss.backward()
    gradient = model.contact_head.antibody_query.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert float(gradient.abs().sum()) > 0.0


def test_degree_matched_rewiring_is_consumed_only_by_geometry_path() -> None:
    import torch

    from interaction_first.framecontact_interventions import (
        build_rewired_neighbor_override,
    )

    model = _model()
    _activate_geometry(model)
    batch = _batch(batch_size=1)
    original = model(batch, return_attention=True)
    override = build_rewired_neighbor_override(
        batch["antigen_backbone"],
        batch["antigen_mask"],
        batch["antigen_chain_ids"],
        generator=torch.Generator().manual_seed(53),
        neighbor_k=4,
        neighbor_radius=None,
    )
    rewired_batch = dict(batch)
    rewired_batch["antigen_neighbor_mask_override"] = override
    rewired = model(rewired_batch, return_attention=True)
    original_geometry = original["self_attention_details"]["antigen"]["geometry"]
    rewired_geometry = rewired["self_attention_details"]["antigen"]["geometry"]
    assert torch.equal(rewired_geometry["neighbor_mask"], override)
    assert not torch.allclose(original["contact_logits"], rewired["contact_logits"])


def test_degree_matched_graph_rewiring_builds_both_partner_overrides() -> None:
    import torch

    from interaction_first.framecontact_interventions import (
        degree_matched_graph_rewiring,
    )

    batch = _batch(batch_size=1)
    rewired = degree_matched_graph_rewiring(
        batch,
        generator=torch.Generator().manual_seed(59),
        neighbor_k=4,
        neighbor_radius=None,
    )
    assert rewired["intervention"] == "degree_matched_graph_rewiring"
    for partner in ("antibody", "antigen"):
        override = rewired[f"{partner}_neighbor_mask_override"]
        assert override.dtype == torch.bool
        assert override.shape == (
            1,
            batch[f"{partner}_mask"].shape[1],
            batch[f"{partner}_mask"].shape[1],
        )
        assert bool(torch.diagonal(override, dim1=-2, dim2=-1)[
            batch[f"{partner}_mask"]
        ].all())


def test_partner_shuffle_invalidates_pair_targets_but_preserves_partner_labels() -> None:
    import torch

    from interaction_first.framecontact_interventions import partner_shuffle

    batch = _batch()
    batch["pair_labels"] = torch.zeros(2, 5, 7)
    batch["pair_candidate_mask"] = (
        batch["antibody_mask"][:, :, None] & batch["antigen_mask"][:, None, :]
    )
    batch["antigen_labels"] = torch.arange(14).view(2, 7)
    shuffled = partner_shuffle(
        batch, partner="antigen", permutation=torch.tensor([1, 0])
    )
    assert "pair_labels" not in shuffled
    assert "pair_candidate_mask" not in shuffled
    assert torch.equal(shuffled["antigen_labels"], batch["antigen_labels"].flip(0))
    assert torch.equal(shuffled["antibody_features"], batch["antibody_features"])


def test_residue_output_ablation_paths_are_exact_and_explicit() -> None:
    import torch

    model = _model()
    batch = _batch()
    model.contact_marginalization = "noisy_or"
    model.residue_output_mode = "unary_only"
    unary = model(batch)
    torch.testing.assert_close(unary["paratope_logits"], unary["paratope_unary_logits"])
    torch.testing.assert_close(unary["epitope_logits"], unary["epitope_unary_logits"])

    model.residue_output_mode = "contact_marginal_only"
    contact = model(batch)
    torch.testing.assert_close(
        contact["paratope_logits"], contact["paratope_contact_logits"]
    )
    torch.testing.assert_close(
        contact["epitope_logits"], contact["epitope_contact_logits"]
    )

    model.residue_output_mode = "gated_hybrid"
    with torch.no_grad():
        model.paratope_contact_gate.fill_(0.4)
        model.epitope_contact_gate.fill_(-0.3)
    hybrid = model(batch)
    torch.testing.assert_close(
        hybrid["paratope_logits"],
        hybrid["paratope_unary_logits"]
        + torch.tanh(model.paratope_contact_gate)
        * hybrid["paratope_contact_logits"],
    )
    torch.testing.assert_close(
        hybrid["epitope_logits"],
        hybrid["epitope_unary_logits"]
        + torch.tanh(model.epitope_contact_gate) * hybrid["epitope_contact_logits"],
    )


def test_noisy_or_is_the_probability_of_at_least_one_contact() -> None:
    import torch

    from interaction_first.framecontact import masked_noisy_or_logits

    probabilities = torch.tensor([[[0.2, 0.3, 0.7]]])
    logits = torch.logit(probabilities)
    mask = torch.tensor([[[True, True, False]]])
    result = masked_noisy_or_logits(logits, mask, dim=2)
    expected_probability = 1.0 - (1.0 - 0.2) * (1.0 - 0.3)
    torch.testing.assert_close(
        torch.sigmoid(result), torch.tensor([[expected_probability]])
    )


def test_length_normalized_noisy_or_is_constant_under_partner_duplication() -> None:
    import torch

    from interaction_first.framecontact import (
        masked_length_normalized_noisy_or_logits,
        masked_noisy_or_logits,
    )

    one = torch.tensor([[[0.35]]])
    many = one.expand(1, 1, 12)
    one_mask = torch.ones_like(one, dtype=torch.bool)
    many_mask = torch.ones_like(many, dtype=torch.bool)
    normalized_one = masked_length_normalized_noisy_or_logits(
        torch.logit(one), one_mask, dim=2
    )
    normalized_many = masked_length_normalized_noisy_or_logits(
        torch.logit(many), many_mask, dim=2
    )
    torch.testing.assert_close(normalized_one, normalized_many)
    assert torch.sigmoid(
        masked_noisy_or_logits(torch.logit(many), many_mask, dim=2)
    ).item() > torch.sigmoid(normalized_many).item()


def test_distogram_contact_is_a_projection_of_the_same_pair_head() -> None:
    import torch

    from interaction_first.framecontact import (
        FrameContactModel,
        contact_logit_from_distogram,
    )

    model = FrameContactModel(
        antibody_input_dim=12,
        antigen_input_dim=10,
        d_model=16,
        num_heads=4,
        num_self_layers=1,
        neighbor_k=4,
        neighbor_radius=None,
        dropout=0.0,
        pair_output_mode="distogram",
        residue_output_mode="contact_marginal_only",
        contact_marginalization="noisy_or",
    ).eval()
    result = model(_batch())
    assert result["distogram_logits"].shape == (2, 5, 7, 12)
    assert result["distance_bin_boundaries"].shape == (11,)
    reconstructed = contact_logit_from_distogram(
        result["distogram_logits"], contact_bin_count=4
    ).masked_fill(~result["pair_candidate_mask"], 0.0)
    torch.testing.assert_close(result["contact_logits"], reconstructed)
    assert result["pair_output_mode"] == "distogram"
