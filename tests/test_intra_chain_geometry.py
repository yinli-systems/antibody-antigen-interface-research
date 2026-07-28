"""Tests for the phase-one intra-chain structure-aware attention stage."""

from __future__ import annotations

import importlib.util

import pytest


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="PyTorch is unavailable",
)


def _sample_backbone(batch_size: int, residue_count: int):
    import torch

    ca_atom = torch.cumsum(torch.randn(batch_size, residue_count, 3), dim=1)
    n_atom = ca_atom + torch.tensor([-1.1, 0.8, 0.2])
    c_atom = ca_atom + torch.tensor([1.3, 0.2, -0.1])
    return torch.stack((n_atom, ca_atom, c_atom), dim=-2)


def _random_rotation(batch_size: int):
    import torch

    matrix = torch.randn(batch_size, 3, 3)
    rotation, _ = torch.linalg.qr(matrix)
    determinant = torch.linalg.det(rotation)
    rotation = rotation.clone()
    rotation[:, :, -1] *= torch.where(
        determinant < 0.0,
        -torch.ones_like(determinant),
        torch.ones_like(determinant),
    )[:, None]
    return rotation


def _rigid_transform(backbone, rotation, translation):
    import torch

    return (
        torch.einsum("bij,brkj->brki", rotation, backbone)
        + translation[:, None, None, :]
    )


def _activate_geometry_bias(encoder) -> None:
    import torch

    with torch.no_grad():
        for block in encoder.blocks:
            final = block.geometry_bias.network[-1]
            final.weight.normal_(mean=0.0, std=0.1)
            final.bias.normal_(mean=0.0, std=0.1)


def test_relative_pose_and_dual_outputs_are_independently_se3_invariant() -> None:
    import torch

    from interaction_first.intra_chain_geometry import (
        DualIntraChainGeometryEncoder,
        build_backbone_frames,
        build_intra_chain_relative_geometry,
    )

    torch.manual_seed(20260728)
    antibody_features = torch.randn(2, 7, 16)
    antigen_features = torch.randn(2, 9, 16)
    antibody_mask = torch.ones(2, 7, dtype=torch.bool)
    antigen_mask = torch.ones(2, 9, dtype=torch.bool)
    antibody_backbone = _sample_backbone(2, 7)
    antigen_backbone = _sample_backbone(2, 9)
    encoder = DualIntraChainGeometryEncoder(
        16,
        4,
        2,
        neighbor_k=4,
        dropout=0.0,
    ).eval()
    _activate_geometry_bias(encoder.antibody_encoder)
    _activate_geometry_bias(encoder.antigen_encoder)

    antibody_before, antigen_before, _ = encoder(
        antibody_features,
        antigen_features,
        antibody_mask,
        antigen_mask,
        antibody_backbone,
        antigen_backbone,
    )
    antibody_rotation = _random_rotation(2)
    antigen_rotation = _random_rotation(2)
    antibody_after, antigen_after, _ = encoder(
        antibody_features,
        antigen_features,
        antibody_mask,
        antigen_mask,
        _rigid_transform(
            antibody_backbone,
            antibody_rotation,
            torch.tensor([[31.0, -8.0, 5.0], [-4.0, 13.0, 70.0]]),
        ),
        _rigid_transform(
            antigen_backbone,
            antigen_rotation,
            torch.tensor([[-55.0, 2.0, 9.0], [18.0, -27.0, 3.0]]),
        ),
    )
    torch.testing.assert_close(antibody_before, antibody_after, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(antigen_before, antigen_after, atol=2e-5, rtol=2e-5)

    frames_before = build_backbone_frames(antibody_backbone, antibody_mask)
    frames_after = build_backbone_frames(
        _rigid_transform(
            antibody_backbone,
            antibody_rotation,
            torch.tensor([[31.0, -8.0, 5.0], [-4.0, 13.0, 70.0]]),
        ),
        antibody_mask,
    )
    pose_before = build_intra_chain_relative_geometry(
        frames_before, antibody_mask, neighbor_k=4
    )
    pose_after = build_intra_chain_relative_geometry(
        frames_after, antibody_mask, neighbor_k=4
    )
    torch.testing.assert_close(
        pose_before.features,
        pose_after.features,
        atol=2e-5,
        rtol=2e-5,
    )
    assert torch.equal(pose_before.neighbor_mask, pose_after.neighbor_mask)


def test_neighbor_graph_is_spatial_intra_chain_and_robust_to_missing_frames() -> None:
    import torch

    from interaction_first.intra_chain_geometry import (
        build_backbone_frames,
        build_intra_chain_relative_geometry,
    )

    ca_atom = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [10.0, 0.0, 0.0], [11.0, 0.0, 0.0]]]
    )
    n_atom = ca_atom + torch.tensor([0.0, 1.0, 0.0])
    c_atom = ca_atom + torch.tensor([1.0, 0.0, 0.0])
    backbone = torch.stack((n_atom, ca_atom, c_atom), dim=-2)
    backbone[0, 3] = torch.nan
    residue_mask = torch.ones(1, 4, dtype=torch.bool)
    chain_ids = torch.tensor([[0, 0, 1, 1]])
    geometry = build_intra_chain_relative_geometry(
        build_backbone_frames(backbone, residue_mask),
        residue_mask,
        chain_ids=chain_ids,
        neighbor_k=2,
        neighbor_radius=3.0,
    )

    assert geometry.neighbor_mask[0, 0, 1]
    assert geometry.neighbor_mask[0, 1, 0]
    assert not bool(geometry.neighbor_mask[0, :2, 2:].any())
    assert not bool(geometry.neighbor_mask[0, 2:, :2].any())
    assert torch.equal(
        geometry.neighbor_mask[0, 3],
        torch.tensor([False, False, False, True]),
    )
    assert not geometry.geometry_valid_mask[0, 3, 3]
    assert torch.equal(
        geometry.features[0, 3],
        torch.zeros_like(geometry.features[0, 3]),
    )


def test_dense_sequence_control_does_not_depend_on_frames_or_coordinates() -> None:
    import torch

    from interaction_first.intra_chain_geometry import (
        build_backbone_frames,
        build_intra_chain_relative_geometry,
    )

    backbone = _sample_backbone(1, 6)
    backbone[0, 1] = torch.nan
    residue_mask = torch.tensor([[True, True, True, True, True, False]])
    chain_ids = torch.tensor([[0, 0, 0, 1, 1, -1]])
    geometry = build_intra_chain_relative_geometry(
        build_backbone_frames(backbone, residue_mask),
        residue_mask,
        chain_ids=chain_ids,
        geometry_feature_mode="none",
        neighbor_k=1,
        neighbor_radius=0.1,
        attention_topology="dense_sequence",
    )

    expected = torch.tensor(
        [
            [True, True, True, False, False, False],
            [True, True, True, False, False, False],
            [True, True, True, False, False, False],
            [False, False, False, True, True, False],
            [False, False, False, True, True, False],
            [False, False, False, False, False, False],
        ]
    )
    assert torch.equal(geometry.neighbor_mask[0], expected)
    assert torch.equal(geometry.features, torch.zeros_like(geometry.features))
    assert not geometry.geometry_valid_mask[0, 1].any()
    assert not geometry.geometry_valid_mask[0, :, 1].any()


@pytest.mark.parametrize(
    ("mode", "expected_channels"),
    (
        ("none", 22),
        ("distance", 22),
        ("distance_direction", 22),
        ("full_pose", 22),
    ),
)
def test_geometry_feature_modes_make_ablations_explicit(
    mode: str,
    expected_channels: int,
) -> None:
    import torch

    from interaction_first.intra_chain_geometry import (
        build_backbone_frames,
        build_intra_chain_relative_geometry,
    )

    mask = torch.ones(1, 5, dtype=torch.bool)
    geometry = build_intra_chain_relative_geometry(
        build_backbone_frames(_sample_backbone(1, 5), mask),
        mask,
        geometry_feature_mode=mode,
        num_rbf=8,
        neighbor_k=None,
    )
    assert geometry.features.shape == (1, 5, 5, expected_channels)
    if mode == "none":
        assert torch.equal(
            geometry.features,
            torch.zeros_like(geometry.features),
        )
    elif mode == "distance":
        assert torch.equal(
            geometry.features[..., 8:],
            torch.zeros_like(geometry.features[..., 8:]),
        )
    elif mode == "distance_direction":
        assert torch.equal(
            geometry.features[..., 11:],
            torch.zeros_like(geometry.features[..., 11:]),
        )


def test_geometry_ablation_modes_have_identical_parameter_counts() -> None:
    from interaction_first.intra_chain_geometry import IntraChainGeometryEncoder

    counts = []
    configurations = (
        ("none", "dense_sequence"),
        ("none", "spatial"),
        ("distance", "spatial"),
        ("distance_direction", "spatial"),
        ("full_pose", "spatial"),
    )
    for mode, topology in configurations:
        model = IntraChainGeometryEncoder(
            16,
            4,
            1,
            geometry_feature_mode=mode,
            num_rbf=8,
            attention_topology=topology,
        )
        counts.append(sum(parameter.numel() for parameter in model.parameters()))
    assert len(set(counts)) == 1


def test_unknown_geometry_feature_mode_fails_closed() -> None:
    from interaction_first.intra_chain_geometry import IntraChainGeometryEncoder

    with pytest.raises(ValueError, match="geometry_feature_mode"):
        IntraChainGeometryEncoder(
            16,
            4,
            1,
            geometry_feature_mode="raw_global_xyz",
        )


def test_unknown_attention_topology_fails_closed() -> None:
    from interaction_first.intra_chain_geometry import IntraChainGeometryEncoder

    with pytest.raises(ValueError, match="attention_topology"):
        IntraChainGeometryEncoder(
            16,
            4,
            1,
            attention_topology="coordinate_leaking_dense",
        )


def test_zero_initialized_geometry_projection_is_a_safe_start() -> None:
    import torch

    from interaction_first.intra_chain_geometry import (
        GeometryBiasedSelfAttentionBlock,
        build_backbone_frames,
        build_intra_chain_relative_geometry,
    )

    torch.manual_seed(41)
    features = torch.randn(1, 6, 16)
    mask = torch.ones(1, 6, dtype=torch.bool)
    geometry = build_intra_chain_relative_geometry(
        build_backbone_frames(_sample_backbone(1, 6), mask),
        mask,
        neighbor_k=None,
    )
    block = GeometryBiasedSelfAttentionBlock(
        d_model=16,
        num_heads=4,
        geometry_dim=geometry.features.shape[-1],
        dropout=0.0,
    ).eval()
    output, details = block(
        features,
        geometry,
        mask,
        return_attention=True,
    )
    assert torch.isfinite(output).all()
    assert details is not None
    assert torch.equal(
        details["geometry_bias"],
        torch.zeros_like(details["geometry_bias"]),
    )
    torch.testing.assert_close(
        details["weights"].sum(dim=-1),
        torch.ones_like(details["weights"].sum(dim=-1)),
    )

    with torch.no_grad():
        block.geometry_bias.network[-1].weight.fill_(0.05)
    changed, changed_details = block(
        features,
        geometry,
        mask,
        return_attention=True,
    )
    assert changed_details is not None
    assert not torch.equal(
        changed_details["geometry_bias"],
        torch.zeros_like(changed_details["geometry_bias"]),
    )
    assert not torch.allclose(output, changed)


def test_antibody_and_antigen_attention_parameters_are_not_shared() -> None:
    from interaction_first.intra_chain_geometry import DualIntraChainGeometryEncoder

    encoder = DualIntraChainGeometryEncoder(16, 4, 2, dropout=0.0)
    antibody_storage = {
        parameter.data_ptr() for parameter in encoder.antibody_encoder.parameters()
    }
    antigen_storage = {
        parameter.data_ptr() for parameter in encoder.antigen_encoder.parameters()
    }
    assert antibody_storage
    assert antigen_storage
    assert antibody_storage.isdisjoint(antigen_storage)
    assert (
        encoder.antibody_encoder.blocks[0].qkv
        is not encoder.antigen_encoder.blocks[0].qkv
    )
    assert (
        encoder.antibody_encoder.blocks[0].geometry_bias
        is not encoder.antigen_encoder.blocks[0].geometry_bias
    )


def test_model_integration_is_opt_in_and_backpropagates_through_both_branches() -> None:
    import torch

    from interaction_first.model import PartnerSpecificInteractionModel

    torch.manual_seed(53)
    legacy = PartnerSpecificInteractionModel(
        antigen_input_dim=8,
        antibody_input_dim=8,
        d_model=16,
        pair_dim=8,
        num_heads=4,
        num_layers=0,
        dropout=0.0,
    )
    assert legacy.intra_chain_geometry is None
    assert not any("intra_chain_geometry" in key for key in legacy.state_dict())
    antigen_features = torch.randn(2, 6, 8)
    antibody_features = torch.randn(2, 5, 8)
    with pytest.raises(ValueError, match="intra_chain_geometry_layers"):
        legacy(
            antigen_features,
            antibody_features,
            antigen_backbone=_sample_backbone(2, 6),
            antibody_backbone=_sample_backbone(2, 5),
        )

    model = PartnerSpecificInteractionModel(
        antigen_input_dim=8,
        antibody_input_dim=8,
        d_model=16,
        pair_dim=8,
        num_heads=4,
        num_layers=0,
        dropout=0.0,
        intra_chain_geometry_layers=2,
        intra_chain_geometry_neighbor_k=3,
    )
    with pytest.raises(ValueError, match="backbone"):
        model(antigen_features, antibody_features)
    output = model(
        antigen_features,
        antibody_features,
        antigen_backbone=_sample_backbone(2, 6),
        antibody_backbone=_sample_backbone(2, 5),
        return_intra_chain_attention=True,
    )
    assert output["intra_chain_geometry_enabled"] is True
    assert output["intra_chain_geometry"] is not None
    assert len(output["intra_chain_geometry"]["antibody"]["layers"]) == 2
    assert len(output["intra_chain_geometry"]["antigen"]["layers"]) == 2
    loss = output["pair_logits"].square().mean()
    loss.backward()
    antibody_final = (
        model.intra_chain_geometry
        .antibody_encoder.blocks[0]
        .geometry_bias.network[-1].weight
    )
    antigen_final = (
        model.intra_chain_geometry
        .antigen_encoder.blocks[0]
        .geometry_bias.network[-1].weight
    )
    assert antibody_final.grad is not None
    assert antigen_final.grad is not None
    assert torch.isfinite(antibody_final.grad).all()
    assert torch.isfinite(antigen_final.grad).all()


def test_padding_queries_remain_exactly_zero_and_attention_is_finite() -> None:
    import torch

    from interaction_first.intra_chain_geometry import IntraChainGeometryEncoder

    torch.manual_seed(67)
    features = torch.randn(2, 5, 16)
    mask = torch.tensor(
        [[True, True, True, True, True], [True, True, True, False, False]]
    )
    backbone = _sample_backbone(2, 5)
    encoder = IntraChainGeometryEncoder(
        16,
        4,
        1,
        neighbor_k=3,
        dropout=0.0,
    ).eval()
    output, details = encoder(
        features,
        mask,
        backbone,
        return_attention=True,
    )
    assert details is not None
    assert torch.isfinite(output).all()
    assert torch.isfinite(details["layers"][0]["weights"]).all()
    assert torch.equal(output[1, 3:], torch.zeros_like(output[1, 3:]))
    assert torch.equal(
        details["layers"][0]["weights"][1, :, 3:],
        torch.zeros_like(details["layers"][0]["weights"][1, :, 3:]),
    )
