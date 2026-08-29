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
    for mode in ("none", "distance", "distance_direction", "full_pose"):
        model = IntraChainGeometryEncoder(
            16,
            4,
            1,
            geometry_feature_mode=mode,
            num_rbf=8,
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


def test_unknown_attention_kernel_fails_closed() -> None:
    from interaction_first.intra_chain_geometry import IntraChainGeometryEncoder

    with pytest.raises(ValueError, match="attention_kernel"):
        IntraChainGeometryEncoder(16, 4, 1, attention_kernel="mystery_fusion")


def test_sdpa_self_attention_matches_reference_outputs_and_gradients() -> None:
    import torch

    from interaction_first.intra_chain_geometry import (
        GeometryBiasedSelfAttentionBlock,
        build_backbone_frames,
        build_intra_chain_relative_geometry,
    )

    torch.manual_seed(37)
    mask = torch.tensor([[True, True, True, True, True, False]])
    backbone = _sample_backbone(1, 6)
    backbone[:, -1] = torch.nan
    geometry = build_intra_chain_relative_geometry(
        build_backbone_frames(backbone, mask),
        mask,
        neighbor_k=3,
    )
    reference = GeometryBiasedSelfAttentionBlock(
        16,
        4,
        geometry.features.shape[-1],
        dropout=0.0,
        attention_kernel="reference",
    ).eval()
    sdpa = GeometryBiasedSelfAttentionBlock(
        16,
        4,
        geometry.features.shape[-1],
        dropout=0.0,
        attention_kernel="sdpa",
    ).eval()
    sdpa.load_state_dict(reference.state_dict())
    with torch.no_grad():
        reference.geometry_bias.network[-1].weight.normal_(std=0.05)
        reference.geometry_bias.network[-1].bias.normal_(std=0.05)
        sdpa.load_state_dict(reference.state_dict())
    reference_features = torch.randn(1, 6, 16, requires_grad=True)
    sdpa_features = reference_features.detach().clone().requires_grad_(True)
    reference_output, _ = reference(
        reference_features, geometry, mask, return_attention=False
    )
    sdpa_output, _ = sdpa(sdpa_features, geometry, mask, return_attention=False)
    torch.testing.assert_close(reference_output, sdpa_output, atol=2e-5, rtol=2e-5)
    reference_output.square().sum().backward()
    sdpa_output.square().sum().backward()
    torch.testing.assert_close(
        reference_features.grad, sdpa_features.grad, atol=3e-5, rtol=3e-5
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

    # Attention diagnostics are defined by the explicit reference path even
    # when the fast inference/training path is selected.
    diagnostic_output, diagnostic = sdpa(
        sdpa_features.detach(), geometry, mask, return_attention=True
    )
    torch.testing.assert_close(reference_output.detach(), diagnostic_output)
    assert diagnostic is not None and diagnostic["weights"].shape == (1, 4, 6, 6)


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


def test_invariant_point_bias_is_se3_invariant_and_frame_sensitive() -> None:
    import torch

    from interaction_first.intra_chain_geometry import (
        InvariantPointHeadBias,
        build_backbone_frames,
        build_intra_chain_relative_geometry,
    )

    torch.manual_seed(53)
    backbone = _sample_backbone(2, 7)
    mask = torch.ones(2, 7, dtype=torch.bool)
    geometry = build_intra_chain_relative_geometry(
        build_backbone_frames(backbone, mask),
        mask,
        geometry_feature_mode="none",
        neighbor_k=None,
    )
    layer = InvariantPointHeadBias(4, num_points=3).eval()
    before = layer(geometry)
    transformed_backbone = _rigid_transform(
        backbone,
        _random_rotation(2),
        torch.tensor([[31.0, -8.0, 5.0], [-4.0, 13.0, 70.0]]),
    )
    transformed_geometry = build_intra_chain_relative_geometry(
        build_backbone_frames(transformed_backbone, mask),
        mask,
        geometry_feature_mode="none",
        neighbor_k=None,
    )
    torch.testing.assert_close(
        before,
        layer(transformed_geometry),
        atol=2e-5,
        rtol=2e-5,
    )

    corrupted_backbone = backbone.clone()
    corrupted_backbone[:, 3, 0] += torch.tensor([0.0, 1.5, -0.5])
    corrupted_geometry = build_intra_chain_relative_geometry(
        build_backbone_frames(corrupted_backbone, mask),
        mask,
        geometry_feature_mode="none",
        neighbor_k=None,
    )
    assert not torch.allclose(before, layer(corrupted_geometry))


def test_invariant_point_memory_rewrite_matches_naive_value_and_gradients() -> None:
    import torch

    from interaction_first.intra_chain_geometry import (
        InvariantPointHeadBias,
        build_backbone_frames,
        build_intra_chain_relative_geometry,
    )

    torch.manual_seed(59)
    backbone = _sample_backbone(1, 6)
    mask = torch.ones(1, 6, dtype=torch.bool)
    geometry = build_intra_chain_relative_geometry(
        build_backbone_frames(backbone, mask),
        mask,
        geometry_feature_mode="none",
        neighbor_k=None,
    )
    layer = InvariantPointHeadBias(3, num_points=4)
    optimized = layer(geometry)
    relative_key = torch.einsum(
        "bijxy,hpy->bijhpx",
        geometry.relative_rotation,
        layer.key_points,
    )
    relative_key = relative_key + geometry.displacement_local[..., None, None, :]
    delta = layer.query_points[None, None, None, ...] - relative_key
    naive_squared_distance = delta.square().sum(dim=-1).mean(dim=-1)
    head_weight = torch.nn.functional.softplus(layer.head_weight_unconstrained)
    naive = (
        -0.5
        * naive_squared_distance
        * head_weight[None, None, None, :]
        / (layer.distance_scale**2)
    )
    naive = naive * geometry.geometry_valid_mask[..., None].to(naive.dtype)
    naive = naive.movedim(-1, 1)
    torch.testing.assert_close(optimized, naive, atol=3e-6, rtol=3e-6)

    probe = torch.randn_like(optimized)
    (optimized * probe).sum().backward(retain_graph=True)
    optimized_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in layer.named_parameters()
    }
    layer.zero_grad(set_to_none=True)
    (naive * probe).sum().backward()
    for name, parameter in layer.named_parameters():
        torch.testing.assert_close(
            optimized_gradients[name], parameter.grad, atol=2e-5, rtol=2e-5
        )


def test_invariant_point_bias_masks_missing_frames_and_runs_in_encoder() -> None:
    import torch

    from interaction_first.intra_chain_geometry import IntraChainGeometryEncoder

    torch.manual_seed(59)
    backbone = _sample_backbone(1, 6)
    backbone[:, 4] = torch.nan
    mask = torch.ones(1, 6, dtype=torch.bool)
    features = torch.randn(1, 6, 16)
    encoder = IntraChainGeometryEncoder(
        16,
        4,
        1,
        geometry_feature_mode="none",
        geometry_bias_mode="invariant_points",
        attention_graph_mode="spatial",
        neighbor_k=4,
        dropout=0.0,
    ).eval()
    output, details = encoder(
        features,
        mask,
        backbone,
        return_attention=True,
    )
    assert details is not None
    point_bias = details["layers"][0]["geometry_bias"]
    assert torch.isfinite(output).all()
    assert torch.isfinite(point_bias).all()
    assert torch.equal(point_bias[:, :, 4], torch.zeros_like(point_bias[:, :, 4]))
    assert torch.equal(point_bias[:, :, :, 4], torch.zeros_like(point_bias[:, :, :, 4]))
    assert not torch.equal(point_bias[:, :, :4, :4], torch.zeros_like(point_bias[:, :, :4, :4]))


def test_unknown_geometry_bias_mode_fails_closed() -> None:
    from interaction_first.intra_chain_geometry import IntraChainGeometryEncoder

    with pytest.raises(ValueError, match="geometry_bias_mode"):
        IntraChainGeometryEncoder(
            16,
            4,
            1,
            geometry_bias_mode="raw_xyz_points",
        )


@pytest.mark.parametrize(
    "bias_mode", ("local_pose_multibasis", "local_pose_rotary")
)
def test_local_pose_bias_modes_are_safe_start_invariant_and_missing_frame_safe(
    bias_mode: str,
) -> None:
    import torch

    from interaction_first.intra_chain_geometry import IntraChainGeometryEncoder

    torch.manual_seed(61)
    features = torch.randn(1, 7, 16)
    mask = torch.ones(1, 7, dtype=torch.bool)
    backbone = _sample_backbone(1, 7)
    backbone[:, 5] = torch.nan
    encoder = IntraChainGeometryEncoder(
        16,
        4,
        1,
        geometry_feature_mode="full_pose",
        geometry_bias_mode=bias_mode,
        attention_graph_mode="dense",
        num_rbf=8,
        dropout=0.0,
    ).eval()

    initial_output, initial_details = encoder(
        features,
        mask,
        backbone,
        return_attention=True,
    )
    assert initial_details is not None
    initial_bias = initial_details["layers"][0]["geometry_bias"]
    assert torch.equal(initial_bias, torch.zeros_like(initial_bias))

    with torch.no_grad():
        encoder.blocks[0].geometry_bias.gate_unconstrained.fill_(0.45)
    active_output, active_details = encoder(
        features,
        mask,
        backbone,
        return_attention=True,
    )
    assert active_details is not None
    active_bias = active_details["layers"][0]["geometry_bias"]
    assert torch.isfinite(active_output).all()
    assert not torch.equal(active_bias[:, :, :5, :5], torch.zeros_like(active_bias[:, :, :5, :5]))
    assert torch.equal(active_bias[:, :, 5], torch.zeros_like(active_bias[:, :, 5]))
    assert torch.equal(active_bias[:, :, :, 5], torch.zeros_like(active_bias[:, :, :, 5]))
    assert not torch.allclose(initial_output, active_output)

    transformed = _rigid_transform(
        backbone,
        _random_rotation(1),
        torch.tensor([[17.0, -29.0, 8.0]]),
    )
    transformed_output, transformed_details = encoder(
        features,
        mask,
        transformed,
        return_attention=True,
    )
    assert transformed_details is not None
    torch.testing.assert_close(active_output, transformed_output, atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(
        active_bias,
        transformed_details["layers"][0]["geometry_bias"],
        atol=3e-5,
        rtol=3e-5,
    )

    if bias_mode == "local_pose_multibasis":
        components = active_details["layers"][0]["geometry_bias_components"]
        assert tuple(components) == ("radial", "direction", "rotation")
        torch.testing.assert_close(sum(components.values()), active_bias)
    else:
        assert active_details["layers"][0]["geometry_rotary_channels"] == 4
        assert torch.all(active_details["layers"][0]["geometry_rotary_gate"] != 0)


@pytest.mark.parametrize(
    "bias_mode", ("local_pose_multibasis", "local_pose_rotary")
)
def test_local_pose_bias_parameter_count_is_feature_ablation_matched(
    bias_mode: str,
) -> None:
    from interaction_first.intra_chain_geometry import IntraChainGeometryEncoder

    counts = []
    for feature_mode in ("none", "distance", "distance_direction", "full_pose"):
        encoder = IntraChainGeometryEncoder(
            16,
            4,
            1,
            geometry_feature_mode=feature_mode,
            geometry_bias_mode=bias_mode,
            num_rbf=8,
        )
        counts.append(sum(parameter.numel() for parameter in encoder.parameters()))
    assert len(set(counts)) == 1


def test_local_pose_multibasis_has_staged_nonzero_gate_and_branch_gradients() -> None:
    import torch

    from interaction_first.intra_chain_geometry import (
        LocalPoseMultiBasisHeadBias,
        build_backbone_frames,
        build_intra_chain_relative_geometry,
    )

    torch.manual_seed(63)
    mask = torch.ones(1, 5, dtype=torch.bool)
    geometry = build_intra_chain_relative_geometry(
        build_backbone_frames(_sample_backbone(1, 5), mask),
        mask,
        num_rbf=8,
        neighbor_k=None,
    )
    layer = LocalPoseMultiBasisHeadBias(22, 4, num_rbf=8)
    weights = torch.randn(1, 4, 5, 5)
    bias = layer(geometry.features, geometry.geometry_valid_mask)
    (bias * weights).sum().backward()
    assert layer.gate_unconstrained.grad is not None
    assert bool((layer.gate_unconstrained.grad.abs() > 0).any())

    layer.zero_grad(set_to_none=True)
    with torch.no_grad():
        layer.gate_unconstrained.fill_(0.3)
    (layer(geometry.features, geometry.geometry_valid_mask) * weights).sum().backward()
    final_weight = layer.component_networks["rotation"][-1].weight
    assert final_weight.grad is not None
    assert bool((final_weight.grad.abs() > 0).any())


def test_local_pose_rotary_has_staged_nonzero_gate_and_phase_gradients() -> None:
    import torch

    from interaction_first.intra_chain_geometry import (
        PairConditionedLocalPoseRotaryBias,
        build_backbone_frames,
        build_intra_chain_relative_geometry,
    )

    torch.manual_seed(65)
    mask = torch.ones(1, 5, dtype=torch.bool)
    geometry = build_intra_chain_relative_geometry(
        build_backbone_frames(_sample_backbone(1, 5), mask),
        mask,
        num_rbf=8,
        neighbor_k=None,
    )
    layer = PairConditionedLocalPoseRotaryBias(
        22,
        4,
        4,
        rotary_channels=4,
    )
    query = torch.randn(1, 4, 5, 4)
    key = torch.randn(1, 4, 5, 4)
    weights = torch.randn(1, 4, 5, 5)
    bias = layer(
        query,
        key,
        geometry.features,
        geometry.geometry_valid_mask,
    )
    assert torch.equal(bias, torch.zeros_like(bias))
    (bias * weights).sum().backward()
    assert layer.gate_unconstrained.grad is not None
    assert bool((layer.gate_unconstrained.grad.abs() > 0).any())

    layer.zero_grad(set_to_none=True)
    with torch.no_grad():
        layer.gate_unconstrained.fill_(0.3)
    active = layer(
        query,
        key,
        geometry.features,
        geometry.geometry_valid_mask,
    )
    (active * weights).sum().backward()
    final_weight = layer.phase_network[-1].weight
    assert final_weight.grad is not None
    assert bool((final_weight.grad.abs() > 0).any())


def test_local_pose_rotary_channel_contract_fails_closed() -> None:
    from interaction_first.intra_chain_geometry import (
        PairConditionedLocalPoseRotaryBias,
    )

    with pytest.raises(ValueError, match="rotary_channels"):
        PairConditionedLocalPoseRotaryBias(22, 4, 4, rotary_channels=3)
    with pytest.raises(ValueError, match="rotary_channels"):
        PairConditionedLocalPoseRotaryBias(22, 4, 4, rotary_channels=6)


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


def test_attention_graph_factorization_separates_sequence_and_spatial_edges() -> None:
    import torch

    from interaction_first.intra_chain_geometry import (
        build_backbone_frames,
        build_intra_chain_relative_geometry,
    )

    mask = torch.ones(1, 6, dtype=torch.bool)
    chain_ids = torch.tensor([[0, 0, 0, 0, 1, 1]])
    backbone = _sample_backbone(1, 6)
    backbone[0, 2] = torch.nan
    frames = build_backbone_frames(backbone, mask)

    dense = build_intra_chain_relative_geometry(
        frames,
        mask,
        chain_ids=chain_ids,
        geometry_feature_mode="none",
        attention_graph_mode="dense",
    )
    sequence = build_intra_chain_relative_geometry(
        frames,
        mask,
        chain_ids=chain_ids,
        geometry_feature_mode="none",
        attention_graph_mode="sequence",
        sequence_window=1,
    )
    spatial = build_intra_chain_relative_geometry(
        frames,
        mask,
        chain_ids=chain_ids,
        geometry_feature_mode="none",
        attention_graph_mode="spatial",
        neighbor_k=2,
        neighbor_radius=None,
    )

    assert dense.neighbor_mask[0, :4, :4].all()
    assert dense.neighbor_mask[0, 4:, 4:].all()
    assert not dense.neighbor_mask[0, :4, 4:].any()
    assert sequence.neighbor_mask[0, 0, 1]
    assert not sequence.neighbor_mask[0, 0, 2]
    assert sequence.neighbor_mask[0, 2, 1]
    assert sequence.neighbor_mask[0, 2, 3]
    assert int(spatial.neighbor_mask[0, 2].sum()) == 1
    assert spatial.neighbor_mask[0, 2, 2]


def test_dense_sequence_only_attention_is_coordinate_independent() -> None:
    import torch

    from interaction_first.intra_chain_geometry import IntraChainGeometryEncoder

    torch.manual_seed(79)
    encoder = IntraChainGeometryEncoder(
        16,
        4,
        1,
        geometry_feature_mode="none",
        attention_graph_mode="dense",
        dropout=0.0,
    ).eval()
    features = torch.randn(1, 7, 16)
    mask = torch.ones(1, 7, dtype=torch.bool)
    first = _sample_backbone(1, 7)
    second = torch.randn_like(first) * 100.0
    first_output, first_details = encoder(
        features, mask, first, return_attention=True
    )
    second_output, second_details = encoder(
        features, mask, second, return_attention=True
    )
    torch.testing.assert_close(first_output, second_output, atol=0.0, rtol=0.0)
    assert first_details is not None and second_details is not None
    torch.testing.assert_close(
        first_details["layers"][0]["weights"],
        second_details["layers"][0]["weights"],
        atol=0.0,
        rtol=0.0,
    )


def test_attention_diagnostics_are_head_resolved_and_finite() -> None:
    import torch

    from interaction_first.attention_diagnostics import AttentionDiagnosticsAccumulator
    from interaction_first.intra_chain_geometry import IntraChainGeometryEncoder

    torch.manual_seed(83)
    encoder = IntraChainGeometryEncoder(
        16,
        4,
        1,
        geometry_feature_mode="full_pose",
        attention_graph_mode="spatial",
        neighbor_k=4,
        dropout=0.0,
    ).eval()
    with torch.no_grad():
        encoder.blocks[0].geometry_bias.network[-1].weight.normal_(std=0.05)
    features = torch.randn(2, 7, 16)
    mask = torch.tensor(
        [[True] * 7, [True, True, True, True, True, False, False]]
    )
    _, details = encoder(
        features,
        mask,
        _sample_backbone(2, 7),
        return_attention=True,
    )
    assert details is not None
    accumulator = AttentionDiagnosticsAccumulator()
    accumulator.update("antibody", details, mask)
    result = accumulator.result()
    layer = result["partners"]["antibody"]["layers"][0]
    assert layer["head_count"] == 4
    metrics = layer["metrics_per_head"]
    assert len(metrics["mean_normalized_entropy"]) == 4
    assert all(
        value is not None and torch.isfinite(torch.tensor(value))
        for values in metrics.values()
        for value in values
    )
    assert any(
        value > 0
        for value in metrics["mean_geometry_counterfactual_total_variation"]
    )
