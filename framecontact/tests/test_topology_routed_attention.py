from __future__ import annotations

import torch

from interaction_first.intra_chain_geometry import build_backbone_frames
from interaction_first.topology_routed_attention import (
    TopologyRoutedFrameContactModel,
    TopologyRoutedIntraChainEncoder,
    TopologyRoutedMultiScaleHeadBias,
)


def _backbone(batch_size: int, residue_count: int) -> torch.Tensor:
    ca = torch.cumsum(torch.randn(batch_size, residue_count, 3), dim=1)
    n = ca + torch.tensor([-1.1, 0.7, 0.2])
    c = ca + torch.tensor([1.2, 0.1, -0.3])
    return torch.stack((n, ca, c), dim=-2)


def _random_rotation(batch_size: int) -> torch.Tensor:
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


def _rigid_transform(
    backbone: torch.Tensor,
    rotation: torch.Tensor,
    translation: torch.Tensor,
) -> torch.Tensor:
    return (
        torch.einsum("bij,brkj->brki", rotation, backbone)
        + translation[:, None, None, :]
    )


def _batch() -> dict[str, torch.Tensor]:
    torch.manual_seed(20260731)
    antibody_mask = torch.tensor(
        [[True] * 5, [True, True, True, True, False]]
    )
    antigen_mask = torch.tensor(
        [[True] * 7, [True, True, True, True, True, True, False]]
    )
    antibody_backbone = _backbone(2, 5).masked_fill(
        ~antibody_mask[..., None, None], float("nan")
    )
    antigen_backbone = _backbone(2, 7).masked_fill(
        ~antigen_mask[..., None, None], float("nan")
    )
    antibody_valid = build_backbone_frames(
        antibody_backbone, antibody_mask
    ).valid_mask
    antigen_valid = build_backbone_frames(
        antigen_backbone, antigen_mask
    ).valid_mask
    return {
        "antibody_features": torch.randn(2, 5, 12).masked_fill(
            ~antibody_mask[..., None], 0.0
        ),
        "antigen_features": torch.randn(2, 7, 10).masked_fill(
            ~antigen_mask[..., None], 0.0
        ),
        "antibody_backbone": antibody_backbone,
        "antigen_backbone": antigen_backbone,
        "antibody_mask": antibody_mask,
        "antigen_mask": antigen_mask,
        "antibody_chain_ids": torch.tensor([[0, 0, 0, 1, 1]]).expand(
            2, -1
        ).clone(),
        "antigen_chain_ids": torch.zeros(2, 7, dtype=torch.long),
        "antibody_frame_valid_mask": antibody_valid,
        "antigen_frame_valid_mask": antigen_valid,
        "antibody_missing_frame_mask": antibody_mask & ~antibody_valid,
        "antigen_missing_frame_mask": antigen_mask & ~antigen_valid,
    }


def test_multiscale_bias_is_zero_initialized_scale_local_and_trainable() -> None:
    torch.manual_seed(89)
    module = TopologyRoutedMultiScaleHeadBias(30, 4, num_rbf=16)
    features = torch.randn(1, 2, 2, 30)
    distance = torch.tensor([[[0.0, 7.0], [7.0, 0.0]]])
    valid = torch.ones(1, 2, 2, dtype=torch.bool)
    initial = module(features, distance, valid)
    torch.testing.assert_close(initial, torch.zeros_like(initial))

    with torch.no_grad():
        module.gate_unconstrained[0, 0, 0] = 1.0
    routed = module(features, distance, valid)
    assert routed[0, 0, 0, 0].abs() > 0
    assert routed[0, 0, 0, 1] == 0
    assert torch.count_nonzero(routed[:, 1:]) == 0
    routed.sum().backward()
    assert module.gate_unconstrained.grad is not None
    assert torch.isfinite(module.gate_unconstrained.grad).all()


def test_routed_dense_attention_is_independently_se3_invariant() -> None:
    torch.manual_seed(97)
    encoder = TopologyRoutedIntraChainEncoder(16, 4, 1, dropout=0.0).eval()
    with torch.no_grad():
        encoder.blocks[0].geometry_bias.gate_unconstrained.normal_(std=0.2)
    features = torch.randn(2, 8, 16)
    mask = torch.ones(2, 8, dtype=torch.bool)
    chain_ids = torch.zeros(2, 8, dtype=torch.long)
    backbone = _backbone(2, 8)
    before, before_details = encoder(
        features,
        mask,
        backbone,
        chain_ids=chain_ids,
        return_attention=True,
    )
    transformed = _rigid_transform(
        backbone,
        _random_rotation(2),
        torch.tensor([[31.0, -8.0, 5.0], [-4.0, 13.0, 70.0]]),
    )
    after, after_details = encoder(
        features,
        mask,
        transformed,
        chain_ids=chain_ids,
        return_attention=True,
    )
    torch.testing.assert_close(before, after, atol=2e-5, rtol=2e-5)
    assert before_details is not None and after_details is not None
    torch.testing.assert_close(
        before_details["geometry"]["layers"][0]["geometry_bias"],
        after_details["geometry"]["layers"][0]["geometry_bias"],
        atol=2e-5,
        rtol=2e-5,
    )
    assert torch.count_nonzero(
        before_details["geometry"]["layers"][0]["geometry_bias"]
    ) > 0


def test_framecontact_uses_one_dense_qkv_path_at_exact_content_initialization() -> None:
    model = TopologyRoutedFrameContactModel(
        antibody_input_dim=12,
        antigen_input_dim=10,
        d_model=16,
        num_heads=4,
        num_self_layers=1,
        geometry_feature_mode="full_pose",
        geometry_bias_mode="topology_routed_multiscale",
        neighbor_k=4,
        neighbor_radius=16.0,
        dropout=0.0,
    ).eval()
    result = model(_batch(), return_attention=True)
    assert model.self_encoder is not None
    for stream in (
        model.self_encoder.antibody_encoder,
        model.self_encoder.antigen_encoder,
    ):
        assert stream.geometry_fusion_mode == "shared_qkv_multiscale"
        assert stream.dense_encoder is None
        assert stream.geometry_residual_gate is None
    details = result["self_attention_details"]
    assert details is not None
    for partner in ("antibody", "antigen"):
        geometry = details[partner]["geometry"]
        assert geometry["attention_graph_mode"] == "dense"
        for layer in geometry["layers"]:
            torch.testing.assert_close(
                layer["geometry_bias"], torch.zeros_like(layer["geometry_bias"])
            )
            assert "geometry_bias_components" in layer
