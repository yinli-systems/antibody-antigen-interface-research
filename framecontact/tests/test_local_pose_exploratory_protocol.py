"""Isolation and end-to-end tests for the local-pose exploratory branch."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="PyTorch is unavailable",
)


def _backbone(residue_count: int):
    import torch

    ca = torch.cumsum(torch.randn(1, residue_count, 3), dim=1)
    n = ca + torch.tensor([-1.1, 0.7, 0.2])
    c = ca + torch.tensor([1.2, 0.1, -0.3])
    return torch.stack((n, ca, c), dim=-2)


def _batch():
    import torch

    from interaction_first.intra_chain_geometry import build_backbone_frames

    torch.manual_seed(20260731)
    antibody_mask = torch.ones(1, 5, dtype=torch.bool)
    antigen_mask = torch.ones(1, 7, dtype=torch.bool)
    antibody_backbone = _backbone(5)
    antigen_backbone = _backbone(7)
    antibody_valid = build_backbone_frames(
        antibody_backbone, antibody_mask
    ).valid_mask
    antigen_valid = build_backbone_frames(antigen_backbone, antigen_mask).valid_mask
    return {
        "antibody_features": torch.randn(1, 5, 12),
        "antigen_features": torch.randn(1, 7, 10),
        "antibody_backbone": antibody_backbone,
        "antigen_backbone": antigen_backbone,
        "antibody_mask": antibody_mask,
        "antigen_mask": antigen_mask,
        "antibody_chain_ids": torch.zeros(1, 5, dtype=torch.long),
        "antigen_chain_ids": torch.zeros(1, 7, dtype=torch.long),
        "antibody_frame_valid_mask": antibody_valid,
        "antigen_frame_valid_mask": antigen_valid,
        "antibody_missing_frame_mask": antibody_mask & ~antibody_valid,
        "antigen_missing_frame_mask": antigen_mask & ~antigen_valid,
    }


def test_registry_protocol_and_receipt_are_isolated_and_hash_bound() -> None:
    from interaction_first.exploratory_experiments import EXPLORATORY_EXPERIMENTS

    confirmatory_names = (
        "sequence_dense",
        "spatial_topology",
        "distance",
        "distance_direction",
        "full_pose",
        "frozen_frame_ipa",
        "framecontact",
        "framecontact_interleaved",
    )
    assert set(confirmatory_names).isdisjoint(EXPLORATORY_EXPERIMENTS)
    protocol_path = REPOSITORY_ROOT / "protocols/framecontact_local_pose_exploratory_v1.json"
    receipt_path = (
        REPOSITORY_ROOT
        / "protocols/FRAMECONTACT_LOCAL_POSE_EXPLORATORY_V1_RECEIPT.json"
    )
    protocol_raw = protocol_path.read_bytes()
    protocol = json.loads(protocol_raw)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert tuple(protocol["development_matrix"]) == tuple(EXPLORATORY_EXPERIMENTS)
    assert receipt["frozen_test_authorized"] is False
    assert receipt["confirmatory_registry_isolation_required"] is True
    assert receipt["protocol_file_sha256"] == hashlib.sha256(protocol_raw).hexdigest()
    canonical = (
        json.dumps(protocol, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")
    assert receipt["protocol_canonical_sha256"] == hashlib.sha256(canonical).hexdigest()


def test_confirmatory_and_exploratory_cli_choices_do_not_mix() -> None:
    base = subprocess.run(
        [sys.executable, "scripts/train_framecontact.py", "--help"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    exploratory = subprocess.run(
        [sys.executable, "scripts/train_framecontact_exploratory.py", "--help"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "local_pose_multibasis" not in base
    assert "local_pose_rotary" not in base
    assert "local_pose_multibasis" in exploratory
    assert "local_pose_rotary" in exploratory
    assert "framecontact_interleaved" not in exploratory


@pytest.mark.parametrize(
    ("experiment", "bias_mode"),
    (
        ("local_pose_multibasis", "local_pose_multibasis"),
        ("local_pose_rotary", "local_pose_rotary"),
    ),
)
def test_exploratory_modes_run_through_framecontact(
    experiment: str,
    bias_mode: str,
) -> None:
    import torch

    from interaction_first.exploratory_experiments import EXPLORATORY_EXPERIMENTS
    from interaction_first.framecontact import FrameContactModel

    specification = EXPLORATORY_EXPERIMENTS[experiment]
    model = FrameContactModel(
        antibody_input_dim=12,
        antigen_input_dim=10,
        d_model=16,
        num_heads=4,
        num_self_layers=1,
        geometry_feature_mode=str(specification["geometry_feature_mode"]),
        geometry_bias_mode=str(specification["geometry_bias_mode"]),
        geometry_attention_graph_mode="spatial",
        num_rbf=8,
        neighbor_k=4,
        neighbor_radius=None,
        dropout=0.0,
        fusion_mode=str(specification["fusion_mode"]),
    ).eval()
    assert model.self_encoder is not None
    for stream in (
        model.self_encoder.antibody_encoder,
        model.self_encoder.antigen_encoder,
    ):
        assert stream.geometry_encoder.geometry_bias_mode == bias_mode
        assert stream.dense_encoder.geometry_bias_mode == "pair_mlp"
    output = model(_batch(), return_attention=True)
    assert output["contact_logits"].shape == (1, 5, 7)
    assert torch.isfinite(output["contact_logits"]).all()
    geometry_layer = output["self_attention_details"]["antigen"]["geometry"][
        "layers"
    ][0]
    assert geometry_layer["geometry_bias"].shape == (1, 4, 7, 7)
