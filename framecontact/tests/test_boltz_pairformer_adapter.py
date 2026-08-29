"""Contract tests for the authenticated official Boltz Pairformer adapter."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


_OPTIONAL_RUNTIME_MODULES = ("torch", "einops", "einx", "fairscale")
_MISSING_RUNTIME_MODULES = tuple(
    name
    for name in _OPTIONAL_RUNTIME_MODULES
    if importlib.util.find_spec(name) is None
)

pytestmark = pytest.mark.skipif(
    bool(_MISSING_RUNTIME_MODULES),
    reason=(
        "Optional Boltz Pairformer runtime dependencies are unavailable: "
        + ", ".join(_MISSING_RUNTIME_MODULES)
    ),
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BOLTZ_ROOT = REPOSITORY_ROOT.parent / "external/boltz"


def _artifact(tmp_path: Path):
    import torch

    from interaction_first.boltz_pairformer_adapter import (
        BOLTZ1_CHECKPOINT_SHA256,
        BOLTZ_REVISION,
        PAIRFORMER_ARTIFACT_FORMAT,
    )

    python_root = BOLTZ_ROOT / "src"
    if str(python_root) not in sys.path:
        sys.path.insert(0, str(python_root))
    from boltz.model.modules.trunk import PairformerModule

    architecture = {
        "token_s": 16,
        "token_z": 8,
        "num_blocks": 1,
        "num_heads": 4,
        "dropout": 0.0,
        "pairwise_head_width": 4,
        "pairwise_num_heads": 2,
    }
    pairformer = PairformerModule(**architecture)
    artifact = {
        "format_version": PAIRFORMER_ARTIFACT_FORMAT,
        "source_checkpoint_sha256": BOLTZ1_CHECKPOINT_SHA256,
        "boltz_source_revision": BOLTZ_REVISION,
        "state_prefix": "pairformer_module.",
        "state_tensor_count": len(pairformer.state_dict()),
        "architecture": architecture,
        "state_dict": pairformer.state_dict(),
    }
    artifact_path = tmp_path / "pairformer.pt"
    torch.save(artifact, artifact_path)
    artifact_sha256 = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    receipt = {
        "format_version": "boltz1_pairformer_extraction_receipt_v1",
        "artifact_sha256": artifact_sha256,
        "source_checkpoint_sha256": BOLTZ1_CHECKPOINT_SHA256,
        "boltz_source_revision": BOLTZ_REVISION,
    }
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    return artifact_path, receipt_path, architecture


def test_official_boltz_pairformer_adapter_loads_frozen_weights_and_runs(
    tmp_path: Path,
) -> None:
    import torch

    from interaction_first.boltz_pairformer_adapter import (
        OfficialBoltzPairformerAdapter,
    )

    if not BOLTZ_ROOT.is_dir():
        pytest.skip("Pinned Boltz source is unavailable")
    artifact_path, receipt_path, architecture = _artifact(tmp_path)
    model = OfficialBoltzPairformerAdapter(
        artifact_path=artifact_path,
        receipt_path=receipt_path,
        boltz_source_root=BOLTZ_ROOT,
        single_input_dim=12,
        pair_input_dim=22,
        output_dim=10,
        freeze_pairformer=True,
    ).eval()
    assert not any(parameter.requires_grad for parameter in model.pairformer.parameters())
    assert any(
        parameter.requires_grad
        for parameter in model.single_input_projection.parameters()
    )
    residue_mask = torch.tensor([[True, True, True, False]])
    pair_mask = residue_mask[:, :, None] & residue_mask[:, None, :]
    output = model(
        torch.randn(1, 4, 12),
        torch.randn(1, 4, 4, 22),
        residue_mask,
        pair_mask,
    )
    assert output["single_features"].shape == (1, 4, 10)
    assert output["pair_features"].shape == (
        1,
        4,
        4,
        architecture["token_z"],
    )
    assert torch.isfinite(output["single_features"]).all()
    assert torch.equal(
        output["single_features"][:, 3],
        torch.zeros_like(output["single_features"][:, 3]),
    )
    assert output["pretrained_pairformer_frozen"] is True


def test_pairformer_adapter_rejects_artifact_receipt_mismatch(tmp_path: Path) -> None:
    from interaction_first.boltz_pairformer_adapter import (
        OfficialBoltzPairformerAdapter,
    )

    if not BOLTZ_ROOT.is_dir():
        pytest.skip("Pinned Boltz source is unavailable")
    artifact_path, receipt_path, _ = _artifact(tmp_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["artifact_sha256"] = "0" * 64
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="artifact SHA-256"):
        OfficialBoltzPairformerAdapter(
            artifact_path=artifact_path,
            receipt_path=receipt_path,
            boltz_source_root=BOLTZ_ROOT,
            single_input_dim=12,
            pair_input_dim=22,
            output_dim=10,
        )


def _backbone(batch_size: int = 1, residue_count: int = 4):
    import torch

    ca = torch.stack(
        (
            torch.arange(residue_count, dtype=torch.float32) * 3.8,
            torch.zeros(residue_count),
            torch.zeros(residue_count),
        ),
        dim=-1,
    )
    n = ca + torch.tensor([-1.2, 0.8, 0.0])
    c = ca + torch.tensor([1.3, 0.3, 0.4])
    return torch.stack((n, ca, c), dim=-2).unsqueeze(0).repeat(batch_size, 1, 1, 1)


def test_real_structure_monomer_wrapper_is_se3_invariant_and_masks_frames(
    tmp_path: Path,
) -> None:
    import torch

    from interaction_first.boltz_pairformer_adapter import (
        OfficialBoltzPairformerMonomerEncoder,
    )

    if not BOLTZ_ROOT.is_dir():
        pytest.skip("Pinned Boltz source is unavailable")
    artifact_path, receipt_path, _ = _artifact(tmp_path)
    model = OfficialBoltzPairformerMonomerEncoder(
        artifact_path=artifact_path,
        receipt_path=receipt_path,
        boltz_source_root=BOLTZ_ROOT,
        single_input_dim=12,
        output_dim=10,
        geometry_feature_mode="full_pose",
        attention_graph_mode="dense",
    ).eval()
    features = torch.randn(1, 4, 12)
    residue_mask = torch.ones(1, 4, dtype=torch.bool)
    chain_ids = torch.tensor([[0, 0, 0, 1]])
    backbone = _backbone()
    backbone[:, 2] = torch.nan
    angle = torch.tensor(0.73)
    rotation = torch.tensor(
        [
            [torch.cos(angle), -torch.sin(angle), 0.0],
            [torch.sin(angle), torch.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    translation = torch.tensor([4.2, -3.1, 7.0])
    transformed = torch.einsum("xy,bray->brax", rotation, backbone) + translation
    with torch.inference_mode():
        first = model(features, residue_mask, backbone, chain_ids)
        second = model(features, residue_mask, transformed, chain_ids)
    assert torch.allclose(
        first["single_features"], second["single_features"], atol=2e-5, rtol=2e-5
    )
    assert torch.equal(first["frame_valid_mask"], second["frame_valid_mask"])
    assert first["frame_valid_mask"].tolist() == [[True, True, False, True]]
    assert not bool(first["geometry_valid_mask"][:, 2].any())
    assert not bool(first["pair_mask"][:, :3, 3].any())
    assert first["geometry_feature_mode"] == "full_pose"


def test_dual_pairformer_wrapper_has_no_shared_parameter_storage(
    tmp_path: Path,
) -> None:
    from interaction_first.boltz_pairformer_adapter import (
        DualOfficialBoltzPairformerMonomerEncoder,
    )

    if not BOLTZ_ROOT.is_dir():
        pytest.skip("Pinned Boltz source is unavailable")
    artifact_path, receipt_path, _ = _artifact(tmp_path)
    model = DualOfficialBoltzPairformerMonomerEncoder(
        artifact_path=artifact_path,
        receipt_path=receipt_path,
        boltz_source_root=BOLTZ_ROOT,
        single_input_dim=12,
        output_dim=10,
    )
    antibody_parameters = list(model.antibody_encoder.parameters())
    antigen_parameters = list(model.antigen_encoder.parameters())
    assert len(antibody_parameters) == len(antigen_parameters)
    assert all(
        first.untyped_storage().data_ptr() != second.untyped_storage().data_ptr()
        for first, second in zip(antibody_parameters, antigen_parameters, strict=True)
    )


def test_adapted_pairformer_contact_model_matches_framecontact_output_contract(
    tmp_path: Path,
) -> None:
    import torch

    from interaction_first.intra_chain_geometry import build_backbone_frames
    from interaction_first.pairformer_framecontact import (
        AdaptedBoltzPairformerContactModel,
    )

    if not BOLTZ_ROOT.is_dir():
        pytest.skip("Pinned Boltz source is unavailable")
    artifact_path, receipt_path, _ = _artifact(tmp_path)
    model = AdaptedBoltzPairformerContactModel(
        artifact_path=artifact_path,
        receipt_path=receipt_path,
        boltz_source_root=BOLTZ_ROOT,
        antibody_input_dim=12,
        antigen_input_dim=12,
        d_model=16,
        num_heads=4,
        dropout=0.0,
    ).eval()
    antibody_backbone = _backbone(residue_count=3)
    antigen_backbone = _backbone(residue_count=4) + torch.tensor([0.0, 5.0, 2.0])
    antibody_mask = torch.ones(1, 3, dtype=torch.bool)
    antigen_mask = torch.ones(1, 4, dtype=torch.bool)
    antibody_valid = build_backbone_frames(
        antibody_backbone, antibody_mask
    ).valid_mask
    antigen_valid = build_backbone_frames(antigen_backbone, antigen_mask).valid_mask
    batch = {
        "antibody_features": torch.randn(1, 3, 12),
        "antigen_features": torch.randn(1, 4, 12),
        "antibody_mask": antibody_mask,
        "antigen_mask": antigen_mask,
        "antibody_backbone": antibody_backbone,
        "antigen_backbone": antigen_backbone,
        "antibody_chain_ids": torch.zeros(1, 3, dtype=torch.long),
        "antigen_chain_ids": torch.zeros(1, 4, dtype=torch.long),
        "antibody_frame_valid_mask": antibody_valid,
        "antigen_frame_valid_mask": antigen_valid,
        "antibody_missing_frame_mask": antibody_mask & ~antibody_valid,
        "antigen_missing_frame_mask": antigen_mask & ~antigen_valid,
    }
    with torch.inference_mode():
        output = model(batch)
        altered = model(
            {**batch, "antibody_features": batch["antibody_features"] + 0.7}
        )
    assert output["contact_logits"].shape == (1, 3, 4)
    assert output["paratope_logits"].shape == (1, 3)
    assert output["epitope_logits"].shape == (1, 4)
    assert output["pair_candidate_mask"].all()
    assert output["baseline_family"] == "adapted_boltz1_pairformer_contact"
    assert not torch.equal(
        output["contact_interaction_logits"],
        altered["contact_interaction_logits"],
    )
    assert not any(
        parameter.requires_grad
        for parameter in model.antibody_encoder.adapter.pairformer.parameters()
    )
    assert not any(
        parameter.requires_grad
        for parameter in model.antigen_encoder.adapter.pairformer.parameters()
    )
