"""Tests for the clean paired-sequence conditioning boundary."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="PyTorch is unavailable",
)


def _receipt(artifact: Path, arrays: dict[str, np.ndarray]) -> dict[str, object]:
    from interaction_first.paired_sequence_features import (
        PPLM_ATTENTION_REDUCTION,
        PPLM_CACHE_FORMAT,
        PPLM_LICENSE_BOUNDARY,
        PPLM_OFFICIAL_REPOSITORY,
        PPLM_OFFICIAL_REVISION,
        sequence_sha256,
    )

    return {
        "format_version": PPLM_CACHE_FORMAT,
        "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "source_repository": PPLM_OFFICIAL_REPOSITORY,
        "source_revision": PPLM_OFFICIAL_REVISION,
        "source_checkpoint_sha256": "a" * 64,
        "license_boundary": PPLM_LICENSE_BOUNDARY,
        "attention_reduction": PPLM_ATTENTION_REDUCTION,
        "chain_order": {"sequence_A": "antibody", "sequence_B": "antigen"},
        "antibody_sequence_sha256": sequence_sha256("ACD"),
        "antigen_sequence_sha256": sequence_sha256("EFGHI"),
        "arrays": {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in arrays.items()
        },
    }


def test_safe_pplm_cache_is_hash_and_sequence_bound(tmp_path: Path) -> None:
    from interaction_first.paired_sequence_features import (
        load_pplm_pair_feature_artifact,
        sequence_sha256,
    )

    arrays = {
        "antibody_embedding": np.arange(12, dtype=np.float32).reshape(3, 4),
        "antigen_embedding": np.arange(20, dtype=np.float32).reshape(5, 4),
        "inter_attention": np.full((4, 3, 5), 0.2, dtype=np.float32),
    }
    artifact = tmp_path / "pair.npz"
    np.savez(artifact, **arrays)
    receipt = tmp_path / "pair.receipt.json"
    payload = _receipt(artifact, arrays)
    receipt.write_text(json.dumps(payload), encoding="utf-8")
    loaded = load_pplm_pair_feature_artifact(
        artifact,
        receipt,
        expected_antibody_sequence_sha256=sequence_sha256("ACD"),
        expected_antigen_sequence_sha256=sequence_sha256("EFGHI"),
    )
    np.testing.assert_array_equal(
        loaded["inter_attention"], arrays["inter_attention"]
    )
    assert loaded["provenance"]["artifact_sha256"] == payload["artifact_sha256"]

    payload["artifact_sha256"] = "0" * 64
    receipt.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 differs"):
        load_pplm_pair_feature_artifact(artifact, receipt)


def test_pplm_attention_reduction_is_bidirectional_and_depth_grouped() -> None:
    import torch

    from interaction_first.paired_sequence_features import (
        PPLM_ATTENTION_REDUCTION,
        reduce_pplm_inter_attention,
    )

    attention_ab = torch.empty(33, 20, 2, 3)
    attention_ba = torch.empty(33, 20, 3, 2)
    for layer in range(33):
        attention_ab[layer].fill_(float(layer + 1))
        attention_ba[layer].fill_(float(2 * (layer + 1)))
    reduced = reduce_pplm_inter_attention(attention_ab, attention_ba)
    assert tuple(reduced.shape) == (4, 2, 3)
    for index, (start, stop) in enumerate(
        PPLM_ATTENTION_REDUCTION["layer_groups"]
    ):
        expected = 1.5 * sum(range(start + 1, stop + 1)) / (stop - start)
        torch.testing.assert_close(reduced[index], torch.full((2, 3), expected))


def test_pplm_pair_store_requires_the_exact_ordered_endpoints(tmp_path: Path) -> None:
    from interaction_first.paired_sequence_features import (
        PPLMPairFeatureStore,
        paired_feature_cache_key,
        sequence_sha256,
    )

    antibody_digest = sequence_sha256("ACD")
    antigen_digest = sequence_sha256("EFGHI")
    assert paired_feature_cache_key(antibody_digest, antigen_digest) != (
        paired_feature_cache_key(antigen_digest, antibody_digest)
    )
    store = PPLMPairFeatureStore(tmp_path)
    artifact, receipt = store.paths(antibody_digest, antigen_digest)
    artifact.parent.mkdir(parents=True)
    arrays = {
        "antibody_embedding": np.arange(12, dtype=np.float32).reshape(3, 4),
        "antigen_embedding": np.arange(20, dtype=np.float32).reshape(5, 4),
        "inter_attention": np.full((4, 3, 5), 0.2, dtype=np.float32),
    }
    np.savez(artifact, **arrays)
    receipt.write_text(json.dumps(_receipt(artifact, arrays)), encoding="utf-8")
    loaded = store.load(antibody_digest, antigen_digest)
    assert tuple(loaded["inter_attention"].shape) == (4, 3, 5)
    with pytest.raises(FileNotFoundError, match="exact PPLM"):
        store.load(antigen_digest, antibody_digest)


def test_pplm_conditioner_is_exactly_baseline_preserving_at_initialization() -> None:
    import torch

    from interaction_first.paired_sequence_features import PPLMPairedConditioner

    torch.manual_seed(19)
    module = PPLMPairedConditioner(
        antibody_base_dim=5,
        antigen_base_dim=7,
        paired_embedding_dim=11,
        attention_head_count=4,
    )
    antibody_base = torch.randn(2, 3, 5)
    antigen_base = torch.randn(2, 6, 7)
    antibody_mask = torch.tensor([[True, True, True], [True, True, False]])
    antigen_mask = torch.tensor(
        [[True, True, True, True, True, True], [True, True, True, True, False, False]]
    )
    arguments = (
        antibody_base,
        antigen_base,
        torch.randn(2, 3, 11),
        torch.randn(2, 6, 11),
        torch.rand(2, 4, 3, 6).clamp_min(1e-4),
        antibody_mask,
        antigen_mask,
    )
    first = module(*arguments)
    second = module(
        antibody_base,
        antigen_base,
        torch.randn(2, 3, 11) * 10,
        torch.randn(2, 6, 11) * 10,
        torch.rand(2, 4, 3, 6).clamp_min(1e-4),
        antibody_mask,
        antigen_mask,
    )
    torch.testing.assert_close(
        first["antibody_features"], second["antibody_features"], atol=0, rtol=0
    )
    torch.testing.assert_close(
        first["antigen_features"], second["antigen_features"], atol=0, rtol=0
    )
    assert torch.count_nonzero(first["pair_prior_logits"]) == 0
    antibody_storage = {
        parameter.untyped_storage().data_ptr()
        for parameter in module.antibody_projection.parameters()
    }
    antigen_storage = {
        parameter.untyped_storage().data_ptr()
        for parameter in module.antigen_projection.parameters()
    }
    assert antibody_storage.isdisjoint(antigen_storage)

    loss = first["raw_pair_prior_logits"].square().mean()
    assert torch.isfinite(loss)
    with torch.no_grad():
        module.antibody_residue_gate.fill_(0.4)
        module.antigen_residue_gate.fill_(-0.3)
        module.pair_prior_gate.fill_(0.5)
    active = module(*arguments)
    assert not torch.allclose(active["antibody_features"], antibody_base)
    assert not torch.allclose(active["antigen_features"], antigen_base)
    assert torch.count_nonzero(active["pair_prior_logits"]) > 0


def test_distogram_prior_preserves_exact_contact_projection() -> None:
    import torch

    from interaction_first.framecontact import (
        PairDistogramHead,
        contact_logit_from_distogram,
        inject_pair_contact_prior,
    )

    torch.manual_seed(23)
    head = PairDistogramHead(8, pair_dim=4)
    antibody = torch.randn(2, 3, 8)
    antigen = torch.randn(2, 5, 8)
    pair_mask = torch.ones(2, 3, 5, dtype=torch.bool)
    pair_mask[1, 2, 4] = False
    components = head.components(antibody, antigen, pair_mask)
    prior = torch.randn(2, 3, 5).masked_fill(~pair_mask, 0.0)
    updated = inject_pair_contact_prior(components, prior, pair_mask)
    torch.testing.assert_close(
        updated["logits"][pair_mask],
        (components["logits"] + prior)[pair_mask],
        atol=2e-6,
        rtol=2e-6,
    )
    projected = contact_logit_from_distogram(
        updated["distogram_logits"],
        contact_bin_count=head.contact_bin_count,
    )
    torch.testing.assert_close(
        updated["logits"][pair_mask],
        projected[pair_mask],
        atol=2e-6,
        rtol=2e-6,
    )
    assert torch.count_nonzero(updated["logits"][~pair_mask]) == 0


def _backbone(residue_count: int):
    import torch

    ca = torch.stack(
        (
            torch.arange(residue_count, dtype=torch.float32) * 3.8,
            torch.zeros(residue_count),
            torch.zeros(residue_count),
        ),
        dim=-1,
    )
    n = ca + torch.tensor([-1.2, 0.7, 0.1])
    c = ca + torch.tensor([1.3, 0.2, -0.2])
    return torch.stack((n, ca, c), dim=-2).unsqueeze(0)


def test_framecontact_accepts_zero_gated_paired_sequence_features() -> None:
    import torch

    from interaction_first.framecontact import FrameContactModel
    from interaction_first.intra_chain_geometry import build_backbone_frames

    torch.manual_seed(29)
    antibody_backbone = _backbone(3)
    antigen_backbone = _backbone(5)
    antibody_mask = torch.ones(1, 3, dtype=torch.bool)
    antigen_mask = torch.ones(1, 5, dtype=torch.bool)
    antibody_valid = build_backbone_frames(
        antibody_backbone, antibody_mask
    ).valid_mask
    antigen_valid = build_backbone_frames(antigen_backbone, antigen_mask).valid_mask
    batch = {
        "antibody_features": torch.randn(1, 3, 12),
        "antigen_features": torch.randn(1, 5, 10),
        "antibody_paired_sequence_features": torch.randn(1, 3, 6),
        "antigen_paired_sequence_features": torch.randn(1, 5, 6),
        "paired_sequence_inter_attention": torch.rand(1, 4, 3, 5).clamp_min(
            1e-4
        ),
        "antibody_backbone": antibody_backbone,
        "antigen_backbone": antigen_backbone,
        "antibody_mask": antibody_mask,
        "antigen_mask": antigen_mask,
        "antibody_chain_ids": torch.zeros(1, 3, dtype=torch.long),
        "antigen_chain_ids": torch.zeros(1, 5, dtype=torch.long),
        "antibody_frame_valid_mask": antibody_valid,
        "antigen_frame_valid_mask": antigen_valid,
        "antibody_missing_frame_mask": antibody_mask & ~antibody_valid,
        "antigen_missing_frame_mask": antigen_mask & ~antigen_valid,
    }
    model = FrameContactModel(
        antibody_input_dim=12,
        antigen_input_dim=10,
        d_model=16,
        num_heads=4,
        num_self_layers=1,
        neighbor_k=2,
        neighbor_radius=None,
        dropout=0.0,
        pair_output_mode="distogram",
        paired_sequence_embedding_dim=6,
        paired_sequence_attention_heads=4,
    ).eval()
    with torch.inference_mode():
        first = model(batch)
        changed = dict(batch)
        changed["antibody_paired_sequence_features"] = torch.randn(1, 3, 6) * 9
        changed["antigen_paired_sequence_features"] = torch.randn(1, 5, 6) * 9
        changed["paired_sequence_inter_attention"] = torch.rand(
            1, 4, 3, 5
        ).clamp_min(1e-4)
        second = model(changed)
    torch.testing.assert_close(
        first["contact_logits"], second["contact_logits"], atol=0, rtol=0
    )
    assert torch.count_nonzero(first["paired_sequence_pair_prior_logits"]) == 0

    with torch.no_grad():
        model.paired_sequence_conditioner.pair_prior_gate.fill_(0.6)
    with torch.inference_mode():
        active = model(batch)
    assert not torch.allclose(active["contact_logits"], first["contact_logits"])
    projected = contact_logit_from_distogram_for_test(active, model)
    torch.testing.assert_close(
        active["contact_logits"], projected, atol=2e-6, rtol=2e-6
    )


def contact_logit_from_distogram_for_test(output, model):
    from interaction_first.framecontact import contact_logit_from_distogram

    return contact_logit_from_distogram(
        output["distogram_logits"],
        contact_bin_count=model.contact_head.contact_bin_count,
    )
