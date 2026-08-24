from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from antibody_pretraining.stage_a1_geometry_objective import (
    StageA1GeometryHeads,
    expected_random_contact_loss,
)


ROOT = Path(__file__).resolve().parents[1]


def test_geometry_head_couples_contact_to_distance_mass() -> None:
    torch.manual_seed(17)
    heads = StageA1GeometryHeads(
        embedding_dimension=8,
        projection_dimension=4,
        pair_dimension=6,
        num_distance_bins=5,
        contact_distance_bin_count=2,
    )
    hidden = torch.randn(2, 5, 8, requires_grad=True)
    batch = torch.tensor([0, 0, 1, 1])
    left = torch.tensor([0, 1, 0, 1])
    right = torch.tensor([2, 3, 2, 4])
    distance = heads.distance_logits(hidden, batch, left, right)
    contact = heads.contact_logits(hidden, batch, left, right)
    expected = torch.logsumexp(distance[:, :2], dim=-1) - torch.logsumexp(
        distance[:, 2:], dim=-1
    ) + heads.contact_bias
    torch.testing.assert_close(contact, expected)


def test_geometry_loss_is_finite_and_differentiable() -> None:
    torch.manual_seed(42)
    heads = StageA1GeometryHeads(
        embedding_dimension=8,
        projection_dimension=4,
        pair_dimension=6,
        num_distance_bins=5,
        contact_distance_bin_count=2,
    )
    hidden = torch.randn(2, 5, 8, requires_grad=True)
    contact_loss, distance_loss, count, positive_count = (
        heads.contact_and_distance_loss(
            hidden=hidden,
            pair_batch_indices=torch.tensor([0, 0, 1, 1]),
            pair_left_token_indices=torch.tensor([0, 1, 0, 1]),
            pair_right_token_indices=torch.tensor([2, 3, 2, 4]),
            distance_bin_targets=torch.tensor([0, 2, 1, 4]),
        )
    )
    assert count == 4
    assert positive_count == 2
    (contact_loss + distance_loss).backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    assert all(
        parameter.grad is not None
        for name, parameter in heads.named_parameters()
        if name.startswith("contact_")
    )


def test_geometry_ranking_loss_is_complex_balanced_and_differentiable() -> None:
    torch.manual_seed(43)
    heads = StageA1GeometryHeads(
        embedding_dimension=8,
        projection_dimension=4,
        pair_dimension=6,
        num_distance_bins=5,
        contact_distance_bin_count=2,
    )
    hidden = torch.randn(2, 6, 8, requires_grad=True)
    losses = heads.balanced_contact_distance_and_ranking_loss(
        hidden=hidden,
        pair_batch_indices=torch.tensor([0, 0, 0, 1, 1, 1]),
        pair_left_token_indices=torch.tensor([0, 0, 1, 0, 1, 1]),
        pair_right_token_indices=torch.tensor([2, 3, 4, 2, 3, 4]),
        distance_bin_targets=torch.tensor([0, 2, 4, 1, 3, 4]),
    )
    contact_loss, ranking_loss, distance_loss, count, positives, complexes = losses
    assert count == 6
    assert positives == 2
    assert complexes == 2
    assert all(torch.isfinite(value) for value in losses[:3])
    (contact_loss + ranking_loss + distance_loss).backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()


def test_geometry_ranking_loss_rewards_positive_ordering() -> None:
    positive = torch.tensor([2.0, 1.0])
    negative = torch.tensor([-1.0, -2.0])
    ordered = torch.nn.functional.softplus(
        negative.unsqueeze(0) - positive.unsqueeze(1)
    ).mean()
    reversed_order = torch.nn.functional.softplus(
        positive.unsqueeze(0) - negative.unsqueeze(1)
    ).mean()
    assert ordered < reversed_order


def test_geometry_loss_rejects_binary_labels_as_distance_bins() -> None:
    heads = StageA1GeometryHeads(
        embedding_dimension=4,
        projection_dimension=2,
        pair_dimension=3,
        num_distance_bins=4,
        contact_distance_bin_count=1,
    )
    with pytest.raises(ValueError, match="out of range"):
        heads.contact_and_distance_loss(
            hidden=torch.randn(1, 3, 4),
            pair_batch_indices=torch.tensor([0, 0]),
            pair_left_token_indices=torch.tensor([0, 1]),
            pair_right_token_indices=torch.tensor([2, 2]),
            distance_bin_targets=torch.tensor([0, 4]),
        )


def test_random_contact_loss_uses_sampled_prior() -> None:
    expected = -(0.2 * math.log(0.2) + 0.8 * math.log(0.8))
    assert expected_random_contact_loss(4) == pytest.approx(expected)


def test_geometry_canary_is_train_only_and_uses_4p5a_mass() -> None:
    protocol = json.loads(
        (ROOT / "protocols" / "abpairfm_stage_a1_geometry_canary_v15.json").read_text(
            encoding="utf-8"
        )
    )
    assert protocol["authorization"] == {
        "development_evaluation": False,
        "pilot_training": True,
        "production_training": False,
        "test_access": False,
    }
    contact = protocol["loss"]["contact"]
    assert contact["distance_bin_edges_angstrom"][:3] == [2.0, 4.5, 6.0]
    assert contact["geometry_head"]["contact_distance_bin_count"] == 2
    assert protocol["optimization"]["updates"] == 40
    assert protocol["systems"]["partition"] == "gpu_4090"
    assert protocol["test_access"] is False


def test_geometry_canary_submission_is_frozen_to_gpu_4090() -> None:
    submission = (
        ROOT / "scripts" / "submit_abpairfm_stage_a1_geometry_canary_v15.sbatch"
    ).read_text(encoding="utf-8")
    assert "#SBATCH -p gpu_4090" in submission
    assert "#SBATCH --gres=gpu:8" in submission
    assert "hp_4090" not in submission
    assert "segment-000031" in submission
    assert "--progress-interval 10" in submission


def test_geometry_head_only_canary_freezes_backbone_and_test() -> None:
    protocol = json.loads(
        (
            ROOT
            / "protocols"
            / "abpairfm_stage_a1_geometry_head_only_canary_v16.json"
        ).read_text(encoding="utf-8")
    )
    assert protocol["optimization"]["backbone_learning_rate"] == 0.0
    assert protocol["optimization"]["updates"] == 80
    assert protocol["optimization"]["microbatch_pairs_per_rank_candidates"] == [48]
    assert protocol["efficiency"]["activation_checkpointing_candidates"] == ["off"]
    assert protocol["loss"]["contact"]["distance_bin_weight"] == 1.0
    assert protocol["loss"]["masked_language_modeling"]["weight"] == 0.0
    assert protocol["loss"]["retrieval"]["weight"] == 0.0
    assert protocol["systems"]["world_size"] == 4
    assert protocol["authorization"]["development_evaluation"] is False
    assert protocol["test_access"] is False

    submission = (
        ROOT
        / "scripts"
        / "submit_abpairfm_stage_a1_geometry_head_only_canary_v16.sbatch"
    ).read_text(encoding="utf-8")
    assert "#SBATCH -p gpu_4090" in submission
    assert "#SBATCH --gres=gpu:4" in submission
    assert "--efficiency-microbatch-pairs-per-rank 48" in submission
    assert "--efficiency-activation-checkpoint-mode off" in submission
    assert "hp_4090" not in submission


def test_geometry_ranking_canary_freezes_internal_holdout_and_test() -> None:
    protocol = json.loads(
        (
            ROOT
            / "protocols"
            / "abpairfm_stage_a1_geometry_ranking_canary_v17.json"
        ).read_text(encoding="utf-8")
    )
    assert protocol["optimization"]["updates"] == 80
    assert protocol["optimization"]["head_learning_rate"] == 0.0001
    assert protocol["loss"]["contact"]["ranking_weight"] == 1.0
    assert protocol["loss"]["contact"]["distance_bin_weight"] == 1.0
    assert protocol["internal_holdout"] == {
        "expected_pdb_count": 511,
        "grouping": "lowercase_pdb_prefix",
        "pdb_inventory_sha256": (
            "0166ecee25b61f333fe36197ca9625ed3a134221946a38e023f6022b4cc49028"
        ),
        "selection_seed": 1701,
        "target_records": 512,
        "test_access": False,
    }
    assert protocol["authorization"]["development_evaluation"] is False
    assert protocol["test_access"] is False
    submission = (
        ROOT
        / "scripts"
        / "submit_abpairfm_stage_a1_geometry_ranking_canary_v17.sbatch"
    ).read_text(encoding="utf-8")
    assert "#SBATCH -p gpu_4090" in submission
    assert "#SBATCH --gres=gpu:4" in submission
    assert "--efficiency-microbatch-pairs-per-rank 48" in submission
    assert "--efficiency-activation-checkpoint-mode off" in submission
    assert "hp_4090" not in submission
