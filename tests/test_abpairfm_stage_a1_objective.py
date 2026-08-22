from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from antibody_pretraining.stage_a1_objective import StageA1Heads


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols" / "abpairfm_stage_a1_multitask_v1.json"
GLOBAL_PILOT_PROTOCOL = (
    ROOT / "protocols" / "abpairfm_stage_a1_multitask_pilot_v4.json"
)


def test_stage_a1_protocol_is_train_only_and_fail_closed() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert protocol["status"] == "FROZEN_IMPLEMENTATION_AND_SOURCE_GATE"
    assert protocol["authorization"] == {
        "development_model_selection": False,
        "pilot_training_after_source_admission": True,
        "production_training": False,
        "test_access": False,
    }
    assert protocol["sources"]["pinder"]["admitted"] is False
    assert protocol["systems"]["partition"] == "gpu_4090"
    assert protocol["systems"]["hp_4090_permitted"] is False


def test_stage_a1_global_retrieval_pilot_remains_train_only() -> None:
    protocol = json.loads(GLOBAL_PILOT_PROTOCOL.read_text(encoding="utf-8"))
    assert protocol["authorization"] == {
        "development_evaluation": False,
        "pilot_training": True,
        "production_training": False,
        "test_access": False,
    }
    assert protocol["loss"]["retrieval"]["candidate_scope"] == "same_node_all_ranks"
    assert protocol["loss"]["retrieval"]["ppiref_cluster_boundary"] == (
        "exact_sequence_fallback_only"
    )
    assert protocol["optimization"]["microbatch_pairs_per_rank"] == 6
    assert protocol["promotion_gates"]["maximum_peak_memory_fraction"] == 0.9
    assert protocol["test_access"] is False


def test_stage_a1_multitask_loss_is_finite_and_differentiable() -> None:
    torch.manual_seed(17)
    heads = StageA1Heads(embedding_dimension=8, projection_dimension=4)
    hidden = torch.randn(3, 7, 8, requires_grad=True)
    chain_ids = torch.tensor(
        [
            [0, 0, 0, 1, 1, -1, -1],
            [0, 0, 1, 1, 1, -1, -1],
            [0, 0, 0, 0, 1, 1, -1],
        ]
    )
    allowed = torch.ones((3, 3), dtype=torch.bool)
    result = heads.loss(
        hidden=hidden,
        chain_ids=chain_ids,
        mlm_loss=hidden.square().mean(),
        retrieval_allowed=allowed,
        pair_batch_indices=torch.tensor([0, 0, 1, 1, 2, 2]),
        pair_left_token_indices=torch.tensor([0, 1, 0, 1, 0, 2]),
        pair_right_token_indices=torch.tensor([3, 4, 2, 3, 4, 5]),
        pair_labels=torch.tensor([1, 0, 1, 0, 1, 0]),
    )
    assert torch.isfinite(result.total)
    assert result.contact_count == 6
    assert result.contact_positive_count == 3
    result.total.backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    assert all(parameter.grad is not None for parameter in heads.parameters())


def test_stage_a1_rejects_false_negative_mask_without_negatives() -> None:
    heads = StageA1Heads(embedding_dimension=4, projection_dimension=2)
    hidden = torch.randn(2, 4, 4)
    chain_ids = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]])
    with pytest.raises(ValueError, match="at least one negative"):
        heads.loss(
            hidden=hidden,
            chain_ids=chain_ids,
            mlm_loss=torch.tensor(1.0),
            retrieval_allowed=torch.eye(2, dtype=torch.bool),
            pair_batch_indices=torch.tensor([0, 0]),
            pair_left_token_indices=torch.tensor([0, 1]),
            pair_right_token_indices=torch.tensor([2, 3]),
            pair_labels=torch.tensor([1, 0]),
        )


def test_stage_a1_supports_rectangular_global_retrieval_candidates() -> None:
    torch.manual_seed(73)
    heads = StageA1Heads(embedding_dimension=4, projection_dimension=2)
    hidden = torch.randn(2, 4, 4, requires_grad=True)
    chain_ids = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]])
    candidate_left = torch.randn(5, 2, requires_grad=True)
    candidate_right = torch.randn(5, 2, requires_grad=True)
    allowed = torch.ones((2, 5), dtype=torch.bool)
    allowed[0, 4] = False
    result = heads.loss(
        hidden=hidden,
        chain_ids=chain_ids,
        mlm_loss=hidden.square().mean(),
        retrieval_allowed=allowed,
        retrieval_candidate_left=candidate_left,
        retrieval_candidate_right=candidate_right,
        retrieval_target_indices=torch.tensor([1, 3]),
        pair_batch_indices=torch.tensor([0, 0, 1, 1]),
        pair_left_token_indices=torch.tensor([0, 1, 0, 1]),
        pair_right_token_indices=torch.tensor([2, 3, 2, 3]),
        pair_labels=torch.tensor([1, 0, 1, 0]),
    )
    assert torch.isfinite(result.total)
    result.total.backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    assert candidate_left.grad is not None and torch.isfinite(candidate_left.grad).all()
    assert candidate_right.grad is not None and torch.isfinite(candidate_right.grad).all()
