from __future__ import annotations

import pytest
import torch

from antibody_pretraining.stage_a1_objective import (
    StageA1Heads,
    _interaction_centering_penalty,
    _matched_interaction_ranking_loss,
)


def test_contact_logit_decomposes_into_unary_and_interaction_terms() -> None:
    heads = StageA1Heads(embedding_dimension=2, projection_dimension=2)
    with torch.no_grad():
        heads.contact_left.weight.copy_(torch.eye(2))
        heads.contact_right.weight.copy_(torch.eye(2))
        heads.contact_left_unary.weight.copy_(torch.tensor([[2.0, -1.0]]))
        heads.contact_right_unary.weight.copy_(torch.tensor([[0.5, 3.0]]))
        heads.contact_bias.fill_(0.25)
    hidden = torch.tensor(
        [[[1.0, 2.0], [0.0, 0.0], [3.0, 4.0], [0.0, 0.0]]]
    )
    components = heads.contact_components(
        hidden,
        torch.tensor([0]),
        torch.tensor([0]),
        torch.tensor([2]),
    )
    expected_interaction = torch.tensor([(1.0 * 3.0 + 2.0 * 4.0) / (2.0**0.5)])
    torch.testing.assert_close(components.interaction_logits, expected_interaction)
    torch.testing.assert_close(components.raw_interaction_logits, expected_interaction)
    torch.testing.assert_close(components.left_unary_logits, torch.tensor([0.0]))
    torch.testing.assert_close(components.right_unary_logits, torch.tensor([13.5]))
    torch.testing.assert_close(
        components.full_logits,
        expected_interaction + 13.5 + 0.25,
    )


def test_endpoint_zero_sum_residual_removes_both_endpoint_marginals() -> None:
    heads = StageA1Heads(
        embedding_dimension=2,
        projection_dimension=2,
        interaction_residualization="endpoint_zero_sum",
    )
    with torch.no_grad():
        heads.contact_left.weight.copy_(torch.eye(2))
        heads.contact_right.weight.copy_(torch.eye(2))
    hidden = torch.tensor([[[1.0, 0.0], [3.0, 2.0], [2.0, 1.0], [4.0, 5.0]]])
    components = heads.contact_components(
        hidden,
        torch.zeros(4, dtype=torch.long),
        torch.tensor([0, 0, 1, 1]),
        torch.tensor([2, 3, 2, 3]),
    )
    matrix = components.interaction_logits.reshape(2, 2)
    torch.testing.assert_close(matrix.mean(dim=0), torch.zeros(2), atol=1e-6, rtol=0)
    torch.testing.assert_close(matrix.mean(dim=1), torch.zeros(2), atol=1e-6, rtol=0)
    torch.testing.assert_close(components.centering_error, torch.tensor(0.0), atol=1e-12, rtol=0)
    assert not torch.allclose(
        components.raw_interaction_logits,
        components.interaction_logits,
    )


def test_endpoint_zero_sum_residual_is_invariant_to_sample_degree() -> None:
    heads = StageA1Heads(
        embedding_dimension=2,
        projection_dimension=2,
        interaction_residualization="endpoint_zero_sum",
    )
    hidden = torch.tensor([[[1.0, 0.0], [3.0, 2.0], [2.0, 1.0], [4.0, 5.0]]])
    base = heads.contact_components(
        hidden,
        torch.zeros(4, dtype=torch.long),
        torch.tensor([0, 0, 1, 1]),
        torch.tensor([2, 3, 2, 3]),
    )
    duplicated = heads.contact_components(
        hidden,
        torch.zeros(6, dtype=torch.long),
        torch.tensor([0, 0, 1, 1, 0, 0]),
        torch.tensor([2, 3, 2, 3, 2, 2]),
    )
    torch.testing.assert_close(
        duplicated.interaction_logits[:4],
        base.interaction_logits,
    )


def test_endpoint_zero_sum_residual_centers_bfloat16_inputs_in_float32() -> None:
    heads = StageA1Heads(
        embedding_dimension=4,
        projection_dimension=4,
        interaction_residualization="endpoint_zero_sum",
    ).to(dtype=torch.bfloat16)
    hidden = torch.tensor(
        [[[1.125, -0.75, 0.5, 2.0], [3.0, 2.25, -1.0, 0.25],
          [2.0, 1.5, 0.75, -0.5], [4.5, 5.0, -2.0, 1.0]]],
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    components = heads.contact_components(
        hidden,
        torch.zeros(4, dtype=torch.long),
        torch.tensor([0, 0, 1, 1]),
        torch.tensor([2, 3, 2, 3]),
    )
    assert components.interaction_logits.dtype == torch.float32
    matrix = components.interaction_logits.reshape(2, 2)
    torch.testing.assert_close(matrix.mean(dim=0), torch.zeros(2), atol=1e-8, rtol=0)
    torch.testing.assert_close(matrix.mean(dim=1), torch.zeros(2), atol=1e-8, rtol=0)
    assert float(components.centering_error.detach()) <= 1e-12
    components.full_logits.sum().backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad.float()).all()


def test_matched_ranking_rewards_positive_interaction_compatibility() -> None:
    labels = torch.tensor([1.0, 0.0, 0.0])
    batch = torch.tensor([0, 0, 0])
    left = torch.tensor([1, 1, 2])
    right = torch.tensor([3, 4, 3])
    high_loss, high_rows, high_columns = _matched_interaction_ranking_loss(
        torch.tensor([2.0, 0.0, 0.0]),
        labels,
        batch,
        left,
        right,
        margin=0.5,
    )
    low_loss, low_rows, low_columns = _matched_interaction_ranking_loss(
        torch.tensor([-2.0, 0.0, 0.0]),
        labels,
        batch,
        left,
        right,
        margin=0.5,
    )
    assert high_loss < low_loss
    assert (high_rows, high_columns) == (1, 1)
    assert (low_rows, low_columns) == (1, 1)


def test_interaction_centering_penalty_uses_both_pair_axes() -> None:
    batch = torch.zeros(4, dtype=torch.long)
    left = torch.tensor([0, 0, 1, 1])
    right = torch.tensor([2, 3, 2, 3])
    balanced = _interaction_centering_penalty(
        torch.tensor([1.0, -1.0, -1.0, 1.0]),
        batch,
        left,
        right,
    )
    unbalanced = _interaction_centering_penalty(
        torch.ones(4),
        batch,
        left,
        right,
    )
    torch.testing.assert_close(balanced, torch.tensor(0.0))
    torch.testing.assert_close(unbalanced, torch.tensor(1.0))


def test_centered_ranked_contact_objective_is_finite_and_differentiable() -> None:
    torch.manual_seed(1701)
    heads = StageA1Heads(embedding_dimension=4, projection_dimension=2)
    hidden = torch.randn(2, 5, 4, requires_grad=True)
    result = heads.loss(
        hidden=hidden,
        chain_ids=torch.tensor([[0, 0, 1, 1, -1], [0, 0, 1, 1, -1]]),
        mlm_loss=hidden.square().mean(),
        retrieval_allowed=torch.ones((2, 2), dtype=torch.bool),
        pair_batch_indices=torch.tensor([0, 0, 0, 1, 1, 1]),
        pair_left_token_indices=torch.tensor([0, 0, 1, 0, 0, 1]),
        pair_right_token_indices=torch.tensor([2, 3, 2, 2, 3, 2]),
        pair_labels=torch.tensor([1, 0, 0, 1, 0, 0]),
        contact_ranking_weight=0.5,
        contact_centering_weight=0.1,
    )
    assert torch.isfinite(result.total)
    assert torch.isfinite(result.contact_bce)
    assert torch.isfinite(result.contact_ranking)
    assert torch.isfinite(result.contact_centering)
    assert result.contact_row_comparison_count == 2
    assert result.contact_column_comparison_count == 2
    result.total.backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    assert all(parameter.grad is not None for parameter in heads.parameters())


def test_ranked_contact_objective_fails_without_matched_negatives() -> None:
    heads = StageA1Heads(embedding_dimension=4, projection_dimension=2)
    hidden = torch.randn(2, 4, 4)
    with pytest.raises(ValueError, match="row- or column-matched negative"):
        heads.loss(
            hidden=hidden,
            chain_ids=torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]]),
            mlm_loss=torch.tensor(1.0),
            retrieval_allowed=torch.ones((2, 2), dtype=torch.bool),
            pair_batch_indices=torch.tensor([0, 0]),
            pair_left_token_indices=torch.tensor([0, 1]),
            pair_right_token_indices=torch.tensor([2, 3]),
            pair_labels=torch.tensor([1, 0]),
            contact_ranking_weight=0.5,
        )


def test_zero_new_weights_preserve_bce_contact_objective() -> None:
    heads = StageA1Heads(embedding_dimension=4, projection_dimension=2)
    hidden = torch.randn(2, 4, 4)
    result = heads.loss(
        hidden=hidden,
        chain_ids=torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]]),
        mlm_loss=torch.tensor(1.0),
        retrieval_allowed=torch.ones((2, 2), dtype=torch.bool),
        pair_batch_indices=torch.tensor([0, 0, 1, 1]),
        pair_left_token_indices=torch.tensor([0, 1, 0, 1]),
        pair_right_token_indices=torch.tensor([2, 3, 2, 3]),
        pair_labels=torch.tensor([1, 0, 1, 0]),
    )
    torch.testing.assert_close(result.contact, result.contact_bce)


def test_contact_only_entrypoint_matches_multitask_contact_terms() -> None:
    torch.manual_seed(173)
    heads = StageA1Heads(embedding_dimension=4, projection_dimension=2)
    hidden = torch.randn(2, 4, 4)
    pairs = {
        "pair_batch_indices": torch.tensor([0, 0, 0, 1, 1, 1]),
        "pair_left_token_indices": torch.tensor([0, 0, 1, 0, 0, 1]),
        "pair_right_token_indices": torch.tensor([2, 3, 2, 2, 3, 2]),
        "pair_labels": torch.tensor([1, 0, 0, 1, 0, 0]),
    }
    contact = heads.contact_objective(
        hidden=hidden,
        **pairs,
        ranking_weight=0.5,
        centering_weight=0.1,
        ranking_margin=0.5,
    )
    multitask = heads.loss(
        hidden=hidden,
        chain_ids=torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]]),
        mlm_loss=torch.tensor(0.0),
        retrieval_allowed=torch.ones((2, 2), dtype=torch.bool),
        **pairs,
        mlm_weight=0.0,
        retrieval_weight=0.0,
        contact_weight=1.0,
        contact_ranking_weight=0.5,
        contact_centering_weight=0.1,
        contact_ranking_margin=0.5,
    )
    torch.testing.assert_close(contact.total, multitask.contact)
    torch.testing.assert_close(contact.bce, multitask.contact_bce)
    torch.testing.assert_close(contact.ranking, multitask.contact_ranking)
    torch.testing.assert_close(contact.centering, multitask.contact_centering)
    assert contact.row_comparison_count == multitask.contact_row_comparison_count
    assert contact.column_comparison_count == multitask.contact_column_comparison_count
