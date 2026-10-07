import pytest
import torch

from antibody_pretraining.paircontact_pretraining import (
    BipartiteEdgeRefinementContactHead,
    SymmetricLowRankPairContactHead,
    adaptive_negative_weights,
    build_partner_swap_permutation,
    paircontact_pretraining_loss,
    paircontact_ranking_metrics,
)


def test_partner_swap_matching_respects_false_negative_mask() -> None:
    allowed = torch.tensor(
        [
            [False, True, True, False],
            [True, False, False, True],
            [False, True, False, True],
            [True, False, True, False],
        ]
    )
    permutation = build_partner_swap_permutation(allowed, seed=1701)
    assert sorted(permutation.tolist()) == [0, 1, 2, 3]
    assert bool(allowed[torch.arange(4), permutation].all())


def test_partner_swap_fails_closed_without_perfect_matching() -> None:
    allowed = torch.tensor(
        [[False, True, False], [False, True, False], [True, False, False]]
    )
    with pytest.raises(ValueError, match="perfect matching"):
        build_partner_swap_permutation(allowed, seed=17)


def test_adaptive_weights_emphasize_hard_negatives_and_normalize() -> None:
    logits = torch.tensor([-3.0, -1.0, 2.0])
    weights = adaptive_negative_weights(
        logits,
        torch.tensor([0, 0, 1]),
        torch.tensor([0, 1, 1]),
        temperature=0.5,
    )
    assert float(weights.mean()) == pytest.approx(1.0)
    assert weights[2] > weights[1] > weights[0]


def test_paircontact_objective_is_finite_and_backpropagates() -> None:
    logits = torch.tensor(
        [
            [[2.0, -1.0, -2.0], [-0.5, 1.5, -1.0]],
            [[-1.0, 1.0, -2.0], [1.2, -0.5, -1.5]],
        ],
        requires_grad=True,
    )
    labels = torch.tensor(
        [
            [[True, False, False], [False, True, False]],
            [[False, True, False], [True, False, False]],
        ]
    )
    valid = torch.ones_like(labels)
    swapped = torch.tensor(
        [[[-2.0, -1.0], [-0.5, 0.2]], [[-1.0, -0.4], [-2.0, -0.2]]],
        requires_grad=True,
    )
    result = paircontact_pretraining_loss(
        logits,
        labels,
        valid,
        system_weights=torch.tensor([1.0, 0.5]),
        swapped_pair_logits=swapped,
        swapped_pair_valid=torch.ones_like(swapped, dtype=torch.bool),
    )
    assert result.positive_pairs == 4
    assert result.negative_pairs == 8
    assert result.valid_systems == 2
    result.total.backward()
    assert logits.grad is not None and bool(torch.isfinite(logits.grad).all())
    assert swapped.grad is not None and bool(torch.isfinite(swapped.grad).all())


def test_paircontact_objective_rejects_one_class_system() -> None:
    logits = torch.zeros((1, 2, 2))
    labels = torch.ones((1, 2, 2), dtype=torch.bool)
    valid = torch.ones_like(labels)
    with pytest.raises(ValueError, match="both classes"):
        paircontact_pretraining_loss(
            logits,
            labels,
            valid,
            partner_swap_weight=0.0,
        )


def test_balanced_focal_pair_loss_is_finite_and_focuses_hard_edges() -> None:
    easy = paircontact_pretraining_loss(
        torch.tensor([[[5.0, -5.0], [-4.0, 4.0]]]),
        torch.tensor([[[True, False], [False, True]]]),
        torch.ones((1, 2, 2), dtype=torch.bool),
        pair_loss_mode="balanced_focal",
        focal_gamma=2.0,
        partner_swap_weight=0.0,
    )
    hard = paircontact_pretraining_loss(
        torch.tensor([[[0.1, -0.1], [-0.2, 0.2]]]),
        torch.tensor([[[True, False], [False, True]]]),
        torch.ones((1, 2, 2), dtype=torch.bool),
        pair_loss_mode="balanced_focal",
        focal_gamma=2.0,
        partner_swap_weight=0.0,
    )
    assert bool(torch.isfinite(easy.total))
    assert hard.pair > easy.pair


def test_paircontact_objective_rejects_unknown_pair_loss() -> None:
    with pytest.raises(ValueError, match="not reviewed"):
        paircontact_pretraining_loss(
            torch.zeros((1, 2, 2)),
            torch.tensor([[[True, False], [False, True]]]),
            torch.ones((1, 2, 2), dtype=torch.bool),
            pair_loss_mode="unreviewed",
            partner_swap_weight=0.0,
        )


def test_symmetric_low_rank_head_is_order_invariant() -> None:
    torch.manual_seed(5)
    head = SymmetricLowRankPairContactHead(embedding_dimension=8, pair_dimension=4)
    left = torch.randn(2, 3, 8, requires_grad=True)
    right = torch.randn(2, 5, 8, requires_grad=True)
    logits = head(left, right)
    reverse = head(right, left).transpose(-1, -2)
    torch.testing.assert_close(logits, reverse)
    logits.square().mean().backward()
    assert left.grad is not None and right.grad is not None


def test_bipartite_edge_refinement_is_order_invariant_and_backpropagates() -> None:
    torch.manual_seed(7)
    head = BipartiteEdgeRefinementContactHead(
        embedding_dimension=8,
        pair_dimension=4,
        edge_dimension=6,
        refinement_steps=2,
    )
    left = torch.randn(2, 3, 8, requires_grad=True)
    right = torch.randn(2, 5, 8, requires_grad=True)
    valid = torch.ones((2, 3, 5), dtype=torch.bool)
    valid[0, 0, 4] = False
    valid[1, 2, 0] = False
    logits = head(left, right, valid)
    reverse = head(right, left, valid.transpose(1, 2)).transpose(1, 2)
    assert logits.shape == (2, 3, 5)
    assert bool(torch.isfinite(logits).all())
    torch.testing.assert_close(logits, reverse, atol=1e-6, rtol=1e-6)
    logits.square().mean().backward()
    assert left.grad is not None and bool(torch.isfinite(left.grad).all())
    assert right.grad is not None and bool(torch.isfinite(right.grad).all())
    assert head.output.weight.grad is not None


def test_bipartite_edge_refinement_rejects_disconnected_nodes() -> None:
    head = BipartiteEdgeRefinementContactHead(embedding_dimension=4)
    left = torch.randn(1, 2, 4)
    right = torch.randn(1, 3, 4)
    valid = torch.ones((1, 2, 3), dtype=torch.bool)
    valid[:, 1, :] = False
    with pytest.raises(ValueError, match="at least one valid edge"):
        head(left, right, valid)


def test_paircontact_ranking_metrics_are_exact_and_tie_aware() -> None:
    scores = torch.tensor([[0.9, 0.4, 0.4], [0.1, 0.3, 0.2]])
    labels = torch.tensor([[True, False, True], [False, False, False]])
    metrics = paircontact_ranking_metrics(scores, labels)
    assert metrics.average_precision == pytest.approx((1.0 + 2.0 / 3.0) / 2.0)
    assert metrics.auroc == pytest.approx(0.9375)
    assert metrics.mean_reciprocal_rank == 1.0
    assert metrics.recall_at_1 == 1.0
    assert metrics.recall_at_l_over_10 == pytest.approx(0.5)
    assert metrics.precision_at_l_over_10 == 1.0
    assert metrics.prevalence == pytest.approx(1.0 / 3.0)
