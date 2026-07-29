"""Tests for bidirectional score-free E1/E2 partner objectives."""

from __future__ import annotations

from dataclasses import replace
import importlib.util

import pytest

from interaction_first.development_contact_dataset import DevelopmentContactExample
from interaction_first.pairscale_partner_specificity import (
    bidirectional_specificity_losses,
    build_single_anchor_score_row,
    multi_positive_margin_loss,
    multi_positive_retrieval_loss,
    raw_specificity_partner_score,
    require_specificity_partner_score,
    select_score_free_bidirectional_decoys,
)


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="PyTorch is unavailable",
)


def _example(
    index: int,
    antibody_id: str,
    antigen_id: str,
    antibody_length: int,
    antigen_length: int,
) -> DevelopmentContactExample:
    import torch

    return DevelopmentContactExample(
        instance_id=f"instance-{index}",
        antibody_id=antibody_id,
        antigen_id=antigen_id,
        antibody_embeddings=torch.full((antibody_length, 3), float(index)),
        antigen_embeddings=torch.full((antigen_length, 3), float(index)),
        contact_targets=torch.full(
            (antibody_length, antigen_length),
            index % 2,
        ),
        contact_mask=torch.ones(
            (antibody_length, antigen_length),
            dtype=torch.bool,
        ),
        positive_pair_count=index + 1,
    )


def _examples() -> tuple[DevelopmentContactExample, ...]:
    return (
        _example(0, "ab-0", "ag-0", 10, 20),
        _example(1, "ab-0", "ag-1", 10, 21),
        _example(2, "ab-1", "ag-0", 11, 20),
        _example(3, "ab-2", "ag-2", 12, 22),
        _example(4, "ab-3", "ag-3", 13, 23),
        _example(5, "ab-4", "ag-4", 14, 24),
        _example(6, "ab-5", "ag-5", 15, 25),
    )


def test_bidirectional_decoys_are_deterministic_and_exclude_observed_pairs() -> None:
    examples = _examples()
    first = select_score_free_bidirectional_decoys(examples, count=2, seed=17)
    replay = select_score_free_bidirectional_decoys(examples, count=2, seed=17)
    assert first == replay
    first.validate(len(examples))
    observed = {(example.antibody_id, example.antigen_id) for example in examples}
    for matched_index, matched in enumerate(examples):
        antigen_ids = {
            examples[index].antigen_id
            for index in first.antibody_to_antigen[matched_index]
        }
        antibody_ids = {
            examples[index].antibody_id
            for index in first.antigen_to_antibody[matched_index]
        }
        assert len(antigen_ids) == 2
        assert len(antibody_ids) == 2
        assert all(
            (matched.antibody_id, antigen_id) not in observed
            for antigen_id in antigen_ids
        )
        assert all(
            (antibody_id, matched.antigen_id) not in observed
            for antibody_id in antibody_ids
        )
    assert 1 not in first.antibody_to_antigen[0]
    assert 2 not in first.antigen_to_antibody[0]


def test_decoy_selection_reads_shapes_and_identifiers_not_values_or_labels() -> None:
    import torch

    examples = _examples()
    baseline = select_score_free_bidirectional_decoys(examples, count=2, seed=17)
    altered = tuple(
        replace(
            example,
            antibody_embeddings=torch.full_like(
                example.antibody_embeddings,
                float("nan"),
            ),
            antigen_embeddings=torch.full_like(
                example.antigen_embeddings,
                float("nan"),
            ),
            contact_targets=torch.full_like(example.contact_targets, 99),
            contact_mask=torch.zeros_like(example.contact_mask),
            positive_pair_count=1000 + index,
        )
        for index, example in enumerate(examples)
    )
    assert select_score_free_bidirectional_decoys(
        altered,
        count=2,
        seed=17,
    ) == baseline


def test_multi_positive_losses_reward_separation_and_backpropagate() -> None:
    import torch

    positive_mask = torch.tensor([[True, True, False, False]])
    separated = torch.tensor([[4.0, 3.0, -1.0, -2.0]], requires_grad=True)
    inverted = torch.tensor([[-1.0, -2.0, 4.0, 3.0]])
    separated_retrieval = multi_positive_retrieval_loss(
        separated,
        positive_mask,
        temperature=0.5,
    )
    inverted_retrieval = multi_positive_retrieval_loss(
        inverted,
        positive_mask,
        temperature=0.5,
    )
    separated_margin = multi_positive_margin_loss(
        separated,
        positive_mask,
        margin=0.2,
        temperature=0.5,
    )
    inverted_margin = multi_positive_margin_loss(
        inverted,
        positive_mask,
        margin=0.2,
        temperature=0.5,
    )
    assert float(separated_retrieval.detach()) < float(inverted_retrieval)
    assert float(separated_margin.detach()) < float(inverted_margin)
    (separated_retrieval + separated_margin).backward()
    assert separated.grad is not None
    assert bool(torch.isfinite(separated.grad).all())


def test_known_second_positive_is_not_treated_as_a_negative() -> None:
    import torch

    scores = torch.tensor([[3.0, 2.5, -2.0]])
    multi_positive = torch.tensor([[True, True, False]])
    false_negative = torch.tensor([[True, False, False]])
    assert float(
        multi_positive_retrieval_loss(
            scores,
            multi_positive,
            temperature=1.0,
        )
    ) < float(
        multi_positive_retrieval_loss(
            scores,
            false_negative,
            temperature=1.0,
        )
    )
    assert float(
        multi_positive_margin_loss(
            scores,
            multi_positive,
            margin=0.2,
            temperature=1.0,
        )
    ) < float(
        multi_positive_margin_loss(
            scores,
            false_negative,
            margin=0.2,
            temperature=1.0,
        )
    )


def test_losses_are_invariant_to_joint_candidate_permutation() -> None:
    import torch

    scores = torch.tensor([[0.2, 2.0, -0.5, 1.5]])
    mask = torch.tensor([[False, True, False, True]])
    permutation = torch.tensor([3, 0, 2, 1])
    for loss in (multi_positive_retrieval_loss, multi_positive_margin_loss):
        kwargs = {"temperature": 0.7}
        if loss is multi_positive_margin_loss:
            kwargs["margin"] = 0.1
        baseline = loss(scores, mask, **kwargs)
        permuted = loss(scores[:, permutation], mask[:, permutation], **kwargs)
        assert float(baseline) == pytest.approx(float(permuted), abs=1e-7)


def test_bidirectional_objective_equal_weights_both_directions() -> None:
    import torch

    ab_scores = torch.tensor([[2.0, -1.0, 0.0]], requires_grad=True)
    ag_scores = torch.tensor([[1.5, -0.5, -1.0]], requires_grad=True)
    positive = torch.tensor([[True, False, False]])
    losses = bidirectional_specificity_losses(
        antibody_to_antigen_scores=ab_scores,
        antibody_to_antigen_positive_mask=positive,
        antigen_to_antibody_scores=ag_scores,
        antigen_to_antibody_positive_mask=positive,
        retrieval_temperature=0.5,
        margin=0.2,
        margin_temperature=0.5,
    )
    assert float(losses.retrieval.detach()) == pytest.approx(
        0.5
        * (
            float(losses.antibody_to_antigen_retrieval.detach())
            + float(losses.antigen_to_antibody_retrieval.detach())
        )
    )
    assert float(losses.margin.detach()) == pytest.approx(
        0.5
        * (
            float(losses.antibody_to_antigen_margin.detach())
            + float(losses.antigen_to_antibody_margin.detach())
        )
    )
    (losses.retrieval + losses.margin).backward()
    assert ab_scores.grad is not None
    assert ag_scores.grad is not None


def test_score_row_accepts_only_independent_centered_residual_scores() -> None:
    import torch

    def output(score: float) -> dict[str, object]:
        return {
            "partner_score": torch.tensor([score]),
            "partner_score_source": (
                "double_centered_independent_specificity_interaction_residual"
            ),
        }

    row = build_single_anchor_score_row(
        output(3.0),
        [output(1.0), output(-1.0)],
    )
    torch.testing.assert_close(row, torch.tensor([[3.0, 1.0, -1.0]]))
    with pytest.raises(ValueError, match="independent centered"):
        require_specificity_partner_score(
            {
                "partner_score": torch.tensor([3.0]),
                "partner_score_source": "double_centered_pair_interactions",
            }
        )


def test_raw_score_reads_only_the_independent_specificity_residual() -> None:
    import torch

    residual = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    mask = torch.ones_like(residual, dtype=torch.bool)
    output = {
        "specificity_interaction_residual_logits": residual,
        "candidate_pair_mask": mask,
        "partner_score_source": (
            "double_centered_independent_specificity_interaction_residual"
        ),
    }
    score = raw_specificity_partner_score(output)
    expected = torch.logsumexp(residual.reshape(1, -1), dim=1) - torch.log(
        torch.tensor(4.0)
    )
    torch.testing.assert_close(score, expected)
    with pytest.raises(ValueError, match="independent specificity"):
        raw_specificity_partner_score(
            {
                **output,
                "partner_score_source": "double_centered_pair_interactions",
            }
        )


@pytest.mark.parametrize(
    ("scores", "mask", "message"),
    [
        ([[1.0, 2.0]], [[False, False]], "at least one positive"),
        ([[1.0, 2.0]], [[True, True]], "at least one negative"),
    ],
)
def test_loss_inputs_fail_closed(
    scores: list[list[float]],
    mask: list[list[bool]],
    message: str,
) -> None:
    import torch

    with pytest.raises(ValueError, match=message):
        multi_positive_retrieval_loss(
            torch.tensor(scores),
            torch.tensor(mask),
            temperature=1.0,
        )
