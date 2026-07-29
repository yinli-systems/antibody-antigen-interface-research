"""Tests for score-free centered multi-decoy PAIRScale objectives."""

from __future__ import annotations

from dataclasses import replace
import importlib.util

import pytest

from interaction_first.development_contact_dataset import DevelopmentContactExample
from interaction_first.pairscale_centered_multidecoy import (
    DECOY_COUNT,
    centered_multidecoy_partner_ranking_loss,
    locked_centered_multidecoy_config,
    select_score_free_length_matched_decoys,
)
from interaction_first.pairscale_no_centering_training import locked_no_centering_config


def _example(index: int, antigen_id: str, antigen_length: int) -> DevelopmentContactExample:
    import torch

    return DevelopmentContactExample(
        instance_id=f"instance-{index}",
        antibody_id=f"antibody-{index}",
        antigen_id=antigen_id,
        antibody_embeddings=torch.full((3, 2), float(index)),
        antigen_embeddings=torch.full((antigen_length, 2), float(index)),
        contact_targets=torch.full((3, antigen_length), index % 2),
        contact_mask=torch.ones((3, antigen_length), dtype=torch.bool),
        positive_pair_count=index + 1,
    )


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="PyTorch is unavailable",
)
def test_decoy_selection_is_distinct_length_matched_and_deterministic() -> None:
    examples = tuple(
        _example(index, f"antigen-{index}", 10 + index)
        for index in range(8)
    )
    first = select_score_free_length_matched_decoys(examples, seed=17)
    replay = select_score_free_length_matched_decoys(examples, seed=17)
    assert first == replay
    assert all(len(indices) == DECOY_COUNT for indices in first)
    for matched_index, indices in enumerate(first):
        assert len({examples[index].antigen_id for index in indices}) == DECOY_COUNT
        assert matched_index not in indices
    closest_distance = min(abs((10 + index) - 10) for index in range(1, 8))
    assert abs((10 + first[0][0]) - 10) == closest_distance


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="PyTorch is unavailable",
)
def test_decoy_selection_does_not_read_values_targets_or_positive_count() -> None:
    import torch

    examples = tuple(
        _example(index, f"antigen-{index}", 10 + index)
        for index in range(8)
    )
    baseline = select_score_free_length_matched_decoys(examples, seed=17)
    altered = tuple(
        replace(
            example,
            antibody_embeddings=torch.full_like(example.antibody_embeddings, float("nan")),
            antigen_embeddings=torch.full_like(example.antigen_embeddings, float("nan")),
            contact_targets=torch.full_like(example.contact_targets, 99),
            contact_mask=torch.zeros_like(example.contact_mask),
            positive_pair_count=10_000 + index,
        )
        for index, example in enumerate(examples)
    )
    assert select_score_free_length_matched_decoys(altered, seed=17) == baseline


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="PyTorch is unavailable",
)
def test_centered_multidecoy_loss_rewards_partner_separation_and_backpropagates() -> None:
    import torch

    mask = torch.ones((1, 3, 3), dtype=torch.bool)
    matched_logits = torch.tensor(
        [[[8.0, -2.0, -2.0], [-2.0, 7.0, -2.0], [-2.0, -2.0, 6.0]]],
        requires_grad=True,
    )
    weak_logits = torch.zeros((1, 3, 3), requires_grad=True)
    decoys = [
        {"contact_logits": weak_logits + offset, "candidate_pair_mask": mask}
        for offset in (0.0, 0.1, -0.1, 0.2)
    ]
    separated = centered_multidecoy_partner_ranking_loss(
        {"contact_logits": matched_logits, "candidate_pair_mask": mask},
        decoys,
        margin=0.1,
        temperature=1.0,
    )
    inverted = centered_multidecoy_partner_ranking_loss(
        {"contact_logits": weak_logits, "candidate_pair_mask": mask},
        [
            {"contact_logits": matched_logits + offset, "candidate_pair_mask": mask}
            for offset in (0.0, 0.1, -0.1, 0.2)
        ],
        margin=0.1,
        temperature=1.0,
    )
    assert float(separated.detach()) < float(inverted.detach())
    separated.backward()
    assert matched_logits.grad is not None
    assert bool(torch.isfinite(matched_logits.grad).all())


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="PyTorch is unavailable",
)
def test_multidecoy_loss_is_invariant_to_decoy_order() -> None:
    import torch

    mask = torch.ones((1, 2, 2), dtype=torch.bool)
    matched = {"contact_logits": torch.randn(1, 2, 2), "candidate_pair_mask": mask}
    decoys = [
        {"contact_logits": torch.randn(1, 2, 2), "candidate_pair_mask": mask}
        for _ in range(4)
    ]
    first = centered_multidecoy_partner_ranking_loss(
        matched,
        decoys,
        margin=0.1,
        temperature=1.0,
    )
    reverse = centered_multidecoy_partner_ranking_loss(
        matched,
        list(reversed(decoys)),
        margin=0.1,
        temperature=1.0,
    )
    assert float(first) == pytest.approx(float(reverse), abs=1e-7)


def test_repair_config_changes_no_architecture_or_optimizer_fields() -> None:
    assert locked_centered_multidecoy_config().canonical_payload() == (
        locked_no_centering_config().canonical_payload()
    )
