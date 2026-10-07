from __future__ import annotations

import torch

from antibody_pretraining.contact_guided_cdr_generation import (
    contact_guidance_objective,
    generate_contact_guided_cdr_proposals,
)


class ToyDenoiser:
    def __call__(self, antibodies, antigens, timesteps):
        logits = torch.full((*antibodies.shape, 8), -8.0)
        logits[..., 1] = 1.0
        logits[..., 2] = 1.0
        return logits


class TokenHotspotScorer:
    def __call__(self, antibodies, antigens):
        batch, antibody_length = antibodies.shape
        antigen_length = antigens.shape[1]
        logits = torch.full((batch, antibody_length, antigen_length), -4.0)
        logits[:, :, 0] = antibodies.eq(2).float() * 8.0 - 4.0
        return logits


def test_contact_guided_generation_preserves_framework_and_is_deterministic() -> None:
    arguments = dict(
        framework_tokens=torch.tensor([5, 5, 5, 5, 5]),
        antigen_tokens=torch.tensor([6, 6, 6]),
        cdr_mask=torch.tensor([False, True, True, False, False]),
        hotspot_mask=torch.tensor([True, False, False]),
        denoiser=ToyDenoiser(),
        contact_scorer=TokenHotspotScorer(),
        canonical_token_ids=[1, 2],
        mask_token_id=7,
        steps=3,
        proposals_per_step=12,
        output_count=3,
        contact_guidance_weight=4.0,
        seed=23,
    )
    first = generate_contact_guided_cdr_proposals(**arguments)
    second = generate_contact_guided_cdr_proposals(**arguments)
    assert first == second
    assert len(first) == 3
    for proposal in first:
        assert proposal.antibody_token_ids[0] == 5
        assert proposal.antibody_token_ids[3:] == (5, 5)
        assert set(proposal.antibody_token_ids[1:3]).issubset({1, 2})
    assert first[0].hotspot_contact_score > 0.5


def test_contact_objective_penalizes_off_target_contacts() -> None:
    pair_logits = torch.tensor(
        [
            [[4.0, -4.0], [4.0, -4.0]],
            [[4.0, 4.0], [4.0, 4.0]],
        ]
    )
    objective, hotspot, off_target = contact_guidance_objective(
        pair_logits,
        torch.tensor([True, True]),
        torch.tensor([True, False]),
        off_target_weight=1.0,
    )
    assert hotspot[0] == hotspot[1]
    assert off_target[0] < off_target[1]
    assert objective[0] > objective[1]


def test_generation_rejects_empty_hotspot() -> None:
    try:
        generate_contact_guided_cdr_proposals(
            torch.tensor([5, 5]),
            torch.tensor([6, 6]),
            torch.tensor([False, True]),
            torch.tensor([False, False]),
            ToyDenoiser(),
            TokenHotspotScorer(),
            [1, 2],
            mask_token_id=7,
        )
    except ValueError as error:
        assert "non-empty" in str(error)
    else:
        raise AssertionError("empty hotspot was not rejected")


def test_generation_rejects_masked_framework_position() -> None:
    try:
        generate_contact_guided_cdr_proposals(
            torch.tensor([7, 5]),
            torch.tensor([6, 6]),
            torch.tensor([False, True]),
            torch.tensor([True, False]),
            ToyDenoiser(),
            TokenHotspotScorer(),
            [1, 2],
            mask_token_id=7,
        )
    except ValueError as error:
        assert "framework" in str(error)
    else:
        raise AssertionError("masked framework position was not rejected")
