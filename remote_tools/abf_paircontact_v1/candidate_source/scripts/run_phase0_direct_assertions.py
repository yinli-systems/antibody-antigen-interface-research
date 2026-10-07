#!/usr/bin/env python3
"""Run dependency-free ABF-PairContact Phase-0 assertions."""

from __future__ import annotations

import json

import torch

from antibody_pretraining.stage_a1_batch import _sample_negative_pairs
from antibody_pretraining.stage_a1_objective import StageA1Heads


def main() -> None:
    positives = {(0, 0), (2, 3)}
    negatives = _sample_negative_pairs(4, 5, positives, seed=17, count=8)
    if negatives != _sample_negative_pairs(4, 5, positives, seed=17, count=8):
        raise RuntimeError("Matched negative sampling is not deterministic")
    if set(negatives).intersection(positives) or len(set(negatives)) != 8:
        raise RuntimeError("Matched negative sampling is invalid")
    for left, right in positives:
        if not any(candidate_left == left for candidate_left, _ in negatives):
            raise RuntimeError("A feasible positive row lacks a matched negative")
        if not any(candidate_right == right for _, candidate_right in negatives):
            raise RuntimeError("A feasible positive column lacks a matched negative")

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
    result.total.backward()
    if not bool(torch.isfinite(result.total)):
        raise RuntimeError("Phase-0 total loss is nonfinite")
    if hidden.grad is None or not bool(torch.isfinite(hidden.grad).all()):
        raise RuntimeError("Phase-0 hidden-state gradients are invalid")
    if any(
        parameter.grad is None or not bool(torch.isfinite(parameter.grad).all())
        for parameter in heads.parameters()
    ):
        raise RuntimeError("Phase-0 head gradients are invalid")
    if (
        result.contact_row_comparison_count != 2
        or result.contact_column_comparison_count != 2
    ):
        raise RuntimeError("Phase-0 matched comparison counts differ")

    fail_closed = False
    try:
        heads.loss(
            hidden=hidden.detach(),
            chain_ids=torch.tensor([[0, 0, 1, 1, -1], [0, 0, 1, 1, -1]]),
            mlm_loss=torch.tensor(1.0),
            retrieval_allowed=torch.ones((2, 2), dtype=torch.bool),
            pair_batch_indices=torch.tensor([0, 0]),
            pair_left_token_indices=torch.tensor([0, 1]),
            pair_right_token_indices=torch.tensor([2, 3]),
            pair_labels=torch.tensor([1, 0]),
            contact_ranking_weight=0.5,
        )
    except ValueError as error:
        fail_closed = "row- or column-matched negative" in str(error)
    if not fail_closed:
        raise RuntimeError("Phase-0 unmatched ranking did not fail closed")

    print(
        json.dumps(
            {
                "column_comparisons": result.contact_column_comparison_count,
                "contact_bce": float(result.contact_bce.detach()),
                "contact_centering": float(result.contact_centering.detach()),
                "contact_ranking": float(result.contact_ranking.detach()),
                "matched_negative_count": len(negatives),
                "row_comparisons": result.contact_row_comparison_count,
                "status": "PASS_PHASE0_DIRECT_ASSERTIONS",
                "torch": torch.__version__,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
