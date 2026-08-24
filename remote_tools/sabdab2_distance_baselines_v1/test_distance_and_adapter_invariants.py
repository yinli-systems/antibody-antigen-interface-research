#!/usr/bin/env python3
"""CPU invariants for the leakage-safe distance head and protocol adapters."""

from __future__ import annotations

import argparse
import inspect
import math
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    arguments = parser.parse_args()
    sys.path.insert(0, str(arguments.source_root / "src"))

    import torch

    from interaction_first.framecontact import (
        BidirectionalPairContactHead,
        PairDistanceBinHead,
    )
    from interaction_first.framecontact_objective import (
        FrameContactLossWeights,
        FrameContactObjective,
    )

    torch.manual_seed(7)
    batch_size, antibody_count, antigen_count, d_model, bin_count = 2, 3, 4, 8, 5
    antibody = torch.randn(batch_size, antibody_count, d_model, requires_grad=True)
    antigen = torch.randn(batch_size, antigen_count, d_model, requires_grad=True)
    pair_mask = torch.ones(
        batch_size, antibody_count, antigen_count, dtype=torch.bool
    )
    distance_head = PairDistanceBinHead(d_model, num_bins=bin_count, pair_dim=4)
    if tuple(inspect.signature(distance_head.forward).parameters) != (
        "antibody",
        "antigen",
        "pair_mask",
    ):
        raise AssertionError("distance head accepts an unauthorized input")
    coordinates = (
        torch.randn(batch_size, antibody_count, 3),
        torch.randn(batch_size, antigen_count, 3),
    )
    distance_logits = distance_head(antibody, antigen, pair_mask)
    coordinates = (coordinates[0] + 1000.0, coordinates[1] - 777.0)
    if not torch.equal(distance_logits, distance_head(antibody, antigen, pair_mask)):
        raise AssertionError("distance-head output depends on external coordinates")

    pair_head = BidirectionalPairContactHead(d_model, pair_dim=4)
    forward = pair_head(antibody, antigen, pair_mask)
    reverse = pair_head(
        antigen, antibody, pair_mask.transpose(1, 2)
    ).transpose(1, 2)
    if not torch.allclose(forward, reverse, atol=1e-6, rtol=1e-6):
        raise AssertionError("bidirectional decoder is not swap-symmetric")

    pair_labels = torch.zeros(
        batch_size, antibody_count, antigen_count, dtype=torch.long
    )
    pair_labels[:, 0, 0] = 1
    output = {
        "contact_logits": torch.randn(
            batch_size, antibody_count, antigen_count, requires_grad=True
        ),
        "contact_interaction_logits": torch.randn(
            batch_size, antibody_count, antigen_count, requires_grad=True
        ),
        "pair_candidate_mask": pair_mask,
        "paratope_logits": torch.randn(
            batch_size, antibody_count, requires_grad=True
        ),
        "epitope_logits": torch.randn(batch_size, antigen_count, requires_grad=True),
        "antibody_mask": torch.ones(
            batch_size, antibody_count, dtype=torch.bool
        ),
        "antigen_mask": torch.ones(batch_size, antigen_count, dtype=torch.bool),
        "distance_bin_logits": distance_logits,
        "contrastive_views": {
            "antibody_sequence": torch.randn(
                batch_size, antibody_count, 6, requires_grad=True
            ),
            "antibody_geometry": torch.randn(
                batch_size, antibody_count, 6, requires_grad=True
            ),
            "antigen_sequence": torch.randn(
                batch_size, antigen_count, 6, requires_grad=True
            ),
            "antigen_geometry": torch.randn(
                batch_size, antigen_count, 6, requires_grad=True
            ),
        },
    }
    batch = {
        "pair_labels": pair_labels,
        "antibody_labels": torch.tensor([[1, 0, 0], [0, 1, 0]]),
        "antigen_labels": torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0]]),
        "antibody_backbone": torch.randn(
            batch_size, antibody_count, 3, 3
        ),
        "antigen_backbone": torch.randn(batch_size, antigen_count, 3, 3),
        "antibody_frame_valid_mask": torch.ones(
            batch_size, antibody_count, dtype=torch.bool
        ),
        "antigen_frame_valid_mask": torch.ones(
            batch_size, antigen_count, dtype=torch.bool
        ),
    }
    objective = FrameContactObjective(
        weights=FrameContactLossWeights(
            specificity=0.0, distance=0.1, contrastive=0.05
        )
    )
    losses = objective(output, batch)
    if not all(
        math.isfinite(float(value.detach()))
        for value in losses.values()
        if torch.is_tensor(value)
    ):
        raise AssertionError("objective produced a non-finite value")
    losses["loss"].backward()
    if distance_head.antibody_query.weight.grad is None:
        raise AssertionError("distance auxiliary head received no gradient")
    print(
        {
            "status": "PASS",
            "distance_shape": tuple(distance_logits.shape),
            "coordinate_input_to_head": False,
            "bidirectional_decoder_swap_symmetric": True,
        }
    )


if __name__ == "__main__":
    main()
