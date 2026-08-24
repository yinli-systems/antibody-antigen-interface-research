#!/usr/bin/env python3
"""Check matched-local ranking direction, gradients, and candidate-mask guards."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    arguments = parser.parse_args()
    script = arguments.source_root / "scripts/train_framecontact.py"
    specification = importlib.util.spec_from_file_location(
        "matched_local_train", script
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("Cannot load the candidate training script")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)

    decoy = {
        "instance_id": "synthetic",
        "antigen_length": 2,
        "decoy_count": 1,
        "positive_offsets": torch.tensor([0, 1]),
        "positive_flat": torch.tensor([0]),
        "negative_offsets": torch.tensor([0, 1]),
        "negative_flat": torch.tensor([1]),
    }
    mask = torch.ones((1, 2, 2), dtype=torch.bool)
    correctly_ranked = torch.tensor(
        [[[4.0, -4.0], [0.0, 0.0]]], requires_grad=True
    )
    reversed_ranking = torch.tensor(
        [[[-4.0, 4.0], [0.0, 0.0]]], requires_grad=True
    )
    good_loss = module._matched_local_ranking_loss(
        correctly_ranked, mask, [decoy], margin=0.05
    )
    bad_loss = module._matched_local_ranking_loss(
        reversed_ranking, mask, [decoy], margin=0.05
    )
    if not torch.isfinite(good_loss) or not torch.isfinite(bad_loss):
        raise AssertionError("Matched-local loss is not finite")
    if not bool(good_loss < bad_loss):
        raise AssertionError("Matched-local loss rewards the wrong ranking direction")
    good_loss.backward()
    if (
        not bool(torch.isfinite(correctly_ranked.grad).all())
        or not bool(correctly_ranked.grad[0, 0, 0] < 0)
        or not bool(correctly_ranked.grad[0, 0, 1] > 0)
    ):
        raise AssertionError("Matched-local gradient direction is invalid")
    invalid_mask = mask.clone()
    invalid_mask[0, 0, 1] = False
    try:
        module._matched_local_ranking_loss(
            correctly_ranked.detach(), invalid_mask, [decoy], margin=0.05
        )
    except ValueError:
        mask_guard = True
    else:
        raise AssertionError("Matched-local candidate-mask guard did not fire")
    print(
        json.dumps(
            {
                "bad_loss": float(bad_loss),
                "good_loss": float(good_loss),
                "gradient_finite": True,
                "gradient_direction": "positive_up_negative_down",
                "mask_guard": mask_guard,
                "status": "PASS",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
