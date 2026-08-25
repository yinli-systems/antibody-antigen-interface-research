#!/usr/bin/env python3
"""Focused contract test for the masked EpiFormer top-2 marginal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source_root / "src"))

    import torch

    from interaction_first.framecontact import (
        masked_top_k_mean_probability_logits,
    )

    logits = torch.tensor(
        [[[0.0, 2.0, -2.0, 1.0], [3.0, -1.0, 0.5, -4.0]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    mask = torch.tensor(
        [[[True, True, False, True], [True, False, True, False]]],
        dtype=torch.bool,
    )
    pooled_logits = masked_top_k_mean_probability_logits(
        logits, mask, dim=2, k=2
    )
    expected = torch.stack(
        [
            torch.sigmoid(logits[0, 0, torch.tensor([1, 3])]).mean(),
            torch.sigmoid(logits[0, 1, torch.tensor([0, 2])]).mean(),
        ]
    ).unsqueeze(0)
    observed = torch.sigmoid(pooled_logits.float())
    torch.testing.assert_close(observed, expected, rtol=1e-6, atol=1e-6)

    sparse_mask = torch.tensor(
        [[[True, False], [False, False]]], dtype=torch.bool
    )
    sparse_logits = torch.tensor(
        [[[0.75, -9.0], [4.0, -4.0]]], dtype=torch.float32
    )
    sparse = masked_top_k_mean_probability_logits(
        sparse_logits, sparse_mask, dim=2, k=2
    )
    torch.testing.assert_close(
        torch.sigmoid(sparse[0, 0].float()),
        torch.sigmoid(sparse_logits[0, 0, 0]),
        rtol=1e-6,
        atol=1e-6,
    )
    if float(sparse[0, 1]) != 0.0:
        raise AssertionError("empty observed set must map to neutral zero logit")

    pooled_logits.sum().backward()
    if logits.grad is None or not bool(torch.isfinite(logits.grad).all()):
        raise AssertionError("top-2 marginal gradient must be finite")
    if bool((logits.grad[~mask]).ne(0).any()):
        raise AssertionError("masked pairs must not receive marginal gradients")

    print(
        json.dumps(
            {
                "status": "PASS_EPIFORMER_TOP2_MARGINAL_CONTRACT",
                "shape": list(pooled_logits.shape),
                "finite_gradient": True,
                "masked_gradient_zero": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
