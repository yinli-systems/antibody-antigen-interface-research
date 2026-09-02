#!/usr/bin/env python3
"""Unit and GPU-canary checks for the frozen FrameContact modifications."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args()


def backbone(length: int, *, device: torch.device) -> torch.Tensor:
    index = torch.arange(length, dtype=torch.float32, device=device)
    ca = torch.stack((index, torch.zeros_like(index), torch.zeros_like(index)), dim=-1)
    n = ca + torch.tensor([0.0, 1.0, 0.0], device=device)
    c = ca + torch.tensor([0.0, 0.0, 1.0], device=device)
    return torch.stack((n, ca, c), dim=1).unsqueeze(0)


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.source_root / "src"))
    from interaction_first.framecontact import (  # pylint: disable=import-outside-toplevel
        FrameContactModel,
        chain_internal_ca_surface_score,
    )
    from interaction_first.framecontact_objective import (  # pylint: disable=import-outside-toplevel
        hard_negative_pair_ranking_loss_per_complex,
    )

    device = torch.device(args.device)
    torch.manual_seed(20260830)
    antigen_backbone = backbone(5, device=device)
    antigen_mask = torch.ones((1, 5), dtype=torch.bool, device=device)
    antigen_chain_ids = torch.tensor([[0, 0, 0, 1, 1]], device=device)
    score = chain_internal_ca_surface_score(
        antigen_backbone, antigen_mask, antigen_chain_ids, radius=10.0
    )
    for chain_id in (0, 1):
        local = antigen_chain_ids == chain_id
        assert abs(float(score[local].mean())) < 1e-5

    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        device=device,
    )
    transformed = antigen_backbone @ rotation.T + torch.tensor(
        [7.0, -3.0, 2.0], device=device
    )
    transformed_score = chain_internal_ca_surface_score(
        transformed, antigen_mask, antigen_chain_ids, radius=10.0
    )
    torch.testing.assert_close(score, transformed_score, rtol=1e-5, atol=1e-5)

    ranking_labels = torch.tensor([[[1, 1, 0, 0]]], device=device)
    ranking_mask = torch.ones_like(ranking_labels, dtype=torch.bool)
    bad = hard_negative_pair_ranking_loss_per_complex(
        torch.tensor([[[-1.0, -0.5, 2.0, 1.0]]], device=device),
        ranking_labels,
        ranking_mask,
    )
    good = hard_negative_pair_ranking_loss_per_complex(
        torch.tensor([[[2.0, 1.0, -1.0, -2.0]]], device=device),
        ranking_labels,
        ranking_mask,
    )
    assert math.isfinite(float(bad)) and math.isfinite(float(good))
    assert float(good) < float(bad)

    common = dict(
        antibody_input_dim=4,
        antigen_input_dim=4,
        d_model=8,
        num_heads=2,
        num_self_layers=1,
        num_rbf=4,
        neighbor_k=4,
        neighbor_radius=16.0,
        sequence_window=4,
        dropout=0.0,
        fusion_mode="interleaved",
        cross_attention_contact_gate_mode="content_only",
        adapter_mode="epiformer_style_same_protocol",
    )
    original = FrameContactModel(**common, surface_prior_mode="none").to(device).eval()
    modified = FrameContactModel(
        **common, surface_prior_mode="ca_density", surface_radius=10.0
    ).to(device).eval()
    missing, unexpected = modified.load_state_dict(original.state_dict(), strict=False)
    assert missing == ["antigen_surface_contact_gate"] and not unexpected
    assert not any("surface" in key for key in original.state_dict())

    antibody_backbone = backbone(3, device=device)
    batch = {
        "antibody_features": torch.randn((1, 3, 4), device=device),
        "antigen_features": torch.randn((1, 5, 4), device=device),
        "antibody_mask": torch.ones((1, 3), dtype=torch.bool, device=device),
        "antigen_mask": antigen_mask,
        "antibody_backbone": antibody_backbone,
        "antigen_backbone": antigen_backbone,
        "antibody_chain_ids": torch.zeros((1, 3), dtype=torch.long, device=device),
        "antigen_chain_ids": antigen_chain_ids,
        "antibody_frame_valid_mask": torch.ones(
            (1, 3), dtype=torch.bool, device=device
        ),
        "antigen_frame_valid_mask": antigen_mask.clone(),
        "antibody_missing_frame_mask": torch.zeros(
            (1, 3), dtype=torch.bool, device=device
        ),
        "antigen_missing_frame_mask": torch.zeros(
            (1, 5), dtype=torch.bool, device=device
        ),
    }
    with torch.no_grad():
        original_output = original(batch)
        modified_output = modified(batch)
    for key in ("contact_logits", "paratope_logits", "epitope_logits"):
        assert torch.equal(original_output[key], modified_output[key]), key

    modified.train()
    output = modified(batch)
    pair_labels = torch.zeros_like(output["contact_logits"], dtype=torch.long)
    pair_labels[:, 0, 0] = 1
    ranking = hard_negative_pair_ranking_loss_per_complex(
        output["contact_logits"],
        pair_labels,
        output["pair_candidate_mask"],
        negative_fraction=0.01,
        maximum_pairs=64,
        margin=0.5,
    )
    loss = output["contact_logits"].square().mean() + 0.25 * ranking
    loss.backward()
    assert torch.isfinite(loss)
    assert modified.antigen_surface_contact_gate is not None
    assert modified.antigen_surface_contact_gate.grad is not None
    assert torch.isfinite(modified.antigen_surface_contact_gate.grad)
    print(
        {
            "status": "PASS_SURFACE_HARDNEGATIVE_CANARY",
            "device": str(device),
            "ranking_bad": float(bad),
            "ranking_good": float(good),
            "loss": float(loss),
        }
    )


if __name__ == "__main__":
    main()
