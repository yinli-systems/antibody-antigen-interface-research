#!/usr/bin/env python3
"""Verify optimized invariant-point bias against the naive six-dimensional form."""

from __future__ import annotations

import argparse
import json

import torch

from interaction_first.intra_chain_geometry import (
    InvariantPointHeadBias,
    build_backbone_frames,
    build_intra_chain_relative_geometry,
)


def _sample_backbone(residues: int) -> torch.Tensor:
    ca = torch.stack(
        (
            torch.arange(residues, dtype=torch.float32) * 3.8,
            torch.sin(torch.arange(residues, dtype=torch.float32) * 0.4),
            torch.cos(torch.arange(residues, dtype=torch.float32) * 0.3),
        ),
        dim=-1,
    )
    n = ca + torch.tensor([-1.2, 0.8, 0.1])
    c = ca + torch.tensor([1.3, 0.2, 0.4])
    return torch.stack((n, ca, c), dim=-2).unsqueeze(0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--residues", type=int, default=6)
    parser.add_argument("--heads", type=int, default=3)
    parser.add_argument("--points", type=int, default=4)
    arguments = parser.parse_args()
    if min(arguments.residues, arguments.heads, arguments.points) <= 0:
        raise ValueError("residues, heads, and points must be positive")
    torch.manual_seed(59)
    residue_mask = torch.ones(1, arguments.residues, dtype=torch.bool)
    geometry = build_intra_chain_relative_geometry(
        build_backbone_frames(_sample_backbone(arguments.residues), residue_mask),
        residue_mask,
        geometry_feature_mode="none",
        neighbor_k=None,
    )
    layer = InvariantPointHeadBias(
        arguments.heads, num_points=arguments.points
    )
    optimized = layer(geometry)
    relative_key = torch.einsum(
        "bijxy,hpy->bijhpx",
        geometry.relative_rotation,
        layer.key_points,
    )
    relative_key = relative_key + geometry.displacement_local[..., None, None, :]
    delta = layer.query_points[None, None, None, ...] - relative_key
    naive_distance = delta.square().sum(dim=-1).mean(dim=-1)
    head_weight = torch.nn.functional.softplus(layer.head_weight_unconstrained)
    naive = (
        -0.5
        * naive_distance
        * head_weight[None, None, None, :]
        / (layer.distance_scale**2)
    )
    naive = naive * geometry.geometry_valid_mask[..., None].to(naive.dtype)
    naive = naive.movedim(-1, 1)
    torch.testing.assert_close(optimized, naive, atol=3e-6, rtol=3e-6)
    probe = torch.randn_like(optimized)
    (optimized * probe).sum().backward(retain_graph=True)
    optimized_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in layer.named_parameters()
    }
    layer.zero_grad(set_to_none=True)
    (naive * probe).sum().backward()
    for name, parameter in layer.named_parameters():
        torch.testing.assert_close(
            optimized_gradients[name],
            parameter.grad,
            atol=2e-5,
            rtol=2e-5,
        )
    naive_delta_elements = (
        arguments.residues**2
        * arguments.heads
        * arguments.points
        * 3
    )
    optimized_pair_head_elements = (
        arguments.residues**2 * arguments.heads
    )
    print(
        json.dumps(
            {
                "format_version": "invariant_point_memory_equivalence_v1",
                "value_equivalent": True,
                "gradient_equivalent": True,
                "naive_delta_elements": naive_delta_elements,
                "optimized_pair_head_elements": optimized_pair_head_elements,
                "delta_to_pair_head_element_ratio": (
                    naive_delta_elements / optimized_pair_head_elements
                ),
                "claim_boundary": (
                    "Algebraic value/gradient equivalence on a deterministic "
                    "synthetic input; not a GPU peak-memory or biological result."
                ),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

