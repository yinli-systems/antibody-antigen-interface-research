#!/usr/bin/env python3
"""Fail-closed CUDA smoke test for the six factorized experiments."""

from __future__ import annotations

import argparse
import json
import re

import torch

from interaction_first.sabdab2_structure import IntraChainResidueClassifier


EXPERIMENTS = {
    "sequence_dense": ("none", "dense"),
    "sequence_local": ("none", "sequence"),
    "spatial_topology": ("none", "spatial"),
    "distance": ("distance", "spatial"),
    "distance_direction": ("distance_direction", "spatial"),
    "full_pose": ("full_pose", "spatial"),
}


def _backbone(batch: int, residues: int, device: torch.device) -> torch.Tensor:
    ca = torch.cumsum(torch.randn(batch, residues, 3, device=device), dim=1)
    n_atom = ca + torch.tensor([-1.1, 0.8, 0.2], device=device)
    c_atom = ca + torch.tensor([1.3, 0.2, -0.1], device=device)
    return torch.stack((n_atom, ca, c_atom), dim=-2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-name-regex", default=r"A100")
    arguments = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(device)
    if re.search(arguments.gpu_name_regex, gpu_name, flags=re.IGNORECASE) is None:
        raise RuntimeError(
            f"GPU {gpu_name!r} does not match required regex {arguments.gpu_name_regex!r}"
        )
    torch.manual_seed(20260729)
    batch_size, antibody_length, antigen_length, input_dim = 2, 48, 64, 128
    antibody_backbone = _backbone(batch_size, antibody_length, device)
    antigen_backbone = _backbone(batch_size, antigen_length, device)
    batch = {
        "instance_ids": ("smoke_0", "smoke_1"),
        "antibody_features": torch.randn(
            batch_size, antibody_length, input_dim, device=device
        ),
        "antigen_features": torch.randn(
            batch_size, antigen_length, input_dim, device=device
        ),
        "antibody_backbone": antibody_backbone,
        "antigen_backbone": antigen_backbone,
        "antibody_mask": torch.ones(
            batch_size, antibody_length, dtype=torch.bool, device=device
        ),
        "antigen_mask": torch.ones(
            batch_size, antigen_length, dtype=torch.bool, device=device
        ),
        "antibody_chain_ids": torch.zeros(
            batch_size, antibody_length, dtype=torch.long, device=device
        ),
        "antigen_chain_ids": torch.zeros(
            batch_size, antigen_length, dtype=torch.long, device=device
        ),
        "antibody_frame_valid_mask": torch.ones(
            batch_size, antibody_length, dtype=torch.bool, device=device
        ),
        "antigen_frame_valid_mask": torch.ones(
            batch_size, antigen_length, dtype=torch.bool, device=device
        ),
        "antibody_missing_frame_mask": torch.zeros(
            batch_size, antibody_length, dtype=torch.bool, device=device
        ),
        "antigen_missing_frame_mask": torch.zeros(
            batch_size, antigen_length, dtype=torch.bool, device=device
        ),
    }
    parameter_counts = set()
    results = []
    for experiment, (geometry_mode, graph_mode) in EXPERIMENTS.items():
        torch.manual_seed(7)
        model = IntraChainResidueClassifier(
            antibody_input_dim=input_dim,
            antigen_input_dim=input_dim,
            d_model=64,
            num_heads=4,
            num_layers=1,
            geometry_feature_mode=geometry_mode,
            attention_graph_mode=graph_mode,
            num_rbf=16,
            neighbor_k=32,
            neighbor_radius=16.0,
            sequence_window=32,
            dropout=0.0,
        ).to(device)
        parameter_counts.add(sum(parameter.numel() for parameter in model.parameters()))
        model.train()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(batch, return_attention=True)
            loss = output["paratope_logits"].square().mean()
            loss = loss + output["epitope_logits"].square().mean()
        loss.backward()
        end_event.record()
        torch.cuda.synchronize(device)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss for {experiment}")
        if output["attention_details"] is None:
            raise RuntimeError(f"Missing attention details for {experiment}")
        results.append(
            {
                "experiment": experiment,
                "loss": float(loss.detach().cpu()),
                "single_pass_forward_backward_ms": float(
                    start_event.elapsed_time(end_event)
                ),
                "peak_memory_allocated_bytes": int(
                    torch.cuda.max_memory_allocated(device)
                ),
            }
        )
        del model, output, loss
    if len(parameter_counts) != 1:
        raise RuntimeError("Six experiments do not have equal parameter counts")
    torch.cuda.synchronize()
    print(
        json.dumps(
            {
                "status": "pass",
                "gpu": gpu_name,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "bf16_supported": torch.cuda.is_bf16_supported(),
                "model_parameter_count": next(iter(parameter_counts)),
                "synthetic_batch": {
                    "batch_size": batch_size,
                    "antibody_residues": antibody_length,
                    "antigen_residues": antigen_length,
                    "input_dim": input_dim,
                },
                "measurement_scope": (
                    "one synthetic BF16 forward+backward pass per experiment; "
                    "correctness and gross resource smoke only, not a benchmark"
                ),
                "experiments": results,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
