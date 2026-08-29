#!/usr/bin/env python3
"""Synthetic GPU smoke for future-only paired conditioning and local EDM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform

import torch


def _autocast(device: torch.device, *, enabled: bool = True):
    """Return a fresh autocast context for each measured forward pass."""
    if device.type == "cuda" and enabled:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.autocast(device_type=device.type, enabled=False)


def _backbone(residue_count: int, *, device: torch.device) -> torch.Tensor:
    ca = torch.stack(
        (
            torch.arange(residue_count, device=device, dtype=torch.float32) * 3.8,
            torch.zeros(residue_count, device=device),
            torch.zeros(residue_count, device=device),
        ),
        dim=-1,
    )
    n = ca + ca.new_tensor([-1.2, 0.7, 0.1])
    c = ca + ca.new_tensor([1.3, 0.2, -0.2])
    return torch.stack((n, ca, c), dim=-2).unsqueeze(0)


def _rotation(angle: float, *, device: torch.device) -> torch.Tensor:
    value = torch.tensor(angle, device=device)
    cosine = torch.cos(value)
    sine = torch.sin(value)
    return torch.stack(
        (
            torch.stack((cosine, -sine, value.new_zeros(()))),
            torch.stack((sine, cosine, value.new_zeros(()))),
            value.new_tensor([0.0, 0.0, 1.0]),
        )
    )


def _transform(
    backbone: torch.Tensor, rotation: torch.Tensor, translation: torch.Tensor
) -> torch.Tensor:
    return torch.einsum("xy,bray->brax", rotation, backbone) + translation


def _batch(device: torch.device) -> dict[str, torch.Tensor]:
    from interaction_first.intra_chain_geometry import build_backbone_frames

    antibody_mask = torch.ones(1, 3, dtype=torch.bool, device=device)
    antigen_mask = torch.ones(1, 5, dtype=torch.bool, device=device)
    antibody_backbone = _backbone(3, device=device)
    antigen_backbone = _backbone(5, device=device)
    antibody_valid = build_backbone_frames(
        antibody_backbone, antibody_mask
    ).valid_mask
    antigen_valid = build_backbone_frames(antigen_backbone, antigen_mask).valid_mask
    pair_labels = torch.zeros(1, 3, 5, device=device)
    pair_labels[0, 1, 2] = 1.0
    return {
        "antibody_features": torch.randn(1, 3, 12, device=device),
        "antigen_features": torch.randn(1, 5, 10, device=device),
        "antibody_paired_sequence_features": torch.randn(1, 3, 6, device=device),
        "antigen_paired_sequence_features": torch.randn(1, 5, 6, device=device),
        "paired_sequence_inter_attention": torch.rand(
            1, 4, 3, 5, device=device
        ).clamp_min(1e-4),
        "antibody_backbone": antibody_backbone,
        "antigen_backbone": antigen_backbone,
        "antibody_mask": antibody_mask,
        "antigen_mask": antigen_mask,
        "antibody_chain_ids": torch.zeros(1, 3, dtype=torch.long, device=device),
        "antigen_chain_ids": torch.zeros(1, 5, dtype=torch.long, device=device),
        "antibody_frame_valid_mask": antibody_valid,
        "antigen_frame_valid_mask": antigen_valid,
        "antibody_missing_frame_mask": antibody_mask & ~antibody_valid,
        "antigen_missing_frame_mask": antigen_mask & ~antigen_valid,
        "pair_labels": pair_labels,
        "antibody_labels": torch.tensor([[0.0, 1.0, 0.0]], device=device),
        "antigen_labels": torch.tensor(
            [[0.0, 0.0, 1.0, 0.0, 0.0]], device=device
        ),
        "antibody_cdr_mask": antibody_mask.clone(),
        "antigen_surface_mask": antigen_mask.clone(),
    }


def _activate_future_paths(model: torch.nn.Module) -> None:
    with torch.no_grad():
        conditioner = model.paired_sequence_conditioner
        conditioner.antibody_residue_gate.fill_(0.35)
        conditioner.antigen_residue_gate.fill_(-0.25)
        conditioner.pair_prior_gate.fill_(0.55)
        for stream in (
            model.self_encoder.antibody_encoder,
            model.self_encoder.antigen_encoder,
        ):
            stream.geometry_residual_gate.fill_(0.45)
        model.cross_attention.antibody_from_antigen.residual_gate.fill_(0.3)
        model.cross_attention.antigen_from_antibody.residual_gate.fill_(0.3)
        model.paratope_contact_gate.fill_(0.2)
        model.epitope_contact_gate.fill_(0.2)


def main() -> None:
    from interaction_first.framecontact import (
        FrameContactModel,
        contact_logit_from_distogram,
    )
    from interaction_first.framecontact_objective import (
        FrameContactLossWeights,
        FrameContactObjective,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()
    device = torch.device(arguments.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.manual_seed(20260801)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(20260801)
    batch = _batch(device)
    model = FrameContactModel(
        antibody_input_dim=12,
        antigen_input_dim=10,
        d_model=16,
        num_heads=4,
        num_self_layers=1,
        neighbor_k=2,
        neighbor_radius=None,
        dropout=0.0,
        pair_output_mode="distogram",
        residue_output_mode="gated_hybrid",
        contact_marginalization="length_normalized_noisy_or",
        paired_sequence_embedding_dim=6,
        paired_sequence_attention_heads=4,
    ).to(device).eval()
    with torch.inference_mode(), _autocast(device):
        baseline = model(batch)
        counterfactual_batch = dict(batch)
        counterfactual_batch["antibody_paired_sequence_features"] = torch.randn_like(
            batch["antibody_paired_sequence_features"]
        )
        counterfactual_batch["antigen_paired_sequence_features"] = torch.randn_like(
            batch["antigen_paired_sequence_features"]
        )
        counterfactual_batch["paired_sequence_inter_attention"] = torch.rand_like(
            batch["paired_sequence_inter_attention"]
        ).clamp_min(1e-4)
        baseline_counterfactual = model(counterfactual_batch)
    zero_gate_max_difference = float(
        (baseline["contact_logits"] - baseline_counterfactual["contact_logits"])
        .abs()
        .max()
        .cpu()
    )
    if zero_gate_max_difference != 0.0:
        raise RuntimeError("zero-gated PPLM path changed the established baseline")

    _activate_future_paths(model)
    transformed_batch = dict(batch)
    transformed_batch["antibody_backbone"] = _transform(
        batch["antibody_backbone"],
        _rotation(0.71, device=device),
        torch.tensor([4.0, -3.0, 7.0], device=device),
    )
    transformed_batch["antigen_backbone"] = _transform(
        batch["antigen_backbone"],
        _rotation(-1.13, device=device),
        torch.tensor([-8.0, 2.0, 1.5], device=device),
    )
    # Check the algebraic invariant in FP32, then measure BF16's numerical
    # envelope separately.  A mixed-precision discrepancy is not evidence that
    # the representation itself violates independent-monomer SE(3) invariance.
    with torch.inference_mode(), _autocast(device, enabled=False):
        active_fp32 = model(batch)
        transformed_fp32 = model(transformed_batch)
    with torch.inference_mode(), _autocast(device):
        active = model(batch)
        transformed = model(transformed_batch)
    active_change = float(
        (active["contact_logits"] - baseline["contact_logits"]).abs().max().cpu()
    )
    if active_change <= 1e-5:
        raise RuntimeError("activated paired conditioning did not change contact logits")
    projected = contact_logit_from_distogram(
        active["distogram_logits"],
        contact_bin_count=model.contact_head.contact_bin_count,
    ).masked_fill(~active["pair_candidate_mask"], 0.0)
    projection_error = float(
        (projected - active["contact_logits"]).abs().max().cpu()
    )
    if projection_error > 5e-4:
        raise RuntimeError("paired prior broke the exact distogram-to-contact projection")
    independent_se3_fp32_error = float(
        (
            active_fp32["contact_logits"]
            - transformed_fp32["contact_logits"]
        )
        .abs()
        .max()
        .cpu()
    )
    independent_se3_bf16_error = float(
        (active["contact_logits"] - transformed["contact_logits"]).abs().max().cpu()
    )
    active_bf16_drift = float(
        (active["contact_logits"] - active_fp32["contact_logits"])
        .abs()
        .max()
        .cpu()
    )
    transformed_bf16_drift = float(
        (transformed["contact_logits"] - transformed_fp32["contact_logits"])
        .abs()
        .max()
        .cpu()
    )
    bf16_numerical_envelope = (
        active_bf16_drift + transformed_bf16_drift + 1e-3
    )
    if independent_se3_fp32_error > 5e-5:
        raise RuntimeError("FP32 independent-monomer SE(3) invariance failed")
    if independent_se3_bf16_error > bf16_numerical_envelope:
        raise RuntimeError(
            "BF16 transform discrepancy exceeds the measured precision envelope"
        )

    model.train()
    model.zero_grad(set_to_none=True)
    with _autocast(device):
        training_output = model(batch)
        objective = FrameContactObjective(
            weights=FrameContactLossWeights(
                contact=0.25,
                paratope=0.0,
                epitope=0.0,
                specificity=0.0,
                distogram=0.0,
                triangle=0.0,
                edm=1.0,
            ),
            antigen_patch_top_seeds=1,
            antigen_patch_radius=20.0,
        )
        losses = objective(training_output, batch)
    losses["loss"].backward()
    pair_gate_gradient = float(
        model.paired_sequence_conditioner.pair_prior_gate.grad.abs().cpu()
    )
    distogram_gradient = float(
        sum(
            parameter.grad.float().abs().sum()
            for parameter in model.contact_head.parameters()
            if parameter.grad is not None
        ).cpu()
    )
    if not torch.isfinite(losses["edm_loss"]) or losses["edm_loss"].item() <= 0:
        raise RuntimeError("local EDM objective did not detect an unrealizable prediction")
    if pair_gate_gradient <= 0 or distogram_gradient <= 0:
        raise RuntimeError("future conditioning/objective gradients did not reach the model")

    payload = {
        "schema_version": "framecontact_frontier_paired_conditioning_smoke_v2",
        "claim_boundary": (
            "Synthetic GPU engineering smoke only; no trained improvement, external "
            "generalization, benchmark, novelty, or production-performance claim."
        ),
        "source_manifest_sha256": arguments.source_manifest_sha256,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
        "gpu_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "bf16_supported": (
            torch.cuda.is_bf16_supported() if device.type == "cuda" else None
        ),
        "checks": {
            "zero_gate_max_difference": zero_gate_max_difference,
            "activated_contact_change": active_change,
            "distogram_projection_max_error": projection_error,
            "independent_monomer_se3_fp32_max_error": (
                independent_se3_fp32_error
            ),
            "independent_monomer_se3_bf16_max_error": (
                independent_se3_bf16_error
            ),
            "active_bf16_vs_fp32_max_error": active_bf16_drift,
            "transformed_bf16_vs_fp32_max_error": transformed_bf16_drift,
            "bf16_numerical_envelope": bf16_numerical_envelope,
            "local_edm_loss": float(losses["edm_loss"].detach().cpu()),
            "total_training_loss": float(losses["loss"].detach().cpu()),
            "paired_prior_gate_gradient_abs": pair_gate_gradient,
            "distogram_head_gradient_l1": distogram_gradient,
        },
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    if arguments.output.exists() or arguments.output.is_symlink():
        raise FileExistsError(f"refusing to overwrite {arguments.output}")
    arguments.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
