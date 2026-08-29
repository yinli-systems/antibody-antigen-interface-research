#!/usr/bin/env python3
"""GPU smoke for the alignment-free local EDM realizability diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import torch

from interaction_first.framecontact_objective import (
    local_edm_realizability_loss,
    sparse_triangle_inequality_loss,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def backbone(ca: torch.Tensor) -> torch.Tensor:
    output = torch.zeros(1, ca.shape[0], 3, 3, device=ca.device)
    output[0, :, 0, :] = ca + ca.new_tensor([-1.0, 0.0, 0.0])
    output[0, :, 1, :] = ca
    output[0, :, 2, :] = ca + ca.new_tensor([1.0, 0.0, 0.0])
    return output


def rigid_transform(
    points: torch.Tensor, rotation: torch.Tensor, shift: torch.Tensor
) -> torch.Tensor:
    return points @ rotation.transpose(0, 1) + shift


def main() -> None:
    arguments = parse_arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("This smoke requires CUDA")
    device = torch.device("cuda")
    active = torch.ones(1, 2, dtype=torch.bool, device=device)
    chain = torch.zeros(1, 2, dtype=torch.long, device=device)

    antibody_ca = torch.tensor(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], device=device
    )
    antigen_ca = torch.tensor(
        [[0.0, 2.0, 1.0], [2.0, 2.0, 1.0]], device=device
    )
    true_prediction = torch.cdist(antibody_ca, antigen_ca)[None].requires_grad_(True)
    pair_mask = torch.ones_like(true_prediction, dtype=torch.bool)
    true_loss = local_edm_realizability_loss(
        true_prediction,
        pair_mask,
        backbone(antibody_ca),
        backbone(antigen_ca),
        active,
        active,
    )
    true_loss.backward()
    if true_loss.item() >= 1e-10 or not torch.isfinite(true_prediction.grad).all():
        raise AssertionError("a true joint embedding did not pass the EDM diagnostic")

    angle = torch.tensor(0.7, device=device)
    cosine, sine = torch.cos(angle), torch.sin(angle)
    antibody_rotation = torch.stack(
        (
            torch.stack((cosine, -sine, angle.new_zeros(()))),
            torch.stack((sine, cosine, angle.new_zeros(()))),
            angle.new_tensor([0.0, 0.0, 1.0]),
        )
    )
    antigen_rotation = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], device=device
    )
    transformed_loss = local_edm_realizability_loss(
        true_prediction.detach(),
        pair_mask,
        backbone(
            rigid_transform(
                antibody_ca, antibody_rotation, antibody_ca.new_tensor([3.0, -2.0, 5.0])
            )
        ),
        backbone(
            rigid_transform(
                antigen_ca, antigen_rotation, antigen_ca.new_tensor([-4.0, 1.0, 2.0])
            )
        ),
        active,
        active,
    )
    if abs(transformed_loss.item() - true_loss.item()) >= 1e-10:
        raise AssertionError("EDM diagnostic changed under independent rigid transforms")

    antibody_ca = torch.tensor(
        [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], device=device
    )
    antigen_ca = torch.tensor(
        [[0.0, -1.0, 0.0], [0.0, 1.0, 0.0]], device=device
    )
    constant = torch.ones(1, 2, 2, device=device, requires_grad=True)
    pair_mask = torch.ones_like(constant, dtype=torch.bool)
    triangle_loss = sparse_triangle_inequality_loss(
        constant,
        pair_mask,
        backbone(antibody_ca),
        backbone(antigen_ca),
        chain,
        chain,
        active,
        active,
        neighbor_radius=3.0,
        neighbor_k=1,
        slack=0.0,
    )
    edm_loss = local_edm_realizability_loss(
        constant,
        pair_mask,
        backbone(antibody_ca),
        backbone(antigen_ca),
        active,
        active,
    )
    edm_loss.backward()
    gradient_l1 = constant.grad.abs().sum()
    if triangle_loss.item() != 0.0 or edm_loss.item() <= 1e-4:
        raise AssertionError("constant-distance triangle escape was not distinguished")
    if not torch.isfinite(constant.grad).all() or gradient_l1.item() <= 0:
        raise AssertionError("EDM diagnostic did not produce a finite useful gradient")

    source_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "interaction_first"
        / "framecontact_objective.py"
    )
    receipt = {
        "format_version": "framecontact_local_edm_gpu_smoke_v1",
        "claim_boundary": (
            "Synthetic GPU engineering smoke only. This is not a trained-model, "
            "generalization, benchmark, or novelty result."
        ),
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
        },
        "source_sha256": sha256_file(source_path),
        "true_joint_edm_loss": true_loss.item(),
        "independent_rigid_transform_edm_loss": transformed_loss.item(),
        "constant_cross_triangle_loss": triangle_loss.item(),
        "constant_cross_edm_loss": edm_loss.item(),
        "constant_cross_gradient_l1": gradient_l1.item(),
        "valid": True,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
