#!/usr/bin/env python3
"""Exercise the isolated topology-routed model on one real AsEP train batch."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as functional


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from interaction_first.asep_external import AsepDevelopmentDataset  # noqa: E402
from interaction_first.sabdab2_contacts import (  # noqa: E402
    Sabdab2ContactBatchCollator,
)
from interaction_first.sabdab2_structure import (  # noqa: E402
    authenticate_file,
    canonical_json_bytes,
    sha256_file,
)
from interaction_first.topology_routed_attention import (  # noqa: E402
    TopologyRoutedFrameContactModel,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--cache-manifest-sha256", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--protocol-receipt", type=Path, required=True)
    parser.add_argument("--protocol-receipt-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    return parser.parse_args()


def _loss(output: dict[str, torch.Tensor], batch: dict[str, object]) -> torch.Tensor:
    pair_mask = output["pair_candidate_mask"]
    antibody_mask = output["antibody_mask"]
    antigen_mask = output["antigen_mask"]
    return (
        functional.binary_cross_entropy_with_logits(
            output["contact_logits"][pair_mask],
            batch["pair_labels"][pair_mask],
        )
        + 0.25
        * functional.binary_cross_entropy_with_logits(
            output["paratope_logits"][antibody_mask],
            batch["antibody_labels"][antibody_mask],
        )
        + 0.25
        * functional.binary_cross_entropy_with_logits(
            output["epitope_logits"][antigen_mask],
            batch["antigen_labels"][antigen_mask],
        )
    )


def _transform_backbone(
    backbone: torch.Tensor,
    rotation: torch.Tensor,
    translation: torch.Tensor,
) -> torch.Tensor:
    return (
        torch.einsum("xy,blay->blax", rotation, backbone)
        + translation[None, None, None, :]
    )


def _router_parameters(
    model: TopologyRoutedFrameContactModel,
) -> tuple[torch.nn.Parameter, ...]:
    assert model.self_encoder is not None
    return tuple(
        block.geometry_bias.gate_unconstrained
        for stream in (
            model.self_encoder.antibody_encoder,
            model.self_encoder.antigen_encoder,
        )
        for block in stream.blocks
    )


def main() -> None:
    arguments = _arguments()
    if arguments.output.exists() or arguments.output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite smoke receipt: {arguments.output}")
    if arguments.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if arguments.precision == "bf16" and arguments.device != "cuda":
        raise ValueError("BF16 smoke requires CUDA")
    if arguments.device == "cuda" and os.environ.get(
        "CUBLAS_WORKSPACE_CONFIG"
    ) not in {":4096:8", ":16:8"}:
        raise ValueError("CUDA smoke requires deterministic CUBLAS_WORKSPACE_CONFIG")
    protocol_sha256 = authenticate_file(
        arguments.protocol, arguments.protocol_sha256, "topology routing protocol"
    )
    receipt_sha256 = authenticate_file(
        arguments.protocol_receipt,
        arguments.protocol_receipt_sha256,
        "topology routing receipt",
    )
    protocol = json.loads(arguments.protocol.read_text(encoding="utf-8"))
    receipt = json.loads(arguments.protocol_receipt.read_text(encoding="utf-8"))
    if (
        receipt.get("protocol_file_sha256") != protocol_sha256
        or receipt.get("training_started") is not False
        or receipt.get("frozen_test_authorized") is not False
        or protocol.get("condition") != "topology_routed_multiscale"
    ):
        raise ValueError("Topology routing authority is not eligible")

    device = torch.device(arguments.device)
    torch.manual_seed(17)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(17)
        torch.cuda.reset_peak_memory_stats(device)
    torch.use_deterministic_algorithms(True, warn_only=True)
    dataset = AsepDevelopmentDataset(
        arguments.cache_root,
        manifest_sha256=arguments.cache_manifest_sha256,
        split_role="train",
        feature_mode="official_esm2_35m",
    )
    example = dataset[0]
    raw_batch = Sabdab2ContactBatchCollator()([example])
    batch = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in raw_batch.items()
    }
    model = TopologyRoutedFrameContactModel(
        antibody_input_dim=int(example["antibody_features"].shape[-1]),
        antigen_input_dim=int(example["antigen_features"].shape[-1]),
        d_model=128,
        num_heads=8,
        num_self_layers=2,
        geometry_feature_mode="full_pose",
        geometry_bias_mode="topology_routed_multiscale",
        num_rbf=16,
        sequence_window=32,
        dropout=0.1,
        fusion_mode="late",
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    model.train()
    context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if arguments.precision == "bf16"
        else nullcontext()
    )
    started = time.monotonic()
    with context:
        output = model(batch)
        loss = _loss(output, batch)
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("Topology-routed smoke produced a non-finite loss")
    loss.backward()
    router_parameters = _router_parameters(model)
    if not all(
        parameter.grad is not None
        and bool(torch.isfinite(parameter.grad).all())
        and bool((parameter.grad != 0).any())
        for parameter in router_parameters
    ):
        raise RuntimeError("A topology router lacks a finite nonzero first-step gradient")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    if not all(bool((parameter.detach() != 0).any()) for parameter in router_parameters):
        raise RuntimeError("A topology router remained identically zero after one step")

    model.eval()
    with torch.no_grad():
        baseline = model(batch, return_attention=True)
        transformed_batch = dict(batch)
        transformed_batch["antibody_backbone"] = _transform_backbone(
            batch["antibody_backbone"],
            torch.tensor(
                [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                device=device,
                dtype=batch["antibody_backbone"].dtype,
            ),
            torch.tensor(
                [31.0, -8.0, 5.0],
                device=device,
                dtype=batch["antibody_backbone"].dtype,
            ),
        )
        transformed_batch["antigen_backbone"] = _transform_backbone(
            batch["antigen_backbone"],
            torch.tensor(
                [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
                device=device,
                dtype=batch["antigen_backbone"].dtype,
            ),
            torch.tensor(
                [-17.0, 23.0, 9.0],
                device=device,
                dtype=batch["antigen_backbone"].dtype,
            ),
        )
        transformed = model(transformed_batch)
    invariance_max_abs = {
        name: float(
            (
                baseline[name].float() - transformed[name].float()
            ).abs().max().cpu()
        )
        for name in ("contact_logits", "paratope_logits", "epitope_logits")
    }
    if max(invariance_max_abs.values()) > 2e-5:
        raise RuntimeError(
            f"Independent SE(3) invariance failed: {invariance_max_abs}"
        )
    geometry_bias_nonzero = all(
        bool(
            (
                baseline["self_attention_details"][partner]["geometry"]["layers"][
                    layer_index
                ]["geometry_bias"]
                != 0
            ).any()
        )
        for partner in ("antibody", "antigen")
        for layer_index in range(2)
    )
    if not geometry_bias_nonzero:
        raise RuntimeError("Learned topology routing produced a zero geometry bias")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    payload = {
        "format_version": "topology_routed_asep_gpu_smoke_v1",
        "authority": {
            "cache_manifest_sha256": arguments.cache_manifest_sha256,
            "protocol_file_sha256": protocol_sha256,
            "protocol_receipt_sha256": receipt_sha256,
            "model_source_sha256": sha256_file(
                REPOSITORY_ROOT
                / "src/interaction_first/topology_routed_attention.py"
            ),
            "trainer_source_sha256": sha256_file(
                REPOSITORY_ROOT
                / "scripts/train_framecontact_topology_routing.py"
            ),
        },
        "runtime": {
            "device": str(device),
            "precision": arguments.precision,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else None
            ),
            "elapsed_seconds": time.monotonic() - started,
            "peak_cuda_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else None
            ),
        },
        "instance_id": example["instance_id"],
        "antibody_length": int(example["pair_labels"].shape[0]),
        "antigen_length": int(example["pair_labels"].shape[1]),
        "loss": float(loss.detach().cpu()),
        "router_parameter_count": sum(
            parameter.numel() for parameter in router_parameters
        ),
        "router_gate_nonzero_after_one_step": True,
        "geometry_bias_nonzero_after_one_step": True,
        "independent_se3_invariance_max_abs": invariance_max_abs,
        "checks": {
            "real_asep_train_member": True,
            "forward_backward_finite": True,
            "optimizer_step_performed": True,
            "official_test_loaded": False,
            "performance_metric_computed": False,
        },
        "claim_boundary": (
            "One authenticated AsEP development-train batch runtime and numerical "
            "smoke only. Loss is not a benchmark metric; no validation selection "
            "or official test access occurred."
        ),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_bytes(canonical_json_bytes(payload))
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
