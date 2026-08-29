#!/usr/bin/env python3
"""Record a non-authorizing SM120 MXFP8 linear capability observation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import torch


FORMAT_VERSION = "transformer_engine_sm120_mxfp8_capability_v1"


class Mxfp8CapabilityError(RuntimeError):
    """Report an invalid runtime or a failed finite-compute capability gate."""


def _canonical(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _publish(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise Mxfp8CapabilityError("Capability receipt output must be create-only")
    unsigned = _canonical(payload)
    payload["receipt_sha256"] = hashlib.sha256(unsigned).hexdigest()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(_canonical(payload))
        handle.flush()
        os.fsync(handle.fileno())


def _allocated_gpu_count() -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    return len([item for item in visible.split(",") if item.strip()])


def _relative_l2(observed: torch.Tensor, reference: torch.Tensor) -> float:
    observed_f32 = observed.detach().float()
    reference_f32 = reference.detach().float()
    denominator = float(torch.linalg.vector_norm(reference_f32).item())
    numerator = float(
        torch.linalg.vector_norm(observed_f32 - reference_f32).item()
    )
    value = numerator / max(denominator, torch.finfo(torch.float32).tiny)
    if not math.isfinite(value):
        raise Mxfp8CapabilityError("Relative L2 observation is non-finite")
    return value


def _run_linear(
    *,
    te: Any,
    recipe: Any | None,
    weight: torch.Tensor,
    input_value: torch.Tensor,
) -> dict[str, Any]:
    layer = te.Linear(
        1024,
        1024,
        bias=False,
        params_dtype=torch.bfloat16,
    ).cuda()
    with torch.no_grad():
        layer.weight.copy_(weight)
    input_tensor = input_value.detach().clone().requires_grad_(True)
    context = (
        te.autocast(enabled=True, recipe=recipe)
        if recipe is not None
        else torch.autocast(device_type="cuda", enabled=False)
    )
    with context:
        output = layer(input_tensor)
        loss = output.float().square().mean()
    loss.backward()
    torch.cuda.synchronize()
    input_gradient = input_tensor.grad
    weight_gradient = layer.weight.grad
    if input_gradient is None or weight_gradient is None:
        raise Mxfp8CapabilityError("Linear gradients are unavailable")
    finite = {
        "input_gradient": bool(torch.isfinite(input_gradient).all().item()),
        "loss": bool(torch.isfinite(loss).item()),
        "output": bool(torch.isfinite(output).all().item()),
        "weight_gradient": bool(torch.isfinite(weight_gradient).all().item()),
    }
    return {
        "finite": finite,
        "input_gradient": input_gradient.detach().clone(),
        "loss": float(loss.detach().cpu()),
        "output": output.detach().clone(),
        "passed": all(finite.values()),
        "weight_gradient": weight_gradient.detach().clone(),
    }


def probe(
    *,
    expected_device_name: str,
    expected_compute_capability: tuple[int, int],
    expected_transformer_engine_version: str,
    prerequisite_receipt: Path,
) -> dict[str, Any]:
    """Compare finite BF16, current-scaling FP8, and MXFP8 linear passes."""
    if _allocated_gpu_count() != 1:
        raise Mxfp8CapabilityError("Probe requires exactly one scheduler GPU")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise Mxfp8CapabilityError("Exactly one CUDA device must be available")
    if prerequisite_receipt.is_symlink() or not prerequisite_receipt.is_file():
        raise Mxfp8CapabilityError("Prerequisite capability receipt is unavailable")

    import transformer_engine
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import (
        Float8CurrentScaling,
        Format,
        MXFP8BlockScaling,
    )

    properties = torch.cuda.get_device_properties(0)
    compute_capability = (properties.major, properties.minor)
    if properties.name != expected_device_name:
        raise Mxfp8CapabilityError("GPU device name differs")
    if compute_capability != expected_compute_capability:
        raise Mxfp8CapabilityError("GPU compute capability differs")
    if transformer_engine.__version__ != expected_transformer_engine_version:
        raise Mxfp8CapabilityError("Transformer Engine version differs")

    availability = te.is_mxfp8_available(return_reason=True)
    if not isinstance(availability, tuple) or len(availability) != 2:
        raise Mxfp8CapabilityError("MXFP8 availability API returned invalid data")
    mxfp8_available, availability_reason = availability
    if not isinstance(mxfp8_available, bool) or not isinstance(
        availability_reason, str
    ):
        raise Mxfp8CapabilityError("MXFP8 availability result has invalid types")
    if not mxfp8_available:
        raise Mxfp8CapabilityError(
            "MXFP8 is unavailable: " + availability_reason
        )

    torch.manual_seed(2026081601)
    torch.cuda.manual_seed_all(2026081601)
    source_layer = te.Linear(
        1024,
        1024,
        bias=False,
        params_dtype=torch.bfloat16,
    ).cuda()
    weight = source_layer.weight.detach().clone()
    del source_layer
    input_value = torch.randn(64, 1024, device="cuda", dtype=torch.bfloat16)

    bf16 = _run_linear(
        te=te,
        recipe=None,
        weight=weight,
        input_value=input_value,
    )
    current = _run_linear(
        te=te,
        recipe=Float8CurrentScaling(),
        weight=weight,
        input_value=input_value,
    )
    mxfp8 = _run_linear(
        te=te,
        recipe=MXFP8BlockScaling(fp8_format=Format.E4M3),
        weight=weight,
        input_value=input_value,
    )
    passed = bool(
        mxfp8_available
        and bf16["passed"]
        and current["passed"]
        and mxfp8["passed"]
    )

    def summarize(run: dict[str, Any]) -> dict[str, Any]:
        return {
            "finite": run["finite"],
            "loss": run["loss"],
            "passed": run["passed"],
        }

    return {
        "authorization": {
            "four_gpu_training": False,
            "mxfp8_parity": False,
            "production_pretraining": False,
            "scientific_claim": False,
        },
        "claim_boundary": (
            "Single-GPU Transformer Engine linear-GEMM capability and raw "
            "numerical observations only; no tolerance was selected and this "
            "is not an attention, numerical-parity, throughput, MFU, four-GPU, "
            "production, biological, or state-of-the-art result"
        ),
        "format_version": FORMAT_VERSION,
        "gates": {
            "bf16_finite": bf16["passed"],
            "current_scaling_fp8_finite": current["passed"],
            "mxfp8_available": mxfp8_available,
            "mxfp8_finite": mxfp8["passed"],
            "passed": passed,
        },
        "hardware": {
            "compute_capability": list(compute_capability),
            "device_name": properties.name,
            "visible_device_count": torch.cuda.device_count(),
        },
        "inputs": {
            "batch": 64,
            "hidden": 1024,
            "prerequisite_receipt_file_sha256": _sha256(prerequisite_receipt),
            "shape_divisible_by_16": True,
            "shape_divisible_by_32": True,
        },
        "observations": {
            "bf16": summarize(bf16),
            "current_scaling_fp8": {
                **summarize(current),
                "relative_l2_to_bf16": {
                    "input_gradient": _relative_l2(
                        current["input_gradient"], bf16["input_gradient"]
                    ),
                    "output": _relative_l2(current["output"], bf16["output"]),
                    "weight_gradient": _relative_l2(
                        current["weight_gradient"], bf16["weight_gradient"]
                    ),
                },
            },
            "mxfp8_e4m3": {
                **summarize(mxfp8),
                "relative_l2_to_bf16": {
                    "input_gradient": _relative_l2(
                        mxfp8["input_gradient"], bf16["input_gradient"]
                    ),
                    "output": _relative_l2(mxfp8["output"], bf16["output"]),
                    "weight_gradient": _relative_l2(
                        mxfp8["weight_gradient"], bf16["weight_gradient"]
                    ),
                },
            },
        },
        "runtime": {
            "cuda_build": torch.version.cuda,
            "mxfp8_availability_reason": availability_reason,
            "pytorch": torch.__version__,
            "transformer_engine": transformer_engine.__version__,
        },
        "test_access": {
            "downstream_labels_used": False,
            "sabdab2_heldout_accessed": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-device-name", required=True)
    parser.add_argument("--expected-compute-capability", required=True)
    parser.add_argument("--expected-transformer-engine-version", required=True)
    parser.add_argument("--prerequisite-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    parts = arguments.expected_compute_capability.split(".")
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        raise Mxfp8CapabilityError("Expected compute capability is invalid")
    payload = probe(
        expected_device_name=arguments.expected_device_name,
        expected_compute_capability=(int(parts[0]), int(parts[1])),
        expected_transformer_engine_version=(
            arguments.expected_transformer_engine_version
        ),
        prerequisite_receipt=arguments.prerequisite_receipt,
    )
    if not payload["gates"]["passed"]:
        raise Mxfp8CapabilityError("MXFP8 finite-compute gate failed")
    _publish(arguments.output, payload)
    print(json.dumps(payload, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
