#!/usr/bin/env python3
"""Record a non-authorizing Transformer Engine FP8 capability smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch

FORMAT_VERSION = "transformer_engine_sm120_capability_v1"


class CapabilityProbeError(RuntimeError):
    """Report a capability or provenance mismatch."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _publish(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise CapabilityProbeError("Capability receipt output must be create-only")
    unsigned = _canonical(payload)
    payload["receipt_sha256"] = hashlib.sha256(unsigned).hexdigest()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(_canonical(payload))
        handle.flush()
        os.fsync(handle.fileno())


def _allocated_gpu_count() -> int:
    value = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    return len([item for item in value.split(",") if item.strip()])


def probe(
    *,
    expected_device_name: str,
    expected_compute_capability: tuple[int, int],
    expected_transformer_engine_version: str,
    install_report_nvcc: Path,
    install_report_te: Path,
) -> dict[str, Any]:
    """Run one finite FP8 forward/backward pass on the allocated GPU."""
    if _allocated_gpu_count() != 1:
        raise CapabilityProbeError("Probe requires exactly one visible GPU")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise CapabilityProbeError("Exactly one CUDA device must be available")
    import transformer_engine
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import Float8CurrentScaling

    properties = torch.cuda.get_device_properties(0)
    compute_capability = (properties.major, properties.minor)
    if properties.name != expected_device_name:
        raise CapabilityProbeError("GPU device name differs")
    if compute_capability != expected_compute_capability:
        raise CapabilityProbeError("GPU compute capability differs")
    if transformer_engine.__version__ != expected_transformer_engine_version:
        raise CapabilityProbeError("Transformer Engine version differs")
    for path, label in (
        (install_report_nvcc, "NVCC install report"),
        (install_report_te, "Transformer Engine install report"),
    ):
        if path.is_symlink() or not path.is_file():
            raise CapabilityProbeError(label + " is absent or symbolic")

    torch.manual_seed(1729)
    torch.cuda.manual_seed_all(1729)
    layer = te.Linear(
        1024,
        1024,
        bias=False,
        params_dtype=torch.bfloat16,
    ).cuda()
    input_tensor = torch.randn(
        64,
        1024,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    torch.cuda.reset_peak_memory_stats()
    recipe = Float8CurrentScaling()
    with te.autocast(enabled=True, recipe=recipe):
        output = layer(input_tensor)
        loss = output.float().square().mean()
    loss.backward()
    torch.cuda.synchronize()
    output_finite = bool(torch.isfinite(output).all().item())
    input_gradient_finite = bool(
        input_tensor.grad is not None
        and torch.isfinite(input_tensor.grad).all().item()
    )
    parameter_gradients = [
        parameter.grad for parameter in layer.parameters() if parameter.requires_grad
    ]
    parameter_gradients_finite = bool(
        parameter_gradients
        and all(
            gradient is not None and bool(torch.isfinite(gradient).all().item())
            for gradient in parameter_gradients
        )
    )
    passed = bool(
        output_finite and input_gradient_finite and parameter_gradients_finite
    )
    return {
        "authorization": {
            "four_gpu_training": False,
            "fp8_candidate": passed,
            "production_pretraining": False,
            "scientific_claim": False,
        },
        "claim_boundary": (
            "Single-GPU FP8 capability and finite-gradient smoke only; this is "
            "not a numerical-parity, throughput, MFU, four-GPU, production, "
            "biological, or state-of-the-art result"
        ),
        "format_version": FORMAT_VERSION,
        "gates": {
            "input_gradient_finite": input_gradient_finite,
            "output_finite": output_finite,
            "parameter_gradients_finite": parameter_gradients_finite,
            "passed": passed,
        },
        "hardware": {
            "compute_capability": list(compute_capability),
            "device_name": properties.name,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "visible_device_count": torch.cuda.device_count(),
        },
        "inputs": {
            "install_report_nvcc_file_sha256": _sha256(install_report_nvcc),
            "install_report_te_file_sha256": _sha256(install_report_te),
        },
        "precision": {
            "activation_parameter_dtype": "bfloat16",
            "fp8_recipe": "Float8CurrentScaling.HYBRID",
            "linear_dimensions": [1024, 1024],
            "micro_batch": 64,
        },
        "runtime": {
            "cuda_build": torch.version.cuda,
            "loss": float(loss.detach().cpu()),
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
    parser.add_argument("--install-report-nvcc", type=Path, required=True)
    parser.add_argument("--install-report-te", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    parts = arguments.expected_compute_capability.split(".")
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        raise CapabilityProbeError("Expected compute capability is invalid")
    payload = probe(
        expected_device_name=arguments.expected_device_name,
        expected_compute_capability=(int(parts[0]), int(parts[1])),
        expected_transformer_engine_version=(
            arguments.expected_transformer_engine_version
        ),
        install_report_nvcc=arguments.install_report_nvcc,
        install_report_te=arguments.install_report_te,
    )
    if not payload["gates"]["passed"]:
        raise CapabilityProbeError("FP8 finite-gradient gate failed")
    _publish(arguments.output, payload)
    print(json.dumps(payload, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
