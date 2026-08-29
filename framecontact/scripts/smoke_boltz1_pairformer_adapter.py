#!/usr/bin/env python3
"""Run and receipt a strict minimal forward with real Boltz-1 Pairformer weights."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from interaction_first.boltz_pairformer_adapter import (  # noqa: E402
    OfficialBoltzPairformerAdapter,
    read_pairformer_receipt,
)


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--boltz-source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--residues", type=int, default=2)
    arguments = parser.parse_args()
    if arguments.output.exists() or arguments.output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite: {arguments.output}")
    if arguments.residues <= 0:
        raise ValueError("residues must be positive")
    device = torch.device(arguments.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    extraction = read_pairformer_receipt(arguments.receipt)
    load_start = time.perf_counter()
    model = OfficialBoltzPairformerAdapter(
        artifact_path=arguments.artifact,
        receipt_path=arguments.receipt,
        boltz_source_root=arguments.boltz_source_root,
        single_input_dim=32,
        pair_input_dim=16,
        output_dim=24,
    ).eval().to(device)
    load_seconds = time.perf_counter() - load_start
    residue_mask = torch.ones(
        1, arguments.residues, dtype=torch.bool, device=device
    )
    pair_mask = residue_mask[:, :, None] & residue_mask[:, None, :]
    generator = torch.Generator(device=device).manual_seed(20260731)
    single = torch.randn(1, arguments.residues, 32, generator=generator, device=device)
    pair = torch.randn(
        1,
        arguments.residues,
        arguments.residues,
        16,
        generator=generator,
        device=device,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    forward_start = time.perf_counter()
    with torch.inference_mode():
        output = model(single, pair, residue_mask, pair_mask)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    forward_seconds = time.perf_counter() - forward_start
    payload = {
        "format_version": "boltz1_pairformer_real_weight_smoke_v1",
        "artifact_path": str(arguments.artifact.resolve()),
        "artifact_sha256": _sha256(arguments.artifact),
        "extraction_receipt_path": str(arguments.receipt.resolve()),
        "extraction_receipt_sha256": _sha256(arguments.receipt),
        "boltz_source_root": str(arguments.boltz_source_root.resolve()),
        "boltz_source_revision": extraction["boltz_source_revision"],
        "source_checkpoint_sha256": extraction["source_checkpoint_sha256"],
        "state_tensor_count": extraction["state_tensor_count"],
        "architecture": extraction["architecture"],
        "strict_state_load": True,
        "real_weight_forward": True,
        "pairformer_parameter_count": sum(
            parameter.numel() for parameter in model.pairformer.parameters()
        ),
        "trainable_adapter_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "pairformer_frozen": not any(
            parameter.requires_grad for parameter in model.pairformer.parameters()
        ),
        "single_output_shape": list(output["single_features"].shape),
        "pair_output_shape": list(output["pair_features"].shape),
        "single_output_finite": bool(
            torch.isfinite(output["single_features"]).all().item()
        ),
        "pair_output_finite": bool(
            torch.isfinite(output["pair_features"]).all().item()
        ),
        "device": str(device),
        "accelerator_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "load_seconds_smoke_only": round(load_seconds, 6),
        "forward_seconds_smoke_only": round(forward_seconds, 6),
        "cuda_peak_allocated_bytes_smoke_only": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else None
        ),
        "biological_data_read": False,
        "benchmark_result": False,
        "native_boltz_pipeline_reproduced": False,
        "claim_boundary": (
            "Strict real-weight load and synthetic minimal forward only; not a "
            "throughput, biological, native-Boltz, or same-split baseline result."
        ),
    }
    if not (
        payload["pairformer_frozen"]
        and payload["single_output_finite"]
        and payload["pair_output_finite"]
    ):
        raise RuntimeError("Real-weight smoke contract failed")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_bytes(_canonical_bytes(payload))
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
