#!/usr/bin/env python3
"""Profile FrameContact attention kernels without touching research splits.

This is a synthetic systems diagnostic, not a model benchmark.  It compares the
current explicit attention implementation with PyTorch SDPA under the exact
operation needed by FrameContact: a dense, per-head additive geometry/contact
bias plus a boolean pair mask.  It also profiles a complete synthetic
FrameContact forward/backward pass at representative antibody/antigen lengths.

The script fails closed unless reference and SDPA outputs *and gradients* agree
within dtype-specific tolerances.  It writes one atomic JSON receipt containing
hardware/runtime provenance, peak allocated memory, latency distributions, and
the CUDA operator selected by PyTorch's dispatcher.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import statistics
import sys
import tempfile
import time
from typing import Any, Callable

import torch
import torch.nn.functional as functional


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from interaction_first.framecontact import FrameContactModel  # noqa: E402
from interaction_first.intra_chain_geometry import build_backbone_frames  # noqa: E402
from interaction_first.sabdab2_structure import canonical_json_bytes, sha256_file  # noqa: E402


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(canonical_json_bytes(payload))
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _reference_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    additive_bias: torch.Tensor,
    permitted: torch.Tensor,
) -> torch.Tensor:
    logits = torch.einsum("bhid,bhjd->bhij", query, key) / math.sqrt(
        query.shape[-1]
    )
    logits = (logits + additive_bias).masked_fill(
        ~permitted, torch.finfo(logits.dtype).min
    )
    weights = torch.softmax(logits, dim=-1)
    return torch.einsum("bhij,bhjd->bhid", weights, value)


def _sdpa_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    additive_bias: torch.Tensor,
    permitted: torch.Tensor,
) -> torch.Tensor:
    mask_value = torch.full_like(additive_bias, float("-inf"))
    attention_bias = torch.where(permitted, additive_bias, mask_value)
    return functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_bias,
        dropout_p=0.0,
        is_causal=False,
    )


def _make_attention_inputs(
    *,
    query_count: int,
    key_count: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device=device).manual_seed(seed)
    shape_q = (1, 8, query_count, 16)
    shape_k = (1, 8, key_count, 16)
    query = torch.randn(shape_q, generator=generator, device=device, dtype=dtype)
    key = torch.randn(shape_k, generator=generator, device=device, dtype=dtype)
    value = torch.randn(shape_k, generator=generator, device=device, dtype=dtype)
    additive_bias = (
        torch.randn(
            (1, 8, query_count, key_count),
            generator=generator,
            device=device,
            dtype=dtype,
        )
        * 0.2
    )
    # Exercise nontrivial masks while guaranteeing every query has a candidate.
    permitted = (
        torch.rand(
            (1, 1, query_count, key_count), generator=generator, device=device
        )
        < 0.35
    )
    permitted[..., 0] = True
    return query, key, value, additive_bias, permitted


def _correctness_receipt(
    *, dtype: torch.dtype, device: torch.device, query_count: int, key_count: int
) -> dict[str, Any]:
    base = _make_attention_inputs(
        query_count=query_count,
        key_count=key_count,
        dtype=dtype,
        device=device,
        seed=20260801,
    )
    reference_inputs = tuple(
        value.detach().clone().requires_grad_(index < 4)
        if torch.is_floating_point(value)
        else value
        for index, value in enumerate(base)
    )
    sdpa_inputs = tuple(
        value.detach().clone().requires_grad_(index < 4)
        if torch.is_floating_point(value)
        else value
        for index, value in enumerate(base)
    )
    reference = _reference_attention(*reference_inputs)
    sdpa = _sdpa_attention(*sdpa_inputs)
    projection = torch.linspace(
        -0.5, 0.5, reference.numel(), device=device, dtype=dtype
    ).reshape_as(reference)
    (reference * projection).sum().backward()
    (sdpa * projection).sum().backward()
    output_max_abs = float((reference - sdpa).abs().max().float())
    output_mean_abs = float((reference - sdpa).abs().mean().float())
    gradient_max_abs = {}
    gradient_mean_abs = {}
    names = ("query", "key", "value", "additive_bias")
    for name, reference_input, sdpa_input in zip(
        names, reference_inputs[:4], sdpa_inputs[:4], strict=True
    ):
        difference = (reference_input.grad - sdpa_input.grad).abs().float()
        gradient_max_abs[name] = float(difference.max())
        gradient_mean_abs[name] = float(difference.mean())
    tolerance = 5e-5 if dtype == torch.float32 else 3e-2
    passed = output_max_abs <= tolerance and max(gradient_max_abs.values()) <= tolerance
    if not passed:
        raise RuntimeError(
            f"SDPA equivalence failed for {dtype}: output={output_max_abs}, "
            f"gradient={max(gradient_max_abs.values())}, tolerance={tolerance}"
        )
    return {
        "dtype": str(dtype).removeprefix("torch."),
        "query_count": query_count,
        "key_count": key_count,
        "tolerance": tolerance,
        "output_max_abs": output_max_abs,
        "output_mean_abs": output_mean_abs,
        "gradient_max_abs": gradient_max_abs,
        "gradient_mean_abs": gradient_mean_abs,
        "passed": True,
    }


def _latency_distribution(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    return {
        "minimum_ms": ordered[0],
        "median_ms": statistics.median(ordered),
        "mean_ms": statistics.fmean(ordered),
        "p90_ms": ordered[min(len(ordered) - 1, math.ceil(0.9 * len(ordered)) - 1)],
        "maximum_ms": ordered[-1],
    }


def _benchmark_cuda_callable(
    callback: Callable[[], torch.Tensor],
    *,
    warmup: int,
    repeats: int,
    device: torch.device,
) -> dict[str, Any]:
    for _ in range(warmup):
        output = callback()
        if not bool(torch.isfinite(output).all()):
            raise RuntimeError("benchmark output is non-finite")
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = callback()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return {
        "latency": _latency_distribution(samples),
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "repeats": repeats,
        "warmup": warmup,
    }


def _profile_operator_names(callback: Callable[[], torch.Tensor]) -> list[str]:
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as profile:
        callback()
        torch.cuda.synchronize()
    interesting = {
        item.key
        for item in profile.key_averages()
        if any(
            token in item.key.lower()
            for token in ("attention", "softmax", "einsum", "bmm", "gemm")
        )
    }
    return sorted(interesting)


def _backbone(length: int, device: torch.device) -> torch.Tensor:
    residue = torch.arange(length, device=device, dtype=torch.float32)
    ca = torch.stack(
        (3.8 * residue, 1.5 * torch.sin(residue / 3.0), torch.cos(residue / 4.0)),
        dim=-1,
    )[None]
    n = ca + torch.tensor([-1.15, 0.75, 0.15], device=device)
    c = ca + torch.tensor([1.25, 0.2, -0.2], device=device)
    return torch.stack((n, ca, c), dim=-2)


def _synthetic_framecontact_batch(
    *, antibody_length: int, antigen_length: int, device: torch.device
) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(
        20260801 + antibody_length + antigen_length
    )
    antibody_mask = torch.ones((1, antibody_length), dtype=torch.bool, device=device)
    antigen_mask = torch.ones((1, antigen_length), dtype=torch.bool, device=device)
    antibody_backbone = _backbone(antibody_length, device)
    antigen_backbone = _backbone(antigen_length, device)
    antibody_valid = build_backbone_frames(antibody_backbone, antibody_mask).valid_mask
    antigen_valid = build_backbone_frames(antigen_backbone, antigen_mask).valid_mask
    return {
        "antibody_features": torch.randn(
            (1, antibody_length, 1280), generator=generator, device=device
        ),
        "antigen_features": torch.randn(
            (1, antigen_length, 1280), generator=generator, device=device
        ),
        "antibody_backbone": antibody_backbone,
        "antigen_backbone": antigen_backbone,
        "antibody_mask": antibody_mask,
        "antigen_mask": antigen_mask,
        "antibody_chain_ids": torch.zeros(
            (1, antibody_length), dtype=torch.long, device=device
        ),
        "antigen_chain_ids": torch.zeros(
            (1, antigen_length), dtype=torch.long, device=device
        ),
        "antibody_frame_valid_mask": antibody_valid,
        "antigen_frame_valid_mask": antigen_valid,
        "antibody_missing_frame_mask": antibody_mask & ~antibody_valid,
        "antigen_missing_frame_mask": antigen_mask & ~antigen_valid,
    }


def _full_model_benchmark(
    *,
    antibody_length: int,
    antigen_length: int,
    device: torch.device,
    warmup: int,
    repeats: int,
    attention_kernel: str,
) -> dict[str, Any]:
    torch.manual_seed(20260801)
    model = FrameContactModel(
        antibody_input_dim=1280,
        antigen_input_dim=1280,
        d_model=128,
        num_heads=8,
        num_self_layers=2,
        geometry_feature_mode="full_pose",
        geometry_bias_mode="pair_mlp",
        geometry_attention_graph_mode="spatial",
        num_rbf=16,
        neighbor_k=32,
        neighbor_radius=16.0,
        pair_dim=128,
        dropout=0.0,
        fusion_mode="late",
        pair_output_mode="distogram",
        residue_output_mode="gated_hybrid",
        attention_kernel=attention_kernel,
    ).to(device)
    batch = _synthetic_framecontact_batch(
        antibody_length=antibody_length,
        antigen_length=antigen_length,
        device=device,
    )

    def inference() -> torch.Tensor:
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            output = model(batch, return_attention=False)
            return output["contact_logits"]

    model.eval()
    inference_receipt = _benchmark_cuda_callable(
        inference, warmup=warmup, repeats=repeats, device=device
    )

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    def train_step() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(batch, return_attention=False)
            pair_mask = output["pair_candidate_mask"]
            loss = output["contact_logits"][pair_mask].float().square().mean()
            loss = loss + 0.01 * (
                output["paratope_logits"][batch["antibody_mask"]].float().square().mean()
                + output["epitope_logits"][batch["antigen_mask"]].float().square().mean()
            )
        loss.backward()
        return loss.detach()

    training_receipt = _benchmark_cuda_callable(
        train_step, warmup=max(1, warmup // 2), repeats=max(3, repeats // 2), device=device
    )
    del optimizer, model, batch
    torch.cuda.empty_cache()
    return {
        "attention_kernel": attention_kernel,
        "antibody_length": antibody_length,
        "antigen_length": antigen_length,
        "pair_count": antibody_length * antigen_length,
        "inference_bf16": inference_receipt,
        "forward_backward_bf16": training_receipt,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--lengths",
        default="220x128,220x512,220x1000",
        help="comma-separated antibody x antigen lengths",
    )
    arguments = parser.parse_args()
    if arguments.output.exists() or arguments.output.is_symlink():
        raise FileExistsError("Refusing to overwrite kernel benchmark receipt")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if arguments.warmup < 1 or arguments.repeats < 3:
        raise ValueError("warmup must be >=1 and repeats must be >=3")
    lengths = []
    for item in arguments.lengths.split(","):
        antibody_text, antigen_text = item.lower().split("x", maxsplit=1)
        pair = (int(antibody_text), int(antigen_text))
        if min(pair) <= 0:
            raise ValueError("all sequence lengths must be positive")
        lengths.append(pair)
    device = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.set_device(device.index)
    started = time.time()

    correctness = [
        _correctness_receipt(
            dtype=dtype,
            device=device,
            query_count=220,
            key_count=512,
        )
        for dtype in (torch.float32, torch.bfloat16)
    ]
    primitive = []
    for dtype in (torch.float32, torch.bfloat16):
        for query_count, key_count in ((220, 220), (220, 512), (220, 1000)):
            inputs = _make_attention_inputs(
                query_count=query_count,
                key_count=key_count,
                dtype=dtype,
                device=device,
                seed=20260801 + key_count,
            )
            callbacks = {
                "reference": lambda values=inputs: _reference_attention(*values),
                "sdpa": lambda values=inputs: _sdpa_attention(*values),
            }
            methods = {}
            for name, callback in callbacks.items():
                methods[name] = _benchmark_cuda_callable(
                    callback,
                    warmup=arguments.warmup,
                    repeats=arguments.repeats,
                    device=device,
                )
                methods[name]["profile_operator_names"] = _profile_operator_names(
                    callback
                )
            methods["median_speedup_sdpa_over_reference"] = (
                methods["reference"]["latency"]["median_ms"]
                / methods["sdpa"]["latency"]["median_ms"]
            )
            methods["peak_memory_ratio_sdpa_over_reference"] = (
                methods["sdpa"]["peak_cuda_allocated_bytes"]
                / methods["reference"]["peak_cuda_allocated_bytes"]
            )
            primitive.append(
                {
                    "dtype": str(dtype).removeprefix("torch."),
                    "query_count": query_count,
                    "key_count": key_count,
                    "methods": methods,
                }
            )
            del inputs
            torch.cuda.empty_cache()

    full_model = []
    for antibody_length, antigen_length in lengths:
        methods = {
            attention_kernel: _full_model_benchmark(
                antibody_length=antibody_length,
                antigen_length=antigen_length,
                device=device,
                warmup=arguments.warmup,
                repeats=arguments.repeats,
                attention_kernel=attention_kernel,
            )
            for attention_kernel in ("reference", "sdpa")
        }
        methods["inference_median_speedup_sdpa_over_reference"] = (
            methods["reference"]["inference_bf16"]["latency"]["median_ms"]
            / methods["sdpa"]["inference_bf16"]["latency"]["median_ms"]
        )
        methods["training_median_speedup_sdpa_over_reference"] = (
            methods["reference"]["forward_backward_bf16"]["latency"]["median_ms"]
            / methods["sdpa"]["forward_backward_bf16"]["latency"]["median_ms"]
        )
        methods["training_peak_memory_ratio_sdpa_over_reference"] = (
            methods["sdpa"]["forward_backward_bf16"]["peak_cuda_allocated_bytes"]
            / methods["reference"]["forward_backward_bf16"]["peak_cuda_allocated_bytes"]
        )
        full_model.append(
            {
                "antibody_length": antibody_length,
                "antigen_length": antigen_length,
                "pair_count": antibody_length * antigen_length,
                "methods": methods,
            }
        )
    payload = {
        "format_version": "framecontact_attention_kernel_benchmark_v2",
        "created_unix_seconds": started,
        "elapsed_wall_seconds": time.time() - started,
        "runtime": {
            "gpu": torch.cuda.get_device_name(device),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "python": sys.version.split()[0],
        },
        "configuration": {
            "warmup": arguments.warmup,
            "repeats": arguments.repeats,
            "heads": 8,
            "head_dim": 16,
            "additive_bias_materialized": True,
            "mask_density": 0.35,
            "dropout": 0.0,
        },
        "correctness": correctness,
        "primitive_attention": primitive,
        "full_model_synthetic": full_model,
        "authority": {
            "benchmark_source_sha256": sha256_file(Path(__file__).resolve()),
            "model_source_sha256": sha256_file(
                SOURCE_ROOT / "interaction_first/framecontact.py"
            ),
            "geometry_source_sha256": sha256_file(
                SOURCE_ROOT / "interaction_first/intra_chain_geometry.py"
            ),
        },
        "checks": {
            "reference_sdpa_forward_and_gradient_equivalence": True,
            "real_dataset_loaded": False,
            "optimizer_step_performed": False,
            "validation_loaded": False,
            "test_loaded": False,
            "model_quality_metric_computed": False,
        },
        "claim_boundary": (
            "Synthetic single-GPU systems diagnostic only. Latency and memory apply "
            "only to the recorded hardware, runtime, shapes, and materialized additive "
            "bias. This is not a training-throughput, convergence, generalization, "
            "benchmark-improvement, fused-kernel, or state-of-the-art claim."
        ),
    }
    _atomic_write(arguments.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
