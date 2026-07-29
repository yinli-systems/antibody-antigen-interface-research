#!/usr/bin/env python3
"""Sweep NCCL collective payload sizes without accessing biological data."""

from __future__ import annotations

import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import socket
import time
from typing import Any

import torch
import torch.distributed as dist

from distributed_launch_contract import (
    LaunchContractError,
    publish_create_only_json,
    require_sha256,
    safe_environment_snapshot,
    source_tree_digest,
    validate_distributed_identity,
    verify_launch_manifest,
)

DTYPES = {
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}
OPERATIONS = ("all_reduce", "broadcast")
MAX_PAYLOAD_EXPONENT = 30
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _event(rank: int, stage: str, **fields: Any) -> None:
    """Emit one flush-safe rank-scoped diagnostic event."""
    print(
        json.dumps(
            {"rank": rank, "stage": stage, **fields},
            sort_keys=True,
            allow_nan=False,
        ),
        flush=True,
    )


def _write_create_only_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically publish one JSON object without replacing an existing artifact."""
    publish_create_only_json(path, payload)


def _source_digest() -> str:
    """Hash the exact message-sweep source and launcher boundary."""
    return source_tree_digest(
        REPOSITORY_ROOT,
        (
            "scripts/distributed_launch_contract.py",
            "scripts/generate_runtime_manifest.py",
            "scripts/nccl_message_size_sweep.py",
            "slurm/nccl_message_size_sweep.sbatch",
        ),
    )


def _initialize_process_group(
    expected_world_size: int,
) -> tuple[int, int, int, torch.device]:
    """Bind the local CUDA device before eagerly initializing NCCL."""
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2:
        raise RuntimeError(
            f"This threshold diagnostic requires exactly two ranks, got {world_size}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    validate_distributed_identity(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        visible_device_count=torch.cuda.device_count(),
        minimum_world_size=2,
        expected_world_size=expected_world_size,
    )
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(minutes=5),
        device_id=device,
    )
    return rank, local_rank, world_size, device


def _payload_sizes(min_exponent: int, max_exponent: int) -> list[int]:
    """Return an inclusive ascending power-of-two payload plan."""
    if min_exponent < 0:
        raise ValueError("--min-exponent must be non-negative")
    if max_exponent < min_exponent:
        raise ValueError("--max-exponent must be at least --min-exponent")
    if max_exponent > MAX_PAYLOAD_EXPONENT:
        raise ValueError(
            f"--max-exponent cannot exceed {MAX_PAYLOAD_EXPONENT}"
        )
    return [1 << exponent for exponent in range(min_exponent, max_exponent + 1)]


def _validate_tensor(
    tensor: torch.Tensor,
    *,
    expected_value: float,
    requested_bytes: int,
) -> None:
    """Require every payload element to equal the exactly representable sentinel."""
    if tensor.numel() * tensor.element_size() != requested_bytes:
        raise RuntimeError("Allocated tensor byte count differs from the sweep plan")
    if not bool(torch.isfinite(tensor).all().item()):
        raise RuntimeError("Collective output contains a non-finite value")
    if not bool(torch.eq(tensor, expected_value).all().item()):
        first = float(tensor.reshape(-1)[0].detach().cpu())
        last = float(tensor.reshape(-1)[-1].detach().cpu())
        raise RuntimeError(
            "Collective output failed full-payload validation: "
            f"expected={expected_value}, first={first}, last={last}"
        )


def _run_one_size(
    *,
    rank: int,
    local_rank: int,
    world_size: int,
    device: torch.device,
    operation: str,
    dtype: torch.dtype,
    requested_bytes: int,
    size_index: int,
) -> dict[str, Any]:
    """Run, synchronize, and fully validate one collective payload."""
    element_size = torch.empty((), dtype=dtype).element_size()
    if requested_bytes % element_size:
        raise ValueError("Requested byte count is not divisible by the dtype size")
    element_count = requested_bytes // element_size

    if operation == "broadcast":
        expected_value = float((size_index % 13) + 1)
        initial_value = expected_value if rank == 0 else -expected_value
    elif operation == "all_reduce":
        initial_value = float(rank + 1)
        expected_value = float(world_size * (world_size + 1) // 2)
    else:
        raise ValueError(f"Unsupported operation: {operation}")

    payload = torch.full(
        (element_count,),
        initial_value,
        dtype=dtype,
        device=device,
    )
    torch.cuda.synchronize(device)
    dist.barrier(device_ids=[local_rank])
    torch.cuda.synchronize(device)
    _event(
        rank,
        "collective_start",
        operation=operation,
        requested_bytes=requested_bytes,
        element_count=element_count,
        dtype=str(dtype),
    )

    started = time.perf_counter()
    if operation == "broadcast":
        dist.broadcast(payload, src=0)
    else:
        dist.all_reduce(payload, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize(device)
    elapsed_seconds = time.perf_counter() - started

    _validate_tensor(
        payload,
        expected_value=expected_value,
        requested_bytes=requested_bytes,
    )
    torch.cuda.synchronize(device)
    dist.barrier(device_ids=[local_rank])
    torch.cuda.synchronize(device)
    _event(
        rank,
        "collective_passed",
        operation=operation,
        requested_bytes=requested_bytes,
        elapsed_seconds=elapsed_seconds,
        expected_value=expected_value,
    )
    del payload
    torch.cuda.empty_cache()
    return {
        "requested_bytes": requested_bytes,
        "element_count": element_count,
        "element_size_bytes": element_size,
        "expected_value": expected_value,
        "rank_zero_elapsed_seconds": elapsed_seconds,
    }


def main() -> None:
    """Run an ascending two-GPU message-size sweep with durable pass receipts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--operation", choices=OPERATIONS, required=True)
    parser.add_argument("--dtype", choices=tuple(DTYPES), required=True)
    parser.add_argument("--min-exponent", type=int, default=10)
    parser.add_argument("--max-exponent", type=int, default=28)
    parser.add_argument("--expected-world-size", type=int, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--launch-manifest", type=Path, required=True)
    parser.add_argument("--expected-launch-manifest-sha256", required=True)
    parser.add_argument("--synthetic-only", action="store_true")
    args = parser.parse_args()
    if not args.synthetic_only:
        raise ValueError("--synthetic-only is mandatory")

    source_sha256 = _source_digest()
    expected_source_sha256 = require_sha256(
        args.expected_source_sha256,
        field="expected_source_sha256",
    )
    if source_sha256 != expected_source_sha256:
        raise LaunchContractError(
            "Executed source boundary differs from the externally frozen digest"
        )
    verify_launch_manifest(
        args.launch_manifest,
        expected_file_sha256=args.expected_launch_manifest_sha256,
        expected_source_sha256=source_sha256,
        expected_world_size=args.expected_world_size,
        expected_backend="nccl",
        expected_parallelism="collective_diagnostic",
        expected_precision=args.dtype,
    )
    sizes = _payload_sizes(args.min_exponent, args.max_exponent)
    rank, local_rank, world_size, device = _initialize_process_group(
        args.expected_world_size
    )
    receipt_root = args.output_root / "passed"
    attempt_root = args.output_root / "attempted"
    try:
        properties = torch.cuda.get_device_properties(device)
        _event(
            rank,
            "process_group_initialized",
            hostname=socket.gethostname(),
            device=str(device),
            device_name=properties.name,
            compute_capability=f"{properties.major}.{properties.minor}",
            world_size=world_size,
            operation=args.operation,
            dtype=args.dtype,
            payload_sizes=sizes,
        )
        if rank == 0:
            if (args.output_root / "result.json").exists():
                raise FileExistsError("Refusing to overwrite result.json")
            receipt_root.mkdir(parents=True, exist_ok=False)
            attempt_root.mkdir(parents=True, exist_ok=False)
        dist.barrier(device_ids=[local_rank])

        receipts: list[dict[str, Any]] = []
        for size_index, requested_bytes in enumerate(sizes):
            attempt_name = f"{size_index:03d}_{requested_bytes:012d}.json"
            if rank == 0:
                _write_create_only_json(
                    attempt_root / attempt_name,
                    {
                        "format_version": "frontier_nccl_message_attempt_v1",
                        "operation": args.operation,
                        "dtype": args.dtype,
                        "size_index": size_index,
                        "requested_bytes": requested_bytes,
                    },
                )
            dist.barrier(device_ids=[local_rank])
            receipt = _run_one_size(
                rank=rank,
                local_rank=local_rank,
                world_size=world_size,
                device=device,
                operation=args.operation,
                dtype=DTYPES[args.dtype],
                requested_bytes=requested_bytes,
                size_index=size_index,
            )
            receipt.update(
                {
                    "format_version": "frontier_nccl_message_pass_v1",
                    "operation": args.operation,
                    "dtype": args.dtype,
                    "size_index": size_index,
                }
            )
            receipts.append(receipt)
            if rank == 0:
                _write_create_only_json(receipt_root / attempt_name, receipt)
            dist.barrier(device_ids=[local_rank])

        if rank == 0:
            _write_create_only_json(
                args.output_root / "result.json",
                {
                    "format_version": "frontier_nccl_message_size_sweep_v1",
                    "status": "passed",
                    "scope": "synthetic two-GPU collective correctness only",
                    "operation": args.operation,
                    "dtype": args.dtype,
                    "min_exponent": args.min_exponent,
                    "max_exponent": args.max_exponent,
                    "payload_sizes_bytes": sizes,
                    "last_passed_bytes": receipts[-1]["requested_bytes"],
                    "receipt_count": len(receipts),
                    "world_size": world_size,
                    "source_sha256": source_sha256,
                    "launch_manifest_sha256": (
                        args.expected_launch_manifest_sha256
                    ),
                    "torch_version": torch.__version__,
                    "cuda_version": torch.version.cuda,
                    "nccl_version": list(torch.cuda.nccl.version()),
                    "device_name": properties.name,
                    "compute_capability": (
                        f"{properties.major}.{properties.minor}"
                    ),
                    "environment": safe_environment_snapshot(),
                    "claim_boundary": (
                        "One synchronized iteration per size is a correctness and "
                        "fault-isolation observation, not a throughput or scaling "
                        "benchmark and not biological-performance evidence."
                    ),
                },
            )
        dist.barrier(device_ids=[local_rank])
        _event(
            rank,
            "sweep_passed",
            operation=args.operation,
            dtype=args.dtype,
            last_passed_bytes=receipts[-1]["requested_bytes"],
        )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
