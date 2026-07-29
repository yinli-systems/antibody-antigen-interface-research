#!/usr/bin/env python3
"""Isolate DDP all-reduce from differentiable all-gather backward failures."""

from __future__ import annotations

import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from distributed_launch_contract import (
    LaunchContractError,
    publish_create_only_json,
    require_sha256,
    source_tree_digest,
    validate_distributed_identity,
    verify_launch_manifest,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _event(rank: int, stage: str, **fields: Any) -> None:
    """Emit one flush-safe diagnostic boundary."""
    print(
        json.dumps(
            {"rank": rank, "stage": stage, **fields},
            sort_keys=True,
            allow_nan=False,
        ),
        flush=True,
    )


def _write_result(path: Path, payload: dict[str, Any]) -> None:
    """Publish one create-only result after every rank passes."""
    publish_create_only_json(path, payload)


def _source_digest() -> str:
    """Hash the exact diagnostic source and launcher boundary."""
    return source_tree_digest(
        REPOSITORY_ROOT,
        (
            "scripts/distributed_launch_contract.py",
            "scripts/generate_runtime_manifest.py",
            "scripts/nccl_backward_isolation.py",
            "slurm/nccl_backward_isolation.sbatch",
        ),
    )


def _initialize(expected_world_size: int) -> tuple[int, int, int, torch.device]:
    """Bind the local CUDA device before eagerly creating the NCCL communicator."""
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
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


def _ddp_only(
    rank: int,
    local_rank: int,
    device: torch.device,
) -> float:
    """Exercise DDP parameter all-reduce without any embedding gather."""
    torch.manual_seed(20260724)
    model = torch.nn.Linear(16, 8).to(device)
    distributed = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
        gradient_as_bucket_view=False,
        static_graph=False,
    )
    features = torch.arange(64, dtype=torch.float32, device=device).reshape(4, 16)
    target = torch.full((4, 8), float(rank), device=device)
    loss = torch.nn.functional.mse_loss(distributed(features), target)
    _event(rank, "ddp_forward_complete", loss=float(loss.detach().cpu()))
    loss.backward()
    torch.cuda.synchronize(device)
    _event(rank, "ddp_backward_synchronized")
    norm = torch.nn.utils.clip_grad_norm_(distributed.parameters(), 1.0)
    torch.cuda.synchronize(device)
    return float(norm.detach().cpu())


def _autograd_all_gather_only(
    rank: int,
    world_size: int,
    device: torch.device,
) -> float:
    """Exercise differentiable all-gather backward without DDP."""
    from torch.distributed.nn.functional import all_gather

    local = (
        torch.arange(32, dtype=torch.float32, device=device).reshape(4, 8)
        + float(rank)
    ).requires_grad_(True)
    gathered = torch.cat(tuple(all_gather(local.contiguous())), dim=0)
    loss = gathered.square().mean()
    _event(
        rank,
        "autograd_all_gather_forward_complete",
        gathered_rows=int(gathered.shape[0]),
        expected_rows=4 * world_size,
        loss=float(loss.detach().cpu()),
    )
    loss.backward()
    torch.cuda.synchronize(device)
    _event(rank, "autograd_all_gather_backward_synchronized")
    if local.grad is None or not bool(torch.isfinite(local.grad).all()):
        raise RuntimeError("Differentiable all-gather returned an invalid gradient")
    return float(torch.linalg.vector_norm(local.grad).detach().cpu())


def main() -> None:
    """Run exactly one collective family and publish a finite receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("ddp_only", "autograd_all_gather_only"),
        required=True,
    )
    parser.add_argument("--output-root", type=Path, required=True)
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
        expected_precision="float32",
    )
    rank, local_rank, world_size, device = _initialize(args.expected_world_size)
    try:
        _event(
            rank,
            "process_group_initialized",
            device=str(device),
            world_size=world_size,
        )
        if args.mode == "ddp_only":
            value = _ddp_only(rank, local_rank, device)
        else:
            value = _autograd_all_gather_only(rank, world_size, device)
        local_value = torch.tensor(value, dtype=torch.float64, device=device)
        gathered_values = [torch.empty_like(local_value) for _ in range(world_size)]
        dist.all_gather(gathered_values, local_value)
        rank_values = [float(item.detach().cpu()) for item in gathered_values]
        if not all(torch.isfinite(item).item() for item in gathered_values):
            raise RuntimeError("A rank reported a non-finite diagnostic value")
        if args.mode == "ddp_only" and len(set(rank_values)) != 1:
            raise RuntimeError("DDP-reduced gradient norms differ across ranks")
        dist.barrier(device_ids=[local_rank])
        _event(rank, "diagnostic_passed", value=value)
        if rank == 0:
            _write_result(
                args.output_root / "result.json",
                {
                    "format_version": "frontier_nccl_backward_isolation_v1",
                    "mode": args.mode,
                    "scope": "synthetic collective correctness only",
                    "status": "passed",
                    "rank_values": rank_values,
                    "world_size": world_size,
                    "source_sha256": source_sha256,
                    "launch_manifest_sha256": (
                        args.expected_launch_manifest_sha256
                    ),
                    "torch_version": torch.__version__,
                    "cuda_version": torch.version.cuda,
                    "nccl_version": list(torch.cuda.nccl.version()),
                    "claim_boundary": (
                        "This synthetic diagnostic provides no throughput, scaling, "
                        "biological-performance, or state-of-the-art evidence."
                    ),
                },
            )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
