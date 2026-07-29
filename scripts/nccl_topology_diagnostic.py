#!/usr/bin/env python3
"""Diagnose single-node NCCL collectives without accessing biological data."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import socket
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from distributed_launch_contract import (
    LaunchContractError,
    publish_create_only_json,
    require_safe_label,
    require_sha256,
    safe_environment_snapshot,
    source_tree_digest,
    validate_distributed_identity,
    verify_launch_manifest,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _event(rank: int, stage: str, **fields: Any) -> None:
    """Emit one rank-scoped diagnostic event immediately."""
    print(
        json.dumps(
            {"rank": rank, "stage": stage, **fields},
            sort_keys=True,
            allow_nan=False,
        ),
        flush=True,
    )


def _atomic_json(path: Path, value: Any) -> None:
    """Publish one create-only finite JSON result."""
    publish_create_only_json(path, value)


def _source_digest() -> str:
    """Hash the exact topology-diagnostic source and launcher boundary."""
    return source_tree_digest(
        REPOSITORY_ROOT,
        (
            "scripts/distributed_launch_contract.py",
            "scripts/generate_runtime_manifest.py",
            "scripts/nccl_topology_diagnostic.py",
            "slurm/nccl_topology_diagnostic.sbatch",
        ),
    )


def _tensor_digest(values: list[tuple[str, torch.Tensor]]) -> str:
    """Hash sorted tensor names, metadata, and exact CPU bytes."""
    digest = hashlib.sha256()
    for name, tensor in sorted(values):
        value = tensor.detach().cpu().contiguous()
        header = json.dumps(
            {
                "name": name,
                "dtype": str(value.dtype),
                "shape": list(value.shape),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        payload = value.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _require_rank_digest_agreement(
    digest: str,
    *,
    device: torch.device,
    world_size: int,
) -> None:
    """Require one SHA-256 receipt to be bitwise identical on every rank."""
    local = torch.tensor(list(bytes.fromhex(digest)), dtype=torch.uint8, device=device)
    gathered = [torch.empty_like(local) for _ in range(world_size)]
    dist.all_gather(gathered, local)
    expected = local.detach().cpu()
    if any(not torch.equal(value.detach().cpu(), expected) for value in gathered):
        raise RuntimeError("DDP state or gradient digest differs across ranks")


def main() -> None:
    """Test initialization, all-reduce, DDP broadcast, and DDP backward."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--expected-world-size", type=int, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--launch-manifest", type=Path, required=True)
    parser.add_argument("--expected-launch-manifest-sha256", required=True)
    parser.add_argument("--synthetic-only", action="store_true")
    args = parser.parse_args()
    if not args.synthetic_only:
        raise ValueError("--synthetic-only is mandatory")
    require_safe_label(args.mode, field="mode")
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
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if not torch.cuda.is_available():
        raise RuntimeError("This diagnostic requires CUDA")
    validate_distributed_identity(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        visible_device_count=torch.cuda.device_count(),
        minimum_world_size=2,
        expected_world_size=args.expected_world_size,
    )
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    properties = torch.cuda.get_device_properties(device)
    _event(
        rank,
        "device_selected",
        local_rank=local_rank,
        name=properties.name,
        compute_capability=f"{properties.major}.{properties.minor}",
        hostname=socket.gethostname(),
        pid=os.getpid(),
    )
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        timeout=datetime.timedelta(seconds=120),
    )
    try:
        _event(rank, "process_group_initialized")
        value = torch.tensor(float(rank + 1), device=device)
        dist.all_reduce(value)
        expected = world_size * (world_size + 1) / 2
        if float(value.cpu()) != expected:
            raise RuntimeError("NCCL all-reduce returned an incorrect value")
        _event(rank, "all_reduce_passed", value=float(value.cpu()))
        torch.manual_seed(20260724)
        model = torch.nn.Linear(4, 4).to(device)
        distributed_model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
        )
        _event(rank, "ddp_constructor_passed")
        features = torch.arange(8, dtype=torch.float32, device=device).reshape(2, 4)
        loss = distributed_model(features).square().mean()
        loss.backward()
        if any(
            parameter.grad is None
            or not bool(torch.isfinite(parameter.grad).all().item())
            for parameter in distributed_model.parameters()
        ):
            raise RuntimeError("DDP backward produced a missing or non-finite gradient")
        parameter_sha256 = _tensor_digest(
            list(distributed_model.named_parameters())
        )
        gradient_sha256 = _tensor_digest(
            [
                (name, parameter.grad)
                for name, parameter in distributed_model.named_parameters()
                if parameter.grad is not None
            ]
        )
        _require_rank_digest_agreement(
            parameter_sha256,
            device=device,
            world_size=world_size,
        )
        _require_rank_digest_agreement(
            gradient_sha256,
            device=device,
            world_size=world_size,
        )
        torch.cuda.synchronize(device)
        _event(rank, "ddp_backward_passed", loss=float(loss.detach().cpu()))
        dist.barrier(device_ids=[local_rank])
        if rank == 0:
            result = {
                "format": "frontier_nccl_topology_diagnostic_v1",
                "status": "passed",
                "scope": "synthetic collective correctness only",
                "mode": args.mode,
                "world_size": world_size,
                "source_sha256": source_sha256,
                "launch_manifest_sha256": args.expected_launch_manifest_sha256,
                "parameter_sha256": parameter_sha256,
                "gradient_sha256": gradient_sha256,
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "nccl_version": list(torch.cuda.nccl.version()),
                "environment": safe_environment_snapshot(),
                "claim_boundary": (
                    "This synthetic single-node diagnostic provides no throughput, "
                    "multi-node scaling, biological-performance, or "
                    "state-of-the-art evidence."
                ),
            }
            _atomic_json(args.output_root / "result.json", result)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
