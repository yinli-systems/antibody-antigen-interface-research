#!/usr/bin/env python3
"""Isolate forward collectives from a subsequent DDP parameter all-reduce."""

from __future__ import annotations

import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable

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

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from interaction_first.frontier_model import ProtocolD0Retriever  # noqa: E402


MODES = (
    "list_all_gather_default",
    "all_gather_into_tensor_default",
    "zero_buffer_all_reduce_default",
    "list_all_gather_separate_group",
    "protocol_d0_ddp_local_loss",
)


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
    """Hash the exact diagnostic source, model, and launcher boundary."""
    return source_tree_digest(
        REPOSITORY_ROOT,
        (
            "scripts/distributed_launch_contract.py",
            "scripts/generate_runtime_manifest.py",
            "scripts/nccl_forward_collective_ddp_isolation.py",
            "slurm/nccl_forward_collective_ddp_isolation.sbatch",
            "src/interaction_first/__init__.py",
            "src/interaction_first/asymmetric_retrieval.py",
            "src/interaction_first/contacts.py",
            "src/interaction_first/frontier_model.py",
            "src/interaction_first/partners.py",
            "src/interaction_first/runtime.py",
            "src/interaction_first/splits.py",
        ),
    )


def _initialize(expected_world_size: int) -> tuple[int, int, int, torch.device]:
    """Bind the local CUDA device before creating the default NCCL group."""
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size < 2:
        raise RuntimeError("This diagnostic requires at least two ranks")
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


def _wrap_ddp(
    module: torch.nn.Module,
    *,
    local_rank: int,
) -> torch.nn.parallel.DistributedDataParallel:
    """Use the same conservative DDP reducer settings as the failing gate."""
    return torch.nn.parallel.DistributedDataParallel(
        module,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
        gradient_as_bucket_view=False,
        static_graph=False,
    )


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    """Return a stable byte representation that also supports scalar tensors."""
    contiguous = tensor.detach().to(device="cpu").contiguous()
    return contiguous.reshape(-1).view(torch.uint8).numpy().tobytes()


def _named_tensor_sha256(
    values: Iterable[tuple[str, torch.Tensor]],
) -> str:
    """Hash names, metadata, and raw tensor bytes in a deterministic order."""
    digest = hashlib.sha256()
    for name, tensor in values:
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(_tensor_bytes(tensor))
        digest.update(b"\0")
    return digest.hexdigest()


def _state_receipt(
    module: torch.nn.Module,
) -> tuple[str, str, float]:
    """Hash parameters and gradients while rejecting missing or invalid values."""
    parameter_values: list[tuple[str, torch.Tensor]] = []
    gradient_values: list[tuple[str, torch.Tensor]] = []
    squared_gradient_norm = 0.0
    for name, parameter in module.named_parameters():
        parameter_values.append((name, parameter))
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            raise RuntimeError(f"Missing gradient for trainable parameter: {name}")
        if not bool(torch.isfinite(parameter.grad).all()):
            raise RuntimeError(f"Non-finite gradient for trainable parameter: {name}")
        gradient_values.append((name, parameter.grad))
        squared_gradient_norm += float(
            parameter.grad.detach().double().square().sum().cpu()
        )
    if not gradient_values:
        raise RuntimeError("The diagnostic produced no trainable gradients")
    return (
        _named_tensor_sha256(parameter_values),
        _named_tensor_sha256(gradient_values),
        squared_gradient_norm**0.5,
    )


def _assert_receipt_agreement(
    *,
    parameter_sha256: str,
    gradient_sha256: str,
    device: torch.device,
    world_size: int,
) -> None:
    """Require bitwise-identical parameters and DDP-reduced gradients on every rank."""
    payload = bytes.fromhex(parameter_sha256) + bytes.fromhex(gradient_sha256)
    local = torch.tensor(list(payload), dtype=torch.uint8, device=device)
    gathered = [torch.empty_like(local) for _ in range(world_size)]
    dist.all_gather(gathered, local)
    torch.cuda.synchronize(device)
    expected = local.detach().cpu()
    if any(not torch.equal(item.detach().cpu(), expected) for item in gathered):
        raise RuntimeError("Parameter or gradient SHA-256 differs across ranks")


def _linear_forward(
    *,
    rank: int,
    local_rank: int,
    world_size: int,
    device: torch.device,
    mode: str,
    separate_group: Any,
) -> tuple[torch.nn.Module, torch.Tensor, dict[str, Any]]:
    """Run one synthetic linear forward path and the selected collective."""
    torch.manual_seed(20260724)
    model = _wrap_ddp(torch.nn.Linear(16, 8).to(device), local_rank=local_rank)
    features = (
        torch.arange(64, dtype=torch.float32, device=device).reshape(4, 16)
        / 64.0
        + float(rank) / 10.0
    )
    local = model(features)
    torch.cuda.synchronize(device)
    _event(rank, "ddp_forward_synchronized", local_shape=list(local.shape))

    if mode == "list_all_gather_default":
        gathered = [torch.empty_like(local) for _ in range(world_size)]
        dist.all_gather(gathered, local.detach().contiguous())
        gathered[rank] = local
        loss_input = torch.cat(gathered, dim=0)
    elif mode == "all_gather_into_tensor_default":
        gathered_tensor = torch.empty(
            (world_size * int(local.shape[0]), int(local.shape[1])),
            dtype=local.dtype,
            device=device,
        )
        dist.all_gather_into_tensor(
            gathered_tensor,
            local.detach().contiguous(),
        )
        gathered = list(gathered_tensor.chunk(world_size, dim=0))
        gathered[rank] = local
        loss_input = torch.cat(gathered, dim=0)
    elif mode == "zero_buffer_all_reduce_default":
        zero_buffer = torch.zeros_like(local)
        dist.all_reduce(zero_buffer, op=dist.ReduceOp.SUM)
        if bool((zero_buffer != 0).any()):
            raise RuntimeError("SUM all-reduce changed an all-zero buffer")
        loss_input = local
    elif mode == "list_all_gather_separate_group":
        if separate_group is None:
            raise AssertionError("The separate NCCL group was not created")
        gathered = [torch.empty_like(local) for _ in range(world_size)]
        dist.all_gather(
            gathered,
            local.detach().contiguous(),
            group=separate_group,
        )
        gathered[rank] = local
        loss_input = torch.cat(gathered, dim=0)
    else:
        raise ValueError(f"Unsupported linear diagnostic mode: {mode}")

    torch.cuda.synchronize(device)
    _event(
        rank,
        "forward_collective_synchronized",
        collective_mode=mode,
        loss_rows=int(loss_input.shape[0]),
    )
    loss = loss_input.float().square().mean()
    return model, loss, {"loss_rows": int(loss_input.shape[0])}


def _protocol_d0_forward(
    *,
    rank: int,
    local_rank: int,
    device: torch.device,
) -> tuple[torch.nn.Module, torch.Tensor, dict[str, Any]]:
    """Run the production D0 architecture with a local synthetic loss and no gather."""
    torch.manual_seed(20260724)
    model = _wrap_ddp(
        ProtocolD0Retriever(
            antibody_input_dim=512,
            antigen_input_dim=1280,
            embedding_dim=128,
            projection_hidden_dim=256,
            dropout=0.0,
        ).to(device),
        local_rank=local_rank,
    )
    batch_size = 2
    antibody_length = 5
    antigen_length = 7
    antibody = (
        torch.arange(
            batch_size * antibody_length * 512,
            dtype=torch.float32,
            device=device,
        ).reshape(batch_size, antibody_length, 512)
        / float(batch_size * antibody_length * 512)
        + float(rank) / 100.0
    )
    antigen = (
        torch.arange(
            batch_size * antigen_length * 1280,
            dtype=torch.float32,
            device=device,
        ).reshape(batch_size, antigen_length, 1280)
        / float(batch_size * antigen_length * 1280)
        - float(rank) / 100.0
    )
    antibody_mask = torch.ones(
        (batch_size, antibody_length),
        dtype=torch.bool,
        device=device,
    )
    antigen_mask = torch.ones(
        (batch_size, antigen_length),
        dtype=torch.bool,
        device=device,
    )
    antibody_chain_ids = (
        torch.arange(antibody_length, device=device)
        .remainder(2)
        .unsqueeze(0)
        .expand(batch_size, -1)
        .contiguous()
    )
    output = model(
        antibody,
        antigen,
        antibody_mask,
        antigen_mask,
        antibody_chain_ids,
    )
    similarities = output["similarity_matrix"].float()
    loss = -torch.diagonal(similarities).mean()
    torch.cuda.synchronize(device)
    _event(
        rank,
        "protocol_d0_local_forward_synchronized",
        similarity_shape=list(similarities.shape),
        collective_mode="none",
    )
    return model, loss, {
        "antibody_shape": list(antibody.shape),
        "antigen_shape": list(antigen.shape),
        "similarity_shape": list(similarities.shape),
    }


def main() -> None:
    """Run one forward-collective/DDP-backward combination and publish a receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, required=True)
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
    separate_group = None
    try:
        _event(
            rank,
            "process_group_initialized",
            device=str(device),
            world_size=world_size,
        )
        if args.mode == "list_all_gather_separate_group":
            separate_group = dist.new_group(
                ranks=list(range(world_size)),
                backend="nccl",
                timeout=timedelta(minutes=5),
            )
            torch.cuda.synchronize(device)
            _event(rank, "separate_nccl_group_initialized")

        if args.mode == "protocol_d0_ddp_local_loss":
            model, loss, forward_metadata = _protocol_d0_forward(
                rank=rank,
                local_rank=local_rank,
                device=device,
            )
        else:
            model, loss, forward_metadata = _linear_forward(
                rank=rank,
                local_rank=local_rank,
                world_size=world_size,
                device=device,
                mode=args.mode,
                separate_group=separate_group,
            )

        _event(rank, "backward_started", loss=float(loss.detach().cpu()))
        loss.backward()
        torch.cuda.synchronize(device)
        _event(rank, "ddp_backward_synchronized")

        parameter_sha256, gradient_sha256, gradient_norm = _state_receipt(model)
        _assert_receipt_agreement(
            parameter_sha256=parameter_sha256,
            gradient_sha256=gradient_sha256,
            device=device,
            world_size=world_size,
        )
        _event(
            rank,
            "state_receipt_agreement_passed",
            gradient_norm=gradient_norm,
            gradient_sha256=gradient_sha256,
            parameter_sha256=parameter_sha256,
        )
        dist.barrier(device_ids=[local_rank])
        if rank == 0:
            _write_result(
                args.output_root / "result.json",
                {
                    "format_version": "frontier_nccl_forward_ddp_isolation_v1",
                    "mode": args.mode,
                    "scope": "synthetic collective and DDP correctness only",
                    "status": "passed",
                    "world_size": world_size,
                    "source_sha256": source_sha256,
                    "launch_manifest_sha256": (
                        args.expected_launch_manifest_sha256
                    ),
                    "loss_rank0": float(loss.detach().cpu()),
                    "gradient_norm": gradient_norm,
                    "gradient_sha256": gradient_sha256,
                    "parameter_sha256": parameter_sha256,
                    "forward_metadata": forward_metadata,
                    "torch_version": torch.__version__,
                    "cuda_version": torch.version.cuda,
                    "nccl_version": list(torch.cuda.nccl.version()),
                    "environment": safe_environment_snapshot(),
                    "claim_boundary": (
                        "This synthetic diagnostic provides no throughput, scaling, "
                        "biological-performance, or state-of-the-art evidence."
                    ),
                },
            )
    finally:
        if separate_group is not None:
            dist.destroy_process_group(separate_group)
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
