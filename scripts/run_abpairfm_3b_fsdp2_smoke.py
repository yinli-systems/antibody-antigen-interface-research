#!/usr/bin/env python3
"""Run a real-3B, synthetic-data FSDP2 systems gate on RTX 4090 GPUs."""

from __future__ import annotations

import argparse
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import socket
import tempfile
import time

import torch

from antibody_pretraining.pair_aware_distributed import (
    fully_shard_pair_aware_esm2,
    load_authenticated_esm2_into_fsdp2,
    optimizer_after_fsdp2,
)
from antibody_pretraining.pair_aware_esm2 import (
    PairAwareESM2,
    PairAwareESM2Config,
    reset_selective_attention_checkpoint_policy_counts,
    selective_attention_checkpoint_policy_counts,
)


CLAIM_BOUNDARY = (
    "Synthetic dual-chain MLM systems evidence using the full 2.839B-parameter "
    "model. It does not authorize biological training or establish data quality, "
    "development performance, scaling efficiency, or SOTA."
)


def _trace(rank: int, phase: str, **values: object) -> None:
    suffix = " ".join(f"{key}={value}" for key, value in sorted(values.items()))
    print(f"ABPAIRFM_SYSTEM rank={rank} phase={phase} {suffix}".rstrip(), flush=True)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-size", type=int, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-world-size", type=int, required=True)
    parser.add_argument(
        "--shard-group-size",
        type=int,
        default=0,
        help="FSDP shard width; zero uses the full world, a divisor enables HSDP.",
    )
    parser.add_argument("--sequence-tokens", type=int, default=128)
    parser.add_argument(
        "--microbatch-pairs",
        type=int,
        default=1,
        help="Independent dual-chain pairs processed per rank and optimizer step.",
    )
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--measured-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument(
        "--activation-checkpoint-policy",
        choices=("full", "selective_attention"),
        default="full",
    )
    parser.add_argument(
        "--reshard-after-forward",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Select FSDP2 full-shard (true) or keep gathered parameters for backward (false).",
    )
    return parser.parse_args()


def _validate(arguments: argparse.Namespace) -> None:
    if arguments.expected_world_size not in {4, 8}:
        raise ValueError("systems gate requires exactly four or eight GPUs")
    if arguments.sequence_tokens < 16 or arguments.sequence_tokens % 2:
        raise ValueError("sequence-tokens must be even and at least 16")
    if arguments.sequence_tokens > 1024:
        raise ValueError("sequence-tokens exceeds the frozen model limit")
    if not 1 <= arguments.microbatch_pairs <= 8:
        raise ValueError("microbatch-pairs must be in [1, 8]")
    if arguments.warmup_steps < 0 or arguments.measured_steps < 1:
        raise ValueError("invalid warmup/measured step counts")
    if arguments.seed < 0:
        raise ValueError("seed must be non-negative")
    _mesh_shape(arguments.expected_world_size, arguments.shard_group_size)


def _mesh_shape(world_size: int, shard_group_size: int) -> tuple[int, int] | None:
    if shard_group_size == 0 or shard_group_size == world_size:
        return None
    if shard_group_size < 1 or shard_group_size > world_size:
        raise ValueError("shard-group-size must be zero or in [1, world-size]")
    if world_size % shard_group_size:
        raise ValueError("shard-group-size must divide world-size")
    return world_size // shard_group_size, shard_group_size


def _synthetic_pair(
    *,
    rank: int,
    sequence_tokens: int,
    microbatch_pairs: int = 1,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + rank * 1009)
    tokens = torch.randint(
        4,
        24,
        (microbatch_pairs, sequence_tokens),
        generator=generator,
    )
    half = sequence_tokens // 2
    tokens[:, 0] = 0
    tokens[:, half - 1] = 2
    tokens[:, half] = 0
    tokens[:, -1] = 2
    chain_ids = torch.cat(
        (torch.zeros(half, dtype=torch.long), torch.ones(half, dtype=torch.long))
    ).unsqueeze(0).expand(microbatch_pairs, -1).clone()
    labels = torch.full_like(tokens, -100)
    eligible = torch.ones_like(tokens, dtype=torch.bool)
    eligible[:, [0, half - 1, half, sequence_tokens - 1]] = False
    selected = eligible & (torch.rand(tokens.shape, generator=generator) < 0.15)
    for batch_index in range(microbatch_pairs):
        for offset in (0, half):
            chain_selected = selected[batch_index, offset : offset + half]
            if not bool(chain_selected.any()):
                selected[batch_index, offset + 1] = True
    labels[selected] = tokens[selected]
    replacement = torch.rand(tokens.shape, generator=generator)
    tokens[selected & replacement.lt(0.8)] = 32
    random_positions = selected & replacement.ge(0.8) & replacement.lt(0.9)
    random_count = int(random_positions.sum())
    if random_count:
        tokens[random_positions] = torch.randint(
            4, 24, (random_count,), generator=generator
        )
    residue_count = microbatch_pairs * (sequence_tokens - 4)
    return (
        tokens.to(device, non_blocking=True),
        chain_ids.to(device, non_blocking=True),
        labels.to(device, non_blocking=True),
        residue_count,
    )


def _collect(value: object, world_size: int) -> list[object]:
    gathered: list[object] = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(gathered, value)
    return gathered


def _publish_create_only(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_path, path)
    except FileExistsError as error:
        raise FileExistsError(f"Refusing to overwrite systems receipt: {path}") from error
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> None:
    arguments = _arguments()
    _validate(arguments)
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.distributed.init_process_group(
        backend="nccl",
        device_id=device,
        timeout=timedelta(minutes=5),
    )
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    try:
        _trace(rank, "process_group_ready", world_size=world_size)
        if world_size != arguments.expected_world_size:
            raise RuntimeError(
                f"world size {world_size} differs from {arguments.expected_world_size}"
            )
        properties = torch.cuda.get_device_properties(device)
        if properties.name != "NVIDIA GeForce RTX 4090":
            raise RuntimeError(f"Forbidden GPU model: {properties.name}")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("Selected RTX 4090 runtime does not support bfloat16")
        torch.manual_seed(arguments.seed + rank)
        torch.cuda.manual_seed(arguments.seed + rank)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

        mesh_shape = _mesh_shape(world_size, arguments.shard_group_size)
        mesh = None
        if mesh_shape is not None:
            from torch.distributed.device_mesh import init_device_mesh

            mesh = init_device_mesh(
                "cuda",
                mesh_shape,
                mesh_dim_names=("replicate", "shard"),
            )
        _trace(rank, "data_parallel_mesh_ready", mesh_shape=mesh_shape or (1, world_size))

        config = PairAwareESM2Config(
            activation_checkpointing=True,
            activation_checkpoint_policy=arguments.activation_checkpoint_policy,
        )
        with torch.device("meta"):
            model = PairAwareESM2(config)
        _trace(rank, "meta_model_ready")
        fully_shard_pair_aware_esm2(
            model,
            mesh=mesh,
            precision="bfloat16",
            reshard_after_forward=arguments.reshard_after_forward,
        )
        _trace(rank, "fsdp2_sharding_ready")
        initialization = load_authenticated_esm2_into_fsdp2(
            model,
            arguments.checkpoint,
            expected_size=arguments.expected_size,
            expected_sha256=arguments.expected_sha256,
        )
        _trace(rank, "checkpoint_loaded")
        optimizer = optimizer_after_fsdp2(
            model,
            learning_rate=arguments.learning_rate,
            weight_decay=arguments.weight_decay,
        )
        _trace(rank, "optimizer_ready")
        tokens, chain_ids, labels, residues_per_rank = _synthetic_pair(
            rank=rank,
            sequence_tokens=arguments.sequence_tokens,
            microbatch_pairs=arguments.microbatch_pairs,
            seed=arguments.seed,
            device=device,
        )
        model.train()
        reset_selective_attention_checkpoint_policy_counts()
        torch.cuda.reset_peak_memory_stats(device)
        step_records: list[dict[str, float | int]] = []
        from torch.nn.attention import SDPBackend, sdpa_kernel

        total_steps = arguments.warmup_steps + arguments.measured_steps
        for step in range(total_steps):
            _trace(rank, "step_start", step=step)
            optimizer.zero_grad(set_to_none=True)
            torch.distributed.barrier()
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                output = model(tokens, chain_ids, labels=labels)
                if output.balanced_chain_loss_sum is None:
                    raise RuntimeError("MLM model did not return a training loss")
                local_chain_count = output.contributing_chain_count.clone()
                global_chain_count = local_chain_count.clone()
                torch.distributed.all_reduce(global_chain_count)
                loss = (
                    output.balanced_chain_loss_sum
                    * world_size
                    / global_chain_count.to(output.balanced_chain_loss_sum.dtype)
                )
            _trace(rank, "forward_done", step=step)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Non-finite synthetic MLM loss")
            loss.backward()
            _trace(rank, "backward_done", step=step)
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not bool(torch.isfinite(gradient_norm)):
                raise RuntimeError("Non-finite gradient norm")
            optimizer.step()
            _trace(rank, "optimizer_step_done", step=step)
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            detached_loss = output.balanced_chain_loss_sum.detach()
            torch.distributed.all_reduce(detached_loss)
            global_mean_loss = detached_loss / global_chain_count
            if step >= arguments.warmup_steps:
                step_records.append(
                    {
                        "elapsed_seconds": elapsed,
                        "global_mean_loss": float(global_mean_loss),
                        "gradient_norm": float(gradient_norm),
                        "step": step,
                    }
                )

        rank_record = {
            "device_index": local_rank,
            "device_name": properties.name,
            "hostname": socket.gethostname(),
            "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "max_memory_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "rank": rank,
            "selective_checkpoint_dispatch": (
                selective_attention_checkpoint_policy_counts()
            ),
            "steps": step_records,
            "total_memory_bytes": properties.total_memory,
        }
        ranks = _collect(rank_record, world_size)
        _trace(rank, "rank_metrics_gathered")
        if rank == 0:
            elapsed_maxima = [
                max(float(record["steps"][index]["elapsed_seconds"]) for record in ranks)  # type: ignore[index]
                for index in range(arguments.measured_steps)
            ]
            tokens_per_step = residues_per_rank * world_size
            throughputs = [tokens_per_step / elapsed for elapsed in elapsed_maxima]
            payload: dict[str, object] = {
                "architecture": {
                    "activation_checkpointing": True,
                    "activation_checkpoint_policy": config.activation_checkpoint_policy,
                    "attention": "pair_aware_lifted_flash_sdpa",
                    "embed_dimension": config.embed_dimension,
                    "head_count": config.head_count,
                    "layer_count": config.layer_count,
                    "parameter_count": 2839005633,
                },
                "claim_boundary": CLAIM_BOUNDARY,
                "format_version": "abpairfm_3b_fsdp2_systems_smoke_v1",
                "fsdp2": {
                    "mesh_shape_replicate_by_shard": list(mesh_shape or (1, world_size)),
                    "reshard_after_forward": arguments.reshard_after_forward,
                },
                "initialization": initialization,
                "microbatch_pairs_per_rank": arguments.microbatch_pairs,
                "precision": "bfloat16",
                "ranks": ranks,
                "sequence_tokens_per_rank": arguments.sequence_tokens,
                "status": "PASS",
                "synthetic_data": True,
                "test_access": {
                    "asep_test_accessed": False,
                    "sabdab2_heldout_accessed": False,
                },
                "throughput": {
                    "global_residue_tokens_per_step": tokens_per_step,
                    "measured_step_count": arguments.measured_steps,
                    "residue_tokens_per_second_by_step": throughputs,
                    "residue_tokens_per_second_mean": sum(throughputs) / len(throughputs),
                    "step_seconds_max_by_step": elapsed_maxima,
                    "warmup_step_count": arguments.warmup_steps,
                },
                "torch": {
                    "cuda_version": torch.version.cuda,
                    "flash_sdpa_forced": True,
                    "version": torch.__version__,
                },
                "world_size": world_size,
            }
            _publish_create_only(arguments.output, payload)
            _trace(rank, "receipt_published", output=arguments.output)
        torch.distributed.barrier()
    finally:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
