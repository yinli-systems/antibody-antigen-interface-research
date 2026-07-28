#!/usr/bin/env python3
"""Run a receipt-bound optimized antibody MLM training candidate."""

from __future__ import annotations

import argparse
import contextlib
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import socket
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from antibody_pretraining.dense_shard import load_dense_shard
from antibody_pretraining.masking import apply_uniform_mlm
from antibody_pretraining.model import (
    AntibodyMaskedLanguageModel,
    config_from_preset,
    load_model_family,
)
from antibody_pretraining.tokenizer import PAD_ID
from antibody_pretraining.training_contract import (
    build_update_plan,
    learning_rate_at_token,
    load_training_protocol,
)
from run_antibody_pretraining import (
    _distributed_tree_digest,
    _load_dcp,
    _parameter_moments,
    _publish,
    _save_dcp,
    _sha256,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--model-family", type=Path, required=True)
    parser.add_argument("--training-protocol", type=Path, required=True)
    parser.add_argument("--efficiency-protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--preset",
        choices=("abf_300m", "abf_650m", "abf_1_3b"),
        required=True,
    )
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--micro-batch", type=int, default=16)
    parser.add_argument("--target-tokens", type=int, default=16384)
    parser.add_argument("--batch-quantum", type=int, default=128)
    parser.add_argument("--backend", choices=("ddp", "fsdp2"), required=True)
    parser.add_argument("--resume-root", type=Path)
    parser.add_argument("--compile-blocks", action="store_true")
    parser.add_argument("--disable-compiled-cudagraphs", action="store_true")
    parser.add_argument("--deterministic-algorithms", action="store_true")
    parser.add_argument("--compile-prewarm-buckets", action="store_true")
    parser.add_argument("--fused-adamw", action="store_true")
    parser.add_argument("--pinned-staging", action="store_true")
    parser.add_argument("--bucketed-padding", action="store_true")
    parser.add_argument("--length-bucketing", action="store_true")
    parser.add_argument(
        "--length-bucket-boundaries",
        default="96,128,160,194",
    )
    parser.add_argument(
        "--fsdp-output-dtype",
        choices=("bfloat16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--ddp-bucket-cap-mb", type=int, default=25)
    parser.add_argument("--skip-checkpoint", action="store_true")
    args = parser.parse_args()
    if args.steps <= 0 or args.micro_batch <= 0 or args.target_tokens <= 0:
        raise ValueError("Pilot dimensions must be positive")
    if args.batch_quantum <= 0 or args.ddp_bucket_cap_mb <= 0:
        raise ValueError("Batch quantum and DDP bucket size must be positive")
    if args.compile_prewarm_buckets and not args.compile_blocks:
        raise ValueError("Compile prewarm requires compiled Transformer blocks")
    if args.disable_compiled_cudagraphs and not args.compile_blocks:
        raise ValueError("CUDAGraph control requires compiled Transformer blocks")
    return args


def _load_efficiency_protocol(path: Path) -> tuple[dict[str, object], str]:
    payload = path.read_bytes()
    try:
        protocol = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Efficiency protocol must be ASCII JSON") from error
    if protocol.get("format_version") != "antibody_efficiency_gate_v1":
        raise ValueError("Unexpected efficiency protocol version")
    return protocol, hashlib.sha256(payload).hexdigest()


def _parse_boundaries(raw: str, maximum: int) -> tuple[int, ...]:
    try:
        boundaries = tuple(int(value) for value in raw.split(","))
    except ValueError as error:
        raise ValueError("Length bucket boundaries must be integers") from error
    if (
        not boundaries
        or any(value <= 0 for value in boundaries)
        or tuple(sorted(set(boundaries))) != boundaries
        or boundaries[-1] != maximum
    ):
        raise ValueError("Length bucket boundaries must increase to model maximum")
    return boundaries


def _bucket_width(maximum_length: int, boundaries: tuple[int, ...]) -> int:
    for boundary in boundaries:
        if maximum_length <= boundary:
            return boundary
    raise RuntimeError("Sequence length exceeds the frozen bucket boundaries")


def _ordered_indices(
    indices: np.ndarray,
    residue_lengths: np.ndarray,
    enabled: bool,
) -> np.ndarray:
    if not enabled:
        return indices
    positions = np.arange(len(indices), dtype=np.int64)
    order = np.lexsort((positions, residue_lengths[indices]))
    ordered = indices[order]
    if sorted(ordered.tolist()) != sorted(indices.tolist()):
        raise RuntimeError("Length bucketing changed update membership")
    return ordered


def _membership_update(
    digest: hashlib._Hash,
    *,
    update: int,
    sample_cursor: int,
    indices: np.ndarray,
) -> None:
    header = json.dumps(
        {
            "sample_cursor": sample_cursor,
            "sequence_count": len(indices),
            "update": update,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    digest.update(header + b"\0")
    digest.update(np.asarray(indices, dtype="<i8").tobytes())


def _stage_tensor(array: np.ndarray, pinned: bool) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(array))
    return tensor.pin_memory() if pinned else tensor


def _compile_transformer_blocks(
    model: AntibodyMaskedLanguageModel,
    *,
    disable_cudagraphs: bool,
) -> None:
    options = {"triton.cudagraphs": False} if disable_cudagraphs else None
    for block in model.blocks:
        block.compile(backend="inductor", fullgraph=True, options=options)


def _wrap_model(
    model: AntibodyMaskedLanguageModel,
    *,
    args: argparse.Namespace,
    device: torch.device,
    local_rank: int,
    world_size: int,
) -> torch.nn.Module:
    if args.compile_blocks:
        _compile_transformer_blocks(
            model,
            disable_cudagraphs=args.disable_compiled_cudagraphs,
        )
    if args.backend == "ddp":
        return DistributedDataParallel(
            model,
            device_ids=[local_rank],
            bucket_cap_mb=args.ddp_bucket_cap_mb,
            gradient_as_bucket_view=True,
            static_graph=True,
        )
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    mesh = init_device_mesh("cuda", (world_size,))
    output_dtype = (
        torch.bfloat16 if args.fsdp_output_dtype == "bfloat16" else torch.float32
    )
    policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        output_dtype=output_dtype,
    )
    for block in model.blocks:
        fully_shard(block, mesh=mesh, mp_policy=policy)
    fully_shard(model, mesh=mesh, mp_policy=policy)
    return model


def _optimizer(
    model: torch.nn.Module,
    protocol: dict[str, object],
    preset: str,
    fused: bool,
) -> torch.optim.Optimizer:
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for parameter in model.parameters():
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    optimizer_spec = protocol["optimizer"]
    return torch.optim.AdamW(
        [
            {
                "params": decay,
                "weight_decay": optimizer_spec["weight_decay"],
            },
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=protocol["presets"][preset]["learning_rate"],
        betas=tuple(optimizer_spec["betas"]),
        eps=optimizer_spec["epsilon"],
        fused=fused,
    )


def _prewarm_compiled_buckets(
    model: torch.nn.Module,
    *,
    boundaries: tuple[int, ...],
    device: torch.device,
    micro_batch: int,
    world_size: int,
) -> list[dict[str, float | int]]:
    trace: list[dict[str, float | int]] = []
    model.train()
    for width in boundaries:
        model.zero_grad(set_to_none=True)
        input_ids = torch.ones(
            (micro_batch, width),
            dtype=torch.long,
            device=device,
        )
        type_ids = torch.ones_like(input_ids)
        attention = torch.ones_like(input_ids, dtype=torch.bool)
        labels = torch.full_like(input_ids, -100)
        labels[:, 0] = 1
        started = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(input_ids, type_ids, attention, labels)
            loss = output.loss_sum * world_size / (micro_batch * world_size)
        loss.backward()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        trace.append(
            {
                "micro_batch_per_gpu": micro_batch,
                "seconds": elapsed,
                "width": width,
            }
        )
    model.zero_grad(set_to_none=True)
    dist.barrier()
    return trace


def _rng_state_payload(rank: int, device: torch.device) -> dict[str, object]:
    """Capture every process-local RNG stream without pickle serialization."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    payload = {
        "format_version": "antibody_rank_rng_state_v1",
        "numpy": {
            "algorithm": numpy_state[0],
            "cached_gaussian": float(numpy_state[4]),
            "has_gaussian": int(numpy_state[3]),
            "keys": [int(value) for value in numpy_state[1]],
            "position": int(numpy_state[2]),
        },
        "python": {
            "gaussian_next": python_state[2],
            "internal_state": [int(value) for value in python_state[1]],
            "version": int(python_state[0]),
        },
        "rank": rank,
        "torch_cpu_state_hex": torch.get_rng_state().cpu().numpy().tobytes().hex(),
        "torch_cuda_state_hex": (
            torch.cuda.get_rng_state(device).cpu().numpy().tobytes().hex()
        ),
    }
    payload["state_sha256"] = hashlib.sha256(
        (
            json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("ascii")
    ).hexdigest()
    return payload


def _validated_rng_state(path: Path, expected_rank: int) -> dict[str, object]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("RNG state must be a regular non-symlink file")
    try:
        payload = json.loads(path.read_text(encoding="ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("RNG state must be ASCII JSON") from error
    if (
        not isinstance(payload, dict)
        or payload.get("format_version") != "antibody_rank_rng_state_v1"
        or payload.get("rank") != expected_rank
    ):
        raise RuntimeError("RNG state identity differs")
    unsigned = dict(payload)
    recorded = unsigned.pop("state_sha256", None)
    observed = hashlib.sha256(
        (
            json.dumps(unsigned, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("ascii")
    ).hexdigest()
    if recorded != observed:
        raise RuntimeError("RNG state self-hash differs")
    return payload


def _save_rng_checkpoint(
    checkpoint_root: Path,
    state: dict[str, object],
    *,
    rank: int,
    world_size: int,
) -> str:
    """Publish one authenticated RNG state per rank and a shared manifest."""
    rng_root = checkpoint_root / "rng"
    if rank == 0:
        rng_root.mkdir()
    dist.barrier()
    _publish(rng_root / f"rank-{rank:05d}.json", state)
    dist.barrier()
    manifest_path = checkpoint_root / "rng_manifest.json"
    if rank == 0:
        paths = sorted(rng_root.glob("rank-*.json"))
        if len(paths) != world_size or any(path.is_symlink() for path in paths):
            raise RuntimeError("RNG checkpoint rank inventory differs")
        files = []
        for expected_rank, path in enumerate(paths):
            payload = _validated_rng_state(path, expected_rank)
            files.append(
                {
                    "file_sha256": _sha256(path),
                    "path": f"rng/{path.name}",
                    "rank": expected_rank,
                    "state_sha256": payload["state_sha256"],
                }
            )
        manifest = {
            "files": files,
            "format_version": "antibody_rng_checkpoint_manifest_v1",
            "world_size": world_size,
        }
        manifest["manifest_sha256"] = hashlib.sha256(
            (
                json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
            ).encode("ascii")
        ).hexdigest()
        _publish(manifest_path, manifest)
    dist.barrier()
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    return str(manifest["manifest_sha256"])


def _restore_rng_checkpoint(
    checkpoint_root: Path,
    *,
    rank: int,
    world_size: int,
    device: torch.device,
) -> str:
    """Authenticate and restore one rank-local RNG checkpoint."""
    manifest_path = checkpoint_root / "rng_manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise RuntimeError("RNG checkpoint manifest is absent")
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    unsigned = dict(manifest)
    recorded = unsigned.pop("manifest_sha256", None)
    observed = hashlib.sha256(
        (
            json.dumps(unsigned, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("ascii")
    ).hexdigest()
    files = manifest.get("files")
    if (
        recorded != observed
        or manifest.get("format_version") != "antibody_rng_checkpoint_manifest_v1"
        or manifest.get("world_size") != world_size
        or not isinstance(files, list)
        or len(files) != world_size
    ):
        raise RuntimeError("RNG checkpoint manifest differs")
    entry = files[rank]
    expected_path = f"rng/rank-{rank:05d}.json"
    if entry.get("rank") != rank or entry.get("path") != expected_path:
        raise RuntimeError("RNG checkpoint rank mapping differs")
    state_path = checkpoint_root / expected_path
    if _sha256(state_path) != entry.get("file_sha256"):
        raise RuntimeError("RNG checkpoint file hash differs")
    state = _validated_rng_state(state_path, rank)
    if state.get("state_sha256") != entry.get("state_sha256"):
        raise RuntimeError("RNG checkpoint state hash differs")
    python_state = state["python"]
    numpy_state = state["numpy"]
    random.setstate(
        (
            int(python_state["version"]),
            tuple(int(value) for value in python_state["internal_state"]),
            python_state["gaussian_next"],
        )
    )
    np.random.set_state(
        (
            str(numpy_state["algorithm"]),
            np.asarray(numpy_state["keys"], dtype=np.uint32),
            int(numpy_state["position"]),
            int(numpy_state["has_gaussian"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(
        torch.tensor(
            list(bytes.fromhex(str(state["torch_cpu_state_hex"]))),
            dtype=torch.uint8,
        )
    )
    torch.cuda.set_rng_state(
        torch.tensor(
            list(bytes.fromhex(str(state["torch_cuda_state_hex"]))),
            dtype=torch.uint8,
        ),
        device,
    )
    return str(recorded)


def main() -> None:
    args = _arguments()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    telemetry_interval_seconds = int(
        os.environ.get("ABF_TELEMETRY_INTERVAL_SECONDS", "30")
    )
    if telemetry_interval_seconds <= 0:
        raise ValueError("Telemetry interval must be positive")
    if world_size <= 1:
        raise RuntimeError("The optimized runner requires distributed execution")
    if not torch.cuda.is_available() or local_rank >= torch.cuda.device_count():
        raise RuntimeError("CUDA rank identity is invalid")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if rank == 0:
        if args.output.exists():
            raise ValueError("Output must be create-only")
        args.output.mkdir(parents=True)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        timeout=datetime.timedelta(minutes=10),
        device_id=device,
    )
    dist.barrier()
    random.seed(20260728)
    np.random.seed(20260728)
    torch.manual_seed(20260728)
    torch.cuda.manual_seed_all(20260728)
    torch.use_deterministic_algorithms(
        args.deterministic_algorithms,
        warn_only=False,
    )
    torch.backends.cudnn.deterministic = args.deterministic_algorithms
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats(device)

    efficiency_protocol, efficiency_protocol_sha256 = _load_efficiency_protocol(
        args.efficiency_protocol
    )
    shard = load_dense_shard(
        args.shard,
        expected_manifest_sha256=args.manifest_sha256,
    )
    family = load_model_family(args.model_family)
    protocol = load_training_protocol(args.training_protocol)
    boundaries = _parse_boundaries(
        args.length_bucket_boundaries,
        family["maximum_sequence_tokens"],
    )
    residue_lengths = np.asarray(shard.lengths, dtype=np.int64) - 2
    model = AntibodyMaskedLanguageModel(config_from_preset(family, args.preset)).to(
        device
    )
    model = _wrap_model(
        model,
        args=args,
        device=device,
        local_rank=local_rank,
        world_size=world_size,
    )
    optimizer = _optimizer(model, protocol, args.preset, args.fused_adamw)

    compile_prewarm_trace: list[dict[str, float | int]] = []
    if args.compile_prewarm_buckets:
        compile_prewarm_trace = _prewarm_compiled_buckets(
            model,
            boundaries=boundaries,
            device=device,
            micro_batch=args.micro_batch,
            world_size=world_size,
        )

    model_family_sha256 = _sha256(args.model_family)
    training_protocol_sha256 = _sha256(args.training_protocol)
    cursor = 0
    consumed = 0
    start_update = 0
    resume_rng_manifest_sha256 = None
    if args.resume_root is not None:
        state = json.loads(
            (args.resume_root / "trainer_state.json").read_text(encoding="ascii")
        )
        expected = {
            "backend": args.backend,
            "batch_quantum": args.batch_quantum,
            "efficiency_protocol_sha256": efficiency_protocol_sha256,
            "length_bucket_boundaries": list(boundaries),
            "micro_batch_per_gpu": args.micro_batch,
            "model_family_sha256": model_family_sha256,
            "optimization": {
                "bucketed_padding": args.bucketed_padding,
                "compile_blocks": args.compile_blocks,
                "disable_compiled_cudagraphs": args.disable_compiled_cudagraphs,
                "compile_prewarm_buckets": args.compile_prewarm_buckets,
                "ddp_bucket_cap_mb": args.ddp_bucket_cap_mb,
                "deterministic_algorithms": args.deterministic_algorithms,
                "fsdp_output_dtype": args.fsdp_output_dtype,
                "fused_adamw": args.fused_adamw,
                "length_bucketing": args.length_bucketing,
                "pinned_staging": args.pinned_staging,
            },
            "preset": args.preset,
            "shard_manifest_sha256": args.manifest_sha256,
            "training_protocol_sha256": training_protocol_sha256,
            "target_residue_tokens": args.target_tokens,
            "world_size": world_size,
        }
        if any(state.get(key) != value for key, value in expected.items()):
            raise RuntimeError("Resume provenance differs from this launch")
        cursor = int(state["sample_cursor"])
        consumed = int(state["consumed_residue_tokens"])
        start_update = int(state["next_update"])
        _load_dcp(model, optimizer, args.resume_root / "checkpoint")
        resume_rng_manifest_sha256 = _restore_rng_checkpoint(
            args.resume_root / "checkpoint",
            rank=rank,
            world_size=world_size,
            device=device,
        )

    trace: list[dict[str, float | int]] = []
    membership_digest = hashlib.sha256()
    data_preparation_seconds = 0.0
    for update in range(start_update, start_update + args.steps):
        preparation_started = time.perf_counter()
        plan = build_update_plan(
            residue_lengths,
            sample_cursor=cursor,
            target_residue_tokens=args.target_tokens,
            batch_quantum_sequences=args.batch_quantum,
            seed=protocol["random_seed"],
        )
        indices_for_compute = _ordered_indices(
            plan.indices,
            residue_lengths,
            args.length_bucketing,
        )
        _membership_update(
            membership_digest,
            update=update + 1,
            sample_cursor=cursor,
            indices=plan.indices,
        )
        global_micro_batch = world_size * args.micro_batch
        if len(indices_for_compute) % global_micro_batch:
            raise RuntimeError("Batch quantum must divide the global micro batch")
        prepared: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        local_target_count = 0
        for start in range(0, len(indices_for_compute), global_micro_batch):
            local_start = start + rank * args.micro_batch
            indices = indices_for_compute[local_start : local_start + args.micro_batch]
            tokens = np.asarray(shard.tokens[indices], dtype=np.int64)
            masked = apply_uniform_mlm(
                tokens,
                np.asarray(shard.sample_keys[indices]),
                update=update,
                seed=protocol["random_seed"],
                mask_fraction=protocol["masking"]["fraction"],
            )
            maximum_length = int(np.asarray(shard.lengths[indices]).max())
            width = (
                _bucket_width(maximum_length, boundaries)
                if args.bucketed_padding or args.length_bucketing
                else maximum_length
            )
            prepared.append(
                (
                    _stage_tensor(masked.input_ids[:, :width], args.pinned_staging),
                    _stage_tensor(masked.labels[:, :width], args.pinned_staging),
                    _stage_tensor(
                        np.asarray(shard.chain_types[indices], dtype=np.int64),
                        args.pinned_staging,
                    ),
                )
            )
            local_target_count += masked.selected_target_count
        target_tensor = torch.tensor(local_target_count, dtype=torch.long, device=device)
        dist.all_reduce(target_tensor, op=dist.ReduceOp.SUM)
        target_count = int(target_tensor)
        if target_count <= 0:
            raise RuntimeError("Update has no selected MLM targets")
        data_preparation_seconds += time.perf_counter() - preparation_started

        lr = learning_rate_at_token(
            consumed,
            peak_learning_rate=protocol["presets"][args.preset]["learning_rate"],
            warmup_residue_tokens=protocol["schedule"]["warmup_residue_tokens"],
            total_residue_tokens=protocol["schedule"]["total_residue_tokens"],
            minimum_learning_rate_ratio=protocol["schedule"][
                "minimum_learning_rate_ratio"
            ],
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        started = time.perf_counter()
        loss_sum_value = 0.0
        for micro_index, (input_cpu, labels_cpu, chain_cpu) in enumerate(prepared):
            synchronize = micro_index + 1 == len(prepared)
            synchronization_context = (
                model.no_sync()
                if args.backend == "ddp" and not synchronize
                else contextlib.nullcontext()
            )
            if args.backend == "fsdp2":
                model.set_requires_gradient_sync(synchronize)
            input_ids = input_cpu.to(
                device,
                non_blocking=args.pinned_staging,
            )
            labels = labels_cpu.to(device, non_blocking=args.pinned_staging)
            chain = chain_cpu.to(device, non_blocking=args.pinned_staging)
            attention = input_ids.ne(PAD_ID)
            type_ids = chain[:, None].expand_as(input_ids).masked_fill(~attention, 0)
            with synchronization_context:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = model(input_ids, type_ids, attention, labels)
                    loss = output.loss_sum * world_size / target_count
                loss.backward()
            loss_sum_value += float(output.loss_sum.detach())
        loss_tensor = torch.tensor(loss_sum_value, dtype=torch.float64, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        loss_sum_value = float(loss_tensor)
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                protocol["gradient_clip_norm"],
            )
        )
        optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        if not math.isfinite(loss_sum_value) or not math.isfinite(gradient_norm):
            raise RuntimeError("Training trace contains a non-finite value")
        consumed += plan.residue_token_count
        cursor = plan.next_sample_cursor
        if rank == 0:
            trace.append(
                {
                    "consumed_residue_tokens": consumed,
                    "gradient_norm": gradient_norm,
                    "learning_rate": lr,
                    "loss": loss_sum_value / target_count,
                    "micro_steps": len(prepared),
                    "residue_tokens": plan.residue_token_count,
                    "seconds": elapsed,
                    "selected_targets": target_count,
                    "sequence_count": len(plan.indices),
                    "update": update + 1,
                }
            )

    rng_state = _rng_state_payload(rank, device)
    parameter_moments = _parameter_moments(model, args.backend, rank)
    module = model.module if hasattr(model, "module") else model
    final_model_sha256 = _distributed_tree_digest(
        module.state_dict(), args.backend, rank
    )
    final_optimizer_sha256 = _distributed_tree_digest(
        optimizer.state_dict(), args.backend, rank
    )
    optimization = {
        "bucketed_padding": args.bucketed_padding,
        "compile_blocks": args.compile_blocks,
        "disable_compiled_cudagraphs": args.disable_compiled_cudagraphs,
        "compile_prewarm_buckets": args.compile_prewarm_buckets,
        "ddp_bucket_cap_mb": args.ddp_bucket_cap_mb,
        "deterministic_algorithms": args.deterministic_algorithms,
        "fsdp_output_dtype": args.fsdp_output_dtype,
        "fused_adamw": args.fused_adamw,
        "length_bucketing": args.length_bucketing,
        "pinned_staging": args.pinned_staging,
    }
    checkpoint_format = None
    checkpoint_rng_manifest_sha256 = None
    if not args.skip_checkpoint:
        checkpoint_format = "torch_distributed_checkpoint"
        checkpoint_root = args.output / "checkpoint"
        _save_dcp(model, optimizer, checkpoint_root)
        checkpoint_rng_manifest_sha256 = _save_rng_checkpoint(
            checkpoint_root,
            rng_state,
            rank=rank,
            world_size=world_size,
        )
    trainer_state = {
        "backend": args.backend,
        "batch_quantum": args.batch_quantum,
        "checkpoint_rng_manifest_sha256": checkpoint_rng_manifest_sha256,
        "consumed_residue_tokens": consumed,
        "efficiency_protocol_sha256": efficiency_protocol_sha256,
        "length_bucket_boundaries": list(boundaries),
        "micro_batch_per_gpu": args.micro_batch,
        "model_family_sha256": model_family_sha256,
        "next_update": start_update + args.steps,
        "optimization": optimization,
        "preset": args.preset,
        "sample_cursor": cursor,
        "shard_manifest_sha256": args.manifest_sha256,
        "training_protocol_sha256": training_protocol_sha256,
        "target_residue_tokens": args.target_tokens,
        "world_size": world_size,
    }
    if rank == 0:
        _publish(args.output / "trainer_state.json", trainer_state)
        _publish(
            args.output / "receipt.json",
            {
                "allocation": {
                    "hostname": socket.gethostname(),
                    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                    "slurm_job_nodelist": os.environ.get("SLURM_JOB_NODELIST"),
                },
                "authorization": {
                    "production_pretraining": False,
                    "scientific_claim": False,
                },
                "backend": args.backend,
                "batch_quantum": args.batch_quantum,
                "checkpoint_format": checkpoint_format,
                "checkpoint_rng_manifest_sha256": (
                    checkpoint_rng_manifest_sha256
                ),
                "claim_boundary": "Repeated-shard systems candidate; no production-corpus, biological, or state-of-the-art claim",
                "compile_prewarm_trace": compile_prewarm_trace,
                "data_preparation_seconds": data_preparation_seconds,
                "device": torch.cuda.get_device_name(device),
                "deterministic_algorithms": (
                    torch.are_deterministic_algorithms_enabled()
                ),
                "efficiency_protocol_sha256": efficiency_protocol_sha256,
                "final_model_sha256": final_model_sha256,
                "final_optimizer_sha256": final_optimizer_sha256,
                "format_version": "antibody_pretraining_optimized_pilot_v1",
                "launch_argv": sys.argv[1:],
                "length_bucket_boundaries": list(boundaries),
                "membership_sha256": membership_digest.hexdigest(),
                "micro_batch_per_gpu": args.micro_batch,
                "model_family_sha256": model_family_sha256,
                "optimization": optimization,
                "parameter_moments": parameter_moments,
                "peak_allocated_memory_mib": torch.cuda.max_memory_allocated(device)
                / (1024 * 1024),
                "peak_reserved_memory_mib": torch.cuda.max_memory_reserved(device)
                / (1024 * 1024),
                "preset": args.preset,
                "resume_used": args.resume_root is not None,
                "resume_rng_manifest_sha256": resume_rng_manifest_sha256,
                "runtime": {
                    "cuda": torch.version.cuda,
                    "cudnn": torch.backends.cudnn.version(),
                    "torch": torch.__version__,
                },
                "shard_manifest_sha256": args.manifest_sha256,
                "target_residue_tokens": args.target_tokens,
                "telemetry_interval_seconds": telemetry_interval_seconds,
                "test_access": {
                    "downstream_labels_used": False,
                    "sabdab2_heldout_accessed": False,
                },
                "trace": trace,
                "training_protocol_sha256": training_protocol_sha256,
                "world_size": world_size,
            },
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
