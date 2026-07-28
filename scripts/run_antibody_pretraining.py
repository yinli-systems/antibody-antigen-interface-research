#!/usr/bin/env python3
"""Run an audited antibody MLM engineering pilot on a dense OAS shard."""

from __future__ import annotations

import argparse
import contextlib
import datetime
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from antibody_pretraining.dense_shard import load_dense_shard
from antibody_pretraining.masking import apply_uniform_mlm
from antibody_pretraining.model import AntibodyMaskedLanguageModel, config_from_preset, load_model_family
from antibody_pretraining.tokenizer import PAD_ID
from antibody_pretraining.training_contract import build_update_plan, learning_rate_at_token, load_training_protocol


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _publish(path: Path, payload: object) -> None:
    data = (json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("ascii")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)


def _save_dcp(model: torch.nn.Module, optimizer: torch.optim.Optimizer, path: Path) -> None:
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict

    model_state, optimizer_state = get_state_dict(model, optimizer)
    dcp.save({"model": model_state, "optimizer": optimizer_state}, checkpoint_id=str(path))


def _load_dcp(model: torch.nn.Module, optimizer: torch.optim.Optimizer, path: Path) -> None:
    """Restore model and optimizer state through PyTorch DCP."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

    model_state, optimizer_state = get_state_dict(model, optimizer)
    state = {"model": model_state, "optimizer": optimizer_state}
    dcp.load(state, checkpoint_id=str(path))
    set_state_dict(
        model,
        optimizer,
        model_state_dict=state["model"],
        optim_state_dict=state["optimizer"],
    )


def _tree_digest(value: object) -> str:
    """Hash a nested tensor state without relying on checkpoint byte layout."""
    digest = hashlib.sha256()

    def visit(item: object) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            header = json.dumps(
                {"dtype": str(tensor.dtype), "shape": list(tensor.shape)},
                sort_keys=True,
                separators=(",", ":"),
            ).encode("ascii")
            digest.update(b"tensor\0" + header + b"\0")
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict\0")
            for key in sorted(item, key=lambda current: str(current)):
                visit(str(key))
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(b"sequence\0")
            for child in item:
                visit(child)
        else:
            digest.update(
                json.dumps(item, sort_keys=True, separators=(",", ":")).encode(
                    "ascii"
                )
                + b"\0"
            )

    visit(value)
    return digest.hexdigest()


def _distributed_tree_digest(value: object, backend: str, rank: int) -> str | None:
    """Hash full tensor values while all FSDP ranks join materialization."""
    digest = hashlib.sha256() if rank == 0 else None

    def update(payload: bytes) -> None:
        if digest is not None:
            digest.update(payload)

    def visit(item: object) -> None:
        if isinstance(item, torch.Tensor):
            tensor = (
                item.full_tensor()
                if backend == "fsdp2" and hasattr(item, "full_tensor")
                else item
            )
            if rank == 0:
                tensor = tensor.detach().cpu().contiguous()
                header = json.dumps(
                    {"dtype": str(tensor.dtype), "shape": list(tensor.shape)},
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("ascii")
                update(b"tensor\0" + header + b"\0")
                update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            update(b"dict\0")
            for key in sorted(item, key=lambda current: str(current)):
                visit(str(key))
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            update(b"sequence\0")
            for child in item:
                visit(child)
        else:
            update(
                json.dumps(item, sort_keys=True, separators=(",", ":")).encode(
                    "ascii"
                )
                + b"\0"
            )

    visit(value)
    return digest.hexdigest() if digest is not None else None


def _parameter_moments(model: torch.nn.Module, backend: str, rank: int) -> dict[str, float | int] | None:
    """Compute full-parameter moments for cross-backend numerical audits."""
    module = model.module if hasattr(model, "module") else model
    count = 0
    total = 0.0
    square_total = 0.0
    absolute_maximum = 0.0
    for parameter in module.parameters():
        value = parameter
        if backend == "fsdp2":
            value = parameter.full_tensor()
        if backend == "zero3":
            import deepspeed

            context = deepspeed.zero.GatheredParameters([parameter], modifier_rank=None)
        else:
            context = contextlib.nullcontext()
        with context:
            if rank == 0:
                value64 = value.detach().to(device="cpu", dtype=torch.float64)
                count += value64.numel()
                total += float(value64.sum())
                square_total += float(value64.square().sum())
                absolute_maximum = max(absolute_maximum, float(value64.abs().max()))
    if rank != 0:
        return None
    return {
        "absolute_maximum": absolute_maximum,
        "count": count,
        "square_sum": square_total,
        "sum": total,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--model-family", type=Path, required=True)
    parser.add_argument("--training-protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preset", default="abf_35m")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--micro-batch", type=int, default=8)
    parser.add_argument("--target-tokens", type=int, default=4096)
    parser.add_argument("--batch-quantum", type=int, default=32)
    parser.add_argument("--resume-root", type=Path)
    parser.add_argument(
        "--backend",
        choices=("single", "ddp", "fsdp2", "zero2", "zero3"),
        default="single",
    )
    parser.add_argument("--skip-checkpoint", action="store_true")
    args = parser.parse_args()
    if args.steps <= 0 or args.micro_batch <= 0:
        raise ValueError("Output must be create-only and pilot sizes must be positive")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if args.backend == "single" and world_size != 1:
        raise RuntimeError("The single backend requires one process")
    if args.backend != "single" and world_size <= 1:
        raise RuntimeError("Distributed backends require multiple processes")
    if not torch.cuda.is_available() or local_rank >= torch.cuda.device_count():
        raise RuntimeError("CUDA rank identity is invalid")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if rank == 0:
        if args.output.exists():
            raise ValueError("Output must be create-only")
        args.output.mkdir(parents=True)
    if world_size > 1:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=datetime.timedelta(minutes=5),
            device_id=device,
        )
        dist.barrier()
    torch.manual_seed(20260728)
    torch.cuda.manual_seed_all(20260728)
    torch.backends.cuda.matmul.allow_tf32 = False
    shard = load_dense_shard(args.shard, expected_manifest_sha256=args.manifest_sha256)
    family = load_model_family(args.model_family)
    protocol = load_training_protocol(args.training_protocol)
    residue_lengths = np.asarray(shard.lengths, dtype=np.int64) - 2
    model = AntibodyMaskedLanguageModel(config_from_preset(family, args.preset)).to(device)
    if args.backend == "ddp":
        model = DistributedDataParallel(model, device_ids=[local_rank])
    elif args.backend == "fsdp2":
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

        mesh = init_device_mesh("cuda", (world_size,))
        policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            output_dtype=torch.float32,
        )
        for block in model.blocks:
            fully_shard(block, mesh=mesh, mp_policy=policy)
        fully_shard(model, mesh=mesh, mp_policy=policy)
    decay, no_decay = [], []
    for parameter in model.parameters():
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": protocol["optimizer"]["weight_decay"]}, {"params": no_decay, "weight_decay": 0.0}],
        lr=protocol["presets"][args.preset]["learning_rate"],
        betas=tuple(protocol["optimizer"]["betas"]),
        eps=protocol["optimizer"]["epsilon"],
    )
    deepspeed_gradient_accumulation_steps = None
    if args.backend in {"zero2", "zero3"}:
        import deepspeed

        first_plan = build_update_plan(
            residue_lengths,
            sample_cursor=0,
            target_residue_tokens=args.target_tokens,
            batch_quantum_sequences=args.batch_quantum,
            seed=protocol["random_seed"],
        )
        global_micro_batch = world_size * args.micro_batch
        if len(first_plan.indices) % global_micro_batch:
            raise RuntimeError("DeepSpeed plan does not divide into global micro batches")
        deepspeed_gradient_accumulation_steps = len(first_plan.indices) // global_micro_batch
        stage = 2 if args.backend == "zero2" else 3
        zero_config = {
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_scatter": True,
            "stage": stage,
        }
        if stage == 3:
            zero_config.update(
                {
                    "stage3_gather_16bit_weights_on_model_save": False,
                    "stage3_param_persistence_threshold": 100_000,
                    "stage3_prefetch_bucket_size": 50_000_000,
                }
            )
        engine_config = {
            "gradient_accumulation_steps": deepspeed_gradient_accumulation_steps,
            "gradient_clipping": protocol["gradient_clip_norm"],
            "steps_per_print": 1_000_000_000,
            "train_batch_size": len(first_plan.indices),
            "train_micro_batch_size_per_gpu": args.micro_batch,
            "wall_clock_breakdown": False,
            "zero_optimization": zero_config,
        }
        model, optimizer, _, _ = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            config=engine_config,
        )
    cursor = 0
    consumed = 0
    start_update = 0
    model_family_sha256 = _sha256(args.model_family)
    training_protocol_sha256 = _sha256(args.training_protocol)
    if args.resume_root is not None:
        state = json.loads(
            (args.resume_root / "trainer_state.json").read_text(encoding="ascii")
        )
        expected = {
            "model_family_sha256": model_family_sha256,
            "backend": args.backend,
            "preset": args.preset,
            "shard_manifest_sha256": args.manifest_sha256,
            "training_protocol_sha256": training_protocol_sha256,
            "world_size": world_size,
        }
        if any(state.get(key) != value for key, value in expected.items()):
            raise RuntimeError("Resume provenance differs from this launch")
        cursor = int(state["sample_cursor"])
        consumed = int(state["consumed_residue_tokens"])
        start_update = int(state["next_update"])
        _load_dcp(model, optimizer, args.resume_root / "checkpoint")
    trace = []
    for update in range(start_update, start_update + args.steps):
        plan = build_update_plan(residue_lengths, sample_cursor=cursor, target_residue_tokens=args.target_tokens, batch_quantum_sequences=args.batch_quantum, seed=protocol["random_seed"])
        global_micro_batch = world_size * args.micro_batch
        if len(plan.indices) % global_micro_batch:
            raise RuntimeError("Pilot batch quantum must divide the global micro batch")
        prepared = []
        target_count = 0
        for start in range(0, len(plan.indices), global_micro_batch):
            local_start = start + rank * args.micro_batch
            indices = plan.indices[local_start : local_start + args.micro_batch]
            tokens = np.asarray(shard.tokens[indices], dtype=np.int64)
            masked = apply_uniform_mlm(tokens, np.asarray(shard.sample_keys[indices]), update=update, seed=protocol["random_seed"], mask_fraction=protocol["masking"]["fraction"])
            prepared.append((indices, masked))
            target_count += masked.selected_target_count
        if world_size > 1:
            target_tensor = torch.tensor(target_count, dtype=torch.long, device=device)
            dist.all_reduce(target_tensor, op=dist.ReduceOp.SUM)
            target_count = int(target_tensor)
        if target_count <= 0:
            raise RuntimeError("Update has no selected MLM targets")
        lr = learning_rate_at_token(consumed, peak_learning_rate=protocol["presets"][args.preset]["learning_rate"], warmup_residue_tokens=protocol["schedule"]["warmup_residue_tokens"], total_residue_tokens=protocol["schedule"]["total_residue_tokens"], minimum_learning_rate_ratio=protocol["schedule"]["minimum_learning_rate_ratio"])
        for group in optimizer.param_groups:
            group["lr"] = lr
        if args.backend not in {"zero2", "zero3"}:
            optimizer.zero_grad(set_to_none=True)
        started = time.perf_counter()
        loss_sum_value = 0.0
        for micro_index, (indices, masked) in enumerate(prepared):
            synchronize = micro_index + 1 == len(prepared)
            if args.backend == "ddp" and not synchronize:
                synchronization_context = model.no_sync()
            else:
                synchronization_context = contextlib.nullcontext()
            if args.backend == "fsdp2":
                model.set_requires_gradient_sync(synchronize)
            width = int(np.asarray(shard.lengths[indices]).max())
            input_ids = torch.from_numpy(masked.input_ids[:, :width]).to(device)
            labels = torch.from_numpy(masked.labels[:, :width]).to(device)
            attention = input_ids.ne(PAD_ID)
            chain = torch.from_numpy(np.asarray(shard.chain_types[indices], dtype=np.int64)).to(device)
            type_ids = chain[:, None].expand_as(input_ids).masked_fill(~attention, 0)
            with synchronization_context:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = model(input_ids, type_ids, attention, labels)
                    loss_multiplier = world_size
                    if args.backend in {"zero2", "zero3"}:
                        if len(prepared) != deepspeed_gradient_accumulation_steps:
                            raise RuntimeError("DeepSpeed accumulation window changed")
                        loss_multiplier *= deepspeed_gradient_accumulation_steps
                    loss = output.loss_sum * loss_multiplier / target_count
                if args.backend in {"zero2", "zero3"}:
                    model.backward(loss)
                    model.step()
                else:
                    loss.backward()
            loss_sum_value += float(output.loss_sum.detach())
        if world_size > 1:
            loss_tensor = torch.tensor(loss_sum_value, dtype=torch.float64, device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            loss_sum_value = float(loss_tensor)
        if args.backend in {"zero2", "zero3"}:
            gradient_norm = float(model.get_global_grad_norm())
        else:
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), protocol["gradient_clip_norm"]))
            optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        consumed += plan.residue_token_count
        cursor = plan.next_sample_cursor
        if rank == 0:
            trace.append({"consumed_residue_tokens": consumed, "gradient_norm": gradient_norm, "learning_rate": lr, "loss": loss_sum_value / target_count, "residue_tokens": plan.residue_token_count, "seconds": elapsed, "selected_targets": target_count, "update": update + 1})
    parameter_moments = _parameter_moments(model, args.backend, rank)
    module = model.module if hasattr(model, "module") else model
    if args.backend in {"zero2", "zero3"}:
        final_model_sha256 = None
        final_optimizer_sha256 = None
    else:
        final_model_sha256 = _distributed_tree_digest(
            module.state_dict(), args.backend, rank
        )
        final_optimizer_sha256 = _distributed_tree_digest(
            optimizer.state_dict(), args.backend, rank
        )
    if not args.skip_checkpoint:
        _save_dcp(model, optimizer, args.output / "checkpoint")
    trainer_state = {"backend": args.backend, "consumed_residue_tokens": consumed, "model_family_sha256": model_family_sha256, "next_update": start_update + args.steps, "preset": args.preset, "sample_cursor": cursor, "shard_manifest_sha256": args.manifest_sha256, "training_protocol_sha256": training_protocol_sha256, "world_size": world_size}
    if rank == 0:
        _publish(args.output / "trainer_state.json", trainer_state)
        receipt = {"backend": args.backend, "checkpoint_format": None if args.skip_checkpoint else "torch_distributed_checkpoint", "claim_boundary": "engineering pilot only; no validation or downstream metric", "device": torch.cuda.get_device_name(local_rank), "final_model_sha256": final_model_sha256, "final_optimizer_sha256": final_optimizer_sha256, "format_version": "antibody_pretraining_pilot_v4", "model_family_sha256": model_family_sha256, "parameter_moments": parameter_moments, "preset": args.preset, "resume_used": args.resume_root is not None, "shard_manifest_sha256": args.manifest_sha256, "start_update": start_update, "test_access": {"downstream_labels_used": False, "sabdab2_heldout_accessed": False}, "trace": trace, "training_protocol_sha256": training_protocol_sha256, "world_size": world_size}
        _publish(args.output / "receipt.json", receipt)
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
