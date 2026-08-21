#!/usr/bin/env python3
"""Run a receipt-bound optimized antibody MLM training candidate."""

from __future__ import annotations

import argparse
import contextlib
import datetime
import hashlib
import inspect
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
import torch.nn.functional as F

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
    _parameter_moments,
    _publish,
    _sha256,
    _tree_digest,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--model-family", type=Path, required=True)
    parser.add_argument("--training-protocol", type=Path, required=True)
    parser.add_argument("--efficiency-protocol", type=Path, required=True)
    parser.add_argument("--ddp-communication-protocol", type=Path)
    parser.add_argument("--parity-protocol", type=Path)
    parser.add_argument("--parity-split-root", type=Path)
    parser.add_argument("--run-seed", type=int)
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
    parser.add_argument("--audit-gradient-rank-parity", action="store_true")
    parser.add_argument("--disable-ddp-gradient-as-bucket-view", action="store_true")
    parser.add_argument("--ddp-batched-grad-copy", action="store_true")
    parser.add_argument(
        "--ddp-communication-dtype",
        choices=("float32", "bfloat16"),
        default="float32",
    )
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
    if args.backend != "ddp" and (
        args.audit_gradient_rank_parity
        or args.ddp_batched_grad_copy
        or args.ddp_communication_dtype != "float32"
        or args.disable_ddp_gradient_as_bucket_view
    ):
        raise ValueError("DDP diagnostics require the DDP backend")
    if (
        args.ddp_batched_grad_copy
        or args.ddp_communication_dtype != "float32"
    ) and args.ddp_communication_protocol is None:
        raise ValueError("DDP communication optimization requires its protocol")
    parity_arguments = (
        args.parity_protocol,
        args.parity_split_root,
        args.run_seed,
    )
    if any(value is not None for value in parity_arguments) and not all(
        value is not None for value in parity_arguments
    ):
        raise ValueError("Parity protocol, split, and run seed are indivisible")
    if args.run_seed is not None and args.run_seed < 0:
        raise ValueError("Run seed must be non-negative")
    if args.parity_protocol is not None and (
        args.resume_root is not None or not args.skip_checkpoint
    ):
        raise ValueError(
            "Parity gate runs prohibit resume and require skipped checkpoints"
        )
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


def _load_ddp_communication_protocol(
    path: Path | None,
    *,
    efficiency_protocol_sha256: str,
) -> tuple[dict[str, object] | None, str | None]:
    if path is None:
        return None, None
    payload = path.read_bytes()
    try:
        protocol = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("DDP communication protocol must be ASCII JSON") from error
    if (
        protocol.get("format_version")
        != "antibody_ddp_communication_optimization_v1"
        or protocol.get("parent_efficiency_protocol_sha256")
        != efficiency_protocol_sha256
        or protocol.get("test_access")
        != {
            "downstream_labels_used": False,
            "sabdab2_heldout_accessed": False,
        }
    ):
        raise ValueError("Unexpected DDP communication protocol")
    return protocol, hashlib.sha256(payload).hexdigest()


def _canonical_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _load_parity_execution(
    protocol_path: Path,
    split_root: Path,
    *,
    shard: object,
    manifest_sha256: str,
    efficiency_protocol_sha256: str,
    run_seed: int,
    steps: int,
    backend: str,
    world_size: int,
) -> tuple[
    dict[str, object],
    str,
    dict[str, object],
    str,
    np.ndarray,
    np.ndarray,
]:
    protocol_payload = protocol_path.read_bytes()
    try:
        protocol = json.loads(protocol_payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("Parity protocol must be ASCII JSON") from error
    protocol_sha256 = hashlib.sha256(protocol_payload).hexdigest()
    if (
        protocol.get("format_version")
        != "antibody_ddp_scientific_parity_gate_v2"
        or protocol.get("parent", {}).get("efficiency_protocol_sha256")
        != efficiency_protocol_sha256
        or protocol.get("pilot_split", {}).get(
            "dense_manifest_payload_sha256"
        )
        != manifest_sha256
        or run_seed not in protocol.get("benchmark", {}).get(
            "training_seeds", []
        )
        or steps != protocol.get("benchmark", {}).get("updates_per_run")
        or backend != "ddp"
        or world_size != protocol.get("allocation", {}).get("gpu_count")
        or protocol.get("test_access")
        != {
            "downstream_labels_used": False,
            "sabdab2_heldout_accessed": False,
        }
    ):
        raise RuntimeError("Parity execution differs from the frozen protocol")
    receipt_path = split_root / "receipt.json"
    split_file_sha256 = _sha256(receipt_path)
    try:
        receipt = json.loads(receipt_path.read_text(encoding="ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("Parity split receipt must be ASCII JSON") from error
    unsigned = dict(receipt)
    recorded_receipt_sha256 = unsigned.pop("receipt_sha256", None)
    if (
        receipt.get("format_version")
        != "antibody_ddp_parity_split_receipt_v2"
        or receipt.get("dense_manifest_payload_sha256") != manifest_sha256
        or receipt.get("source", {}).get("protocol_sha256")
        != protocol_sha256
        or recorded_receipt_sha256
        != hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest()
    ):
        raise RuntimeError("Parity split receipt differs")
    arrays = {}
    for name, expected in receipt.get("files", {}).items():
        path = split_root / name
        if _sha256(path) != expected.get("sha256"):
            raise RuntimeError(f"Parity split file digest differs: {name}")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if (
            list(array.shape) != expected.get("shape")
            or str(array.dtype) != expected.get("dtype")
        ):
            raise RuntimeError(f"Parity split array schema differs: {name}")
        arrays[name] = array
    required = {
        "evaluation_indices.npy",
        "evaluation_sample_keys.npy",
        "training_indices.npy",
        "training_sample_keys.npy",
    }
    if set(arrays) != required:
        raise RuntimeError("Parity split array inventory differs")
    training = np.asarray(arrays["training_indices.npy"], dtype=np.int64)
    evaluation = np.asarray(arrays["evaluation_indices.npy"], dtype=np.int64)
    sample_keys = np.asarray(shard.sample_keys)
    split_policy = protocol["pilot_split"]
    if (
        len(sample_keys) != split_policy["expected_record_count"]
        or len(training) != split_policy["training_count"]
        or len(evaluation) != split_policy["evaluation_count"]
        or np.any(training < 0)
        or np.any(evaluation < 0)
        or np.any(training >= len(sample_keys))
        or np.any(evaluation >= len(sample_keys))
        or len(np.intersect1d(training, evaluation)) != 0
        or not np.array_equal(
            np.union1d(training, evaluation),
            np.arange(len(sample_keys), dtype=np.int64),
        )
        or not np.array_equal(
            sample_keys[training],
            arrays["training_sample_keys.npy"],
        )
        or not np.array_equal(
            sample_keys[evaluation],
            arrays["evaluation_sample_keys.npy"],
        )
    ):
        raise RuntimeError("Parity split membership differs")
    return (
        protocol,
        protocol_sha256,
        receipt,
        split_file_sha256,
        training,
        evaluation,
    )


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


def _evaluate_parity_panel(
    model: torch.nn.Module,
    *,
    shard: object,
    evaluation_indices: np.ndarray,
    mask_streams: list[dict[str, int]],
    boundaries: tuple[int, ...],
    device: torch.device,
    micro_batch: int,
    rank: int,
    world_size: int,
) -> dict[str, object] | None:
    """Score fixed MLM masks without changing training membership or state."""
    if not len(evaluation_indices) or not mask_streams:
        raise RuntimeError("Parity evaluation panel and masks must be nonempty")
    expected_keys = np.asarray(
        shard.sample_keys[evaluation_indices],
        dtype=np.uint64,
    )
    local_indices = np.asarray(
        evaluation_indices[rank::world_size],
        dtype=np.int64,
    )
    accumulators = {
        int(shard.sample_keys[index]): {
            "correct": 0,
            "loss_sum": 0.0,
            "selected_targets": 0,
        }
        for index in local_indices
    }
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for stream in mask_streams:
            for start in range(0, len(local_indices), micro_batch):
                valid_indices = local_indices[start : start + micro_batch]
                valid_count = len(valid_indices)
                if valid_count == 0:
                    continue
                if valid_count < micro_batch:
                    padded_indices = np.pad(
                        valid_indices,
                        (0, micro_batch - valid_count),
                        mode="edge",
                    )
                else:
                    padded_indices = valid_indices
                tokens = np.asarray(
                    shard.tokens[padded_indices],
                    dtype=np.int64,
                )
                sample_keys = np.asarray(
                    shard.sample_keys[padded_indices],
                    dtype=np.uint64,
                )
                masked = apply_uniform_mlm(
                    tokens,
                    sample_keys,
                    update=int(stream["update"]),
                    seed=int(stream["seed"]),
                    mask_fraction=0.15,
                )
                maximum_length = int(
                    np.asarray(shard.lengths[padded_indices]).max()
                )
                width = _bucket_width(maximum_length, boundaries)
                input_ids = torch.from_numpy(
                    np.ascontiguousarray(masked.input_ids[:, :width])
                ).to(device)
                labels = torch.from_numpy(
                    np.ascontiguousarray(masked.labels[:, :width])
                ).to(device)
                chain = torch.from_numpy(
                    np.asarray(
                        shard.chain_types[padded_indices],
                        dtype=np.int64,
                    )
                ).to(device)
                attention = input_ids.ne(PAD_ID)
                type_ids = chain[:, None].expand_as(input_ids).masked_fill(
                    ~attention,
                    0,
                )
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = model(input_ids, type_ids, attention)
                selected = labels.ne(-100)
                losses = F.cross_entropy(
                    output.logits.float().flatten(0, 1),
                    labels.flatten(),
                    ignore_index=-100,
                    reduction="none",
                ).view_as(labels)
                correct = (
                    output.logits.argmax(dim=-1).eq(labels) & selected
                ).sum(dim=1)
                loss_sums = losses.sum(dim=1)
                target_counts = selected.sum(dim=1)
                for offset in range(valid_count):
                    key = int(sample_keys[offset])
                    accumulator = accumulators[key]
                    accumulator["correct"] += int(correct[offset])
                    accumulator["loss_sum"] += float(loss_sums[offset])
                    accumulator["selected_targets"] += int(
                        target_counts[offset]
                    )
    if was_training:
        model.train()
    local_records = []
    for sample_key in sorted(accumulators):
        item = accumulators[sample_key]
        target_count = int(item["selected_targets"])
        if target_count <= 0:
            raise RuntimeError("Parity evaluation sample has no MLM targets")
        local_records.append(
            {
                "masked_accuracy": item["correct"] / target_count,
                "masked_cross_entropy": item["loss_sum"] / target_count,
                "sample_key": sample_key,
                **item,
            }
        )
    gathered: list[list[dict[str, object]] | None] = [None] * world_size
    dist.all_gather_object(gathered, local_records)
    if rank != 0:
        return None
    records = sorted(
        (
            record
            for rank_records in gathered
            if rank_records is not None
            for record in rank_records
        ),
        key=lambda record: int(record["sample_key"]),
    )
    observed_keys = np.asarray(
        [record["sample_key"] for record in records],
        dtype=np.uint64,
    )
    if (
        len(records) != len(evaluation_indices)
        or not np.array_equal(observed_keys, np.sort(expected_keys))
    ):
        raise RuntimeError("Parity evaluation sample accounting differs")
    total_loss = sum(float(record["loss_sum"]) for record in records)
    total_correct = sum(int(record["correct"]) for record in records)
    total_targets = sum(
        int(record["selected_targets"]) for record in records
    )
    return {
        "masked_accuracy": total_correct / total_targets,
        "masked_cross_entropy": total_loss / total_targets,
        "mask_stream_count": len(mask_streams),
        "sample_count": len(records),
        "samples": records,
        "selected_targets": total_targets,
    }


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
        ddp_arguments = {
            "device_ids": [local_rank],
            "bucket_cap_mb": args.ddp_bucket_cap_mb,
            "gradient_as_bucket_view": (
                not args.disable_ddp_gradient_as_bucket_view
            ),
            "static_graph": False,
        }
        ddp_parameters = inspect.signature(DistributedDataParallel).parameters
        if "batched_grad_copy" in ddp_parameters:
            ddp_arguments["batched_grad_copy"] = args.ddp_batched_grad_copy
        elif args.ddp_batched_grad_copy:
            raise RuntimeError(
                "This PyTorch DistributedDataParallel does not support "
                "batched_grad_copy"
            )
        wrapped = DistributedDataParallel(model, **ddp_arguments)
        if args.ddp_communication_dtype == "bfloat16":
            from torch.distributed.algorithms.ddp_comm_hooks import default_hooks

            wrapped.register_comm_hook(None, default_hooks.bf16_compress_hook)
        return wrapped
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


def _replicated_rank_digests(value: object, label: str) -> list[str]:
    """Prove that every DDP rank holds the same complete state tree."""
    local_sha256 = _tree_digest(value)
    gathered: list[str | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local_sha256)
    if any(value is None for value in gathered) or len(set(gathered)) != 1:
        detail = json.dumps(gathered, separators=(",", ":"))
        raise RuntimeError(
            f"Replicated {label} state differs across DDP ranks: {detail}"
        )
    return [str(value) for value in gathered]


def _assert_scalar_rank_parity(value: float, label: str) -> list[float]:
    """Fail when a deterministic DDP scalar differs exactly across ranks."""
    gathered: list[float | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, value)
    if any(item is None for item in gathered) or len(set(gathered)) != 1:
        detail = json.dumps(gathered, separators=(",", ":"))
        raise RuntimeError(f"Replicated {label} differs across DDP ranks: {detail}")
    return [float(item) for item in gathered]


def _save_authenticated_dcp(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_root: Path,
    *,
    backend: str,
    rank: int,
) -> tuple[str | None, str | None, list[str] | None, list[str] | None]:
    """Save DCP state and return canonical pre-serialization digests."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict

    model_state, optimizer_state = get_state_dict(model, optimizer)
    model_rank_sha256 = None
    optimizer_rank_sha256 = None
    if backend == "ddp":
        model_rank_sha256 = _replicated_rank_digests(model_state, "DCP model")
        optimizer_rank_sha256 = _replicated_rank_digests(
            optimizer_state, "DCP optimizer"
        )
        model_sha256 = model_rank_sha256[0]
        optimizer_sha256 = optimizer_rank_sha256[0]
    else:
        model_sha256 = _distributed_tree_digest(model_state, backend, rank)
        optimizer_sha256 = _distributed_tree_digest(optimizer_state, backend, rank)
    dcp.save(
        {"model": model_state, "optimizer": optimizer_state},
        checkpoint_id=str(checkpoint_root),
    )
    return (
        model_sha256,
        optimizer_sha256,
        model_rank_sha256,
        optimizer_rank_sha256,
    )


def _load_authenticated_dcp(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_root: Path,
    *,
    backend: str,
    rank: int,
) -> tuple[str | None, str | None, list[str] | None, list[str] | None]:
    """Load DCP state and hash the canonical payload before applying it."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

    model_state, optimizer_state = get_state_dict(model, optimizer)
    state = {"model": model_state, "optimizer": optimizer_state}
    dcp.load(
        state,
        checkpoint_id=str(checkpoint_root),
    )
    model_rank_sha256 = None
    optimizer_rank_sha256 = None
    if backend == "ddp":
        model_rank_sha256 = _replicated_rank_digests(
            state["model"], "loaded DCP model payload"
        )
        optimizer_rank_sha256 = _replicated_rank_digests(
            state["optimizer"], "loaded DCP optimizer payload"
        )
        model_sha256 = model_rank_sha256[0]
        optimizer_sha256 = optimizer_rank_sha256[0]
    else:
        model_sha256 = _distributed_tree_digest(state["model"], backend, rank)
        optimizer_sha256 = _distributed_tree_digest(
            state["optimizer"], backend, rank
        )
    set_state_dict(
        model,
        optimizer,
        model_state_dict=state["model"],
        optim_state_dict=state["optimizer"],
    )
    return (
        model_sha256,
        optimizer_sha256,
        model_rank_sha256,
        optimizer_rank_sha256,
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
    effective_seed = (
        args.run_seed if args.run_seed is not None else 20260728
    )
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
    random.seed(effective_seed)
    np.random.seed(effective_seed)
    torch.manual_seed(effective_seed)
    torch.cuda.manual_seed_all(effective_seed)
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
    (
        ddp_communication_protocol,
        ddp_communication_protocol_sha256,
    ) = _load_ddp_communication_protocol(
        args.ddp_communication_protocol,
        efficiency_protocol_sha256=efficiency_protocol_sha256,
    )
    if ddp_communication_protocol is not None and not any(
        candidate.get("batched_grad_copy") == args.ddp_batched_grad_copy
        and candidate.get("communication_dtype")
        == args.ddp_communication_dtype
        for candidate in ddp_communication_protocol["candidates"]
    ):
        raise RuntimeError("DDP communication candidate is not registered")
    if args.ddp_communication_dtype == "bfloat16":
        nccl_version = tuple(int(value) for value in torch.cuda.nccl.version())
        if nccl_version <= (2, 9, 6) or not torch.__version__.startswith("2.13."):
            raise RuntimeError("BF16 DDP communication runtime is unsupported")
    shard = load_dense_shard(
        args.shard,
        expected_manifest_sha256=args.manifest_sha256,
    )
    family = load_model_family(args.model_family)
    protocol = load_training_protocol(args.training_protocol)
    parity_protocol = None
    parity_protocol_sha256 = None
    parity_split_receipt = None
    parity_split_receipt_file_sha256 = None
    parity_training_indices = None
    parity_evaluation_indices = None
    if args.parity_protocol is not None:
        (
            parity_protocol,
            parity_protocol_sha256,
            parity_split_receipt,
            parity_split_receipt_file_sha256,
            parity_training_indices,
            parity_evaluation_indices,
        ) = _load_parity_execution(
            args.parity_protocol,
            args.parity_split_root,
            shard=shard,
            manifest_sha256=args.manifest_sha256,
            efficiency_protocol_sha256=efficiency_protocol_sha256,
            run_seed=effective_seed,
            steps=args.steps,
            backend=args.backend,
            world_size=world_size,
        )
    boundaries = _parse_boundaries(
        args.length_bucket_boundaries,
        family["maximum_sequence_tokens"],
    )
    all_residue_lengths = np.asarray(shard.lengths, dtype=np.int64) - 2
    residue_lengths = (
        all_residue_lengths[parity_training_indices]
        if parity_training_indices is not None
        else all_residue_lengths
    )
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
    initial_model_rank_sha256 = None
    if args.backend == "ddp":
        module = model.module if hasattr(model, "module") else model
        initial_model_rank_sha256 = _replicated_rank_digests(
            module.state_dict(), "initial model"
        )

    model_family_sha256 = _sha256(args.model_family)
    training_protocol_sha256 = _sha256(args.training_protocol)
    cursor = 0
    consumed = 0
    start_update = 0
    resume_loaded_model_sha256 = None
    resume_loaded_optimizer_sha256 = None
    resume_loaded_model_rank_sha256 = None
    resume_loaded_optimizer_rank_sha256 = None
    resume_payload_model_sha256 = None
    resume_payload_optimizer_sha256 = None
    resume_payload_model_rank_sha256 = None
    resume_payload_optimizer_rank_sha256 = None
    resume_rng_manifest_sha256 = None
    resume_rng_checkpoint_root = None
    if args.resume_root is not None:
        state = json.loads(
            (args.resume_root / "trainer_state.json").read_text(encoding="ascii")
        )
        expected = {
            "backend": args.backend,
            "batch_quantum": args.batch_quantum,
            "efficiency_protocol_sha256": efficiency_protocol_sha256,
            "ddp_communication_protocol_sha256": (
                ddp_communication_protocol_sha256
            ),
            "length_bucket_boundaries": list(boundaries),
            "micro_batch_per_gpu": args.micro_batch,
            "model_family_sha256": model_family_sha256,
            "optimization": {
                "audit_gradient_rank_parity": args.audit_gradient_rank_parity,
                "bucketed_padding": args.bucketed_padding,
                "compile_blocks": args.compile_blocks,
                "disable_compiled_cudagraphs": args.disable_compiled_cudagraphs,
                "compile_prewarm_buckets": args.compile_prewarm_buckets,
                "ddp_bucket_cap_mb": args.ddp_bucket_cap_mb,
                "ddp_batched_grad_copy": args.ddp_batched_grad_copy,
                "ddp_communication_dtype": args.ddp_communication_dtype,
                "disable_ddp_gradient_as_bucket_view": (
                    args.disable_ddp_gradient_as_bucket_view
                ),
                "ddp_static_graph": False,
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
        (
            resume_payload_model_sha256,
            resume_payload_optimizer_sha256,
            resume_payload_model_rank_sha256,
            resume_payload_optimizer_rank_sha256,
        ) = _load_authenticated_dcp(
            model,
            optimizer,
            args.resume_root / "checkpoint",
            backend=args.backend,
            rank=rank,
        )
        module = model.module if hasattr(model, "module") else model
        if args.backend == "ddp":
            resume_loaded_model_rank_sha256 = _replicated_rank_digests(
                module.state_dict(), "applied model"
            )
            resume_loaded_optimizer_rank_sha256 = _replicated_rank_digests(
                optimizer.state_dict(), "applied optimizer"
            )
            resume_loaded_model_sha256 = resume_loaded_model_rank_sha256[0]
            resume_loaded_optimizer_sha256 = (
                resume_loaded_optimizer_rank_sha256[0]
            )
        else:
            resume_loaded_model_sha256 = _distributed_tree_digest(
                module.state_dict(), args.backend, rank
            )
            resume_loaded_optimizer_sha256 = _distributed_tree_digest(
                optimizer.state_dict(), args.backend, rank
            )
        resume_rng_checkpoint_root = args.resume_root / "checkpoint"

    compile_prewarm_trace: list[dict[str, float | int]] = []
    if args.compile_prewarm_buckets:
        compile_prewarm_trace = _prewarm_compiled_buckets(
            model,
            boundaries=boundaries,
            device=device,
            micro_batch=args.micro_batch,
            world_size=world_size,
        )
    if resume_rng_checkpoint_root is not None:
        resume_rng_manifest_sha256 = _restore_rng_checkpoint(
            resume_rng_checkpoint_root,
            rank=rank,
            world_size=world_size,
            device=device,
        )

    pretraining_evaluation = None
    if parity_protocol is not None:
        dist.barrier()
        pretraining_evaluation = _evaluate_parity_panel(
            model,
            shard=shard,
            evaluation_indices=parity_evaluation_indices,
            mask_streams=parity_protocol["evaluation"]["mask_streams"],
            boundaries=boundaries,
            device=device,
            micro_batch=args.micro_batch,
            rank=rank,
            world_size=world_size,
        )
        dist.barrier()

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
            seed=effective_seed,
        )
        logical_indices_for_compute = _ordered_indices(
            plan.indices,
            residue_lengths,
            args.length_bucketing,
        )
        indices_for_compute = (
            parity_training_indices[logical_indices_for_compute]
            if parity_training_indices is not None
            else logical_indices_for_compute
        )
        membership_indices = (
            parity_training_indices[plan.indices]
            if parity_training_indices is not None
            else plan.indices
        )
        _membership_update(
            membership_digest,
            update=update + 1,
            sample_cursor=cursor,
            indices=membership_indices,
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
                seed=effective_seed,
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
        if args.backend == "ddp" and args.audit_gradient_rank_parity:
            _assert_scalar_rank_parity(
                gradient_norm,
                f"post-reduction gradient norm at update {update + 1}",
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

    posttraining_evaluation = None
    if parity_protocol is not None:
        dist.barrier()
        posttraining_evaluation = _evaluate_parity_panel(
            model,
            shard=shard,
            evaluation_indices=parity_evaluation_indices,
            mask_streams=parity_protocol["evaluation"]["mask_streams"],
            boundaries=boundaries,
            device=device,
            micro_batch=args.micro_batch,
            rank=rank,
            world_size=world_size,
        )
        dist.barrier()

    rng_state = _rng_state_payload(rank, device)
    parameter_moments = _parameter_moments(model, args.backend, rank)
    ddp_logging_data = None
    if args.backend == "ddp":
        raw_ddp_logging_data = model._get_ddp_logging_data()
        ddp_logging_data = {
            str(key): value
            for key, value in sorted(raw_ddp_logging_data.items())
            if isinstance(value, (bool, float, int, str))
        }
    module = model.module if hasattr(model, "module") else model
    final_model_rank_sha256 = None
    final_optimizer_rank_sha256 = None
    if args.backend == "ddp":
        final_model_rank_sha256 = _replicated_rank_digests(
            module.state_dict(), "final model"
        )
        final_optimizer_rank_sha256 = _replicated_rank_digests(
            optimizer.state_dict(), "final optimizer"
        )
        final_model_sha256 = final_model_rank_sha256[0]
        final_optimizer_sha256 = final_optimizer_rank_sha256[0]
    else:
        final_model_sha256 = _distributed_tree_digest(
            module.state_dict(), args.backend, rank
        )
        final_optimizer_sha256 = _distributed_tree_digest(
            optimizer.state_dict(), args.backend, rank
        )
    optimization = {
        "audit_gradient_rank_parity": args.audit_gradient_rank_parity,
        "bucketed_padding": args.bucketed_padding,
        "compile_blocks": args.compile_blocks,
        "disable_compiled_cudagraphs": args.disable_compiled_cudagraphs,
        "compile_prewarm_buckets": args.compile_prewarm_buckets,
        "ddp_bucket_cap_mb": args.ddp_bucket_cap_mb,
        "ddp_batched_grad_copy": args.ddp_batched_grad_copy,
        "ddp_communication_dtype": args.ddp_communication_dtype,
        "disable_ddp_gradient_as_bucket_view": (
            args.disable_ddp_gradient_as_bucket_view
        ),
        "ddp_static_graph": False,
        "deterministic_algorithms": args.deterministic_algorithms,
        "fsdp_output_dtype": args.fsdp_output_dtype,
        "fused_adamw": args.fused_adamw,
        "length_bucketing": args.length_bucketing,
        "pinned_staging": args.pinned_staging,
    }
    checkpoint_format = None
    checkpoint_payload_model_sha256 = None
    checkpoint_payload_optimizer_sha256 = None
    checkpoint_payload_model_rank_sha256 = None
    checkpoint_payload_optimizer_rank_sha256 = None
    checkpoint_rng_manifest_sha256 = None
    if not args.skip_checkpoint:
        checkpoint_format = "torch_distributed_checkpoint"
        checkpoint_root = args.output / "checkpoint"
        (
            checkpoint_payload_model_sha256,
            checkpoint_payload_optimizer_sha256,
            checkpoint_payload_model_rank_sha256,
            checkpoint_payload_optimizer_rank_sha256,
        ) = _save_authenticated_dcp(
            model,
            optimizer,
            checkpoint_root,
            backend=args.backend,
            rank=rank,
        )
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
        "ddp_communication_protocol_sha256": (
            ddp_communication_protocol_sha256
        ),
        "length_bucket_boundaries": list(boundaries),
        "micro_batch_per_gpu": args.micro_batch,
        "model_family_sha256": model_family_sha256,
        "next_update": start_update + args.steps,
        "optimization": optimization,
        "parity_protocol_sha256": parity_protocol_sha256,
        "parity_split_receipt_file_sha256": (
            parity_split_receipt_file_sha256
        ),
        "parity_split_receipt_sha256": (
            parity_split_receipt.get("receipt_sha256")
            if parity_split_receipt is not None
            else None
        ),
        "preset": args.preset,
        "run_seed": effective_seed,
        "sample_cursor": cursor,
        "shard_manifest_sha256": args.manifest_sha256,
        "training_protocol_sha256": training_protocol_sha256,
        "training_universe_count": len(residue_lengths),
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
                "checkpoint_payload_model_sha256": (
                    checkpoint_payload_model_sha256
                ),
                "checkpoint_payload_optimizer_sha256": (
                    checkpoint_payload_optimizer_sha256
                ),
                "checkpoint_payload_model_rank_sha256": (
                    checkpoint_payload_model_rank_sha256
                ),
                "checkpoint_payload_optimizer_rank_sha256": (
                    checkpoint_payload_optimizer_rank_sha256
                ),
                "checkpoint_rng_manifest_sha256": (
                    checkpoint_rng_manifest_sha256
                ),
                "claim_boundary": (
                    "Training-excluded dense-pilot MLM systems parity only; "
                    "no production-corpus, biological, scaling, or "
                    "state-of-the-art claim"
                    if parity_protocol is not None
                    else "Repeated-shard systems candidate; no "
                    "production-corpus, biological, or state-of-the-art claim"
                ),
                "compile_prewarm_trace": compile_prewarm_trace,
                "data_preparation_seconds": data_preparation_seconds,
                "device": torch.cuda.get_device_name(device),
                "deterministic_algorithms": (
                    torch.are_deterministic_algorithms_enabled()
                ),
                "ddp_logging_data": ddp_logging_data,
                "efficiency_protocol_sha256": efficiency_protocol_sha256,
                "ddp_communication_protocol_sha256": (
                    ddp_communication_protocol_sha256
                ),
                "final_model_sha256": final_model_sha256,
                "final_model_rank_sha256": final_model_rank_sha256,
                "final_optimizer_sha256": final_optimizer_sha256,
                "final_optimizer_rank_sha256": final_optimizer_rank_sha256,
                "format_version": "antibody_pretraining_optimized_pilot_v1",
                "initial_model_rank_sha256": initial_model_rank_sha256,
                "launch_argv": sys.argv[1:],
                "length_bucket_boundaries": list(boundaries),
                "membership_sha256": membership_digest.hexdigest(),
                "micro_batch_per_gpu": args.micro_batch,
                "model_family_sha256": model_family_sha256,
                "optimization": optimization,
                "parity_evaluation_count": (
                    len(parity_evaluation_indices)
                    if parity_evaluation_indices is not None
                    else None
                ),
                "parity_protocol_sha256": parity_protocol_sha256,
                "parity_split_receipt_file_sha256": (
                    parity_split_receipt_file_sha256
                ),
                "parity_split_receipt_sha256": (
                    parity_split_receipt.get("receipt_sha256")
                    if parity_split_receipt is not None
                    else None
                ),
                "parameter_moments": parameter_moments,
                "peak_allocated_memory_mib": torch.cuda.max_memory_allocated(device)
                / (1024 * 1024),
                "peak_reserved_memory_mib": torch.cuda.max_memory_reserved(device)
                / (1024 * 1024),
                "preset": args.preset,
                "pretraining_evaluation": pretraining_evaluation,
                "resume_used": args.resume_root is not None,
                "resume_loaded_model_sha256": resume_loaded_model_sha256,
                "resume_loaded_model_rank_sha256": (
                    resume_loaded_model_rank_sha256
                ),
                "resume_loaded_optimizer_sha256": (
                    resume_loaded_optimizer_sha256
                ),
                "resume_loaded_optimizer_rank_sha256": (
                    resume_loaded_optimizer_rank_sha256
                ),
                "resume_payload_model_sha256": resume_payload_model_sha256,
                "resume_payload_model_rank_sha256": (
                    resume_payload_model_rank_sha256
                ),
                "resume_payload_optimizer_sha256": (
                    resume_payload_optimizer_sha256
                ),
                "resume_payload_optimizer_rank_sha256": (
                    resume_payload_optimizer_rank_sha256
                ),
                "resume_rng_manifest_sha256": resume_rng_manifest_sha256,
                "resume_rng_restored_after_compile_prewarm": (
                    resume_rng_checkpoint_root is not None
                ),
                "runtime": {
                    "cuda": torch.version.cuda,
                    "cudnn": torch.backends.cudnn.version(),
                    "torch": torch.__version__,
                },
                "shard_manifest_sha256": args.manifest_sha256,
                "posttraining_evaluation": posttraining_evaluation,
                "run_seed": effective_seed,
                "target_residue_tokens": args.target_tokens,
                "telemetry_interval_seconds": telemetry_interval_seconds,
                "test_access": {
                    "downstream_labels_used": False,
                    "sabdab2_heldout_accessed": False,
                },
                "trace": trace,
                "training_protocol_sha256": training_protocol_sha256,
                "training_universe_count": len(residue_lengths),
                "world_size": world_size,
            },
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
