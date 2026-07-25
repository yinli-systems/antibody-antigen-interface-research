#!/usr/bin/env python3
"""Run a data-free D0/R2 distributed correctness and recovery smoke test."""

from __future__ import annotations

import argparse
import contextlib
import copy
import datetime
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from distributed_launch_contract import (
    LaunchContractError,
    publish_create_only_json,
    require_sha256,
    source_tree_digest,
    validate_distributed_identity,
    validate_training_mode,
    verify_launch_manifest,
)
from interaction_first.frontier_batching import (
    FrontierBatch,
    FrontierBatchCollator,
    FrontierExample,
)
from interaction_first.frontier_checkpoint import (
    capture_rng_state,
    create_checkpoint_generation,
    load_checkpoint_generation,
)
from interaction_first.frontier_distributed import (
    exact_distributed_mean,
    symmetric_multi_positive_info_nce,
    symmetric_score_matrix_info_nce,
)
from interaction_first.frontier_model import (
    ProtocolD0Retriever,
    ProtocolReranker,
    balanced_contact_bce,
)
from interaction_first.runtime import set_reproducible_seed


def _canonical_json(value: Any) -> bytes:
    """Serialize finite JSON metadata with stable ordering."""
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _source_digest(repository_root: Path) -> str:
    """Bind the smoke receipt to every executed frontier source file."""
    return source_tree_digest(
        repository_root,
        (
            "scripts/distributed_launch_contract.py",
            "scripts/frontier_ddp_smoke.py",
            "scripts/frontier_ddp_staged_smoke.py",
            "scripts/generate_runtime_manifest.py",
            "slurm/frontier_ddp_smoke.sbatch",
            "slurm/frontier_r2_smoke.sbatch",
            "src/interaction_first/__init__.py",
            "src/interaction_first/asymmetric_retrieval.py",
            "src/interaction_first/contacts.py",
            "src/interaction_first/frontier_batching.py",
            "src/interaction_first/frontier_checkpoint.py",
            "src/interaction_first/frontier_distributed.py",
            "src/interaction_first/frontier_model.py",
            "src/interaction_first/partners.py",
            "src/interaction_first/runtime.py",
            "src/interaction_first/splits.py",
        ),
    )


def _state_dict_digest(module: torch.nn.Module) -> str:
    """Hash parameter names, metadata, and exact CPU tensor bytes."""
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        metadata = {
            "name": name,
            "dtype": str(value.dtype),
            "shape": list(value.shape),
        }
        header = _canonical_json(metadata)
        payload = value.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    """Publish one create-only JSON artifact through an adjacent temporary file."""
    publish_create_only_json(path, value)


def _rank_log(rank: int, stage: str, **fields: Any) -> None:
    """Emit one flush-safe rank event for diagnosing collective stalls."""
    print(
        json.dumps(
            {"rank": rank, "stage": stage, **fields},
            sort_keys=True,
            allow_nan=False,
        ),
        flush=True,
    )


def _distributed_context(
    *,
    expected_world_size: int | None = None,
    minimum_world_size: int = 1,
) -> tuple[int, int, int, torch.device]:
    """Initialize one NCCL or Gloo process group from torchrun variables."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        visible_device_count = torch.cuda.device_count()
        validate_distributed_identity(
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            visible_device_count=visible_device_count,
            minimum_world_size=minimum_world_size,
            expected_world_size=expected_world_size,
        )
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        if local_rank != 0:
            raise LaunchContractError("CPU diagnostics require LOCAL_RANK=0")
        if expected_world_size is not None and world_size != expected_world_size:
            raise LaunchContractError(
                "WORLD_SIZE differs from the externally declared launch size"
            )
        if not 0 <= rank < world_size or world_size < minimum_world_size:
            raise LaunchContractError("Invalid CPU distributed rank identity")
        torch.set_num_threads(1)
        device = torch.device("cpu")
    _rank_log(
        rank,
        "device_selected",
        local_rank=local_rank,
        device=str(device),
        world_size=world_size,
    )
    if world_size > 1:
        backend = "nccl" if device.type == "cuda" else "gloo"
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            timeout=datetime.timedelta(minutes=5),
        )
        _rank_log(rank, "process_group_initialized", backend=backend)
    return world_size, rank, local_rank, device


def _all_positive_graph(total_size: int) -> tuple[tuple[str, str], ...]:
    """Create a deterministic multi-positive graph for collective testing."""
    edges = {
        (f"ab-{index:05d}", f"ag-{index:05d}") for index in range(total_size)
    }
    if total_size > 1:
        edges.update(
            (f"ab-{index:05d}", f"ag-{(index + 1) % total_size:05d}")
            for index in range(0, total_size, 2)
        )
    return tuple(sorted(edges))


def _graph_masks(
    total_size: int,
    positive_graph: tuple[tuple[str, str], ...],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build identical global positive and ambiguity masks on every rank."""
    graph = frozenset(positive_graph)
    positive = torch.tensor(
        [
            [
                (f"ab-{antibody_index:05d}", f"ag-{antigen_index:05d}") in graph
                for antigen_index in range(total_size)
            ]
            for antibody_index in range(total_size)
        ],
        dtype=torch.bool,
        device=device,
    )
    excluded = torch.zeros_like(positive)
    if total_size > 3:
        for query in range(total_size):
            candidate = (query + 2) % total_size
            if not bool(positive[query, candidate]):
                excluded[query, candidate] = True
    return positive, excluded


def _synthetic_batch(
    *,
    rank: int,
    batch_size: int,
    seed: int,
    antibody_dim: int,
    antigen_dim: int,
    positive_graph: tuple[tuple[str, str], ...],
) -> FrontierBatch:
    """Generate deterministic residue tensors with explicit resolved contact labels."""
    examples = []
    for local_index in range(batch_size):
        global_index = rank * batch_size + local_index
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + 104729 * global_index)
        antibody_length = 10 + global_index % 5
        antigen_length = 14 + global_index % 7
        antibody = torch.randn(
            antibody_length,
            antibody_dim,
            generator=generator,
        )
        antigen = torch.randn(
            antigen_length,
            antigen_dim,
            generator=generator,
        )
        heavy_length = max(1, (2 * antibody_length) // 3)
        chain_ids = torch.cat(
            (
                torch.zeros(heavy_length, dtype=torch.long),
                torch.ones(antibody_length - heavy_length, dtype=torch.long),
            )
        )
        contact_target = torch.zeros(
            antigen_length,
            antibody_length,
            dtype=torch.uint8,
        )
        contact_target[global_index % antigen_length, global_index % antibody_length] = 1
        contact_target[
            (global_index + 3) % antigen_length,
            (global_index + 2) % antibody_length,
        ] = 1
        contact_mask = torch.ones_like(contact_target, dtype=torch.bool)
        examples.append(
            FrontierExample(
                instance_id=f"synthetic-{global_index:05d}",
                antibody_id=f"ab-{global_index:05d}",
                antigen_id=f"ag-{global_index:05d}",
                antibody_embeddings=antibody,
                antigen_embeddings=antigen,
                antibody_chain_ids=chain_ids,
                contact_target=contact_target,
                contact_mask=contact_mask,
            )
        )
    return FrontierBatchCollator(positive_graph)(examples)


def _ddp(
    module: torch.nn.Module,
    device: torch.device,
    local_rank: int,
    *,
    init_sync: bool = True,
) -> torch.nn.Module:
    """Wrap a module with explicit single-device DDP configuration."""
    if not dist.is_available() or not dist.is_initialized():
        return module
    if device.type == "cuda":
        return DistributedDataParallel(
            module,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=init_sync,
            init_sync=init_sync,
        )
    return DistributedDataParallel(
        module,
        broadcast_buffers=init_sync,
        init_sync=init_sync,
    )


def _unwrap(module: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying model for both single-process and DDP execution."""
    return module.module if isinstance(module, DistributedDataParallel) else module


def _autocast(
    device: torch.device,
    precision: str,
) -> contextlib.AbstractContextManager[Any]:
    """Use the requested audited precision mode."""
    if device.type == "cuda" and precision == "bfloat16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _synchronize_stage(rank: int, stage: str, device: torch.device) -> None:
    """Force asynchronous CUDA failures to surface at a named boundary."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    _rank_log(rank, stage)


def _finite_gradient_norm(module: torch.nn.Module) -> float:
    """Clip and validate the global gradient norm."""
    norm = torch.nn.utils.clip_grad_norm_(module.parameters(), max_norm=1.0)
    value = float(norm.detach().cpu())
    if not np.isfinite(value):
        raise FloatingPointError("Distributed smoke produced a non-finite gradient norm")
    return value


def _assert_rank_parameter_agreement(module: torch.nn.Module) -> float:
    """Verify exact rank agreement and return a diagnostic checksum."""
    device = next(module.parameters()).device
    checksum = torch.zeros((), dtype=torch.float64, device=device)
    for parameter in module.parameters():
        checksum = checksum + parameter.detach().to(torch.float64).sum()
    if dist.is_available() and dist.is_initialized():
        local_digest = _state_dict_digest(module)
        digests = _gather_objects(local_digest, dist.get_world_size())
        if any(digest != local_digest for digest in digests):
            raise RuntimeError("DDP model-state SHA-256 differs across ranks")
    return float(checksum.cpu())


def _train_d0(
    batch: FrontierBatch,
    *,
    steps: int,
    total_size: int,
    positive_mask: torch.Tensor,
    exclusion_mask: torch.Tensor,
    antibody_dim: int,
    antigen_dim: int,
    device: torch.device,
    local_rank: int,
    precision: str,
) -> tuple[ProtocolD0Retriever, list[dict[str, float]]]:
    """Exercise global all-rank multi-positive D0 optimization."""
    model = ProtocolD0Retriever(
        antibody_input_dim=antibody_dim,
        antigen_input_dim=antigen_dim,
        embedding_dim=32,
        projection_hidden_dim=64,
        dropout=0.0,
    ).to(device)
    _synchronize_stage(
        dist.get_rank() if dist.is_initialized() else 0,
        "d0_model_constructed",
        device,
    )
    distributed_model = _ddp(model, device, local_rank)
    _synchronize_stage(
        dist.get_rank() if dist.is_initialized() else 0,
        "d0_ddp_constructed",
        device,
    )
    optimizer = torch.optim.AdamW(
        distributed_model.parameters(),
        lr=1e-3,
        weight_decay=0.2,
        betas=(0.9, 0.98),
        eps=1e-8,
    )
    history = []
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, precision):
            output = distributed_model(
                batch.antibody_embeddings,
                batch.antigen_embeddings,
                batch.antibody_mask,
                batch.antigen_mask,
                batch.antibody_chain_ids,
            )
            loss = symmetric_multi_positive_info_nce(
                output["antibody_embeddings"],
                output["antigen_embeddings"],
                positive_mask,
                exclusion_mask,
                temperature=output["temperature"],
            )
        _synchronize_stage(
            dist.get_rank() if dist.is_initialized() else 0,
            f"d0_forward_{step + 1}_synchronized",
            device,
        )
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError("D0 smoke loss is non-finite")
        loss.backward()
        _synchronize_stage(
            dist.get_rank() if dist.is_initialized() else 0,
            f"d0_backward_{step + 1}_synchronized",
            device,
        )
        gradient_norm = _finite_gradient_norm(distributed_model)
        optimizer.step()
        history.append(
            {
                "step": float(step + 1),
                "loss": float(loss.detach().cpu()),
                "gradient_norm_before_clip": gradient_norm,
                "global_candidate_count": float(total_size),
            }
        )
        if not dist.is_initialized() or dist.get_rank() == 0:
            _rank_log(
                0,
                "d0_step_complete",
                step=step + 1,
                loss=float(loss.detach().cpu()),
            )
    unwrapped = _unwrap(distributed_model)
    _assert_rank_parameter_agreement(unwrapped)
    return copy.deepcopy(unwrapped), history


def _exact_balanced_contact_mean(
    logits: torch.Tensor,
    targets: torch.Tensor,
    resolved_mask: torch.Tensor,
    *,
    group: Any = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Reduce contact BCE by the exact global dual-class complex count."""
    details = balanced_contact_bce(
        logits,
        targets,
        resolved_mask,
        return_details=True,
    )
    loss = exact_distributed_mean(
        details["loss_sum"],
        details["valid_complex_count"],
        group=group,
    )
    return loss, details


def _train_r2(
    coarse: ProtocolD0Retriever,
    batch: FrontierBatch,
    *,
    steps: int,
    antibody_dim: int,
    antigen_dim: int,
    device: torch.device,
    local_rank: int,
    precision: str,
    verified_equal_initialization: bool = False,
    model_initialization_seed: int = 20260725,
) -> tuple[
    ProtocolReranker,
    torch.optim.Optimizer,
    torch.optim.lr_scheduler.LRScheduler,
    list[dict[str, float]],
]:
    """Exercise indexed pair reranking, native contacts, and exact global means."""
    cpu_rng_state = torch.get_rng_state()
    try:
        if verified_equal_initialization:
            torch.default_generator.manual_seed(model_initialization_seed)
        model = ProtocolReranker(
            coarse,
            antibody_input_dim=antibody_dim,
            antigen_input_dim=antigen_dim,
            d_model=64,
            num_heads=8,
            num_layers=2,
            ffn_dim=128,
            dropout=0.1,
            contact_pair_dim=32,
        ).to(device)
    finally:
        if verified_equal_initialization:
            torch.set_rng_state(cpu_rng_state)
    _synchronize_stage(
        dist.get_rank() if dist.is_initialized() else 0,
        "r2_model_constructed",
        device,
    )
    if verified_equal_initialization:
        local_digest = _state_dict_digest(model)
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank_digests = _gather_objects(local_digest, world_size)
        if any(digest != local_digest for digest in rank_digests):
            raise RuntimeError(
                "R2 verified-equal initialization produced divergent rank states"
            )
        _rank_log(
            dist.get_rank() if dist.is_initialized() else 0,
            "r2_initial_state_verified_equal",
            sha256=local_digest,
            world_size=world_size,
        )
    distributed_model = _ddp(
        model,
        device,
        local_rank,
        init_sync=not verified_equal_initialization,
    )
    _synchronize_stage(
        dist.get_rank() if dist.is_initialized() else 0,
        "r2_ddp_constructed",
        device,
    )
    trainable = [parameter for parameter in distributed_model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=1e-4,
        weight_decay=0.01,
        betas=(0.9, 0.98),
        eps=1e-8,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, steps),
    )
    batch_size = batch.batch_size
    antibody_indices = torch.arange(batch_size, device=device).repeat_interleave(
        batch_size
    )
    antigen_indices = torch.arange(batch_size, device=device).repeat(batch_size)
    diagonal_positions = torch.arange(batch_size, device=device) * (
        batch_size + 1
    )
    local_positive = batch.positive_pair_mask
    local_excluded = torch.zeros_like(local_positive)
    if batch_size > 2:
        for query in range(batch_size):
            candidate = (query + 2) % batch_size
            if not bool(local_positive[query, candidate]):
                local_excluded[query, candidate] = True

    history = []
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, precision):
            output = distributed_model(
                batch.antibody_embeddings.index_select(0, antibody_indices),
                batch.antigen_embeddings.index_select(0, antigen_indices),
                batch.antibody_mask.index_select(0, antibody_indices),
                batch.antigen_mask.index_select(0, antigen_indices),
                batch.antibody_chain_ids.index_select(0, antibody_indices),
            )
            score_matrix = output["retrieval_score"].reshape(batch_size, batch_size)
            local_retrieval = symmetric_score_matrix_info_nce(
                score_matrix,
                local_positive,
                local_excluded,
                temperature=0.07,
            )
            retrieval_loss = exact_distributed_mean(
                local_retrieval * batch_size,
                batch_size,
            )
            contact_loss, contact_details = _exact_balanced_contact_mean(
                output["contact_logits"].index_select(0, diagonal_positions),
                batch.contact_target,
                batch.contact_mask,
            )
            loss = retrieval_loss + 0.10 * contact_loss
        _synchronize_stage(
            dist.get_rank() if dist.is_initialized() else 0,
            f"r2_forward_{step + 1}_synchronized",
            device,
        )
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError("R2 smoke loss is non-finite")
        loss.backward()
        _synchronize_stage(
            dist.get_rank() if dist.is_initialized() else 0,
            f"r2_backward_{step + 1}_synchronized",
            device,
        )
        gradient_norm = _finite_gradient_norm(distributed_model)
        optimizer.step()
        scheduler.step()
        history.append(
            {
                "step": float(step + 1),
                "loss": float(loss.detach().cpu()),
                "retrieval_loss": float(retrieval_loss.detach().cpu()),
                "contact_loss": float(contact_loss.detach().cpu()),
                "contact_valid_complex_count": float(
                    contact_details["valid_complex_count"]
                ),
                "contact_observed_complex_count": float(
                    contact_details["observed_complex_count"]
                ),
                "gradient_norm_before_clip": gradient_norm,
            }
        )
        if not dist.is_initialized() or dist.get_rank() == 0:
            _rank_log(
                0,
                "r2_step_complete",
                step=step + 1,
                loss=float(loss.detach().cpu()),
            )
    unwrapped = _unwrap(distributed_model)
    _assert_rank_parameter_agreement(unwrapped)
    return unwrapped, optimizer, scheduler, history


def _gather_objects(value: Any, world_size: int) -> list[Any]:
    """Gather rank-ordered Python objects on every process."""
    if world_size == 1:
        return [value]
    gathered: list[Any] = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, value)
    return gathered


def main() -> None:
    """Run the sealed synthetic-only smoke and publish a verified receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--batch-size-per-rank", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument(
        "--precision",
        choices=("float32", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--parallelism", choices=("ddp",), required=True)
    parser.add_argument("--expected-world-size", type=int, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--launch-manifest", type=Path, required=True)
    parser.add_argument("--expected-launch-manifest-sha256", required=True)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument(
        "--synthetic-only",
        action="store_true",
        help="Required acknowledgement that no biological data may be mounted or read.",
    )
    args = parser.parse_args()
    if not args.synthetic_only:
        raise ValueError("--synthetic-only is mandatory for this smoke test")
    if args.steps <= 0 or args.batch_size_per_rank < 2:
        raise ValueError("steps must be positive and batch size must be at least two")
    repository_root = Path(__file__).resolve().parents[1]
    source_sha256 = _source_digest(repository_root)
    expected_source_sha256 = require_sha256(
        args.expected_source_sha256,
        field="expected_source_sha256",
    )
    if source_sha256 != expected_source_sha256:
        raise LaunchContractError(
            "Executed source boundary differs from the externally frozen digest: "
            f"expected={expected_source_sha256}, observed={source_sha256}"
        )
    verify_launch_manifest(
        args.launch_manifest,
        expected_file_sha256=args.expected_launch_manifest_sha256,
        expected_source_sha256=source_sha256,
        expected_world_size=args.expected_world_size,
        expected_backend="nccl" if torch.cuda.is_available() else "gloo",
        expected_parallelism=args.parallelism,
        expected_precision=args.precision,
    )
    world_size, rank, local_rank, device = _distributed_context(
        expected_world_size=args.expected_world_size,
    )
    training_mode = validate_training_mode(
        parallelism=args.parallelism,
        precision=args.precision,
        cuda_available=device.type == "cuda",
        bf16_supported=(
            bool(torch.cuda.is_bf16_supported()) if device.type == "cuda" else False
        ),
        require_cuda=args.require_cuda,
    )
    try:
        set_reproducible_seed(args.seed + rank, deterministic=True)
        _rank_log(rank, "seed_configured", seed=args.seed + rank)
        antibody_dim = 48
        antigen_dim = 64
        total_size = world_size * args.batch_size_per_rank
        positive_graph = _all_positive_graph(total_size)
        positive_mask, exclusion_mask = _graph_masks(
            total_size,
            positive_graph,
            device,
        )
        batch = _synthetic_batch(
            rank=rank,
            batch_size=args.batch_size_per_rank,
            seed=args.seed,
            antibody_dim=antibody_dim,
            antigen_dim=antigen_dim,
            positive_graph=positive_graph,
        ).to(device)
        _rank_log(rank, "synthetic_batch_ready", batch_size=batch.batch_size)
        coarse, d0_history = _train_d0(
            batch,
            steps=args.steps,
            total_size=total_size,
            positive_mask=positive_mask,
            exclusion_mask=exclusion_mask,
            antibody_dim=antibody_dim,
            antigen_dim=antigen_dim,
            device=device,
            local_rank=local_rank,
            precision=args.precision,
        )
        _rank_log(rank, "d0_complete")
        coarse_sha256 = _state_dict_digest(coarse)
        _rank_log(rank, "coarse_state_hashed", sha256=coarse_sha256)
        model, optimizer, scheduler, r2_history = _train_r2(
            coarse,
            batch,
            steps=args.steps,
            antibody_dim=antibody_dim,
            antigen_dim=antigen_dim,
            device=device,
            local_rank=local_rank,
            precision=args.precision,
        )
        _rank_log(rank, "r2_complete")
        final_model_sha256 = _state_dict_digest(model)
        rng_states = _gather_objects(capture_rng_state(), world_size)
        _rank_log(rank, "rng_states_gathered", rng_world_size=len(rng_states))
        histories = _gather_objects(
            {
                "rank": rank,
                "d0": d0_history,
                "r2": r2_history,
            },
            world_size,
        )
        synthetic_manifest = {
            "format": "frontier_synthetic_smoke_manifest_v1",
            "seed": args.seed,
            "world_size": world_size,
            "batch_size_per_rank": args.batch_size_per_rank,
            "steps": args.steps,
            "antibody_dim": antibody_dim,
            "antigen_dim": antigen_dim,
            "positive_pairs": positive_graph,
            "biological_data_access": "forbidden",
        }
        synthetic_sha256 = _sha256_bytes(_canonical_json(synthetic_manifest))
        provenance = {
            "coarse_state": coarse_sha256,
            "source_tree": source_sha256,
            "synthetic_manifest": synthetic_sha256,
        }
        config = {
            "format": "frontier_ddp_smoke_config_v1",
            "world_size": world_size,
            "steps": args.steps,
            "batch_size_per_rank": args.batch_size_per_rank,
            "precision": args.precision if device.type == "cuda" else "float32",
            "backend": (
                dist.get_backend()
                if dist.is_available() and dist.is_initialized()
                else "single_process"
            ),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device_type": device.type,
            "candidate": "D0_then_R2_synthetic_correctness_only",
            "parallelism": training_mode["parallelism"],
            "gradient_reduction": training_mode["gradient_reduction"],
            "global_loss_gradient_semantics": (
                "D0 uses local_slice_ddp_exact under fixed standard-DDP averaging; "
                "R2 exact means use SUM-reduced numerators and denominators with "
                "world_size/total_count backward scaling."
            ),
            "metric_claim": training_mode["metric_claim"],
            "efficiency_claim": training_mode["efficiency_claim"],
            "launch_manifest_sha256": args.expected_launch_manifest_sha256,
        }
        generation = f"step_{args.steps:06d}"
        if rank == 0:
            receipt = create_checkpoint_generation(
                args.output_root / "checkpoints",
                generation,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=None,
                epoch=0,
                global_step=args.steps,
                provenance_hashes=provenance,
                config=config,
                rng_states=rng_states,
            )
            summary = {
                "format": "frontier_ddp_smoke_result_v1",
                "status": "passed",
                "scope": "synthetic distributed correctness only",
                "claim_boundary": (
                    "This run contains no biological data and provides no model "
                    "performance or state-of-the-art evidence."
                ),
                "world_size": world_size,
                "device_type": device.type,
                "source_sha256": source_sha256,
                "launch_manifest_sha256": args.expected_launch_manifest_sha256,
                "synthetic_manifest_sha256": synthetic_sha256,
                "coarse_state_sha256": coarse_sha256,
                "final_model_state_sha256": final_model_sha256,
                "checkpoint_receipt": receipt,
                "rank_histories": histories,
            }
            _atomic_json(args.output_root / "result.json", summary)
            _rank_log(rank, "checkpoint_published", generation=generation)
        if world_size > 1:
            dist.barrier()
        payload = load_checkpoint_generation(
            args.output_root / "checkpoints",
            generation,
            expected_provenance_hashes=provenance,
        )
        if int(payload["receipt"]["rng_world_size"]) != world_size:
            raise RuntimeError("Checkpoint did not preserve every distributed RNG rank")
        _rank_log(rank, "checkpoint_verified", generation=generation)
        if rank == 0:
            print(
                json.dumps(
                    {
                        "status": "passed",
                        "world_size": world_size,
                        "source_sha256": source_sha256,
                        "checkpoint_sha256": payload["checkpoint_sha256"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
