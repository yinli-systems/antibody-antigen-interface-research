#!/usr/bin/env python3
"""Run a synthetic R2 FSDP2 parity and DCP restart diagnostic."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
import torch.distributed as dist
import torch.nn.functional as functional

from distributed_launch_contract import (
    LaunchContractError,
    publish_create_only_json,
    require_sha256,
    source_tree_digest,
    validate_distributed_identity,
    verify_launch_manifest,
)
from interaction_first.frontier_fsdp2 import (
    FSDP2_CLAIM_BOUNDARY,
    fully_shard_reranker,
    load_dcp_generation,
    percentile,
    save_dcp_generation,
    validate_fsdp2_runtime,
)
from interaction_first.frontier_model import ProtocolD0Retriever, ProtocolReranker


SOURCE_PATHS = (
    "scripts/distributed_launch_contract.py",
    "scripts/frontier_fsdp2_dcp_smoke.py",
    "slurm/frontier_fsdp2_dcp_smoke.sbatch",
    "src/interaction_first/frontier_fsdp2.py",
    "src/interaction_first/frontier_model.py",
)


def _canonical_json_bytes(value: Any) -> bytes:
    """Serialize finite JSON deterministically."""
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _source_digest(repository_root: Path) -> str:
    """Bind every executable source file used by this diagnostic."""
    return source_tree_digest(repository_root, SOURCE_PATHS)


def _sha256_json(value: Any) -> str:
    """Hash one canonical JSON value."""
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _initialize_distributed(expected_world_size: int) -> tuple[int, int, torch.device]:
    """Initialize NCCL and validate torchrun and Slurm identity."""
    if not torch.cuda.is_available():
        raise RuntimeError("This FSDP2 diagnostic requires CUDA")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    device = torch.device("cuda", local_rank)
    validate_distributed_identity(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        visible_device_count=torch.cuda.device_count(),
        minimum_world_size=1,
        expected_world_size=expected_world_size,
    )
    return rank, local_rank, device


def _autocast(precision: str) -> contextlib.AbstractContextManager[Any]:
    """Return the explicitly selected CUDA precision context."""
    if precision == "bfloat16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _build_model(seed: int, device: torch.device) -> ProtocolReranker:
    """Construct the reduced, code-path-equivalent R2 fixture."""
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state(device)
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        coarse = ProtocolD0Retriever(
            antibody_input_dim=24,
            antigen_input_dim=32,
            embedding_dim=16,
            projection_hidden_dim=32,
            dropout=0.0,
        )
        model = ProtocolReranker(
            coarse,
            antibody_input_dim=24,
            antigen_input_dim=32,
            d_model=32,
            num_heads=4,
            num_layers=2,
            ffn_dim=64,
            dropout=0.0,
            contact_pair_dim=16,
        )
        return model.to(device)
    finally:
        torch.set_rng_state(cpu_state)
        torch.cuda.set_rng_state(cuda_state, device)


def _synthetic_batch(
    indices: Iterable[int],
    *,
    seed: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Generate stateless fixed-shape train-only inputs from global indices."""
    rows = [int(value) for value in indices]
    if not rows:
        raise ValueError("A synthetic batch cannot be empty")
    antibody_rows: list[torch.Tensor] = []
    antigen_rows: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    retrieval_targets: list[float] = []
    for global_index in rows:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + 104729 * global_index)
        antibody_rows.append(torch.randn(7, 24, generator=generator))
        antigen_rows.append(torch.randn(9, 32, generator=generator))
        target = torch.zeros(9, 7)
        target[global_index % 9, global_index % 7] = 1.0
        target[(global_index + 3) % 9, (global_index + 2) % 7] = 1.0
        targets.append(target)
        retrieval_targets.append(math.sin(global_index + 0.5))
    batch_size = len(rows)
    return {
        "antibody_features": torch.stack(antibody_rows).to(device),
        "antigen_features": torch.stack(antigen_rows).to(device),
        "antibody_mask": torch.ones(batch_size, 7, dtype=torch.bool, device=device),
        "antigen_mask": torch.ones(batch_size, 9, dtype=torch.bool, device=device),
        "antibody_chain_ids": torch.tensor(
            [[0, 0, 0, 0, 1, 1, 1] for _ in rows],
            dtype=torch.long,
            device=device,
        ),
        "contact_target": torch.stack(targets).to(device),
        "retrieval_target": torch.tensor(
            retrieval_targets,
            dtype=torch.float32,
            device=device,
        ),
    }


def _loss(
    model: torch.nn.Module,
    batch: Mapping[str, torch.Tensor],
    precision: str,
) -> torch.Tensor:
    """Evaluate one deterministic paired retrieval and native-contact objective."""
    with _autocast(precision):
        output = model(
            batch["antibody_features"],
            batch["antigen_features"],
            batch["antibody_mask"],
            batch["antigen_mask"],
            batch["antibody_chain_ids"],
        )
        retrieval = functional.mse_loss(
            output["retrieval_score"].float(),
            batch["retrieval_target"],
        )
        contact = functional.binary_cross_entropy_with_logits(
            output["contact_logits"].float(),
            batch["contact_target"],
        )
        return retrieval + 0.10 * contact


def _train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Mapping[str, torch.Tensor],
    *,
    precision: str,
) -> tuple[float, float]:
    """Run one synchronized optimizer step and return loss and wall time."""
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    started = time.perf_counter()
    loss = _loss(model, batch, precision)
    if not bool(torch.isfinite(loss).item()):
        raise FloatingPointError("Synthetic FSDP2 loss is non-finite")
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return float(loss.detach().cpu()), elapsed


def _optimizer(model: torch.nn.Module) -> torch.optim.Optimizer:
    """Create the locked optimizer after optional sharding."""
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    return torch.optim.AdamW(
        trainable,
        lr=1e-4,
        betas=(0.9, 0.98),
        eps=1e-8,
        weight_decay=0.01,
    )


def _full_model_state(model: torch.nn.Module) -> dict[str, Any]:
    """Collect a rank-zero CPU full model state through the DCP state API."""
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
    )

    return get_model_state_dict(
        model,
        options=StateDictOptions(
            full_state_dict=True,
            cpu_offload=True,
            strict=True,
        ),
    )


def _state_difference(
    observed: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> dict[str, float | int]:
    """Return maximum absolute and relative tensor differences."""
    if set(observed) != set(expected):
        raise RuntimeError("Compared model states have different keys")
    maximum_absolute = 0.0
    maximum_relative = 0.0
    tensor_count = 0
    element_count = 0
    for name in sorted(observed):
        left = observed[name]
        right = expected[name]
        if not torch.is_tensor(left) or not torch.is_tensor(right):
            if left != right:
                raise RuntimeError(f"Non-tensor state differs for {name}")
            continue
        left_tensor = left.detach().cpu().float()
        right_tensor = right.detach().cpu().float()
        if tuple(left_tensor.shape) != tuple(right_tensor.shape):
            raise RuntimeError(f"Tensor state shape differs for {name}")
        absolute = (left_tensor - right_tensor).abs()
        denominator = torch.maximum(right_tensor.abs(), torch.full_like(right_tensor, 1e-12))
        maximum_absolute = max(
            maximum_absolute,
            float(absolute.max()) if absolute.numel() else 0.0,
        )
        maximum_relative = max(
            maximum_relative,
            float((absolute / denominator).max()) if absolute.numel() else 0.0,
        )
        tensor_count += 1
        element_count += int(left_tensor.numel())
    return {
        "max_absolute": maximum_absolute,
        "max_relative": maximum_relative,
        "tensor_count": tensor_count,
        "element_count": element_count,
    }


def _assert_difference(
    difference: Mapping[str, float | int],
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
    label: str,
) -> None:
    """Fail closed when a state comparison exceeds either locked tolerance."""
    if (
        float(difference["max_absolute"]) > absolute_tolerance
        and float(difference["max_relative"]) > relative_tolerance
    ):
        raise RuntimeError(
            f"{label} exceeded parity tolerance: {dict(difference)}, "
            f"atol={absolute_tolerance}, rtol={relative_tolerance}"
        )


def _gather_rank_objects(value: Any, world_size: int) -> list[Any]:
    """Gather a rank-ordered Python value on every rank."""
    values: list[Any] = [None for _ in range(world_size)]
    dist.all_gather_object(values, value)
    return values


def _global_mean_scalar(value: float, device: torch.device) -> float:
    """Average one rank-local scalar for reference-loss comparison."""
    tensor = torch.tensor(value, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= dist.get_world_size()
    return float(tensor.cpu())


def main() -> None:
    """Run FSDP2 parity, immutable DCP save/load, and exact continuation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--steps-before-checkpoint", type=int, default=3)
    parser.add_argument("--batch-size-per-rank", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--precision", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--parallelism", choices=("fsdp2",), required=True)
    parser.add_argument("--expected-world-size", type=int, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--launch-manifest", type=Path, required=True)
    parser.add_argument("--expected-launch-manifest-sha256", required=True)
    parser.add_argument("--absolute-tolerance", type=float, default=2e-5)
    parser.add_argument("--relative-tolerance", type=float, default=2e-4)
    parser.add_argument("--synthetic-only", action="store_true")
    args = parser.parse_args()
    if not args.synthetic_only:
        raise ValueError("--synthetic-only is mandatory")
    if args.steps_before_checkpoint < 1 or args.batch_size_per_rank < 1:
        raise ValueError("Step and batch counts must be positive")
    if args.absolute_tolerance <= 0.0 or args.relative_tolerance <= 0.0:
        raise ValueError("Parity tolerances must be positive")
    repository_root = Path(__file__).resolve().parents[1]
    observed_source = _source_digest(repository_root)
    expected_source = require_sha256(
        args.expected_source_sha256,
        field="expected_source_sha256",
    )
    if observed_source != expected_source:
        raise LaunchContractError(
            "Executed source boundary differs from the frozen digest: "
            f"expected={expected_source}, observed={observed_source}"
        )
    rank, _, device = _initialize_distributed(args.expected_world_size)
    world_size = dist.get_world_size()
    try:
        verify_launch_manifest(
            args.launch_manifest,
            expected_file_sha256=args.expected_launch_manifest_sha256,
            expected_source_sha256=observed_source,
            expected_world_size=world_size,
            expected_backend="nccl",
            expected_parallelism="fsdp2",
            expected_precision=args.precision,
        )
        runtime_receipt = validate_fsdp2_runtime(
            parallelism=args.parallelism,
            precision=args.precision,
            cuda_available=True,
            bf16_supported=bool(torch.cuda.is_bf16_supported()),
            distributed_initialized=dist.is_initialized(),
            world_size=world_size,
            torch_version=torch.__version__,
        )
        output_root = args.output_root.resolve()
        if rank == 0:
            if output_root.exists() or output_root.is_symlink():
                raise FileExistsError(f"Refusing to overwrite output root: {output_root}")
            output_root.mkdir(parents=True)
        dist.barrier()
        global_size = world_size * args.batch_size_per_rank
        synthetic_manifest = {
            "format": "frontier_fsdp2_synthetic_fixture_v1",
            "seed": args.seed,
            "world_size": world_size,
            "batch_size_per_rank": args.batch_size_per_rank,
            "global_indices": list(range(global_size)),
            "antibody_length": 7,
            "antigen_length": 9,
            "biological_data_access": "forbidden",
        }
        provenance = {
            "launch_manifest": args.expected_launch_manifest_sha256,
            "source_tree": observed_source,
            "synthetic_fixture": _sha256_json(synthetic_manifest),
        }
        local_indices = range(
            rank * args.batch_size_per_rank,
            (rank + 1) * args.batch_size_per_rank,
        )
        local_batch = _synthetic_batch(local_indices, seed=args.seed, device=device)
        full_batch = (
            _synthetic_batch(range(global_size), seed=args.seed, device=device)
            if rank == 0
            else None
        )
        model = _build_model(args.seed, device)
        fully_shard_reranker(model, precision=args.precision)
        optimizer = _optimizer(model)
        reference = _build_model(args.seed, device) if rank == 0 else None
        reference_optimizer = _optimizer(reference) if reference is not None else None
        step_receipts: list[dict[str, Any]] = []
        for step in range(args.steps_before_checkpoint):
            local_loss, elapsed = _train_step(
                model,
                optimizer,
                local_batch,
                precision=args.precision,
            )
            global_loss = _global_mean_scalar(local_loss, device)
            if rank == 0:
                assert reference is not None
                assert reference_optimizer is not None
                assert full_batch is not None
                reference_loss, reference_elapsed = _train_step(
                    reference,
                    reference_optimizer,
                    full_batch,
                    precision=args.precision,
                )
                loss_difference = abs(global_loss - reference_loss)
            else:
                reference_loss = 0.0
                reference_elapsed = 0.0
                loss_difference = 0.0
            step_receipts.append(
                {
                    "step": step + 1,
                    "rank": rank,
                    "local_loss": local_loss,
                    "global_loss": global_loss,
                    "rank_step_seconds": elapsed,
                    "reference_loss_rank0": reference_loss,
                    "reference_step_seconds_rank0": reference_elapsed,
                    "loss_absolute_difference_rank0": loss_difference,
                }
            )
        checkpoint_receipt = save_dcp_generation(
            output_root / "checkpoints",
            f"step_{args.steps_before_checkpoint:06d}",
            model=model,
            optimizer=optimizer,
            provenance_hashes=provenance,
            training_state={
                "global_step": args.steps_before_checkpoint,
                "seed": args.seed,
                "synthetic_only": True,
            },
        )
        continuation_loss, continuation_seconds = _train_step(
            model,
            optimizer,
            local_batch,
            precision=args.precision,
        )
        continuation_global_loss = _global_mean_scalar(continuation_loss, device)
        if rank == 0:
            assert reference is not None
            assert reference_optimizer is not None
            assert full_batch is not None
            reference_continuation_loss, _ = _train_step(
                reference,
                reference_optimizer,
                full_batch,
                precision=args.precision,
            )
        else:
            reference_continuation_loss = 0.0
        uninterrupted_state = _full_model_state(model)
        restored = _build_model(args.seed, device)
        fully_shard_reranker(restored, precision=args.precision)
        restored_optimizer = _optimizer(restored)
        load_started = time.perf_counter()
        loaded_receipt = load_dcp_generation(
            output_root / "checkpoints",
            f"step_{args.steps_before_checkpoint:06d}",
            model=restored,
            optimizer=restored_optimizer,
            expected_provenance_hashes=provenance,
        )
        torch.cuda.synchronize()
        load_seconds = time.perf_counter() - load_started
        restored_loss, restored_seconds = _train_step(
            restored,
            restored_optimizer,
            local_batch,
            precision=args.precision,
        )
        restored_global_loss = _global_mean_scalar(restored_loss, device)
        restored_state = _full_model_state(restored)
        if rank == 0:
            assert reference is not None
            reference_state = {
                name: value.detach().cpu()
                for name, value in reference.state_dict().items()
            }
            fsdp_reference_difference = _state_difference(uninterrupted_state, reference_state)
            restart_difference = _state_difference(restored_state, uninterrupted_state)
            _assert_difference(
                fsdp_reference_difference,
                absolute_tolerance=args.absolute_tolerance,
                relative_tolerance=args.relative_tolerance,
                label="FSDP2 versus single-process reference",
            )
            _assert_difference(
                restart_difference,
                absolute_tolerance=0.0,
                relative_tolerance=0.0,
                label="DCP continuation",
            )
            if abs(restored_global_loss - continuation_global_loss) > 1e-8:
                raise RuntimeError("DCP continuation loss differs from uninterrupted execution")
        else:
            fsdp_reference_difference = {}
            restart_difference = {}
        rank_payload = {
            "rank": rank,
            "steps": step_receipts,
            "continuation_seconds": continuation_seconds,
            "restored_seconds": restored_seconds,
            "load_seconds": load_seconds,
            "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(device),
        }
        all_rank_payloads = _gather_rank_objects(rank_payload, world_size)
        if rank == 0:
            all_step_times = [
                float(step["rank_step_seconds"])
                for payload in all_rank_payloads
                for step in payload["steps"]
            ]
            total_synthetic_residue_tokens = (
                global_size * (7 + 9) * args.steps_before_checkpoint
            )
            total_rank_step_seconds = sum(all_step_times)
            result = {
                "format": "frontier_fsdp2_dcp_smoke_result_v1",
                "status": "passed",
                "scope": "synthetic R2 FSDP2 and DCP correctness only",
                "claim_boundary": FSDP2_CLAIM_BOUNDARY,
                "source_sha256": observed_source,
                "launch_manifest_sha256": args.expected_launch_manifest_sha256,
                "synthetic_manifest": synthetic_manifest,
                "synthetic_manifest_sha256": provenance["synthetic_fixture"],
                "runtime": runtime_receipt,
                "parity": {
                    "absolute_tolerance": args.absolute_tolerance,
                    "relative_tolerance": args.relative_tolerance,
                    "fsdp2_vs_single_process": fsdp_reference_difference,
                    "dcp_continuation": restart_difference,
                    "uninterrupted_continuation_global_loss": continuation_global_loss,
                    "restored_continuation_global_loss": restored_global_loss,
                    "single_process_continuation_loss": reference_continuation_loss,
                },
                "checkpoint": {
                    "save_receipt_content_sha256": checkpoint_receipt["content_sha256"],
                    "loaded_receipt_content_sha256": loaded_receipt["content_sha256"],
                    "source_world_size": checkpoint_receipt["source_world_size"],
                    "load_world_size": world_size,
                },
                "telemetry": {
                    "step_seconds_p50": percentile(all_step_times, 0.50),
                    "step_seconds_p95": percentile(all_step_times, 0.95),
                    "step_seconds_p99": percentile(all_step_times, 0.99),
                    "synthetic_residue_tokens": total_synthetic_residue_tokens,
                    "diagnostic_residue_tokens_per_rank_second": (
                        total_synthetic_residue_tokens / total_rank_step_seconds
                    ),
                    "rank_receipts": all_rank_payloads,
                    "efficiency_claim": "forbidden",
                },
                "outer_or_test_access": False,
                "biological_training_authorized": False,
            }
            publish_create_only_json(output_root / "result.json", result)
            os.chmod(output_root / "result.json", 0o444)
            print(
                json.dumps(
                    {
                        "status": "passed",
                        "world_size": world_size,
                        "source_sha256": observed_source,
                        "checkpoint_receipt": checkpoint_receipt["content_sha256"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
