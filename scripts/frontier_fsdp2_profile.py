#!/usr/bin/env python3
"""Measure warmed FSDP2 throughput for the reduced synthetic R2 fixture."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from distributed_launch_contract import (
    LaunchContractError,
    publish_create_only_json,
    require_sha256,
    source_tree_digest,
    verify_launch_manifest,
)
from frontier_fsdp2_dcp_smoke import (
    _assert_difference,
    _build_model,
    _full_model_state,
    _gather_rank_objects,
    _initialize_distributed,
    _optimizer,
    _sha256_json,
    _state_difference,
    _synthetic_batch,
    _train_step,
)
from interaction_first.frontier_fsdp2 import (
    FSDP2_CLAIM_BOUNDARY,
    fully_shard_reranker,
    load_dcp_generation,
    percentile,
    promote_scalar_parameters_for_fsdp2,
    save_dcp_generation,
    validate_fsdp2_runtime,
)


SOURCE_PATHS = (
    "scripts/distributed_launch_contract.py",
    "scripts/frontier_fsdp2_dcp_smoke.py",
    "scripts/frontier_fsdp2_profile.py",
    "scripts/generate_runtime_manifest.py",
    "slurm/frontier_fsdp2_profile.sbatch",
    "slurm/frontier_fsdp2_multinode_profile.sbatch",
    "src/interaction_first/frontier_fsdp2.py",
    "src/interaction_first/frontier_model.py",
)
PROFILE_FORMAT = "frontier_fsdp2_warmed_profile_result_v1"
PROFILE_CLAIM_BOUNDARY = (
    "Exact reduced-fixture systems measurement only; no full-model, biological, "
    "scaling-efficiency, production-readiness, SOTA, or world-leading claim."
)


def _source_digest(repository_root: Path) -> str:
    """Bind every executable source file used by this profile."""
    return source_tree_digest(repository_root, SOURCE_PATHS)


def _percentile_map(values: list[float]) -> dict[str, float]:
    """Summarize a non-empty timing series with locked quantiles."""
    return {
        "minimum": min(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "maximum": max(values),
    }


def _build_sharded(
    seed: int,
    *,
    precision: str,
    device: torch.device,
) -> tuple[torch.nn.Module, torch.optim.Optimizer]:
    """Build and shard the reviewed reduced R2 fixture."""
    model = _build_model(seed, device)
    promoted = promote_scalar_parameters_for_fsdp2(model)
    if promoted != ("contact_head.bias",):
        raise RuntimeError(f"Unexpected scalar promotion set: {promoted}")
    fully_shard_reranker(model, precision=precision)
    return model, _optimizer(model)


def main() -> None:
    """Run warmup, measure aligned global steps, and time DCP save/load."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--measured-steps", type=int, default=500)
    parser.add_argument("--batch-size-per-rank", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--precision", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--parallelism", choices=("fsdp2",), required=True)
    parser.add_argument("--expected-world-size", type=int, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--launch-manifest", type=Path, required=True)
    parser.add_argument("--expected-launch-manifest-sha256", required=True)
    parser.add_argument("--synthetic-only", action="store_true")
    args = parser.parse_args()
    if not args.synthetic_only:
        raise ValueError("--synthetic-only is mandatory")
    if args.warmup_steps != 50 or args.measured_steps != 500:
        raise ValueError("The authoritative profile requires 50 warmup and 500 measured steps")
    if args.batch_size_per_rank < 1:
        raise ValueError("Batch size per rank must be positive")
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
        local_indices = range(
            rank * args.batch_size_per_rank,
            (rank + 1) * args.batch_size_per_rank,
        )
        batch = _synthetic_batch(local_indices, seed=args.seed, device=device)
        model, optimizer = _build_sharded(
            args.seed,
            precision=args.precision,
            device=device,
        )
        for _ in range(args.warmup_steps):
            _train_step(model, optimizer, batch, precision=args.precision)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
        measured_seconds: list[float] = []
        measured_losses: list[float] = []
        for _ in range(args.measured_steps):
            loss, elapsed = _train_step(
                model,
                optimizer,
                batch,
                precision=args.precision,
            )
            measured_losses.append(loss)
            measured_seconds.append(elapsed)
        rank_payload = {
            "rank": rank,
            "step_seconds": measured_seconds,
            "first_loss": measured_losses[0],
            "last_loss": measured_losses[-1],
            "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(device),
        }
        rank_payloads = _gather_rank_objects(rank_payload, world_size)
        before_checkpoint = _full_model_state(model)
        profile_manifest = {
            "format": "frontier_fsdp2_synthetic_profile_fixture_v1",
            "seed": args.seed,
            "world_size": world_size,
            "batch_size_per_rank": args.batch_size_per_rank,
            "warmup_steps": args.warmup_steps,
            "measured_steps": args.measured_steps,
            "antibody_length": 7,
            "antigen_length": 9,
            "biological_data_access": "forbidden",
        }
        provenance = {
            "launch_manifest": args.expected_launch_manifest_sha256,
            "source_tree": observed_source,
            "synthetic_profile": _sha256_json(profile_manifest),
        }
        checkpoint_receipt = save_dcp_generation(
            output_root / "checkpoints",
            "step_000550",
            model=model,
            optimizer=optimizer,
            provenance_hashes=provenance,
            training_state={
                "global_step": args.warmup_steps + args.measured_steps,
                "warmup_steps": args.warmup_steps,
                "measured_steps": args.measured_steps,
                "synthetic_only": True,
            },
        )
        restored, restored_optimizer = _build_sharded(
            args.seed,
            precision=args.precision,
            device=device,
        )
        torch.cuda.synchronize()
        load_started = time.perf_counter()
        loaded_receipt = load_dcp_generation(
            output_root / "checkpoints",
            "step_000550",
            model=restored,
            optimizer=restored_optimizer,
            expected_provenance_hashes=provenance,
        )
        torch.cuda.synchronize()
        load_seconds = time.perf_counter() - load_started
        load_seconds_by_rank = _gather_rank_objects(load_seconds, world_size)
        after_restore = _full_model_state(restored)
        if rank == 0:
            restore_difference = _state_difference(
                after_restore,
                before_checkpoint,
                absolute_tolerance=0.0,
                relative_tolerance=0.0,
            )
            _assert_difference(
                restore_difference,
                absolute_tolerance=0.0,
                relative_tolerance=0.0,
                label="Profile DCP restore",
            )
            per_step_global_seconds = [
                max(float(payload["step_seconds"][index]) for payload in rank_payloads)
                for index in range(args.measured_steps)
            ]
            global_examples = world_size * args.batch_size_per_rank
            global_residue_tokens = global_examples * (7 + 9)
            examples_per_second = [
                global_examples / elapsed for elapsed in per_step_global_seconds
            ]
            residue_tokens_per_second = [
                global_residue_tokens / elapsed for elapsed in per_step_global_seconds
            ]
            result = {
                "format": PROFILE_FORMAT,
                "status": "passed",
                "scope": "reduced synthetic R2 warmed FSDP2 profile",
                "claim_boundary": PROFILE_CLAIM_BOUNDARY,
                "fsdp2_claim_boundary": FSDP2_CLAIM_BOUNDARY,
                "source_sha256": observed_source,
                "launch_manifest_sha256": args.expected_launch_manifest_sha256,
                "synthetic_profile_manifest": profile_manifest,
                "synthetic_profile_manifest_sha256": provenance["synthetic_profile"],
                "runtime": runtime_receipt,
                "precision": args.precision,
                "world_size": world_size,
                "warmup_steps": args.warmup_steps,
                "measured_steps": args.measured_steps,
                "batch_size_per_rank": args.batch_size_per_rank,
                "global_examples_per_step": global_examples,
                "global_residue_tokens_per_step": global_residue_tokens,
                "scaling_coordinate": {
                    "fixed_global_examples": global_examples,
                    "world_size": world_size,
                    "batch_size_per_rank": args.batch_size_per_rank,
                },
                "authoritative_timing": "maximum rank wall time for each aligned optimizer step",
                "step_seconds": _percentile_map(per_step_global_seconds),
                "global_examples_per_second": _percentile_map(examples_per_second),
                "global_residue_tokens_per_second": _percentile_map(residue_tokens_per_second),
                "checkpoint": {
                    "save_seconds_max": checkpoint_receipt["save_seconds_max"],
                    "load_seconds_max": max(float(value) for value in load_seconds_by_rank),
                    "receipt_content_sha256": checkpoint_receipt["content_sha256"],
                    "loaded_receipt_content_sha256": loaded_receipt["content_sha256"],
                    "restore_state_difference": restore_difference,
                },
                "rank_receipts": rank_payloads,
                "outer_or_test_access": False,
                "biological_training_authorized": False,
            }
            publish_create_only_json(output_root / "result.json", result)
            os.chmod(output_root / "result.json", 0o444)
            print(json.dumps({"status": "passed", "world_size": world_size, "measured_steps": args.measured_steps}, sort_keys=True), flush=True)
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
