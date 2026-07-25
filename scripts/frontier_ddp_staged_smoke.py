#!/usr/bin/env python3
"""Run one process-isolated stage of the synthetic distributed smoke test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.distributed as dist

from distributed_launch_contract import (
    LaunchContractError,
    require_sha256,
    validate_training_mode,
    verify_launch_manifest,
)
from frontier_ddp_smoke import (
    _all_positive_graph,
    _atomic_json,
    _canonical_json,
    _distributed_context,
    _gather_objects,
    _graph_masks,
    _rank_log,
    _sha256_bytes,
    _source_digest,
    _state_dict_digest,
    _synthetic_batch,
    _train_d0,
    _train_r2,
)
from interaction_first.frontier_checkpoint import (
    capture_rng_state,
    create_checkpoint_generation,
    load_checkpoint_generation,
)
from interaction_first.frontier_model import ProtocolD0Retriever
from interaction_first.runtime import set_reproducible_seed


def _manifest(
    *,
    seed: int,
    world_size: int,
    batch_size: int,
    steps: int,
    antibody_dim: int,
    antigen_dim: int,
    positive_graph: tuple[tuple[str, str], ...],
) -> dict[str, object]:
    """Return the identical synthetic contract used by both process stages."""
    return {
        "format": "frontier_staged_synthetic_manifest_v1",
        "seed": seed,
        "world_size": world_size,
        "batch_size_per_rank": batch_size,
        "steps": steps,
        "antibody_dim": antibody_dim,
        "antigen_dim": antigen_dim,
        "positive_pairs": positive_graph,
        "biological_data_access": "forbidden",
    }


def _base_config(
    *,
    stage: str,
    world_size: int,
    steps: int,
    batch_size: int,
    precision: str,
    device: torch.device,
    ddp_initialization: str,
    parallelism: str,
    launch_manifest_sha256: str,
) -> dict[str, object]:
    """Create finite checkpoint metadata shared across stage receipts."""
    return {
        "format": "frontier_staged_ddp_smoke_config_v1",
        "stage": stage,
        "world_size": world_size,
        "steps": steps,
        "batch_size_per_rank": batch_size,
        "precision": precision if device.type == "cuda" else "float32",
        "backend": dist.get_backend() if dist.is_initialized() else "single_process",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device_type": device.type,
        "ddp_initialization": ddp_initialization,
        "parallelism": parallelism,
        "gradient_reduction": "ddp_average",
        "global_loss_gradient_semantics": (
            "D0 uses local_slice_ddp_exact under fixed standard-DDP averaging; "
            "R2 exact means use SUM-reduced numerators and denominators with "
            "world_size/total_count backward scaling."
        ),
        "metric_claim": "forbidden",
        "efficiency_claim": "forbidden",
        "launch_manifest_sha256": launch_manifest_sha256,
    }


def main() -> None:
    """Execute exactly one stage so every DDP model gets a fresh process group."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("d0", "r2"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--coarse-root", type=Path)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--batch-size-per-rank", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument(
        "--precision",
        choices=("float32", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument(
        "--ddp-initialization",
        choices=("default", "verified-equal"),
        default="default",
        help=(
            "Use default DDP synchronization or a hash-gated identical "
            "initialization that safely skips the initial broadcast."
        ),
    )
    parser.add_argument("--parallelism", choices=("ddp",), required=True)
    parser.add_argument("--expected-world-size", type=int, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--launch-manifest", type=Path, required=True)
    parser.add_argument("--expected-launch-manifest-sha256", required=True)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--synthetic-only", action="store_true")
    args = parser.parse_args()
    if not args.synthetic_only:
        raise ValueError("--synthetic-only is mandatory")
    if args.steps <= 0 or args.batch_size_per_rank < 2:
        raise ValueError("steps must be positive and batch size must be at least two")
    if args.stage == "r2" and args.coarse_root is None:
        raise ValueError("R2 requires --coarse-root")
    if args.stage == "d0" and args.coarse_root is not None:
        raise ValueError("D0 cannot receive --coarse-root")

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
    validate_training_mode(
        parallelism=args.parallelism,
        precision=args.precision,
        cuda_available=device.type == "cuda",
        bf16_supported=(
            bool(torch.cuda.is_bf16_supported()) if device.type == "cuda" else False
        ),
        require_cuda=args.require_cuda,
    )
    generation = f"step_{args.steps:06d}"
    try:
        set_reproducible_seed(args.seed + rank, deterministic=True)
        antibody_dim = 48
        antigen_dim = 64
        total_size = world_size * args.batch_size_per_rank
        positive_graph = _all_positive_graph(total_size)
        manifest = _manifest(
            seed=args.seed,
            world_size=world_size,
            batch_size=args.batch_size_per_rank,
            steps=args.steps,
            antibody_dim=antibody_dim,
            antigen_dim=antigen_dim,
            positive_graph=positive_graph,
        )
        manifest_sha256 = _sha256_bytes(_canonical_json(manifest))
        batch = _synthetic_batch(
            rank=rank,
            batch_size=args.batch_size_per_rank,
            seed=args.seed,
            antibody_dim=antibody_dim,
            antigen_dim=antigen_dim,
            positive_graph=positive_graph,
        ).to(device)
        _rank_log(rank, "staged_batch_ready", stage_name=args.stage)

        if args.stage == "d0":
            positive_mask, exclusion_mask = _graph_masks(
                total_size,
                positive_graph,
                device,
            )
            model, history = _train_d0(
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
            model_sha256 = _state_dict_digest(model)
            rng_states = _gather_objects(capture_rng_state(), world_size)
            histories = _gather_objects({"rank": rank, "history": history}, world_size)
            provenance = {
                "source_tree": source_sha256,
                "synthetic_manifest": manifest_sha256,
            }
            if rank == 0:
                receipt = create_checkpoint_generation(
                    args.output_root / "checkpoints",
                    generation,
                    model=model,
                    optimizer=None,
                    scheduler=None,
                    scaler=None,
                    epoch=0,
                    global_step=args.steps,
                    provenance_hashes=provenance,
                    config=_base_config(
                        stage="d0",
                        world_size=world_size,
                        steps=args.steps,
                        batch_size=args.batch_size_per_rank,
                        precision=args.precision,
                        device=device,
                        ddp_initialization="default",
                        parallelism=args.parallelism,
                        launch_manifest_sha256=(
                            args.expected_launch_manifest_sha256
                        ),
                    ),
                    rng_states=rng_states,
                )
                _atomic_json(
                    args.output_root / "result.json",
                    {
                        "format": "frontier_staged_d0_result_v1",
                        "status": "passed",
                        "scope": "synthetic distributed correctness only",
                        "source_sha256": source_sha256,
                        "launch_manifest_sha256": (
                            args.expected_launch_manifest_sha256
                        ),
                        "synthetic_manifest_sha256": manifest_sha256,
                        "model_state_sha256": model_sha256,
                        "checkpoint_receipt": receipt,
                        "rank_histories": histories,
                    },
                )
            if dist.is_initialized():
                dist.barrier()
            payload = load_checkpoint_generation(
                args.output_root / "checkpoints",
                generation,
                expected_provenance_hashes=provenance,
            )
            if payload["checkpoint_sha256"] != payload["receipt"]["checkpoint_sha256"]:
                raise RuntimeError("D0 checkpoint verification failed")
            _rank_log(rank, "d0_handoff_verified", generation=generation)
        else:
            coarse_provenance = {
                "source_tree": source_sha256,
                "synthetic_manifest": manifest_sha256,
            }
            coarse_payload = load_checkpoint_generation(
                args.coarse_root / "checkpoints",
                generation,
                expected_provenance_hashes=coarse_provenance,
            )
            coarse = ProtocolD0Retriever(
                antibody_input_dim=antibody_dim,
                antigen_input_dim=antigen_dim,
                embedding_dim=32,
                projection_hidden_dim=64,
                dropout=0.0,
            )
            coarse.load_state_dict(coarse_payload["model_state"], strict=True)
            coarse_sha256 = _state_dict_digest(coarse)
            _rank_log(
                rank,
                "coarse_checkpoint_loaded",
                checkpoint_sha256=coarse_payload["checkpoint_sha256"],
            )
            model, optimizer, scheduler, history = _train_r2(
                coarse,
                batch,
                steps=args.steps,
                antibody_dim=antibody_dim,
                antigen_dim=antigen_dim,
                device=device,
                local_rank=local_rank,
                precision=args.precision,
                verified_equal_initialization=(
                    args.ddp_initialization == "verified-equal"
                ),
                model_initialization_seed=args.seed + 1,
            )
            final_sha256 = _state_dict_digest(model)
            rng_states = _gather_objects(capture_rng_state(), world_size)
            histories = _gather_objects({"rank": rank, "history": history}, world_size)
            provenance = {
                "coarse_checkpoint": coarse_payload["checkpoint_sha256"],
                "coarse_state": coarse_sha256,
                "source_tree": source_sha256,
                "synthetic_manifest": manifest_sha256,
            }
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
                    config=_base_config(
                        stage="r2",
                        world_size=world_size,
                        steps=args.steps,
                        batch_size=args.batch_size_per_rank,
                        precision=args.precision,
                        device=device,
                        ddp_initialization=args.ddp_initialization,
                        parallelism=args.parallelism,
                        launch_manifest_sha256=(
                            args.expected_launch_manifest_sha256
                        ),
                    ),
                    rng_states=rng_states,
                )
                _atomic_json(
                    args.output_root / "result.json",
                    {
                        "format": "frontier_staged_r2_result_v1",
                        "status": "passed",
                        "scope": "synthetic distributed correctness only",
                        "claim_boundary": "No biological data or performance evidence.",
                        "source_sha256": source_sha256,
                        "launch_manifest_sha256": (
                            args.expected_launch_manifest_sha256
                        ),
                        "synthetic_manifest_sha256": manifest_sha256,
                        "coarse_checkpoint_sha256": coarse_payload[
                            "checkpoint_sha256"
                        ],
                        "coarse_state_sha256": coarse_sha256,
                        "final_model_state_sha256": final_sha256,
                        "checkpoint_receipt": receipt,
                        "rank_histories": histories,
                    },
                )
            if dist.is_initialized():
                dist.barrier()
            payload = load_checkpoint_generation(
                args.output_root / "checkpoints",
                generation,
                expected_provenance_hashes=provenance,
            )
            _rank_log(
                rank,
                "r2_checkpoint_verified",
                checkpoint_sha256=payload["checkpoint_sha256"],
            )
        if rank == 0:
            print(
                json.dumps(
                    {
                        "status": "passed",
                        "stage": args.stage,
                        "world_size": world_size,
                        "source_sha256": source_sha256,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
