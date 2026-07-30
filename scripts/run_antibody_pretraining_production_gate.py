#!/usr/bin/env python3
"""Run a protocol-bound production-corpus loader and recovery gate."""

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

from antibody_pretraining.masking import apply_uniform_mlm
from antibody_pretraining.model import (
    AntibodyMaskedLanguageModel,
    config_from_preset,
    load_model_family,
)
from antibody_pretraining.production_corpus import load_production_corpus
from antibody_pretraining.production_update_plan import build_indexed_update_plan
from antibody_pretraining.tokenizer import PAD_ID
from antibody_pretraining.training_contract import (
    learning_rate_at_token,
    load_training_protocol,
)
import run_antibody_pretraining_optimized as engine


FORMAT_VERSION = "antibody_production_loader_recovery_run_v1"
GATE_FORMAT = "antibody_production_loader_recovery_gate_v1"
TEST_ACCESS = {
    "downstream_labels_used": False,
    "sabdab2_heldout_accessed": False,
}
ROOT = Path(__file__).resolve().parents[1]


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--production-corpus-root", type=Path, required=True)
    parser.add_argument("--production-corpus-manifest-file-sha256", required=True)
    parser.add_argument("--production-corpus-manifest-payload-sha256", required=True)
    parser.add_argument("--production-corpus-protocol-file-sha256", required=True)
    parser.add_argument(
        "--production-corpus-independent-audit",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--production-corpus-independent-audit-file-sha256",
        required=True,
    )
    parser.add_argument(
        "--production-corpus-independent-audit-payload-sha256",
        required=True,
    )
    parser.add_argument("--gate-protocol", type=Path, required=True)
    parser.add_argument("--model-family", type=Path, required=True)
    parser.add_argument("--training-protocol", type=Path, required=True)
    parser.add_argument("--efficiency-protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume-root", type=Path)
    parser.add_argument(
        "--run-role",
        choices=("continuous", "stage1", "resumed"),
        required=True,
    )
    parser.add_argument(
        "--preset",
        choices=("abf_300m", "abf_650m"),
        required=True,
    )
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--micro-batch", type=int, required=True)
    parser.add_argument("--target-tokens", type=int, required=True)
    parser.add_argument("--batch-quantum", type=int, required=True)
    parser.add_argument("--run-seed", type=int, required=True)
    parser.add_argument("--length-bucket-boundaries", required=True)
    parser.add_argument("--ddp-bucket-cap-mb", type=int, default=25)
    parser.add_argument("--compile-blocks", action="store_true")
    parser.add_argument("--disable-compiled-cudagraphs", action="store_true")
    parser.add_argument("--compile-prewarm-buckets", action="store_true")
    parser.add_argument("--fused-adamw", action="store_true")
    parser.add_argument("--pinned-staging", action="store_true")
    parser.add_argument("--bucketed-padding", action="store_true")
    parser.add_argument("--length-bucketing", action="store_true")
    parser.add_argument("--skip-checkpoint", action="store_true")
    args = parser.parse_args()
    if any(
        value <= 0
        for value in (
            args.steps,
            args.micro_batch,
            args.target_tokens,
            args.batch_quantum,
            args.ddp_bucket_cap_mb,
        )
    ):
        raise ValueError("Gate dimensions must be positive")
    if args.run_seed < 0:
        raise ValueError("Run seed must be non-negative")
    if args.compile_prewarm_buckets and not args.compile_blocks:
        raise ValueError("Compile prewarm requires compiled Transformer blocks")
    if args.disable_compiled_cudagraphs and not args.compile_blocks:
        raise ValueError("CUDAGraph control requires compiled Transformer blocks")
    if args.run_role == "resumed" and args.resume_root is None:
        raise ValueError("Resumed role requires a checkpoint root")
    if args.run_role != "resumed" and args.resume_root is not None:
        raise ValueError("Only the resumed role accepts a checkpoint root")
    return args


def _canonical_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def _load_gate_protocol(path: Path) -> tuple[dict[str, object], str]:
    raw = path.read_bytes()
    try:
        protocol = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("Gate protocol must be strict ASCII JSON") from error
    if (
        not isinstance(protocol, dict)
        or protocol.get("format_version") != GATE_FORMAT
        or protocol.get("authorization")
        != {
            "loader_recovery_gate": True,
            "production_pretraining": False,
            "scale_out": False,
            "scientific_claim": False,
        }
        or protocol.get("test_access") != TEST_ACCESS
    ):
        raise RuntimeError("Gate protocol identity or authorization differs")
    return protocol, hashlib.sha256(raw).hexdigest()


def _optimization(args: argparse.Namespace) -> dict[str, object]:
    return {
        "bucketed_padding": args.bucketed_padding,
        "compile_blocks": args.compile_blocks,
        "compile_prewarm_buckets": args.compile_prewarm_buckets,
        "ddp_bucket_cap_mb": args.ddp_bucket_cap_mb,
        "deterministic_algorithms": True,
        "disable_compiled_cudagraphs": args.disable_compiled_cudagraphs,
        "fused_adamw": args.fused_adamw,
        "length_bucketing": args.length_bucketing,
        "pinned_staging": args.pinned_staging,
    }


def _data_contract(args: argparse.Namespace) -> dict[str, object]:
    return {
        "independent_audit_file_sha256": (
            args.production_corpus_independent_audit_file_sha256
        ),
        "independent_audit_payload_sha256": (
            args.production_corpus_independent_audit_payload_sha256
        ),
        "manifest_file_sha256": args.production_corpus_manifest_file_sha256,
        "manifest_payload_sha256": args.production_corpus_manifest_payload_sha256,
        "protocol_file_sha256": args.production_corpus_protocol_file_sha256,
        "split": "train",
    }


def _implementation_contract() -> dict[str, str]:
    """Bind every project source file executed by the isolated gate."""
    files = (
        "scripts/run_antibody_pretraining.py",
        "scripts/run_antibody_pretraining_optimized.py",
        "scripts/run_antibody_pretraining_production_gate.py",
        "src/antibody_pretraining/dense_shard.py",
        "src/antibody_pretraining/masking.py",
        "src/antibody_pretraining/model.py",
        "src/antibody_pretraining/production_corpus.py",
        "src/antibody_pretraining/production_update_plan.py",
        "src/antibody_pretraining/tokenizer.py",
        "src/antibody_pretraining/training_contract.py",
    )
    return {relative: engine._sha256(ROOT / relative) for relative in files}


def _validate_launch(
    args: argparse.Namespace,
    protocol: dict[str, object],
    *,
    gate_protocol_sha256: str,
    model_family_sha256: str,
    training_protocol_sha256: str,
    efficiency_protocol_sha256: str,
    world_size: int,
) -> None:
    run = protocol.get("runs", {}).get(args.run_role)
    expected = {
        "batch_quantum_sequences": args.batch_quantum,
        "checkpoint": not args.skip_checkpoint,
        "micro_batch_per_gpu": args.micro_batch,
        "preset": args.preset,
        "resume": args.resume_root is not None,
        "run_seed": args.run_seed,
        "steps": args.steps,
        "target_residue_tokens": args.target_tokens,
    }
    if run != expected:
        raise RuntimeError("Launch differs from the frozen run role")
    inputs = protocol.get("inputs")
    expected_inputs = {
        **_data_contract(args),
        "efficiency_protocol_file_sha256": efficiency_protocol_sha256,
        "model_family_file_sha256": model_family_sha256,
        "training_protocol_file_sha256": training_protocol_sha256,
    }
    if inputs != expected_inputs:
        raise RuntimeError("Launch input evidence differs from the gate protocol")
    if protocol.get("implementation") != _implementation_contract():
        raise RuntimeError("Executed source differs from the gate protocol")
    execution = protocol.get("execution")
    if execution != {
        "backend": "ddp",
        "length_bucket_boundaries": args.length_bucket_boundaries,
        "optimization": _optimization(args),
        "world_size": world_size,
    }:
        raise RuntimeError("Launch execution settings differ from the gate protocol")
    if len(gate_protocol_sha256) != 64:
        raise AssertionError("Gate protocol digest is invalid")


def _ordered_indices(
    indices: np.ndarray,
    lengths: np.ndarray,
    *,
    enabled: bool,
) -> np.ndarray:
    if not enabled:
        return indices
    selected_lengths = np.asarray(lengths)
    if selected_lengths.shape != indices.shape:
        raise RuntimeError("Length bucketing input differs from update membership")
    positions = np.arange(len(indices), dtype=np.int64)
    ordered = indices[np.lexsort((positions, selected_lengths))]
    if sorted(ordered.tolist()) != sorted(indices.tolist()):
        raise RuntimeError("Length bucketing changed update membership")
    return ordered


def _publish(path: Path, payload: dict[str, object]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_json_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def main() -> None:
    args = _arguments()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size != 1:
        raise RuntimeError("The production loader recovery gate requires one GPU")
    if not torch.cuda.is_available() or local_rank >= torch.cuda.device_count():
        raise RuntimeError("CUDA rank identity is invalid")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if rank == 0:
        if args.output.exists():
            raise RuntimeError("Output must be create-only")
        args.output.mkdir(parents=True)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        timeout=datetime.timedelta(minutes=10),
        device_id=device,
    )
    dist.barrier()
    random.seed(args.run_seed)
    np.random.seed(args.run_seed)
    torch.manual_seed(args.run_seed)
    torch.cuda.manual_seed_all(args.run_seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats(device)

    gate_protocol, gate_protocol_sha256 = _load_gate_protocol(args.gate_protocol)
    family = load_model_family(args.model_family)
    training_protocol = load_training_protocol(args.training_protocol)
    _, efficiency_protocol_sha256 = engine._load_efficiency_protocol(
        args.efficiency_protocol
    )
    model_family_sha256 = engine._sha256(args.model_family)
    training_protocol_sha256 = engine._sha256(args.training_protocol)
    _validate_launch(
        args,
        gate_protocol,
        gate_protocol_sha256=gate_protocol_sha256,
        model_family_sha256=model_family_sha256,
        training_protocol_sha256=training_protocol_sha256,
        efficiency_protocol_sha256=efficiency_protocol_sha256,
        world_size=world_size,
    )
    corpus = load_production_corpus(
        args.production_corpus_root,
        split="train",
        expected_manifest_file_sha256=(
            args.production_corpus_manifest_file_sha256
        ),
        expected_manifest_payload_sha256=(
            args.production_corpus_manifest_payload_sha256
        ),
        expected_protocol_file_sha256=args.production_corpus_protocol_file_sha256,
        independent_audit_path=args.production_corpus_independent_audit,
        expected_independent_audit_file_sha256=(
            args.production_corpus_independent_audit_file_sha256
        ),
        expected_independent_audit_payload_sha256=(
            args.production_corpus_independent_audit_payload_sha256
        ),
    )
    boundaries = engine._parse_boundaries(
        args.length_bucket_boundaries,
        family["maximum_sequence_tokens"],
    )
    engine_args = argparse.Namespace(
        audit_gradient_rank_parity=False,
        backend="ddp",
        compile_blocks=args.compile_blocks,
        compile_prewarm_buckets=args.compile_prewarm_buckets,
        ddp_batched_grad_copy=False,
        ddp_bucket_cap_mb=args.ddp_bucket_cap_mb,
        ddp_communication_dtype="float32",
        disable_compiled_cudagraphs=args.disable_compiled_cudagraphs,
        disable_ddp_gradient_as_bucket_view=False,
        fsdp_output_dtype="bfloat16",
    )
    model = AntibodyMaskedLanguageModel(config_from_preset(family, args.preset)).to(
        device
    )
    model = engine._wrap_model(
        model,
        args=engine_args,
        device=device,
        local_rank=local_rank,
        world_size=world_size,
    )
    optimizer = engine._optimizer(
        model,
        training_protocol,
        args.preset,
        args.fused_adamw,
    )
    module = model.module if hasattr(model, "module") else model
    initial_model_sha256 = engine._replicated_rank_digests(
        module.state_dict(),
        "initial model",
    )[0]
    cursor = 0
    consumed = 0
    start_update = 0
    resume_rng_root = None
    resume_payload_model_sha256 = None
    resume_payload_optimizer_sha256 = None
    resume_loaded_model_sha256 = None
    resume_loaded_optimizer_sha256 = None
    if args.resume_root is not None:
        state = json.loads(
            (args.resume_root / "trainer_state.json").read_text(encoding="ascii")
        )
        expected_state = {
            "batch_quantum": args.batch_quantum,
            "data_source": _data_contract(args),
            "efficiency_protocol_sha256": efficiency_protocol_sha256,
            "gate_protocol_sha256": gate_protocol_sha256,
            "length_bucket_boundaries": list(boundaries),
            "micro_batch_per_gpu": args.micro_batch,
            "model_family_sha256": model_family_sha256,
            "optimization": _optimization(args),
            "preset": args.preset,
            "run_seed": args.run_seed,
            "target_residue_tokens": args.target_tokens,
            "training_protocol_sha256": training_protocol_sha256,
            "training_universe_count": len(corpus),
            "world_size": world_size,
        }
        if any(state.get(key) != value for key, value in expected_state.items()):
            raise RuntimeError("Resume provenance differs from this gate")
        cursor = int(state["sample_cursor"])
        consumed = int(state["consumed_residue_tokens"])
        start_update = int(state["next_update"])
        (
            resume_payload_model_sha256,
            resume_payload_optimizer_sha256,
            _,
            _,
        ) = engine._load_authenticated_dcp(
            model,
            optimizer,
            args.resume_root / "checkpoint",
            backend="ddp",
            rank=rank,
        )
        resume_loaded_model_sha256 = engine._replicated_rank_digests(
            module.state_dict(),
            "resumed model",
        )[0]
        resume_loaded_optimizer_sha256 = engine._replicated_rank_digests(
            optimizer.state_dict(),
            "resumed optimizer",
        )[0]
        resume_rng_root = args.resume_root / "checkpoint"

    compile_prewarm_trace: list[dict[str, float | int]] = []
    if args.compile_prewarm_buckets:
        compile_prewarm_trace = engine._prewarm_compiled_buckets(
            model,
            boundaries=boundaries,
            device=device,
            micro_batch=args.micro_batch,
            world_size=world_size,
        )
    resume_rng_manifest_sha256 = None
    if resume_rng_root is not None:
        resume_rng_manifest_sha256 = engine._restore_rng_checkpoint(
            resume_rng_root,
            rank=rank,
            world_size=world_size,
            device=device,
        )

    membership_digest = hashlib.sha256()
    trace: list[dict[str, float | int]] = []
    data_preparation_seconds = 0.0
    for update in range(start_update, start_update + args.steps):
        preparation_started = time.perf_counter()
        plan = build_indexed_update_plan(
            record_count=len(corpus),
            residue_length_lookup=corpus.residue_lengths,
            sample_cursor=cursor,
            target_residue_tokens=args.target_tokens,
            batch_quantum_sequences=args.batch_quantum,
            seed=args.run_seed,
        )
        compute_indices = _ordered_indices(
            plan.indices,
            corpus.residue_lengths(plan.indices),
            enabled=args.length_bucketing,
        )
        engine._membership_update(
            membership_digest,
            update=update + 1,
            sample_cursor=cursor,
            indices=plan.indices,
        )
        if len(compute_indices) % args.micro_batch:
            raise RuntimeError("Batch quantum must divide the micro batch")
        prepared: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        local_target_count = 0
        for start in range(0, len(compute_indices), args.micro_batch):
            indices = compute_indices[start : start + args.micro_batch]
            batch = corpus.batch(
                indices,
                fields=(
                    "chain_types.npy",
                    "lengths.npy",
                    "sample_keys.npy",
                    "tokens.npy",
                ),
            )
            masked = apply_uniform_mlm(
                batch["tokens.npy"].astype(np.int64, copy=False),
                batch["sample_keys.npy"],
                update=update,
                seed=args.run_seed,
                mask_fraction=training_protocol["masking"]["fraction"],
            )
            maximum_length = int(batch["lengths.npy"].max())
            width = (
                engine._bucket_width(maximum_length, boundaries)
                if args.bucketed_padding or args.length_bucketing
                else maximum_length
            )
            prepared.append(
                (
                    engine._stage_tensor(
                        masked.input_ids[:, :width],
                        args.pinned_staging,
                    ),
                    engine._stage_tensor(
                        masked.labels[:, :width],
                        args.pinned_staging,
                    ),
                    engine._stage_tensor(
                        batch["chain_types.npy"].astype(np.int64, copy=False),
                        args.pinned_staging,
                    ),
                )
            )
            local_target_count += masked.selected_target_count
        target_tensor = torch.tensor(
            local_target_count,
            dtype=torch.long,
            device=device,
        )
        dist.all_reduce(target_tensor, op=dist.ReduceOp.SUM)
        target_count = int(target_tensor)
        if target_count <= 0:
            raise RuntimeError("Update has no selected MLM targets")
        data_preparation_seconds += time.perf_counter() - preparation_started

        learning_rate = learning_rate_at_token(
            consumed,
            peak_learning_rate=training_protocol["presets"][args.preset][
                "learning_rate"
            ],
            warmup_residue_tokens=training_protocol["schedule"][
                "warmup_residue_tokens"
            ],
            total_residue_tokens=training_protocol["schedule"][
                "total_residue_tokens"
            ],
            minimum_learning_rate_ratio=training_protocol["schedule"][
                "minimum_learning_rate_ratio"
            ],
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        started = time.perf_counter()
        loss_sum_value = 0.0
        for micro_index, (input_cpu, labels_cpu, chain_cpu) in enumerate(prepared):
            synchronize = micro_index + 1 == len(prepared)
            sync_context = model.no_sync() if not synchronize else contextlib.nullcontext()
            input_ids = input_cpu.to(device, non_blocking=args.pinned_staging)
            labels = labels_cpu.to(device, non_blocking=args.pinned_staging)
            chain = chain_cpu.to(device, non_blocking=args.pinned_staging)
            attention = input_ids.ne(PAD_ID)
            type_ids = chain[:, None].expand_as(input_ids).masked_fill(~attention, 0)
            with sync_context:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = model(input_ids, type_ids, attention, labels)
                    loss = output.loss_sum / target_count
                loss.backward()
            loss_sum_value += float(output.loss_sum.detach())
        loss_tensor = torch.tensor(loss_sum_value, dtype=torch.float64, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        loss_sum_value = float(loss_tensor)
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                training_protocol["gradient_clip_norm"],
            )
        )
        optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        if not math.isfinite(loss_sum_value) or not math.isfinite(gradient_norm):
            raise RuntimeError("Gate trace contains a non-finite value")
        consumed += plan.residue_token_count
        cursor = plan.next_sample_cursor
        if rank == 0:
            trace.append(
                {
                    "consumed_residue_tokens": consumed,
                    "gradient_norm": gradient_norm,
                    "learning_rate": learning_rate,
                    "loss": loss_sum_value / target_count,
                    "micro_steps": len(prepared),
                    "residue_tokens": plan.residue_token_count,
                    "seconds": elapsed,
                    "selected_targets": target_count,
                    "sequence_count": len(plan.indices),
                    "update": update + 1,
                }
            )

    rng_state = engine._rng_state_payload(rank, device)
    final_model_sha256 = engine._replicated_rank_digests(
        module.state_dict(),
        "final model",
    )[0]
    final_optimizer_sha256 = engine._replicated_rank_digests(
        optimizer.state_dict(),
        "final optimizer",
    )[0]
    checkpoint_model_sha256 = None
    checkpoint_optimizer_sha256 = None
    checkpoint_rng_manifest_sha256 = None
    if not args.skip_checkpoint:
        checkpoint_root = args.output / "checkpoint"
        (
            checkpoint_model_sha256,
            checkpoint_optimizer_sha256,
            _,
            _,
        ) = engine._save_authenticated_dcp(
            model,
            optimizer,
            checkpoint_root,
            backend="ddp",
            rank=rank,
        )
        checkpoint_rng_manifest_sha256 = engine._save_rng_checkpoint(
            checkpoint_root,
            rng_state,
            rank=rank,
            world_size=world_size,
        )
    trainer_state = {
        "batch_quantum": args.batch_quantum,
        "checkpoint_rng_manifest_sha256": checkpoint_rng_manifest_sha256,
        "consumed_residue_tokens": consumed,
        "data_source": _data_contract(args),
        "efficiency_protocol_sha256": efficiency_protocol_sha256,
        "gate_protocol_sha256": gate_protocol_sha256,
        "length_bucket_boundaries": list(boundaries),
        "micro_batch_per_gpu": args.micro_batch,
        "model_family_sha256": model_family_sha256,
        "next_update": start_update + args.steps,
        "optimization": _optimization(args),
        "preset": args.preset,
        "run_seed": args.run_seed,
        "sample_cursor": cursor,
        "target_residue_tokens": args.target_tokens,
        "training_protocol_sha256": training_protocol_sha256,
        "training_universe_count": len(corpus),
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
                    "loader_recovery_gate": True,
                    "production_pretraining": False,
                    "scale_out": False,
                    "scientific_claim": False,
                },
                "checkpoint_model_sha256": checkpoint_model_sha256,
                "checkpoint_optimizer_sha256": checkpoint_optimizer_sha256,
                "checkpoint_rng_manifest_sha256": checkpoint_rng_manifest_sha256,
                "claim_boundary": (
                    "Audited OAS loader and exact-recovery gate only; no "
                    "production-pretraining, scale, biological, or SOTA claim"
                ),
                "compile_prewarm_trace": compile_prewarm_trace,
                "data_preparation_seconds": data_preparation_seconds,
                "data_source": _data_contract(args),
                "device": torch.cuda.get_device_name(device),
                "efficiency_protocol_sha256": efficiency_protocol_sha256,
                "final_model_sha256": final_model_sha256,
                "final_optimizer_sha256": final_optimizer_sha256,
                "format_version": FORMAT_VERSION,
                "gate_protocol_sha256": gate_protocol_sha256,
                "initial_model_sha256": initial_model_sha256,
                "implementation": _implementation_contract(),
                "launch_argv": sys.argv[1:],
                "membership_sha256": membership_digest.hexdigest(),
                "model_family_sha256": model_family_sha256,
                "optimization": _optimization(args),
                "peak_allocated_memory_mib": (
                    torch.cuda.max_memory_allocated(device) / (1024 * 1024)
                ),
                "peak_reserved_memory_mib": (
                    torch.cuda.max_memory_reserved(device) / (1024 * 1024)
                ),
                "preset": args.preset,
                "resume_loaded_model_sha256": resume_loaded_model_sha256,
                "resume_loaded_optimizer_sha256": resume_loaded_optimizer_sha256,
                "resume_payload_model_sha256": resume_payload_model_sha256,
                "resume_payload_optimizer_sha256": (
                    resume_payload_optimizer_sha256
                ),
                "resume_rng_manifest_sha256": resume_rng_manifest_sha256,
                "resume_used": args.resume_root is not None,
                "run_role": args.run_role,
                "runtime": {
                    "cuda": torch.version.cuda,
                    "cudnn": torch.backends.cudnn.version(),
                    "torch": torch.__version__,
                },
                "test_access": TEST_ACCESS,
                "trace": trace,
                "training_protocol_sha256": training_protocol_sha256,
                "training_universe_count": len(corpus),
                "world_size": world_size,
            },
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
