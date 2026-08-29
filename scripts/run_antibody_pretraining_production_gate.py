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
import resource
import socket
import sys
import time

import numpy as np
import torch
import torch.distributed as dist

from antibody_pretraining.masking import apply_uniform_mlm
from antibody_pretraining.elastic_resume import (
    authenticate_source_rng_manifest,
    destination_rank_seed,
    semantic_optimizer_state,
    semantic_trainer_state_sha256,
    validate_randomness_contract,
)
from antibody_pretraining.model import (
    AntibodyMaskedLanguageModel,
    config_from_preset,
    load_model_family,
)
from antibody_pretraining.production_corpus import load_production_corpus
from antibody_pretraining.production_update_plan import (
    build_indexed_update_plan,
    partition_update_indices,
)
from antibody_pretraining.tokenizer import PAD_ID
from antibody_pretraining.training_contract import (
    learning_rate_at_token,
    load_training_protocol,
)
import run_antibody_pretraining_optimized as engine


RECOVERY_RUN_FORMAT = "antibody_production_loader_recovery_run_v1"
RECOVERY_GATE_FORMAT = "antibody_production_loader_recovery_gate_v1"
PRETRAINING_RUN_FORMAT = "antibody_four_gpu_pretraining_segment_run_v1"
PRETRAINING_GATE_FORMAT = "antibody_four_gpu_pretraining_segment_gate_v1"
FORMAT_VERSION = RECOVERY_RUN_FORMAT
GATE_FORMAT = RECOVERY_GATE_FORMAT
ELASTIC_FORMAT_VERSION = "antibody_elastic_recovery_run_v1"
ELASTIC_GATE_FORMAT = "antibody_elastic_recovery_gate_v1"
RECOVERY_AUTHORIZATION = {
    "loader_recovery_gate": True,
    "production_pretraining": False,
    "scale_out": False,
    "scientific_claim": False,
}
PRETRAINING_AUTHORIZATION = {
    "configuration_promotion": True,
    "downstream_transfer": False,
    "production_pretraining": True,
    "scientific_claim": False,
}
ELASTIC_AUTHORIZATION = {
    "elastic_recovery_gate": True,
    "production_pretraining": False,
    "scientific_claim": False,
    "test_access": False,
}
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
        choices=("abf_35m", "abf_300m", "abf_650m"),
        required=True,
    )
    parser.add_argument("--elastic-run-key")
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--micro-batch", type=int, required=True)
    parser.add_argument("--target-tokens", type=int, required=True)
    parser.add_argument("--batch-quantum", type=int, required=True)
    parser.add_argument("--run-seed", type=int, required=True)
    parser.add_argument("--length-bucket-boundaries", required=True)
    parser.add_argument("--ddp-bucket-cap-mb", type=int, default=25)
    parser.add_argument("--maximum-open-arrays", type=int, required=True)
    parser.add_argument("--minimum-soft-open-file-limit", type=int, required=True)
    parser.add_argument("--campaign-token-limit", type=int)
    parser.add_argument("--compile-blocks", action="store_true")
    parser.add_argument("--disable-compiled-cudagraphs", action="store_true")
    parser.add_argument("--compile-prewarm-buckets", action="store_true")
    parser.add_argument("--fused-adamw", action="store_true")
    parser.add_argument("--pinned-staging", action="store_true")
    parser.add_argument("--bucketed-padding", action="store_true")
    parser.add_argument("--length-bucketing", action="store_true")
    parser.add_argument("--skip-checkpoint", action="store_true")
    parser.add_argument(
        "--authorization-mode",
        choices=("recovery_gate", "production_pretraining"),
        default="recovery_gate",
    )
    args = parser.parse_args()
    if any(
        value <= 0
        for value in (
            args.steps,
            args.micro_batch,
            args.target_tokens,
            args.batch_quantum,
            args.ddp_bucket_cap_mb,
            args.maximum_open_arrays,
            args.minimum_soft_open_file_limit,
        )
    ):
        raise ValueError("Gate dimensions must be positive")
    if args.run_seed < 0:
        raise ValueError("Run seed must be non-negative")
    if args.campaign_token_limit is not None and args.campaign_token_limit <= 0:
        raise ValueError("Campaign token limit must be positive")
    if args.minimum_soft_open_file_limit < args.maximum_open_arrays + 512:
        raise ValueError("Open-file limit does not preserve descriptor headroom")
    if args.compile_prewarm_buckets and not args.compile_blocks:
        raise ValueError("Compile prewarm requires compiled Transformer blocks")
    if args.disable_compiled_cudagraphs and not args.compile_blocks:
        raise ValueError("CUDAGraph control requires compiled Transformer blocks")
    if args.run_role == "resumed" and args.resume_root is None:
        raise ValueError("Resumed role requires a checkpoint root")
    if args.run_role != "resumed" and args.resume_root is not None:
        raise ValueError("Only the resumed role accepts a checkpoint root")
    if args.elastic_run_key is not None and not args.elastic_run_key.strip():
        raise ValueError("Elastic run key cannot be empty")
    if args.elastic_run_key is not None and args.authorization_mode != "recovery_gate":
        raise ValueError("Elastic recovery cannot authorize production pretraining")
    if (
        args.campaign_token_limit is not None
        and args.authorization_mode != "production_pretraining"
    ):
        raise ValueError("Campaign token limit requires production authorization")
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


def _load_canonical_json(path: Path, label: str) -> dict[str, object]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"{label} must be a regular file")
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{label} must be strict ASCII JSON") from error
    if not isinstance(value, dict) or raw != _canonical_json_bytes(value):
        raise RuntimeError(f"{label} must be a canonical JSON object")
    return value


def _mode_contract(
    authorization_mode: str,
) -> tuple[str, str, dict[str, bool], str]:
    if authorization_mode == "recovery_gate":
        return (
            RECOVERY_GATE_FORMAT,
            RECOVERY_RUN_FORMAT,
            RECOVERY_AUTHORIZATION,
            "Audited OAS loader and exact-recovery gate only; no "
            "production-pretraining, scale, biological, or SOTA claim",
        )
    if authorization_mode == "production_pretraining":
        return (
            PRETRAINING_GATE_FORMAT,
            PRETRAINING_RUN_FORMAT,
            PRETRAINING_AUTHORIZATION,
            "Four-GPU OAS masked-language-model pretraining segment; no "
            "downstream, partner-specificity, official-test, biological, "
            "model-promotion, or SOTA claim",
        )
    raise ValueError("Unknown authorization mode")


def _load_gate_protocol(
    path: Path,
    elastic_run_key: str | None = None,
    authorization_mode: str = "recovery_gate",
) -> tuple[dict[str, object], str, bool]:
    raw = path.read_bytes()
    try:
        protocol = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("Gate protocol must be strict ASCII JSON") from error
    if elastic_run_key is not None:
        if raw != _canonical_json_bytes(protocol):
            raise RuntimeError("Elastic gate protocol must be canonical JSON")
        if (
            not isinstance(protocol, dict)
            or protocol.get("format_version") != ELASTIC_GATE_FORMAT
            or protocol.get("authorization") != ELASTIC_AUTHORIZATION
            or protocol.get("test_access") != TEST_ACCESS
            or elastic_run_key not in protocol.get("runs", {})
        ):
            raise RuntimeError("Elastic gate protocol identity differs")
        return protocol, hashlib.sha256(raw).hexdigest(), True
    gate_format, _, authorization, _ = _mode_contract(authorization_mode)
    if (
        not isinstance(protocol, dict)
        or protocol.get("format_version") != gate_format
        or protocol.get("authorization") != authorization
        or protocol.get("test_access") != TEST_ACCESS
    ):
        raise RuntimeError("Gate protocol identity or authorization differs")
    return protocol, hashlib.sha256(raw).hexdigest(), False


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


def _data_loader_contract(args: argparse.Namespace) -> dict[str, object]:
    return {
        "grouped_update_read": True,
        "maximum_open_arrays": args.maximum_open_arrays,
        "minimum_soft_open_file_limit": args.minimum_soft_open_file_limit,
    }


def _open_file_limits() -> tuple[int, int]:
    """Return finite receipt-safe soft and hard descriptor limits."""
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    normalized = tuple(
        sys.maxsize if value == resource.RLIM_INFINITY else int(value)
        for value in (soft, hard)
    )
    return normalized[0], normalized[1]


def _implementation_contract(*, include_elastic: bool = False) -> dict[str, str]:
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
    if include_elastic:
        files = (*files, "src/antibody_pretraining/elastic_resume.py")
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
    elastic_run_key: str | None,
    resume_source_world_size: int | None,
) -> None:
    run_key = elastic_run_key if elastic_run_key is not None else args.run_role
    run = protocol.get("runs", {}).get(run_key)
    expected: dict[str, object] = {
        "batch_quantum_sequences": args.batch_quantum,
        "checkpoint": not args.skip_checkpoint,
        "micro_batch_per_gpu": args.micro_batch,
        "preset": args.preset,
        "resume": args.resume_root is not None,
        "run_seed": args.run_seed,
        "steps": args.steps,
        "target_residue_tokens": args.target_tokens,
    }
    if elastic_run_key is not None:
        expected.update(
            {
                "role": args.run_role,
                "source_world_size": resume_source_world_size,
                "world_size": world_size,
            }
        )
    elif args.campaign_token_limit is not None:
        expected["campaign_token_limit"] = args.campaign_token_limit
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
    if protocol.get("implementation") != _implementation_contract(
        include_elastic=elastic_run_key is not None
    ):
        raise RuntimeError("Executed source differs from the gate protocol")
    expected_execution = {
        "backend": "ddp",
        "data_loader": _data_loader_contract(args),
        "length_bucket_boundaries": args.length_bucket_boundaries,
        "optimization": _optimization(args),
    }
    if elastic_run_key is None:
        expected_execution["world_size"] = world_size
    if protocol.get("execution") != expected_execution:
        raise RuntimeError("Launch execution settings differ from the gate protocol")
    if elastic_run_key is not None:
        corpus_verification = protocol.get("corpus")
        if (
            not isinstance(corpus_verification, dict)
            or corpus_verification.get("array_file_hashes_verified_during_freeze")
            is not True
            or corpus_verification.get("stage_revalidation")
            != "immutable_tree_manifest_size_dtype_shape"
        ):
            raise RuntimeError("Elastic corpus verification handoff differs")
        topology = protocol.get("topology")
        if (
            not isinstance(topology, dict)
            or world_size not in topology.get("allowed_world_sizes", [])
            or topology.get("node_count") != 1
        ):
            raise RuntimeError("Elastic launch topology differs from the gate")
        transition = [resume_source_world_size, world_size]
        if (
            resume_source_world_size is not None
            and transition not in topology.get("allowed_transitions", [])
        ):
            raise RuntimeError("Elastic world-size transition is not frozen")
    soft_open_file_limit, _ = _open_file_limits()
    if soft_open_file_limit < args.minimum_soft_open_file_limit:
        raise RuntimeError("Soft open-file limit is below the frozen minimum")
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
    resume_state: dict[str, object] | None = None
    resume_source_world_size: int | None = None
    if args.resume_root is not None:
        resume_state_path = args.resume_root / "trainer_state.json"
        resume_state = _load_canonical_json(
            resume_state_path,
            "Resume trainer state",
        )
        resume_source_world_size = int(resume_state.get("world_size", 0))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", str(world_size)))
    if world_size <= 0 or world_size != local_world_size:
        raise RuntimeError("The production loader gate requires one local GPU node")
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
    rank_hosts: list[str | None] = [None] * world_size
    dist.all_gather_object(rank_hosts, socket.gethostname())
    if len(set(rank_hosts)) != 1:
        raise RuntimeError("The production loader gate requires one shared host")
    random.seed(args.run_seed)
    np.random.seed(args.run_seed)
    torch.manual_seed(args.run_seed)
    torch.cuda.manual_seed_all(args.run_seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats(device)

    (
        gate_protocol,
        gate_protocol_sha256,
        elastic_gate,
    ) = _load_gate_protocol(
        args.gate_protocol,
        args.elastic_run_key,
        args.authorization_mode,
    )
    family = load_model_family(args.model_family)
    training_protocol = load_training_protocol(args.training_protocol)
    if elastic_gate:
        validate_randomness_contract(
            family,
            masking_implementation=(
                "counter_derived_by_sample_key_update_and_seed"
            ),
        )
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
        elastic_run_key=args.elastic_run_key,
        resume_source_world_size=resume_source_world_size,
    )
    corpus_arguments = {
        "split": "train",
        "expected_manifest_file_sha256": (
            args.production_corpus_manifest_file_sha256
        ),
        "expected_manifest_payload_sha256": (
            args.production_corpus_manifest_payload_sha256
        ),
        "expected_protocol_file_sha256": (
            args.production_corpus_protocol_file_sha256
        ),
        "independent_audit_path": args.production_corpus_independent_audit,
        "expected_independent_audit_file_sha256": (
            args.production_corpus_independent_audit_file_sha256
        ),
        "expected_independent_audit_payload_sha256": (
            args.production_corpus_independent_audit_payload_sha256
        ),
        "maximum_open_arrays": args.maximum_open_arrays,
    }
    corpus = None
    if local_rank == 0:
        corpus = load_production_corpus(
            args.production_corpus_root,
            **corpus_arguments,
            verify_array_file_hashes=not elastic_gate,
        )
    dist.barrier()
    if local_rank != 0:
        corpus = load_production_corpus(
            args.production_corpus_root,
            **corpus_arguments,
            verify_array_file_hashes=False,
        )
    if corpus is None:
        raise AssertionError("Production corpus was not opened")
    expected_hash_mode = local_rank == 0 and not elastic_gate
    if corpus.contract.array_file_hashes_verified is not expected_hash_mode:
        raise RuntimeError("Distributed corpus hash-verification mode differs")
    corpus_counts: list[int | None] = [None] * world_size
    dist.all_gather_object(corpus_counts, len(corpus))
    if len(set(corpus_counts)) != 1:
        raise RuntimeError("Distributed corpus record counts differ")
    dist.barrier()
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
    resume_loaded_semantic_optimizer_sha256 = None
    resume_receipt: dict[str, object] | None = None
    resume_semantic_state_sha256 = None
    if args.resume_root is not None:
        if resume_state is None:
            raise AssertionError("Resume state was not loaded")
        state = resume_state
        expected_state = {
            "batch_quantum": args.batch_quantum,
            "data_source": _data_contract(args),
            "data_loader": _data_loader_contract(args),
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
        }
        if not elastic_gate:
            expected_state["world_size"] = world_size
        if any(state.get(key) != value for key, value in expected_state.items()):
            raise RuntimeError("Resume provenance differs from this gate")
        if elastic_gate:
            resume_semantic_state_sha256 = semantic_trainer_state_sha256(state)
            resume_receipt = _load_canonical_json(
                args.resume_root / "receipt.json",
                "Elastic resume receipt",
            )
            elastic_receipt = resume_receipt.get("elastic_resume")
            if (
                resume_receipt.get("format_version") != ELASTIC_FORMAT_VERSION
                or resume_receipt.get("authorization") != ELASTIC_AUTHORIZATION
                or resume_receipt.get("gate_protocol_sha256")
                != gate_protocol_sha256
                or resume_receipt.get("checkpoint_model_sha256")
                != resume_receipt.get("final_model_sha256")
                or resume_receipt.get("checkpoint_rng_manifest_sha256")
                != state.get("checkpoint_rng_manifest_sha256")
                or not isinstance(elastic_receipt, dict)
                or elastic_receipt.get("destination_semantic_state_sha256")
                != resume_semantic_state_sha256
            ):
                raise RuntimeError("Elastic resume receipt authentication failed")
        else:
            resume_receipt = _load_canonical_json(
                args.resume_root / "receipt.json",
                "Resume receipt",
            )
            _, expected_resume_format, expected_authorization, _ = _mode_contract(
                args.authorization_mode
            )
            if (
                resume_receipt.get("format_version") != expected_resume_format
                or resume_receipt.get("authorization") != expected_authorization
                or resume_receipt.get("gate_protocol_sha256")
                != gate_protocol_sha256
                or resume_receipt.get("checkpoint_model_sha256")
                != resume_receipt.get("final_model_sha256")
                or resume_receipt.get("checkpoint_optimizer_sha256") is None
                or resume_receipt.get("checkpoint_rng_manifest_sha256")
                != state.get("checkpoint_rng_manifest_sha256")
                or resume_receipt.get("test_access") != TEST_ACCESS
            ):
                raise RuntimeError("Resume receipt differs from this gate")
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
        resume_loaded_semantic_optimizer_sha256 = engine._replicated_rank_digests(
            semantic_optimizer_state(module, optimizer),
            "resumed named optimizer",
        )[0]
        if elastic_gate and (
            resume_payload_model_sha256
            != resume_receipt.get("checkpoint_model_sha256")
            or resume_payload_optimizer_sha256
            != resume_receipt.get("checkpoint_optimizer_sha256")
            or resume_loaded_model_sha256 != resume_receipt.get("final_model_sha256")
            or resume_loaded_optimizer_sha256
            != resume_receipt.get("final_optimizer_sha256")
            or resume_loaded_semantic_optimizer_sha256
            != resume_receipt.get("final_semantic_optimizer_sha256")
        ):
            raise RuntimeError("Elastic DCP payload differs from its source receipt")
        if not elastic_gate and (
            resume_receipt is None
            or resume_payload_model_sha256
            != resume_receipt.get("checkpoint_model_sha256")
            or resume_payload_optimizer_sha256
            != resume_receipt.get("checkpoint_optimizer_sha256")
            or resume_loaded_model_sha256 != resume_receipt.get("final_model_sha256")
            or resume_loaded_optimizer_sha256
            != resume_receipt.get("final_optimizer_sha256")
        ):
            raise RuntimeError(
                "Resume checkpoint differs from the authenticated receipt"
            )
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
    elastic_source_rng_authentication = None
    elastic_destination_rank_seeds = None
    if resume_rng_root is not None:
        if elastic_gate and resume_source_world_size != world_size:
            elastic_source_rng_authentication = authenticate_source_rng_manifest(
                resume_rng_root,
                expected_manifest_sha256=str(
                    state["checkpoint_rng_manifest_sha256"]
                ),
            )
            if (
                elastic_source_rng_authentication["source_world_size"]
                != resume_source_world_size
            ):
                raise RuntimeError("Elastic RNG source topology differs")
            rank_seed = destination_rank_seed(
                run_seed=args.run_seed,
                next_update=start_update,
                rank=rank,
                world_size=world_size,
            )
            random.seed(rank_seed)
            np.random.seed(rank_seed % (1 << 32))
            torch.manual_seed(rank_seed)
            torch.cuda.manual_seed(rank_seed)
            gathered_seeds: list[int | None] = [None] * world_size
            dist.all_gather_object(gathered_seeds, rank_seed)
            if any(value is None for value in gathered_seeds):
                raise RuntimeError("Elastic destination rank seed inventory differs")
            elastic_destination_rank_seeds = [
                int(value) for value in gathered_seeds if value is not None
            ]
            resume_rng_manifest_sha256 = str(
                elastic_source_rng_authentication["manifest_sha256"]
            )
        else:
            resume_rng_manifest_sha256 = engine._restore_rng_checkpoint(
                resume_rng_root,
                rank=rank,
                world_size=world_size,
                device=device,
            )
            if (
                not elastic_gate
                and resume_receipt is not None
                and resume_rng_manifest_sha256
                != resume_receipt.get("checkpoint_rng_manifest_sha256")
            ):
                raise RuntimeError(
                    "Resume RNG state differs from the authenticated receipt"
                )

    membership_digest = hashlib.sha256()
    trace: list[dict[str, float | int]] = []
    data_preparation_seconds = 0.0
    executed_steps = 0
    for update in range(start_update, start_update + args.steps):
        if (
            args.campaign_token_limit is not None
            and consumed >= args.campaign_token_limit
        ):
            break
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
        local_micro_batches = partition_update_indices(
            compute_indices,
            micro_batch_per_rank=args.micro_batch,
            world_size=world_size,
            rank=rank,
        )
        prepared: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        local_target_count = 0
        grouped_batches = corpus.batch_groups(
            local_micro_batches,
            fields=(
                "chain_types.npy",
                "lengths.npy",
                "sample_keys.npy",
                "tokens.npy",
            ),
        )
        for batch in grouped_batches:
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
                    loss = output.loss_sum * world_size / target_count
                loss.backward()
            loss_sum_value += float(output.loss_sum.detach())
        local_loss_sum_value = loss_sum_value
        loss_tensor = torch.tensor(loss_sum_value, dtype=torch.float64, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        loss_sum_value = float(loss_tensor)
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                training_protocol["gradient_clip_norm"],
            )
        )
        if not math.isfinite(loss_sum_value) or not math.isfinite(gradient_norm):
            nonfinite_gradient_names: list[str] = []
            maximum_gradient_magnitude = 0.0
            for name, parameter in model.named_parameters():
                gradient = parameter.grad
                if gradient is None:
                    continue
                if not bool(torch.isfinite(gradient).all().item()):
                    if len(nonfinite_gradient_names) < 32:
                        nonfinite_gradient_names.append(name)
                    continue
                maximum_gradient_magnitude = max(
                    maximum_gradient_magnitude,
                    float(gradient.detach().abs().max()),
                )
            local_diagnostic = {
                "global_gradient_norm": gradient_norm,
                "global_loss_sum": loss_sum_value,
                "local_loss_sum": local_loss_sum_value,
                "maximum_finite_gradient_magnitude": maximum_gradient_magnitude,
                "nonfinite_gradient_names": nonfinite_gradient_names,
                "rank": rank,
                "update": update + 1,
            }
            rank_diagnostics: list[object] = [None] * world_size
            dist.all_gather_object(rank_diagnostics, local_diagnostic)
            if rank == 0:
                print(
                    "NONFINITE_GATE_DIAGNOSTIC="
                    + json.dumps(
                        rank_diagnostics,
                        allow_nan=True,
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                    flush=True,
                )
            raise RuntimeError(
                "Gate trace contains a non-finite value before optimizer.step "
                f"at update {update + 1}"
            )
        optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        consumed += plan.residue_token_count
        cursor = plan.next_sample_cursor
        executed_steps += 1
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

    if executed_steps <= 0:
        raise RuntimeError("Segment would execute zero optimizer updates")

    rng_state = engine._rng_state_payload(rank, device)
    final_model_sha256 = engine._replicated_rank_digests(
        module.state_dict(),
        "final model",
    )[0]
    final_optimizer_sha256 = engine._replicated_rank_digests(
        optimizer.state_dict(),
        "final optimizer",
    )[0]
    final_semantic_optimizer_sha256 = engine._replicated_rank_digests(
        semantic_optimizer_state(module, optimizer),
        "final named optimizer",
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
        "data_loader": _data_loader_contract(args),
        "efficiency_protocol_sha256": efficiency_protocol_sha256,
        "gate_protocol_sha256": gate_protocol_sha256,
        "length_bucket_boundaries": list(boundaries),
        "micro_batch_per_gpu": args.micro_batch,
        "model_family_sha256": model_family_sha256,
        "next_update": start_update + executed_steps,
        "optimization": _optimization(args),
        "preset": args.preset,
        "run_seed": args.run_seed,
        "sample_cursor": cursor,
        "target_residue_tokens": args.target_tokens,
        "training_protocol_sha256": training_protocol_sha256,
        "training_universe_count": len(corpus),
        "world_size": world_size,
    }
    destination_semantic_state_sha256 = (
        semantic_trainer_state_sha256(trainer_state) if elastic_gate else None
    )
    if elastic_gate:
        run_format = ELASTIC_FORMAT_VERSION
        authorization = ELASTIC_AUTHORIZATION
        claim_boundary = (
            "Real-OAS elastic recovery systems gate only; no "
            "production-pretraining, downstream, biological, test, "
            "model-selection, or SOTA claim"
        )
    else:
        _, run_format, authorization, claim_boundary = _mode_contract(
            args.authorization_mode
        )
    if rank == 0:
        soft_open_file_limit, hard_open_file_limit = _open_file_limits()
        _publish(args.output / "trainer_state.json", trainer_state)
        _publish(
            args.output / "receipt.json",
            {
                "allocation": {
                    "hostname": socket.gethostname(),
                    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                    "slurm_job_nodelist": os.environ.get("SLURM_JOB_NODELIST"),
                },
                "authorization": authorization,
                "campaign": {
                    "complete": (
                        args.campaign_token_limit is not None
                        and consumed >= args.campaign_token_limit
                    ),
                    "configured_maximum_steps": args.steps,
                    "executed_steps": executed_steps,
                    "token_limit": args.campaign_token_limit,
                },
                "checkpoint_model_sha256": checkpoint_model_sha256,
                "checkpoint_optimizer_sha256": checkpoint_optimizer_sha256,
                "checkpoint_rng_manifest_sha256": checkpoint_rng_manifest_sha256,
                "checkpoint_state_hash_domains": {
                    "model": "torch_distributed_checkpoint_canonical_state_dict_v1",
                    "optimizer": (
                        "torch_distributed_checkpoint_canonical_optimizer_state_dict_v1"
                    ),
                    "rng": "authenticated_rank_rng_manifest_v1",
                },
                "claim_boundary": claim_boundary,
                "compile_prewarm_trace": compile_prewarm_trace,
                "corpus_open": {
                    "array_file_hashes_verified_by_local_rank_zero": (
                        not elastic_gate
                    ),
                    "array_file_hashes_verified_during_gate_freeze": (
                        elastic_gate
                    ),
                    "follower_validation": "size_dtype_shape_after_barrier",
                    "hard_open_file_limit": hard_open_file_limit,
                    "maximum_open_arrays": args.maximum_open_arrays,
                    "minimum_soft_open_file_limit": (
                        args.minimum_soft_open_file_limit
                    ),
                    "node_count": 1,
                    "rank_hosts": rank_hosts,
                    "soft_open_file_limit": soft_open_file_limit,
                    "stage_revalidation": (
                        "immutable_tree_manifest_size_dtype_shape"
                        if elastic_gate
                        else "rank_zero_full_hash_then_size_dtype_shape"
                    ),
                },
                "data_loader": _data_loader_contract(args),
                "data_preparation_seconds": data_preparation_seconds,
                "data_source": _data_contract(args),
                "device": torch.cuda.get_device_name(device),
                "efficiency_protocol_sha256": efficiency_protocol_sha256,
                "elastic_resume": (
                    {
                        "cross_world": (
                            resume_source_world_size is not None
                            and resume_source_world_size != world_size
                        ),
                        "destination_rank_seeds": (
                            elastic_destination_rank_seeds
                        ),
                        "destination_semantic_state_sha256": (
                            destination_semantic_state_sha256
                        ),
                        "destination_world_size": world_size,
                        "enabled": True,
                        "rng_policy": (
                            "authenticate_source_then_topology_reseed"
                            if resume_source_world_size is not None
                            and resume_source_world_size != world_size
                            else "initial_seed_or_exact_same_world_restore"
                        ),
                        "run_key": args.elastic_run_key,
                        "source_rng_authentication": (
                            elastic_source_rng_authentication
                        ),
                        "source_semantic_state_sha256": (
                            resume_semantic_state_sha256
                        ),
                        "source_world_size": resume_source_world_size,
                    }
                    if elastic_gate
                    else None
                ),
                "final_model_sha256": final_model_sha256,
                "final_optimizer_sha256": final_optimizer_sha256,
                "final_semantic_optimizer_sha256": (
                    final_semantic_optimizer_sha256
                ),
                "final_state_hash_domains": {
                    "model": "unwrapped_module_state_dict_v1",
                    "optimizer": "native_optimizer_state_dict_v1",
                },
                "format_version": run_format,
                "gate_protocol_sha256": gate_protocol_sha256,
                "initial_model_sha256": initial_model_sha256,
                "implementation": _implementation_contract(
                    include_elastic=elastic_gate
                ),
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
                "resume_loaded_semantic_optimizer_sha256": (
                    resume_loaded_semantic_optimizer_sha256
                ),
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
