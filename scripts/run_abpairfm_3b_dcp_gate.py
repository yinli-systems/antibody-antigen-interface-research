#!/usr/bin/env python3
"""Prove fresh-process DCP continuation for the full pair-aware 3B model."""

from __future__ import annotations

import argparse
from datetime import timedelta
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Sequence

import torch

from antibody_pretraining.pair_aware_distributed import (
    fully_shard_pair_aware_esm2,
    load_authenticated_esm2_into_fsdp2,
    optimizer_after_fsdp2,
    reshard_pair_aware_esm2,
)
from antibody_pretraining.pair_aware_esm2 import PairAwareESM2, PairAwareESM2Config
from run_abpairfm_3b_fsdp2_smoke import _synthetic_pair


def _load_frontier_dcp_module() -> Any:
    """Load the reviewed standalone DCP primitive without package side effects."""
    path = Path(__file__).resolve().parents[1] / "src/interaction_first/frontier_fsdp2.py"
    specification = importlib.util.spec_from_file_location(
        "abpairfm_frontier_fsdp2", path
    )
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Cannot load DCP primitive: {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


_FRONTIER_DCP = _load_frontier_dcp_module()
load_dcp_generation = _FRONTIER_DCP.load_dcp_generation
save_dcp_generation = _FRONTIER_DCP.save_dcp_generation


SOURCE_PATHS = (
    "protocols/abpairfm_3b_dcp_recovery_gate_v3.json",
    "scripts/run_abpairfm_3b_dcp_gate.py",
    "scripts/run_abpairfm_3b_fsdp2_smoke.py",
    "src/antibody_pretraining/pair_aware_attention.py",
    "src/antibody_pretraining/pair_aware_distributed.py",
    "src/antibody_pretraining/pair_aware_esm2.py",
    "src/interaction_first/frontier_fsdp2.py",
)
CLAIM_BOUNDARY = (
    "Full-model synthetic FSDP2 and DCP recovery evidence only. Passing does "
    "not authorize biological pretraining, development model selection, test "
    "access, or SOTA claims."
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_sha256(project_root: Path) -> str:
    digest = hashlib.sha256()
    for relative in SOURCE_PATHS:
        path = project_root / relative
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"Reviewed source is missing or not regular: {relative}")
        encoded_name = relative.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(8, "big"))
        digest.update(encoded_name)
        size = path.stat().st_size
        digest.update(size.to_bytes(8, "big"))
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _publish_create_only(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_path, path)
    except FileExistsError as error:
        raise FileExistsError(f"Refusing to overwrite recovery result: {path}") from error
    finally:
        temporary_path.unlink(missing_ok=True)


def _update_tree_digest(digest: Any, value: object) -> None:
    if isinstance(value, torch.Tensor):
        local = value.to_local() if hasattr(value, "to_local") else value
        tensor = local.detach().to(device="cpu").contiguous()
        metadata = json.dumps(
            {"dtype": str(tensor.dtype), "shape": list(tensor.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(b"tensor")
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
        return
    if isinstance(value, dict):
        digest.update(b"dict")
        for key in sorted(value, key=lambda item: (type(item).__name__, repr(item))):
            encoded = f"{type(key).__name__}:{key!r}".encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
            _update_tree_digest(digest, value[key])
        return
    if isinstance(value, (list, tuple)):
        digest.update(type(value).__name__.encode("ascii"))
        digest.update(len(value).to_bytes(8, "big"))
        for item in value:
            _update_tree_digest(digest, item)
        return
    encoded = f"{type(value).__name__}:{value!r}".encode("utf-8")
    digest.update(b"scalar")
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)


def _canonical_local_state_digests(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer
) -> dict[str, str]:
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_state_dict,
    )

    model_state, optimizer_state = get_state_dict(
        model,
        optimizer,
        options=StateDictOptions(
            full_state_dict=False,
            cpu_offload=False,
            strict=True,
        ),
    )
    result: dict[str, str] = {}
    for label, state in (
        ("model", model_state),
        ("optimizer", optimizer_state),
    ):
        digest = hashlib.sha256()
        _update_tree_digest(digest, state)
        result[label] = digest.hexdigest()
    return result


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    digest_parser = subparsers.add_parser("digest")
    digest_parser.add_argument("--project-root", type=Path, required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--mode", choices=("save", "restore"), required=True)
    run_parser.add_argument("--project-root", type=Path, required=True)
    run_parser.add_argument("--checkpoint", type=Path, required=True)
    run_parser.add_argument("--expected-checkpoint-size", type=int, required=True)
    run_parser.add_argument("--expected-checkpoint-sha256", required=True)
    run_parser.add_argument("--expected-source-sha256", required=True)
    run_parser.add_argument("--protocol", type=Path, required=True)
    run_parser.add_argument("--expected-protocol-sha256", required=True)
    run_parser.add_argument("--output-root", type=Path, required=True)
    run_parser.add_argument("--expected-world-size", type=int, default=4)
    run_parser.add_argument("--sequence-tokens", type=int, default=1024)
    run_parser.add_argument("--steps-before-checkpoint", type=int, default=2)
    run_parser.add_argument("--seed", type=int, default=1701)
    run_parser.add_argument("--save-result", type=Path)
    run_parser.add_argument("--expected-save-result-sha256")
    return parser.parse_args(argv)


def _validate_run(arguments: argparse.Namespace) -> None:
    if arguments.expected_world_size != 4:
        raise ValueError("The frozen recovery gate requires exactly four GPUs")
    if arguments.sequence_tokens != 1024:
        raise ValueError("The frozen recovery gate requires 1024 tokens per rank")
    if arguments.steps_before_checkpoint != 2:
        raise ValueError("The frozen recovery gate requires two pre-checkpoint steps")
    for name in (
        "expected_checkpoint_sha256",
        "expected_source_sha256",
        "expected_protocol_sha256",
    ):
        value = getattr(arguments, name)
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"{name} must be a lowercase SHA-256")
    if arguments.mode == "restore" and (
        arguments.save_result is None
        or arguments.expected_save_result_sha256 is None
    ):
        raise ValueError("Restore mode requires an authenticated save result")


def _trace(rank: int, phase: str, **values: object) -> None:
    suffix = " ".join(f"{key}={value}" for key, value in sorted(values.items()))
    print(f"ABPAIRFM_DCP rank={rank} phase={phase} {suffix}".rstrip(), flush=True)


def _build(
    arguments: argparse.Namespace,
    *,
    rank: int,
) -> tuple[PairAwareESM2, torch.optim.AdamW, dict[str, Any]]:
    config = PairAwareESM2Config(activation_checkpointing=True)
    with torch.device("meta"):
        model = PairAwareESM2(config)
    fully_shard_pair_aware_esm2(
        model,
        precision="bfloat16",
        reshard_after_forward=False,
    )
    initialization = load_authenticated_esm2_into_fsdp2(
        model,
        arguments.checkpoint,
        expected_size=arguments.expected_checkpoint_size,
        expected_sha256=arguments.expected_checkpoint_sha256,
    )
    optimizer = optimizer_after_fsdp2(
        model,
        learning_rate=1e-5,
        weight_decay=0.01,
    )
    _trace(rank, "model_optimizer_ready")
    return model, optimizer, initialization


def _train_step(
    model: PairAwareESM2,
    optimizer: torch.optim.AdamW,
    *,
    rank: int,
    world_size: int,
    device: torch.device,
    sequence_tokens: int,
    seed: int,
    step: int,
) -> dict[str, float | int]:
    tokens, chain_ids, labels, residues_per_rank = _synthetic_pair(
        rank=rank,
        sequence_tokens=sequence_tokens,
        seed=seed + step * 100003,
        device=device,
    )
    optimizer.zero_grad(set_to_none=True)
    torch.distributed.barrier()
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    from torch.nn.attention import SDPBackend, sdpa_kernel

    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        output = model(tokens, chain_ids, labels=labels)
        if output.balanced_chain_loss_sum is None:
            raise RuntimeError("MLM model did not return a training loss")
        global_chain_count = output.contributing_chain_count.clone()
        torch.distributed.all_reduce(global_chain_count)
        loss = (
            output.balanced_chain_loss_sum
            * world_size
            / global_chain_count.to(output.balanced_chain_loss_sum.dtype)
        )
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("Non-finite recovery-gate loss")
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if not bool(torch.isfinite(gradient_norm)):
        raise RuntimeError("Non-finite recovery-gate gradient norm")
    optimizer.step()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    global_loss_sum = output.balanced_chain_loss_sum.detach().clone()
    torch.distributed.all_reduce(global_loss_sum)
    global_loss = global_loss_sum / global_chain_count
    record: dict[str, float | int] = {
        "elapsed_seconds": elapsed,
        "global_mean_loss": float(global_loss),
        "gradient_norm": float(gradient_norm),
        "global_residue_tokens": residues_per_rank * world_size,
        "step": step,
    }
    _trace(rank, "step_complete", loss=record["global_mean_loss"], step=step)
    return record


def _gather(value: object, world_size: int) -> list[object]:
    values: list[object] = [None] * world_size
    torch.distributed.all_gather_object(values, value)
    return values


def _read_authenticated_result(path: Path, expected_sha256: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("Save result must be a regular file")
    if _file_sha256(path) != expected_sha256:
        raise RuntimeError("Save result SHA-256 differs from expectation")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("status") != "PASS":
        raise RuntimeError("Save result is not a passing JSON object")
    return value


def _save_mode(
    arguments: argparse.Namespace,
    *,
    rank: int,
    world_size: int,
    device: torch.device,
    source_sha256: str,
    protocol_sha256: str,
) -> None:
    model, optimizer, initialization = _build(arguments, rank=rank)
    pre_checkpoint = [
        _train_step(
            model,
            optimizer,
            rank=rank,
            world_size=world_size,
            device=device,
            sequence_tokens=arguments.sequence_tokens,
            seed=arguments.seed,
            step=step,
        )
        for step in range(arguments.steps_before_checkpoint)
    ]
    fixture = {
        "seed": arguments.seed,
        "sequence_tokens_per_rank": arguments.sequence_tokens,
        "steps_before_checkpoint": arguments.steps_before_checkpoint,
        "world_size": world_size,
    }
    provenance = {
        "official_checkpoint": arguments.expected_checkpoint_sha256,
        "protocol": protocol_sha256,
        "source_tree": source_sha256,
        "synthetic_fixture": _json_sha256(fixture),
    }
    generation = f"step-{arguments.steps_before_checkpoint:06d}"
    reshard_pair_aware_esm2(model)
    _trace(rank, "resharded_before_checkpoint")
    checkpoint_digests = _canonical_local_state_digests(model, optimizer)
    checkpoint_rank_evidence = _gather(
        {"digests": checkpoint_digests, "rank": rank},
        world_size,
    )
    checkpoint_receipt = save_dcp_generation(
        arguments.output_root / "checkpoints",
        generation,
        model=model,
        optimizer=optimizer,
        provenance_hashes=provenance,
        training_state={
            "global_step": arguments.steps_before_checkpoint,
            "next_synthetic_step": arguments.steps_before_checkpoint,
            "seed": arguments.seed,
            "sequence_tokens_per_rank": arguments.sequence_tokens,
            "synthetic_only": True,
        },
    )
    continuation = _train_step(
        model,
        optimizer,
        rank=rank,
        world_size=world_size,
        device=device,
        sequence_tokens=arguments.sequence_tokens,
        seed=arguments.seed,
        step=arguments.steps_before_checkpoint,
    )
    continuation_rank_evidence = _gather(
        {"continuation": continuation, "rank": rank},
        world_size,
    )
    if rank == 0:
        result = {
            "checkpoint": {
                "content_sha256": checkpoint_receipt["content_sha256"],
                "generation": generation,
                "root": str((arguments.output_root / "checkpoints").resolve()),
                "source_world_size": checkpoint_receipt["source_world_size"],
            },
            "claim_boundary": CLAIM_BOUNDARY,
            "fixture": fixture,
            "checkpoint_rank_evidence": checkpoint_rank_evidence,
            "continuation_rank_evidence": continuation_rank_evidence,
            "format_version": "abpairfm_3b_dcp_save_result_v3",
            "initialization": initialization,
            "mode": "save",
            "pre_checkpoint_rank0_steps": pre_checkpoint,
            "provenance_hashes": provenance,
            "status": "PASS",
            "test_access": {
                "asep_test_accessed": False,
                "sabdab2_heldout_accessed": False,
            },
        }
        _publish_create_only(arguments.output_root / "save_result.json", result)
    torch.distributed.barrier()


def _restore_mode(
    arguments: argparse.Namespace,
    *,
    rank: int,
    world_size: int,
    device: torch.device,
    source_sha256: str,
    protocol_sha256: str,
) -> None:
    assert arguments.save_result is not None
    assert arguments.expected_save_result_sha256 is not None
    save_result = _read_authenticated_result(
        arguments.save_result,
        arguments.expected_save_result_sha256,
    )
    if save_result.get("provenance_hashes", {}).get("source_tree") != source_sha256:
        raise RuntimeError("Save and restore source digests differ")
    if save_result.get("provenance_hashes", {}).get("protocol") != protocol_sha256:
        raise RuntimeError("Save and restore protocol digests differ")
    model, optimizer, initialization = _build(arguments, rank=rank)
    checkpoint = save_result["checkpoint"]
    loaded = load_dcp_generation(
        checkpoint["root"],
        checkpoint["generation"],
        model=model,
        optimizer=optimizer,
        expected_provenance_hashes=save_result["provenance_hashes"],
    )
    if loaded["content_sha256"] != checkpoint["content_sha256"]:
        raise RuntimeError("Loaded DCP receipt differs from saved receipt")
    reshard_pair_aware_esm2(model)
    restored_checkpoint_digests = _canonical_local_state_digests(model, optimizer)
    restored_checkpoint_rank_evidence = _gather(
        {"digests": restored_checkpoint_digests, "rank": rank},
        world_size,
    )
    continuation = _train_step(
        model,
        optimizer,
        rank=rank,
        world_size=world_size,
        device=device,
        sequence_tokens=arguments.sequence_tokens,
        seed=arguments.seed,
        step=arguments.steps_before_checkpoint,
    )
    continuation_rank_evidence = _gather(
        {"continuation": continuation, "rank": rank},
        world_size,
    )
    failure: str | None = None
    if rank == 0:
        try:
            expected_checkpoint = save_result["checkpoint_rank_evidence"]
            if (
                len(expected_checkpoint) != world_size
                or len(restored_checkpoint_rank_evidence) != world_size
            ):
                raise RuntimeError("Checkpoint rank evidence cardinality differs")
            for expected_rank, observed_rank in zip(
                expected_checkpoint,
                restored_checkpoint_rank_evidence,
                strict=True,
            ):
                if expected_rank["rank"] != observed_rank["rank"]:
                    raise RuntimeError("Checkpoint rank evidence ordering differs")
                if expected_rank["digests"] != observed_rank["digests"]:
                    raise RuntimeError(
                        "Restored checkpoint-point state digest differs on rank "
                        f"{observed_rank['rank']}"
                    )
            expected_continuation = save_result["continuation_rank_evidence"]
            if (
                len(expected_continuation) != world_size
                or len(continuation_rank_evidence) != world_size
            ):
                raise RuntimeError("Continuation rank evidence cardinality differs")
            for expected_rank, observed_rank in zip(
                expected_continuation,
                continuation_rank_evidence,
                strict=True,
            ):
                if expected_rank["rank"] != observed_rank["rank"]:
                    raise RuntimeError("Continuation rank evidence ordering differs")
        except BaseException as error:
            failure = f"{type(error).__name__}: {error}"
    messages = [failure]
    torch.distributed.broadcast_object_list(messages, src=0)
    if messages[0] is not None:
        raise RuntimeError(messages[0])
    if rank == 0:
        result = {
            "checkpoint": checkpoint,
            "claim_boundary": CLAIM_BOUNDARY,
            "exact_parity": {
                "model_checkpoint_canonical_shard_digest_by_rank": True,
                "optimizer_checkpoint_canonical_shard_digest_by_rank": True,
            },
            "format_version": "abpairfm_3b_dcp_restore_result_v3",
            "initialization": initialization,
            "mode": "restore",
            "nonexact_diagnostics": {
                "continuation_gradient_norm_by_rank": [
                    {
                        "rank": observed["rank"],
                        "restored": observed["continuation"]["gradient_norm"],
                        "uninterrupted": expected["continuation"]["gradient_norm"],
                    }
                    for expected, observed in zip(
                        save_result["continuation_rank_evidence"],
                        continuation_rank_evidence,
                        strict=True,
                    )
                ],
                "continuation_global_loss_by_rank": [
                    {
                        "rank": observed["rank"],
                        "restored": observed["continuation"]["global_mean_loss"],
                        "uninterrupted": expected["continuation"]["global_mean_loss"],
                    }
                    for expected, observed in zip(
                        save_result["continuation_rank_evidence"],
                        continuation_rank_evidence,
                        strict=True,
                    )
                ],
                "continuation_operational_and_finite": True,
                "post_continuation_parameter_digest_gated": False,
                "reason": (
                    "Production Flash SDPA backward is non-deterministic; exact "
                    "parity is enforced at the checkpoint point before continuation, "
                    "while continuation loss and gradient norm remain diagnostics."
                ),
            },
            "restored_checkpoint_rank_evidence": restored_checkpoint_rank_evidence,
            "continuation_rank_evidence": continuation_rank_evidence,
            "save_result_sha256": arguments.expected_save_result_sha256,
            "status": "PASS",
            "test_access": {
                "asep_test_accessed": False,
                "sabdab2_heldout_accessed": False,
            },
        }
        _publish_create_only(arguments.output_root / "restore_result.json", result)
    torch.distributed.barrier()


def main(argv: Sequence[str] | None = None) -> None:
    arguments = _arguments(argv)
    if arguments.command == "digest":
        print(_source_sha256(arguments.project_root.resolve()))
        return
    _validate_run(arguments)
    project_root = arguments.project_root.resolve()
    source_sha256 = _source_sha256(project_root)
    if source_sha256 != arguments.expected_source_sha256:
        raise RuntimeError("Reviewed source digest differs from launch anchor")
    protocol_sha256 = _file_sha256(arguments.protocol.resolve())
    if protocol_sha256 != arguments.expected_protocol_sha256:
        raise RuntimeError("Protocol digest differs from launch anchor")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.distributed.init_process_group(
        backend="nccl",
        device_id=device,
        timeout=timedelta(minutes=15),
    )
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    try:
        if world_size != arguments.expected_world_size:
            raise RuntimeError("Distributed world size differs from the frozen gate")
        properties = torch.cuda.get_device_properties(device)
        if properties.name != "NVIDIA GeForce RTX 4090":
            raise RuntimeError(f"Forbidden GPU model: {properties.name}")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("Selected GPU does not support bfloat16")
        torch.manual_seed(arguments.seed + rank)
        torch.cuda.manual_seed(arguments.seed + rank)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        if rank == 0:
            if arguments.mode == "save":
                arguments.output_root.mkdir(parents=True, exist_ok=False)
            elif (
                arguments.output_root.is_symlink()
                or not arguments.output_root.is_dir()
            ):
                raise RuntimeError("Restore output root is missing or not regular")
        torch.distributed.barrier()
        _trace(rank, "run_start", mode=arguments.mode, world_size=world_size)
        if arguments.mode == "save":
            _save_mode(
                arguments,
                rank=rank,
                world_size=world_size,
                device=device,
                source_sha256=source_sha256,
                protocol_sha256=protocol_sha256,
            )
        else:
            _restore_mode(
                arguments,
                rank=rank,
                world_size=world_size,
                device=device,
                source_sha256=source_sha256,
                protocol_sha256=protocol_sha256,
            )
    finally:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
