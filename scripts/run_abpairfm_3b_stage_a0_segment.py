#!/usr/bin/env python3
"""Run one auditable, resumable 3B Stage-A0 production segment."""

from __future__ import annotations

import argparse
from datetime import timedelta
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import socket
import statistics
import sys
import time
from typing import Any

import torch

from antibody_pretraining.pair_aware_distributed import (
    fully_shard_pair_aware_esm2,
    load_authenticated_esm2_into_fsdp2,
    optimizer_after_fsdp2,
    reshard_pair_aware_esm2,
)
from antibody_pretraining.pair_aware_esm2 import PairAwareESM2, PairAwareESM2Config


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

from run_abpairfm_3b_real_data_gate_v2 import (  # noqa: E402
    _build_real_microbatch,
    _collect,
    _nearest_rank_p95,
    _publish_create_only,
    _publish_progress,
    _sha256,
    _zstd_lines,
)


CLAIM_BOUNDARY = (
    "User-authorized production Stage-A0 pair-aware MLM warm-start over licensed, "
    "strictly development-decontaminated STRING physical pairs. This is resumable "
    "real pretraining, but not the complete Stage-A contact/retrieval objective and "
    "not development, test, biological, or SOTA evidence."
)
SOURCE_PATHS = (
    "scripts/run_abpairfm_3b_stage_a0_segment.py",
    "scripts/run_abpairfm_3b_real_data_gate_v2.py",
    "src/antibody_pretraining/pair_aware_attention.py",
    "src/antibody_pretraining/pair_aware_distributed.py",
    "src/antibody_pretraining/pair_aware_esm2.py",
    "src/interaction_first/frontier_fsdp2.py",
)


def _frontier_module() -> Any:
    path = Path(__file__).resolve().parents[1] / "src/interaction_first/frontier_fsdp2.py"
    specification = importlib.util.spec_from_file_location("abpairfm_stage_a0_dcp", path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Cannot load DCP implementation: {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


_DCP = _frontier_module()


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--campaign-protocol", type=Path, required=True)
    parser.add_argument("--expected-campaign-protocol-sha256", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-size", type=int, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-receipt", type=Path, required=True)
    parser.add_argument("--segment-index", type=int, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--input-checkpoint-root", type=Path)
    parser.add_argument("--input-generation")
    parser.add_argument("--expected-input-content-sha256")
    parser.add_argument("--expected-world-size", type=int, default=8)
    parser.add_argument("--shard-group-size", type=int, default=4)
    parser.add_argument("--microbatch-pairs", type=int, default=7)
    parser.add_argument("--accumulation-steps", type=int, default=2)
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--maximum-tokens", type=int, default=1024)
    parser.add_argument("--maximum-learning-rate", type=float, default=1e-6)
    parser.add_argument("--warmup-residue-tokens", type=int, default=100_000_000)
    parser.add_argument("--target-residue-tokens", type=int, default=5_000_000_000)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--progress-interval", type=int, default=10)
    return parser.parse_args()


def _validate(arguments: argparse.Namespace) -> None:
    if arguments.expected_world_size != 8 or arguments.shard_group_size != 4:
        raise ValueError("Stage-A0 production requires the promoted HSDP 2x4 mesh")
    if arguments.microbatch_pairs != 7 or arguments.accumulation_steps != 2:
        raise ValueError("Stage-A0 production batch differs from the promoted gate")
    if arguments.updates != 200 or arguments.maximum_tokens != 1024:
        raise ValueError("Stage-A0 segment shape differs from the promoted gate")
    if arguments.maximum_learning_rate != 1e-6 or arguments.weight_decay != 0.01:
        raise ValueError("Stage-A0 optimizer differs from the frozen campaign")
    if arguments.seed != 1701 or arguments.segment_index < 0:
        raise ValueError("Stage-A0 seed or segment index differs")
    if not 0 < arguments.warmup_residue_tokens < arguments.target_residue_tokens:
        raise ValueError("Stage-A0 token schedule is invalid")
    resume = (
        arguments.input_checkpoint_root,
        arguments.input_generation,
        arguments.expected_input_content_sha256,
    )
    if any(value is not None for value in resume) != all(value is not None for value in resume):
        raise ValueError("Checkpoint continuation arguments must be supplied together")
    if arguments.segment_index == 0 and any(value is not None for value in resume):
        raise ValueError("The first Stage-A0 segment cannot restore a campaign checkpoint")
    if arguments.segment_index > 0 and not all(value is not None for value in resume):
        raise ValueError("A successor Stage-A0 segment requires an authenticated checkpoint")
    for value in (
        arguments.expected_campaign_protocol_sha256,
        arguments.expected_checkpoint_sha256,
        arguments.expected_source_sha256,
    ):
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("Expected SHA-256 is malformed")


def _source_sha256(project_root: Path) -> str:
    digest = hashlib.sha256()
    for relative in SOURCE_PATHS:
        path = project_root / relative
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"Reviewed source is missing or symlinked: {relative}")
        name = relative.encode("utf-8")
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def _load_campaign(arguments: argparse.Namespace) -> dict[str, Any]:
    path = arguments.campaign_protocol.resolve()
    if _sha256(path) != arguments.expected_campaign_protocol_sha256:
        raise RuntimeError("Campaign protocol SHA-256 differs")
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if (
        protocol.get("format_version") != "abpairfm_3b_stage_a0_campaign_v1"
        or protocol.get("authorization", {}).get("production_stage_a0_pair_mlm") is not True
        or protocol.get("authorization", {}).get("production_stage_a_full_objective") is not False
        or protocol.get("authorization", {}).get("test_access") is not False
        or protocol.get("systems", {}).get("partition") != "gpu_4090"
    ):
        raise RuntimeError("Campaign authorization differs")
    return protocol


def _load_manifest(arguments: argparse.Namespace) -> tuple[list[dict[str, Any]], str, str]:
    manifest = arguments.manifest.resolve()
    receipt_path = arguments.manifest_receipt.resolve()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    manifest_sha256 = _sha256(manifest)
    if (
        receipt.get("format_version") != "abpairfm_3b_stage_a0_segment_manifest_v1"
        or receipt.get("status") != "PRODUCTION_STAGE_A0_SEGMENT_READY"
        or receipt.get("authorization", {}).get("production_stage_a0_pair_mlm") is not True
        or receipt.get("authorization", {}).get("production_stage_a_full_objective") is not False
        or receipt.get("test_access") is not False
        or receipt.get("segment", {}).get("index") != arguments.segment_index
        or receipt.get("inputs", {}).get("protocol_sha256")
        != arguments.expected_campaign_protocol_sha256
        or receipt.get("output", {}).get("sha256") != manifest_sha256
    ):
        raise RuntimeError("Stage-A0 segment manifest evidence differs")
    records: list[dict[str, Any]] = []
    hashes: set[str] = set()
    for line_number, line in enumerate(_zstd_lines(manifest), start=1):
        value = json.loads(line)
        pair_hash = value.get("pair_hash_sha256")
        if (
            value.get("segment_record_index") != line_number - 1
            or not isinstance(pair_hash, str)
            or len(pair_hash) != 64
            or pair_hash in hashes
            or value.get("evidence_role") not in {"direct", "transferred_only"}
            or min(len(value.get("sequence1", "")), len(value.get("sequence2", ""))) < 20
        ):
            raise RuntimeError("Stage-A0 segment record differs")
        hashes.add(pair_hash)
        records.append(value)
    required = (
        arguments.updates
        * arguments.accumulation_steps
        * arguments.expected_world_size
        * arguments.microbatch_pairs
    )
    if len(records) != required or len(records) != receipt.get("counts", {}).get("manifest_records"):
        raise RuntimeError("Stage-A0 segment record count differs")
    return records, manifest_sha256, _sha256(receipt_path)


def _learning_rate(consumed: int, *, warmup: int, target: int, maximum: float) -> float:
    if consumed < warmup:
        return maximum * max(consumed, 1) / warmup
    return maximum * max(target - consumed, 0) / (target - warmup)


def _input_receipt(arguments: argparse.Namespace) -> dict[str, Any] | None:
    if arguments.input_checkpoint_root is None:
        return None
    assert arguments.input_generation is not None
    assert arguments.expected_input_content_sha256 is not None
    path = arguments.input_checkpoint_root.resolve() / arguments.input_generation / "receipt.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("content_sha256") != arguments.expected_input_content_sha256:
        raise RuntimeError("Input checkpoint content SHA-256 differs")
    state = value.get("training_state", {})
    if (
        state.get("next_segment_index") != arguments.segment_index
        or state.get("test_access") is not False
        or state.get("target_residue_tokens") != arguments.target_residue_tokens
    ):
        raise RuntimeError("Input checkpoint training cursor differs")
    return value


def main() -> None:
    arguments = _arguments()
    _validate(arguments)
    project_root = arguments.project_root.resolve()
    campaign = _load_campaign(arguments)
    source_sha256 = _source_sha256(project_root)
    if source_sha256 != arguments.expected_source_sha256:
        raise RuntimeError("Reviewed Stage-A0 source digest differs from launch anchor")
    records, manifest_sha256, manifest_receipt_sha256 = _load_manifest(arguments)
    prior_receipt = _input_receipt(arguments)
    provenance = {
        "campaign_protocol": arguments.expected_campaign_protocol_sha256,
        "initial_checkpoint": arguments.expected_checkpoint_sha256,
        "training_source": source_sha256,
    }

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.distributed.init_process_group(
        backend="nccl", device_id=device, timeout=timedelta(minutes=15)
    )
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    try:
        if world_size != arguments.expected_world_size:
            raise RuntimeError("Stage-A0 world size differs")
        properties = torch.cuda.get_device_properties(device)
        if properties.name != "NVIDIA GeForce RTX 4090":
            raise RuntimeError(f"Forbidden GPU model: {properties.name}")
        torch.manual_seed(arguments.seed + rank)
        torch.cuda.manual_seed(arguments.seed + rank)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

        from torch.distributed.device_mesh import init_device_mesh

        mesh = init_device_mesh(
            "cuda", (2, 4), mesh_dim_names=("replicate", "shard")
        )
        config = PairAwareESM2Config(activation_checkpointing=True)
        with torch.device("meta"):
            model = PairAwareESM2(config)
        fully_shard_pair_aware_esm2(
            model, mesh=mesh, precision="bfloat16", reshard_after_forward=False
        )
        initialization = load_authenticated_esm2_into_fsdp2(
            model,
            arguments.checkpoint,
            expected_size=arguments.expected_checkpoint_size,
            expected_sha256=arguments.expected_checkpoint_sha256,
        )
        optimizer = optimizer_after_fsdp2(
            model,
            learning_rate=arguments.maximum_learning_rate,
            weight_decay=arguments.weight_decay,
            betas=(0.9, 0.98),
        )
        consumed_before = 0
        global_update_before = 0
        input_checkpoint = None
        if prior_receipt is not None:
            input_checkpoint = _DCP.load_dcp_generation(
                arguments.input_checkpoint_root,
                arguments.input_generation,
                model=model,
                optimizer=optimizer,
                expected_provenance_hashes=prior_receipt["provenance_hashes"],
            )
            if input_checkpoint["content_sha256"] != arguments.expected_input_content_sha256:
                raise RuntimeError("Collectively loaded checkpoint content differs")
            consumed_before = int(input_checkpoint["training_state"]["consumed_residue_tokens"])
            global_update_before = int(input_checkpoint["training_state"]["global_update"])
        if not hasattr(model, "set_requires_gradient_sync") or not hasattr(
            model, "set_is_last_backward"
        ):
            raise RuntimeError("FSDP2 accumulation controls are unavailable")

        model.train()
        torch.cuda.reset_peak_memory_stats(device)
        updates: list[dict[str, float | int]] = []
        consumed_segment = 0
        local_swapped = 0
        from torch.nn.attention import SDPBackend, sdpa_kernel

        for update in range(arguments.updates):
            optimizer.zero_grad(set_to_none=True)
            learning_rate = _learning_rate(
                consumed_before + consumed_segment,
                warmup=arguments.warmup_residue_tokens,
                target=arguments.target_residue_tokens,
                maximum=arguments.maximum_learning_rate,
            )
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            torch.distributed.barrier()
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            update_loss = 0.0
            update_residues = 0
            for accumulation in range(arguments.accumulation_steps):
                microstep = update * arguments.accumulation_steps + accumulation
                start = microstep * world_size * arguments.microbatch_pairs + rank * arguments.microbatch_pairs
                indices = list(range(start, start + arguments.microbatch_pairs))
                cpu_tokens, cpu_chain_ids, cpu_labels, local_residues, swapped = _build_real_microbatch(
                    records,
                    indices,
                    maximum_tokens=arguments.maximum_tokens,
                    seed=arguments.seed + arguments.segment_index * 1_000_003,
                )
                local_swapped += swapped
                tokens = cpu_tokens.to(device, non_blocking=True)
                chain_ids = cpu_chain_ids.to(device, non_blocking=True)
                labels = cpu_labels.to(device, non_blocking=True)
                synchronize = accumulation == arguments.accumulation_steps - 1
                model.set_requires_gradient_sync(synchronize)
                model.set_is_last_backward(synchronize)
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    output = model(tokens, chain_ids, labels=labels)
                    if output.balanced_chain_loss_sum is None:
                        raise RuntimeError("Stage-A0 model did not return MLM loss")
                    global_chain_count = output.contributing_chain_count.clone()
                    torch.distributed.all_reduce(global_chain_count)
                    loss = (
                        output.balanced_chain_loss_sum
                        * world_size
                        / global_chain_count.to(output.balanced_chain_loss_sum.dtype)
                        / arguments.accumulation_steps
                    )
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("Non-finite Stage-A0 MLM loss")
                loss.backward()
                detached_loss = output.balanced_chain_loss_sum.detach().clone()
                torch.distributed.all_reduce(detached_loss)
                update_loss += float(detached_loss / global_chain_count) / arguments.accumulation_steps
                residue_tensor = torch.tensor(local_residues, device=device, dtype=torch.long)
                torch.distributed.all_reduce(residue_tensor)
                update_residues += int(residue_tensor)
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not bool(torch.isfinite(gradient_norm)):
                raise RuntimeError("Non-finite Stage-A0 gradient norm")
            optimizer.step()
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            consumed_segment += update_residues
            record = {
                "elapsed_seconds": elapsed,
                "global_mean_mlm_loss": update_loss,
                "global_residue_tokens": update_residues,
                "global_update": global_update_before + update + 1,
                "gradient_norm": float(gradient_norm),
                "learning_rate": learning_rate,
                "segment_update": update + 1,
            }
            updates.append(record)
            if rank == 0 and ((update + 1) % arguments.progress_interval == 0 or update + 1 == arguments.updates):
                _publish_progress(
                    arguments.progress.resolve(),
                    {
                        "consumed_residue_tokens_campaign": consumed_before + consumed_segment,
                        "consumed_residue_tokens_segment": consumed_segment,
                        "format_version": "abpairfm_3b_stage_a0_progress_v1",
                        "global_update": global_update_before + update + 1,
                        "latest_global_mean_mlm_loss": update_loss,
                        "latest_gradient_norm": float(gradient_norm),
                        "learning_rate": learning_rate,
                        "segment_index": arguments.segment_index,
                        "segment_update": update + 1,
                        "status": "RUNNING" if update + 1 < arguments.updates else "TRAINING_COMPLETE_CHECKPOINT_PENDING",
                        "test_access": False,
                    },
                )

        rank_record = {
            "device_name": properties.name,
            "hostname": socket.gethostname(),
            "local_pair_order_swaps": local_swapped,
            "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "max_memory_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "rank": rank,
            "total_memory_bytes": properties.total_memory,
        }
        ranks = _collect(rank_record, world_size)
        reshard_pair_aware_esm2(model)
        generation = f"segment-{arguments.segment_index:06d}"
        training_state = {
            "consumed_residue_tokens": consumed_before + consumed_segment,
            "global_update": global_update_before + arguments.updates,
            "last_manifest_receipt_sha256": manifest_receipt_sha256,
            "last_manifest_sha256": manifest_sha256,
            "next_segment_index": arguments.segment_index + 1,
            "target_residue_tokens": arguments.target_residue_tokens,
            "test_access": False,
        }
        checkpoint_receipt = _DCP.save_dcp_generation(
            arguments.checkpoint_root.resolve(),
            generation,
            model=model,
            optimizer=optimizer,
            provenance_hashes=provenance,
            training_state=training_state,
        )
        if rank == 0:
            losses = [float(value["global_mean_mlm_loss"]) for value in updates]
            gradients = [float(value["gradient_norm"]) for value in updates]
            elapsed = [float(value["elapsed_seconds"]) for value in updates]
            residues = [int(value["global_residue_tokens"]) for value in updates]
            result = {
                "authorization": {
                    "production_stage_a0_pair_mlm": True,
                    "production_stage_a_full_objective": False,
                },
                "campaign_protocol_sha256": arguments.expected_campaign_protocol_sha256,
                "checkpoint": {
                    "content_sha256": checkpoint_receipt["content_sha256"],
                    "generation": generation,
                    "root": str(arguments.checkpoint_root.resolve()),
                },
                "claim_boundary": CLAIM_BOUNDARY,
                "format_version": "abpairfm_3b_stage_a0_segment_result_v1",
                "initialization": initialization,
                "input_checkpoint_content_sha256": (
                    None if input_checkpoint is None else input_checkpoint["content_sha256"]
                ),
                "loss": {
                    "final_20_update_mean": statistics.mean(losses[-20:]),
                    "first_20_update_mean": statistics.mean(losses[:20]),
                    "maximum": max(losses),
                    "minimum": min(losses),
                    "nonfinite_updates": 0,
                },
                "manifest": {
                    "records_consumed": len(records),
                    "receipt_sha256": manifest_receipt_sha256,
                    "sha256": manifest_sha256,
                },
                "optimization": {
                    "accumulation_steps": arguments.accumulation_steps,
                    "maximum_learning_rate": arguments.maximum_learning_rate,
                    "microbatch_pairs_per_rank": arguments.microbatch_pairs,
                    "updates": arguments.updates,
                    "warmup_residue_tokens": arguments.warmup_residue_tokens,
                },
                "ranks": ranks,
                "segment_index": arguments.segment_index,
                "source_sha256": source_sha256,
                "stability": {
                    "maximum_gradient_norm_before_clipping": max(gradients),
                    "p95_to_median_update_time": _nearest_rank_p95(elapsed) / statistics.median(elapsed),
                },
                "status": "PASS_CHECKPOINT_VERIFIED",
                "test_access": False,
                "throughput": {
                    "consumed_residue_tokens_campaign": consumed_before + consumed_segment,
                    "consumed_residue_tokens_segment": consumed_segment,
                    "residue_tokens_per_second_mean": statistics.mean(
                        value / seconds for value, seconds in zip(residues, elapsed, strict=True)
                    ),
                },
                "training_state": training_state,
                "world_size": world_size,
            }
            _publish_create_only(arguments.result.resolve(), result)
            _publish_progress(
                arguments.progress.resolve(),
                {
                    "checkpoint_content_sha256": checkpoint_receipt["content_sha256"],
                    "consumed_residue_tokens_campaign": consumed_before + consumed_segment,
                    "format_version": "abpairfm_3b_stage_a0_progress_v1",
                    "global_update": global_update_before + arguments.updates,
                    "segment_index": arguments.segment_index,
                    "segment_update": arguments.updates,
                    "status": "PASS_CHECKPOINT_VERIFIED",
                    "test_access": False,
                },
            )
            print(json.dumps(result, sort_keys=True, separators=(",", ":")))
        torch.distributed.barrier()
    finally:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
