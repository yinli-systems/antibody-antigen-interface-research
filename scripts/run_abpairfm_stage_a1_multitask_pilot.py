#!/usr/bin/env python3
"""Run the frozen 200-update ABPairFM Stage-A1 multitask pilot."""

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
from typing import Any, Mapping

import torch
from torch import nn


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
SCRIPT_ROOT = REPOSITORY_ROOT / "scripts"
for path in (SOURCE_ROOT, SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from antibody_pretraining.pair_aware_distributed import (  # noqa: E402
    fully_shard_pair_aware_esm2,
    load_authenticated_esm2_into_fsdp2,
    reshard_pair_aware_esm2,
)
from antibody_pretraining.pair_aware_esm2 import (  # noqa: E402
    PairAwareESM2,
    PairAwareESM2Config,
)
from antibody_pretraining.stage_a1_batch import (  # noqa: E402
    StageA1Microbatch,
    build_stage_a1_microbatch,
)
from antibody_pretraining.stage_a1_objective import StageA1Heads  # noqa: E402
from interaction_first.frontier_fsdp2 import verify_dcp_generation  # noqa: E402
from run_abpairfm_3b_real_data_gate_v2 import (  # noqa: E402
    _collect,
    _nearest_rank_p95,
    _publish_create_only,
    _publish_progress,
    _sha256,
    _zstd_lines,
)


EXPECTED_PROTOCOL_FORMAT = "abpairfm_stage_a1_multitask_pilot_v3"
SOURCE_PATHS = (
    "scripts/run_abpairfm_stage_a1_multitask_pilot.py",
    "scripts/run_abpairfm_3b_real_data_gate_v2.py",
    "src/antibody_pretraining/pair_aware_attention.py",
    "src/antibody_pretraining/pair_aware_distributed.py",
    "src/antibody_pretraining/pair_aware_esm2.py",
    "src/antibody_pretraining/stage_a1_batch.py",
    "src/antibody_pretraining/stage_a1_objective.py",
    "src/interaction_first/frontier_fsdp2.py",
)


class StageA1System(nn.Module):
    """Register the sharded backbone and replicated heads under one DCP root."""

    def __init__(self, backbone: PairAwareESM2, heads: StageA1Heads) -> None:
        super().__init__()
        self.backbone = backbone
        self.heads = heads


def _frontier_module() -> Any:
    path = SOURCE_ROOT / "interaction_first" / "frontier_fsdp2.py"
    specification = importlib.util.spec_from_file_location("stage_a1_dcp", path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Cannot load DCP implementation: {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


_DCP = _frontier_module()


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-receipt", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-initial-checkpoint-size", type=int, required=True)
    parser.add_argument("--expected-initial-checkpoint-sha256", required=True)
    parser.add_argument("--input-checkpoint-root", type=Path, required=True)
    parser.add_argument("--input-generation", required=True)
    parser.add_argument("--expected-input-content-sha256", required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--expected-world-size", type=int, default=8)
    parser.add_argument("--shard-group-size", type=int, default=4)
    parser.add_argument("--maximum-tokens", type=int, default=1024)
    parser.add_argument("--progress-interval", type=int, default=10)
    return parser.parse_args()


def source_sha256(project_root: Path) -> str:
    digest = hashlib.sha256()
    for relative in SOURCE_PATHS:
        path = project_root / relative
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"Reviewed Stage-A1 source is missing: {relative}")
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def _load_protocol(arguments: argparse.Namespace) -> dict[str, Any]:
    if _sha256(arguments.protocol) != arguments.expected_protocol_sha256:
        raise RuntimeError("Stage-A1 protocol SHA-256 differs")
    protocol = json.loads(arguments.protocol.read_text(encoding="utf-8"))
    authorization = protocol.get("authorization", {})
    systems = protocol.get("systems", {})
    optimization = protocol.get("optimization", {})
    if (
        protocol.get("format_version") != EXPECTED_PROTOCOL_FORMAT
        or protocol.get("status") != "FROZEN_BEFORE_PILOT_MANIFEST_AND_GPU_TRAINING"
        or authorization.get("pilot_training") is not True
        or authorization.get("production_training") is not False
        or authorization.get("test_access") is not False
        or protocol.get("test_access") is not False
        or systems.get("partition") != "gpu_4090"
        or systems.get("world_size") != arguments.expected_world_size
        or optimization.get("updates") != 200
        or optimization.get("microbatch_pairs_per_rank") != 7
        or optimization.get("accumulation_steps") != 2
    ):
        raise RuntimeError("Stage-A1 pilot authorization or shape differs")
    if protocol["checkpoint"]["generation"] != arguments.input_generation:
        raise RuntimeError("Stage-A1 input generation differs from protocol")
    if (
        protocol["checkpoint"]["content_sha256"]
        != arguments.expected_input_content_sha256
    ):
        raise RuntimeError("Stage-A1 input checkpoint anchor differs from protocol")
    return protocol


def _load_manifest(
    arguments: argparse.Namespace, protocol: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], str, str]:
    receipt = json.loads(arguments.manifest_receipt.read_text(encoding="utf-8"))
    manifest_sha256 = _sha256(arguments.manifest)
    receipt_sha256 = _sha256(arguments.manifest_receipt)
    if (
        receipt.get("status") != "PASS_STAGE_A1_PILOT_SOURCE_ADMISSION"
        or receipt.get("test_access") is not False
        or receipt.get("inputs", {}).get("protocol_sha256")
        != arguments.expected_protocol_sha256
        or receipt.get("output", {}).get("sha256") != manifest_sha256
        or receipt.get("counts", {}).get("records")
        != protocol["pilot_manifest"]["total_records"]
        or receipt.get("cross_source_exact_pair_overlap_after_filter") != 0
    ):
        raise RuntimeError("Stage-A1 manifest receipt differs")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(_zstd_lines(arguments.manifest), start=1):
        value = json.loads(line)
        if (
            value.get("pilot_record_index") != line_number - 1
            or value.get("source") not in {"ppiref50k", "string_v12_physical"}
            or value.get("contact_supervision")
            != (value.get("source") == "ppiref50k")
            or not isinstance(value.get("retrieval_supervision"), bool)
            or min(len(value.get("sequence1", "")), len(value.get("sequence2", "")))
            < 3
        ):
            raise RuntimeError("Stage-A1 manifest record differs")
        records.append(value)
    if len(records) != protocol["pilot_manifest"]["total_records"]:
        raise RuntimeError("Stage-A1 manifest count differs")
    return records, manifest_sha256, receipt_sha256


def _verify_input_dcp_collectively(
    arguments: argparse.Namespace, protocol: Mapping[str, Any]
) -> dict[str, Any]:
    rank = torch.distributed.get_rank()
    payload: list[dict[str, Any]] = [{"error": None, "receipt": None}]
    if rank == 0:
        try:
            receipt = verify_dcp_generation(
                arguments.input_checkpoint_root, arguments.input_generation
            )
            if receipt.get("content_sha256") != arguments.expected_input_content_sha256:
                raise RuntimeError("Stage-A1 immutable DCP content SHA-256 differs")
            if (
                receipt.get("provenance_hashes", {}).get("initial_checkpoint")
                != arguments.expected_initial_checkpoint_sha256
            ):
                raise RuntimeError("Stage-A1 DCP initial-checkpoint provenance differs")
            consumed = receipt.get("training_state", {}).get(
                "consumed_residue_tokens"
            )
            if (
                not isinstance(consumed, int)
                or consumed < protocol["checkpoint"]["minimum_consumed_residue_tokens"]
                or receipt.get("training_state", {}).get("test_access") is not False
            ):
                raise RuntimeError("Stage-A1 input DCP training cursor differs")
            payload[0]["receipt"] = receipt
        except BaseException as error:
            payload[0]["error"] = f"{type(error).__name__}: {error}"
    torch.distributed.broadcast_object_list(payload, src=0)
    if payload[0]["error"] is not None:
        raise RuntimeError(f"Stage-A1 DCP verification failed: {payload[0]['error']}")
    receipt = payload[0]["receipt"]
    if not isinstance(receipt, dict):
        raise RuntimeError("Stage-A1 DCP receipt was not broadcast")
    return receipt


def _load_model_only_dcp(
    model: PairAwareESM2, *, root: Path, generation: str
) -> None:
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
        set_model_state_dict,
    )

    options = StateDictOptions(full_state_dict=False, cpu_offload=False, strict=True)
    state = get_model_state_dict(model, options=options)
    dcp.load(
        {"model": state},
        checkpoint_id=root.resolve() / generation,
        no_dist=False,
    )
    incompatible = set_model_state_dict(model, state, options=options)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("Stage-A1 model-only DCP restore returned incompatible keys")
    torch.distributed.barrier()


def _to_device(batch: StageA1Microbatch, device: torch.device) -> StageA1Microbatch:
    values: dict[str, Any] = {}
    for field in batch.__dataclass_fields__:
        value = getattr(batch, field)
        values[field] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return StageA1Microbatch(**values)


def _sync_replicated_head_gradients(heads: StageA1Heads, world_size: int) -> None:
    for parameter in heads.parameters():
        if parameter.grad is None:
            raise RuntimeError("Stage-A1 head parameter did not receive a gradient")
        torch.distributed.all_reduce(parameter.grad)
        parameter.grad.div_(world_size)


def _clip_gradients(
    backbone: PairAwareESM2, heads: StageA1Heads, maximum_norm: float
) -> float:
    backbone_norm = torch.nn.utils.clip_grad_norm_(
        backbone.parameters(), float("inf"), foreach=False
    )
    head_norm = torch.nn.utils.clip_grad_norm_(
        heads.parameters(), float("inf"), foreach=False
    )
    total = math.sqrt(float(backbone_norm) ** 2 + float(head_norm) ** 2)
    if not math.isfinite(total):
        raise RuntimeError("Stage-A1 gradient norm is nonfinite")
    scale = min(1.0, maximum_norm / (total + 1e-6))
    if scale < 1.0:
        for parameter in [*backbone.parameters(), *heads.parameters()]:
            if parameter.grad is not None:
                parameter.grad.mul_(scale)
    return total


def _mean_records(records: list[dict[str, float]], key: str, count: int) -> float:
    return statistics.mean(record[key] for record in records[-count:])


def main() -> None:
    arguments = _arguments()
    protocol = _load_protocol(arguments)
    project_root = arguments.project_root.resolve()
    observed_source_sha256 = source_sha256(project_root)
    if observed_source_sha256 != arguments.expected_source_sha256:
        raise RuntimeError("Reviewed Stage-A1 source SHA-256 differs")
    records, manifest_sha256, manifest_receipt_sha256 = _load_manifest(
        arguments, protocol
    )

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
            raise RuntimeError("Stage-A1 world size differs")
        properties = torch.cuda.get_device_properties(device)
        if properties.name != "NVIDIA GeForce RTX 4090":
            raise RuntimeError(f"Forbidden GPU model: {properties.name}")
        input_receipt = _verify_input_dcp_collectively(arguments, protocol)
        torch.manual_seed(1701)
        torch.cuda.manual_seed(1701)
        from torch.distributed.device_mesh import init_device_mesh

        mesh = init_device_mesh(
            "cuda", (2, arguments.shard_group_size), mesh_dim_names=("replicate", "shard")
        )
        with torch.device("meta"):
            backbone = PairAwareESM2(PairAwareESM2Config(activation_checkpointing=True))
        fully_shard_pair_aware_esm2(
            backbone, mesh=mesh, precision="bfloat16", reshard_after_forward=False
        )
        initialization = load_authenticated_esm2_into_fsdp2(
            backbone,
            arguments.initial_checkpoint,
            expected_size=arguments.expected_initial_checkpoint_size,
            expected_sha256=arguments.expected_initial_checkpoint_sha256,
        )
        _load_model_only_dcp(
            backbone,
            root=arguments.input_checkpoint_root,
            generation=arguments.input_generation,
        )
        heads = StageA1Heads(temperature=protocol["loss"]["retrieval"]["temperature"]).to(
            device
        )
        system = StageA1System(backbone, heads)
        optimization = protocol["optimization"]
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": list(backbone.parameters()),
                    "lr": optimization["backbone_learning_rate"],
                },
                {
                    "params": list(heads.parameters()),
                    "lr": optimization["head_learning_rate"],
                },
            ],
            betas=tuple(optimization["betas"]),
            weight_decay=optimization["weight_decay"],
            fused=True,
        )
        if not hasattr(backbone, "set_requires_gradient_sync") or not hasattr(
            backbone, "set_is_last_backward"
        ):
            raise RuntimeError("Stage-A1 FSDP2 accumulation controls are unavailable")
        torch.manual_seed(1701 + rank)
        torch.cuda.manual_seed(1701 + rank)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        backbone.train()
        heads.train()
        torch.cuda.reset_peak_memory_stats(device)
        captured: list[torch.Tensor] = []
        hook = backbone.emb_layer_norm_after.register_forward_hook(
            lambda _module, _inputs, output: captured.append(output)
        )
        update_records: list[dict[str, float]] = []
        from torch.nn.attention import SDPBackend, sdpa_kernel

        for update in range(optimization["updates"]):
            optimizer.zero_grad(set_to_none=True)
            torch.distributed.barrier()
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            totals = torch.zeros(7, device=device, dtype=torch.float64)
            for accumulation in range(optimization["accumulation_steps"]):
                microstep = update * optimization["accumulation_steps"] + accumulation
                start = (
                    microstep * world_size + rank
                ) * optimization["microbatch_pairs_per_rank"]
                cpu_batch = build_stage_a1_microbatch(
                    records[
                        start : start + optimization["microbatch_pairs_per_rank"]
                    ],
                    seed=1701 + microstep * world_size + rank,
                    maximum_tokens=arguments.maximum_tokens,
                    maximum_positive_pairs_per_complex=protocol["loss"]["contact"][
                        "maximum_positive_pairs_per_complex"
                    ],
                    maximum_sampled_pairs_per_complex=protocol["loss"]["contact"][
                        "maximum_sampled_pairs_per_complex"
                    ],
                    maximum_negative_to_positive_ratio=protocol["loss"]["contact"][
                        "maximum_negative_to_positive_ratio"
                    ],
                )
                batch = _to_device(cpu_batch, device)
                synchronize = accumulation == optimization["accumulation_steps"] - 1
                backbone.set_requires_gradient_sync(synchronize)
                backbone.set_is_last_backward(synchronize)
                captured.clear()
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION), torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16
                ):
                    output = backbone(batch.tokens, batch.chain_ids, labels=batch.labels)
                    if output.balanced_chain_loss_sum is None or len(captured) != 1:
                        raise RuntimeError("Stage-A1 backbone did not expose MLM and hidden states")
                    mlm_loss = output.balanced_chain_loss_sum / output.contributing_chain_count
                    retrieval_indices = batch.retrieval_batch_indices
                    multitask = heads.loss(
                        hidden=captured[0][retrieval_indices],
                        chain_ids=batch.chain_ids[retrieval_indices],
                        mlm_loss=mlm_loss,
                        retrieval_allowed=batch.retrieval_allowed,
                        pair_batch_indices=torch.searchsorted(
                            retrieval_indices, batch.contact_batch_indices
                        ),
                        pair_left_token_indices=batch.contact_left_token_indices,
                        pair_right_token_indices=batch.contact_right_token_indices,
                        pair_labels=batch.contact_labels,
                        mlm_weight=protocol["loss"]["masked_language_modeling"]["weight"],
                        retrieval_weight=protocol["loss"]["retrieval"]["weight"],
                        contact_weight=protocol["loss"]["contact"]["weight"],
                    )
                    loss = multitask.total / optimization["accumulation_steps"]
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("Stage-A1 multitask loss is nonfinite")
                loss.backward()
                totals += torch.tensor(
                    [
                        float(multitask.total.detach()),
                        float(multitask.mlm.detach()),
                        float(multitask.retrieval.detach()),
                        float(multitask.contact.detach()),
                        float(multitask.retrieval_accuracy_left_to_right.detach()),
                        float(multitask.retrieval_accuracy_right_to_left.detach()),
                        float(batch.residue_count),
                    ],
                    device=device,
                    dtype=torch.float64,
                )
            _sync_replicated_head_gradients(heads, world_size)
            gradient_norm = _clip_gradients(
                backbone, heads, optimization["gradient_clip_norm"]
            )
            optimizer.step()
            torch.distributed.all_reduce(totals)
            totals[:6] /= world_size * optimization["accumulation_steps"]
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            record = {
                "contact_loss": float(totals[3]),
                "elapsed_seconds": elapsed,
                "gradient_norm": gradient_norm,
                "mlm_loss": float(totals[1]),
                "residue_tokens": float(totals[6]),
                "retrieval_accuracy_left_to_right": float(totals[4]),
                "retrieval_accuracy_right_to_left": float(totals[5]),
                "retrieval_loss": float(totals[2]),
                "total_loss": float(totals[0]),
            }
            update_records.append(record)
            if rank == 0 and (
                (update + 1) % arguments.progress_interval == 0
                or update + 1 == optimization["updates"]
            ):
                _publish_progress(
                    arguments.progress,
                    {
                        "format_version": "abpairfm_stage_a1_pilot_progress_v1",
                        "latest": record,
                        "status": (
                            "RUNNING"
                            if update + 1 < optimization["updates"]
                            else "TRAINING_COMPLETE_CHECKPOINT_PENDING"
                        ),
                        "test_access": False,
                        "update": update + 1,
                    },
                )
        hook.remove()
        rank_record = {
            "device_name": properties.name,
            "hostname": socket.gethostname(),
            "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "max_memory_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "rank": rank,
            "total_memory_bytes": properties.total_memory,
        }
        ranks = _collect(rank_record, world_size)
        reshard_pair_aware_esm2(backbone)
        provenance = {
            "input_checkpoint": arguments.expected_input_content_sha256,
            "manifest": manifest_sha256,
            "manifest_receipt": manifest_receipt_sha256,
            "protocol": arguments.expected_protocol_sha256,
            "source": observed_source_sha256,
        }
        checkpoint_receipt = _DCP.save_dcp_generation(
            arguments.checkpoint_root,
            "pilot-000000",
            model=system,
            optimizer=optimizer,
            provenance_hashes=provenance,
            training_state={
                "input_consumed_residue_tokens": input_receipt["training_state"][
                    "consumed_residue_tokens"
                ],
                "pilot_updates": optimization["updates"],
                "test_access": False,
            },
        )
        restore_receipt = _DCP.load_dcp_generation(
            arguments.checkpoint_root,
            "pilot-000000",
            model=system,
            optimizer=optimizer,
            expected_provenance_hashes=provenance,
        )
        peak_fraction = max(
            rank_value["max_memory_allocated_bytes"] / rank_value["total_memory_bytes"]
            for rank_value in ranks
        )
        status = (
            "PASS_STAGE_A1_MULTITASK_PILOT_CHECKPOINT_VERIFIED"
            if peak_fraction
            <= protocol["promotion_gates"]["maximum_peak_memory_fraction"]
            else "FAIL_STAGE_A1_MULTITASK_PILOT_MEMORY_GATE"
        )
        if rank == 0:
            elapsed_values = [record["elapsed_seconds"] for record in update_records]
            result = {
                "authorization": protocol["authorization"],
                "checkpoint": {
                    "content_sha256": checkpoint_receipt["content_sha256"],
                    "generation": "pilot-000000",
                    "restore_content_sha256": restore_receipt["content_sha256"],
                    "root": str(arguments.checkpoint_root.resolve()),
                },
                "claim_boundary": protocol["claim_boundary"],
                "format_version": "abpairfm_stage_a1_multitask_pilot_result_v1",
                "input_checkpoint": {
                    "consumed_residue_tokens": input_receipt["training_state"][
                        "consumed_residue_tokens"
                    ],
                    "content_sha256": arguments.expected_input_content_sha256,
                    "generation": arguments.input_generation,
                },
                "initialization": initialization,
                "loss": {
                    name: {
                        "final_20_update_mean": _mean_records(update_records, name, 20),
                        "first_20_update_mean": statistics.mean(
                            record[name] for record in update_records[:20]
                        ),
                    }
                    for name in ("total_loss", "mlm_loss", "retrieval_loss", "contact_loss")
                },
                "manifest": {
                    "records_consumed": len(records),
                    "receipt_sha256": manifest_receipt_sha256,
                    "sha256": manifest_sha256,
                },
                "memory": {
                    "maximum_allocated_fraction": peak_fraction,
                    "required_maximum_fraction": protocol["promotion_gates"][
                        "maximum_peak_memory_fraction"
                    ],
                },
                "optimization": optimization,
                "ranks": ranks,
                "source_sha256": observed_source_sha256,
                "stability": {
                    "maximum_gradient_norm_before_clipping": max(
                        record["gradient_norm"] for record in update_records
                    ),
                    "p95_to_median_update_time": _nearest_rank_p95(elapsed_values)
                    / statistics.median(elapsed_values),
                },
                "status": status,
                "test_access": False,
                "throughput": {
                    "mean_residue_tokens_per_second": statistics.mean(
                        record["residue_tokens"] / record["elapsed_seconds"]
                        for record in update_records
                    ),
                    "pilot_residue_tokens": sum(
                        record["residue_tokens"] for record in update_records
                    ),
                },
            }
            _publish_create_only(arguments.result, result)
            _publish_progress(
                arguments.progress,
                {
                    "checkpoint_content_sha256": checkpoint_receipt["content_sha256"],
                    "format_version": "abpairfm_stage_a1_pilot_progress_v1",
                    "latest": update_records[-1],
                    "status": status,
                    "test_access": False,
                    "update": optimization["updates"],
                },
            )
        torch.distributed.barrier()
        if status.startswith("FAIL_"):
            raise RuntimeError("Stage-A1 pilot failed its frozen memory gate")
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
