#!/usr/bin/env python3
"""Evaluate immutable Stage-A1 retrieval heads on frozen development pairs."""

from __future__ import annotations

import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
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
)
from antibody_pretraining.pair_aware_esm2 import (  # noqa: E402
    ESM2_ALPHABET,
    PairAwareESM2,
    PairAwareESM2Config,
)
from antibody_pretraining.stage_a1_objective import StageA1Heads  # noqa: E402
from interaction_first.frontier_fsdp2 import verify_dcp_generation  # noqa: E402
from run_mint_frozen_development_retrieval import (  # noqa: E402
    batch_work,
    endpoint_windows,
    evaluate,
    load_inputs,
    publish_create_only,
    retrieval_pairs,
    sha256_file,
)


EXPECTED_PROTOCOL_FORMAT = "abpairfm_stage_a1_development_retrieval_v1"
EXPECTED_INITIAL_CHECKPOINT_SHA256 = (
    "7de8b4082ba15891959ab368b77ce3886697af1efb16d3c9e9e7b0c5d3f07500"
)
EXPECTED_INITIAL_CHECKPOINT_SIZE = 5_678_116_398
SOURCE_PATHS = (
    "scripts/run_abpairfm_stage_a1_frozen_development_retrieval.py",
    "scripts/run_mint_frozen_development_retrieval.py",
    "src/antibody_pretraining/pair_aware_attention.py",
    "src/antibody_pretraining/pair_aware_distributed.py",
    "src/antibody_pretraining/pair_aware_esm2.py",
    "src/antibody_pretraining/stage_a1_objective.py",
    "src/interaction_first/frontier_fsdp2.py",
)


class StageA1System(nn.Module):
    def __init__(self, backbone: PairAwareESM2, heads: StageA1Heads) -> None:
        super().__init__()
        self.backbone = backbone
        self.heads = heads


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--dcp-root", type=Path, required=True)
    parser.add_argument("--dcp-generation", required=True)
    parser.add_argument("--expected-dcp-content-sha256", required=True)
    parser.add_argument("--promotion-result", type=Path, required=True)
    parser.add_argument("--pair-panel", type=Path, required=True)
    parser.add_argument("--pair-panel-receipt", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--maximum-batch-size", type=int, default=7)
    parser.add_argument("--maximum-attention-cells", type=int, default=7_340_032)
    return parser.parse_args()


def _source_sha256(project_root: Path) -> str:
    digest = hashlib.sha256()
    for relative in SOURCE_PATHS:
        path = project_root / relative
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"Reviewed source is missing or symlinked: {relative}")
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def _load_protocol(arguments: argparse.Namespace) -> dict[str, Any]:
    if sha256_file(arguments.protocol) != arguments.expected_protocol_sha256:
        raise RuntimeError("Stage-A1 development protocol SHA-256 differs")
    protocol = json.loads(arguments.protocol.read_text(encoding="utf-8"))
    authorization = protocol.get("authorization", {})
    compute = protocol.get("compute", {})
    if (
        protocol.get("format_version") != EXPECTED_PROTOCOL_FORMAT
        or protocol.get("status") != "FROZEN_BEFORE_GPU_EVALUATION"
        or authorization.get("development_pair_retrieval") is not True
        or authorization.get("development_contact") is not False
        or authorization.get("production_training") is not False
        or authorization.get("test_access") is not False
        or protocol.get("test_access") is not False
        or compute.get("partition") != "gpu_4090"
        or compute.get("world_size") != 8
        or arguments.maximum_batch_size
        != compute.get("maximum_batch_pairs_per_rank")
        or arguments.maximum_attention_cells
        != compute.get("maximum_attention_cells_per_rank")
    ):
        raise RuntimeError("Stage-A1 development protocol authorization differs")
    return protocol


def _verify_checkpoint_collectively(
    arguments: argparse.Namespace, protocol: Mapping[str, Any]
) -> dict[str, Any]:
    rank = torch.distributed.get_rank()
    payload: list[dict[str, Any]] = [{"error": None, "receipt": None}]
    if rank == 0:
        try:
            promotion = json.loads(arguments.promotion_result.read_text(encoding="utf-8"))
            selection = protocol["checkpoint_selection"]
            receipt = verify_dcp_generation(arguments.dcp_root, arguments.dcp_generation)
            if (
                promotion.get("status") != selection["required_result_status"]
                or promotion.get("test_access") is not False
                or promotion.get("checkpoint", {}).get("content_sha256")
                != arguments.expected_dcp_content_sha256
                or promotion.get("checkpoint", {}).get("generation")
                != arguments.dcp_generation
                or promotion.get("source_sha256")
                != selection["expected_training_source_sha256"]
                or receipt.get("content_sha256")
                != arguments.expected_dcp_content_sha256
                or receipt.get("generation") != arguments.dcp_generation
                or receipt.get("provenance_hashes", {}).get("protocol")
                != selection["expected_training_protocol_sha256"]
                or receipt.get("provenance_hashes", {}).get("source")
                != selection["expected_training_source_sha256"]
                or receipt.get("training_state", {}).get("production_training")
                is not selection["expected_production_training"]
                or receipt.get("training_state", {}).get("stage_a1_updates")
                != selection["expected_stage_a1_updates"]
                or receipt.get("training_state", {}).get("test_access") is not False
            ):
                raise RuntimeError("Stage-A1 promotion result or DCP receipt differs")
            payload[0]["receipt"] = receipt
        except BaseException as error:
            payload[0]["error"] = f"{type(error).__name__}: {error}"
    torch.distributed.broadcast_object_list(payload, src=0)
    if payload[0]["error"] is not None:
        raise RuntimeError(f"Stage-A1 checkpoint verification failed: {payload[0]['error']}")
    receipt = payload[0]["receipt"]
    if not isinstance(receipt, dict):
        raise RuntimeError("Rank zero did not broadcast the Stage-A1 receipt")
    return receipt


def _load_system_dcp(
    system: StageA1System, *, dcp_root: Path, generation: str
) -> None:
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
        set_model_state_dict,
    )

    options = StateDictOptions(full_state_dict=False, cpu_offload=False, strict=True)
    state = get_model_state_dict(system, options=options)
    dcp.load(
        {"model": state},
        checkpoint_id=dcp_root.resolve() / generation,
        no_dist=False,
    )
    incompatible = set_model_state_dict(system, state, options=options)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("Stage-A1 DCP restore returned incompatible model keys")
    torch.distributed.barrier()


def _token_ids(sequence: str) -> list[int]:
    index = {token: position for position, token in enumerate(ESM2_ALPHABET)}
    unknown = index["<unk>"]
    return [
        index["<cls>"],
        *(index.get(residue, unknown) for residue in sequence),
        index["<eos>"],
    ]


def build_unmasked_batch(
    batch: list[tuple[tuple[str, str, str], str, str]],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    encoded = [(_token_ids(left), _token_ids(right)) for _, left, right in batch]
    maximum = max(len(left) + len(right) for left, right in encoded)
    tokens = torch.full((len(batch), maximum), 1, dtype=torch.long)
    chain_ids = torch.full((len(batch), maximum), -1, dtype=torch.long)
    residue_count = 0
    for row, ((_, left_sequence, right_sequence), (left, right)) in enumerate(
        zip(batch, encoded)
    ):
        left_end = len(left)
        right_end = left_end + len(right)
        tokens[row, :left_end] = torch.tensor(left)
        tokens[row, left_end:right_end] = torch.tensor(right)
        chain_ids[row, :left_end] = 0
        chain_ids[row, left_end:right_end] = 1
        residue_count += len(left_sequence) + len(right_sequence)
    return (
        tokens.to(device, non_blocking=True),
        chain_ids.to(device, non_blocking=True),
        residue_count,
    )


def _aggregate_score_parts(
    parts: list[dict[tuple[str, str, str], tuple[float, int]]]
) -> dict[tuple[str, str, str], float]:
    sums: dict[tuple[str, str, str], float] = {}
    counts: dict[tuple[str, str, str], int] = {}
    for part in parts:
        for key, (value, count) in part.items():
            sums[key] = sums.get(key, 0.0) + value
            counts[key] = counts.get(key, 0) + count
    if not sums or set(sums) != set(counts) or any(count <= 0 for count in counts.values()):
        raise RuntimeError("Distributed Stage-A1 score aggregation is incomplete")
    return {key: sums[key] / counts[key] for key in sums}


def score_pairs(
    *,
    backbone: PairAwareESM2,
    heads: StageA1Heads,
    pairs: dict[tuple[str, str, str], tuple[str, str]],
    maximum_batch_size: int,
    maximum_attention_cells: int,
    device: torch.device,
) -> tuple[dict[tuple[str, str, str], float] | None, dict[str, Any]]:
    work: list[tuple[tuple[str, str, str], str, str]] = []
    expected_windows: dict[tuple[str, str, str], int] = {}
    for key, (left, right) in pairs.items():
        left_windows = endpoint_windows(left)
        right_windows = endpoint_windows(right)
        expected_windows[key] = len(left_windows) * len(right_windows)
        for left_window in left_windows:
            for right_window in right_windows:
                work.append((key, left_window, right_window))
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    local_batches = batch_work(
        work[rank::world_size],
        maximum_batch_size=maximum_batch_size,
        maximum_attention_cells=maximum_attention_cells,
    )
    if not local_batches:
        raise RuntimeError("Every rank requires Stage-A1 retrieval work")
    local_batch_count = torch.tensor(len(local_batches), device=device, dtype=torch.long)
    torch.distributed.all_reduce(local_batch_count, op=torch.distributed.ReduceOp.MAX)
    batch_count = int(local_batch_count.item())
    partial: dict[tuple[str, str, str], tuple[float, int]] = {}
    local_residues = 0
    local_windows = 0
    backbone.eval()
    heads.eval()
    torch.cuda.reset_peak_memory_stats(device)
    torch.distributed.barrier()
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for batch_index in range(batch_count):
        active = batch_index < len(local_batches)
        batch = local_batches[batch_index] if active else local_batches[-1]
        tokens, chain_ids, residues = build_unmasked_batch(batch, device=device)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = backbone(
                tokens,
                chain_ids,
                allow_cross_chain_attention=False,
                return_hidden_states=True,
                compute_logits=False,
            )
            if output.hidden_states is None:
                raise RuntimeError("Stage-A1 retrieval hidden states are missing")
            left, right = heads.retrieval_embeddings(output.hidden_states, chain_ids)
            logits = (left * right).sum(dim=-1) / heads.temperature
        if active:
            local_residues += residues
            local_windows += len(batch)
            for item, score in zip(batch, logits.float().tolist()):
                value, count = partial.get(item[0], (0.0, 0))
                partial[item[0]] = (value + float(score), count + 1)
        if rank == 0 and (batch_index == 0 or (batch_index + 1) % 100 == 0):
            print(
                json.dumps(
                    {"batch": batch_index + 1, "batches": batch_count, "phase": "retrieval"},
                    sort_keys=True,
                ),
                flush=True,
            )
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    gathered: list[dict[tuple[str, str, str], tuple[float, int]]] = [
        {} for _ in range(world_size)
    ]
    torch.distributed.all_gather_object(gathered, partial)
    counters = torch.tensor(
        [local_residues, local_windows, torch.cuda.max_memory_allocated(device)],
        device=device,
        dtype=torch.long,
    )
    all_counters = [torch.zeros_like(counters) for _ in range(world_size)]
    torch.distributed.all_gather(all_counters, counters)
    elapsed_tensor = torch.tensor(elapsed, device=device, dtype=torch.float64)
    torch.distributed.all_reduce(elapsed_tensor, op=torch.distributed.ReduceOp.MAX)
    maximum_elapsed = float(elapsed_tensor.item())
    scores = _aggregate_score_parts(gathered) if rank == 0 else None
    if rank == 0:
        assert scores is not None
        if set(scores) != set(pairs):
            raise RuntimeError("Stage-A1 retrieval pair score set differs")
        observed_counts: dict[tuple[str, str, str], int] = {}
        for part in gathered:
            for key, (_, count) in part.items():
                observed_counts[key] = observed_counts.get(key, 0) + count
        if observed_counts != expected_windows:
            raise RuntimeError("Stage-A1 retrieval window count differs")
    total_residues = sum(int(value[0].item()) for value in all_counters)
    total_windows = sum(int(value[1].item()) for value in all_counters)
    return scores, {
        "elapsed_seconds": maximum_elapsed,
        "gpu_hours": maximum_elapsed * world_size / 3600.0,
        "nonpadding_residues": total_residues,
        "nonpadding_residues_per_second": total_residues / maximum_elapsed,
        "pair_windows": total_windows,
        "pair_windows_per_second": total_windows / maximum_elapsed,
        "peak_cuda_bytes_max_rank": max(int(value[2].item()) for value in all_counters),
        "world_size": world_size,
    }


def _publish_dataset(
    *,
    arguments: argparse.Namespace,
    dataset: str,
    records: list[dict[str, Any]],
    index: dict[str, dict[str, Any]],
    swapped: dict[str, tuple[str, str, str]],
    scores: Mapping[tuple[str, str, str], float],
    runtime: Mapping[str, Any],
    checkpoint_receipt: Mapping[str, Any],
    source_sha256: str,
) -> dict[str, Any]:
    predictions, metrics = evaluate(records, index, swapped, scores)
    for row in predictions:
        row["score"] = row.pop("raw_logit")
        row.pop("probability")
        row["score_semantics"] = "stage_a1_independent_view_dual_encoder_logit"
    prediction_path = arguments.output_root / f"{dataset}.predictions.jsonl"
    payload = b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        for row in predictions
    )
    publish_create_only(prediction_path, payload)
    summary = {
        "authorization": {
            "development_pair_retrieval": True,
            "development_contact": False,
            "production_training": False,
            "test_access": False,
        },
        "checkpoint": {
            "content_sha256": checkpoint_receipt["content_sha256"],
            "generation": checkpoint_receipt["generation"],
            "training_state": checkpoint_receipt["training_state"],
        },
        "claim_boundary": "Frozen-development Stage-A1 retrieval only; contact and test remain sealed.",
        "dataset": dataset,
        "format_version": "abpairfm_stage_a1_development_retrieval_result_v1",
        "metrics": metrics,
        "predictions": {
            "bytes": prediction_path.stat().st_size,
            "row_count": len(predictions),
            "sha256": sha256_file(prediction_path),
        },
        "protocol_sha256": arguments.expected_protocol_sha256,
        "runtime": {
            **runtime,
            "gpu_name": torch.cuda.get_device_name(torch.cuda.current_device()),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
        "source_sha256": source_sha256,
        "status": "FROZEN_DEVELOPMENT_RETRIEVAL_COMPLETE",
        "test_access": False,
    }
    publish_create_only(
        arguments.output_root / f"{dataset}.summary.json",
        (json.dumps(summary, sort_keys=True, separators=(",", ":")) + "\n").encode(
            "utf-8"
        ),
    )
    return summary


def main() -> None:
    arguments = _arguments()
    protocol = _load_protocol(arguments)
    project_root = arguments.project_root.resolve()
    source_sha256 = _source_sha256(project_root)
    if source_sha256 != arguments.expected_source_sha256:
        raise RuntimeError("Reviewed Stage-A1 retrieval source digest differs")
    if os.environ.get("SLURM_JOB_PARTITION") != "gpu_4090":
        raise RuntimeError("Stage-A1 retrieval must run under Slurm gpu_4090")
    if (
        arguments.initial_checkpoint.stat().st_size != EXPECTED_INITIAL_CHECKPOINT_SIZE
        or sha256_file(arguments.initial_checkpoint) != EXPECTED_INITIAL_CHECKPOINT_SHA256
    ):
        raise RuntimeError("Initial ESM2 checkpoint authentication failed")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.distributed.init_process_group(
        backend="nccl", device_id=device, timeout=timedelta(minutes=20)
    )
    rank = torch.distributed.get_rank()
    try:
        if torch.distributed.get_world_size() != 8:
            raise RuntimeError("Stage-A1 retrieval requires exactly eight ranks")
        properties = torch.cuda.get_device_properties(device)
        if properties.name != "NVIDIA GeForce RTX 4090":
            raise RuntimeError(f"Forbidden GPU model: {properties.name}")
        torch.manual_seed(20260824 + rank)
        torch.cuda.manual_seed(20260824 + rank)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

        receipt = _verify_checkpoint_collectively(arguments, protocol)
        from torch.distributed.device_mesh import init_device_mesh

        mesh = init_device_mesh("cuda", (2, 4), mesh_dim_names=("replicate", "shard"))
        with torch.device("meta"):
            backbone = PairAwareESM2(PairAwareESM2Config(activation_checkpointing=False))
        fully_shard_pair_aware_esm2(
            backbone, mesh=mesh, precision="bfloat16", reshard_after_forward=True
        )
        load_authenticated_esm2_into_fsdp2(
            backbone,
            arguments.initial_checkpoint,
            expected_size=EXPECTED_INITIAL_CHECKPOINT_SIZE,
            expected_sha256=EXPECTED_INITIAL_CHECKPOINT_SHA256,
        )
        heads = StageA1Heads(temperature=protocol["score"]["temperature"]).to(
            device=device, dtype=torch.float32
        )
        system = StageA1System(backbone, heads)
        _load_system_dcp(
            system,
            dcp_root=arguments.dcp_root,
            generation=arguments.dcp_generation,
        )

        summaries: dict[str, Any] = {}
        for dataset in protocol["frozen_panel"]["datasets"]:
            records = load_inputs(arguments.pair_panel, arguments.pair_panel_receipt, dataset)
            pairs, index, swapped = retrieval_pairs(records, records)
            scores, runtime = score_pairs(
                backbone=backbone,
                heads=heads,
                pairs=pairs,
                maximum_batch_size=arguments.maximum_batch_size,
                maximum_attention_cells=arguments.maximum_attention_cells,
                device=device,
            )
            if rank == 0:
                assert scores is not None
                summaries[dataset] = _publish_dataset(
                    arguments=arguments,
                    dataset=dataset,
                    records=records,
                    index=index,
                    swapped=swapped,
                    scores=scores,
                    runtime=runtime,
                    checkpoint_receipt=receipt,
                    source_sha256=source_sha256,
                )
            torch.distributed.barrier()
        if rank == 0:
            result = {
                "claim_boundary": protocol["claim_boundary"],
                "datasets": summaries,
                "format_version": "abpairfm_stage_a1_development_retrieval_job_v1",
                "protocol_sha256": arguments.expected_protocol_sha256,
                "source_sha256": source_sha256,
                "status": "FROZEN_DEVELOPMENT_RETRIEVAL_COMPLETE",
                "test_access": False,
            }
            publish_create_only(
                arguments.output_root / "result.json",
                (json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n").encode(
                    "utf-8"
                ),
            )
            print(json.dumps(result, sort_keys=True, separators=(",", ":")), flush=True)
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
