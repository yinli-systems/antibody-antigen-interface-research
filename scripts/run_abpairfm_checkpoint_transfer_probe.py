#!/usr/bin/env python3
"""Evaluate one ABPairFM checkpoint without fitting a task head."""

from __future__ import annotations

import argparse
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import torch
import torch.nn.functional as F


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


EXPECTED_PROTOCOL_FORMAT = "abpairfm_checkpoint_transfer_probe_v1"
EXPECTED_650M_PROTOCOL_FORMAT = "abpairfm_650m_scale_control_v1"
EXPECTED_INITIAL_CHECKPOINT_SHA256 = (
    "7de8b4082ba15891959ab368b77ce3886697af1efb16d3c9e9e7b0c5d3f07500"
)
EXPECTED_INITIAL_CHECKPOINT_SIZE = 5_678_116_398
SOURCE_PATHS = (
    "scripts/run_abpairfm_checkpoint_transfer_probe.py",
    "scripts/run_mint_frozen_development_retrieval.py",
    "src/antibody_pretraining/pair_aware_attention.py",
    "src/antibody_pretraining/pair_aware_distributed.py",
    "src/antibody_pretraining/pair_aware_esm2.py",
    "src/interaction_first/frontier_fsdp2.py",
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-variant",
        choices=("base_esm2_3b", "base_esm2_650m", "stage_a0_3b"),
        required=True,
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--expected-initial-checkpoint-size",
        type=int,
        default=EXPECTED_INITIAL_CHECKPOINT_SIZE,
    )
    parser.add_argument(
        "--expected-initial-checkpoint-sha256",
        default=EXPECTED_INITIAL_CHECKPOINT_SHA256,
    )
    parser.add_argument("--layer-count", type=int, default=36)
    parser.add_argument("--embed-dimension", type=int, default=2560)
    parser.add_argument("--head-count", type=int, default=40)
    parser.add_argument("--ffn-dimension", type=int, default=10240)
    parser.add_argument("--pair-panel", type=Path, required=True)
    parser.add_argument("--pair-panel-receipt", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint-target-residue-tokens", type=int)
    parser.add_argument("--dcp-root", type=Path)
    parser.add_argument("--dcp-generation")
    parser.add_argument("--expected-dcp-content-sha256")
    parser.add_argument("--maximum-batch-size", type=int, default=7)
    parser.add_argument("--maximum-attention-cells", type=int, default=7_340_032)
    parser.add_argument("--smoke-anchor-count", type=int, default=0)
    return parser.parse_args()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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
        raise RuntimeError("Transfer-probe protocol SHA-256 differs")
    protocol = json.loads(arguments.protocol.read_text(encoding="utf-8"))
    authorization = protocol.get("authorization", {})
    compute = protocol.get("compute", {})
    expected_format = (
        EXPECTED_650M_PROTOCOL_FORMAT
        if arguments.model_variant == "base_esm2_650m"
        else EXPECTED_PROTOCOL_FORMAT
    )
    if (
        protocol.get("format_version") != expected_format
        or protocol.get("status") != "FROZEN_BEFORE_GPU_EVALUATION"
        or authorization.get("development_pair_retrieval") is not True
        or authorization.get("development_contact") is not False
        or authorization.get("production_training") is not False
        or authorization.get("test_access") is not False
        or protocol.get("test_access") is not False
        or compute.get("partition") != "gpu_4090"
        or compute.get("world_size") != 8
    ):
        raise RuntimeError("Transfer-probe protocol authorization differs")
    if arguments.maximum_batch_size != compute.get("maximum_batch_pairs_per_rank"):
        raise RuntimeError("Transfer-probe batch size differs from the protocol")
    if arguments.maximum_attention_cells != compute.get(
        "maximum_attention_cells_per_rank"
    ):
        raise RuntimeError("Transfer-probe attention budget differs from the protocol")
    return protocol


def _validate_arguments(arguments: argparse.Namespace, protocol: Mapping[str, Any]) -> None:
    if arguments.smoke_anchor_count < 0:
        raise ValueError("smoke-anchor-count cannot be negative")
    dcp_values = (
        arguments.checkpoint_target_residue_tokens,
        arguments.dcp_root,
        arguments.dcp_generation,
        arguments.expected_dcp_content_sha256,
    )
    if arguments.model_variant in {"base_esm2_3b", "base_esm2_650m"}:
        if any(value is not None for value in dcp_values):
            raise ValueError("The base ESM2 control cannot receive a Stage-A0 DCP")
    elif not all(value is not None for value in dcp_values):
        raise ValueError("A Stage-A0 probe requires a complete immutable DCP anchor")
    if arguments.checkpoint_target_residue_tokens is not None:
        targets = protocol["checkpoint_ladder"]["targets"]
        if arguments.checkpoint_target_residue_tokens not in targets:
            raise ValueError("Checkpoint target is absent from the frozen ladder")
    expected_dimensions = (
        (33, 1280, 20, 5120)
        if arguments.model_variant == "base_esm2_650m"
        else (36, 2560, 40, 10240)
    )
    observed_dimensions = (
        arguments.layer_count,
        arguments.embed_dimension,
        arguments.head_count,
        arguments.ffn_dimension,
    )
    if observed_dimensions != expected_dimensions:
        raise ValueError("Transfer-probe model dimensions differ")
    control = (
        protocol.get("model_control", {})
        if arguments.model_variant == "base_esm2_650m"
        else protocol.get("model_controls", {}).get(arguments.model_variant, {})
    )
    if (
        control.get("checkpoint_sha256")
        not in {None, arguments.expected_initial_checkpoint_sha256}
        or control.get("checkpoint_size_bytes")
        not in {None, arguments.expected_initial_checkpoint_size}
    ):
        raise ValueError("Transfer-probe checkpoint anchor differs from the protocol")
    for digest in (
        arguments.expected_protocol_sha256,
        arguments.expected_source_sha256,
        arguments.expected_dcp_content_sha256,
        arguments.expected_initial_checkpoint_sha256,
    ):
        if digest is not None and (
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("Expected SHA-256 is malformed")


def _verify_dcp_collectively(
    arguments: argparse.Namespace, protocol: Mapping[str, Any]
) -> dict[str, Any]:
    assert arguments.dcp_root is not None
    assert arguments.dcp_generation is not None
    assert arguments.expected_dcp_content_sha256 is not None
    assert arguments.checkpoint_target_residue_tokens is not None
    rank = torch.distributed.get_rank()
    payload: list[dict[str, Any]] = [{"error": None, "receipt": None}]
    if rank == 0:
        try:
            receipt = verify_dcp_generation(arguments.dcp_root, arguments.dcp_generation)
            target = arguments.checkpoint_target_residue_tokens
            consumed = receipt.get("training_state", {}).get("consumed_residue_tokens")
            maximum_overshoot = protocol["checkpoint_ladder"][
                "maximum_target_overshoot_residue_tokens"
            ]
            if receipt.get("content_sha256") != arguments.expected_dcp_content_sha256:
                raise RuntimeError("DCP content SHA-256 differs from the launch anchor")
            if (
                not isinstance(consumed, int)
                or consumed < target
                or consumed - target > maximum_overshoot
                or receipt.get("training_state", {}).get("test_access") is not False
            ):
                raise RuntimeError("DCP token cursor is outside the frozen target window")
            payload[0]["receipt"] = receipt
        except BaseException as error:
            payload[0]["error"] = f"{type(error).__name__}: {error}"
    torch.distributed.broadcast_object_list(payload, src=0)
    if payload[0]["error"] is not None:
        raise RuntimeError(f"Immutable DCP verification failed: {payload[0]['error']}")
    receipt = payload[0]["receipt"]
    if not isinstance(receipt, dict):
        raise RuntimeError("Rank zero did not broadcast an immutable DCP receipt")
    return receipt


def _load_model_only_dcp(
    model: PairAwareESM2,
    *,
    dcp_root: Path,
    generation: str,
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
        checkpoint_id=dcp_root.resolve() / generation,
        no_dist=False,
    )
    incompatible = set_model_state_dict(model, state, options=options)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Model-only DCP restore returned incompatible keys: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    torch.distributed.barrier()


def _token_ids(sequence: str) -> list[int]:
    index = {token: position for position, token in enumerate(ESM2_ALPHABET)}
    unknown = index["<unk>"]
    return [index["<cls>"], *(index.get(residue, unknown) for residue in sequence), index["<eos>"]]


def _selected_positions(
    *,
    sequence: str,
    pair_key: tuple[str, str, str],
    endpoint: str,
    mask_seed: int,
) -> list[int]:
    count = max(1, int(math.ceil(len(sequence) * 0.15)))
    framed_key = "|".join(pair_key)
    ranked = sorted(
        range(len(sequence)),
        key=lambda position: _sha256_text(
            f"{mask_seed}|{framed_key}|{endpoint}|{sequence}|{position}"
        ),
    )
    return sorted(ranked[:count])


def _build_batch(
    batch: list[tuple[tuple[str, str, str], str, str]],
    *,
    pair_aware: bool,
    mask_seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    encoded = [(_token_ids(left), _token_ids(right)) for _, left, right in batch]
    maximum = max(len(left) + len(right) for left, right in encoded)
    tokens = torch.full((len(batch), maximum), 1, dtype=torch.long)
    chain_ids = torch.full((len(batch), maximum), -1, dtype=torch.int32)
    endpoint_ids = torch.full((len(batch), maximum), -1, dtype=torch.int8)
    labels = torch.full((len(batch), maximum), -100, dtype=torch.long)
    nonpadding_residues = 0
    for row, ((key, left_sequence, right_sequence), (left, right)) in enumerate(
        zip(batch, encoded)
    ):
        left_end = len(left)
        right_end = left_end + len(right)
        tokens[row, :left_end] = torch.tensor(left)
        tokens[row, left_end:right_end] = torch.tensor(right)
        chain_ids[row, :right_end] = 0
        if pair_aware:
            chain_ids[row, left_end:right_end] = 1
        endpoint_ids[row, :left_end] = 0
        endpoint_ids[row, left_end:right_end] = 1
        for position in _selected_positions(
            sequence=left_sequence,
            pair_key=key,
            endpoint="left",
            mask_seed=mask_seed,
        ):
            token_position = 1 + position
            labels[row, token_position] = tokens[row, token_position]
            tokens[row, token_position] = 32
        for position in _selected_positions(
            sequence=right_sequence,
            pair_key=key,
            endpoint="right",
            mask_seed=mask_seed,
        ):
            token_position = left_end + 1 + position
            labels[row, token_position] = tokens[row, token_position]
            tokens[row, token_position] = 32
        nonpadding_residues += len(left_sequence) + len(right_sequence)
    return (
        tokens.to(device, non_blocking=True),
        chain_ids.to(device, non_blocking=True),
        endpoint_ids.to(device, non_blocking=True),
        labels.to(device, non_blocking=True),
        nonpadding_residues,
    )


def _row_scores(logits: torch.Tensor, labels: torch.Tensor, endpoint_ids: torch.Tensor) -> list[float]:
    losses = F.cross_entropy(
        logits.float().flatten(0, 1),
        labels.flatten(),
        ignore_index=-100,
        reduction="none",
    ).view_as(labels)
    scores: list[float] = []
    selected = labels.ne(-100)
    for row in range(labels.shape[0]):
        endpoint_losses: list[torch.Tensor] = []
        for endpoint in (0, 1):
            mask = selected[row] & endpoint_ids[row].eq(endpoint)
            if not bool(mask.any()):
                raise RuntimeError("Every endpoint requires at least one deterministic mask")
            endpoint_losses.append(losses[row, mask].mean())
        scores.append(float(-torch.stack(endpoint_losses).mean().item()))
    return scores


def _aggregate_score_parts(
    parts: list[dict[tuple[str, str, str], tuple[float, int]]]
) -> dict[tuple[str, str, str], float]:
    sums: dict[tuple[str, str, str], float] = {}
    counts: dict[tuple[str, str, str], int] = {}
    for part in parts:
        for key, (value, count) in part.items():
            sums[key] = sums.get(key, 0.0) + float(value)
            counts[key] = counts.get(key, 0) + int(count)
    if not sums or set(sums) != set(counts) or any(count <= 0 for count in counts.values()):
        raise RuntimeError("Distributed transfer-probe score aggregation is incomplete")
    return {key: sums[key] / counts[key] for key in sums}


def _score_pairs(
    *,
    model: PairAwareESM2,
    pairs: dict[tuple[str, str, str], tuple[str, str]],
    pair_aware: bool,
    mask_seed: int,
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
    local_work = work[rank::world_size]
    local_batches = batch_work(
        local_work,
        maximum_batch_size=maximum_batch_size,
        maximum_attention_cells=maximum_attention_cells,
    )
    if not local_batches:
        raise RuntimeError("Every rank requires transfer-probe work")
    local_batch_count = torch.tensor(len(local_batches), device=device, dtype=torch.long)
    torch.distributed.all_reduce(local_batch_count, op=torch.distributed.ReduceOp.MAX)
    batch_count = int(local_batch_count.item())
    partial: dict[tuple[str, str, str], tuple[float, int]] = {}
    local_residues = 0
    local_windows = 0
    model.eval()
    torch.cuda.reset_peak_memory_stats(device)
    torch.distributed.barrier()
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    from torch.nn.attention import SDPBackend, sdpa_kernel

    for batch_index in range(batch_count):
        active = batch_index < len(local_batches)
        batch = local_batches[batch_index] if active else local_batches[-1]
        tokens, chain_ids, endpoint_ids, labels, residues = _build_batch(
            batch,
            pair_aware=pair_aware,
            mask_seed=mask_seed,
            device=device,
        )
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                output = model(tokens, chain_ids)
        scores = _row_scores(output.logits, labels, endpoint_ids)
        if active:
            local_residues += residues
            local_windows += len(batch)
            for item, score in zip(batch, scores):
                key = item[0]
                value, count = partial.get(key, (0.0, 0))
                partial[key] = (value + score, count + 1)
        if rank == 0 and (batch_index == 0 or (batch_index + 1) % 100 == 0):
            print(
                json.dumps(
                    {
                        "batch": batch_index + 1,
                        "batches": batch_count,
                        "phase": "transfer_probe",
                    },
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
            raise RuntimeError("Transfer-probe pair score set differs")
        observed_counts: dict[tuple[str, str, str], int] = {}
        for part in gathered:
            for key, (_, count) in part.items():
                observed_counts[key] = observed_counts.get(key, 0) + count
        if observed_counts != expected_windows:
            raise RuntimeError("Transfer-probe pair window count differs")
    total_residues = sum(int(value[0].item()) for value in all_counters)
    total_windows = sum(int(value[1].item()) for value in all_counters)
    runtime = {
        "elapsed_seconds": maximum_elapsed,
        "gpu_hours": maximum_elapsed * world_size / 3600.0,
        "nonpadding_residues": total_residues,
        "nonpadding_residues_per_second": total_residues / maximum_elapsed,
        "pair_windows": total_windows,
        "pair_windows_per_second": total_windows / maximum_elapsed,
        "peak_cuda_bytes_max_rank": max(int(value[2].item()) for value in all_counters),
        "world_size": world_size,
    }
    return scores, runtime


def _publish_dataset(
    *,
    arguments: argparse.Namespace,
    dataset: str,
    records: list[dict[str, Any]],
    index: dict[str, dict[str, Any]],
    swapped: dict[str, tuple[str, str, str]],
    scores: Mapping[tuple[str, str, str], float],
    runtime: Mapping[str, Any],
    checkpoint_receipt: Mapping[str, Any] | None,
    source_sha256: str,
) -> dict[str, Any]:
    predictions, metrics = evaluate(records, index, swapped, scores)
    for row in predictions:
        row["score"] = row.pop("raw_logit")
        row.pop("probability")
        row["score_semantics"] = "uncalibrated_masked_chain_conditional_log_likelihood"
    prediction_path = arguments.output_root / f"{dataset}.predictions.jsonl"
    prediction_payload = b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        for row in predictions
    )
    publish_create_only(prediction_path, prediction_payload)
    summary = {
        "authorization": {
            "development_pair_retrieval": arguments.smoke_anchor_count == 0,
            "development_contact": False,
            "production_training": False,
            "test_access": False,
        },
        "checkpoint": (
            None
            if checkpoint_receipt is None
            else {
                "content_sha256": checkpoint_receipt["content_sha256"],
                "generation": checkpoint_receipt["generation"],
                "training_state": checkpoint_receipt["training_state"],
            }
        ),
        "claim_boundary": (
            "Deterministic smoke only; no quality conclusion is authorized."
            if arguments.smoke_anchor_count
            else "Frozen-development head-free retrieval probe. Scores are uncalibrated and decoys are ranking controls; contact, test, biological-improvement, and SOTA claims remain closed."
        ),
        "dataset": dataset,
        "format_version": "abpairfm_checkpoint_transfer_probe_result_v1",
        "input": {
            "anchor_instance_count": len(records),
            "unique_canonical_exact_pair_count": len(
                {record["canonical_pair_sha256"] for record in records}
            ),
        },
        "metrics": metrics,
        "model_variant": arguments.model_variant,
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
        "status": (
            "SMOKE_NONCLAIM_COMPLETE"
            if arguments.smoke_anchor_count
            else "FROZEN_DEVELOPMENT_RETRIEVAL_COMPLETE"
        ),
        "test_access": False,
    }
    summary_path = arguments.output_root / f"{dataset}.summary.json"
    publish_create_only(
        summary_path,
        (json.dumps(summary, sort_keys=True, separators=(",", ":")) + "\n").encode(
            "utf-8"
        ),
    )
    return summary


def main() -> None:
    arguments = _arguments()
    project_root = arguments.project_root.resolve()
    protocol = _load_protocol(arguments)
    _validate_arguments(arguments, protocol)
    if os.environ.get("SLURM_JOB_PARTITION") != "gpu_4090":
        raise RuntimeError("Transfer probes must run under Slurm gpu_4090")
    source_sha256 = _source_sha256(project_root)
    if source_sha256 != arguments.expected_source_sha256:
        raise RuntimeError("Reviewed transfer-probe source digest differs")
    if (
        arguments.initial_checkpoint.stat().st_size
        != arguments.expected_initial_checkpoint_size
        or sha256_file(arguments.initial_checkpoint)
        != arguments.expected_initial_checkpoint_sha256
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
        world_size = torch.distributed.get_world_size()
        if world_size != 8:
            raise RuntimeError("Transfer probe requires exactly eight ranks")
        properties = torch.cuda.get_device_properties(device)
        if properties.name != "NVIDIA GeForce RTX 4090":
            raise RuntimeError(f"Forbidden GPU model: {properties.name}")
        torch.manual_seed(1701 + rank)
        torch.cuda.manual_seed(1701 + rank)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

        from torch.distributed.device_mesh import init_device_mesh

        mesh = init_device_mesh("cuda", (2, 4), mesh_dim_names=("replicate", "shard"))
        with torch.device("meta"):
            model = PairAwareESM2(
                PairAwareESM2Config(
                    layer_count=arguments.layer_count,
                    embed_dimension=arguments.embed_dimension,
                    head_count=arguments.head_count,
                    ffn_dimension=arguments.ffn_dimension,
                    activation_checkpointing=False,
                )
            )
        fully_shard_pair_aware_esm2(
            model, mesh=mesh, precision="bfloat16", reshard_after_forward=True
        )
        load_authenticated_esm2_into_fsdp2(
            model,
            arguments.initial_checkpoint,
            expected_size=arguments.expected_initial_checkpoint_size,
            expected_sha256=arguments.expected_initial_checkpoint_sha256,
        )
        checkpoint_receipt = None
        if arguments.model_variant == "stage_a0_3b":
            checkpoint_receipt = _verify_dcp_collectively(arguments, protocol)
            assert arguments.dcp_root is not None
            assert arguments.dcp_generation is not None
            _load_model_only_dcp(
                model,
                dcp_root=arguments.dcp_root,
                generation=arguments.dcp_generation,
            )

        summaries: dict[str, Any] = {}
        for dataset in protocol["frozen_panel"]["datasets"]:
            authority_records = load_inputs(
                arguments.pair_panel, arguments.pair_panel_receipt, dataset
            )
            records = (
                authority_records[: arguments.smoke_anchor_count]
                if arguments.smoke_anchor_count
                else authority_records
            )
            pairs, index, swapped = retrieval_pairs(records, authority_records)
            scores, runtime = _score_pairs(
                model=model,
                pairs=pairs,
                pair_aware=arguments.model_variant == "stage_a0_3b",
                mask_seed=protocol["score"]["mask_seed"],
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
                    checkpoint_receipt=checkpoint_receipt,
                    source_sha256=source_sha256,
                )
            torch.distributed.barrier()
        if rank == 0:
            result = {
                "claim_boundary": "Development-only transfer probe; test remains sealed.",
                "datasets": summaries,
                "format_version": "abpairfm_checkpoint_transfer_probe_job_v1",
                "model_variant": arguments.model_variant,
                "protocol_sha256": arguments.expected_protocol_sha256,
                "source_sha256": source_sha256,
                "status": (
                    "SMOKE_NONCLAIM_COMPLETE"
                    if arguments.smoke_anchor_count
                    else "FROZEN_DEVELOPMENT_RETRIEVAL_COMPLETE"
                ),
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
