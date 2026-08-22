#!/usr/bin/env python3
"""Run the frozen 200-update ABPairFM 3B train-only real-data gate."""

from __future__ import annotations

import argparse
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import statistics
import subprocess
import tempfile
import time
from typing import Any

import torch

from antibody_pretraining.pair_aware_distributed import (
    fully_shard_pair_aware_esm2,
    load_authenticated_esm2_into_fsdp2,
    optimizer_after_fsdp2,
)
from antibody_pretraining.pair_aware_esm2 import (
    ESM2_ALPHABET,
    PairAwareESM2,
    PairAwareESM2Config,
    reset_selective_attention_checkpoint_policy_counts,
    selective_attention_checkpoint_policy_counts,
)


CLAIM_BOUNDARY = (
    "Train-only development-decontaminated STRING numerical and systems evidence "
    "using the full 2.839B-parameter model. It does not authorize production "
    "pretraining or establish validation, test, biological, or SOTA performance."
)
TOKEN_INDEX = {token: index for index, token in enumerate(ESM2_ALPHABET)}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-size", type=int, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--expected-world-size", type=int, default=8)
    parser.add_argument("--shard-group-size", type=int, default=4)
    parser.add_argument("--microbatch-pairs", type=int, default=8)
    parser.add_argument("--accumulation-steps", type=int, default=2)
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--maximum-tokens", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--progress-interval", type=int, default=10)
    parser.add_argument(
        "--activation-checkpoint-policy",
        choices=("full", "selective_attention"),
        default="full",
    )
    return parser.parse_args()


def _validate(arguments: argparse.Namespace) -> None:
    if arguments.expected_world_size != 8 or arguments.shard_group_size != 4:
        raise ValueError("real-data gate requires the promoted eight-rank HSDP 2x4 mesh")
    if arguments.microbatch_pairs not in {7, 8} or arguments.accumulation_steps != 2:
        raise ValueError("real-data gate requires microbatch 7 or 8 and frozen accumulation")
    if arguments.updates != 200 or arguments.maximum_tokens != 1024:
        raise ValueError("real-data gate requires 200 updates and 1024-token pairs")
    if arguments.learning_rate != 1e-6 or arguments.weight_decay != 0.01:
        raise ValueError("real-data gate optimizer differs from the frozen candidate")
    if arguments.seed != 1701 or arguments.progress_interval < 1:
        raise ValueError("real-data gate seed or progress interval differs")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _zstd_lines(path: Path):
    process = subprocess.Popen(
        ["zstd", "-q", "-dc", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    assert process.stdout is not None
    try:
        yield from process.stdout
    finally:
        process.stdout.close()
        error = process.stderr.read() if process.stderr is not None else ""
        return_code = process.wait()
        if return_code:
            raise RuntimeError(f"zstd failed with {return_code}: {error[-1000:]}")


def _load_manifest(path: Path, receipt_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("format_version") != "abpairfm_3b_real_gate_slice_v1"
        or receipt.get("status") != "REAL_GATE_SLICE_READY_NOT_TRAINED"
        or receipt.get("test_access") is not False
        or receipt.get("authorization", {}).get("real_data_gpu_gate") is not True
        or receipt["output"]["sha256"] != _sha256(path)
    ):
        raise ValueError("ABPairFM real-gate manifest evidence differs")
    records: list[dict[str, Any]] = []
    pair_hashes: set[str] = set()
    for line_number, line in enumerate(_zstd_lines(path), start=1):
        value = json.loads(line)
        if not isinstance(value, dict) or value.get("selection_index") != line_number - 1:
            raise ValueError("Real-gate manifest ordering differs")
        pair_hash = value.get("pair_hash_sha256")
        sequence1 = value.get("sequence1")
        sequence2 = value.get("sequence2")
        if (
            not isinstance(pair_hash, str)
            or len(pair_hash) != 64
            or pair_hash in pair_hashes
            or not isinstance(sequence1, str)
            or not isinstance(sequence2, str)
            or min(len(sequence1), len(sequence2)) < 20
        ):
            raise ValueError("Real-gate manifest record differs")
        pair_hashes.add(pair_hash)
        records.append(value)
    if len(records) != int(receipt["counts"]["manifest_records"]):
        raise ValueError("Real-gate manifest count differs")
    return records, receipt


def _seed(*values: object) -> int:
    payload = "\0".join(str(value) for value in values).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def _crop_sequence(sequence: str, length: int, *, seed: int) -> str:
    if length > len(sequence) or length < 1:
        raise ValueError("Invalid deterministic crop length")
    if length == len(sequence):
        return sequence
    start = seed % (len(sequence) - length + 1)
    return sequence[start : start + length]


def _crop_pair(
    sequence1: str,
    sequence2: str,
    *,
    maximum_residues: int,
    seed: int,
) -> tuple[str, str]:
    if min(len(sequence1), len(sequence2)) < 20 or maximum_residues < 40:
        raise ValueError("Pair crop cannot preserve both chains")
    if len(sequence1) + len(sequence2) <= maximum_residues:
        return sequence1, sequence2
    length1 = round(maximum_residues * len(sequence1) / (len(sequence1) + len(sequence2)))
    length1 = max(20, min(length1, len(sequence1), maximum_residues - 20))
    length2 = min(len(sequence2), maximum_residues - length1)
    if length2 < 20:
        length2 = 20
        length1 = min(len(sequence1), maximum_residues - length2)
    remaining = maximum_residues - length1 - length2
    if remaining > 0:
        add1 = min(remaining, len(sequence1) - length1)
        length1 += add1
        remaining -= add1
        length2 += min(remaining, len(sequence2) - length2)
    if length1 + length2 != maximum_residues:
        raise ValueError("Deterministic pair crop did not fill the token budget")
    return (
        _crop_sequence(sequence1, length1, seed=_seed(seed, "chain1")),
        _crop_sequence(sequence2, length2, seed=_seed(seed, "chain2")),
    )


def _tokenize(sequence: str) -> list[int]:
    if not sequence or any(character.isspace() for character in sequence):
        raise ValueError("Malformed protein sequence")
    return [TOKEN_INDEX.get(character, TOKEN_INDEX["<unk>"]) for character in sequence]


def _build_real_microbatch(
    records: list[dict[str, Any]],
    indices: list[int],
    *,
    maximum_tokens: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    batch_size = len(indices)
    tokens = torch.full((batch_size, maximum_tokens), TOKEN_INDEX["<pad>"], dtype=torch.long)
    chain_ids = torch.full((batch_size, maximum_tokens), -1, dtype=torch.long)
    labels = torch.full((batch_size, maximum_tokens), -100, dtype=torch.long)
    total_residues = 0
    swapped_pairs = 0
    for batch_index, record_index in enumerate(indices):
        record = records[record_index]
        sequence1 = str(record["sequence1"])
        sequence2 = str(record["sequence2"])
        augmentation_seed = _seed(seed, record_index, "pair_order")
        if augmentation_seed % 2:
            sequence1, sequence2 = sequence2, sequence1
            swapped_pairs += 1
        sequence1, sequence2 = _crop_pair(
            sequence1,
            sequence2,
            maximum_residues=maximum_tokens - 4,
            seed=_seed(seed, record_index, "crop"),
        )
        first = [TOKEN_INDEX["<cls>"], *_tokenize(sequence1), TOKEN_INDEX["<eos>"]]
        second = [TOKEN_INDEX["<cls>"], *_tokenize(sequence2), TOKEN_INDEX["<eos>"]]
        combined = first + second
        tokens[batch_index, : len(combined)] = torch.tensor(combined)
        chain_ids[batch_index, : len(first)] = 0
        chain_ids[batch_index, len(first) : len(combined)] = 1
        eligible = torch.zeros(maximum_tokens, dtype=torch.bool)
        eligible[1 : len(first) - 1] = True
        eligible[len(first) + 1 : len(combined) - 1] = True
        generator = torch.Generator(device="cpu")
        generator.manual_seed(_seed(seed, record_index, "mask"))
        selected = eligible & (torch.rand(maximum_tokens, generator=generator) < 0.15)
        for start, end in ((1, len(first) - 1), (len(first) + 1, len(combined) - 1)):
            if not bool(selected[start:end].any()):
                selected[start + int(torch.randint(end - start, (1,), generator=generator))] = True
        labels[batch_index, selected] = tokens[batch_index, selected]
        replacement = torch.rand(maximum_tokens, generator=generator)
        tokens[batch_index, selected & replacement.lt(0.8)] = TOKEN_INDEX["<mask>"]
        random_positions = selected & replacement.ge(0.8) & replacement.lt(0.9)
        random_count = int(random_positions.sum())
        if random_count:
            tokens[batch_index, random_positions] = torch.randint(
                TOKEN_INDEX["L"],
                TOKEN_INDEX["C"] + 1,
                (random_count,),
                generator=generator,
            )
        total_residues += len(sequence1) + len(sequence2)
    return tokens, chain_ids, labels, total_residues, swapped_pairs


def _collect(value: object, world_size: int) -> list[object]:
    gathered: list[object] = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(gathered, value)
    return gathered


def _publish_create_only(path: Path, value: object) -> None:
    payload = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_path, path)
    except FileExistsError as error:
        raise FileExistsError(f"Refusing to overwrite real-gate receipt: {path}") from error
    finally:
        temporary_path.unlink(missing_ok=True)


def _publish_progress(path: Path, value: object) -> None:
    payload = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _nearest_rank_p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


def main() -> None:
    arguments = _arguments()
    _validate(arguments)
    manifest = arguments.manifest.resolve()
    manifest_receipt_path = arguments.manifest_receipt.resolve()
    records, manifest_receipt = _load_manifest(manifest, manifest_receipt_path)
    required_records = (
        arguments.updates
        * arguments.accumulation_steps
        * arguments.expected_world_size
        * arguments.microbatch_pairs
    )
    if len(records) != required_records:
        raise ValueError(f"Real-gate manifest must contain exactly {required_records} records")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.distributed.init_process_group(
        backend="nccl",
        device_id=device,
        timeout=timedelta(minutes=10),
    )
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    try:
        if world_size != arguments.expected_world_size:
            raise RuntimeError("Real-gate world size differs")
        properties = torch.cuda.get_device_properties(device)
        if properties.name != "NVIDIA GeForce RTX 4090":
            raise RuntimeError(f"Forbidden GPU model: {properties.name}")
        torch.manual_seed(arguments.seed + rank)
        torch.cuda.manual_seed(arguments.seed + rank)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

        from torch.distributed.device_mesh import init_device_mesh

        mesh = init_device_mesh(
            "cuda",
            (world_size // arguments.shard_group_size, arguments.shard_group_size),
            mesh_dim_names=("replicate", "shard"),
        )
        config = PairAwareESM2Config(
            activation_checkpointing=True,
            activation_checkpoint_policy=arguments.activation_checkpoint_policy,
        )
        with torch.device("meta"):
            model = PairAwareESM2(config)
        fully_shard_pair_aware_esm2(
            model,
            mesh=mesh,
            precision="bfloat16",
            reshard_after_forward=False,
        )
        initialization = load_authenticated_esm2_into_fsdp2(
            model,
            arguments.checkpoint,
            expected_size=arguments.expected_size,
            expected_sha256=arguments.expected_sha256,
        )
        optimizer = optimizer_after_fsdp2(
            model,
            learning_rate=arguments.learning_rate,
            weight_decay=arguments.weight_decay,
            betas=(0.9, 0.98),
        )
        if not hasattr(model, "set_requires_gradient_sync") or not hasattr(
            model, "set_is_last_backward"
        ):
            raise RuntimeError("FSDP2 gradient-accumulation controls are unavailable")
        model.train()
        reset_selective_attention_checkpoint_policy_counts()
        torch.cuda.reset_peak_memory_stats(device)
        update_records: list[dict[str, float | int]] = []
        local_swapped = 0
        from torch.nn.attention import SDPBackend, sdpa_kernel

        for update in range(arguments.updates):
            optimizer.zero_grad(set_to_none=True)
            torch.distributed.barrier()
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            update_loss = 0.0
            update_residues = 0
            for accumulation in range(arguments.accumulation_steps):
                microstep = update * arguments.accumulation_steps + accumulation
                start = (
                    microstep * world_size * arguments.microbatch_pairs
                    + rank * arguments.microbatch_pairs
                )
                indices = list(range(start, start + arguments.microbatch_pairs))
                cpu_tokens, cpu_chain_ids, cpu_labels, local_residues, swapped = (
                    _build_real_microbatch(
                        records,
                        indices,
                        maximum_tokens=arguments.maximum_tokens,
                        seed=arguments.seed,
                    )
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
                        raise RuntimeError("Real MLM model did not return a loss")
                    local_chain_count = output.contributing_chain_count.clone()
                    global_chain_count = local_chain_count.clone()
                    torch.distributed.all_reduce(global_chain_count)
                    loss = (
                        output.balanced_chain_loss_sum
                        * world_size
                        / global_chain_count.to(output.balanced_chain_loss_sum.dtype)
                        / arguments.accumulation_steps
                    )
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("Non-finite real-data MLM loss")
                loss.backward()
                detached_loss = output.balanced_chain_loss_sum.detach()
                torch.distributed.all_reduce(detached_loss)
                update_loss += float(detached_loss / global_chain_count) / arguments.accumulation_steps
                residue_tensor = torch.tensor(local_residues, device=device, dtype=torch.long)
                torch.distributed.all_reduce(residue_tensor)
                update_residues += int(residue_tensor)
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not bool(torch.isfinite(gradient_norm)):
                raise RuntimeError("Non-finite real-data gradient norm")
            optimizer.step()
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            update_records.append(
                {
                    "elapsed_seconds": elapsed,
                    "global_mean_mlm_loss": update_loss,
                    "global_residue_tokens": update_residues,
                    "gradient_norm": float(gradient_norm),
                    "update": update + 1,
                }
            )
            if rank == 0 and (
                (update + 1) % arguments.progress_interval == 0
                or update + 1 == arguments.updates
            ):
                _publish_progress(
                    arguments.progress.resolve(),
                    {
                        "completed_updates": update + 1,
                        "format_version": "abpairfm_3b_real_data_gate_progress_v1",
                        "latest_global_mean_mlm_loss": update_loss,
                        "latest_gradient_norm": float(gradient_norm),
                        "manifest_records_consumed": (
                            (update + 1)
                            * arguments.accumulation_steps
                            * world_size
                            * arguments.microbatch_pairs
                        ),
                        "status": "RUNNING" if update + 1 < arguments.updates else "TRAINING_COMPLETE_AUDIT_PENDING",
                        "test_access": False,
                    },
                )

        rank_record = {
            "device_index": local_rank,
            "device_name": properties.name,
            "hostname": socket.gethostname(),
            "local_manifest_records": (
                arguments.updates
                * arguments.accumulation_steps
                * arguments.microbatch_pairs
            ),
            "local_pair_order_swaps": local_swapped,
            "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "max_memory_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "rank": rank,
            "selective_checkpoint_dispatch": (
                selective_attention_checkpoint_policy_counts()
            ),
            "total_memory_bytes": properties.total_memory,
            "updates": update_records,
        }
        ranks = _collect(rank_record, world_size)
        if rank == 0:
            elapsed_maxima = [
                max(float(record["updates"][index]["elapsed_seconds"]) for record in ranks)  # type: ignore[index]
                for index in range(arguments.updates)
            ]
            global_residues = [
                int(update_records[index]["global_residue_tokens"])
                for index in range(arguments.updates)
            ]
            throughputs = [
                residues / elapsed for residues, elapsed in zip(global_residues, elapsed_maxima, strict=True)
            ]
            losses = [float(record["global_mean_mlm_loss"]) for record in update_records]
            gradients = [float(record["gradient_norm"]) for record in update_records]
            payload = {
                "authorization": {
                    "production_pretraining": False,
                    "real_data_gpu_gate_passed": True,
                },
                "claim_boundary": CLAIM_BOUNDARY,
                "data": {
                    "development_decontaminated": True,
                    "duplicate_manifest_records": 0,
                    "manifest_records_consumed": required_records,
                    "manifest_records_expected": required_records,
                    "manifest_sha256": _sha256(manifest),
                    "manifest_skipped_records": 0,
                    "receipt_sha256": _sha256(manifest_receipt_path),
                    "source_cluster_pairs": manifest_receipt["counts"]["source_cluster_pairs"],
                },
                "format_version": "abpairfm_3b_real_data_gate_v2",
                "fsdp2": {
                    "mesh_shape_replicate_by_shard": [2, 4],
                    "reshard_after_forward": False,
                },
                "initialization": initialization,
                "loss": {
                    "final_20_update_mean": statistics.mean(losses[-20:]),
                    "first_20_update_mean": statistics.mean(losses[:20]),
                    "maximum": max(losses),
                    "minimum": min(losses),
                    "nonfinite_updates": 0,
                },
                "optimization": {
                    "activation_checkpoint_policy": config.activation_checkpoint_policy,
                    "accumulation_steps": arguments.accumulation_steps,
                    "adamw_betas": [0.9, 0.98],
                    "gradient_clip_norm": 1.0,
                    "learning_rate": arguments.learning_rate,
                    "microbatch_pairs_per_rank": arguments.microbatch_pairs,
                    "updates": arguments.updates,
                    "weight_decay": arguments.weight_decay,
                },
                "precision": "bfloat16_compute_fp32_optimizer",
                "ranks": ranks,
                "stability": {
                    "maximum_gradient_norm_before_clipping": max(gradients),
                    "nonfinite_gradient_updates": 0,
                    "p95_to_median_update_time": _nearest_rank_p95(elapsed_maxima)
                    / statistics.median(elapsed_maxima),
                },
                "status": "PASS",
                "test_access": {
                    "asep_test_accessed": False,
                    "sabdab2_heldout_accessed": False,
                },
                "throughput": {
                    "consumed_residue_tokens": sum(global_residues),
                    "residue_tokens_per_second_mean": statistics.mean(throughputs),
                    "residue_tokens_per_second_median": statistics.median(throughputs),
                    "update_seconds_max_by_update": elapsed_maxima,
                },
                "torch": {
                    "cuda_version": torch.version.cuda,
                    "version": torch.__version__,
                },
                "world_size": world_size,
            }
            _publish_create_only(arguments.output.resolve(), payload)
            print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    finally:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
