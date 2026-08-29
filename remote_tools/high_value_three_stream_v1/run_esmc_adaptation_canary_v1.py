#!/usr/bin/env python3
"""Four-GPU correctness canary for whole-chain versus CDR-focused ESM-C adaptation."""

from __future__ import annotations

import argparse
import contextlib
import csv
import gzip
import hashlib
import heapq
import json
import math
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from minimal_esmc_mlm_training_v1 import load_base_model, sha256_file, state_probe_sha256
from minimal_esmc_runtime_v1 import PAD_TOKEN_ID, SEQUENCE_VOCAB, TOKEN_TO_ID, tokenize_sequence


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def atomic_write(path: Path, value: object) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(canonical_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def select_span_rows(path: Path, count: int) -> list[dict[str, str]]:
    selected: list[tuple[int, str, dict[str, str]]] = []
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if row["split"] != "train":
                continue
            score = int(hashlib.sha256(("esmc-canary-v1|" + row["row_hash"]).encode()).hexdigest(), 16)
            item = (-score, row["row_hash"], row)
            if len(selected) < count:
                heapq.heappush(selected, item)
            elif item > selected[0]:
                heapq.heapreplace(selected, item)
    if len(selected) != count:
        raise RuntimeError("insufficient train rows for canary")
    return [item[2] for item in sorted(selected, key=lambda value: -value[0])]


def load_examples(source: Path, span_path: Path, count: int) -> list[dict[str, object]]:
    spans = select_span_rows(span_path, count)
    by_index = {int(row["source_row_index"]): row for row in spans}
    examples: dict[int, dict[str, object]] = {}
    with gzip.open(source, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for index, row in enumerate(reader):
            span = by_index.get(index)
            if span is None:
                continue
            if row["row_hash"] != span["row_hash"]:
                raise RuntimeError("span/source row hash differs")
            examples[index] = {
                "row_hash": row["row_hash"],
                "heavy": row["heavy_sequence"],
                "light": row["light_sequence"],
                "spans": tuple((int(span[f"{name}_start"]), int(span[f"{name}_end"])) for name in ("h1", "h2", "h3", "l1", "l2", "l3")),
            }
    if len(examples) != count:
        raise RuntimeError("selected canary rows were not all recovered")
    return [examples[int(row["source_row_index"])] for row in spans]


def make_batch(
    examples: list[dict[str, object]], indices: list[int], arm: str, generator: torch.Generator, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    sequences = []
    eligible_rows = []
    for index in indices:
        example = examples[index % len(examples)]
        heavy = str(example["heavy"])
        light = str(example["light"])
        tokens = tokenize_sequence(f"{heavy}-{light}")
        eligible = torch.zeros_like(tokens, dtype=torch.bool)
        if arm == "whole_chain_control":
            eligible[1 : 1 + len(heavy)] = True
            eligible[2 + len(heavy) : -1] = True
        else:
            spans = example["spans"]
            for start, end in spans[:3]:
                eligible[1 + start : 1 + end] = True
            light_offset = 2 + len(heavy)
            for start, end in spans[3:]:
                eligible[light_offset + start : light_offset + end] = True
        sequences.append(tokens)
        eligible_rows.append(eligible)
    length = max(value.numel() for value in sequences)
    tokens = torch.full((len(sequences), length), PAD_TOKEN_ID, dtype=torch.long)
    eligible = torch.zeros((len(sequences), length), dtype=torch.bool)
    for row, (sequence, mask) in enumerate(zip(sequences, eligible_rows)):
        tokens[row, : sequence.numel()] = sequence
        eligible[row, : mask.numel()] = mask
    tokens = tokens.to(device)
    eligible = eligible.to(device)
    probability = 0.15 if arm == "whole_chain_control" else 0.50
    masked = (torch.rand(tokens.shape, generator=generator, device=device) < probability) & eligible
    if not masked.any():
        first = eligible.nonzero(as_tuple=False)[0]
        masked[first[0], first[1]] = True
    labels = tokens.clone()
    labels[~masked] = -100
    corruption = torch.rand(tokens.shape, generator=generator, device=device)
    replace = masked & (corruption < 0.8)
    randomize = masked & (corruption >= 0.8) & (corruption < 0.9)
    tokens[replace] = TOKEN_TO_ID["<mask>"]
    random_tokens = torch.randint(0, len(SEQUENCE_VOCAB), tokens.shape, generator=generator, device=device)
    tokens[randomize] = random_tokens[randomize]
    return tokens, labels, int(masked.sum()), int(eligible.sum())


def reduce_sum(value: float, device: torch.device) -> float:
    tensor = torch.tensor(value, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--spans", type=Path, required=True)
    parser.add_argument("--arm", choices=("whole_chain_control", "cdr_focused"), required=True)
    parser.add_argument("--safetensors-site", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 4:
        raise RuntimeError("canary requires exactly four ranks")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")
    authority = json.loads(args.authority.read_text())
    if sha256_file(args.checkpoint) != authority["model"]["weight_sha256"]:
        raise RuntimeError("base checkpoint SHA differs")
    if sha256_file(args.source) != authority["data"]["source_sha256"]:
        raise RuntimeError("canary source SHA differs")
    if sha256_file(args.spans) != authority["data"]["span_sha256"]:
        raise RuntimeError("canary span SHA differs")
    examples = load_examples(args.source, args.spans, authority["canary"]["sample_rows"])
    torch.manual_seed(authority["canary"]["seed"])
    model = load_base_model(args.checkpoint, args.safetensors_site)
    initial_probe = state_probe_sha256(model)
    model = model.to(device=device, dtype=torch.bfloat16)
    torch.cuda.reset_peak_memory_stats(device)
    wrapped = DistributedDataParallel(
        model, device_ids=[local_rank], output_device=local_rank,
        gradient_as_bucket_view=True, static_graph=True,
    )
    optimizer = torch.optim.AdamW(
        wrapped.parameters(), lr=authority["optimization"]["learning_rate"],
        weight_decay=authority["optimization"]["weight_decay"],
    )
    updates = authority["canary"]["updates"]
    accumulation = authority["canary"]["gradient_accumulation_steps"]
    microbatch = authority["canary"]["per_rank_microbatch"]
    generator = torch.Generator(device=device).manual_seed(
        authority["canary"]["seed"] * 1000 + rank
    )
    losses: list[float] = []
    grad_norms: list[float] = []
    masked_total = 0.0
    eligible_total = 0.0
    nonfinite = 0
    start_time = time.monotonic()
    for update in range(updates):
        optimizer.zero_grad(set_to_none=True)
        local_loss_sum = 0.0
        for micro in range(accumulation):
            base = (update * accumulation * world_size + micro * world_size + rank) * microbatch
            indices = list(range(base, base + microbatch))
            tokens, labels, masked, eligible = make_batch(examples, indices, args.arm, generator, device)
            sync_context = contextlib.nullcontext() if micro + 1 == accumulation else wrapped.no_sync()
            with sync_context:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = wrapped(tokens)
                    loss = F.cross_entropy(logits.view(-1, 64), labels.view(-1), ignore_index=-100)
                    scaled = loss / accumulation
                if not torch.isfinite(loss):
                    nonfinite += 1
                scaled.backward()
            local_loss_sum += float(loss.detach())
            masked_total += reduce_sum(masked, device)
            eligible_total += reduce_sum(eligible, device)
        grad_norm = torch.nn.utils.clip_grad_norm_(wrapped.parameters(), max_norm=1.0)
        if not torch.isfinite(grad_norm):
            nonfinite += 1
        warmup = max(1, math.ceil(updates * authority["optimization"]["warmup_ratio"]))
        if update < warmup:
            factor = (update + 1) / warmup
        else:
            factor = max(0.0, (updates - update) / max(1, updates - warmup))
        for group in optimizer.param_groups:
            group["lr"] = authority["optimization"]["learning_rate"] * factor
        optimizer.step()
        losses.append(reduce_sum(local_loss_sum / accumulation, device) / world_size)
        grad_norms.append(reduce_sum(float(grad_norm), device) / world_size)
    torch.cuda.synchronize(device)
    elapsed = time.monotonic() - start_time
    peak = torch.cuda.max_memory_allocated(device)
    peak_all = [torch.zeros((), dtype=torch.int64, device=device) for _ in range(world_size)]
    dist.all_gather(peak_all, torch.tensor(peak, dtype=torch.int64, device=device))
    nonfinite_global = int(reduce_sum(nonfinite, device))
    total_memory = torch.cuda.get_device_properties(device).total_memory
    dist.barrier()
    if rank == 0:
        final_probe = state_probe_sha256(wrapped.module)
        result = {
            "format_version": "paired_esmc_adaptation_canary_result_v1",
            "status": "PASS_FINITE_FOUR_GPU_ESMC_ADAPTATION_CANARY" if nonfinite_global == 0 and final_probe != initial_probe else "FAIL_CLOSED_ESMC_ADAPTATION_CANARY",
            "arm": args.arm,
            "authority_sha256": sha256_file(args.authority),
            "base_checkpoint_sha256": authority["model"]["weight_sha256"],
            "span_sha256": authority["data"]["span_sha256"],
            "world_size": world_size,
            "updates": updates,
            "global_batch_size": world_size * microbatch * accumulation,
            "losses": losses,
            "grad_norms": grad_norms,
            "masked_tokens": masked_total,
            "eligible_tokens": eligible_total,
            "masked_fraction_of_eligible": masked_total / eligible_total,
            "nonfinite": nonfinite_global,
            "initial_state_probe_sha256": initial_probe,
            "final_state_probe_sha256": final_probe,
            "elapsed_seconds": elapsed,
            "maximum_peak_allocated_bytes": max(int(value.item()) for value in peak_all),
            "peak_allocated_fraction": max(int(value.item()) for value in peak_all) / total_memory,
            "audit": {
                "strict_base_load": True,
                "full_model_trainable": True,
                "gradient_checkpointing": True,
                "bert_80_10_10_corruption": True,
                "random_tokens_from_tokenizer_vocabulary": True,
                "checkpoint_saved": False,
                "abcdr_validation_or_test_rows_used": False,
                "task_labels_read": False,
                "sabdab2_test_access": False,
            },
        }
        atomic_write(args.output, result)
        print(json.dumps({**result, "result_sha256": sha256_file(args.output)}, sort_keys=True))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
