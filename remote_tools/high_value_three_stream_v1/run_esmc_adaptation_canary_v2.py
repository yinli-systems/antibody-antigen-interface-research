#!/usr/bin/env python3
"""Recovery canary with a synchronized first DDP static-graph update."""

from __future__ import annotations

import argparse
import contextlib
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
from run_esmc_adaptation_canary_v1 import atomic_write, load_examples, make_batch, reduce_sum


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
    generator = torch.Generator(device=device).manual_seed(authority["canary"]["seed"] * 1000 + rank)
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
            # PyTorch #143580: static_graph cannot begin its first backward inside no_sync.
            # Synchronizing all microbatches of update zero is gradient-equivalent and
            # initializes the reducer before normal accumulation begins.
            synchronize = update == 0 or micro + 1 == accumulation
            sync_context = contextlib.nullcontext() if synchronize else wrapped.no_sync()
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
        factor = (update + 1) / warmup if update < warmup else max(0.0, (updates - update) / max(1, updates - warmup))
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
            "format_version": "paired_esmc_adaptation_canary_result_v2",
            "status": "PASS_FINITE_FOUR_GPU_ESMC_ADAPTATION_CANARY" if nonfinite_global == 0 and final_probe != initial_probe else "FAIL_CLOSED_ESMC_ADAPTATION_CANARY",
            "arm": args.arm, "authority_sha256": sha256_file(args.authority),
            "base_checkpoint_sha256": authority["model"]["weight_sha256"],
            "span_sha256": authority["data"]["span_sha256"], "world_size": world_size,
            "updates": updates, "global_batch_size": world_size * microbatch * accumulation,
            "losses": losses, "grad_norms": grad_norms,
            "masked_tokens": masked_total, "eligible_tokens": eligible_total,
            "masked_fraction_of_eligible": masked_total / eligible_total,
            "nonfinite": nonfinite_global, "initial_state_probe_sha256": initial_probe,
            "final_state_probe_sha256": final_probe, "elapsed_seconds": elapsed,
            "maximum_peak_allocated_bytes": max(int(value.item()) for value in peak_all),
            "peak_allocated_fraction": max(int(value.item()) for value in peak_all) / total_memory,
            "audit": {
                "strict_base_load": True, "full_model_trainable": True,
                "gradient_checkpointing": True, "bert_80_10_10_corruption": True,
                "random_tokens_from_tokenizer_vocabulary": True, "checkpoint_saved": False,
                "ddp_static_graph_first_update_synchronized": True,
                "pytorch_issue_143580_workaround_only": True,
                "abcdr_validation_or_test_rows_used": False, "task_labels_read": False,
                "sabdab2_test_access": False,
            },
        }
        atomic_write(args.output, result)
        print(json.dumps({**result, "result_sha256": sha256_file(args.output)}, sort_keys=True))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
