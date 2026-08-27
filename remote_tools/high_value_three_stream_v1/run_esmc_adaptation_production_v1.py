#!/usr/bin/env python3
"""Frozen four-epoch ESM-C whole-chain versus CDR-focused adaptation."""

from __future__ import annotations

import argparse
import contextlib
import csv
import gzip
import json
import math
import os
from pathlib import Path
import sys
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from minimal_esmc_mlm_training_v1 import load_base_model, sha256_file, state_probe_sha256
from run_esmc_adaptation_canary_v1 import atomic_write, make_batch


def reduce_vector(values: list[float], device: torch.device) -> list[float]:
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return [float(value) for value in tensor.cpu().tolist()]


def load_examples(source: Path, spans_path: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    by_index: dict[int, tuple[str, str, tuple[tuple[int, int], ...]]] = {}
    split_counts = {"train": 0, "validation": 0}
    with gzip.open(spans_path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            split = row["split"]
            if split not in split_counts:
                raise RuntimeError(f"unexpected span split: {split}")
            index = int(row["source_row_index"])
            if index in by_index:
                raise RuntimeError("duplicate source row index in spans")
            spans = tuple(
                (int(row[f"{name}_start"]), int(row[f"{name}_end"]))
                for name in ("h1", "h2", "h3", "l1", "l2", "l3")
            )
            by_index[index] = (row["row_hash"], split, spans)
            split_counts[split] += 1
    if split_counts != {"train": 757_640, "validation": 15_415}:
        raise RuntimeError(f"span split counts differ: {split_counts}")

    examples: dict[str, list[dict[str, object]]] = {"train": [], "validation": []}
    allowed = set("ACDEFGHIKLMNPQRSTVWY")
    with gzip.open(source, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for index, row in enumerate(reader):
            authority = by_index.get(index)
            if authority is None:
                continue
            row_hash, split, spans = authority
            heavy = row["heavy_sequence"]
            light = row["light_sequence"]
            if row["row_hash"] != row_hash:
                raise RuntimeError("span/source row hash differs")
            if (set(heavy) | set(light)) - allowed:
                raise RuntimeError("noncanonical residue found after frozen quality filtering")
            if any(not (0 <= start < end <= len(heavy if i < 3 else light)) for i, (start, end) in enumerate(spans)):
                raise RuntimeError("CDR span falls outside chain")
            examples[split].append({"row_hash": row_hash, "heavy": heavy, "light": light, "spans": spans})
    for split in examples:
        examples[split].sort(key=lambda value: str(value["row_hash"]))
        if len(examples[split]) != split_counts[split]:
            raise RuntimeError(f"not all {split} examples were recovered")
    return examples["train"], examples["validation"]


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    examples: list[dict[str, object]],
    mask_arm: str,
    batch_size: int,
    rank: int,
    world_size: int,
    seed: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed + rank)
    local_loss_sum = 0.0
    local_correct = 0.0
    local_masked = 0.0
    local_eligible = 0.0
    local_examples = 0.0
    indices = list(range(rank, len(examples), world_size))
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        tokens, labels, masked, eligible = make_batch(examples, batch_indices, mask_arm, generator, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(tokens)
        active = labels != -100
        local_loss_sum += float(F.cross_entropy(logits[active].float(), labels[active], reduction="sum"))
        local_correct += float((logits.argmax(dim=-1)[active] == labels[active]).sum())
        local_masked += masked
        local_eligible += eligible
        local_examples += len(batch_indices)
    loss_sum, correct, masked, eligible, count = reduce_vector(
        [local_loss_sum, local_correct, local_masked, local_eligible, local_examples], device
    )
    if int(count) != len(examples) or masked <= 0 or eligible <= 0:
        raise RuntimeError("validation accounting differs")
    return {
        "loss": loss_sum / masked,
        "accuracy": correct / masked,
        "masked_tokens": masked,
        "eligible_tokens": eligible,
        "masked_fraction_of_eligible": masked / eligible,
        "examples": count,
    }


def save_checkpoint(model: torch.nn.Module, path: Path, safetensors_site: Path) -> dict[str, object]:
    sys.path.append(str(safetensors_site))
    from safetensors.torch import save_file

    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    state = {name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()}
    save_file(state, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--spans", type=Path, required=True)
    parser.add_argument("--arm", choices=("whole_chain_control", "cdr_focused"), required=True)
    parser.add_argument("--safetensors-site", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    authority = json.loads(args.authority.read_text())
    protocol_sha = sha256_file(args.authority)
    if world_size != authority["training"]["world_size"] or world_size != 4:
        raise RuntimeError("production world size differs")
    if sha256_file(args.checkpoint) != authority["model"]["weight_sha256"]:
        raise RuntimeError("base checkpoint SHA differs")
    if sha256_file(args.source) != authority["data"]["source_sha256"]:
        raise RuntimeError("source SHA differs")
    if sha256_file(args.spans) != authority["data"]["span_sha256"]:
        raise RuntimeError("span SHA differs")
    train_examples, validation_examples = load_examples(args.source, args.spans)
    if len(train_examples) != authority["data"]["train_rows"] or len(validation_examples) != authority["data"]["validation_rows"]:
        raise RuntimeError("loaded split sizes differ")

    seed = authority["training"]["seed"]
    torch.manual_seed(seed)
    model = load_base_model(args.checkpoint, args.safetensors_site)
    initial_probe = state_probe_sha256(model)
    model = model.to(device=device, dtype=torch.bfloat16)
    wrapped = DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        gradient_as_bucket_view=True,
        static_graph=True,
    )
    optimizer = torch.optim.AdamW(
        wrapped.parameters(),
        lr=authority["optimization"]["learning_rate"],
        weight_decay=authority["optimization"]["weight_decay"],
    )
    epochs = authority["training"]["epochs"]
    microbatch = authority["training"]["per_rank_microbatch"]
    accumulation = authority["training"]["gradient_accumulation_steps"]
    updates_per_epoch = authority["training"]["updates_per_epoch"]
    total_updates = updates_per_epoch * epochs
    warmup = math.ceil(total_updates * authority["optimization"]["warmup_ratio"])
    train_generator = torch.Generator(device=device).manual_seed(seed * 1000 + rank)
    global_update = 0
    epoch_receipts: list[dict[str, object]] = []
    total_start = time.monotonic()
    for epoch in range(1, epochs + 1):
        wrapped.train()
        permutation_generator = torch.Generator().manual_seed(seed + epoch)
        permutation = torch.randperm(len(train_examples), generator=permutation_generator).tolist()
        local_indices = permutation[rank::world_size]
        usable = updates_per_epoch * accumulation * microbatch
        local_indices = local_indices[:usable]
        if len(local_indices) != usable:
            raise RuntimeError("local epoch sample accounting differs")
        local_loss_sum = 0.0
        local_masked = 0.0
        local_eligible = 0.0
        local_nonfinite = 0.0
        epoch_start = time.monotonic()
        for update_in_epoch in range(updates_per_epoch):
            optimizer.zero_grad(set_to_none=True)
            for micro in range(accumulation):
                start = (update_in_epoch * accumulation + micro) * microbatch
                indices = local_indices[start : start + microbatch]
                tokens, labels, masked, eligible = make_batch(
                    train_examples, indices, args.arm, train_generator, device
                )
                synchronize = global_update == 0 or micro + 1 == accumulation
                sync_context = contextlib.nullcontext() if synchronize else wrapped.no_sync()
                with sync_context:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        logits = wrapped(tokens)
                        loss = F.cross_entropy(logits.view(-1, 64), labels.view(-1), ignore_index=-100)
                        scaled = loss / accumulation
                    if not torch.isfinite(loss):
                        local_nonfinite += 1
                    scaled.backward()
                local_loss_sum += float(loss.detach()) * masked
                local_masked += masked
                local_eligible += eligible
            grad_norm = torch.nn.utils.clip_grad_norm_(wrapped.parameters(), max_norm=authority["optimization"]["gradient_clip_norm"])
            if not torch.isfinite(grad_norm):
                local_nonfinite += 1
            if global_update < warmup:
                factor = (global_update + 1) / warmup
            else:
                factor = max(0.0, (total_updates - global_update) / max(1, total_updates - warmup))
            for group in optimizer.param_groups:
                group["lr"] = authority["optimization"]["learning_rate"] * factor
            optimizer.step()
            global_update += 1
            if rank == 0 and global_update % authority["training"]["log_every_updates"] == 0:
                print(json.dumps({"arm": args.arm, "epoch": epoch, "global_update": global_update, "lr": optimizer.param_groups[0]["lr"]}, sort_keys=True), flush=True)
        torch.cuda.synchronize(device)
        loss_sum, masked, eligible, nonfinite = reduce_vector(
            [local_loss_sum, local_masked, local_eligible, local_nonfinite], device
        )
        validation = {
            "whole_chain_15pct": evaluate(
                wrapped.module, validation_examples, "whole_chain_control", microbatch,
                rank, world_size, authority["validation"]["whole_chain_seed"], device,
            ),
            "cdr_50pct": evaluate(
                wrapped.module, validation_examples, "cdr_focused", microbatch,
                rank, world_size, authority["validation"]["cdr_seed"], device,
            ),
        }
        wrapped.train()
        dist.barrier()
        if rank == 0:
            checkpoint = save_checkpoint(
                wrapped.module,
                args.output_root / args.arm / f"checkpoint-epoch-{epoch}.safetensors",
                args.safetensors_site,
            )
            epoch_result = {
                "format_version": "paired_esmc_adaptation_epoch_result_v1",
                "status": "PASS_FINITE_EPOCH_CHECKPOINT_VERIFIED" if nonfinite == 0 else "FAIL_CLOSED_NONFINITE_EPOCH",
                "arm": args.arm,
                "epoch": epoch,
                "authority_sha256": protocol_sha,
                "global_update": global_update,
                "train": {
                    "mean_masked_token_cross_entropy": loss_sum / masked,
                    "masked_tokens": masked,
                    "eligible_tokens": eligible,
                    "masked_fraction_of_eligible": masked / eligible,
                    "nonfinite": int(nonfinite),
                    "elapsed_seconds": time.monotonic() - epoch_start,
                },
                "validation": validation,
                "checkpoint": checkpoint,
                "selected_for_downstream": epoch == authority["training"]["fixed_checkpoint_epoch"],
                "audit": {
                    "direct_from_frozen_base": True,
                    "checkpoint_selected_without_downstream_metrics": True,
                    "abcdr_validation_or_test_rows_used": False,
                    "task_labels_read": False,
                    "sabdab2_test_access": False,
                },
            }
            epoch_path = args.output_root / args.arm / f"epoch-{epoch}-result.json"
            atomic_write(epoch_path, epoch_result)
            epoch_receipts.append({
                "epoch": epoch,
                "result_path": str(epoch_path),
                "result_sha256": sha256_file(epoch_path),
                "checkpoint": checkpoint,
            })
            print(json.dumps({"epoch_result": epoch_receipts[-1]}, sort_keys=True), flush=True)
        dist.barrier()
        if nonfinite != 0:
            raise RuntimeError("nonfinite training state")
    if rank == 0:
        final_probe = state_probe_sha256(wrapped.module)
        result = {
            "format_version": "paired_esmc_adaptation_production_result_v1",
            "status": "PASS_FIXED_EPOCH4_ADAPTATION_CHECKPOINT_VERIFIED" if final_probe != initial_probe else "FAIL_CLOSED_UNCHANGED_MODEL",
            "arm": args.arm,
            "authority_sha256": protocol_sha,
            "base_checkpoint_sha256": authority["model"]["weight_sha256"],
            "initial_state_probe_sha256": initial_probe,
            "final_state_probe_sha256": final_probe,
            "epochs": epochs,
            "global_updates": global_update,
            "epoch_receipts": epoch_receipts,
            "fixed_checkpoint_epoch": authority["training"]["fixed_checkpoint_epoch"],
            "fixed_checkpoint": epoch_receipts[authority["training"]["fixed_checkpoint_epoch"] - 1]["checkpoint"],
            "elapsed_seconds": time.monotonic() - total_start,
            "audit": {
                "base_and_span_sha_verified": True,
                "same_seed_shuffle_batch_optimizer_and_schedule_across_arms": True,
                "fixed_validation_masks_across_epochs_and_arms": True,
                "no_downstream_checkpoint_selection": True,
                "abcdr_validation_or_test_rows_used": False,
                "task_labels_read": False,
                "sabdab2_test_access": False,
            },
        }
        atomic_write(args.output_root / args.arm / "result.json", result)
        print(json.dumps({**result, "result_sha256": sha256_file(args.output_root / args.arm / "result.json")}, sort_keys=True), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
