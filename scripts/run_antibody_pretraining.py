#!/usr/bin/env python3
"""Run an audited antibody MLM engineering pilot on a dense OAS shard."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from antibody_pretraining.dense_shard import load_dense_shard
from antibody_pretraining.masking import apply_uniform_mlm
from antibody_pretraining.model import AntibodyMaskedLanguageModel, config_from_preset, load_model_family
from antibody_pretraining.tokenizer import PAD_ID
from antibody_pretraining.training_contract import build_update_plan, learning_rate_at_token, load_training_protocol


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _publish(path: Path, payload: object) -> None:
    data = (json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("ascii")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)


def _save_dcp(model: torch.nn.Module, optimizer: torch.optim.Optimizer, path: Path) -> None:
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict

    model_state, optimizer_state = get_state_dict(model, optimizer)
    dcp.save({"model": model_state, "optimizer": optimizer_state}, checkpoint_id=str(path))


def _load_dcp(model: torch.nn.Module, optimizer: torch.optim.Optimizer, path: Path) -> None:
    """Restore model and optimizer state through PyTorch DCP."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

    model_state, optimizer_state = get_state_dict(model, optimizer)
    state = {"model": model_state, "optimizer": optimizer_state}
    dcp.load(state, checkpoint_id=str(path))
    set_state_dict(
        model,
        optimizer,
        model_state_dict=state["model"],
        optim_state_dict=state["optimizer"],
    )


def _tree_digest(value: object) -> str:
    """Hash a nested tensor state without relying on checkpoint byte layout."""
    digest = hashlib.sha256()

    def visit(item: object) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            header = json.dumps(
                {"dtype": str(tensor.dtype), "shape": list(tensor.shape)},
                sort_keys=True,
                separators=(",", ":"),
            ).encode("ascii")
            digest.update(b"tensor\0" + header + b"\0")
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict\0")
            for key in sorted(item, key=lambda current: str(current)):
                visit(str(key))
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(b"sequence\0")
            for child in item:
                visit(child)
        else:
            digest.update(
                json.dumps(item, sort_keys=True, separators=(",", ":")).encode(
                    "ascii"
                )
                + b"\0"
            )

    visit(value)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--model-family", type=Path, required=True)
    parser.add_argument("--training-protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preset", default="abf_35m")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--micro-batch", type=int, default=8)
    parser.add_argument("--target-tokens", type=int, default=4096)
    parser.add_argument("--batch-quantum", type=int, default=32)
    parser.add_argument("--resume-root", type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.steps <= 0 or args.micro_batch <= 0:
        raise ValueError("Output must be create-only and pilot sizes must be positive")
    args.output.mkdir(parents=True)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("This gate requires exactly one visible CUDA GPU")
    torch.cuda.set_device(0)
    torch.manual_seed(20260728)
    torch.cuda.manual_seed_all(20260728)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda", 0)
    shard = load_dense_shard(args.shard, expected_manifest_sha256=args.manifest_sha256)
    family = load_model_family(args.model_family)
    protocol = load_training_protocol(args.training_protocol)
    model = AntibodyMaskedLanguageModel(config_from_preset(family, args.preset)).to(device)
    decay, no_decay = [], []
    for parameter in model.parameters():
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": protocol["optimizer"]["weight_decay"]}, {"params": no_decay, "weight_decay": 0.0}],
        lr=protocol["presets"][args.preset]["learning_rate"],
        betas=tuple(protocol["optimizer"]["betas"]),
        eps=protocol["optimizer"]["epsilon"],
    )
    residue_lengths = np.asarray(shard.lengths, dtype=np.int64) - 2
    cursor = 0
    consumed = 0
    start_update = 0
    model_family_sha256 = _sha256(args.model_family)
    training_protocol_sha256 = _sha256(args.training_protocol)
    if args.resume_root is not None:
        state = json.loads(
            (args.resume_root / "trainer_state.json").read_text(encoding="ascii")
        )
        expected = {
            "model_family_sha256": model_family_sha256,
            "preset": args.preset,
            "shard_manifest_sha256": args.manifest_sha256,
            "training_protocol_sha256": training_protocol_sha256,
        }
        if any(state.get(key) != value for key, value in expected.items()):
            raise RuntimeError("Resume provenance differs from this launch")
        cursor = int(state["sample_cursor"])
        consumed = int(state["consumed_residue_tokens"])
        start_update = int(state["next_update"])
        _load_dcp(model, optimizer, args.resume_root / "checkpoint")
    trace = []
    for update in range(start_update, start_update + args.steps):
        plan = build_update_plan(residue_lengths, sample_cursor=cursor, target_residue_tokens=args.target_tokens, batch_quantum_sequences=args.batch_quantum, seed=protocol["random_seed"])
        if len(plan.indices) % args.micro_batch:
            raise RuntimeError("Pilot batch quantum must divide the micro batch")
        prepared = []
        target_count = 0
        for start in range(0, len(plan.indices), args.micro_batch):
            indices = plan.indices[start : start + args.micro_batch]
            tokens = np.asarray(shard.tokens[indices], dtype=np.int64)
            masked = apply_uniform_mlm(tokens, np.asarray(shard.sample_keys[indices]), update=update, seed=protocol["random_seed"], mask_fraction=protocol["masking"]["fraction"])
            prepared.append((indices, masked))
            target_count += masked.selected_target_count
        if target_count <= 0:
            raise RuntimeError("Update has no selected MLM targets")
        lr = learning_rate_at_token(consumed, peak_learning_rate=protocol["presets"][args.preset]["learning_rate"], warmup_residue_tokens=protocol["schedule"]["warmup_residue_tokens"], total_residue_tokens=protocol["schedule"]["total_residue_tokens"], minimum_learning_rate_ratio=protocol["schedule"]["minimum_learning_rate_ratio"])
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        started = time.perf_counter()
        loss_sum_value = 0.0
        for indices, masked in prepared:
            width = int(np.asarray(shard.lengths[indices]).max())
            input_ids = torch.from_numpy(masked.input_ids[:, :width]).to(device)
            labels = torch.from_numpy(masked.labels[:, :width]).to(device)
            attention = input_ids.ne(PAD_ID)
            chain = torch.from_numpy(np.asarray(shard.chain_types[indices], dtype=np.int64)).to(device)
            type_ids = chain[:, None].expand_as(input_ids).masked_fill(~attention, 0)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(input_ids, type_ids, attention, labels)
                loss = output.loss_sum / target_count
            loss.backward()
            loss_sum_value += float(output.loss_sum.detach())
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), protocol["gradient_clip_norm"]))
        optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        consumed += plan.residue_token_count
        cursor = plan.next_sample_cursor
        trace.append({"consumed_residue_tokens": consumed, "gradient_norm": gradient_norm, "learning_rate": lr, "loss": loss_sum_value / target_count, "residue_tokens": plan.residue_token_count, "seconds": elapsed, "selected_targets": target_count, "update": update + 1})
    _save_dcp(model, optimizer, args.output / "checkpoint")
    trainer_state = {"consumed_residue_tokens": consumed, "model_family_sha256": model_family_sha256, "next_update": start_update + args.steps, "preset": args.preset, "sample_cursor": cursor, "shard_manifest_sha256": args.manifest_sha256, "training_protocol_sha256": training_protocol_sha256}
    _publish(args.output / "trainer_state.json", trainer_state)
    receipt = {"backend": "single_gpu", "checkpoint_format": "torch_distributed_checkpoint", "claim_boundary": "engineering pilot only; no validation or downstream metric", "device": torch.cuda.get_device_name(0), "final_model_sha256": _tree_digest(model.state_dict()), "final_optimizer_sha256": _tree_digest(optimizer.state_dict()), "format_version": "antibody_pretraining_pilot_v2", "model_family_sha256": model_family_sha256, "preset": args.preset, "resume_used": args.resume_root is not None, "shard_manifest_sha256": args.manifest_sha256, "start_update": start_update, "test_access": {"downstream_labels_used": False, "sabdab2_heldout_accessed": False}, "trace": trace, "training_protocol_sha256": training_protocol_sha256}
    _publish(args.output / "receipt.json", receipt)


if __name__ == "__main__":
    main()
