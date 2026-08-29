#!/usr/bin/env python3
"""Run a frozen, checkpoint-free paired/CDR-aware 650M training canary."""

from __future__ import annotations

import argparse
from dataclasses import replace
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import torch
import torch.distributed.checkpoint as dcp

from paired_pretraining import (
    REGION_CDR1,
    REGION_CDR2,
    REGION_CDR3,
    REGION_FRAMEWORK,
    apply_cdr3_balanced_mlm,
    focal_mlm_loss_sum,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class PairedBinaryShard:
    """Read one authenticated variable-length paired binary shard."""

    def __init__(self, root: Path, *, verify_hashes: bool) -> None:
        self.root = root
        self.manifest_path = root / "MANIFEST.json"
        self.manifest = json.loads(self.manifest_path.read_text(encoding="ascii"))
        if (
            self.manifest.get("format_version")
            != "oas_paired_development_clean_binary_v1"
            or self.manifest.get("status")
            != "PASS_PAIRED_DEVELOPMENT_CLEAN_MATERIALIZATION"
            or self.manifest.get("test_access") is not False
        ):
            raise RuntimeError("Paired shard authority differs")
        artifacts = self.manifest["artifacts"]
        if verify_hashes:
            for specification in artifacts.values():
                path = root / specification["name"]
                if path.stat().st_size != specification["bytes"] or _sha256(path) != specification["sha256"]:
                    raise RuntimeError(f"Paired shard artifact differs: {path.name}")
        self.input_ids = np.memmap(root / artifacts["input_ids"]["name"], dtype=np.uint8, mode="r")
        self.type_ids = np.memmap(root / artifacts["type_ids"]["name"], dtype=np.uint8, mode="r")
        self.region_ids = np.memmap(root / artifacts["region_ids"]["name"], dtype=np.uint8, mode="r")
        self.offsets = np.memmap(root / artifacts["offsets"]["name"], dtype="<u8", mode="r")
        self.sample_keys = np.memmap(root / artifacts["sample_keys"]["name"], dtype="<u8", mode="r")
        self.split = np.memmap(root / artifacts["split"]["name"], dtype=np.uint8, mode="r")
        kept_rows = int(self.manifest["counts"]["kept_rows"])
        if (
            len(self.offsets) != kept_rows + 1
            or len(self.sample_keys) != kept_rows
            or len(self.split) != kept_rows
            or int(self.offsets[-1]) != len(self.input_ids)
            or len(self.type_ids) != len(self.input_ids)
            or len(self.region_ids) != len(self.input_ids)
        ):
            raise RuntimeError("Paired shard binary dimensions differ")

    def batch(self, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        lengths = self.offsets[indices + 1] - self.offsets[indices]
        width = int(lengths.max())
        width = min(320, ((width + 7) // 8) * 8)
        tokens = np.zeros((len(indices), width), dtype=np.uint8)
        types = np.zeros_like(tokens)
        regions = np.zeros_like(tokens)
        for row, index in enumerate(indices):
            start = int(self.offsets[index])
            stop = int(self.offsets[index + 1])
            length = stop - start
            tokens[row, :length] = self.input_ids[start:stop]
            types[row, :length] = self.type_ids[start:stop]
            regions[row, :length] = self.region_ids[start:stop]
        return tokens, types, regions, np.asarray(self.sample_keys[indices], dtype=np.uint64)


def _splitmix64(value: np.ndarray) -> np.ndarray:
    mask = np.uint64(0xFFFFFFFFFFFFFFFF)
    value = (value + np.uint64(0x9E3779B97F4A7C15)) & mask
    value = ((value ^ (value >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & mask
    value = ((value ^ (value >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & mask
    return value ^ (value >> np.uint64(31))


def select_indices(shard: PairedBinaryShard, split: int, count: int, seed: int) -> np.ndarray:
    candidates = np.flatnonzero(np.asarray(shard.split) == split)
    if len(candidates) < count:
        raise RuntimeError("Frozen shard partition is too small")
    scores = _splitmix64(np.asarray(shard.sample_keys[candidates]) ^ np.uint64(seed))
    selected = np.argpartition(scores, count - 1)[:count]
    order = np.lexsort((candidates[selected], scores[selected]))
    return candidates[selected[order]].astype(np.int64, copy=False)


def _masked_batch(
    shard: PairedBinaryShard,
    indices: np.ndarray,
    *,
    update: int,
    seed: int,
    cdr3_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tokens, types, regions, keys = shard.batch(indices)
    masked = apply_cdr3_balanced_mlm(
        tokens,
        regions,
        keys,
        update=update,
        seed=seed,
        overall_fraction=0.15,
        cdr3_fraction=cdr3_fraction,
    )
    return masked.input_ids, types, regions, masked.labels, tokens != 0


def evaluate(
    model: torch.nn.Module,
    shard: PairedBinaryShard,
    indices: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
    seed: int,
) -> dict[str, object]:
    totals = {
        name: {"correct": 0, "count": 0, "loss": 0.0}
        for name in ("all", "framework", "cdr1", "cdr2", "cdr3", "heavy", "light")
    }
    model.eval()
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        input_ids, types, regions, labels, attention = _masked_batch(
            shard,
            batch_indices,
            update=10_000_000 + start // batch_size,
            seed=seed,
            cdr3_fraction=0.15,
        )
        input_tensor = torch.as_tensor(input_ids, dtype=torch.long, device=device)
        type_tensor = torch.as_tensor(types, dtype=torch.long, device=device)
        label_tensor = torch.as_tensor(labels, dtype=torch.long, device=device)
        attention_tensor = torch.as_tensor(attention, dtype=torch.bool, device=device)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(input_tensor, type_tensor, attention_mask=attention_tensor)
        selected = label_tensor.ne(-100)
        log_probability = torch.log_softmax(output.logits.float(), dim=-1)
        safe_labels = label_tensor.clamp_min(0)
        nll = -log_probability.gather(-1, safe_labels[..., None]).squeeze(-1)
        predictions = output.logits.argmax(dim=-1)
        region_tensor = torch.as_tensor(regions, dtype=torch.long, device=device)
        masks = {
            "all": selected,
            "framework": selected & region_tensor.eq(REGION_FRAMEWORK),
            "cdr1": selected & region_tensor.eq(REGION_CDR1),
            "cdr2": selected & region_tensor.eq(REGION_CDR2),
            "cdr3": selected & region_tensor.eq(REGION_CDR3),
            "heavy": selected & type_tensor.eq(5),
            "light": selected & type_tensor.eq(6),
        }
        for name, mask in masks.items():
            count = int(mask.sum())
            totals[name]["count"] += count
            totals[name]["correct"] += int((predictions.eq(label_tensor) & mask).sum())
            totals[name]["loss"] += float(nll[mask].sum())
    return {
        name: {
            "masked_accuracy": values["correct"] / values["count"],
            "masked_cross_entropy": values["loss"] / values["count"],
            "selected_targets": values["count"],
        }
        for name, values in totals.items()
    }


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--abf-source-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--checkpoint-receipt", type=Path, required=True)
    parser.add_argument("--model-family", type=Path, required=True)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite canary result")
    protocol_bytes = args.protocol.read_bytes()
    protocol = json.loads(protocol_bytes.decode("ascii"))
    expected_test_access = {
        "downstream_labels_used": False,
        "sabdab2_heldout_accessed": False,
        "temporal_holdout_accessed": False,
    }
    if (
        protocol.get("format_version") != "oas_paired_650m_training_canary_v1"
        or protocol.get("test_access") != expected_test_access
        or protocol.get("checkpoint_saved") is not False
    ):
        raise RuntimeError("Training canary protocol differs")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Canary requires exactly one CUDA device")
    device_name = torch.cuda.get_device_name(0)
    if "4090" not in device_name:
        raise RuntimeError("Canary requires RTX 4090")
    receipt = json.loads(args.checkpoint_receipt.read_text(encoding="ascii"))
    if (
        _sha256(args.checkpoint_receipt) != protocol["checkpoint_receipt_sha256"]
        or receipt.get("checkpoint_model_sha256") != protocol["checkpoint_model_sha256"]
        or _sha256(args.model_family) != protocol["model_family_sha256"]
    ):
        raise RuntimeError("Checkpoint provenance differs")

    shard = PairedBinaryShard(args.shard_root, verify_hashes=True)
    shard_manifest_sha256 = _sha256(shard.manifest_path)
    if shard.manifest.get("protocol_sha256") != protocol["materialization_protocol_sha256"]:
        raise RuntimeError("Paired materialization protocol differs")

    sys.path.insert(0, str(args.abf_source_root / "src"))
    sys.path.insert(0, str(args.abf_source_root / "scripts"))
    from antibody_pretraining.model import (  # pylint: disable=import-outside-toplevel
        AntibodyMaskedLanguageModel,
        config_from_preset,
        load_model_family,
    )
    from run_antibody_pretraining import _tree_digest  # pylint: disable=import-outside-toplevel

    family = load_model_family(args.model_family)
    config = replace(config_from_preset(family, "abf_650m"), maximum_sequence_tokens=320)
    model = AntibodyMaskedLanguageModel(config)
    state = {"model": model.state_dict()}
    dcp.load(state, checkpoint_id=str(args.checkpoint_root))
    model.load_state_dict(state["model"], strict=True)
    del state
    observed_model_sha256 = _tree_digest(model.state_dict())
    if observed_model_sha256 != protocol["checkpoint_model_sha256"]:
        raise RuntimeError("Loaded model hash differs")

    device = torch.device("cuda", 0)
    model.to(device)
    gc.collect()
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision("high")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(protocol["learning_rate"]),
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1,
        fused=True,
    )
    steps = int(protocol["steps"])
    micro_batch = int(protocol["micro_batch"])
    accumulation = int(protocol["accumulation"])
    seed = int(protocol["seed"])
    training_indices = select_indices(shard, 0, steps * micro_batch * accumulation, seed)
    validation_indices = select_indices(shard, 1, int(protocol["validation_pairs"]), seed + 1)
    before = evaluate(
        model,
        shard,
        validation_indices,
        device=device,
        batch_size=micro_batch,
        seed=seed + 2,
    )
    trace: list[dict[str, object]] = []
    torch.cuda.reset_peak_memory_stats(device)
    cursor = 0
    objective = str(protocol["objective"])
    cdr3_fraction = 0.15 if objective == "uniform_ce" else 0.25
    for update in range(steps):
        prepared = []
        selected_targets = 0
        for micro_step in range(accumulation):
            indices = training_indices[cursor : cursor + micro_batch]
            cursor += micro_batch
            item = _masked_batch(
                shard,
                indices,
                update=update * accumulation + micro_step,
                seed=seed,
                cdr3_fraction=cdr3_fraction,
            )
            selected_targets += int((item[3] != -100).sum())
            prepared.append(item)
        optimizer.zero_grad(set_to_none=True)
        loss_sum_value = 0.0
        model.train()
        for input_ids, types, _regions, labels, attention in prepared:
            input_tensor = torch.as_tensor(input_ids, dtype=torch.long, device=device)
            type_tensor = torch.as_tensor(types, dtype=torch.long, device=device)
            label_tensor = torch.as_tensor(labels, dtype=torch.long, device=device)
            attention_tensor = torch.as_tensor(attention, dtype=torch.bool, device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(
                    input_tensor,
                    type_tensor,
                    attention_mask=attention_tensor,
                    labels=label_tensor if objective != "cdr3_focal" else None,
                )
                loss_sum = (
                    focal_mlm_loss_sum(output.logits, label_tensor, gamma=2.0)
                    if objective == "cdr3_focal"
                    else output.loss_sum
                )
                loss = loss_sum / selected_targets
            loss.backward()
            loss_sum_value += float(loss_sum.detach())
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not math.isfinite(float(gradient_norm)) or not math.isfinite(loss_sum_value):
            raise RuntimeError("Canary produced nonfinite loss or gradient")
        optimizer.step()
        trace.append(
            {
                "gradient_norm": float(gradient_norm),
                "loss": loss_sum_value / selected_targets,
                "selected_targets": selected_targets,
                "update": update + 1,
            }
        )
    after = evaluate(
        model,
        shard,
        validation_indices,
        device=device,
        batch_size=micro_batch,
        seed=seed + 2,
    )
    nonfinite_parameter_count = sum(
        int((~torch.isfinite(parameter)).sum()) for parameter in model.parameters()
    )
    peak_allocated_bytes = torch.cuda.max_memory_allocated(device)
    total_device_bytes = torch.cuda.get_device_properties(device).total_memory
    peak_fraction = peak_allocated_bytes / total_device_bytes
    if nonfinite_parameter_count or peak_fraction > 0.92:
        raise RuntimeError("Canary parameter or memory gate failed")
    result = {
        "checkpoint_model_sha256": observed_model_sha256,
        "checkpoint_saved": False,
        "data_manifest_sha256": shard_manifest_sha256,
        "device_name": device_name,
        "format_version": "oas_paired_650m_training_canary_result_v1",
        "nonfinite_parameter_count": nonfinite_parameter_count,
        "objective": objective,
        "peak_allocated_bytes": peak_allocated_bytes,
        "peak_allocated_fraction": peak_fraction,
        "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "status": "PASS_PAIRED_650M_TRAINING_CANARY",
        "test_access": expected_test_access,
        "trace": trace,
        "validation_after": after,
        "validation_before": before,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("ascii")
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
