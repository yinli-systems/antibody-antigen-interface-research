#!/usr/bin/env python3
"""Materialize development antibody embeddings from fixed epoch-4 adapted ESM-C."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import torch

from minimal_esmc_runtime_v1 import MinimalESMC600M, PAD_TOKEN_ID, tokenize_sequence


ROLES = {
    "whole_chain_adapted": "local/ESMC-600M-whole-chain-adapted-epoch4",
    "cdr_focused_adapted": "local/ESMC-600M-cdr-focused-adapted-epoch4",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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


def load_model(checkpoint: Path, expected_sha: str, safetensors_site: Path) -> MinimalESMC600M:
    if sha256_file(checkpoint) != expected_sha:
        raise RuntimeError("adapted checkpoint SHA differs")
    sys.path.append(str(safetensors_site))
    from safetensors import safe_open

    state = {}
    with safe_open(checkpoint, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if key.startswith("sequence_head."):
                continue
            if not key.startswith("body."):
                raise RuntimeError(f"unexpected adapted checkpoint key: {key}")
            normalized = key.removeprefix("body.")
            if normalized in state:
                raise RuntimeError("duplicate normalized adapted state key")
            state[normalized] = handle.get_tensor(key)
    if len(state) != 362:
        raise RuntimeError(f"adapted ESM-C body tensor count differs: {len(state)}")
    with torch.device("meta"):
        model = MinimalESMC600M()
    incompatible = model.load_state_dict(state, strict=True, assign=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("strict adapted ESM-C body load differs")
    model.materialize_nonpersistent_buffers(torch.device("cpu"))
    return model.to(device="cuda", dtype=torch.bfloat16).eval()


def selected_units(
    plan: dict[str, object], shard_index: int, shard_count: int, canary: bool
) -> list[dict[str, object]]:
    antibodies = list(plan["antibodies"])
    if canary:
        return [
            next(value for value in antibodies if value["paired"] is True),
            next(value for value in antibodies if value["paired"] is False),
        ]
    return [
        unit
        for unit in antibodies
        if int(str(unit["sequence_sha256"])[:16], 16) % shard_count == shard_index
    ]


def batches(units: list[dict[str, object]]) -> list[list[dict[str, object]]]:
    ordered = sorted(
        units, key=lambda value: (len(value["model_input_sequence"]), value["sequence_sha256"])
    )
    result: list[list[dict[str, object]]] = []
    current: list[dict[str, object]] = []
    current_max = 0
    for unit in ordered:
        length = len(unit["model_input_sequence"]) + 2
        proposed_max = max(current_max, length)
        if current and (
            len(current) >= 16
            or proposed_max * proposed_max * (len(current) + 1) > 1_000_000
        ):
            result.append(current)
            current = []
            current_max = 0
        current.append(unit)
        current_max = max(current_max, length)
    if current:
        result.append(current)
    return result


def save_artifact(
    unit: dict[str, object],
    embedding: torch.Tensor,
    root: Path,
    role: str,
    checkpoint_sha: str,
) -> dict[str, object]:
    embedding = embedding.float().cpu().contiguous()
    expected_shape = (int(unit["sequence_length"]), 1152)
    if tuple(embedding.shape) != expected_shape or not bool(torch.isfinite(embedding).all()):
        raise RuntimeError("adapted residue embedding contract differs")
    key = str(unit["sequence_sha256"])
    path = root / key[:2] / f"{key}.pt"
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    artifact = {
        "format_version": "sppider_seq_embedding_v1",
        "sequence_sha256": key,
        "sequence_length": unit["sequence_length"],
        "model_name": ROLES[role],
        "model_revision": "fixed-epoch-4",
        "model_weight_sha256": checkpoint_sha,
        "embedding_dimension": 1152,
        "input_kind": "antibody",
        "paired_input": bool(unit.get("paired", False)),
        "separator_token_id": 30 if unit.get("paired", False) else None,
        "max_tokens": 2048,
        "stride": 2046,
        "starts": [0],
        "ends": [unit["sequence_length"]],
        "chunks": [embedding],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(artifact, temporary)
    os.replace(temporary, path)
    return {
        "sequence_sha256": key,
        "sequence_length": unit["sequence_length"],
        "artifact_path": path.relative_to(root).as_posix(),
        "artifact_sha256": sha256_file(path),
        "artifact_bytes": path.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--role", choices=tuple(ROLES), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--safetensors-site", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--canary", action="store_true")
    args = parser.parse_args()
    if sha256_file(args.protocol) != args.protocol_sha256:
        raise RuntimeError("adapted embedding protocol SHA differs")
    protocol = json.loads(args.protocol.read_text())
    binding = protocol["adapted_antibody_models"][args.role]
    if (
        protocol.get("status") != "FROZEN_AFTER_ADAPTATION_BEFORE_EMBEDDING"
        or protocol.get("test_access") is not False
        or binding.get("checkpoint") != str(args.checkpoint)
        or binding.get("checkpoint_sha256") != args.checkpoint_sha256
        or sha256_file(args.plan) != args.plan_sha256
        or protocol.get("development_sequence_plan_sha256") != args.plan_sha256
    ):
        raise RuntimeError("adapted embedding authority differs")
    plan = json.loads(args.plan.read_text())
    if (
        plan.get("status") != "FROZEN_TRAIN_VALIDATION_ONLY_BEFORE_EMBEDDING"
        or plan.get("authorization", {}).get("test_access") is not False
    ):
        raise RuntimeError("development sequence plan contract differs")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard")
    units = selected_units(plan, args.shard_index, args.shard_count, args.canary)
    model = load_model(args.checkpoint, args.checkpoint_sha256, args.safetensors_site)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    records = []
    with torch.inference_mode():
        for group in batches(units):
            encoded = [tokenize_sequence(str(unit["model_input_sequence"])) for unit in group]
            maximum = max(value.shape[0] for value in encoded)
            token_batch = torch.full((len(group), maximum), PAD_TOKEN_ID, dtype=torch.int64)
            for index, value in enumerate(encoded):
                token_batch[index, : value.shape[0]] = value
            output = model(token_batch.cuda()).embeddings
            for index, unit in enumerate(group):
                length = int(unit["sequence_length"])
                if bool(unit["paired"]):
                    heavy = int(unit["heavy_length"])
                    light = int(unit["light_length"])
                    residue_embedding = torch.cat(
                        (
                            output[index, 1 : 1 + heavy],
                            output[index, 2 + heavy : 2 + heavy + light],
                        ),
                        dim=0,
                    )
                else:
                    residue_embedding = output[index, 1 : 1 + length]
                records.append(
                    save_artifact(unit, residue_embedding, args.output_root, args.role, args.checkpoint_sha256)
                )
    result = {
        "format_version": "sabdab2_adapted_esmc_embedding_result_v1",
        "status": (
            "PASS_ADAPTED_ESMC_EMBEDDING_CANARY"
            if args.canary
            else "PASS_ADAPTED_ESMC_EMBEDDING_SHARD"
        ),
        "protocol_sha256": args.protocol_sha256,
        "plan_sha256": args.plan_sha256,
        "role": args.role,
        "checkpoint_sha256": args.checkpoint_sha256,
        "canary": args.canary,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "record_count": len(records),
        "records": records,
        "runtime": {
            "elapsed_seconds": time.monotonic() - started,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
        },
        "audit": {
            "strict_model_load": True,
            "embedding_dimension": 1152,
            "finite": True,
            "task_labels_read": False,
            "test_access": False,
            "training_started": False,
            "antigen_embedding_materialized": False,
        },
    }
    atomic_write(args.result, result)
    print(json.dumps({**result, "records": "omitted", "result_sha256": sha256_file(args.result)}, sort_keys=True))


if __name__ == "__main__":
    main()
