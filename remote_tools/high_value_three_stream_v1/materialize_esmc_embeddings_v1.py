#!/usr/bin/env python3
"""Materialize frozen SAbDab2 development ESM-C residue embeddings."""

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
from verify_esmc_model_runtime_v1 import normalize_body_key


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def atomic_write(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def artifact_path(root: Path, sequence_key: str) -> Path:
    return root / sequence_key[:2] / f"{sequence_key}.pt"


def load_model(
    checkpoint: Path, expected_sha: str, role: str, safetensors_site: Path
) -> MinimalESMC600M:
    if sha256_file(checkpoint) != expected_sha:
        raise RuntimeError("checkpoint SHA differs")
    sys.path.append(str(safetensors_site))
    from safetensors import safe_open

    state = {}
    with safe_open(checkpoint, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            normalized = normalize_body_key(key, role)
            if normalized is None:
                continue
            if normalized in state:
                raise RuntimeError("duplicate normalized state key")
            state[normalized] = handle.get_tensor(key)
    if len(state) != 362:
        raise RuntimeError("ESM-C body tensor count differs")
    with torch.device("meta"):
        model = MinimalESMC600M()
    incompatible = model.load_state_dict(state, strict=True, assign=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("strict ESM-C body load differs")
    model.materialize_nonpersistent_buffers(torch.device("cpu"))
    return model.to(device="cuda", dtype=torch.bfloat16).eval()


def select_canary(plan: dict[str, object], role: str) -> list[dict[str, object]]:
    antibodies = list(plan["antibodies"])
    paired = next(value for value in antibodies if value["paired"] is True)
    single = next(value for value in antibodies if value["paired"] is False)
    if role == "published_cdr":
        return [{**paired, "kind": "antibody"}, {**single, "kind": "antibody"}]
    antigens = list(plan["antigens"])
    shortest = min(antigens, key=lambda value: value["sequence_length"])
    longest = max(antigens, key=lambda value: value["sequence_length"])
    return [
        {**paired, "kind": "antibody"},
        {**single, "kind": "antibody"},
        {**shortest, "kind": "antigen"},
        {**longest, "kind": "antigen"},
    ]


def selected_units(
    plan: dict[str, object], role: str, shard_index: int, shard_count: int
) -> list[dict[str, object]]:
    units = [{**value, "kind": "antibody"} for value in plan["antibodies"]]
    if role == "base":
        units.extend({**value, "kind": "antigen"} for value in plan["antigens"])
    return [
        unit
        for unit in units
        if int(str(unit["sequence_sha256"])[:16], 16) % shard_count == shard_index
    ]


def batches(units: list[dict[str, object]]) -> list[list[dict[str, object]]]:
    ordered = sorted(units, key=lambda value: (len(value["model_input_sequence"]), value["sequence_sha256"]))
    result: list[list[dict[str, object]]] = []
    current: list[dict[str, object]] = []
    current_max = 0
    for unit in ordered:
        length = len(unit["model_input_sequence"]) + 2
        proposed_max = max(current_max, length)
        if current and (len(current) >= 16 or proposed_max * proposed_max * (len(current) + 1) > 1_000_000):
            result.append(current)
            current = []
            current_max = 0
        current.append(unit)
        current_max = max(current_max, length)
    if current:
        result.append(current)
    return result


def save_artifact(
    unit: dict[str, object], embedding: torch.Tensor, root: Path, role: str, checkpoint_sha: str
) -> dict[str, object]:
    expected_shape = (int(unit["sequence_length"]), 1152)
    embedding = embedding.float().cpu().contiguous()
    if tuple(embedding.shape) != expected_shape or not bool(torch.isfinite(embedding).all()):
        raise RuntimeError("residue embedding contract differs")
    path = artifact_path(root, str(unit["sequence_sha256"]))
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    artifact = {
        "format_version": "sppider_seq_embedding_v1",
        "sequence_sha256": unit["sequence_sha256"],
        "sequence_length": unit["sequence_length"],
        "model_name": "biohub/ESMC-600M" if role == "base" else "NOC-Lab/AbCDR-ESMC",
        "model_revision": (
            "a7e82012c83126b9eedb055fea9fa84b6c02f094"
            if role == "base"
            else "d6e08c7a771f6c7b399e9f94cd3b9b36d0ac12b8"
        ),
        "model_weight_sha256": checkpoint_sha,
        "embedding_dimension": 1152,
        "input_kind": unit["kind"],
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
        "sequence_sha256": unit["sequence_sha256"],
        "kind": unit["kind"],
        "sequence_length": unit["sequence_length"],
        "artifact_path": path.relative_to(root).as_posix(),
        "artifact_sha256": sha256_file(path),
        "artifact_bytes": path.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--role", choices=("base", "published_cdr"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--safetensors-site", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--canary", action="store_true")
    args = parser.parse_args()
    if sha256_file(args.plan) != args.plan_sha256:
        raise RuntimeError("development sequence plan SHA differs")
    plan = json.loads(args.plan.read_text())
    if (
        plan.get("status") != "FROZEN_TRAIN_VALIDATION_ONLY_BEFORE_EMBEDDING"
        or plan.get("authorization", {}).get("embedding_canary_authorized") is not True
        or plan.get("authorization", {}).get("test_access") is not False
    ):
        raise RuntimeError("development sequence plan contract differs")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard")
    if args.result.exists() or args.result.is_symlink():
        raise FileExistsError(args.result)
    units = (
        select_canary(plan, args.role)
        if args.canary
        else selected_units(plan, args.role, args.shard_index, args.shard_count)
    )
    model = load_model(args.checkpoint, args.checkpoint_sha256, args.role, args.safetensors_site)
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
                if unit["kind"] == "antibody" and bool(unit["paired"]):
                    heavy = int(unit["heavy_length"])
                    light = int(unit["light_length"])
                    residue_embedding = torch.cat(
                        (output[index, 1 : 1 + heavy], output[index, 2 + heavy : 2 + heavy + light]),
                        dim=0,
                    )
                else:
                    residue_embedding = output[index, 1 : 1 + length]
                records.append(
                    save_artifact(unit, residue_embedding, args.output_root, args.role, args.checkpoint_sha256)
                )
    result = {
        "format_version": "sabdab2_esmc_embedding_materialization_result_v1",
        "status": "PASS_ESMC_EMBEDDING_CANARY" if args.canary else "PASS_ESMC_EMBEDDING_SHARD",
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
        },
    }
    atomic_write(args.result, canonical_bytes(result))
    print(json.dumps({**result, "result_sha256": sha256_file(args.result), "records": "omitted"}, sort_keys=True))


if __name__ == "__main__":
    main()
