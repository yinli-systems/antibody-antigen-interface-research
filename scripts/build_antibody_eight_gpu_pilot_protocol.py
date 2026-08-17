#!/usr/bin/env python3
"""Derive a create-only 8-GPU recovery-pilot protocol from a 4-GPU gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def _load(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    value = json.loads(raw.decode("ascii"))
    if not isinstance(value, dict) or raw != _canonical(value):
        raise ValueError(f"input must be canonical JSON object: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--continuous-steps", type=int, default=200)
    parser.add_argument("--stage1-steps", type=int, default=100)
    parser.add_argument(
        "--target-gpu-model",
        choices=("NVIDIA GeForce RTX 4090", "NVIDIA GeForce RTX 5090"),
        default="NVIDIA GeForce RTX 5090",
    )
    parser.add_argument("--target-world-size", type=int, choices=(6, 8), default=8)
    args = parser.parse_args()
    if args.continuous_steps <= 0 or args.stage1_steps <= 0:
        raise ValueError("step counts must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)
    source = _load(args.source)
    if source.get("format_version") != "antibody_production_loader_recovery_gate_v1":
        raise ValueError("source is not a loader recovery gate")
    if source.get("authorization") != {
        "loader_recovery_gate": True,
        "production_pretraining": False,
        "scale_out": False,
        "scientific_claim": False,
    }:
        raise ValueError("source authorization is not recovery-only")
    if source.get("test_access") != {
        "downstream_labels_used": False,
        "sabdab2_heldout_accessed": False,
    }:
        raise ValueError("source test-access boundary differs")
    execution = source.get("execution")
    if not isinstance(execution, dict) or execution.get("world_size") != 4:
        raise ValueError("source must be the four-GPU gate")
    optimization = execution.get("optimization")
    if not isinstance(optimization, dict) or optimization.get("compile_blocks"):
        raise ValueError("pilot only accepts the safe-eager source gate")
    hardware = source.get("hardware_identity")
    if not isinstance(hardware, dict) or hardware != {
        "gpu_count": 4,
        "gpu_model": "NVIDIA GeForce RTX 5090",
        "node_count": 1,
    }:
        raise ValueError("source hardware identity differs")
    source_gpu_model = str(hardware["gpu_model"])
    cross_hardware = args.target_gpu_model != source_gpu_model
    target_tokens = 131072 * args.target_world_size // 4
    batch_quantum = 256 * args.target_world_size // 4
    result = dict(source)
    result["claim_boundary"] = (
        "Single-node scale real-OAS recovery pilot only; cross-hardware engineering "
        "evidence only when the target GPU differs from the four-GPU source; "
        "no production training, scale promotion, downstream, biological, "
        "official-test, or SOTA claim"
    )
    result["pilot"] = {
        "source_protocol_sha256": hashlib.sha256(_canonical(source)).hexdigest(),
        "world_size": args.target_world_size,
        "continuous_steps": args.continuous_steps,
        "stage1_steps": args.stage1_steps,
        "resumed_steps": args.stage1_steps,
        "target_residue_tokens_per_update": target_tokens,
        "batch_quantum_sequences": batch_quantum,
        "backend": "ddp",
        "authorization_mode": "recovery_gate",
        "cross_hardware": cross_hardware,
        "source_gpu_model": source_gpu_model,
        "target_gpu_model": args.target_gpu_model,
    }
    result["execution"] = dict(execution)
    result["execution"]["world_size"] = args.target_world_size
    result["execution"]["data_loader"] = dict(execution["data_loader"])
    result["hardware_identity"] = dict(hardware)
    result["hardware_identity"]["gpu_count"] = args.target_world_size
    result["hardware_identity"]["gpu_model"] = args.target_gpu_model
    result["runs"] = {
        "continuous": {
            "batch_quantum_sequences": batch_quantum,
            "checkpoint": False,
            "micro_batch_per_gpu": 16,
            "preset": "abf_650m",
            "resume": False,
            "run_seed": 17,
            "steps": args.continuous_steps,
            "target_residue_tokens": target_tokens,
        },
        "stage1": {
            "batch_quantum_sequences": batch_quantum,
            "checkpoint": True,
            "micro_batch_per_gpu": 16,
            "preset": "abf_650m",
            "resume": False,
            "run_seed": 17,
            "steps": args.stage1_steps,
            "target_residue_tokens": target_tokens,
        },
        "resumed": {
            "batch_quantum_sequences": batch_quantum,
            "checkpoint": False,
            "micro_batch_per_gpu": 16,
            "preset": "abf_650m",
            "resume": True,
            "run_seed": 17,
            "steps": args.stage1_steps,
            "target_residue_tokens": target_tokens,
        },
    }
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(_canonical(result))
        handle.flush()
        os.fsync(handle.fileno())


if __name__ == "__main__":
    main()
