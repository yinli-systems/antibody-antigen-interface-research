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
    result = dict(source)
    result["claim_boundary"] = (
        "Eight-GPU real-OAS recovery pilot only; no production training, "
        "scale promotion, downstream, biological, official-test, or SOTA claim"
    )
    result["pilot"] = {
        "source_protocol_sha256": hashlib.sha256(_canonical(source)).hexdigest(),
        "world_size": 8,
        "continuous_steps": args.continuous_steps,
        "stage1_steps": args.stage1_steps,
        "resumed_steps": args.stage1_steps,
        "target_residue_tokens_per_update": 262144,
        "backend": "ddp",
        "authorization_mode": "recovery_gate",
    }
    result["execution"] = dict(execution)
    result["execution"]["world_size"] = 8
    result["execution"]["data_loader"] = dict(execution["data_loader"])
    result["runs"] = {
        "continuous": {
            "batch_quantum_sequences": 512,
            "checkpoint": False,
            "micro_batch_per_gpu": 16,
            "preset": "abf_650m",
            "resume": False,
            "run_seed": 17,
            "steps": args.continuous_steps,
            "target_residue_tokens": 262144,
        },
        "stage1": {
            "batch_quantum_sequences": 512,
            "checkpoint": True,
            "micro_batch_per_gpu": 16,
            "preset": "abf_650m",
            "resume": False,
            "run_seed": 17,
            "steps": args.stage1_steps,
            "target_residue_tokens": 262144,
        },
        "resumed": {
            "batch_quantum_sequences": 512,
            "checkpoint": False,
            "micro_batch_per_gpu": 16,
            "preset": "abf_650m",
            "resume": True,
            "run_seed": 17,
            "steps": args.stage1_steps,
            "target_residue_tokens": 262144,
        },
    }
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(_canonical(result))
        handle.flush()
        os.fsync(handle.fileno())


if __name__ == "__main__":
    main()
