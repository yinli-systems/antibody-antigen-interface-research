#!/usr/bin/env python3
"""Localize a full-model DCP parity failure without weakening the gate."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

import torch


def _load_gate(project_root: Path) -> Any:
    path = project_root / "scripts/run_abpairfm_3b_dcp_gate.py"
    specification = importlib.util.spec_from_file_location("abpairfm_dcp_gate", path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Cannot load recovery gate: {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _digest(gate: Any, value: object) -> str:
    digest = hashlib.sha256()
    gate._update_tree_digest(digest, value)
    return digest.hexdigest()


def _details(gate: Any, model: torch.nn.Module, optimizer: Any) -> dict[str, Any]:
    model_state = model.state_dict()
    optimizer_state = optimizer.state_dict()
    state = optimizer_state["state"]
    groups = optimizer_state["param_groups"]
    return {
        "model": {
            "overall": _digest(gate, model_state),
            "by_key": {name: _digest(gate, value) for name, value in model_state.items()},
        },
        "optimizer": {
            "overall": _digest(gate, optimizer_state),
            "param_groups": _digest(gate, groups),
            "state_by_slot": {
                str(slot): _digest(gate, state[slot])
                for slot in sorted(state, key=lambda item: (type(item).__name__, repr(item)))
            },
        },
    }


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("baseline", "restore"), required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-size", type=int, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--save-result", type=Path, required=True)
    parser.add_argument("--expected-save-result-sha256", required=True)
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    project_root = arguments.project_root.resolve()
    gate = _load_gate(project_root)
    if gate._source_sha256(project_root) != arguments.expected_source_sha256:
        raise RuntimeError("Recovery-gate source anchor differs")
    if gate._file_sha256(arguments.protocol.resolve()) != arguments.expected_protocol_sha256:
        raise RuntimeError("Recovery-gate protocol anchor differs")
    save_result = gate._read_authenticated_result(
        arguments.save_result,
        arguments.expected_save_result_sha256,
    )

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.distributed.init_process_group(backend="nccl", device_id=device)
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    try:
        if world_size != 4:
            raise RuntimeError("Diagnostic requires the frozen four-GPU topology")
        if torch.cuda.get_device_properties(device).name != "NVIDIA GeForce RTX 4090":
            raise RuntimeError("Diagnostic is restricted to gpu_4090")
        torch.manual_seed(1701 + rank)
        torch.cuda.manual_seed(1701 + rank)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        model, optimizer, initialization = gate._build(arguments, rank=rank)
        records = []
        if arguments.mode == "baseline":
            records.extend(
                gate._train_step(
                    model,
                    optimizer,
                    rank=rank,
                    world_size=world_size,
                    device=device,
                    sequence_tokens=1024,
                    seed=1701,
                    step=step,
                )
                for step in range(2)
            )
            gate.reshard_pair_aware_esm2(model)
            from torch.distributed.checkpoint.state_dict import (
                StateDictOptions,
                get_state_dict,
            )

            checkpoint_view = get_state_dict(
                model,
                optimizer,
                options=StateDictOptions(
                    full_state_dict=False,
                    cpu_offload=False,
                    strict=True,
                ),
            )
            del checkpoint_view
        else:
            checkpoint = save_result["checkpoint"]
            gate.load_dcp_generation(
                checkpoint["root"],
                checkpoint["generation"],
                model=model,
                optimizer=optimizer,
                expected_provenance_hashes=save_result["provenance_hashes"],
            )
        records.extend(
            gate._train_step(
                model,
                optimizer,
                rank=rank,
                world_size=world_size,
                device=device,
                sequence_tokens=1024,
                seed=1701,
                step=step,
            )
            for step in range(2, 3)
        )
        gate.reshard_pair_aware_esm2(model)
        evidence = {
            "claim_boundary": "Diagnostic evidence only; not a passing recovery gate.",
            "details": _details(gate, model, optimizer),
            "initialization": initialization,
            "mode": arguments.mode,
            "rank": rank,
            "records": records,
            "source_sha256": arguments.expected_source_sha256,
            "status": "DIAGNOSTIC_ONLY",
            "test_access": False,
        }
        gate._publish_create_only(
            arguments.output_root / f"{arguments.mode}-rank-{rank}.json",
            evidence,
        )
        torch.distributed.barrier()
    finally:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
