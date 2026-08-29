#!/usr/bin/env python3
"""Audit Stage-A1 head tensors without loading the 3B backbone."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
import torch.nn.functional as F

from antibody_pretraining.stage_a1_objective import StageA1Heads


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--initialization-seed", type=int, default=1701)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--temperature", type=float, default=0.07)
    return parser.parse_args()


def _tensor_summary(initial: torch.Tensor, trained: torch.Tensor) -> dict[str, object]:
    delta = trained - initial
    initial_norm = float(initial.norm())
    cosine = None
    if initial.numel() > 1 and initial_norm > 0.0 and float(trained.norm()) > 0.0:
        cosine = float(F.cosine_similarity(initial.flatten(), trained.flatten(), dim=0))
    relative_delta = float(delta.norm()) / initial_norm if initial_norm > 0.0 else None
    values = {
        "cosine_to_initialization": cosine,
        "delta_l2_norm": float(delta.norm()),
        "initial_abs_mean": float(initial.abs().mean()),
        "initial_l2_norm": initial_norm,
        "relative_delta_l2": relative_delta,
        "shape": list(initial.shape),
        "trained_abs_mean": float(trained.abs().mean()),
        "trained_l2_norm": float(trained.norm()),
    }
    if not all(
        value is None or isinstance(value, list) or math.isfinite(value)
        for value in values.values()
    ):
        raise RuntimeError("Nonfinite Stage-A1 head audit statistic")
    return values


def main() -> None:
    arguments = _arguments()
    checkpoint = arguments.checkpoint.resolve()
    if not checkpoint.is_dir() or not (checkpoint / ".metadata").is_file():
        raise FileNotFoundError(f"DCP generation is incomplete: {checkpoint}")

    torch.manual_seed(arguments.initialization_seed)
    initial = StageA1Heads(temperature=arguments.temperature)
    loaded = StageA1Heads(temperature=arguments.temperature)
    state = {"model": {"heads": loaded.state_dict()}}
    dcp.load(state, checkpoint_id=checkpoint)
    loaded.load_state_dict(state["model"]["heads"], strict=True)

    initial_parameters = dict(initial.named_parameters())
    loaded_parameters = dict(loaded.named_parameters())
    if initial_parameters.keys() != loaded_parameters.keys():
        raise RuntimeError("Stage-A1 head parameter names differ")
    parameters = {
        name: _tensor_summary(initial_parameters[name], loaded_parameters[name])
        for name in sorted(initial_parameters)
    }
    payload = {
        "checkpoint": str(checkpoint),
        "format_version": "abpairfm_stage_a1_dcp_head_audit_v1",
        "initialization_seed": arguments.initialization_seed,
        "parameters": parameters,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if arguments.output is None:
        print(rendered, end="")
        return
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("x", encoding="utf-8") as handle:
        handle.write(rendered)


if __name__ == "__main__":
    main()
