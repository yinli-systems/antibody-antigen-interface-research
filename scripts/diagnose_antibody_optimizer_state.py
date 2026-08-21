#!/usr/bin/env python3
"""Compare two DCP checkpoints by canonical model and named optimizer tensors."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

from antibody_pretraining.elastic_resume import semantic_optimizer_state
from antibody_pretraining.model import (
    AntibodyMaskedLanguageModel,
    config_from_preset,
    load_model_family,
)
from antibody_pretraining.training_contract import load_training_protocol
import run_antibody_pretraining_optimized as engine


FORMAT_VERSION = "antibody_optimizer_state_diagnostic_v1"


class OptimizerDiagnosticError(RuntimeError):
    """Report an invalid checkpoint or incomparable state tree."""


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


def _load_checkpoint(
    checkpoint_root: Path,
    *,
    family: dict[str, Any],
    training: dict[str, Any],
    preset: str,
) -> tuple[torch.nn.Module, torch.optim.Optimizer]:
    model = AntibodyMaskedLanguageModel(config_from_preset(family, preset))
    optimizer = engine._optimizer(model, training, preset, False)
    model_state, optimizer_state = get_state_dict(model, optimizer)
    state = {"model": model_state, "optimizer": optimizer_state}
    dcp.load(state, checkpoint_id=str(checkpoint_root))
    set_state_dict(
        model,
        optimizer,
        model_state_dict=state["model"],
        optim_state_dict=state["optimizer"],
    )
    return model, optimizer


def _tensor_delta(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    if left.dtype != right.dtype or left.shape != right.shape:
        raise OptimizerDiagnosticError("Tensor schema differs")
    equal = torch.equal(left, right)
    if left.is_floating_point():
        left64 = left.detach().to(dtype=torch.float64)
        right64 = right.detach().to(dtype=torch.float64)
        difference = (left64 - right64).abs()
        denominator = torch.maximum(left64.abs(), right64.abs())
        relative = difference / denominator.clamp_min(torch.finfo(torch.float64).tiny)
        result = {
            "different_elements": int(left.ne(right).sum()),
            "dtype": str(left.dtype),
            "equal": equal,
            "l2_difference": float(torch.linalg.vector_norm(difference)),
            "maximum_absolute_difference": float(difference.max()) if left.numel() else 0.0,
            "maximum_relative_difference": float(relative.max()) if left.numel() else 0.0,
            "shape": list(left.shape),
        }
        if left.dtype == torch.float32:
            left_bits = left.detach().contiguous().view(torch.int32).to(torch.int64)
            right_bits = right.detach().contiguous().view(torch.int32).to(torch.int64)
            offset = 1 << 31
            left_ordered = torch.where(
                left_bits < 0,
                offset - left_bits,
                offset + left_bits,
            )
            right_ordered = torch.where(
                right_bits < 0,
                offset - right_bits,
                offset + right_bits,
            )
            result["maximum_ulp_distance"] = (
                int((left_ordered - right_ordered).abs().max())
                if left.numel()
                else 0
            )
        return result
    return {
        "different_elements": int(left.ne(right).sum()),
        "dtype": str(left.dtype),
        "equal": equal,
        "shape": list(left.shape),
    }


def compare(
    *,
    left_root: Path,
    right_root: Path,
    model_family: Path,
    training_protocol: Path,
    preset: str,
) -> dict[str, Any]:
    family = load_model_family(model_family)
    training = load_training_protocol(training_protocol)
    left_model, left_optimizer = _load_checkpoint(
        left_root,
        family=family,
        training=training,
        preset=preset,
    )
    right_model, right_optimizer = _load_checkpoint(
        right_root,
        family=family,
        training=training,
        preset=preset,
    )
    left_model_state = left_model.state_dict()
    right_model_state = right_model.state_dict()
    if set(left_model_state) != set(right_model_state):
        raise OptimizerDiagnosticError("Model tensor inventory differs")
    model_differences = {
        name: _tensor_delta(left_model_state[name], right_model_state[name])
        for name in sorted(left_model_state)
        if not torch.equal(left_model_state[name], right_model_state[name])
    }

    left = semantic_optimizer_state(left_model, left_optimizer)
    right = semantic_optimizer_state(right_model, right_optimizer)
    groups_equal = left["param_groups"] == right["param_groups"]
    left_state = left["state"]
    right_state = right["state"]
    if set(left_state) != set(right_state):
        raise OptimizerDiagnosticError("Named optimizer parameter inventory differs")
    field_summaries: dict[str, dict[str, Any]] = {}
    tensor_differences = []
    non_tensor_differences = []
    for name in sorted(left_state):
        left_fields = left_state[name]
        right_fields = right_state[name]
        if set(left_fields) != set(right_fields):
            raise OptimizerDiagnosticError(f"Optimizer field inventory differs: {name}")
        for field in sorted(left_fields):
            left_value = left_fields[field]
            right_value = right_fields[field]
            if isinstance(left_value, torch.Tensor) and isinstance(
                right_value, torch.Tensor
            ):
                delta = _tensor_delta(left_value, right_value)
                aggregate = field_summaries.setdefault(
                    field,
                    {
                        "different_elements": 0,
                        "different_tensors": 0,
                        "maximum_absolute_difference": 0.0,
                        "maximum_relative_difference": 0.0,
                        "maximum_ulp_distance": 0,
                        "tensor_count": 0,
                    },
                )
                aggregate["tensor_count"] += 1
                aggregate["different_elements"] += delta["different_elements"]
                if not delta["equal"]:
                    aggregate["different_tensors"] += 1
                    aggregate["maximum_absolute_difference"] = max(
                        aggregate["maximum_absolute_difference"],
                        delta.get("maximum_absolute_difference", 0.0),
                    )
                    aggregate["maximum_relative_difference"] = max(
                        aggregate["maximum_relative_difference"],
                        delta.get("maximum_relative_difference", 0.0),
                    )
                    aggregate["maximum_ulp_distance"] = max(
                        aggregate["maximum_ulp_distance"],
                        delta.get("maximum_ulp_distance", 0),
                    )
                    tensor_differences.append(
                        {"field": field, "parameter": name, **delta}
                    )
            elif left_value != right_value:
                non_tensor_differences.append(
                    {
                        "field": field,
                        "left": left_value,
                        "parameter": name,
                        "right": right_value,
                    }
                )
    tensor_differences.sort(
        key=lambda value: (
            value.get("maximum_absolute_difference", 0.0),
            value["parameter"],
            value["field"],
        ),
        reverse=True,
    )
    if any(
        not math.isfinite(float(value.get("maximum_absolute_difference", 0.0)))
        for value in tensor_differences
    ):
        raise OptimizerDiagnosticError("Optimizer comparison is non-finite")

    def checkpoint_inventory(root: Path) -> dict[str, Any]:
        files = []
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            files.append(
                {
                    "bytes": path.stat().st_size,
                    "path": str(path.relative_to(root)),
                    "sha256": digest.hexdigest(),
                }
            )
        if not files:
            raise OptimizerDiagnosticError("Checkpoint inventory is empty")
        return {
            "files": files,
            "inventory_sha256": hashlib.sha256(_canonical(files)).hexdigest(),
        }

    checkpoint_inventories = {
        "left": checkpoint_inventory(left_root),
        "right": checkpoint_inventory(right_root),
    }
    return {
        "authorization": {
            "diagnostic_only": True,
            "production_pretraining": False,
            "scientific_claim": False,
        },
        "checkpoint_roots": {
            "left": str(left_root.resolve()),
            "right": str(right_root.resolve()),
        },
        "checkpoint_inventories": checkpoint_inventories,
        "field_summaries": field_summaries,
        "format_version": FORMAT_VERSION,
        "implementation_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "model": {
            "different_tensors": len(model_differences),
            "exact": not model_differences,
            "tensor_differences": model_differences,
        },
        "optimizer": {
            "exact": (
                groups_equal
                and not tensor_differences
                and not non_tensor_differences
            ),
            "named_parameter_count": len(left_state),
            "non_tensor_differences": non_tensor_differences,
            "param_groups_equal": groups_equal,
            "top_tensor_differences": tensor_differences[:32],
        },
        "preset": preset,
        "test_access": {
            "downstream_labels_used": False,
            "sabdab2_heldout_accessed": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--model-family", type=Path, required=True)
    parser.add_argument("--training-protocol", type=Path, required=True)
    parser.add_argument("--preset", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = compare(
        left_root=args.left,
        right_root=args.right,
        model_family=args.model_family,
        training_protocol=args.training_protocol,
        preset=args.preset,
    )
    payload["diagnostic_sha256"] = hashlib.sha256(_canonical(payload)).hexdigest()
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(_canonical(payload))


if __name__ == "__main__":
    main()
