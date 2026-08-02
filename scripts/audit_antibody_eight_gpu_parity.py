#!/usr/bin/env python3
"""Audit DDP and FSDP2 receipts from an identical eight-GPU workload."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


LIMITS = {
    "gradient_norm_absolute": 0.02,
    "loss_absolute": 0.0002,
    "parameter_square_sum_absolute": 1e-6,
    "parameter_sum_absolute": 1e-5,
}


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="ascii"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ddp", type=Path, required=True)
    parser.add_argument("--fsdp2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ddp = _load(args.ddp)
    fsdp2 = _load(args.fsdp2)
    if ddp["backend"] != "ddp" or fsdp2["backend"] != "fsdp2":
        raise RuntimeError("Backend receipts are mislabeled")
    anchors = ("model_family_sha256", "shard_manifest_sha256", "training_protocol_sha256", "world_size")
    anchor_equal = all(ddp[key] == fsdp2[key] for key in anchors)
    if len(ddp["trace"]) != len(fsdp2["trace"]):
        raise RuntimeError("Backend trace lengths differ")
    trace_pairs = list(zip(ddp["trace"], fsdp2["trace"]))
    membership_equal = all(
        all(left[key] == right[key] for key in ("consumed_residue_tokens", "residue_tokens", "selected_targets", "update"))
        for left, right in trace_pairs
    )
    differences = {
        "gradient_norm_absolute_maximum": max(abs(left["gradient_norm"] - right["gradient_norm"]) for left, right in trace_pairs),
        "loss_absolute_maximum": max(abs(left["loss"] - right["loss"]) for left, right in trace_pairs),
        "parameter_square_sum_absolute": abs(ddp["parameter_moments"]["square_sum"] - fsdp2["parameter_moments"]["square_sum"]),
        "parameter_sum_absolute": abs(ddp["parameter_moments"]["sum"] - fsdp2["parameter_moments"]["sum"]),
    }
    numerical_pass = (
        differences["gradient_norm_absolute_maximum"] <= LIMITS["gradient_norm_absolute"]
        and differences["loss_absolute_maximum"] <= LIMITS["loss_absolute"]
        and differences["parameter_square_sum_absolute"] <= LIMITS["parameter_square_sum_absolute"]
        and differences["parameter_sum_absolute"] <= LIMITS["parameter_sum_absolute"]
        and ddp["parameter_moments"]["count"] == fsdp2["parameter_moments"]["count"]
    )
    ddp_steady = sum(item["seconds"] for item in ddp["trace"][1:]) / len(ddp["trace"][1:])
    fsdp2_steady = sum(item["seconds"] for item in fsdp2["trace"][1:]) / len(fsdp2["trace"][1:])
    payload = {
        "anchors_equal": anchor_equal,
        "claim_boundary": "three-update eight-GPU correctness only; efficiency gate remains closed",
        "differences": differences,
        "format_version": "antibody_eight_gpu_parity_v1",
        "limits": LIMITS,
        "membership_equal": membership_equal,
        "numerical_pass": numerical_pass,
        "steady_step_seconds": {"ddp": ddp_steady, "fsdp2": fsdp2_steady},
        "test_access": {"downstream_labels_used": False, "sabdab2_heldout_accessed": False},
    }
    if not anchor_equal or not membership_equal or not numerical_pass:
        raise RuntimeError(f"Eight-GPU parity failed: {payload}")
    data = (json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("ascii")
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)


if __name__ == "__main__":
    main()
