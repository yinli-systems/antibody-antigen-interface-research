#!/usr/bin/env python3
"""Audit exact FSDP2 and bounded DDP recovery evidence for the 300M model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


PARAMETER_COUNT = 308_365_312
PER_PARAMETER_LIMIT = 1e-12


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="ascii"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ddp-continuous", type=Path, required=True)
    parser.add_argument("--ddp-resumed", type=Path, required=True)
    parser.add_argument("--ddp-continuous-state", type=Path, required=True)
    parser.add_argument("--ddp-resumed-state", type=Path, required=True)
    parser.add_argument("--fsdp2-recovery", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    continuous = _load(args.ddp_continuous)
    resumed = _load(args.ddp_resumed)
    continuous_state = _load(args.ddp_continuous_state)
    resumed_state = _load(args.ddp_resumed_state)
    fsdp2 = _load(args.fsdp2_recovery)
    if (
        continuous["backend"] != "ddp"
        or resumed["backend"] != "ddp"
        or continuous["preset"] != "abf_300m"
        or resumed["preset"] != "abf_300m"
        or continuous["parameter_moments"]["count"] != PARAMETER_COUNT
        or resumed["parameter_moments"]["count"] != PARAMETER_COUNT
    ):
        raise RuntimeError("DDP recovery inputs are mislabeled")
    if fsdp2["backend"] != "fsdp2" or fsdp2["world_size"] != 8:
        raise RuntimeError("FSDP2 recovery input is mislabeled")
    keys = (
        "consumed_residue_tokens",
        "gradient_norm",
        "learning_rate",
        "loss",
        "residue_tokens",
        "selected_targets",
        "update",
    )
    continuation_equal = all(
        all(left[key] == right[key] for key in keys)
        for left, right in zip(continuous["trace"][-2:], resumed["trace"])
    )
    moment_differences = {
        "square_sum_per_parameter": abs(
            continuous["parameter_moments"]["square_sum"]
            - resumed["parameter_moments"]["square_sum"]
        )
        / PARAMETER_COUNT,
        "sum_per_parameter": abs(
            continuous["parameter_moments"]["sum"]
            - resumed["parameter_moments"]["sum"]
        )
        / PARAMETER_COUNT,
    }
    ddp_numerical_continuation_pass = (
        continuation_equal
        and continuous_state == resumed_state
        and all(value <= PER_PARAMETER_LIMIT for value in moment_differences.values())
    )
    ddp_exact = {
        "model": continuous["final_model_sha256"]
        == resumed["final_model_sha256"],
        "optimizer": continuous["final_optimizer_sha256"]
        == resumed["final_optimizer_sha256"],
        "trainer_state": continuous_state == resumed_state,
    }
    fsdp2_exact = all(fsdp2["equal"].values())
    payload = {
        "authorization": {
            "exact_recovery_backend": "fsdp2" if fsdp2_exact else None,
            "production_pretraining": False,
            "scale_32_gpu": False,
            "scale_64_gpu": False,
        },
        "claim_boundary": "300M same-world-size eight-GPU engineering recovery; DDP optimizer numerical equivalence is not inferred from model moments",
        "ddp": {
            "exact_equal": ddp_exact,
            "exact_pass": all(ddp_exact.values()),
            "moment_differences": moment_differences,
            "numerical_continuation_pass": ddp_numerical_continuation_pass,
            "optimizer_numerical_equivalence_assessed": False,
            "receipt_sha256": {
                "continuous": _sha256(args.ddp_continuous),
                "resumed": _sha256(args.ddp_resumed),
            },
        },
        "format_version": "antibody_scale_recovery_v1",
        "fsdp2": {
            "exact_equal": fsdp2["equal"],
            "exact_pass": fsdp2_exact,
            "recovery_receipt_sha256": _sha256(args.fsdp2_recovery),
        },
        "limits": {"parameter_moment_difference_per_parameter": PER_PARAMETER_LIMIT},
        "parameter_count": PARAMETER_COUNT,
        "preset": "abf_300m",
        "test_access": {
            "downstream_labels_used": False,
            "sabdab2_heldout_accessed": False,
        },
        "world_size": 8,
    }
    if not ddp_numerical_continuation_pass or not fsdp2_exact:
        raise RuntimeError(f"300M recovery audit failed: {payload}")
    data = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("ascii")
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)


if __name__ == "__main__":
    main()
