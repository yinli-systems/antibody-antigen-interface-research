#!/usr/bin/env python3
"""Audit identical-work ZeRO-2 and ZeRO-3 numerical receipts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


LIMITS = {
    "gradient_norm_absolute": 1e-4,
    "loss_absolute": 3e-8,
    "parameter_square_sum_absolute": 1e-6,
    "parameter_sum_absolute": 1e-6,
}


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="ascii"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zero2", type=Path, required=True)
    parser.add_argument("--zero3", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    zero2 = _load(args.zero2)
    zero3 = _load(args.zero3)
    if zero2["backend"] != "zero2" or zero3["backend"] != "zero3":
        raise RuntimeError("ZeRO receipts are mislabeled")
    if len(zero2["trace"]) != len(zero3["trace"]):
        raise RuntimeError("ZeRO trace lengths differ")
    pairs = list(zip(zero2["trace"], zero3["trace"]))
    membership_equal = all(
        all(left[key] == right[key] for key in ("consumed_residue_tokens", "residue_tokens", "selected_targets", "update"))
        for left, right in pairs
    )
    differences = {
        "gradient_norm_absolute_maximum": max(abs(left["gradient_norm"] - right["gradient_norm"]) for left, right in pairs),
        "loss_absolute_maximum": max(abs(left["loss"] - right["loss"]) for left, right in pairs),
        "parameter_square_sum_absolute": abs(zero2["parameter_moments"]["square_sum"] - zero3["parameter_moments"]["square_sum"]),
        "parameter_sum_absolute": abs(zero2["parameter_moments"]["sum"] - zero3["parameter_moments"]["sum"]),
    }
    passed = (
        membership_equal
        and differences["gradient_norm_absolute_maximum"] <= LIMITS["gradient_norm_absolute"]
        and differences["loss_absolute_maximum"] <= LIMITS["loss_absolute"]
        and differences["parameter_square_sum_absolute"] <= LIMITS["parameter_square_sum_absolute"]
        and differences["parameter_sum_absolute"] <= LIMITS["parameter_sum_absolute"]
        and zero2["parameter_moments"]["count"] == zero3["parameter_moments"]["count"]
    )
    payload = {
        "claim_boundary": "three-update ZeRO-2 versus ZeRO-3 internal parity only; DDP parity and efficiency remain open",
        "differences": differences,
        "format_version": "antibody_deepspeed_parity_v1",
        "limits": LIMITS,
        "membership_equal": membership_equal,
        "numerical_pass": passed,
        "test_access": {"downstream_labels_used": False, "sabdab2_heldout_accessed": False},
    }
    if not passed:
        raise RuntimeError(f"DeepSpeed parity failed: {payload}")
    data = (json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("ascii")
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)


if __name__ == "__main__":
    main()
