#!/usr/bin/env python3
"""Audit paired whole-chain and CDR-focused ESM-C adaptation canaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--whole-chain", type=Path, required=True)
    parser.add_argument("--cdr", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    runs = {
        "whole_chain_control": json.loads(args.whole_chain.read_text()),
        "cdr_focused": json.loads(args.cdr.read_text()),
    }
    gate = protocol["frozen_gate"]
    per_arm = {}
    for arm, run in runs.items():
        losses = run.get("losses", [])
        gradients = run.get("grad_norms", [])
        expected_range = gate["whole_chain_masked_fraction_range"] if arm == "whole_chain_control" else gate["cdr_masked_fraction_range"]
        checks = {
            "run_status": run.get("status") == "PASS_FINITE_FOUR_GPU_ESMC_ADAPTATION_CANARY",
            "arm": run.get("arm") == arm,
            "authority_sha256": run.get("authority_sha256") == sha256_file(args.protocol),
            "world_size": run.get("world_size") == gate["world_size"],
            "updates": run.get("updates") == protocol["canary"]["updates"] == len(losses) == len(gradients),
            "finite_positive_losses": bool(losses) and all(math.isfinite(value) and value > 0 for value in losses),
            "finite_positive_gradients": bool(gradients) and all(math.isfinite(value) and value > 0 for value in gradients),
            "nonfinite": run.get("nonfinite") == 0,
            "masked_fraction": expected_range[0] <= run.get("masked_fraction_of_eligible", -1) <= expected_range[1],
            "peak_memory": run.get("peak_allocated_fraction", 2) <= gate["maximum_peak_allocated_fraction"],
            "state_changed": run.get("initial_state_probe_sha256") != run.get("final_state_probe_sha256"),
            "no_forbidden_access": run.get("audit", {}).get("abcdr_validation_or_test_rows_used") is False and run.get("audit", {}).get("sabdab2_test_access") is False,
        }
        per_arm[arm] = {"result_sha256": sha256_file(args.whole_chain if arm == "whole_chain_control" else args.cdr), "checks": checks}
    cross_checks = {
        "identical_initial_state": runs["whole_chain_control"].get("initial_state_probe_sha256") == runs["cdr_focused"].get("initial_state_probe_sha256"),
        "same_base_checkpoint": runs["whole_chain_control"].get("base_checkpoint_sha256") == runs["cdr_focused"].get("base_checkpoint_sha256") == protocol["model"]["weight_sha256"],
        "same_span_artifact": runs["whole_chain_control"].get("span_sha256") == runs["cdr_focused"].get("span_sha256") == protocol["data"]["span_sha256"],
    }
    passed = all(all(value["checks"].values()) for value in per_arm.values()) and all(cross_checks.values())
    result = {
        "format_version": "paired_esmc_adaptation_canary_analysis_v1",
        "status": "PASS_WHOLE_CHAIN_AND_CDR_FOUR_GPU_CANARIES" if passed else "FAIL_CLOSED_WHOLE_CHAIN_AND_CDR_FOUR_GPU_CANARIES",
        "protocol_sha256": sha256_file(args.protocol),
        "per_arm": per_arm,
        "cross_arm_checks": cross_checks,
        "authorization": {"production_protocol_freeze_authorized": passed, "production_training_authorized": False, "downstream_test_authorized": False},
        "claim_boundary": {"system_correctness_only": True, "performance_gain_claim": False, "task_labels_read": False, "sabdab2_test_access": False},
    }
    atomic_write(args.output, result)
    receipt = {
        "format_version": "paired_esmc_adaptation_canary_analysis_receipt_v1",
        "status": result["status"],
        "analysis_sha256": sha256_file(args.output),
        "production_protocol_freeze_authorized": passed,
        "production_training_authorized": False,
        "sabdab2_test_access": False,
    }
    atomic_write(args.receipt, receipt)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
