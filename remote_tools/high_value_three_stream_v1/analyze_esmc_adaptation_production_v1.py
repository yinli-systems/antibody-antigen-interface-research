#!/usr/bin/env python3
"""Audit the fixed-epoch whole-chain versus CDR-focused ESM-C adaptation."""

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


def atomic_write(path: Path, value: object) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write((json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode())
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
    per_arm = {}
    selected_metrics = {}
    for arm, run in runs.items():
        checkpoint = run.get("fixed_checkpoint", {})
        checkpoint_path = Path(checkpoint.get("path", ""))
        receipts = run.get("epoch_receipts", [])
        checks = {
            "status": run.get("status") == "PASS_FIXED_EPOCH4_ADAPTATION_CHECKPOINT_VERIFIED",
            "arm": run.get("arm") == arm,
            "authority": run.get("authority_sha256") == sha256_file(args.protocol),
            "base": run.get("base_checkpoint_sha256") == protocol["model"]["weight_sha256"],
            "epochs": run.get("epochs") == protocol["training"]["epochs"] == len(receipts),
            "updates": run.get("global_updates") == protocol["training"]["total_updates"],
            "fixed_epoch": run.get("fixed_checkpoint_epoch") == protocol["training"]["fixed_checkpoint_epoch"],
            "checkpoint_exists": checkpoint_path.is_file(),
            "checkpoint_sha": checkpoint_path.is_file() and sha256_file(checkpoint_path) == checkpoint.get("sha256"),
            "state_changed": run.get("initial_state_probe_sha256") != run.get("final_state_probe_sha256"),
            "no_forbidden_access": run.get("audit", {}).get("abcdr_validation_or_test_rows_used") is False and run.get("audit", {}).get("sabdab2_test_access") is False,
        }
        selected = json.loads(Path(receipts[protocol["training"]["fixed_checkpoint_epoch"] - 1]["result_path"]).read_text())
        for evaluation in ("whole_chain_15pct", "cdr_50pct"):
            metrics = selected["validation"][evaluation]
            if not all(math.isfinite(metrics[key]) for key in ("loss", "accuracy", "masked_fraction_of_eligible")):
                checks[f"finite_{evaluation}"] = False
            else:
                checks[f"finite_{evaluation}"] = True
        selected_metrics[arm] = selected["validation"]
        per_arm[arm] = {"result_sha256": sha256_file(args.whole_chain if arm == "whole_chain_control" else args.cdr), "checks": checks, "fixed_checkpoint": checkpoint}
    integrity_passed = all(all(item["checks"].values()) for item in per_arm.values())
    deltas = {}
    for evaluation in ("whole_chain_15pct", "cdr_50pct"):
        deltas[evaluation] = {
            key: selected_metrics["cdr_focused"][evaluation][key] - selected_metrics["whole_chain_control"][evaluation][key]
            for key in ("loss", "accuracy")
        }
    result = {
        "format_version": "paired_esmc_adaptation_production_analysis_v1",
        "status": "PASS_FIXED_EPOCH4_CONTROLLED_ADAPTATION" if integrity_passed else "FAIL_CLOSED_FIXED_EPOCH4_CONTROLLED_ADAPTATION",
        "protocol_sha256": sha256_file(args.protocol),
        "per_arm": per_arm,
        "fixed_epoch_validation": selected_metrics,
        "cdr_minus_whole_chain": deltas,
        "authorization": {
            "development_embedding_materialization_authorized": integrity_passed,
            "downstream_test_authorized": False,
            "performance_gain_claim_authorized": False,
        },
        "claim_boundary": {
            "checkpoint_epoch_fixed_before_training": True,
            "mlm_validation_not_used_for_checkpoint_selection": True,
            "downstream_development_required_for_representation_claim": True,
            "task_labels_read": False,
            "sabdab2_test_access": False,
        },
    }
    atomic_write(args.output, result)
    receipt = {
        "format_version": "paired_esmc_adaptation_production_analysis_receipt_v1",
        "status": result["status"],
        "analysis_sha256": sha256_file(args.output),
        "development_embedding_materialization_authorized": integrity_passed,
        "downstream_test_authorized": False,
        "sabdab2_test_access": False,
    }
    atomic_write(args.receipt, receipt)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
