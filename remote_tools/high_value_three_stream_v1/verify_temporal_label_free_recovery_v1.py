#!/usr/bin/env python3
"""Verify recovered label-free temporal evidence before the single opening."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base-receipt", type=Path, required=True)
    parser.add_argument("--recovery-receipt", type=Path, required=True)
    args = parser.parse_args()
    if sha256_file(args.plan) != args.plan_sha256:
        raise ValueError("recovery plan SHA-256 differs")
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    base_receipt = json.loads(args.base_receipt.read_text(encoding="utf-8"))
    recovery = json.loads(args.recovery_receipt.read_text(encoding="utf-8"))
    if (
        plan.get("format_version") != "framecontact_temporal_label_free_recovery_plan_v3"
        or plan.get("status")
        != "FROZEN_AFTER_THREE_LABEL_FREE_INTERFACE_FAILURES_BEFORE_RECOVERY"
        or plan.get("recovery", {}).get("verifier_script_sha256")
        != sha256_file(Path(__file__))
        or manifest.get("status") != "PASS_LABEL_FREE_DOUBLE_COLD_FREEZE"
        or manifest.get("audit", {}).get("task_labels_read") is not False
        or manifest.get("audit", {}).get("cross_chain_distances_computed") is not False
        or manifest.get("audit", {}).get("contact_labels_computed") is not False
        or base_receipt.get("manifest_sha256") != sha256_file(args.manifest)
        or base_receipt.get("task_labels_read") is not False
        or recovery.get("status") != "PASS_LABEL_FREE_INTERFACE_RECOVERY_BOUND"
        or recovery.get("recovery_plan_sha256") != args.plan_sha256
        or recovery.get("label_free_manifest_sha256") != sha256_file(args.manifest)
        or recovery.get("label_free_receipt_sha256") != sha256_file(args.base_receipt)
        or recovery.get("gate_thresholds_changed") is not False
        or recovery.get("task_labels_read") is not False
        or recovery.get("cross_chain_distances_computed") is not False
        or recovery.get("selection_performed") is not False
        or plan.get("cif_transfer_amendment", {}).get("status")
        != "PASS_CASE_SENSITIVE_TRANSFER_CORRECTION_FROZEN"
    ):
        raise ValueError("recovered label-free evidence differs")
    print(
        json.dumps(
            {
                "status": "PASS_RECOVERED_LABEL_FREE_EVIDENCE_VERIFIED_BEFORE_OPENING",
                "recovery_plan_sha256": args.plan_sha256,
                "label_free_manifest_sha256": sha256_file(args.manifest),
                "double_cold_instances": manifest["counts"]["double_cold_instances"],
                "double_cold_unique_pdbs": manifest["counts"]["double_cold_unique_pdbs"],
                "task_labels_read": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
