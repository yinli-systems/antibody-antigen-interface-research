#!/usr/bin/env python3
"""Audit exact recovery for the development-only eight-GPU OAS pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


TEST_ACCESS = {"downstream_labels_used": False, "sabdab2_heldout_accessed": False}
AUTHORIZATION = {
    "loader_recovery_gate": True,
    "production_pretraining": False,
    "scale_out": False,
    "scientific_claim": False,
}


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("ascii")


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--continuous", type=Path, required=True)
    parser.add_argument("--stage1", type=Path, required=True)
    parser.add_argument("--resumed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    protocol = _load(args.protocol)
    if protocol.get("execution", {}).get("world_size") != 8:
        raise ValueError("protocol is not eight-GPU")
    receipts = {}
    states = {}
    for role, root in (("continuous", args.continuous), ("stage1", args.stage1), ("resumed", args.resumed)):
        receipt = _load(root / "receipt.json")
        state = _load(root / "trainer_state.json")
        if receipt.get("format_version") != "antibody_production_loader_recovery_run_v1":
            raise ValueError(f"{role} receipt format differs")
        if receipt.get("authorization") != AUTHORIZATION or receipt.get("test_access") != TEST_ACCESS:
            raise ValueError(f"{role} authorization boundary differs")
        if receipt.get("run_role") != role or receipt.get("world_size") != 8 or state.get("world_size") != 8:
            raise ValueError(f"{role} world-size identity differs")
        hosts = receipt.get("corpus_open", {}).get("rank_hosts", [])
        if len(hosts) != 8 or len(set(hosts)) != 1:
            raise ValueError(f"{role} is not a single-node eight-rank run")
        receipts[role] = receipt
        states[role] = state
    if receipts["continuous"]["final_model_sha256"] != receipts["resumed"]["final_model_sha256"]:
        raise ValueError("model recovery mismatch")
    if receipts["continuous"]["final_optimizer_sha256"] != receipts["resumed"]["final_optimizer_sha256"]:
        raise ValueError("optimizer recovery mismatch")
    if states["continuous"] != states["resumed"]:
        raise ValueError("trainer-state recovery mismatch")
    payload = {
        "format_version": "antibody_eight_gpu_real_oas_recovery_audit_v1",
        "authorization": AUTHORIZATION,
        "test_access": TEST_ACCESS,
        "claim_boundary": "Eight-GPU real-OAS exact recovery pilot only; no production or scientific claim",
        "protocol_sha256": hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
        "world_size": 8,
        "roles": {
            role: {
                "final_model_sha256": receipts[role]["final_model_sha256"],
                "final_optimizer_sha256": receipts[role]["final_optimizer_sha256"],
                "consumed_residue_tokens": receipts[role]["trace"][-1]["consumed_residue_tokens"],
            }
            for role in receipts
        },
        "exact_recovery": {"model": True, "optimizer": True, "trainer_state": True},
    }
    payload["receipt_sha256"] = hashlib.sha256(_canonical(payload)).hexdigest()
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(_canonical(payload))
        handle.flush()
        os.fsync(handle.fileno())


if __name__ == "__main__":
    main()
