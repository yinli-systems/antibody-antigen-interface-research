#!/usr/bin/env python3
"""Audit exact four-GPU recovery for a confirmed real-OAS candidate."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import audit_antibody_production_batch_efficiency as artifacts
import audit_antibody_production_loader_recovery as recovery
import audit_antibody_production_recovery_gate_8gpu as promoted


FORMAT_VERSION = "antibody_production_loader_recovery_audit_4gpu_v1"
WORLD_SIZE = 4


def _load_run(root: Path, role: str) -> dict[str, Any]:
    receipt, receipt_file_sha256 = recovery._load(root / "receipt.json")
    state, state_file_sha256 = recovery._load(root / "trainer_state.json")
    rank_hosts = receipt.get("corpus_open", {}).get("rank_hosts", [])
    if (
        receipt.get("format_version") != promoted.RUN_FORMAT
        or receipt.get("run_role") != role
        or receipt.get("test_access") != promoted.TEST_ACCESS
        or receipt.get("authorization") != promoted.AUTHORIZATION
        or receipt.get("world_size") != WORLD_SIZE
        or state.get("world_size") != WORLD_SIZE
        or receipt.get("corpus_open", {}).get("node_count") != 1
        or len(rank_hosts) != WORLD_SIZE
        or len(set(rank_hosts)) != 1
    ):
        raise recovery.RecoveryAuditError(
            f"{role} four-GPU identity differs"
        )
    return {
        "receipt": receipt,
        "receipt_file_sha256": receipt_file_sha256,
        "root": str(root),
        "state": state,
        "state_file_sha256": state_file_sha256,
    }


def audit_recovery_4gpu(
    *,
    gate_protocol_path: Path,
    continuous_root: Path,
    stage1_root: Path,
    resumed_root: Path,
    confirmation_protocol_path: Path,
    confirmation_audit_path: Path,
    artifact_root: Path,
) -> dict[str, Any]:
    """Audit source promotion, exact restart, and sealed four-GPU artifacts."""
    protocol, protocol_sha256 = recovery._load(gate_protocol_path)
    if (
        protocol.get("format_version") != promoted.GATE_FORMAT
        or protocol.get("authorization") != promoted.AUTHORIZATION
        or protocol.get("test_access") != promoted.TEST_ACCESS
        or protocol.get("execution", {}).get("world_size") != WORLD_SIZE
        or protocol.get("execution", {}).get("hardware")
        != {
            "gpu_count": WORLD_SIZE,
            "gpu_model": "NVIDIA GeForce RTX 5090",
            "node_count": 1,
        }
        or protocol.get("auditor_sha256") != artifacts._sha256(Path(__file__))
    ):
        raise recovery.RecoveryAuditError("Four-GPU gate protocol differs")
    promotion = promoted._validate_confirmation(
        protocol=protocol,
        confirmation_protocol_path=confirmation_protocol_path,
        confirmation_audit_path=confirmation_audit_path,
    )
    original_loader = recovery._load_run
    original_file = recovery.__file__
    recovery._load_run = _load_run
    recovery.__file__ = str(Path(__file__))
    try:
        result = recovery.audit_recovery(
            gate_protocol_path=gate_protocol_path,
            continuous_root=continuous_root,
            stage1_root=stage1_root,
            resumed_root=resumed_root,
        )
    finally:
        recovery._load_run = original_loader
        recovery.__file__ = original_file
    result["candidate_sidecars"] = {
        role: promoted._candidate_sidecar(
            root,
            protocol=protocol,
            protocol_sha256=protocol_sha256,
        )
        for role, root in {
            "continuous": continuous_root,
            "stage1": stage1_root,
            "resumed": resumed_root,
        }.items()
    }
    artifact_audit = artifacts._verify_artifacts(artifact_root)
    if not artifact_audit["manifest_verified"] or artifact_audit["writable_paths"]:
        raise recovery.RecoveryAuditError("Recovery artifacts are not immutable")
    result.update(
        {
            "artifact_audit": artifact_audit,
            "authorization": {
                "production_pretraining": False,
                "scale_16_gpu": False,
                "scale_32_gpu": False,
                "scientific_claim": False,
            },
            "claim_boundary": (
                "Exact four-GPU real-OAS loader and checkpoint recovery for "
                "one confirmed execution candidate only; no production "
                "training, biological, downstream, official-test, or SOTA claim"
            ),
            "format_version": FORMAT_VERSION,
            "promotion_evidence": promotion,
            "world_size": WORLD_SIZE,
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-protocol", type=Path, required=True)
    parser.add_argument("--continuous", type=Path, required=True)
    parser.add_argument("--stage1", type=Path, required=True)
    parser.add_argument("--resumed", type=Path, required=True)
    parser.add_argument("--confirmation-protocol", type=Path, required=True)
    parser.add_argument("--confirmation-audit", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_recovery_4gpu(
        gate_protocol_path=args.gate_protocol,
        continuous_root=args.continuous,
        stage1_root=args.stage1,
        resumed_root=args.resumed,
        confirmation_protocol_path=args.confirmation_protocol,
        confirmation_audit_path=args.confirmation_audit,
        artifact_root=args.artifact_root,
    )
    recovery._publish(args.output, result)


if __name__ == "__main__":
    main()
