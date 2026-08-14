#!/usr/bin/env python3
"""Freeze a four-GPU real-OAS exact-recovery readiness gate."""

from __future__ import annotations

import argparse
from pathlib import Path

import freeze_antibody_production_recovery_gate_8gpu as parent


WORLD_SIZE = 4
TARGET_RESIDUE_TOKENS = 131_072
BATCH_QUANTUM_SEQUENCES = 256


def freeze_gate(**arguments: object) -> dict[str, object]:
    """Reuse the confirmed execution candidate and freeze four-GPU identity."""
    protocol = parent.freeze_gate(**arguments)
    source_root = Path(arguments["source_root"])
    auditor = source_root / "scripts/audit_antibody_four_gpu_recovery_gate.py"
    protocol["auditor_sha256"] = parent._sha256(auditor)
    protocol["claim_boundary"] = (
        "Authorizes only a four-GPU real-OAS loader and exact-recovery gate "
        "for a confirmed execution candidate; no production training, "
        "biological, downstream, official-test, or SOTA claim"
    )
    execution = protocol["execution"]
    assert isinstance(execution, dict)
    execution["world_size"] = WORLD_SIZE
    execution["hardware"] = {
        "gpu_count": WORLD_SIZE,
        "gpu_model": "NVIDIA GeForce RTX 5090",
        "node_count": 1,
    }
    runs = protocol["runs"]
    assert isinstance(runs, dict)
    for run in runs.values():
        assert isinstance(run, dict)
        run["batch_quantum_sequences"] = BATCH_QUANTUM_SEQUENCES
        run["target_residue_tokens"] = TARGET_RESIDUE_TOKENS
    return protocol


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--manifest-file-sha256", required=True)
    parser.add_argument("--manifest-payload-sha256", required=True)
    parser.add_argument("--corpus-protocol-file-sha256", required=True)
    parser.add_argument("--independent-audit", type=Path, required=True)
    parser.add_argument("--independent-audit-file-sha256", required=True)
    parser.add_argument("--independent-audit-payload-sha256", required=True)
    parser.add_argument("--confirmation-protocol", type=Path, required=True)
    parser.add_argument("--confirmation-protocol-file-sha256", required=True)
    parser.add_argument("--confirmation-audit", type=Path, required=True)
    parser.add_argument("--confirmation-audit-file-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    values = vars(parser.parse_args())
    output = values.pop("output")
    values["independent_audit_path"] = values.pop("independent_audit")
    values["confirmation_protocol_path"] = values.pop("confirmation_protocol")
    values["confirmation_audit_path"] = values.pop("confirmation_audit")
    protocol = freeze_gate(**values)
    parent._publish(output, protocol)


if __name__ == "__main__":
    main()
