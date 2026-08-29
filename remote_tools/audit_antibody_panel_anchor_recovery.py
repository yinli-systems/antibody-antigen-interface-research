#!/usr/bin/env python3
"""Audit a completed segment through the frozen validation-anchor correction."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "recovery"))

import audit_antibody_four_gpu_pretraining_segment as base  # noqa: E402
import freeze_antibody_panel_anchor_recovery as freezer  # noqa: E402


RECOVERY_FORMAT = "antibody_panel_anchor_segment_recovery_v1"


class RecoveryAuditError(RuntimeError):
    """Report a failed recovery-closure invariant."""


def validate_failed_segment(segment_root: Path, failed_job_id: str) -> dict[str, Any]:
    environment = (segment_root / "environment.txt").read_text(encoding="utf-8")
    if f"job_id={failed_job_id}\n" not in environment:
        raise RecoveryAuditError("Segment job identity differs")
    evaluation_log = segment_root / "evaluation.log"
    text = evaluation_log.read_text(encoding="utf-8", errors="replace")
    if text.count("RuntimeError: Validation panel anchor differs") < 4:
        raise RecoveryAuditError("Segment did not fail only at the known panel anchor")
    if (segment_root / "evaluation.json").exists():
        raise RecoveryAuditError("Failed segment unexpectedly contains evaluation JSON")
    artifact_audit = freezer.verify_artifact_tree(segment_root)
    return {
        "artifact_audit": artifact_audit,
        "evaluation_log_sha256": freezer.sha256_file(evaluation_log),
        "failed_job_id": failed_job_id,
    }


def validate_closure(segment_root: Path, closure_root: Path) -> dict[str, Any]:
    allowed_extra = {"evaluation.json", "recovery_evaluation.log"}
    original_files = {
        path.relative_to(segment_root): path
        for path in segment_root.rglob("*")
        if path.is_file() and path.name != "artifact_sha256.txt"
    }
    closure_files = {
        path.relative_to(closure_root): path
        for path in closure_root.rglob("*")
        if path.is_file() and path.name != "artifact_sha256.txt"
    }
    for relative, original in original_files.items():
        copied = closure_files.get(relative)
        if copied is None or freezer.sha256_file(copied) != freezer.sha256_file(original):
            raise RecoveryAuditError(f"Closure changed original artifact: {relative}")
    extras = set(closure_files) - set(original_files)
    if {path.as_posix() for path in extras} != allowed_extra:
        raise RecoveryAuditError("Closure contains unexpected recovery artifacts")
    if any(path.is_symlink() for path in closure_root.rglob("*")):
        raise RecoveryAuditError("Closure contains a symbolic link")
    return freezer.verify_artifact_tree(closure_root)


def audit(
    *,
    recovery_protocol_path: Path,
    recovery_receipt_path: Path,
    gate_protocol_path: Path,
    segment_root: Path,
    closure_root: Path,
    segment_index: int,
    failed_job_id: str,
    previous_segment_root: Path | None,
    previous_segment_audit: Path | None,
) -> dict[str, Any]:
    protocol = freezer.verify(
        protocol_path=recovery_protocol_path,
        receipt_path=recovery_receipt_path,
        source_root=ROOT,
    )
    gate, gate_sha256 = base._load(gate_protocol_path)
    evidence = protocol["evidence"]
    if (
        gate_sha256 != evidence["gate_protocol_sha256"]
        or gate.get("validation_panel", {}).get("indices_sha256")
        != evidence["stale_panel_anchor"]
    ):
        raise RecoveryAuditError("Campaign gate is not the frozen stale-anchor gate")
    failed = validate_failed_segment(segment_root, failed_job_id)
    closure_audit = validate_closure(segment_root, closure_root)
    corrected_protocol = copy.deepcopy(gate)
    corrected_protocol["validation_panel"]["indices_sha256"] = evidence[
        "corrected_panel_anchor"
    ]
    original_validator = base._validate_evaluation

    def corrected_validator(
        value: Mapping[str, Any],
        *,
        protocol: Mapping[str, Any],
        gate_sha256: str,
        run_receipt_sha256: str,
        run_state_sha256: str,
        final_model_sha256: str,
    ) -> dict[str, object]:
        del protocol
        return original_validator(
            value,
            protocol=corrected_protocol,
            gate_sha256=gate_sha256,
            run_receipt_sha256=run_receipt_sha256,
            run_state_sha256=run_state_sha256,
            final_model_sha256=final_model_sha256,
        )

    base._validate_evaluation = corrected_validator
    try:
        result = base.audit_segment(
            gate_protocol_path=gate_protocol_path,
            segment_root=closure_root,
            segment_index=segment_index,
            previous_segment_root=previous_segment_root,
            previous_audit_path=previous_segment_audit,
        )
    finally:
        base._validate_evaluation = original_validator
    result["inputs"]["auditor_sha256"] = freezer.sha256_file(Path(__file__))
    result["inputs"]["segment_root"] = str(segment_root.resolve())
    result["claim_boundary"] = (
        "Immutable four-GPU OAS MLM segment and corrected frozen validation-panel "
        "checkpoint audit only; the correction changes no training, checkpoint, "
        "data, masking, test, downstream, or scientific-claim authority."
    )
    result["recovery"] = {
        "closure_artifact_audit": closure_audit,
        "closure_root": str(closure_root.resolve()),
        "corrected_panel_anchor": evidence["corrected_panel_anchor"],
        "failed_segment": failed,
        "format_version": RECOVERY_FORMAT,
        "original_auditor_sha256": freezer.sha256_file(
            ROOT / "scripts/audit_antibody_four_gpu_pretraining_segment.py"
        ),
        "recovery_protocol_file_sha256": freezer.sha256_file(
            recovery_protocol_path
        ),
        "stale_panel_anchor": evidence["stale_panel_anchor"],
        "test_access": freezer.TEST_ACCESS,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery-protocol", type=Path, required=True)
    parser.add_argument("--recovery-receipt", type=Path, required=True)
    parser.add_argument("--gate-protocol", type=Path, required=True)
    parser.add_argument("--segment-root", type=Path, required=True)
    parser.add_argument("--closure-root", type=Path, required=True)
    parser.add_argument("--segment-index", type=int, required=True)
    parser.add_argument("--failed-job-id", required=True)
    parser.add_argument("--previous-segment-root", type=Path)
    parser.add_argument("--previous-segment-audit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = audit(
        recovery_protocol_path=args.recovery_protocol,
        recovery_receipt_path=args.recovery_receipt,
        gate_protocol_path=args.gate_protocol,
        segment_root=args.segment_root,
        closure_root=args.closure_root,
        segment_index=args.segment_index,
        failed_job_id=args.failed_job_id,
        previous_segment_root=args.previous_segment_root,
        previous_segment_audit=args.previous_segment_audit,
    )
    base._publish(args.output, value)
    print(json.dumps(value["authorization"], sort_keys=True))


if __name__ == "__main__":
    main()
