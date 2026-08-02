#!/usr/bin/env python3
"""Independently audit an OAS corpus plan and emit a create-only receipt."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import antibody_pretraining.independent_corpus_audit as audit_module
from antibody_pretraining.independent_corpus_audit import (
    audit_corpus_plan,
    canonical_json_bytes,
    file_sha256,
    load_ascii_json,
)


def _write_create_only(path: Path, payload: bytes) -> None:
    """Write one immutable audit receipt and reject an existing target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink(missing_ok=True)
        finally:
            raise


def _parser() -> argparse.ArgumentParser:
    """Build the strict independent-audit command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-file-sha256", required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-file-sha256", required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--spec-file-sha256", required=True)
    parser.add_argument("--auditor-module-sha256", required=True)
    parser.add_argument("--auditor-source-sha256", required=True)
    parser.add_argument("--planner-module-sha256", required=True)
    parser.add_argument("--planner-source-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    """Verify byte anchors, rebuild membership, and write the receipt."""
    args = _parser().parse_args()
    observed_controller_sha256 = file_sha256(Path(__file__))
    if observed_controller_sha256 != args.auditor_source_sha256:
        raise ValueError("Auditor controller digest differs")
    observed_module_sha256 = file_sha256(Path(audit_module.__file__))
    if observed_module_sha256 != args.auditor_module_sha256:
        raise ValueError("Auditor module digest differs")
    manifest = load_ascii_json(args.manifest, args.manifest_file_sha256)
    plan = load_ascii_json(args.plan, args.plan_file_sha256)
    spec = load_ascii_json(args.spec, args.spec_file_sha256)
    receipt = audit_corpus_plan(
        manifest,
        plan,
        spec,
        manifest_file_sha256=args.manifest_file_sha256,
        plan_file_sha256=args.plan_file_sha256,
        spec_file_sha256=args.spec_file_sha256,
        auditor_module_sha256=observed_module_sha256,
        auditor_source_sha256=observed_controller_sha256,
        planner_module_sha256=args.planner_module_sha256,
        planner_source_sha256=args.planner_source_sha256,
    )
    payload = canonical_json_bytes(receipt)
    _write_create_only(args.output, payload)
    print(file_sha256(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
