#!/usr/bin/env python3
"""Audit the static-194 efficiency gate with the corrected validation anchor."""

from __future__ import annotations

import argparse
from pathlib import Path

import audit_antibody_four_gpu_efficiency_gate as parent
import freeze_antibody_four_gpu_efficiency_gate_v3 as freezer


FORMAT_VERSION = "antibody_four_gpu_efficiency_result_v3"


def audit_gate(*, run_root: Path, protocol_path: Path, bundle_root: Path) -> dict:
    return parent.audit_gate(
        run_root=run_root,
        protocol_path=protocol_path,
        bundle_root=bundle_root,
        protocol_freezer=freezer,
        result_format_version=FORMAT_VERSION,
        auditor_path=Path(__file__),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("Audit output must be create-only")
    parent._publish(
        args.output,
        audit_gate(
            run_root=args.run_root,
            protocol_path=args.protocol,
            bundle_root=args.bundle_root,
        ),
    )


if __name__ == "__main__":
    main()
