#!/usr/bin/env python3
"""Run authenticated US-align checks for Foldseek threshold-boundary pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from interaction_first.usalign_boundary import (  # noqa: E402
    run_usalign_boundary_spot_checks,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structural-audit", type=Path, required=True)
    parser.add_argument("--structural-audit-sha256", required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--split-manifest-sha256", required=True)
    parser.add_argument("--export-manifest", type=Path, required=True)
    parser.add_argument("--export-manifest-sha256", required=True)
    parser.add_argument("--structure-root", type=Path, required=True)
    parser.add_argument("--usalign-binary", type=Path, required=True)
    parser.add_argument("--usalign-binary-sha256", required=True)
    parser.add_argument("--usalign-lock", type=Path, required=True)
    parser.add_argument("--usalign-lock-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    arguments = parser.parse_args()
    receipt = run_usalign_boundary_spot_checks(
        structural_audit_path=arguments.structural_audit,
        structural_audit_sha256=arguments.structural_audit_sha256,
        split_manifest_path=arguments.split_manifest,
        split_manifest_sha256=arguments.split_manifest_sha256,
        export_manifest_path=arguments.export_manifest,
        export_manifest_sha256=arguments.export_manifest_sha256,
        structure_root=arguments.structure_root,
        usalign_binary_path=arguments.usalign_binary,
        usalign_binary_sha256=arguments.usalign_binary_sha256,
        usalign_lock_path=arguments.usalign_lock,
        usalign_lock_sha256=arguments.usalign_lock_sha256,
        output_root=arguments.output_root,
        workers=arguments.workers,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if not receipt["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
