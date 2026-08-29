#!/usr/bin/env python3
"""Audit every authenticated AsEP train/validation graph without opening test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from interaction_first.asep_external import audit_asep_development_graphs  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority-root", type=Path, required=True)
    parser.add_argument("--inner-audit", type=Path, required=True)
    parser.add_argument("--inner-audit-sha256", required=True)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--split-receipt-sha256", required=True)
    parser.add_argument(
        "--split-method",
        choices=("epitope_group", "epitope_ratio"),
        default="epitope_group",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args()
    result = audit_asep_development_graphs(
        authority_root=arguments.authority_root,
        inner_audit_path=arguments.inner_audit,
        inner_audit_sha256=arguments.inner_audit_sha256,
        split_root=arguments.split_root,
        split_receipt_sha256=arguments.split_receipt_sha256,
        split_method=arguments.split_method,
        output_root=arguments.output_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
