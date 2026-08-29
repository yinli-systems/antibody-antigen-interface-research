#!/usr/bin/env python3
"""Materialize label-free AsEP test metadata after validation selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from interaction_first.asep_frozen_test import (  # noqa: E402
    materialize_asep_label_free_test_metadata,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority-root", type=Path, required=True)
    parser.add_argument("--inner-audit", type=Path, required=True)
    parser.add_argument("--inner-audit-sha256", required=True)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--split-receipt-sha256", required=True)
    parser.add_argument(
        "--split-method", choices=("epitope_group", "epitope_ratio"), required=True
    )
    parser.add_argument("--selection-root", type=Path, required=True)
    parser.add_argument("--selection-receipt-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args()
    result = materialize_asep_label_free_test_metadata(
        authority_root=arguments.authority_root,
        inner_audit_path=arguments.inner_audit,
        inner_audit_sha256=arguments.inner_audit_sha256,
        split_root=arguments.split_root,
        split_receipt_sha256=arguments.split_receipt_sha256,
        split_method=arguments.split_method,
        selection_root=arguments.selection_root,
        selection_receipt_sha256=arguments.selection_receipt_sha256,
        output_root=arguments.output_root,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
