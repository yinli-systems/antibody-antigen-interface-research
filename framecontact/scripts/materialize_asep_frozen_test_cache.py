#!/usr/bin/env python3
"""Open and materialize the AsEP official test graph exactly once."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from interaction_first.asep_frozen_test import (  # noqa: E402
    materialize_asep_frozen_test_cache,
)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--authority-root", type=Path, required=True)
parser.add_argument("--inner-audit", type=Path, required=True)
parser.add_argument("--inner-audit-sha256", required=True)
parser.add_argument("--split-root", type=Path, required=True)
parser.add_argument("--split-receipt-sha256", required=True)
parser.add_argument("--split-method", choices=("epitope_group", "epitope_ratio"), required=True)
parser.add_argument("--opening-receipt", type=Path, required=True)
parser.add_argument("--opening-receipt-sha256", required=True)
parser.add_argument("--output-root", type=Path, required=True)
arguments = parser.parse_args()
print(json.dumps(materialize_asep_frozen_test_cache(
    authority_root=arguments.authority_root,
    inner_audit_path=arguments.inner_audit,
    inner_audit_sha256=arguments.inner_audit_sha256,
    split_root=arguments.split_root,
    split_receipt_sha256=arguments.split_receipt_sha256,
    split_method=arguments.split_method,
    opening_receipt_path=arguments.opening_receipt,
    opening_receipt_sha256=arguments.opening_receipt_sha256,
    output_root=arguments.output_root,
), sort_keys=True))
