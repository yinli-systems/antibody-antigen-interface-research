#!/usr/bin/env python3
"""Prepare authenticated AsEP-test and SAbDab2-dev Foldseek directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from interaction_first.asep_cross_foldseek import (  # noqa: E402
    prepare_asep_sabdab2_cross_foldseek,
)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--label-free-test-root", type=Path, required=True)
parser.add_argument("--label-free-test-receipt-sha256", required=True)
parser.add_argument("--sabdab2-split-manifest", type=Path, required=True)
parser.add_argument("--sabdab2-split-manifest-sha256", required=True)
parser.add_argument("--sabdab2-antigen-export-root", type=Path, required=True)
parser.add_argument("--sabdab2-antigen-export-receipt-sha256", required=True)
parser.add_argument("--output-root", type=Path, required=True)
arguments = parser.parse_args()
print(json.dumps(prepare_asep_sabdab2_cross_foldseek(
    label_free_test_root=arguments.label_free_test_root,
    label_free_test_receipt_sha256=arguments.label_free_test_receipt_sha256,
    sabdab2_split_manifest=arguments.sabdab2_split_manifest,
    sabdab2_split_manifest_sha256=arguments.sabdab2_split_manifest_sha256,
    sabdab2_antigen_export_root=arguments.sabdab2_antigen_export_root,
    sabdab2_antigen_export_receipt_sha256=arguments.sabdab2_antigen_export_receipt_sha256,
    output_root=arguments.output_root,
), sort_keys=True))
