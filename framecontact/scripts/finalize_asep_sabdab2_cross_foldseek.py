#!/usr/bin/env python3
"""Finalize an exhaustive AsEP-test x SAbDab2-dev Foldseek rectangle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from interaction_first.asep_cross_foldseek import (  # noqa: E402
    finalize_asep_sabdab2_cross_foldseek,
)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--preparation-root", type=Path, required=True)
parser.add_argument("--preparation-receipt-sha256", required=True)
parser.add_argument("--raw-alignments", type=Path, required=True)
parser.add_argument("--raw-alignments-sha256", required=True)
parser.add_argument("--command", type=Path, required=True)
parser.add_argument("--foldseek-version", required=True)
parser.add_argument("--normalized-output", type=Path, required=True)
parser.add_argument("--receipt-output", type=Path, required=True)
arguments = parser.parse_args()
print(json.dumps(finalize_asep_sabdab2_cross_foldseek(
    preparation_root=arguments.preparation_root,
    preparation_receipt_sha256=arguments.preparation_receipt_sha256,
    raw_alignments=arguments.raw_alignments,
    raw_alignments_sha256=arguments.raw_alignments_sha256,
    command_path=arguments.command,
    foldseek_version=arguments.foldseek_version,
    normalized_output=arguments.normalized_output,
    receipt_output=arguments.receipt_output,
), sort_keys=True))
