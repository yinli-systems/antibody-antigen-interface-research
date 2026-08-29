#!/usr/bin/env python3
"""Authenticate, merge, and normalize all Foldseek shard outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from interaction_first.foldseek_shards import (  # noqa: E402
    merge_and_normalize_foldseek_shards,
)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--shard-output-root", type=Path, required=True)
parser.add_argument("--plan", type=Path, required=True)
parser.add_argument("--plan-sha256", required=True)
parser.add_argument("--export-manifest", type=Path, required=True)
parser.add_argument("--export-manifest-sha256", required=True)
parser.add_argument("--raw-output", type=Path, required=True)
parser.add_argument("--normalized-output", type=Path, required=True)
parser.add_argument("--receipt", type=Path, required=True)
parser.add_argument("--foldseek-version", required=True)
arguments = parser.parse_args()
print(json.dumps(merge_and_normalize_foldseek_shards(
    shard_output_root=arguments.shard_output_root,
    plan_path=arguments.plan,
    plan_sha256=arguments.plan_sha256,
    export_manifest_path=arguments.export_manifest,
    export_manifest_sha256=arguments.export_manifest_sha256,
    raw_output_path=arguments.raw_output,
    normalized_output_path=arguments.normalized_output,
    receipt_path=arguments.receipt,
    foldseek_version=arguments.foldseek_version,
), sort_keys=True))
