#!/usr/bin/env python3
"""Verify a Foldseek shard rectangle and freeze its receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from interaction_first.foldseek_shards import finalize_foldseek_shard  # noqa: E402


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--plan", type=Path, required=True)
parser.add_argument("--plan-sha256", required=True)
parser.add_argument("--shard-index", type=int, required=True)
parser.add_argument("--raw", type=Path, required=True)
parser.add_argument("--raw-sha256", required=True)
parser.add_argument("--command", type=Path, required=True)
parser.add_argument("--receipt", type=Path, required=True)
parser.add_argument("--foldseek-version", required=True)
arguments = parser.parse_args()
print(json.dumps(finalize_foldseek_shard(
    plan_path=arguments.plan,
    plan_sha256=arguments.plan_sha256,
    shard_index=arguments.shard_index,
    raw_path=arguments.raw,
    raw_sha256=arguments.raw_sha256,
    command_path=arguments.command,
    receipt_path=arguments.receipt,
    foldseek_version=arguments.foldseek_version,
), sort_keys=True))
