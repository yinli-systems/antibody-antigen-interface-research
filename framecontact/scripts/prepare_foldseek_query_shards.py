#!/usr/bin/env python3
"""Create a deterministic residue-balanced Foldseek query shard plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from interaction_first.foldseek_shards import prepare_foldseek_query_shards  # noqa: E402


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--export-root", type=Path, required=True)
parser.add_argument("--manifest-sha256", required=True)
parser.add_argument("--output-root", type=Path, required=True)
parser.add_argument("--shard-count", type=int, required=True)
arguments = parser.parse_args()
result = prepare_foldseek_query_shards(
    export_root=arguments.export_root,
    manifest_sha256=arguments.manifest_sha256,
    output_root=arguments.output_root,
    shard_count=arguments.shard_count,
)
print(json.dumps({
    "output_root": result["output_root"],
    "plan_sha256": result["plan_sha256"],
    "structure_count": result["structure_count"],
    "shard_count": result["shard_count"],
    "query_counts": [value["query_count"] for value in result["shards"]],
    "query_residue_counts": [
        value["query_residue_count"] for value in result["shards"]
    ],
}, sort_keys=True))
