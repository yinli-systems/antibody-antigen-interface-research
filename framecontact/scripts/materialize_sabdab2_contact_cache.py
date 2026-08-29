#!/usr/bin/env python3
"""Authenticate SAbDab2 pairs.parquet and materialize FrameContact targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from interaction_first.sabdab2_contacts import materialize_contact_cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--pairs-sha256", required=True)
    parser.add_argument("--structure-cache-root", type=Path, required=True)
    parser.add_argument("--structure-manifest-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args()
    receipt = materialize_contact_cache(
        pairs_path=arguments.pairs,
        pairs_sha256=arguments.pairs_sha256,
        structure_cache_root=arguments.structure_cache_root,
        structure_manifest_sha256=arguments.structure_manifest_sha256,
        output_root=arguments.output_root,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
