#!/usr/bin/env python3
"""Materialize one split-independent SAbDab2 N/CA/C cohort cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from interaction_first.sabdab2_structure import materialize_structure_cohort_cache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--metadata-sha256", required=True)
    parser.add_argument("--canonical-manifest", type=Path, required=True)
    parser.add_argument("--canonical-manifest-sha256", required=True)
    parser.add_argument("--prepared-instances", type=Path, required=True)
    parser.add_argument("--prepared-instances-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-instances", type=int)
    arguments = parser.parse_args()
    receipt = materialize_structure_cohort_cache(
        archive_path=arguments.archive,
        archive_sha256=arguments.archive_sha256,
        metadata_path=arguments.metadata,
        metadata_sha256=arguments.metadata_sha256,
        canonical_manifest_path=arguments.canonical_manifest,
        canonical_manifest_sha256=arguments.canonical_manifest_sha256,
        prepared_instances_path=arguments.prepared_instances,
        prepared_instances_sha256=arguments.prepared_instances_sha256,
        output_root=arguments.output_root,
        max_instances=arguments.max_instances,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
