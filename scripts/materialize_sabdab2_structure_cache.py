#!/usr/bin/env python3
"""Materialize authenticated SAbDab2 N/CA/C tensors on a locked split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from interaction_first.sabdab2_structure import (  # noqa: E402
    materialize_structure_cache,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--split-sha256", required=True)
    parser.add_argument("--split-verification", type=Path, required=True)
    parser.add_argument("--split-verification-sha256", required=True)
    parser.add_argument("--canonical-manifest", type=Path, required=True)
    parser.add_argument("--canonical-manifest-sha256", required=True)
    parser.add_argument("--prepared-instances", type=Path, required=True)
    parser.add_argument("--prepared-instances-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-instances", type=int)
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    receipt = materialize_structure_cache(
        archive_path=arguments.archive,
        archive_sha256=arguments.archive_sha256,
        split_path=arguments.split,
        split_sha256=arguments.split_sha256,
        split_verification_path=arguments.split_verification,
        split_verification_sha256=arguments.split_verification_sha256,
        canonical_manifest_path=arguments.canonical_manifest,
        canonical_manifest_sha256=arguments.canonical_manifest_sha256,
        prepared_instances_path=arguments.prepared_instances,
        prepared_instances_sha256=arguments.prepared_instances_sha256,
        output_root=arguments.output_root,
        max_instances=arguments.max_instances,
    )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
