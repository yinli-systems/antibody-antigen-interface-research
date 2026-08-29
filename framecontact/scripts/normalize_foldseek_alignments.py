#!/usr/bin/env python3
"""Normalize hashed Foldseek identifiers and freeze a run receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from interaction_first.foldseek_normalize import normalize_foldseek_alignments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--raw-sha256", required=True)
    parser.add_argument("--export-manifest", type=Path, required=True)
    parser.add_argument("--export-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--foldseek-version", required=True)
    parser.add_argument("--command", required=True)
    arguments = parser.parse_args()
    result = normalize_foldseek_alignments(
        raw_path=arguments.raw,
        raw_sha256=arguments.raw_sha256,
        export_manifest_path=arguments.export_manifest,
        export_manifest_sha256=arguments.export_manifest_sha256,
        output_path=arguments.output,
        receipt_path=arguments.receipt,
        foldseek_version=arguments.foldseek_version,
        command=shlex.split(arguments.command),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
