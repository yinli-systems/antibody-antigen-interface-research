#!/usr/bin/env python3
"""Export authenticated SAbDab2 antigen-only backbones for Foldseek."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from interaction_first.sabdab2_antigen_export import export_antigen_structures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--metadata-sha256", required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--prepared-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args()
    result = export_antigen_structures(
        archive_path=arguments.archive,
        archive_sha256=arguments.archive_sha256,
        metadata_path=arguments.metadata,
        metadata_sha256=arguments.metadata_sha256,
        prepared_path=arguments.prepared,
        prepared_sha256=arguments.prepared_sha256,
        output_root=arguments.output_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
