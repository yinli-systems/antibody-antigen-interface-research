#!/usr/bin/env python3
"""Freeze an authenticated train+validation-only pair-geometry selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from interaction_first.sabdab2_pair_geometry import (  # noqa: E402
    freeze_pair_geometry_development_selection,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--split-manifest-sha256", required=True)
    parser.add_argument("--structure-cache-root", type=Path, required=True)
    parser.add_argument("--structure-manifest-sha256", required=True)
    parser.add_argument("--contact-cache-root", type=Path, required=True)
    parser.add_argument("--contact-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = freeze_pair_geometry_development_selection(
        split_path=arguments.split_manifest,
        split_sha256=arguments.split_manifest_sha256,
        structure_cache_root=arguments.structure_cache_root,
        structure_manifest_sha256=arguments.structure_manifest_sha256,
        contact_cache_root=arguments.contact_cache_root,
        contact_manifest_sha256=arguments.contact_manifest_sha256,
        output_path=arguments.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
