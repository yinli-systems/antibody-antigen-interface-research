#!/usr/bin/env python3
"""Build or verify an authenticated FrameContact training source tree."""

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
    canonical_json_bytes,
    sha256_file,
)
from interaction_first.source_manifest import (  # noqa: E402
    build_source_manifest,
    verify_source_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--source-root", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument(
        "--include-directory",
        action="append",
        help=(
            "Directory relative to source root to authenticate; repeat for multiple "
            "directories. Defaults to scripts, src, and protocols."
        ),
    )
    verify = subparsers.add_parser("verify")
    verify.add_argument("--source-root", type=Path, required=True)
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--manifest-sha256", required=True)
    arguments = parser.parse_args()
    if arguments.command == "build":
        if arguments.output.exists() or arguments.output.is_symlink():
            raise FileExistsError(f"Refusing to overwrite source manifest: {arguments.output}")
        build_keywords = {}
        if arguments.include_directory:
            if len(set(arguments.include_directory)) != len(
                arguments.include_directory
            ):
                raise ValueError("Included source directories must be unique")
            build_keywords["directories"] = tuple(arguments.include_directory)
        payload = build_source_manifest(
            arguments.source_root,
            claim_boundary=(
                "Training source inventory and digests only. This contains no model "
                "prediction, validation metric, test access, or hardware-performance claim."
            ),
            **build_keywords,
        )
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_bytes(canonical_json_bytes(payload))
        print(json.dumps({"manifest_sha256": sha256_file(arguments.output), **payload}, indent=2))
    else:
        payload = verify_source_manifest(
            arguments.source_root,
            arguments.manifest,
            arguments.manifest_sha256,
        )
        print(json.dumps({"valid": True, "tree_sha256": payload["tree_sha256"]}))


if __name__ == "__main__":
    main()
