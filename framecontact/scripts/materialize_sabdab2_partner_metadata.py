#!/usr/bin/env python3
"""Materialize label-free SAbDab2 partner matching metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from interaction_first.sabdab2_partner_metadata import (
    bind_partner_metadata_to_split,
    materialize_partner_metadata_base,
    project_split_bound_partner_metadata_to_base,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    base = subparsers.add_parser("base")
    base.add_argument("--archive", type=Path, required=True)
    base.add_argument("--archive-sha256", required=True)
    base.add_argument("--metadata", type=Path, required=True)
    base.add_argument("--metadata-sha256", required=True)
    base.add_argument("--prepared", type=Path, required=True)
    base.add_argument("--prepared-sha256", required=True)
    base.add_argument("--output-root", type=Path, required=True)
    base.add_argument("--workers", type=int, default=1)
    bind = subparsers.add_parser("bind")
    bind.add_argument("--base", type=Path, required=True)
    bind.add_argument("--base-sha256", required=True)
    bind.add_argument("--split", type=Path, required=True)
    bind.add_argument("--split-sha256", required=True)
    bind.add_argument("--output-root", type=Path, required=True)
    project = subparsers.add_parser("project")
    project.add_argument("--bound", type=Path, required=True)
    project.add_argument("--bound-sha256", required=True)
    project.add_argument("--bound-receipt", type=Path, required=True)
    project.add_argument("--bound-receipt-sha256", required=True)
    project.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.command == "base":
        payload = materialize_partner_metadata_base(
            archive_path=arguments.archive,
            archive_sha256=arguments.archive_sha256,
            metadata_path=arguments.metadata,
            metadata_sha256=arguments.metadata_sha256,
            prepared_path=arguments.prepared,
            prepared_sha256=arguments.prepared_sha256,
            output_root=arguments.output_root,
            workers=arguments.workers,
        )
    elif arguments.command == "bind":
        payload = bind_partner_metadata_to_split(
            base_path=arguments.base,
            base_sha256=arguments.base_sha256,
            split_path=arguments.split,
            split_sha256=arguments.split_sha256,
            output_root=arguments.output_root,
        )
    else:
        payload = project_split_bound_partner_metadata_to_base(
            bound_path=arguments.bound,
            bound_sha256=arguments.bound_sha256,
            bound_receipt_path=arguments.bound_receipt,
            bound_receipt_sha256=arguments.bound_receipt_sha256,
            output_root=arguments.output_root,
        )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
