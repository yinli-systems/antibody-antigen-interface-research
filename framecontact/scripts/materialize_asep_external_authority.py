#!/usr/bin/env python3
"""Safely materialize the authenticated AsEP v1.2.0 external authority."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from interaction_first.asep_authority import (  # noqa: E402
    load_asep_dataset_lock,
    materialize_asep_authority,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--authority-lock", type=Path, required=True)
    parser.add_argument("--authority-lock-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--maximum-members", type=int, default=100_000)
    parser.add_argument(
        "--maximum-uncompressed-bytes",
        type=int,
        default=64 * 1024**3,
    )
    arguments = parser.parse_args()
    lock = load_asep_dataset_lock(
        arguments.authority_lock,
        arguments.authority_lock_sha256,
    )
    dataset = lock["dataset"]
    receipt = materialize_asep_authority(
        archive_path=arguments.archive,
        archive_sha256=dataset["archive_sha256"],
        archive_md5=dataset["publisher_md5"],
        expected_size_bytes=dataset["archive_size_bytes"],
        source_code_revision=lock["official_source"]["revision"],
        output_root=arguments.output_root,
        maximum_members=arguments.maximum_members,
        maximum_uncompressed_bytes=arguments.maximum_uncompressed_bytes,
        expected_member_paths=tuple(lock["expected_archive_members"]),
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
