#!/usr/bin/env python3
"""Publish a score-free component-disjoint PAIRScale development holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import uuid


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from interaction_first.development_authority import canonical_json_bytes  # noqa: E402
from interaction_first.development_holdout import (  # noqa: E402
    REQUIRED_IMPLEMENTATION_SOURCES,
    build_component_holdout,
    build_holdout_receipt,
    parse_score_free_subset,
)


def _sha256_file(path: Path) -> str:
    """Hash one regular non-symlink file."""

    path = path.expanduser().absolute()
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected a regular non-symlink file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _publish_json(path: Path, payload: object) -> None:
    """Create one canonical JSON artifact and make it read-only."""

    with path.open("xb") as handle:
        handle.write(canonical_json_bytes(payload))
        handle.flush()
        os.fsync(handle.fileno())
    path.chmod(0o440)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--expected-subset-sha256", required=True)
    parser.add_argument("--expected-builder-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--holdout-pair-count", type=int, default=64)
    parser.add_argument("--seed", type=int, default=29)
    return parser.parse_args()


def main() -> None:
    """Verify all anchors and atomically publish one holdout assignment."""

    args = _parse_args()
    builder_sha256 = _sha256_file(Path(__file__))
    if builder_sha256 != args.expected_builder_sha256:
        raise ValueError("Builder source digest mismatch")
    subset_bytes = args.subset.read_bytes()
    subset = parse_score_free_subset(
        subset_bytes,
        expected_subset_sha256=args.expected_subset_sha256,
    )
    assignment = build_component_holdout(
        subset,
        holdout_pair_count=args.holdout_pair_count,
        seed=args.seed,
    )
    implementation_source_sha256 = {
        "scripts/build_pairscale_development_holdout.py": builder_sha256,
        "src/interaction_first/development_authority.py": _sha256_file(
            SOURCE_ROOT / "interaction_first" / "development_authority.py"
        ),
        "src/interaction_first/development_holdout.py": _sha256_file(
            SOURCE_ROOT / "interaction_first" / "development_holdout.py"
        ),
        "src/interaction_first/development_subset.py": _sha256_file(
            SOURCE_ROOT / "interaction_first" / "development_subset.py"
        ),
    }
    if set(implementation_source_sha256) != REQUIRED_IMPLEMENTATION_SOURCES:
        raise AssertionError("Implementation source inventory is incomplete")
    receipt = build_holdout_receipt(
        assignment,
        implementation_source_sha256=implementation_source_sha256,
    )

    output_root = args.output_root.expanduser().absolute()
    if output_root.exists() or output_root.is_symlink():
        raise ValueError("Output root already exists; publication is create-only")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = output_root.with_name(
        f".{output_root.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    staging.mkdir(mode=0o700)
    try:
        _publish_json(staging / "DEVELOPMENT_COMPONENT_HOLDOUT.json", assignment)
        _publish_json(staging / "HOLDOUT_BUILD_RECEIPT.json", receipt)
        staging.chmod(0o550)
        os.rename(staging, output_root)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    print(
        json.dumps(
            {
                "assignment_content_sha256": assignment[
                    "assignment_content_sha256"
                ],
                "assignment_file_sha256": receipt["assignment_file_sha256"],
                "output_root": str(output_root),
                "receipt_sha256": receipt["receipt_sha256"],
                "train_instance_count": assignment["train"]["instance_count"],
                "holdout_instance_count": assignment["holdout"]["instance_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
