#!/usr/bin/env python3
"""Freeze a deterministic train-only matched-local coverage canary split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-split", type=Path, required=True)
    parser.add_argument("--source-split-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=16)
    arguments = parser.parse_args()
    if _sha256(arguments.source_split) != arguments.source_split_sha256:
        raise ValueError("Source split SHA-256 differs")
    if arguments.count <= 0:
        raise ValueError("Count must be positive")
    with arguments.source_split.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Source split has no header")
        fieldnames = list(reader.fieldnames)
        candidates = [
            row
            for row in reader
            if row.get("role") == "train"
            and row.get("screen_included", "true").casefold() != "false"
        ]
    candidates.sort(
        key=lambda row: hashlib.sha256(
            f"matched-local-canary-v2\0{row['instance_id']}".encode()
        ).hexdigest()
    )
    selected = []
    clusters = set()
    for row in candidates:
        cluster = row.get("antigen_cluster_id", "")
        if not cluster or cluster in clusters:
            continue
        selected.append(row)
        clusters.add(cluster)
        if len(selected) == arguments.count:
            break
    if len(selected) != arguments.count:
        raise ValueError("Not enough unique antigen clusters for requested canary")
    if arguments.output.exists() or arguments.output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {arguments.output}")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(selected)
    print(
        json.dumps(
            {
                "format_version": "matched_local_canary_split_receipt_v2",
                "output": str(arguments.output),
                "output_sha256": _sha256(arguments.output),
                "selected_train_instances": len(selected),
                "unique_antigen_clusters": len(clusters),
                "validation_rows_written": 0,
                "test_rows_written": 0,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
