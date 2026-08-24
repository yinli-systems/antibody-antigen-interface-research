#!/usr/bin/env python3
"""Build a deterministic train/validation screen strictly inside a frozen split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def rank(row: dict[str, str], salt: str) -> bytes:
    return hashlib.sha256(
        f"{salt}\0{row['antigen_cluster_id']}\0{row['instance_id']}".encode()
    ).digest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-count", type=int, default=512)
    parser.add_argument("--validation-count", type=int, default=128)
    parser.add_argument("--salt", default="sabdab2-same-protocol-screen-v3")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = list(csv.DictReader(args.source.open(newline="", encoding="utf-8")))
    development_clusters = {
        role: {
            row["antigen_cluster_id"]
            for row in rows
            if row["role"] == role
        }
        for role in ("train", "validation")
    }
    overlapping_clusters = (
        development_clusters["train"] & development_clusters["validation"]
    )
    selected: list[dict[str, str]] = []
    for role, count in (("train", args.train_count), ("validation", args.validation_count)):
        local = [
            row
            for row in rows
            if row["role"] == role
            and row["antigen_cluster_id"] not in overlapping_clusters
        ]
        if len(local) < count:
            raise ValueError(f"only {len(local)} eligible {role} rows for {count} requested")
        local.sort(key=lambda row: rank(row, args.salt))
        selected.extend(local[:count])
    selected_ids = {row["instance_id"] for row in selected}
    if len(selected_ids) != args.train_count + args.validation_count:
        raise ValueError("selected instance identifiers are not unique")
    output_rows: list[dict[str, str]] = []
    for row in rows:
        output_row = dict(row)
        output_row["source_role"] = row["role"]
        output_row["screen_included"] = str(
            row["instance_id"] in selected_ids
        ).lower()
        if row["instance_id"] not in selected_ids:
            # The cohort loader requires a total role mapping over all cached
            # identifiers. It never instantiates this inactive role in the
            # development screen.
            output_row["role"] = "test"
        output_rows.append(output_row)
    output_rows.sort(key=lambda row: row["instance_id"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[*rows[0].keys(), "source_role", "screen_included"],
        )
        writer.writeheader()
        writer.writerows(output_rows)
    print(
        json.dumps(
            {
                "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                "train_count": args.train_count,
                "validation_count": args.validation_count,
                "excluded_train_validation_antigen_clusters": sorted(
                    overlapping_clusters
                ),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
