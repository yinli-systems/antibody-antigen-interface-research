#!/usr/bin/env python3
"""Bind label-free partner metadata to a frozen screen and build panels."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import tempfile


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from interaction_first.reciprocal_partner_panel import (  # noqa: E402
    build_reciprocal_partner_panel,
)
from interaction_first.sabdab2_structure import (  # noqa: E402
    canonical_json_bytes,
    sha256_file,
)


def _authenticate(path: Path, expected: str, label: str) -> None:
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"{label} SHA-256 differs: {observed}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--metadata-sha256", required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--split-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--panel-size", type=int, default=7)
    parser.add_argument("--protocol-seed", type=int, default=20_260_824)
    arguments = parser.parse_args()
    _authenticate(arguments.metadata, arguments.metadata_sha256, "partner metadata")
    _authenticate(arguments.split, arguments.split_sha256, "screen split")
    if arguments.output_root.exists() or arguments.output_root.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {arguments.output_root}")
    arguments.output_root.parent.mkdir(parents=True, exist_ok=True)
    with arguments.split.open("r", encoding="utf-8", newline="") as handle:
        split_rows = list(csv.DictReader(handle))
    roles = {
        str(row["instance_id"]): str(row["role"])
        for row in split_rows
        if str(row.get("screen_included", "true")).casefold() == "true"
    }
    if sum(value == "train" for value in roles.values()) != 512 or sum(
        value == "validation" for value in roles.values()
    ) != 128:
        raise ValueError("Screen split is not the frozen 512/128 cohort")
    with arguments.metadata.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if fieldnames is None or "role" not in fieldnames:
        raise ValueError("Partner metadata lacks a role column")
    metadata_ids = {str(row["instance_id"]) for row in rows}
    if not set(roles).issubset(metadata_ids):
        raise ValueError("Partner metadata lacks a screen instance")
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{arguments.output_root.name}.", dir=arguments.output_root.parent
        )
    )
    try:
        bound = temporary / "SCREEN_PARTNER_METADATA.csv"
        with bound.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            for row in rows:
                row = dict(row)
                row["role"] = roles.get(str(row["instance_id"]), "excluded")
                writer.writerow(row)
        bound_sha256 = sha256_file(bound)
        panel_hashes = {}
        for role in ("train", "validation"):
            output = temporary / f"{role}-reciprocal-{arguments.panel_size}.json"
            build_reciprocal_partner_panel(
                metadata_path=bound,
                metadata_sha256=bound_sha256,
                output_path=output,
                role=role,
                panel_size_per_direction=arguments.panel_size,
                protocol_seed=arguments.protocol_seed,
            )
            panel_hashes[role] = sha256_file(output)
        receipt = {
            "format_version": "sabdab2_screen_reciprocal_panel_receipt_v1",
            "authority": {
                "source_metadata_sha256": arguments.metadata_sha256,
                "screen_split_sha256": arguments.split_sha256,
                "task_labels_read": False,
                "test_features_or_labels_read": False,
            },
            "screen_partner_metadata_sha256": bound_sha256,
            "panel_size_per_direction": arguments.panel_size,
            "panel_sha256": panel_hashes,
            "role_counts": {"train": 512, "validation": 128},
        }
        (temporary / "SCREEN_RECIPROCAL_PANEL_RECEIPT.json").write_bytes(
            canonical_json_bytes(receipt)
        )
        temporary.replace(arguments.output_root)
    except BaseException:
        import shutil

        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
