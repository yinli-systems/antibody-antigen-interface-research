#!/usr/bin/env python3
"""Freeze label-free same-antigen/different-antibody challenge panels."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from interaction_first.sabdab2_structure import (  # noqa: E402
    canonical_json_bytes,
    sha256_file,
)
from interaction_first.same_antigen_challenge import (  # noqa: E402
    LABEL_FREE_SAME_ANTIGEN_CHALLENGE_FORMAT,
    build_label_free_same_antigen_challenge_entries,
)


RECEIPT_FORMAT = "framecontact_label_free_same_antigen_challenge_receipt_v1"
ROLES = ("train", "validation", "test")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--metadata-sha256", required=True)
    parser.add_argument("--metadata-receipt", type=Path, required=True)
    parser.add_argument("--metadata-receipt-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-alternates-per-focal", type=int, default=31)
    parser.add_argument("--seed", type=int, default=20_260_730)
    return parser.parse_args()


def _authenticate(arguments: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, str]]]:
    if sha256_file(arguments.metadata) != arguments.metadata_sha256:
        raise ValueError("Partner metadata SHA-256 differs")
    if sha256_file(arguments.metadata_receipt) != arguments.metadata_receipt_sha256:
        raise ValueError("Partner metadata receipt SHA-256 differs")
    receipt = json.loads(arguments.metadata_receipt.read_text(encoding="utf-8"))
    if (
        receipt.get("format_version") != "sabdab2_partner_metadata_split_v1"
        or receipt.get("metadata_sha256") != arguments.metadata_sha256
        or receipt.get("authority", {}).get("task_labels_read") is not False
    ):
        raise ValueError("Partner metadata is not a label-free split authority")
    with arguments.metadata.open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    if len(records) != int(receipt["instance_count"]):
        raise ValueError("Partner metadata row count differs from its receipt")
    if {row["role"] for row in records} != set(ROLES):
        raise ValueError("Partner metadata does not contain the frozen three roles")
    if len({row["instance_id"] for row in records}) != len(records):
        raise ValueError("Partner metadata contains duplicate instances")
    return receipt, records


def main() -> None:
    arguments = _arguments()
    if arguments.output_root.exists() or arguments.output_root.is_symlink():
        raise FileExistsError("Refusing to overwrite challenge authority")
    metadata_receipt, records = _authenticate(arguments)
    arguments.output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{arguments.output_root.name}.tmp-",
            dir=arguments.output_root.parent,
        )
    )
    try:
        role_receipts: dict[str, dict[str, Any]] = {}
        for role in ROLES:
            role_records = [row for row in records if row["role"] == role]
            entries = build_label_free_same_antigen_challenge_entries(
                role_records,
                max_alternates_per_focal=arguments.max_alternates_per_focal,
                seed=arguments.seed,
            )
            output = {
                "format_version": LABEL_FREE_SAME_ANTIGEN_CHALLENGE_FORMAT,
                "claim_boundary": (
                    "Exact-antigen/different-antibody counterfactual identities selected "
                    "only from split-bound, task-label-free metadata. No epitope/contact "
                    "label or model prediction was read."
                ),
                "role": role,
                "authority": {
                    "metadata_sha256": arguments.metadata_sha256,
                    "metadata_receipt_sha256": arguments.metadata_receipt_sha256,
                    "split_sha256": metadata_receipt["authority"]["split_sha256"],
                    "task_labels_read": False,
                    "model_predictions_read": False,
                    "seed": arguments.seed,
                    "max_alternates_per_focal": arguments.max_alternates_per_focal,
                    "same_antigen_rule": "exact antigen_sequence_sha256",
                    "different_antibody_rule": (
                        "both antibody_sequence_sha256 and antibody_cluster_id differ"
                    ),
                },
                "source_role_instance_count": len(role_records),
                "exact_antigen_group_count": len(
                    {entry["antigen_sequence_sha256"] for entry in entries}
                ),
                "focal_instance_count": len(entries),
                "ordered_pair_count": sum(
                    len(entry["alternates"]) for entry in entries
                ),
                "entries": entries,
            }
            path = temporary / f"{role}.json"
            path.write_bytes(canonical_json_bytes(output))
            role_receipts[role] = {
                "path": path.name,
                "sha256": sha256_file(path),
                "source_role_instance_count": len(role_records),
                "exact_antigen_group_count": output["exact_antigen_group_count"],
                "focal_instance_count": output["focal_instance_count"],
                "ordered_pair_count": output["ordered_pair_count"],
            }
        receipt = {
            "format_version": RECEIPT_FORMAT,
            "claim_boundary": (
                "Pre-evaluation, label-free same-antigen challenge authority; test "
                "targets and predictions remain unopened."
            ),
            "metadata_sha256": arguments.metadata_sha256,
            "metadata_receipt_sha256": arguments.metadata_receipt_sha256,
            "task_labels_read": False,
            "model_predictions_read": False,
            "roles": role_receipts,
        }
        receipt_path = temporary / "SAME_ANTIGEN_CHALLENGE_RECEIPT.json"
        receipt_path.write_bytes(canonical_json_bytes(receipt))
        os.replace(temporary, arguments.output_root)
        print(
            json.dumps(
                {
                    "output_root": str(arguments.output_root),
                    "receipt_sha256": sha256_file(
                        arguments.output_root
                        / "SAME_ANTIGEN_CHALLENGE_RECEIPT.json"
                    ),
                    "roles": role_receipts,
                },
                sort_keys=True,
            )
        )
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


if __name__ == "__main__":
    main()
