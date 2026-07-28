#!/usr/bin/env python3
"""Materialize completed plan-bound OAS downloads into one pilot dense shard."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import antibody_pretraining.dense_shard as dense_shard_module
import antibody_pretraining.preprocessing as preprocessing_module
from antibody_pretraining.dense_shard import DenseShardRecord, file_sha256, write_dense_shard
from antibody_pretraining.preprocessing import (
    evaluate_unpaired_row,
    load_filter_policy,
    selected_by_plan_fraction,
)


def _parser() -> argparse.ArgumentParser:
    """Build the strict pilot-materialization parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-root", type=Path, required=True)
    parser.add_argument("--plan-payload-sha256", required=True)
    parser.add_argument("--filter-policy", type=Path, required=True)
    parser.add_argument("--filter-policy-file-sha256", required=True)
    parser.add_argument("--maximum-data-units", type=int, required=True)
    parser.add_argument("--maximum-sequence-tokens", type=int, default=194)
    parser.add_argument("--dense-shard-source-sha256", required=True)
    parser.add_argument("--preprocessing-source-sha256", required=True)
    parser.add_argument("--controller-source-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _source_sha256(path: Path) -> str:
    """Hash one controller or imported source file."""
    return file_sha256(path)


def _metadata_and_rows(path: Path) -> tuple[dict[str, Any], Any]:
    """Open one OAS gzip and return metadata with a CSV row iterator."""
    handle = gzip.open(path, "rt", encoding="utf-8", newline="")
    metadata_row = next(csv.reader([handle.readline()]))
    if len(metadata_row) != 1:
        handle.close()
        raise ValueError("OAS metadata line is invalid")
    metadata = json.loads(metadata_row[0])
    return metadata, (handle, csv.DictReader(handle))


def main() -> int:
    """Verify receipts, filter rows, deduplicate, and write a pilot shard."""
    args = _parser().parse_args()
    source_checks = (
        (Path(__file__), args.controller_source_sha256, "controller"),
        (
            Path(dense_shard_module.__file__),
            args.dense_shard_source_sha256,
            "dense shard",
        ),
        (
            Path(preprocessing_module.__file__),
            args.preprocessing_source_sha256,
            "preprocessing",
        ),
    )
    for path, expected, label in source_checks:
        if _source_sha256(path) != expected:
            raise ValueError(f"{label} source digest differs")
    policy, policy_file_sha256 = load_filter_policy(args.filter_policy)
    if policy_file_sha256 != args.filter_policy_file_sha256:
        raise ValueError("Filter-policy file digest differs")
    receipt_root = args.download_root / "receipts" / "unpaired"
    receipt_paths = sorted(receipt_root.rglob("*.receipt.json"))
    if not receipt_paths:
        raise ValueError("Download root contains no completed receipts")
    receipt_paths = receipt_paths[: args.maximum_data_units]
    counters = Counter()
    status_flags = Counter()
    input_receipts = []
    records: list[DenseShardRecord] = []
    seen_sequence_sha256: set[str] = set()
    for receipt_path in receipt_paths:
        receipt_bytes = receipt_path.read_bytes()
        receipt = json.loads(receipt_bytes.decode("ascii"))
        if receipt.get("corpus_plan_payload_sha256") != args.plan_payload_sha256:
            raise ValueError("Download receipt does not bind the frozen corpus plan")
        membership = receipt.get("corpus_plan_membership")
        if not isinstance(membership, dict) or membership.get("split") != "train":
            counters["nontraining_data_unit"] += 1
            continue
        raw_path = (
            args.download_root
            / "raw"
            / "unpaired"
            / receipt["relative_unit"]
        )
        if file_sha256(raw_path) != receipt["compressed_sha256"]:
            raise ValueError("Raw data-unit bytes disagree with the download receipt")
        metadata, opened = _metadata_and_rows(raw_path)
        handle, reader = opened
        authority_chain = metadata.get("Chain")
        observed_rows = 0
        try:
            for row in reader:
                observed_rows += 1
                counters["source_rows"] += 1
                raw_sequence = ""
                for field in policy["sequence_field_priority"]:
                    value = row.get(field)
                    if isinstance(value, str) and value.strip():
                        raw_sequence = value
                        break
                if not selected_by_plan_fraction(
                    raw_sequence,
                    namespace_sha256=membership["selection_hash_namespace"],
                    numerator=membership["selection_fraction_numerator"],
                    denominator=membership["selection_fraction_denominator"],
                ):
                    counters["plan_fraction_rejected"] += 1
                    continue
                counters["plan_fraction_selected"] += 1
                decision = evaluate_unpaired_row(
                    row,
                    authority_chain=authority_chain,
                    policy=policy,
                )
                if decision.accepted is None:
                    counters[f"quality_rejected:{decision.rejection_reason}"] += 1
                    continue
                accepted = decision.accepted
                if accepted.sequence_sha256 in seen_sequence_sha256:
                    counters["exact_duplicate_rejected"] += 1
                    continue
                seen_sequence_sha256.add(accepted.sequence_sha256)
                for flag in accepted.status_flags:
                    status_flags[flag] += 1
                records.append(
                    DenseShardRecord(
                        chain_type=accepted.chain_type,
                        sample_key=int.from_bytes(
                            bytes.fromhex(accepted.sequence_sha256)[:8],
                            "big",
                        ),
                        sequence=accepted.sequence,
                    )
                )
                counters["accepted_unique"] += 1
        finally:
            handle.close()
        if observed_rows != receipt["data_row_count"]:
            raise ValueError("Materializer row count differs from download receipt")
        input_receipts.append(
            {
                "receipt_file_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
                "receipt_payload_sha256": receipt["receipt_sha256"],
                "relative_unit": receipt["relative_unit"],
            }
        )
    provenance = {
        "claim_boundary": {
            "global_production_dedup_complete": False,
            "pilot_only": True,
            "training_started": False,
        },
        "controller_source_sha256": args.controller_source_sha256,
        "dense_shard_source_sha256": args.dense_shard_source_sha256,
        "filter_counts": dict(sorted(counters.items())),
        "filter_policy_file_sha256": policy_file_sha256,
        "input_receipts": input_receipts,
        "plan_payload_sha256": args.plan_payload_sha256,
        "preprocessing_source_sha256": args.preprocessing_source_sha256,
        "status_flag_counts": dict(sorted(status_flags.items())),
    }
    manifest = write_dense_shard(
        args.output,
        records,
        maximum_sequence_tokens=args.maximum_sequence_tokens,
        provenance=provenance,
    )
    print(json.dumps(manifest, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
