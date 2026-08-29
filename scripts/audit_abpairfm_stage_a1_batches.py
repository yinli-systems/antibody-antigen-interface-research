#!/usr/bin/env python3
"""Materialize every frozen Stage-A1 pilot microbatch on CPU before GPU launch."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from antibody_pretraining.stage_a1_batch import (  # noqa: E402
    build_stage_a1_microbatch,
    build_stage_a1_retrieval_allowed,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--expected-protocol-sha256")
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    if sha256_file(arguments.manifest) != arguments.expected_manifest_sha256:
        raise RuntimeError("Stage-A1 batch audit manifest SHA-256 differs")
    if arguments.receipt.exists():
        raise FileExistsError("Stage-A1 batch audit receipt is create-only")
    protocol_sha256 = None
    if (arguments.protocol is None) != (arguments.expected_protocol_sha256 is None):
        raise ValueError("Both Stage-A1 protocol audit arguments are required")
    if arguments.protocol is None:
        layout = {
            "ppiref_records_per_microbatch": 4,
            "string_records_per_microbatch": 3,
            "total_records": 22_400,
        }
        world_size = 8
    else:
        protocol_sha256 = sha256_file(arguments.protocol)
        if protocol_sha256 != arguments.expected_protocol_sha256:
            raise RuntimeError("Stage-A1 batch audit protocol SHA-256 differs")
        protocol = json.loads(arguments.protocol.read_text(encoding="utf-8"))
        if (
            protocol.get("format_version")
            not in {
                "abpairfm_stage_a1_multitask_pilot_v3",
                "abpairfm_stage_a1_multitask_pilot_v4",
            }
            or protocol.get("authorization", {}).get("production_training") is not False
            or protocol.get("test_access") is not False
        ):
            raise RuntimeError("Stage-A1 batch audit protocol boundary differs")
        layout = protocol["pilot_manifest"]
        world_size = protocol["systems"]["world_size"]
    process = subprocess.run(
        ["zstd", "-q", "-dc", str(arguments.manifest)],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    records = [json.loads(line) for line in process.stdout.splitlines()]
    if len(records) != layout["total_records"] or any(
        record.get("pilot_record_index") != index
        for index, record in enumerate(records)
    ):
        raise RuntimeError("Stage-A1 batch audit manifest cursor differs")
    counts = {
        "contact_negative_samples": 0,
        "contact_positive_samples": 0,
        "false_negative_candidate_exclusions": 0,
        "global_false_negative_candidate_exclusions": 0,
        "minimum_global_candidates_per_query": None,
        "microbatches": 0,
        "residue_tokens": 0,
        "retrieval_supervised_records": 0,
    }
    microbatch_size = (
        layout["ppiref_records_per_microbatch"]
        + layout["string_records_per_microbatch"]
    )
    batches = []
    for microbatch_index, start in enumerate(
        range(0, len(records), microbatch_size)
    ):
        rows = records[start : start + microbatch_size]
        if (
            sum(row["source"] == "ppiref50k" for row in rows)
            != layout["ppiref_records_per_microbatch"]
        ):
            raise RuntimeError("Stage-A1 microbatch structural/string mixture differs")
        batch = build_stage_a1_microbatch(
            rows, seed=1701 + microbatch_index, maximum_tokens=1024
        )
        counts["microbatches"] += 1
        counts["residue_tokens"] += batch.residue_count
        counts["retrieval_supervised_records"] += len(
            batch.retrieval_batch_indices
        )
        counts["contact_positive_samples"] += int(batch.contact_labels.sum())
        counts["contact_negative_samples"] += int(
            len(batch.contact_labels) - batch.contact_labels.sum()
        )
        counts["false_negative_candidate_exclusions"] += int(
            (~batch.retrieval_allowed).sum()
        )
        batches.append(batch)
    if len(batches) % world_size:
        raise RuntimeError("Stage-A1 batch audit does not form complete rank groups")
    minimum_candidates = None
    for start in range(0, len(batches), world_size):
        rank_batches = batches[start : start + world_size]
        retrieval_counts = [len(batch.retrieval_batch_indices) for batch in rank_batches]
        if len(set(retrieval_counts)) != 1:
            raise RuntimeError("Stage-A1 rank group has unequal retrieval counts")
        global_metadata = tuple(
            torch.cat([getattr(batch, field) for batch in rank_batches])
            for field in (
                "retrieval_left_exact_ids",
                "retrieval_right_exact_ids",
                "retrieval_left_cluster_ids",
                "retrieval_right_cluster_ids",
                "retrieval_positive_group_ids",
            )
        )
        local_count = retrieval_counts[0]
        for rank, batch in enumerate(rank_batches):
            local_metadata = tuple(
                getattr(batch, field)
                for field in (
                    "retrieval_left_exact_ids",
                    "retrieval_right_exact_ids",
                    "retrieval_left_cluster_ids",
                    "retrieval_right_cluster_ids",
                    "retrieval_positive_group_ids",
                )
            )
            targets = rank * local_count + torch.arange(local_count)
            allowed = build_stage_a1_retrieval_allowed(
                local_metadata, global_metadata, targets
            )
            candidate_counts = allowed.sum(dim=1)
            observed_minimum = int(candidate_counts.min())
            minimum_candidates = (
                observed_minimum
                if minimum_candidates is None
                else min(minimum_candidates, observed_minimum)
            )
            counts["global_false_negative_candidate_exclusions"] += int(
                (~allowed).sum()
            )
    counts["minimum_global_candidates_per_query"] = minimum_candidates
    receipt = {
        "claim_boundary": "CPU materialization of every frozen train-only Stage-A1 pilot microbatch; no model, development, or test ability was measured.",
        "counts": counts,
        "format_version": "abpairfm_stage_a1_batch_audit_v1",
        "manifest_sha256": arguments.expected_manifest_sha256,
        "protocol_sha256": protocol_sha256,
        "status": "PASS_ALL_STAGE_A1_PILOT_BATCHES_MATERIALIZED",
        "test_access": False,
    }
    arguments.receipt.parent.mkdir(parents=True, exist_ok=True)
    with arguments.receipt.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
