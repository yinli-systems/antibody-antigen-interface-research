#!/usr/bin/env python3
"""Materialize every frozen Stage-A1 pilot microbatch on CPU before GPU launch."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from antibody_pretraining.stage_a1_batch import build_stage_a1_microbatch  # noqa: E402


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
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    if sha256_file(arguments.manifest) != arguments.expected_manifest_sha256:
        raise RuntimeError("Stage-A1 batch audit manifest SHA-256 differs")
    if arguments.receipt.exists():
        raise FileExistsError("Stage-A1 batch audit receipt is create-only")
    process = subprocess.run(
        ["zstd", "-q", "-dc", str(arguments.manifest)],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    records = [json.loads(line) for line in process.stdout.splitlines()]
    if len(records) != 22_400 or any(
        record.get("pilot_record_index") != index
        for index, record in enumerate(records)
    ):
        raise RuntimeError("Stage-A1 batch audit manifest cursor differs")
    counts = {
        "contact_negative_samples": 0,
        "contact_positive_samples": 0,
        "false_negative_candidate_exclusions": 0,
        "microbatches": 0,
        "residue_tokens": 0,
        "retrieval_supervised_records": 0,
    }
    for microbatch_index, start in enumerate(range(0, len(records), 7)):
        rows = records[start : start + 7]
        if sum(row["source"] == "ppiref50k" for row in rows) != 4:
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
    receipt = {
        "claim_boundary": "CPU materialization of every frozen train-only Stage-A1 pilot microbatch; no model, development, or test ability was measured.",
        "counts": counts,
        "format_version": "abpairfm_stage_a1_batch_audit_v1",
        "manifest_sha256": arguments.expected_manifest_sha256,
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
