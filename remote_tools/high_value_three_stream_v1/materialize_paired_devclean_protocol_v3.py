#!/usr/bin/env python3
"""Materialize paired OAS data after the frozen protocol-v3 exclusions."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
from pathlib import Path


COLUMNS = (
    "heavy_sequence", "light_sequence", "cdr1_aa_heavy", "cdr2_aa_heavy",
    "cdr3_aa_heavy", "cdr1_aa_light", "cdr2_aa_light", "cdr3_aa_light",
    "study", "run", "species", "donor_hash", "row_hash",
)


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--merged-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.output, args.receipt):
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
    if sha256_file(args.plan) != args.plan_sha256:
        raise ValueError("plan SHA-256 differs")
    plan = json.loads(args.plan.read_text())
    if (
        plan.get("format_version") != "paired_esmc_cdr_decontamination_plan_v2"
        or plan.get("status") != "FROZEN_PROTOCOL_V3_EXCLUSION_BEFORE_REDUCTION"
        or plan.get("scripts", {}).get(Path(__file__).name) != sha256_file(Path(__file__))
        or plan.get("audit", {}).get("test_access") is not False
    ):
        raise ValueError("plan contract differs")
    source_path = Path(plan["source_pairs"]["path"])
    source_receipt_path = Path(plan["source_pairs"]["receipt"])
    if (
        sha256_file(source_path) != plan["source_pairs"]["sha256"]
        or sha256_file(source_receipt_path) != plan["source_pairs"]["receipt_sha256"]
    ):
        raise ValueError("source pairs differ")
    source_receipt = json.loads(source_receipt_path.read_text())
    source_count = plan["source_pairs"]["count"]
    if (
        source_receipt.get("counts", {}).get("unique_exact_pairs") != source_count
        or source_receipt.get("test_access") is not False
    ):
        raise ValueError("source receipt differs")
    result_path = args.merged_root / "MERGED_EXCLUSIONS_RESULT_V2.json"
    merged_receipt_path = args.merged_root / "MERGED_EXCLUSIONS_RESULT_V2_RECEIPT.json"
    result = json.loads(result_path.read_text())
    merged_receipt = json.loads(merged_receipt_path.read_text())
    if (
        result.get("status") != "PASS_PROTOCOL_V3_ALL_SHARDS_MERGED"
        or result.get("plan_sha256") != args.plan_sha256
        or result.get("audit", {}).get("test_access") is not False
        or merged_receipt.get("result_sha256") != sha256_file(result_path)
        or merged_receipt.get("output_sha256") != result["output"]["sha256"]
    ):
        raise ValueError("merged exclusions differ")
    excluded_path = args.merged_root / result["output"]["path"]
    if sha256_file(excluded_path) != result["output"]["sha256"]:
        raise ValueError("exclusion artifact differs")
    excluded: set[str] = set()
    previous: str | None = None
    with gzip.open(excluded_path, "rt", encoding="ascii", newline="") as handle:
        for line in handle:
            if not line.endswith("\n"):
                raise ValueError("unterminated exclusion")
            row_hash = line[:-1]
            if previous is not None and row_hash <= previous:
                raise ValueError("exclusion order differs")
            excluded.add(row_hash)
            previous = row_hash
    if len(excluded) != result["counts"]["excluded_union_pairs"]:
        raise ValueError("exclusion count differs")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    observed = excluded_observed = kept = 0
    try:
        with gzip.open(source_path, "rt", encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source, delimiter="\t")
            if tuple(reader.fieldnames or ()) != COLUMNS:
                raise ValueError("source columns differ")
            with temporary.open("xb") as raw:
                with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
                    with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                        writer = csv.DictWriter(
                            text, fieldnames=COLUMNS, delimiter="\t", lineterminator="\n"
                        )
                        writer.writeheader()
                        for row in reader:
                            observed += 1
                            if row["row_hash"] in excluded:
                                excluded_observed += 1
                            else:
                                writer.writerow(row)
                                kept += 1
        if observed != source_count or excluded_observed != len(excluded):
            raise ValueError("source/exclusion accounting differs")
        os.replace(temporary, args.output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    receipt = {
        "format_version": "abpairfm_oas_stage_b_devclean_protocol_v3_v1",
        "status": "PASS_PROTOCOL_V3_DECONTAMINATED_CORPUS_MATERIALIZED",
        "authorization": {
            "training_authorized": False,
            "requires_separate_corpus_sufficiency_and_model_binding_gate": True,
            "test_access": False,
        },
        "claim_boundary": (
            "Exact native paired OAS subset after the frozen v3 90%-single-chain "
            "or 80%-paired-same-development-instance rule; not the paper's full "
            "5.5M-pair corpus and not yet authorized for training."
        ),
        "plan_sha256": args.plan_sha256,
        "inputs": {
            "source_sha256": plan["source_pairs"]["sha256"],
            "source_receipt_sha256": plan["source_pairs"]["receipt_sha256"],
            "merged_result_sha256": sha256_file(result_path),
            "merged_receipt_sha256": sha256_file(merged_receipt_path),
            "exclusions_sha256": result["output"]["sha256"],
            "hits_sha256": plan["hits"]["sha256"],
        },
        "counts": {
            "source_pairs": observed,
            "excluded_pairs": excluded_observed,
            "kept_pairs": kept,
            **result["counts"],
        },
        "decontamination": plan["decontamination"],
        "output": {
            "path": str(args.output),
            "bytes": args.output.stat().st_size,
            "sha256": sha256_file(args.output),
        },
        "audit": {
            "accounting_verified": True,
            "task_labels_read": False,
            "test_access": False,
            "legacy_partial_sqlite_reused": False,
        },
    }
    atomic_write(args.receipt, canonical_bytes(receipt))
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
