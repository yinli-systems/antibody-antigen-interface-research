#!/usr/bin/env python3
"""Create the audited development-decontaminated paired OAS corpus."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
from pathlib import Path


REQUIRED_COLUMNS = (
    "heavy_sequence",
    "light_sequence",
    "cdr1_aa_heavy",
    "cdr2_aa_heavy",
    "cdr3_aa_heavy",
    "cdr1_aa_light",
    "cdr2_aa_light",
    "cdr3_aa_light",
    "study",
    "run",
    "species",
    "donor_hash",
    "row_hash",
)


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


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


def read_excluded(path: Path, expected: int) -> set[str]:
    excluded: set[str] = set()
    previous: str | None = None
    with gzip.open(path, "rt", encoding="ascii", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.endswith("\n"):
                raise ValueError(f"unterminated exclusion at line {line_number}")
            row_hash = line[:-1]
            if (
                len(row_hash) != 64
                or any(c not in "0123456789abcdef" for c in row_hash)
                or (previous is not None and row_hash <= previous)
            ):
                raise ValueError(f"exclusion order/hash differs at line {line_number}")
            excluded.add(row_hash)
            previous = row_hash
    if len(excluded) != expected:
        raise ValueError("exclusion count differs")
    return excluded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--merged-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    for target in (args.output, args.receipt):
        if target.exists() or target.is_symlink():
            raise FileExistsError(target)
    if sha256_file(args.plan) != args.plan_sha256:
        raise ValueError("execution plan SHA-256 differs")
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if (
        plan.get("format_version") != "paired_esmc_cdr_decontamination_plan_v1"
        or plan.get("status") != "FROZEN_BEFORE_SHARD_REDUCTION"
        or plan.get("scripts", {}).get("materialize_paired_devclean_v1.py")
        != sha256_file(Path(__file__))
        or plan.get("audit", {}).get("test_access") is not False
    ):
        raise ValueError("execution plan contract differs")

    source_path = Path(plan["source_pairs"]["path"])
    source_receipt_path = Path(plan["source_pairs"]["receipt"])
    if (
        sha256_file(source_path) != plan["source_pairs"]["sha256"]
        or sha256_file(source_receipt_path)
        != plan["source_pairs"]["receipt_sha256"]
    ):
        raise ValueError("source pair artifact differs")
    source_receipt = json.loads(source_receipt_path.read_text(encoding="utf-8"))
    expected_source_count = plan["source_pairs"]["count"]
    if (
        source_receipt.get("format_version") != "abpairfm_oas_stage_b_candidates_v2"
        or source_receipt.get("counts", {}).get("unique_exact_pairs")
        != expected_source_count
        or source_receipt.get("test_access") is not False
    ):
        raise ValueError("source pair receipt contract differs")

    merged_result_path = args.merged_root / "MERGED_EXCLUSIONS_RESULT_V1.json"
    merged_receipt_path = args.merged_root / "MERGED_EXCLUSIONS_RESULT_V1_RECEIPT.json"
    merged_result = json.loads(merged_result_path.read_text(encoding="utf-8"))
    merged_receipt = json.loads(merged_receipt_path.read_text(encoding="utf-8"))
    if (
        merged_result.get("status") != "PASS_ALL_SHARDS_MERGED_VERIFIED"
        or merged_result.get("plan_sha256") != args.plan_sha256
        or merged_result.get("audit", {}).get("test_access") is not False
        or merged_receipt.get("result_sha256") != sha256_file(merged_result_path)
        or merged_receipt.get("output_sha256")
        != merged_result.get("output", {}).get("sha256")
    ):
        raise ValueError("merged exclusions contract differs")
    exclusions_path = args.merged_root / merged_result["output"]["path"]
    if sha256_file(exclusions_path) != merged_result["output"]["sha256"]:
        raise ValueError("merged exclusions SHA-256 differs")
    excluded = read_excluded(
        exclusions_path, merged_result["counts"]["excluded_unique_pairs"]
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    observed = 0
    excluded_observed = 0
    kept = 0
    try:
        with gzip.open(source_path, "rt", encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source, delimiter="\t")
            if tuple(reader.fieldnames or ()) != REQUIRED_COLUMNS:
                raise ValueError("source pair columns differ")
            with temporary.open("xb") as raw:
                with gzip.GzipFile(filename="", mode="wb", compresslevel=6, fileobj=raw, mtime=0) as compressed:
                    with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                        writer = csv.DictWriter(
                            text,
                            fieldnames=REQUIRED_COLUMNS,
                            delimiter="\t",
                            lineterminator="\n",
                        )
                        writer.writeheader()
                        for row in reader:
                            observed += 1
                            row_hash = row["row_hash"]
                            if len(row_hash) != 64 or any(
                                c not in "0123456789abcdef" for c in row_hash
                            ):
                                raise ValueError(f"source row_hash differs at row {observed}")
                            if row_hash in excluded:
                                excluded_observed += 1
                                continue
                            writer.writerow(row)
                            kept += 1
        if observed != expected_source_count:
            raise ValueError("source pair count differs")
        if excluded_observed != len(excluded):
            raise ValueError("one or more excluded hashes were absent from source pairs")
        os.replace(temporary, args.output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    receipt = {
        "format_version": "abpairfm_oas_stage_b_devclean_sharded_v1",
        "status": "PASS_SCALE_RESTRICTED_PAIRED_ESMC_DATA_GATE",
        "authorization": {
            "paired_esmc_controlled_training": True,
            "full_5500000_pair_claim": False,
            "test_access": False,
        },
        "claim_boundary": (
            "Development-decontaminated exact native VH/VL OAS pairs for the "
            "frozen resource-bounded paired-only ESM-C controlled study; this "
            "is not the paper's full 5.5M-pair corpus or an exact reproduction."
        ),
        "plan_sha256": args.plan_sha256,
        "inputs": {
            "source_pairs_sha256": plan["source_pairs"]["sha256"],
            "source_receipt_sha256": plan["source_pairs"]["receipt_sha256"],
            "merged_exclusions_result_sha256": sha256_file(merged_result_path),
            "merged_exclusions_receipt_sha256": sha256_file(merged_receipt_path),
            "merged_exclusions_sha256": merged_result["output"]["sha256"],
            "hits_sha256": plan["hits"]["sha256"],
        },
        "counts": {
            "source_pairs": observed,
            "excluded_pairs": excluded_observed,
            "kept_pairs": kept,
            "target_deficit_from_5500000": 5_500_000 - kept,
            "mmseqs_hit_lines": merged_result["counts"]["hit_lines"],
        },
        "decontamination": plan["decontamination"],
        "output": {
            "path": str(args.output),
            "bytes": args.output.stat().st_size,
            "sha256": sha256_file(args.output),
        },
        "audit": {
            "all_excluded_hashes_observed_in_source": True,
            "source_count_verified": True,
            "task_labels_read": False,
            "test_access": False,
            "legacy_partial_sqlite_reused": False,
        },
    }
    atomic_write(args.receipt, canonical_bytes(receipt))
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
