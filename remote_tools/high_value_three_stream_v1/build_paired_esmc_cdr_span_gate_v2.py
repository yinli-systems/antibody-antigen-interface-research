#!/usr/bin/env python3
"""Freeze unique jointly ordered CDR spans and pair-group-disjoint splits."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile


CHAIN_CDR_COLUMNS = {
    "heavy": ("cdr1_aa_heavy", "cdr2_aa_heavy", "cdr3_aa_heavy"),
    "light": ("cdr1_aa_light", "cdr2_aa_light", "cdr3_aa_light"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def atomic_write(path: Path, value: object) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(canonical_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def occurrences(sequence: str, motif: str) -> list[tuple[int, int]]:
    return [
        (index, index + len(motif))
        for index in range(len(sequence) - len(motif) + 1)
        if sequence.startswith(motif, index)
    ]


def ordered_chain_spans(
    sequence: str, motifs: tuple[str, str, str]
) -> tuple[list[tuple[int, int]], str | None]:
    if any(not motif for motif in motifs):
        return [], "empty_cdr"
    candidates = [occurrences(sequence, motif) for motif in motifs]
    if any(not values for values in candidates):
        return [], "missing_cdr"
    ordered = [
        (first, second, third)
        for first in candidates[0]
        for second in candidates[1]
        for third in candidates[2]
        if first[1] <= second[0] and second[1] <= third[0]
    ]
    if len(ordered) == 0:
        return [], "no_ordered_triplet"
    if len(ordered) > 1:
        return [], "ambiguous_ordered_triplet"
    return list(ordered[0]), None


def row_spans(row: dict[str, str]) -> tuple[list[tuple[int, int]], str | None]:
    result = []
    for chain in ("heavy", "light"):
        sequence = row[f"{chain}_sequence"].strip().upper()
        motifs = tuple(row[column].strip().upper() for column in CHAIN_CDR_COLUMNS[chain])
        spans, error = ordered_chain_spans(sequence, motifs)
        if error is not None:
            return [], f"{chain}_{error}"
        result.extend(spans)
    return result, None


def pair_sha256(heavy: str, light: str) -> str:
    return hashlib.sha256((heavy + "\0" + light).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-spans", type=Path, required=True)
    parser.add_argument("--output-result", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    args = parser.parse_args()
    outputs = (args.output_spans, args.output_result, args.output_receipt)
    if any(path.exists() or path.is_symlink() for path in outputs):
        raise FileExistsError("one or more output paths already exist")
    authority = json.loads(args.authority.read_text())
    if sha256_file(args.source) != authority["source"]["sha256"]:
        raise RuntimeError("paired source SHA differs")
    if sha256_file(Path(authority["parent_protocol"]["uri"])) != authority["parent_protocol"]["sha256"]:
        raise RuntimeError("parent protocol SHA differs")
    if authority["supersedes"]["result_sha256"] != sha256_file(
        args.output_result.parent / "PAIRED_ESMC_CDR_SPAN_GATE_RESULT_V1.json"
    ):
        raise RuntimeError("superseded failure evidence differs")

    args.output_spans.parent.mkdir(parents=True, exist_ok=True)
    temporary_fd, temporary_name = tempfile.mkstemp(
        prefix=args.output_spans.name + ".tmp.", dir=args.output_spans.parent
    )
    os.close(temporary_fd)
    counts = Counter()
    exclusions = Counter()
    row_hashes: set[str] = set()
    pair_groups: dict[str, str] = {}
    try:
        with gzip.open(args.source, "rt", newline="") as source_handle, open(temporary_name, "wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
                with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                    reader = csv.DictReader(source_handle, delimiter="\t")
                    required = {
                        "heavy_sequence", "light_sequence", "row_hash",
                        *[column for columns in CHAIN_CDR_COLUMNS.values() for column in columns],
                    }
                    if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                        raise RuntimeError("paired source columns differ")
                    fields = [
                        "source_row_index", "row_hash", "pair_sha256", "split",
                        "heavy_length", "light_length",
                        "h1_start", "h1_end", "h2_start", "h2_end", "h3_start", "h3_end",
                        "l1_start", "l1_end", "l2_start", "l2_end", "l3_start", "l3_end",
                    ]
                    writer = csv.DictWriter(text, fieldnames=fields, delimiter="\t", lineterminator="\n")
                    writer.writeheader()
                    for index, row in enumerate(reader):
                        counts["source_rows"] += 1
                        row_hash = row["row_hash"]
                        if row_hash in row_hashes:
                            exclusions["duplicate_row_hash"] += 1
                            continue
                        row_hashes.add(row_hash)
                        heavy = row["heavy_sequence"].strip().upper()
                        light = row["light_sequence"].strip().upper()
                        if len(heavy) + len(light) + 1 > authority["cdr_span_rule"]["maximum_model_input_residues_including_separator"]:
                            exclusions["model_input_too_long"] += 1
                            continue
                        spans, error = row_spans(row)
                        if error is not None:
                            exclusions[error] += 1
                            continue
                        group = pair_sha256(heavy, light)
                        split = "validation" if int(group[:16], 16) % 100 < 2 else "train"
                        if pair_groups.setdefault(group, split) != split:
                            raise RuntimeError("pair group split differs")
                        counts[f"eligible_{split}_rows"] += 1
                        value: dict[str, object] = {
                            "source_row_index": index,
                            "row_hash": row_hash,
                            "pair_sha256": group,
                            "split": split,
                            "heavy_length": len(heavy),
                            "light_length": len(light),
                        }
                        for name, span in zip(("h1", "h2", "h3", "l1", "l2", "l3"), spans):
                            value[f"{name}_start"], value[f"{name}_end"] = span
                        writer.writerow(value)
                    text.flush()
                raw.flush()
                os.fsync(raw.fileno())
        os.replace(temporary_name, args.output_spans)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise

    counts["eligible_rows"] = counts["eligible_train_rows"] + counts["eligible_validation_rows"]
    counts["excluded_rows"] = counts["source_rows"] - counts["eligible_rows"]
    counts["unique_row_hashes"] = len(row_hashes)
    counts["unique_pair_groups"] = len(pair_groups)
    fraction = counts["eligible_rows"] / counts["source_rows"]
    gate = authority["frozen_gate"]
    checks = {
        "source_row_count": counts["source_rows"] == authority["source"]["expected_rows"],
        "minimum_eligible_rows": counts["eligible_rows"] >= gate["minimum_eligible_rows"],
        "minimum_eligible_fraction": fraction >= gate["minimum_eligible_fraction"],
        "minimum_validation_rows": counts["eligible_validation_rows"] >= gate["minimum_validation_rows"],
        "pair_group_disjoint": True,
        "row_hash_unique": exclusions["duplicate_row_hash"] == 0,
        "finite_accounting": counts["eligible_rows"] + counts["excluded_rows"] == counts["source_rows"],
    }
    passed = all(checks.values())
    result = {
        "format_version": "paired_esmc_cdr_span_gate_result_v2",
        "status": "PASS_PAIRED_ESMC_CDR_ORDERED_SPAN_AND_SPLIT_GATE" if passed else "FAIL_CLOSED_PAIRED_ESMC_CDR_ORDERED_SPAN_AND_SPLIT_GATE",
        "authority_sha256": sha256_file(args.authority),
        "parent_protocol_sha256": authority["parent_protocol"]["sha256"],
        "superseded_result_sha256": authority["supersedes"]["result_sha256"],
        "source_sha256": authority["source"]["sha256"],
        "span_artifact": {"path": str(args.output_spans), "sha256": sha256_file(args.output_spans)},
        "counts": dict(sorted(counts.items())),
        "eligible_fraction": fraction,
        "exclusion_reasons": dict(sorted(exclusions.items())),
        "checks": checks,
        "audit": {"pair_group_disjoint_by_construction": True, "checkpoint_selection": False, "task_labels_read": False, "test_access": False},
        "authorization": {
            "local_whole_chain_vs_cdr_canary_authorized": passed,
            "production_training_authorized": False,
            "downstream_test_authorized": False,
        },
    }
    atomic_write(args.output_result, result)
    receipt = {
        "format_version": "paired_esmc_cdr_span_gate_receipt_v2",
        "status": result["status"],
        "result_sha256": sha256_file(args.output_result),
        "span_artifact_sha256": result["span_artifact"]["sha256"],
        "local_whole_chain_vs_cdr_canary_authorized": passed,
        "production_training_authorized": False,
        "test_access": False,
    }
    atomic_write(args.output_receipt, receipt)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
