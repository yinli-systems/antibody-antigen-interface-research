#!/usr/bin/env python3
"""Prepare and audit positive/negative MMseqs2 gap-aware canary evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

from antibody_pretraining.mmseqs_gap_aware import (
    MMseqsGapAwareError,
    audit_alignment_rows,
    audit_prefilter_database,
    publish_export_receipt,
    sha256_file,
    target_identifier,
)


def _exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _rotate(value: str, positions: int) -> str:
    return value[positions:] + value[:positions]


def prepare(root: Path) -> Mapping[str, Any]:
    root.mkdir(parents=True, exist_ok=False)
    base = "ACDEFGHIKLMNPQRSTVWY" * 5
    targets = [base, _rotate(base, 3), _rotate(base, 7), _rotate(base, 11)]
    changed = "A" if targets[1][50] != "A" else "C"
    queries = [
        targets[0],
        targets[1][:50] + changed + targets[1][51:],
        targets[2][:50] + "A" + targets[2][50:],
        targets[3][1:],
    ]
    names = ["exact", "substitution", "internal_indel", "terminal_indel"]
    query_keys = ["asep_" + hashlib.sha256(name.encode("ascii")).hexdigest() for name in names]
    target_keys = [
        target_identifier(
            task_id=0,
            split="train" if index < 3 else "validation",
            chain_type="heavy",
            authority_ordinal=index + 1,
        )
        for index in range(4)
    ]
    query_path = root / "queries.fasta"
    target_path = root / "targets.fasta"
    _exclusive(
        query_path,
        "".join(f">{key}\n{sequence}\n" for key, sequence in zip(query_keys, queries)).encode("ascii"),
    )
    _exclusive(
        target_path,
        "".join(f">{key}\n{sequence}\n" for key, sequence in zip(target_keys, targets)).encode("ascii"),
    )
    manifest = {
        "expected_pairs": [list(value) for value in zip(query_keys, target_keys)],
        "format_version": "oas_mmseqs_gap_canary_inputs_v1",
        "query_fasta_sha256": sha256_file(query_path),
        "query_lengths": dict(zip(query_keys, map(len, queries))),
        "target_fasta_sha256": sha256_file(target_path),
        "test_access": {
            "development_data_accessed": False,
            "frozen_official_external_test_accessed": False,
        },
    }
    publish_export_receipt(root / "expected.json", manifest)
    return manifest


def _load_expected(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, Mapping) or value.get("format_version") != "oas_mmseqs_gap_canary_inputs_v1":
        raise MMseqsGapAwareError("Canary manifest format differs")
    unsigned = dict(value)
    claimed = unsigned.pop("receipt_sha256", None)
    observed = hashlib.sha256(
        (json.dumps(unsigned, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n").encode("ascii")
    ).hexdigest()
    if claimed != observed:
        raise MMseqsGapAwareError("Canary manifest hash differs")
    return value


def audit_positive(arguments: argparse.Namespace) -> Mapping[str, Any]:
    expected = _load_expected(arguments.expected)
    prefilter = audit_prefilter_database(
        prefix=arguments.prefilter_prefix,
        expected_query_count=4,
        maximum_sequences=arguments.maximum_sequences,
    )
    hits = arguments.output.parent / "hits.tsv"
    with arguments.alignments.open("r", encoding="utf-8") as source, hits.open(
        "x", encoding="ascii", newline=""
    ) as destination:
        alignment = audit_alignment_rows(
            source,
            query_lengths=expected["query_lengths"],
            expected_task_id=0,
            hits_output=destination,
        )
    observed_pairs = set()
    with hits.open("r", encoding="ascii") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            observed_pairs.add(tuple(fields[:2]))
    required_pairs = {tuple(value) for value in expected["expected_pairs"]}
    if not required_pairs.issubset(observed_pairs):
        raise MMseqsGapAwareError("Gap-aware canary lost a registered control")
    result = {
        "alignment_audit": alignment,
        "expected_pair_count": len(required_pairs),
        "format_version": "oas_mmseqs_gap_canary_result_v1",
        "gate_passed": True,
        "hits_sha256": sha256_file(hits),
        "prefilter_audit": prefilter,
        "test_access": expected["test_access"],
    }
    publish_export_receipt(arguments.output, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--output-root", type=Path, required=True)
    prefilter = commands.add_parser("audit-prefilter")
    prefilter.add_argument("--prefix", type=Path, required=True)
    prefilter.add_argument("--query-count", type=int, required=True)
    prefilter.add_argument("--maximum-sequences", type=int, required=True)
    positive = commands.add_parser("audit-positive")
    positive.add_argument("--expected", type=Path, required=True)
    positive.add_argument("--prefilter-prefix", type=Path, required=True)
    positive.add_argument("--maximum-sequences", type=int, required=True)
    positive.add_argument("--alignments", type=Path, required=True)
    positive.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    try:
        if arguments.command == "prepare":
            result = prepare(arguments.output_root)
        elif arguments.command == "audit-prefilter":
            result = audit_prefilter_database(
                prefix=arguments.prefix,
                expected_query_count=arguments.query_count,
                maximum_sequences=arguments.maximum_sequences,
            )
        else:
            result = audit_positive(arguments)
    except MMseqsGapAwareError as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(3) from error
    print(json.dumps(result, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
