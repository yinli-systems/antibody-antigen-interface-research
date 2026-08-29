#!/usr/bin/env python3
"""Create and audit boundary fixtures for the native fixed-IMGT CUDA search."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

from antibody_pretraining.imgt_exact_search import (
    SEED_POSITION_GROUPS,
    build_query_index,
    verify_target_candidates,
)
from antibody_pretraining.imgt_position import (
    CANONICAL_LENGTH,
    GAP,
    MISSING_END,
    CanonicalAlignment,
)
from antibody_pretraining.imgt_query_export import IMGTQueryRecord, pack_imgt_query
from antibody_pretraining.imgt_search_output import (
    FIXED_IMGT_MATCH_RECORD,
    FixedIMGTOutputMatch,
    pack_fixed_imgt_match,
    unpack_fixed_imgt_match,
)
from antibody_pretraining.oas_imgt_export import (
    IMGTTargetRecord,
    pack_imgt_target,
    sequence_from_alignment,
)


FORMAT_VERSION = "oas_imgt_exact_cuda_canary_v1"


class OASIMGTCUDACanaryError(RuntimeError):
    """Report a fixture or native-output mismatch."""


def _alignment(values: bytes) -> CanonicalAlignment:
    result = CanonicalAlignment(
        encoded=values,
        residue_count=sum(value not in {GAP, MISSING_END} for value in values),
    )
    result.validate()
    return result


def _query(
    ordinal: int, values: bytes, *, chain_type: str, role: str
) -> IMGTQueryRecord:
    alignment = _alignment(values)
    sequence = sequence_from_alignment(alignment).encode("ascii")
    return IMGTQueryRecord(
        alignment=alignment,
        chain_type=chain_type,
        domain_sequence_sha256=hashlib.sha256(sequence).hexdigest(),
        query_ordinal=ordinal,
        role=role,
    )


def _target(
    ordinal: int, values: bytes, *, chain_type: str, split: str
) -> IMGTTargetRecord:
    alignment = _alignment(values)
    sequence = sequence_from_alignment(alignment).encode("ascii")
    return IMGTTargetRecord(
        alignment=alignment,
        authority_ordinal=ordinal,
        chain_type=chain_type,
        sequence_sha256=hashlib.sha256(sequence).hexdigest(),
        split=split,
    )


def _mutated(group_count: int) -> bytes:
    values = bytearray(b"A" * CANONICAL_LENGTH)
    for group in SEED_POSITION_GROUPS[:group_count]:
        values[group[0]] = ord("C")
    return bytes(values)


def _fixtures() -> tuple[tuple[IMGTQueryRecord, ...], tuple[IMGTTargetRecord, ...]]:
    if len(SEED_POSITION_GROUPS) != 25:
        raise OASIMGTCUDACanaryError("Frozen seed partition differs")
    queries = (
        _query(7, b"A" * 200, chain_type="heavy", role="train"),
        _query(
            19,
            bytes([MISSING_END]) * 20 + b"A" * 180,
            chain_type="heavy",
            role="val",
        ),
        # The sequence is intentionally cross-labelled.  Conservative
        # decontamination must find it rather than trust the chain annotation.
        _query(23, b"A" * 200, chain_type="kappa", role="train"),
    )
    sparse = bytearray([GAP] * 200)
    for group in SEED_POSITION_GROUPS[:10]:
        for position in group:
            sparse[position] = ord("A")
    five_seed_false_positive = bytearray(b"C" * 200)
    for group in SEED_POSITION_GROUPS[:5]:
        for position in group:
            five_seed_false_positive[position] = ord("A")
    targets = (
        _target(100, b"A" * 200, chain_type="heavy", split="train"),
        _target(101, _mutated(20), chain_type="heavy", split="validation"),
        _target(102, _mutated(21), chain_type="heavy", split="train"),
        _target(
            103,
            bytes([GAP]) * 20 + b"A" * 180,
            chain_type="heavy",
            split="validation",
        ),
        _target(104, bytes(sparse), chain_type="heavy", split="train"),
        _target(
            105,
            bytes(five_seed_false_positive),
            chain_type="heavy",
            split="validation",
        ),
        _target(106, b"A" * 200, chain_type="lambda", split="train"),
    )
    return queries, targets


def _expected_matches(
    queries: Iterable[IMGTQueryRecord], targets: Iterable[IMGTTargetRecord]
) -> tuple[FixedIMGTOutputMatch, ...]:
    index = build_query_index(queries)
    output = []
    for target in targets:
        for match in verify_target_candidates(target, index):
            output.append(
                FixedIMGTOutputMatch(
                    identical_positions=match.metrics.identical_positions,
                    overlapping_positions=match.metrics.overlapping_positions,
                    query_ordinal=match.query_ordinal,
                    query_positions=match.metrics.query_positions,
                    target_authority_ordinal=match.target_authority_ordinal,
                    target_positions=match.metrics.target_positions,
                    target_split=match.target_split,
                    union_positions=match.metrics.union_positions,
                )
            )
    return tuple(
        sorted(output, key=lambda item: (item.target_authority_ordinal, item.query_ordinal))
    )


def _exclusive(path: Path, payload: bytes) -> None:
    descriptor = path.open("xb")
    with descriptor:
        descriptor.write(payload)
        descriptor.flush()


def prepare(output_root: Path) -> dict[str, object]:
    output_root = output_root.expanduser().absolute()
    if output_root.exists() or output_root.is_symlink():
        raise OASIMGTCUDACanaryError("Refusing to overwrite canary fixtures")
    output_root.mkdir(parents=True)
    queries, targets = _fixtures()
    expected = _expected_matches(queries, targets)
    expected_pairs = tuple(
        (item.target_authority_ordinal, item.query_ordinal) for item in expected
    )
    if expected_pairs != (
        (100, 7),
        (100, 19),
        (100, 23),
        (101, 7),
        (101, 23),
        (103, 7),
        (103, 19),
        (103, 23),
        (106, 7),
        (106, 19),
        (106, 23),
    ):
        raise OASIMGTCUDACanaryError("Frozen canary oracle membership differs")
    query_payload = b"".join(pack_imgt_query(item) for item in queries)
    target_payload = b"".join(pack_imgt_target(item) for item in targets)
    expected_payload = b"".join(pack_fixed_imgt_match(item) for item in expected)
    _exclusive(output_root / "queries.bin", query_payload)
    _exclusive(output_root / "targets.bin", target_payload)
    _exclusive(output_root / "expected_matches.bin", expected_payload)
    receipt: dict[str, object] = {
        "expected_match_count": len(expected),
        "expected_match_sha256": hashlib.sha256(expected_payload).hexdigest(),
        "format_version": FORMAT_VERSION,
        "query_count": len(queries),
        "query_sha256": hashlib.sha256(query_payload).hexdigest(),
        "target_count": len(targets),
        "target_sha256": hashlib.sha256(target_payload).hexdigest(),
        "test_cases": {
            "cross_chain_label_recall": True,
            "five_seed_false_positive_rejected": True,
            "gap_missing_end_equivalence": True,
            "identity_boundary_20_disagreements": True,
            "identity_negative_21_disagreements": True,
            "reciprocal_coverage_negative": True,
        },
    }
    receipt["receipt_sha256"] = hashlib.sha256(
        (json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n").encode(
            "ascii"
        )
    ).hexdigest()
    _exclusive(
        output_root / "fixture_receipt.json",
        (json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n").encode(
            "ascii"
        ),
    )
    return receipt


def _records(path: Path) -> tuple[FixedIMGTOutputMatch, ...]:
    if path.is_symlink() or not path.is_file():
        raise OASIMGTCUDACanaryError("Canary match file is absent or symbolic")
    payload = path.read_bytes()
    if len(payload) % FIXED_IMGT_MATCH_RECORD.size:
        raise OASIMGTCUDACanaryError("Canary match output ended mid-record")
    return tuple(
        unpack_fixed_imgt_match(payload[offset : offset + FIXED_IMGT_MATCH_RECORD.size])
        for offset in range(0, len(payload), FIXED_IMGT_MATCH_RECORD.size)
    )


def audit(fixture_root: Path, actual_matches: Path, kernel_stats: Path) -> dict[str, object]:
    fixture_root = fixture_root.expanduser().absolute()
    expected_path = fixture_root / "expected_matches.bin"
    expected = _records(expected_path)
    actual = _records(actual_matches.expanduser().absolute())
    if actual != expected or actual_matches.read_bytes() != expected_path.read_bytes():
        raise OASIMGTCUDACanaryError("CUDA matches differ byte-for-byte from Python oracle")
    if kernel_stats.is_symlink() or not kernel_stats.is_file():
        raise OASIMGTCUDACanaryError("CUDA kernel stats are absent or symbolic")
    try:
        stats = json.loads(kernel_stats.read_text(encoding="ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OASIMGTCUDACanaryError("CUDA kernel stats are invalid JSON") from error
    if (
        not isinstance(stats, dict)
        or stats.get("format_version") != "oas_imgt_exact_cuda_stats_v1"
        or stats.get("query_records") != 3
        or stats.get("target_file_records") != 7
        or stats.get("records_scanned") != 7
        or stats.get("matches") != len(expected)
        or stats.get("seed_lookups") != 175
    ):
        raise OASIMGTCUDACanaryError("CUDA kernel stats differ from the canary domain")
    return {
        "actual_match_count": len(actual),
        "actual_match_sha256": hashlib.sha256(actual_matches.read_bytes()).hexdigest(),
        "cuda_device": stats.get("cuda_device"),
        "format_version": FORMAT_VERSION,
        "result": "PASS",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--output-root", type=Path, required=True)
    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("--fixture-root", type=Path, required=True)
    audit_parser.add_argument("--actual-matches", type=Path, required=True)
    audit_parser.add_argument("--kernel-stats", type=Path, required=True)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    if arguments.command == "prepare":
        result = prepare(arguments.output_root)
    else:
        result = audit(
            arguments.fixture_root, arguments.actual_matches, arguments.kernel_stats
        )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
