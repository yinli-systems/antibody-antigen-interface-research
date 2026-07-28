#!/usr/bin/env python3
"""Audit three immutable PAIRScale E1/E2 screening reports."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Mapping

from interaction_first.development_authority import canonical_json_bytes
from interaction_first.pairscale_e1_e2_gate_statistics import (
    analyze_e1_e2_screening,
)


def _seed_mapping(values: list[str], name: str) -> Mapping[int, str]:
    result: dict[int, str] = {}
    for value in values:
        seed_text, separator, item = value.partition("=")
        if separator != "=" or not seed_text or not item:
            raise ValueError(f"{name} entries must use SEED=VALUE")
        try:
            seed = int(seed_text)
        except ValueError as exc:
            raise ValueError(f"{name} seed must be an integer") from exc
        if seed in result:
            raise ValueError(f"{name} contains duplicate seed {seed}")
        result[seed] = item
    return result


def _create_json(path: Path, payload: Mapping[str, object]) -> None:
    target = path.expanduser().absolute()
    if target.is_symlink() or target.exists():
        raise ValueError("output must be create-only")
    target.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(target, flags, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json_bytes(dict(payload)))
    except BaseException:
        try:
            target.unlink()
        except FileNotFoundError:
            pass
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignment", type=Path, required=True)
    parser.add_argument("--assignment-receipt", type=Path, required=True)
    parser.add_argument("--assignment-file-sha256", required=True)
    parser.add_argument("--assignment-content-sha256", required=True)
    parser.add_argument("--assignment-receipt-sha256", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument(
        "--report",
        action="append",
        default=[],
        metavar="SEED=PATH",
        required=True,
    )
    parser.add_argument(
        "--report-sha256",
        action="append",
        default=[],
        metavar="SEED=SHA256",
        required=True,
    )
    parser.add_argument("--created-at-utc")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    report_values = _seed_mapping(arguments.report, "report")
    report_sha256 = _seed_mapping(arguments.report_sha256, "report-sha256")
    result = analyze_e1_e2_screening(
        assignment_path=arguments.assignment,
        assignment_receipt_path=arguments.assignment_receipt,
        expected_assignment_file_sha256=arguments.assignment_file_sha256,
        expected_assignment_content_sha256=arguments.assignment_content_sha256,
        expected_assignment_receipt_sha256=arguments.assignment_receipt_sha256,
        protocol_path=arguments.protocol,
        expected_protocol_sha256=arguments.protocol_sha256,
        report_paths_by_seed={
            seed: Path(value) for seed, value in report_values.items()
        },
        expected_report_sha256_by_seed=report_sha256,
        created_at_utc=arguments.created_at_utc,
    )
    _create_json(arguments.output, result)


if __name__ == "__main__":
    main()
