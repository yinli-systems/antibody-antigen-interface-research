#!/usr/bin/env python3
"""Compile authenticated OAS near-identity search tracks into exclusions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from antibody_pretraining.near_identity_exclusion import compile_exclusions


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-root", type=Path, required=True)
    parser.add_argument("--gap-root", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-exact-receipt-sha256", required=True)
    parser.add_argument("--expected-gap-receipt-sha256", required=True)
    parser.add_argument("--expected-target-receipt-sha256", required=True)
    parser.add_argument("--controller-source-sha256", required=True)
    parser.add_argument("--task-count", type=int, default=32)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    receipt = compile_exclusions(
        exact_root=arguments.exact_root,
        gap_root=arguments.gap_root,
        target_root=arguments.target_root,
        output_root=arguments.output_root,
        expected_exact_receipt_sha256=arguments.expected_exact_receipt_sha256,
        expected_gap_receipt_sha256=arguments.expected_gap_receipt_sha256,
        expected_target_receipt_sha256=arguments.expected_target_receipt_sha256,
        controller_source_sha256=arguments.controller_source_sha256,
        task_count=arguments.task_count,
    )
    print(json.dumps(receipt, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
