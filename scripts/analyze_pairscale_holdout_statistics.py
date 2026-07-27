#!/usr/bin/env python3
"""Run authenticated PAIRScale development holdout statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from interaction_first.pairscale_holdout_statistics import (
    analyze_development_holdout_statistics,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assignment", type=Path, required=True)
    parser.add_argument("--assignment-receipt", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--assignment-file-sha256", required=True)
    parser.add_argument("--assignment-content-sha256", required=True)
    parser.add_argument("--assignment-receipt-sha256", required=True)
    parser.add_argument("--evaluation-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = analyze_development_holdout_statistics(
        assignment_path=args.assignment,
        assignment_receipt_path=args.assignment_receipt,
        evaluation_path=args.evaluation,
        expected_assignment_file_sha256=args.assignment_file_sha256,
        expected_assignment_content_sha256=args.assignment_content_sha256,
        expected_assignment_receipt_sha256=args.assignment_receipt_sha256,
        expected_evaluation_sha256=args.evaluation_sha256,
        iterations=args.iterations,
        seed=args.seed,
    )
    output = args.output.expanduser().absolute()
    if output.exists() or output.is_symlink() or not output.parent.is_dir():
        raise ValueError("output must be a new file in an existing directory")
    encoded = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    with output.open("xb") as handle:
        handle.write(encoded)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
