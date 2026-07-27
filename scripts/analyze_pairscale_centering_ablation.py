#!/usr/bin/env python3
"""Run the authenticated PAIRScale centering-ablation comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from interaction_first.pairscale_centering_ablation_statistics import (
    compare_pairscale_centering_ablation,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignment", type=Path, required=True)
    parser.add_argument("--assignment-receipt", type=Path, required=True)
    parser.add_argument("--reference-evaluation", type=Path, required=True)
    parser.add_argument("--no-centering-evaluation", type=Path, required=True)
    parser.add_argument("--assignment-file-sha256", required=True)
    parser.add_argument("--assignment-content-sha256", required=True)
    parser.add_argument("--assignment-receipt-sha256", required=True)
    parser.add_argument("--reference-evaluation-sha256", required=True)
    parser.add_argument("--no-centering-evaluation-sha256", required=True)
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--created-at-utc")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = compare_pairscale_centering_ablation(
        assignment_path=arguments.assignment,
        assignment_receipt_path=arguments.assignment_receipt,
        reference_evaluation_path=arguments.reference_evaluation,
        no_centering_evaluation_path=arguments.no_centering_evaluation,
        expected_assignment_file_sha256=arguments.assignment_file_sha256,
        expected_assignment_content_sha256=arguments.assignment_content_sha256,
        expected_assignment_receipt_sha256=arguments.assignment_receipt_sha256,
        expected_reference_evaluation_sha256=arguments.reference_evaluation_sha256,
        expected_no_centering_evaluation_sha256=(
            arguments.no_centering_evaluation_sha256
        ),
        iterations=arguments.iterations,
        seed=arguments.seed,
        created_at_utc=arguments.created_at_utc,
    )
    output = arguments.output.expanduser().absolute()
    if output.exists() or output.is_symlink() or not output.parent.is_dir():
        raise ValueError("output must be a new file in an existing directory")
    raw = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    with output.open("xb") as handle:
        handle.write(raw)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
