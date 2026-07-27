#!/usr/bin/env python3
"""Run authenticated PAIRScale pair-versus-unary component statistics."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from interaction_first.pairscale_unary_statistics import (
    compare_pairscale_unary_controls,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignment", type=Path, required=True)
    parser.add_argument("--assignment-receipt", type=Path, required=True)
    parser.add_argument("--pair-evaluation", type=Path, required=True)
    parser.add_argument("--antibody-only-evaluation", type=Path, required=True)
    parser.add_argument("--antigen-only-evaluation", type=Path, required=True)
    parser.add_argument("--assignment-file-sha256", required=True)
    parser.add_argument("--assignment-content-sha256", required=True)
    parser.add_argument("--assignment-receipt-sha256", required=True)
    parser.add_argument("--pair-evaluation-sha256", required=True)
    parser.add_argument("--antibody-only-evaluation-sha256", required=True)
    parser.add_argument("--antigen-only-evaluation-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()
    result = compare_pairscale_unary_controls(
        assignment_path=arguments.assignment,
        assignment_receipt_path=arguments.assignment_receipt,
        pair_evaluation_path=arguments.pair_evaluation,
        antibody_only_evaluation_path=arguments.antibody_only_evaluation,
        antigen_only_evaluation_path=arguments.antigen_only_evaluation,
        expected_assignment_file_sha256=arguments.assignment_file_sha256,
        expected_assignment_content_sha256=arguments.assignment_content_sha256,
        expected_assignment_receipt_sha256=arguments.assignment_receipt_sha256,
        expected_pair_evaluation_sha256=arguments.pair_evaluation_sha256,
        expected_antibody_only_evaluation_sha256=(
            arguments.antibody_only_evaluation_sha256
        ),
        expected_antigen_only_evaluation_sha256=(
            arguments.antigen_only_evaluation_sha256
        ),
        iterations=arguments.iterations,
        seed=arguments.seed,
    )
    output = arguments.output.expanduser().absolute()
    if output.exists() or output.is_symlink() or not output.parent.is_dir():
        raise ValueError("output must be a new file in an existing directory")
    encoded = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    with output.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
