#!/usr/bin/env python3
"""Run authenticated PAIRScale multi-decoy repair statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from interaction_first.pairscale_multidecoy_repair_statistics import (
    compare_pairscale_multidecoy_repair,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "assignment",
        "assignment-receipt",
        "repair-evaluation",
        "reference-evaluation",
        "no-centering-evaluation",
        "antibody-only-evaluation",
        "antigen-only-evaluation",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in (
        "assignment-file-sha256",
        "assignment-content-sha256",
        "assignment-receipt-sha256",
        "repair-evaluation-sha256",
        "reference-evaluation-sha256",
        "no-centering-evaluation-sha256",
        "antibody-only-evaluation-sha256",
        "antigen-only-evaluation-sha256",
    ):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--created-at-utc")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = compare_pairscale_multidecoy_repair(
        assignment_path=arguments.assignment,
        assignment_receipt_path=arguments.assignment_receipt,
        repair_evaluation_path=arguments.repair_evaluation,
        reference_evaluation_path=arguments.reference_evaluation,
        no_centering_evaluation_path=arguments.no_centering_evaluation,
        antibody_only_evaluation_path=arguments.antibody_only_evaluation,
        antigen_only_evaluation_path=arguments.antigen_only_evaluation,
        expected_assignment_file_sha256=arguments.assignment_file_sha256,
        expected_assignment_content_sha256=arguments.assignment_content_sha256,
        expected_assignment_receipt_sha256=arguments.assignment_receipt_sha256,
        expected_repair_evaluation_sha256=arguments.repair_evaluation_sha256,
        expected_reference_evaluation_sha256=arguments.reference_evaluation_sha256,
        expected_no_centering_evaluation_sha256=(
            arguments.no_centering_evaluation_sha256
        ),
        expected_antibody_only_evaluation_sha256=(
            arguments.antibody_only_evaluation_sha256
        ),
        expected_antigen_only_evaluation_sha256=(
            arguments.antigen_only_evaluation_sha256
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
