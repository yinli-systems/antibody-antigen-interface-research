#!/usr/bin/env python3
"""Run one registered independent full-linear V6-B CP-SAT audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from interaction_first.v6_b_independent_cp_audit import (
    execute_independent_cp_audit,
)
from interaction_first.v6_b_production import read_verified, strict_json


def parse_args() -> argparse.Namespace:
    """Parse only explicitly registered content anchors."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--registration-sha256", required=True)
    parser.add_argument("--primary-result", type=Path, required=True)
    parser.add_argument("--primary-result-sha256", required=True)
    parser.add_argument("--controller-sha256", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    """Authenticate every input and execute one create-only audit."""
    args = parse_args()
    controller = Path(__file__).resolve()
    observed_controller = hashlib.sha256(controller.read_bytes()).hexdigest()
    if observed_controller != args.controller_sha256:
        raise ValueError("independent CP controller hash differs")
    registration = strict_json(
        read_verified(args.registration, args.registration_sha256, "registration"),
        "registration",
    )
    primary_result = strict_json(
        read_verified(
            args.primary_result,
            args.primary_result_sha256,
            "primary result",
        ),
        "primary result",
    )
    receipt = execute_independent_cp_audit(
        registration=registration,
        registration_sha256=args.registration_sha256,
        primary_result=primary_result,
        primary_result_file_sha256=args.primary_result_sha256,
        controller_sha256=args.controller_sha256,
        output_directory=args.output_directory,
    )
    print(
        json.dumps(
            {
                "terminal_status": receipt["terminal_status"],
                "output_directory": str(args.output_directory.resolve()),
                "split_gate0_authorized": False,
                "training_authorized": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
