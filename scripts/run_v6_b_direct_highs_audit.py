#!/usr/bin/env python3
"""Run one registered isolated Direct-HiGHS audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from interaction_first.v6_b_direct_highs_audit import (
    execute_direct_highs_audit,
)
from interaction_first.v6_b_direct_highs_audit import (
    validate_direct_highs_registration,
)
from interaction_first.v6_b_direct_highs_audit import canonical_json_bytes


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--registration-sha256", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    """Authenticate one canonical registration and execute its audit."""
    arguments = _arguments()
    payload = arguments.registration.read_bytes()
    observed_sha256 = hashlib.sha256(payload).hexdigest()
    if observed_sha256 != arguments.registration_sha256:
        raise ValueError("direct-HiGHS registration file hash differs")
    registration = json.loads(payload.decode("ascii"))
    if canonical_json_bytes(registration) != payload:
        raise ValueError("direct-HiGHS registration is not canonical")
    checked = validate_direct_highs_registration(registration)
    controller = Path(__file__).resolve()
    receipt = execute_direct_highs_audit(
        registration=checked,
        registration_sha256=arguments.registration_sha256,
        controller_sha256=hashlib.sha256(controller.read_bytes()).hexdigest(),
        output_directory=arguments.output_directory,
    )
    print(
        json.dumps(
            {
                "terminal_status": receipt["terminal_status"],
                "cross_solver_linear_agreement": receipt[
                    "cross_solver_linear_agreement"
                ],
                "output_directory": str(arguments.output_directory.resolve()),
                "split_gate0_authorized": False,
                "training_authorized": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
