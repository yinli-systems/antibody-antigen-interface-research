#!/usr/bin/env python3
"""Run one registered full-authority V6-B production feasibility attempt."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from interaction_first.v6_b_production import (
    execute_registered_feasibility,
    install_termination_handlers,
    read_verified,
    strict_json,
)


def parse_args() -> argparse.Namespace:
    """Parse explicit content anchors without environment-derived defaults."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--registration-sha256", required=True)
    parser.add_argument("--controller-sha256", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    """Authenticate the registration and execute one create-only attempt."""
    args = parse_args()
    controller_path = Path(__file__).resolve()
    observed_controller = hashlib.sha256(controller_path.read_bytes()).hexdigest()
    if observed_controller != args.controller_sha256:
        raise ValueError("production controller source hash differs")
    registration_bytes = read_verified(
        args.registration,
        args.registration_sha256,
        "production registration",
    )
    registration = strict_json(registration_bytes, "production registration")
    install_termination_handlers()
    result = execute_registered_feasibility(
        registration=registration,
        registration_sha256=args.registration_sha256,
        controller_path=controller_path,
        controller_sha256=args.controller_sha256,
        output_directory=args.output_directory,
    )
    print(
        json.dumps(
            {
                "output_directory": str(args.output_directory.resolve()),
                "terminal_status": result["terminal_status"],
                "split_gate0_authorized": False,
                "training_authorized": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

