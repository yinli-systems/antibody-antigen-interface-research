#!/usr/bin/env python3
"""Run one registered CP-only V6-B linear projection bundle export."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from interaction_first.v6_b_production import read_verified, strict_json
from interaction_first.v6_b_projection_bundle_export import (
    execute_projection_bundle_export,
    validate_export_registration,
)


def parse_args() -> argparse.Namespace:
    """Parse only the export registration and its content hash."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--registration-sha256", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    """Authenticate the registration and perform one create-only export."""
    args = parse_args()
    registration = validate_export_registration(
        strict_json(
            read_verified(
                args.registration,
                args.registration_sha256,
                "bundle export registration",
            ),
            "bundle export registration",
        )
    )
    source = registration["source"]
    controller_reference = source["files"]["controller"]
    controller = Path(__file__).resolve()
    controller_sha256 = hashlib.sha256(controller.read_bytes()).hexdigest()
    if controller_sha256 != controller_reference["sha256"]:
        raise ValueError("bundle export controller hash differs")
    production_reference = registration["production_registration"]
    formal_reference = registration["formal_cp_result"]
    production = strict_json(
        read_verified(
            Path(production_reference["path"]),
            production_reference["sha256"],
            "production registration",
        ),
        "production registration",
    )
    formal = strict_json(
        read_verified(
            Path(formal_reference["path"]),
            formal_reference["sha256"],
            "formal independent CP result",
        ),
        "formal independent CP result",
    )
    receipt = execute_projection_bundle_export(
        export_registration=registration,
        export_registration_sha256=args.registration_sha256,
        production_registration=production,
        production_registration_sha256=production_reference["sha256"],
        formal_cp_result=formal,
        formal_cp_result_file_sha256=formal_reference["sha256"],
        controller_sha256=controller_sha256,
        output_directory=args.output_directory,
    )
    print(
        json.dumps(
            {
                "terminal_status": receipt["terminal_status"],
                "bundle_sha256": receipt["bundle"]["bundle_sha256"],
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
