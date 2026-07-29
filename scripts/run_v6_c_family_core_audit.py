#!/usr/bin/env python3
"""Run one registered create-only V6-C family-core audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from interaction_first.v6_c_family_core_audit import (
    canonical_json_bytes,
    execute_family_core_audit,
)
from interaction_first.v6_c_family_core_registration import (
    validate_family_core_registration,
)


def parse_args() -> argparse.Namespace:
    """Parse only content-addressed execution inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--registration-sha256", required=True)
    parser.add_argument("--controller-sha256", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser.parse_args()


def _strict_registration(payload: bytes) -> dict[str, object]:
    def reject_duplicates(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("V6-C registration contains duplicate keys")
            result[key] = value
        return result

    try:
        parsed = json.loads(
            payload.decode("ascii"),
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(
            "V6-C registration is not canonical ASCII JSON"
        ) from error
    if not isinstance(parsed, dict):
        raise ValueError("V6-C registration is not a JSON object")
    if canonical_json_bytes(parsed) != payload:
        raise ValueError("V6-C registration is not canonically encoded")
    return parsed


def main() -> int:
    """Authenticate the controller and execute one frozen registration."""
    arguments = parse_args()
    controller = Path(__file__).resolve(strict=True)
    observed_controller_sha256 = hashlib.sha256(
        controller.read_bytes()
    ).hexdigest()
    if observed_controller_sha256 != arguments.controller_sha256:
        raise ValueError("V6-C controller hash differs")
    registration_bytes = arguments.registration.read_bytes()
    if (
        hashlib.sha256(registration_bytes).hexdigest()
        != arguments.registration_sha256
    ):
        raise ValueError("V6-C registration file hash differs")
    registration = validate_family_core_registration(
        _strict_registration(registration_bytes)
    )
    receipt = execute_family_core_audit(
        registration=registration,
        registration_sha256=arguments.registration_sha256,
        controller_path=controller,
        controller_sha256=observed_controller_sha256,
        output_directory=arguments.output_directory,
    )
    print(
        json.dumps(
            {
                "full_protocol_infeasibility_claim_authorized": False,
                "output_directory": str(
                    arguments.output_directory.resolve()
                ),
                "split_gate0_authorized": False,
                "terminal_status": receipt["terminal_status"],
                "test_access_authorized": False,
                "threshold_change_authorized": False,
                "training_authorized": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
