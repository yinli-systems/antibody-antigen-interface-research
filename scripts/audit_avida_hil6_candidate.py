#!/usr/bin/env python3
"""Independently verify AVIDa raw-to-candidate normalization semantics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from interaction_first.avida_hil6_independent_auditor import (
    audit_avida_hil6_candidate,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--registered-raw-inventory", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--normalizer-source", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    arguments = parser.parse_args()
    result = audit_avida_hil6_candidate(
        raw_root=arguments.raw_root,
        registered_raw_inventory=arguments.registered_raw_inventory,
        contract_path=arguments.contract,
        candidate_root=arguments.candidate_root,
        normalizer_source_path=arguments.normalizer_source,
        output_receipt=arguments.output_receipt,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
