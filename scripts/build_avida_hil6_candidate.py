#!/usr/bin/env python3
"""Create blocked PAIRBench candidate tables from authenticated AVIDa bytes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from interaction_first.avida_hil6_candidate import build_avida_hil6_candidate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--registry-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args()
    receipt = build_avida_hil6_candidate(
        raw_root=arguments.raw_root,
        registry_root=arguments.registry_root,
        contract_path=arguments.contract,
        output_root=arguments.output_root,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
