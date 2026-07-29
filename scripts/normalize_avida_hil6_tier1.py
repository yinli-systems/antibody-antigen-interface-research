#!/usr/bin/env python3
"""Materialize the frozen AVIDa Tier-1 normalization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from interaction_first.avida_hil6_normalization import normalize_avida_hil6_tier1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--vhh-annotation-root", type=Path, required=True)
    parser.add_argument("--antigen-annotation-root", type=Path, required=True)
    parser.add_argument("--vhh-independent-receipt", type=Path, required=True)
    parser.add_argument("--antigen-independent-receipt", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = normalize_avida_hil6_tier1(
        candidate_root=args.candidate_root,
        vhh_annotation_root=args.vhh_annotation_root,
        antigen_annotation_root=args.antigen_annotation_root,
        vhh_independent_receipt=args.vhh_independent_receipt,
        antigen_independent_receipt=args.antigen_independent_receipt,
        protocol_path=args.protocol,
        output_root=args.output_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
