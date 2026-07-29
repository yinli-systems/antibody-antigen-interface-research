#!/usr/bin/env python3
"""Independently audit an AVIDa Tier-1 normalization root."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from interaction_first.avida_hil6_normalization_independent_auditor import (
    audit_avida_hil6_normalization,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--vhh-annotation-root", type=Path, required=True)
    parser.add_argument("--antigen-annotation-root", type=Path, required=True)
    parser.add_argument("--normalization-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    args = parser.parse_args()
    result = audit_avida_hil6_normalization(
        candidate_root=args.candidate_root,
        vhh_annotation_root=args.vhh_annotation_root,
        antigen_annotation_root=args.antigen_annotation_root,
        normalization_root=args.normalization_root,
        protocol_path=args.protocol,
        output_receipt=args.output_receipt,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
