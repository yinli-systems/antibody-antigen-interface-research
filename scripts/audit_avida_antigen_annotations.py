#!/usr/bin/env python3
"""Independently audit AVIDa IL-6 boundary annotations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from interaction_first.avida_antigen_independent_auditor import (
    audit_avida_antigen_annotations,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--annotation-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--uniprot-json", type=Path, required=True)
    parser.add_argument("--annotator-source", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    arguments = parser.parse_args()
    receipt = audit_avida_antigen_annotations(
        candidate_root=arguments.candidate_root,
        annotation_root=arguments.annotation_root,
        protocol_path=arguments.protocol,
        uniprot_json=arguments.uniprot_json,
        annotator_source=arguments.annotator_source,
        output_receipt=arguments.output_receipt,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
