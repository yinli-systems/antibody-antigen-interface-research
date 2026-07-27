#!/usr/bin/env python3
"""Annotate AVIDa IL-6 boundaries from pinned UniProt authority bytes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from interaction_first.avida_antigen_annotation import (
    annotate_avida_antigen_boundaries,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--uniprot-json", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args()
    receipt = annotate_avida_antigen_boundaries(
        candidate_root=arguments.candidate_root,
        protocol_path=arguments.protocol,
        uniprot_json=arguments.uniprot_json,
        output_root=arguments.output_root,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
