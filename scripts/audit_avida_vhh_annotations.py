#!/usr/bin/env python3
"""Independently replay and audit AVIDa VHH boundary annotations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from interaction_first.avida_vhh_independent_auditor import (
    audit_avida_vhh_annotations,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--annotation-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--anarcii-wheel", type=Path, required=True)
    parser.add_argument("--annotator-source", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--ncpu", type=int, default=4)
    arguments = parser.parse_args()
    receipt = audit_avida_vhh_annotations(
        candidate_root=arguments.candidate_root,
        annotation_root=arguments.annotation_root,
        protocol_path=arguments.protocol,
        anarcii_wheel=arguments.anarcii_wheel,
        annotator_source=arguments.annotator_source,
        output_receipt=arguments.output_receipt,
        batch_size=arguments.batch_size,
        chunk_size=arguments.chunk_size,
        ncpu=arguments.ncpu,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
