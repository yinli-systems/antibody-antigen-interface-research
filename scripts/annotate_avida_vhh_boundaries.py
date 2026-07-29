#!/usr/bin/env python3
"""Annotate AVIDa VHH boundaries using the frozen ANARCII protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from interaction_first.avida_vhh_annotation import annotate_avida_vhh_boundaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--anarcii-wheel", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--ncpu", type=int, default=1)
    arguments = parser.parse_args()
    receipt = annotate_avida_vhh_boundaries(
        candidate_root=arguments.candidate_root,
        protocol_path=arguments.protocol,
        anarcii_wheel=arguments.anarcii_wheel,
        output_root=arguments.output_root,
        batch_size=arguments.batch_size,
        chunk_size=arguments.chunk_size,
        ncpu=arguments.ncpu,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
