#!/usr/bin/env python3
"""Freeze label-free wrong-antigen and wrong-antibody panels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from interaction_first.reciprocal_partner_panel import build_reciprocal_partner_panel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--metadata-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--role", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--panel-size-per-direction", type=int, default=31)
    parser.add_argument("--protocol-seed", type=int, default=20_260_730)
    arguments = parser.parse_args()
    payload = build_reciprocal_partner_panel(
        metadata_path=arguments.metadata,
        metadata_sha256=arguments.metadata_sha256,
        output_path=arguments.output,
        role=arguments.role,
        panel_size_per_direction=arguments.panel_size_per_direction,
        protocol_seed=arguments.protocol_seed,
    )
    print(
        json.dumps(
            {
                "role": payload["role"],
                "directions": 2,
                "entries": len(payload["entries"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
