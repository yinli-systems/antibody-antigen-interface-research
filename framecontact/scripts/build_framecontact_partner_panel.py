#!/usr/bin/env python3
"""Freeze a deterministic label-free wrong-partner panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from interaction_first.partner_panel import build_partner_panel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--metadata-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--role", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--panel-size", type=int, default=31)
    parser.add_argument("--protocol-seed", type=int, default=20_260_730)
    arguments = parser.parse_args()
    payload = build_partner_panel(
        metadata_path=arguments.metadata,
        metadata_sha256=arguments.metadata_sha256,
        output_path=arguments.output,
        role=arguments.role,
        panel_size=arguments.panel_size,
        protocol_seed=arguments.protocol_seed,
    )
    print(json.dumps({"role": payload["role"], "entries": len(payload["entries"])}, sort_keys=True))


if __name__ == "__main__":
    main()
