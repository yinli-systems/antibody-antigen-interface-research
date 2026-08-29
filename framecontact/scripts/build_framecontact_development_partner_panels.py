#!/usr/bin/env python3
"""Freeze label-free train/validation reciprocal panels and no test panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from interaction_first.development_partner_panels import (  # noqa: E402
    build_development_partner_panel_bundle,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--metadata-sha256", required=True)
    parser.add_argument("--metadata-receipt", type=Path, required=True)
    parser.add_argument("--metadata-receipt-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--panel-size-per-direction", type=int, default=31)
    parser.add_argument("--protocol-seed", type=int, default=20_260_730)
    arguments = parser.parse_args()
    receipt = build_development_partner_panel_bundle(
        metadata_path=arguments.metadata,
        metadata_sha256=arguments.metadata_sha256,
        metadata_receipt_path=arguments.metadata_receipt,
        metadata_receipt_sha256=arguments.metadata_receipt_sha256,
        output_root=arguments.output_root,
        panel_size_per_direction=arguments.panel_size_per_direction,
        protocol_seed=arguments.protocol_seed,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
