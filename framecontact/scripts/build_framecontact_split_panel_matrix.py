#!/usr/bin/env python3
"""Build split-bound partner metadata and all reciprocal partner panels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from interaction_first.panel_matrix import build_split_panel_matrix  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partner-metadata-base", type=Path, required=True)
    parser.add_argument("--partner-metadata-base-sha256", required=True)
    parser.add_argument("--partner-metadata-base-receipt", type=Path, required=True)
    parser.add_argument("--partner-metadata-base-receipt-sha256", required=True)
    parser.add_argument("--split-matrix-root", type=Path, required=True)
    parser.add_argument("--split-matrix-receipt-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--panel-size-per-direction", type=int, default=31)
    parser.add_argument("--protocol-seed", type=int, default=20_260_730)
    arguments = parser.parse_args()
    receipt = build_split_panel_matrix(
        partner_metadata_base_path=arguments.partner_metadata_base,
        partner_metadata_base_sha256=arguments.partner_metadata_base_sha256,
        partner_metadata_base_receipt_path=arguments.partner_metadata_base_receipt,
        partner_metadata_base_receipt_sha256=(
            arguments.partner_metadata_base_receipt_sha256
        ),
        split_matrix_root=arguments.split_matrix_root,
        split_matrix_receipt_sha256=arguments.split_matrix_receipt_sha256,
        output_root=arguments.output_root,
        panel_size_per_direction=arguments.panel_size_per_direction,
        protocol_seed=arguments.protocol_seed,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
