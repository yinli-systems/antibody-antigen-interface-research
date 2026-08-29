#!/usr/bin/env python3
"""Audit a frozen FrameContact train/validation/test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from interaction_first.structural_novelty import (
    audit_antigen_structure_split,
    write_structural_audit,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--alignments", type=Path, required=True)
    parser.add_argument("--alignments-sha256", required=True)
    parser.add_argument("--foldseek-receipt", type=Path, required=True)
    parser.add_argument("--foldseek-receipt-sha256", required=True)
    parser.add_argument("--tm-threshold", type=float, default=0.50)
    parser.add_argument(
        "--foldseek-score-rule",
        choices=("reciprocal_min", "geometric_mean", "reciprocal_max"),
        default="reciprocal_max",
    )
    parser.add_argument("--boundary-width", type=float, default=0.02)
    parser.add_argument("--boundary-pair-limit", type=int, default=1_000)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    payload = audit_antigen_structure_split(
        manifest_path=arguments.manifest,
        manifest_sha256=arguments.manifest_sha256,
        alignments_path=arguments.alignments,
        alignments_sha256=arguments.alignments_sha256,
        foldseek_receipt_path=arguments.foldseek_receipt,
        foldseek_receipt_sha256=arguments.foldseek_receipt_sha256,
        tm_threshold=arguments.tm_threshold,
        foldseek_score_rule=arguments.foldseek_score_rule,
        boundary_width=arguments.boundary_width,
        boundary_pair_limit=arguments.boundary_pair_limit,
    )
    write_structural_audit(payload, arguments.output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
