#!/usr/bin/env python3
"""Audit SAbDab2 overlap in the label-free AsEP official test inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from interaction_first.asep_decontamination import (  # noqa: E402
    audit_asep_sabdab2_decontamination,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-free-test-root", type=Path, required=True)
    parser.add_argument("--label-free-test-receipt-sha256", required=True)
    parser.add_argument("--sabdab2-split-manifest", type=Path, required=True)
    parser.add_argument("--sabdab2-split-manifest-sha256", required=True)
    parser.add_argument("--sabdab2-antigen-export-manifest", type=Path, required=True)
    parser.add_argument("--sabdab2-antigen-export-manifest-sha256", required=True)
    parser.add_argument("--cross-alignments", type=Path, required=True)
    parser.add_argument("--cross-alignments-sha256", required=True)
    parser.add_argument("--cross-foldseek-receipt", type=Path, required=True)
    parser.add_argument("--cross-foldseek-receipt-sha256", required=True)
    parser.add_argument("--tm-threshold", type=float, default=0.50)
    parser.add_argument(
        "--foldseek-score-rule",
        choices=("reciprocal_min", "geometric_mean", "reciprocal_max"),
        default="reciprocal_min",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args()
    result = audit_asep_sabdab2_decontamination(
        label_free_test_root=arguments.label_free_test_root,
        label_free_test_receipt_sha256=arguments.label_free_test_receipt_sha256,
        sabdab2_split_manifest=arguments.sabdab2_split_manifest,
        sabdab2_split_manifest_sha256=arguments.sabdab2_split_manifest_sha256,
        sabdab2_antigen_export_manifest=arguments.sabdab2_antigen_export_manifest,
        sabdab2_antigen_export_manifest_sha256=(
            arguments.sabdab2_antigen_export_manifest_sha256
        ),
        cross_alignments=arguments.cross_alignments,
        cross_alignments_sha256=arguments.cross_alignments_sha256,
        cross_foldseek_receipt=arguments.cross_foldseek_receipt,
        cross_foldseek_receipt_sha256=arguments.cross_foldseek_receipt_sha256,
        output_root=arguments.output_root,
        tm_threshold=arguments.tm_threshold,
        foldseek_score_rule=arguments.foldseek_score_rule,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
