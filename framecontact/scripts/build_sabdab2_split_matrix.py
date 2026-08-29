#!/usr/bin/env python3
"""Build frozen original, temporal, and Foldseek antigen-fold manifests."""

from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from interaction_first.sabdab2_split_matrix import build_split_matrix


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--metadata-sha256", required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--prepared-sha256", required=True)
    parser.add_argument("--foldseek-alignments", type=Path, required=True)
    parser.add_argument("--foldseek-alignments-sha256", required=True)
    parser.add_argument("--foldseek-receipt", type=Path, required=True)
    parser.add_argument("--foldseek-receipt-sha256", required=True)
    parser.add_argument("--train-cutoff", type=date.fromisoformat, default=date(2022, 12, 31))
    parser.add_argument(
        "--validation-cutoff", type=date.fromisoformat, default=date(2024, 6, 30)
    )
    parser.add_argument("--tm-threshold", type=float, default=0.50)
    parser.add_argument(
        "--foldseek-score-rule",
        choices=("reciprocal_min", "geometric_mean", "reciprocal_max"),
        default="reciprocal_max",
    )
    parser.add_argument("--assignment-seed", type=int, default=20_260_730)
    parser.add_argument(
        "--metadata-cluster-policy",
        choices=("ignore", "antigen", "double_cold"),
        default="ignore",
        help=(
            "Optional metadata-cluster isolation added to Foldseek/exact-sequence "
            "components. Existing behavior is ignore."
        ),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args()
    result = build_split_matrix(
        metadata_path=arguments.metadata,
        metadata_sha256=arguments.metadata_sha256,
        prepared_path=arguments.prepared,
        prepared_sha256=arguments.prepared_sha256,
        foldseek_alignments_path=arguments.foldseek_alignments,
        foldseek_alignments_sha256=arguments.foldseek_alignments_sha256,
        foldseek_receipt_path=arguments.foldseek_receipt,
        foldseek_receipt_sha256=arguments.foldseek_receipt_sha256,
        output_root=arguments.output_root,
        train_cutoff=arguments.train_cutoff,
        validation_cutoff=arguments.validation_cutoff,
        tm_threshold=arguments.tm_threshold,
        foldseek_score_rule=arguments.foldseek_score_rule,
        assignment_seed=arguments.assignment_seed,
        metadata_cluster_policy=arguments.metadata_cluster_policy,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
