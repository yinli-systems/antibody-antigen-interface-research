#!/usr/bin/env python3
"""Materialize one authenticated modulo slice of decontaminated OAS buckets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from antibody_pretraining.decontaminated_corpus import filter_task


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--task-count", type=int, default=32)
    parser.add_argument("--bucket-count", type=int, default=256)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--exclusion-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--expected-protocol-file-sha256", required=True)
    parser.add_argument("--controller-source-sha256", required=True)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    receipt = filter_task(
        task_id=arguments.task_id,
        task_count=arguments.task_count,
        bucket_count=arguments.bucket_count,
        source_root=arguments.source_root,
        exclusion_root=arguments.exclusion_root,
        output_root=arguments.output_root,
        protocol_path=arguments.protocol,
        expected_protocol_file_sha256=arguments.expected_protocol_file_sha256,
        controller_source_sha256=arguments.controller_source_sha256,
    )
    print(json.dumps(receipt, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
