#!/usr/bin/env python3
"""Publish one authenticated global manifest for filtered OAS shards."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from antibody_pretraining.decontaminated_corpus import (
    canonical_json_bytes,
    finalize_corpus,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--expected-protocol-file-sha256", required=True)
    parser.add_argument("--finalizer-source-sha256", required=True)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    receipt = finalize_corpus(
        output_root=arguments.output_root,
        protocol_path=arguments.protocol,
        expected_protocol_file_sha256=arguments.expected_protocol_file_sha256,
        finalizer_source_path=Path(__file__),
        finalizer_source_sha256=arguments.finalizer_source_sha256,
    )
    sys.stdout.buffer.write(canonical_json_bytes(receipt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
