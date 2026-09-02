#!/usr/bin/env python3
"""Canonicalize the frozen all-versus-all antibody-CDR MMseqs2 relation."""

from __future__ import annotations

import argparse
from pathlib import Path

from finalize_pinder_antibody_s2_mmseqs_v1 import run


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--fasta-sha256", required=True)
    parser.add_argument("--raw-tsv", type=Path, required=True)
    parser.add_argument("--mmseqs", type=Path, required=True)
    parser.add_argument("--mmseqs-sha256", required=True)
    parser.add_argument("--mmseqs-version", required=True)
    parser.add_argument("--identity-threshold", type=float, default=0.4)
    parser.add_argument("--coverage-threshold", type=float, default=0.8)
    parser.add_argument("--output-tsv", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.role = "antibody_cdr"
    return arguments


if __name__ == "__main__":
    run(parse_arguments())
