#!/usr/bin/env python3
"""Verify a PAIRScale materialization and write an external receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from interaction_first.source_materializer import (
    verify_materialized_registered_source,
    write_create_only_receipt,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--registry-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    arguments = parser.parse_args()
    receipt = verify_materialized_registered_source(
        plan_path=arguments.plan,
        registry_root=arguments.registry_root,
        source_root=arguments.source_root,
    )
    write_create_only_receipt(arguments.receipt, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
