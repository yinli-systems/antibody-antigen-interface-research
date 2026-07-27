#!/usr/bin/env python3
"""Materialize one pinned PAIRScale source into a content-addressed inventory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from interaction_first.source_materializer import materialize_registered_source


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--registry-root", type=Path, required=True)
    parser.add_argument("--destination-root", type=Path, required=True)
    parser.add_argument("--curl-binary", default="curl")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--verified-seed-root", type=Path)
    arguments = parser.parse_args()
    inventory = materialize_registered_source(
        plan_path=arguments.plan,
        registry_root=arguments.registry_root,
        destination_root=arguments.destination_root,
        curl_binary=arguments.curl_binary,
        workers=arguments.workers,
        verified_seed_root=arguments.verified_seed_root,
    )
    print(json.dumps(inventory, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
