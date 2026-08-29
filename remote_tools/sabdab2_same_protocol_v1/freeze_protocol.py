#!/usr/bin/env python3
"""Write a minimal immutable receipt for a development-only protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.receipt.exists():
        raise FileExistsError(args.receipt)
    raw = args.protocol.read_bytes()
    protocol = json.loads(raw)
    if protocol.get("claim_boundary", {}).get("test_loaded") is not False:
        raise ValueError("development protocol must forbid test loading")
    receipt = {
        "format_version": "sabdab2_same_protocol_development_receipt_v1",
        "protocol_file_sha256": hashlib.sha256(raw).hexdigest(),
        "protocol_canonical_sha256": hashlib.sha256(canonical(protocol)).hexdigest(),
        "test_loaded": False,
        "frozen_before_training": True,
    }
    args.receipt.write_bytes(canonical(receipt))
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
