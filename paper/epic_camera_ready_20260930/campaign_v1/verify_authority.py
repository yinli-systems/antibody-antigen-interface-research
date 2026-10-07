#!/usr/bin/env python3
"""Verify every frozen parent-bundle artifact once and emit an exclusive receipt."""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import time
from typing import Any


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-protocol", required=True, type=Path)
    parser.add_argument("--campaign-protocol-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    protocol_path = args.campaign_protocol.resolve()
    require(sha(protocol_path) == args.campaign_protocol_sha256, "Campaign protocol hash differs")
    campaign = json.loads(protocol_path.read_text())
    require(campaign["status"] == "PREREGISTERED_BEFORE_GPU_EXECUTION", "Campaign not preregistered")
    require(sha(Path(__file__).resolve()) == campaign["implementation"]["verify_authority.py"],
            "Authority verifier differs from preregistered implementation")
    parent = Path(campaign["parent_authority"]["root"]).resolve()
    require(parent.is_dir() and not parent.is_symlink(), "Parent root invalid")
    expected = campaign["parent_authority"]
    fixed = {
        "BUNDLE.json": expected["bundle_json_sha256"],
        "BUILD_RECEIPT.json": expected["build_receipt_sha256"],
        "PROTOCOL.json": expected["protocol_sha256"],
        "authority/FULL_SPLIT.csv": expected["full_split_sha256"],
        "development/structure/STRUCTURE_CACHE_MANIFEST.json": expected["structure_manifest_sha256"],
        "development/contact/CONTACT_CACHE_MANIFEST.json": expected["contact_manifest_sha256"],
        "source/scripts/train_framecontact.py": expected["train_source_sha256"],
        "source/src/interaction_first/framecontact.py": expected["model_source_sha256"],
        "source/src/interaction_first/framecontact_objective.py": expected["objective_source_sha256"],
    }
    for relative, digest in fixed.items():
        require(sha(parent / relative) == digest, f"Fixed parent artifact changed: {relative}")

    inventory = json.loads((parent / "BUNDLE.json").read_text())["files"]
    started = time.monotonic()
    total_bytes = 0
    for index, (relative, digest) in enumerate(sorted(inventory.items()), 1):
        candidate = parent / relative
        resolved = candidate.resolve()
        require(not Path(relative).is_absolute() and parent in resolved.parents,
                f"Inventory path escapes parent: {relative}")
        require(candidate.is_file() and not candidate.is_symlink(), f"Missing/symlink inventory file: {relative}")
        require(sha(candidate) == digest, f"Inventory hash differs: {relative}")
        total_bytes += candidate.stat().st_size
        if index % 500 == 0 or index == len(inventory):
            print(json.dumps({"verified": index, "total": len(inventory),
                              "bytes": total_bytes, "seconds": time.monotonic() - started},
                             sort_keys=True), flush=True)

    spec = importlib.util.spec_from_file_location("epic_parent_protocol", parent / "runner/protocol.py")
    require(spec is not None and spec.loader is not None, "Cannot load parent protocol helper")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    with (parent / "authority/FULL_SPLIT.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    split_audit = helper.validate_split(rows)
    with (parent / "authority/DEVELOPMENT_SPLIT.csv").open(newline="") as handle:
        development = list(csv.DictReader(handle))
    require(development == [row for row in rows if row["role"] != "test"],
            "Development membership differs from frozen full split")

    structure = json.loads((parent / "development/structure/STRUCTURE_CACHE_MANIFEST.json").read_text())
    contact = json.loads((parent / "development/contact/CONTACT_CACHE_MANIFEST.json").read_text())
    require(structure["authority"]["geometry_source"] == "independent_sequence_predicted_af3",
            "Unexpected geometry source")
    require(contact["authority"]["original455_label_artifacts_opened"] is False,
            "Parent development contact cache claims test-label access")

    payload: dict[str, Any] = {
        "status": "PASS_FULL_PARENT_BUNDLE_VERIFICATION",
        "format_version": campaign["format_version"],
        "campaign_protocol_sha256": args.campaign_protocol_sha256,
        "parent_bundle_json_sha256": expected["bundle_json_sha256"],
        "verified_inventory_file_count": len(inventory),
        "verified_inventory_total_bytes": total_bytes,
        "split_audit": split_audit,
        "development_count": len(development),
        "geometry_source": structure["authority"]["geometry_source"],
        "original455_label_artifacts_opened": False,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "wall_seconds": time.monotonic() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print("COMPLETE", args.output, sha(args.output), flush=True)


if __name__ == "__main__":
    main()
