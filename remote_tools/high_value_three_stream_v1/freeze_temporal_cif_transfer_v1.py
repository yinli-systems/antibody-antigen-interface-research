#!/usr/bin/env python3
"""Freeze externally transported, label-free SAbDab2 cropped mmCIF inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


EXPECTED_IDENTITY_SHA256 = "23e6b9848216cdc57ea3210cba89905a659817b6a49a3de9815d9b5153da5303"
EXPECTED_PLAN_SHA256 = "44a184b8e62986ef21abeff188f1014cd763e7cdd421a3177f12693f39000939"


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha(path: Path, expected: str, label: str) -> None:
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"{label} SHA-256 differs: {observed}")


def atomic_write(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(payload)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity-manifest", type=Path, required=True)
    parser.add_argument("--execution-plan", type=Path, required=True)
    parser.add_argument("--cif-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.output, args.receipt):
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
    require_sha(args.identity_manifest, EXPECTED_IDENTITY_SHA256, "identity manifest")
    require_sha(args.execution_plan, EXPECTED_PLAN_SHA256, "execution plan")
    identity = json.loads(args.identity_manifest.read_text())
    records = []
    expected_names = set()
    for row in sorted(identity["records"], key=lambda value: value["instance"]):
        name = str(row["instance"]).replace("/", "_") + ".cif"
        expected_names.add(name)
        path = args.cif_root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"missing regular mmCIF: {name}")
        prefix = path.read_bytes()[:65536]
        if not prefix.startswith((b"data_", b"#")) or b"_atom_site." not in prefix:
            raise ValueError(f"invalid mmCIF payload: {name}")
        records.append(
            {
                "instance_id": str(row["instance"]),
                "pdb_id": str(row["pdb"]),
                "path": str(path),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    observed_names = {path.name for path in args.cif_root.glob("*.cif") if path.is_file()}
    if observed_names != expected_names:
        raise ValueError("cropped mmCIF filename set differs from identity manifest")
    payload = {
        "format_version": "sabdab2_temporal_cropped_cif_transfer_manifest_v1",
        "status": "PASS_LABEL_FREE_NETWORK_INPUTS_FROZEN",
        "identity_manifest_sha256": EXPECTED_IDENTITY_SHA256,
        "execution_plan_sha256": EXPECTED_PLAN_SHA256,
        "source": {
            "api": "https://sabdab.opig.stats.ox.ac.uk/api/download/structures",
            "request_semantics": "cropped declared antibody instance with include_antigen=true",
            "transport": "downloaded on a network-reachable host and copied byte-for-byte to ParaCloud",
        },
        "count": len(records),
        "records": records,
        "audit": {
            "task_labels_read": False,
            "cross_chain_distances_computed": False,
            "contact_labels_computed": False,
            "old_exposed_test_access": False,
        },
    }
    atomic_write(args.output, canonical_bytes(payload))
    receipt = {
        "format_version": "sabdab2_temporal_cropped_cif_transfer_receipt_v1",
        "status": payload["status"],
        "manifest_sha256": sha256_file(args.output),
        "record_count": len(records),
        "task_labels_read": False,
    }
    atomic_write(args.receipt, canonical_bytes(receipt))
    print(json.dumps({**receipt, "receipt_sha256": sha256_file(args.receipt)}, sort_keys=True))


if __name__ == "__main__":
    main()
