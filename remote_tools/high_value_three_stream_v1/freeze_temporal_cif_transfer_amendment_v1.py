#!/usr/bin/env python3
"""Bind the case-sensitive correction of one temporal cropped mmCIF."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


INSTANCE = "pdb_000026wc-H-L"


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


def atomic_write(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def chain_ids(path: Path) -> list[str]:
    from Bio.PDB import MMCIFParser

    structure = MMCIFParser(QUIET=True).get_structure(INSTANCE, path)
    model = next(structure.get_models())
    return [chain.id for chain in model]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-manifest", type=Path, required=True)
    parser.add_argument("--original-manifest-sha256", required=True)
    parser.add_argument("--identity-manifest", type=Path, required=True)
    parser.add_argument("--identity-manifest-sha256", required=True)
    parser.add_argument("--superseded-file", type=Path, required=True)
    parser.add_argument("--corrected-file", type=Path, required=True)
    parser.add_argument("--failed-stdout", type=Path, required=True)
    parser.add_argument("--failed-stderr", type=Path, required=True)
    parser.add_argument("--failed-job-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.output, args.receipt):
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
    if sha256_file(args.original_manifest) != args.original_manifest_sha256:
        raise ValueError("original transfer manifest SHA-256 differs")
    if sha256_file(args.identity_manifest) != args.identity_manifest_sha256:
        raise ValueError("identity manifest SHA-256 differs")
    original = json.loads(args.original_manifest.read_text(encoding="utf-8"))
    identity = json.loads(args.identity_manifest.read_text(encoding="utf-8"))
    source_record = next(
        row
        for row in original["records"]
        if row.get("instance_id", row.get("instance")) == INSTANCE
    )
    identity_record = next(
        row for row in identity["records"] if row.get("instance") == INSTANCE
    )
    superseded_sha = sha256_file(args.superseded_file)
    corrected_sha = sha256_file(args.corrected_file)
    if (
        original.get("status") != "PASS_LABEL_FREE_NETWORK_INPUTS_FROZEN"
        or original.get("audit", {}).get("task_labels_read") is not False
        or source_record.get("sha256") != superseded_sha
        or superseded_sha == corrected_sha
        or identity_record.get("hchain") != "H"
        or identity_record.get("lchain") != "L"
        or identity_record.get("antigen_chain") != "A"
    ):
        raise ValueError("case-collision recovery inputs differ")
    wrong_chains = chain_ids(args.superseded_file)
    corrected_chains = chain_ids(args.corrected_file)
    if not {"h", "l", "a"}.issubset(wrong_chains) or not {
        "H",
        "L",
        "A",
    }.issubset(corrected_chains):
        raise ValueError("case-sensitive chain evidence differs")
    amendment = {
        "format_version": "framecontact_temporal_cif_transfer_amendment_v1",
        "status": "PASS_CASE_SENSITIVE_TRANSFER_CORRECTION_FROZEN",
        "original_manifest_sha256": args.original_manifest_sha256,
        "identity_manifest_sha256": args.identity_manifest_sha256,
        "instance": INSTANCE,
        "failure": {
            "job_id": args.failed_job_id,
            "state": "FAILED",
            "exit_code": "1:0",
            "stdout_sha256": sha256_file(args.failed_stdout),
            "stderr_sha256": sha256_file(args.failed_stderr),
            "cause": "case-insensitive local staging aliased H/L and h/l instance filenames",
        },
        "superseded": {
            "path": str(args.superseded_file),
            "sha256": superseded_sha,
            "observed_chain_ids": wrong_chains,
        },
        "corrected": {
            "path": str(args.corrected_file),
            "sha256": corrected_sha,
            "bytes": args.corrected_file.stat().st_size,
            "observed_chain_ids": corrected_chains,
        },
        "audit": {
            "identity_h_l_a_chains_present_in_corrected_file": True,
            "task_labels_read": False,
            "cross_chain_distances_computed": False,
            "contact_labels_computed": False,
            "selection_performed": False,
            "temporal_opening_count": 0,
        },
    }
    atomic_write(args.output, canonical_bytes(amendment))
    receipt = {
        "format_version": "framecontact_temporal_cif_transfer_amendment_receipt_v1",
        "status": amendment["status"],
        "amendment_sha256": sha256_file(args.output),
        "corrected_cif_sha256": corrected_sha,
        "failed_job_id": args.failed_job_id,
        "task_labels_read": False,
        "temporal_opening_count": 0,
    }
    atomic_write(args.receipt, canonical_bytes(receipt))
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
