#!/usr/bin/env python3
"""Build one create-only V6-A score-free authority bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from interaction_first.score_free_authority import (
    LOCKED_AUTHORITY_COUNTS,
    REQUIRED_VIEWS,
    build_score_free_authority,
    canonical_json_bytes,
)


@dataclass(frozen=True)
class VerifiedBytes:
    """Bytes consumed from the same regular-file descriptor that was hashed."""

    payload: bytes
    reference: dict[str, object]


def _absolute_without_resolving(path: Path) -> Path:
    return Path(os.path.abspath(os.path.expanduser(str(path))))


def _reject_symlink_components(path: Path, *, include_final: bool) -> None:
    absolute = _absolute_without_resolving(path)
    parts = absolute.parts
    stop = len(parts) if include_final else len(parts) - 1
    current = Path(parts[0])
    for part in parts[1:stop]:
        current /= part
        metadata = os.lstat(current)
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"symlink path component is forbidden: {current}")


def read_verified_regular_file(
    path: Path,
    expected_sha256: str,
    label: str,
) -> VerifiedBytes:
    """Read and authenticate one no-follow regular file exactly once."""
    if (
        len(expected_sha256) != 64
        or not expected_sha256.isascii()
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise ValueError(f"{label} expected SHA-256 is malformed")
    absolute = _absolute_without_resolving(path)
    _reject_symlink_components(absolute, include_final=True)
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(absolute, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} is not a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        payload = b"".join(chunks)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    fingerprint_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    fingerprint_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if fingerprint_before != fingerprint_after or len(payload) != before.st_size:
        raise ValueError(f"{label} changed while it was being authenticated")
    observed_sha256 = hashlib.sha256(payload).hexdigest()
    if observed_sha256 != expected_sha256:
        raise ValueError(f"{label} SHA-256 differs from the external anchor")
    return VerifiedBytes(
        payload=payload,
        reference={
            "path": str(absolute),
            "byte_size": len(payload),
            "sha256": observed_sha256,
            "device": before.st_dev,
            "inode": before.st_ino,
            "mtime_ns": before.st_mtime_ns,
            "nofollow_same_descriptor_read": True,
        },
    )


def create_only_json(path: Path, payload: object) -> None:
    """Publish canonical JSON atomically without replacing an existing path."""
    absolute = _absolute_without_resolving(path)
    parent_existed = absolute.parent.exists()
    _reject_symlink_components(absolute.parent, include_final=False)
    absolute.parent.mkdir(parents=False, exist_ok=True)
    _reject_symlink_components(absolute.parent, include_final=True)
    if absolute.exists() or absolute.is_symlink():
        raise FileExistsError(f"create-only output already exists: {absolute}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{absolute.name}.",
        suffix=".tmp",
        dir=absolute.parent,
    )
    temporary = Path(temporary_name)
    try:
        encoded = canonical_json_bytes(payload)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, absolute)
        os.chmod(absolute, 0o400)
        directory_descriptor = os.open(absolute.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    if not parent_existed:
        os.chmod(absolute.parent, 0o500)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-authority", type=Path, required=True)
    parser.add_argument("--train-authority-sha256", required=True)
    parser.add_argument("--canonical-manifest", type=Path, required=True)
    parser.add_argument("--canonical-manifest-sha256", required=True)
    for view in REQUIRED_VIEWS:
        option = view.replace("_", "-")
        parser.add_argument(f"--edge-ledger-{option}", type=Path, required=True)
        parser.add_argument(f"--edge-ledger-{option}-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, object]:
    authority = read_verified_regular_file(
        args.train_authority,
        args.train_authority_sha256,
        "physical train-only authority",
    )
    manifest = read_verified_regular_file(
        args.canonical_manifest,
        args.canonical_manifest_sha256,
        "physical train-only canonical manifest",
    )
    ledgers: dict[str, bytes] = {}
    verified_inputs: dict[str, object] = {
        "train_authority": authority.reference,
        "canonical_manifest": manifest.reference,
    }
    for view in REQUIRED_VIEWS:
        attribute = view.replace("-", "_")
        verified = read_verified_regular_file(
            getattr(args, f"edge_ledger_{attribute}"),
            getattr(args, f"edge_ledger_{attribute}_sha256"),
            f"{view} direct-edge ledger",
        )
        ledgers[view] = verified.payload
        verified_inputs[f"edge_ledger_{view}"] = verified.reference
    bundle = build_score_free_authority(
        authority_bytes=authority.payload,
        manifest_bytes=manifest.payload,
        edge_ledger_bytes=ledgers,
        expected_counts=LOCKED_AUTHORITY_COUNTS,
    )
    payload = bundle.canonical_payload()
    payload["verified_file_receipts"] = verified_inputs
    payload["publication_controller_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    payload["publication_sha256"] = hashlib.sha256(
        canonical_json_bytes(payload)
    ).hexdigest()
    create_only_json(args.output, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = run(args)
    print(
        json.dumps(
            {
                "output": str(_absolute_without_resolving(args.output)),
                "scientific_payload_sha256": payload[
                    "scientific_payload_sha256"
                ],
                "publication_sha256": payload["publication_sha256"],
                "training_authorized": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
