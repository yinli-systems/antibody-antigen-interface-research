#!/usr/bin/env python3
"""Independently rebuild and audit one V6-A score-free authority bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from interaction_first.score_free_independent_auditor import (
    VIEWS,
    audit_score_free_authority_candidate,
    authority_audit_markdown,
    canonical_json_bytes,
)


@dataclass(frozen=True)
class VerifiedBytes:
    """Authenticated bytes and the descriptor receipt for one input."""

    payload: bytes
    reference: dict[str, object]


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.path.expanduser(str(path))))


def _reject_symlink_components(path: Path, *, include_final: bool) -> None:
    absolute = _absolute(path)
    stop = len(absolute.parts) if include_final else len(absolute.parts) - 1
    current = Path(absolute.parts[0])
    for part in absolute.parts[1:stop]:
        current /= part
        metadata = os.lstat(current)
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"symlink path component is forbidden: {current}")


def _validate_sha256(value: str, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or not value.isascii()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} SHA-256 is malformed")


def read_verified(path: Path, expected_sha256: str, label: str) -> VerifiedBytes:
    """Hash and consume the same no-follow regular-file descriptor."""
    _validate_sha256(expected_sha256, label)
    absolute = _absolute(path)
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
    fingerprint = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
    )
    if fingerprint(before) != fingerprint(after) or len(payload) != before.st_size:
        raise ValueError(f"{label} changed during authentication")
    observed = hashlib.sha256(payload).hexdigest()
    if observed != expected_sha256:
        raise ValueError(f"{label} SHA-256 differs from the external anchor")
    return VerifiedBytes(
        payload=payload,
        reference={
            "path": str(absolute),
            "sha256": observed,
            "byte_size": len(payload),
            "device": before.st_dev,
            "inode": before.st_ino,
            "mtime_ns": before.st_mtime_ns,
            "nofollow_same_descriptor_read": True,
        },
    )


def _write_exclusive(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o400)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def create_only_report_directory(
    output_directory: Path,
    receipt: dict[str, object],
) -> None:
    """Create one immutable directory containing JSON and derived Markdown."""
    absolute = _absolute(output_directory)
    _reject_symlink_components(absolute, include_final=False)
    if absolute.exists() or absolute.is_symlink():
        raise FileExistsError(f"create-only output already exists: {absolute}")
    os.mkdir(absolute, 0o700)
    try:
        _write_exclusive(absolute / "audit_receipt.json", canonical_json_bytes(receipt))
        markdown = authority_audit_markdown(receipt).encode("ascii")
        _write_exclusive(absolute / "audit_report.md", markdown)
        os.chmod(absolute / "audit_receipt.json", 0o400)
        os.chmod(absolute / "audit_report.md", 0o400)
        directory_descriptor = os.open(absolute, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        os.chmod(absolute, 0o500)
    except BaseException:
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--train-authority", type=Path, required=True)
    parser.add_argument("--train-authority-sha256", required=True)
    parser.add_argument("--canonical-manifest", type=Path, required=True)
    parser.add_argument("--canonical-manifest-sha256", required=True)
    parser.add_argument("--primary-authority-source-sha256", required=True)
    parser.add_argument("--primary-statistics-source-sha256", required=True)
    for view in VIEWS:
        option = view.replace("_", "-")
        parser.add_argument(f"--edge-ledger-{option}", type=Path, required=True)
        parser.add_argument(f"--edge-ledger-{option}-sha256", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, object]:
    candidate = read_verified(args.candidate, args.candidate_sha256, "candidate")
    authority = read_verified(
        args.train_authority,
        args.train_authority_sha256,
        "train authority",
    )
    manifest = read_verified(
        args.canonical_manifest,
        args.canonical_manifest_sha256,
        "canonical manifest",
    )
    ledgers: dict[str, bytes] = {}
    references: dict[str, object] = {
        "candidate": candidate.reference,
        "train_authority": authority.reference,
        "canonical_manifest": manifest.reference,
    }
    for view in VIEWS:
        verified = read_verified(
            getattr(args, f"edge_ledger_{view}"),
            getattr(args, f"edge_ledger_{view}_sha256"),
            f"{view} edge ledger",
        )
        ledgers[view] = verified.payload
        references[f"edge_ledger_{view}"] = verified.reference
    receipt = audit_score_free_authority_candidate(
        candidate_bytes=candidate.payload,
        authority_bytes=authority.payload,
        manifest_bytes=manifest.payload,
        edge_ledger_bytes=ledgers,
        expected_primary_implementation_sha256={
            "score_free_authority": args.primary_authority_source_sha256,
            "score_free_split_statistics": args.primary_statistics_source_sha256,
        },
    )
    receipt.pop("receipt_sha256")
    receipt["verified_file_receipts"] = references
    receipt["receipt_sha256"] = hashlib.sha256(
        canonical_json_bytes(receipt)
    ).hexdigest()
    create_only_report_directory(args.output_directory, receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    receipt = run(args)
    print(
        json.dumps(
            {
                "output_directory": str(_absolute(args.output_directory)),
                "receipt_sha256": receipt["receipt_sha256"],
                "passed": receipt["passed"],
                "training_authorized": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
