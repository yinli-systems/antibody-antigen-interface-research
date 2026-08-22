#!/usr/bin/env python3
"""Extract only strict-train holo mapping files from an authenticated PINDER ZIP."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
import time
import zipfile
import zlib


HOLO_COLUMNS = ("holo_R_pdb", "holo_L_pdb")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mapping_member(pdb_name: str) -> str:
    path = PurePosixPath(pdb_name)
    if path.name != pdb_name or path.suffix != ".pdb":
        raise ValueError(f"Invalid PINDER holo filename: {pdb_name!r}")
    return f"mappings/{path.stem}.parquet"


def required_members(manifest: Path) -> set[str]:
    members: set[str] = set()
    with gzip.open(manifest, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = set(reader.fieldnames or ())
        if not set(HOLO_COLUMNS).issubset(fields):
            raise RuntimeError("PINDER manifest is missing holo endpoint columns")
        for row in reader:
            for column in HOLO_COLUMNS:
                members.add(mapping_member(row[column]))
    if not members:
        raise RuntimeError("PINDER manifest selected no holo mapping members")
    return members


def _crc32(path: Path) -> int:
    checksum = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            checksum = zlib.crc32(block, checksum)
    return checksum & 0xFFFFFFFF


def _publish_create_only(path: Path, payload: dict[str, object]) -> None:
    serialized = (
        json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("ascii")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_path, path)
    except FileExistsError as error:
        raise FileExistsError(f"Refusing to overwrite receipt: {path}") from error
    finally:
        temporary_path.unlink(missing_ok=True)


def extract_selected(
    archive: Path,
    output_root: Path,
    selected_names: set[str],
) -> tuple[int, int, int]:
    output_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        selected = [item for item in source.infolist() if item.filename in selected_names]
        observed = {item.filename for item in selected}
        missing = selected_names - observed
        if missing:
            raise RuntimeError(
                f"PINDER mappings ZIP is missing {len(missing)} required members"
            )
        if len(selected) != len(observed):
            raise RuntimeError("PINDER mappings ZIP contains duplicate required members")
        selected.sort(key=lambda item: item.header_offset)
        reused = 0
        extracted_bytes = 0
        for index, item in enumerate(selected, start=1):
            destination = output_root / PurePosixPath(item.filename).name
            if destination.exists():
                if (
                    destination.is_symlink()
                    or destination.stat().st_size != item.file_size
                    or _crc32(destination) != item.CRC
                ):
                    raise RuntimeError(
                        f"Existing extracted mapping differs: {destination}"
                    )
                reused += 1
                extracted_bytes += item.file_size
                continue
            temporary = destination.with_name(f".{destination.name}.partial")
            if temporary.exists():
                temporary.unlink()
            with source.open(item, "r") as input_handle, temporary.open("xb") as output:
                for block in iter(lambda: input_handle.read(4 * 1024 * 1024), b""):
                    output.write(block)
            if temporary.stat().st_size != item.file_size:
                raise RuntimeError(f"Extracted mapping size differs: {item.filename}")
            os.replace(temporary, destination)
            extracted_bytes += item.file_size
            if index % 10_000 == 0:
                print(
                    json.dumps(
                        {
                            "extracted_or_reused": index,
                            "required": len(selected),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    return len(selected), reused, extracted_bytes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--expected-archive-sha256", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--materialization-receipt", type=Path, required=True)
    parser.add_argument("--expected-materialization-receipt-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    arguments = parser.parse_args()

    archive = arguments.archive.resolve()
    manifest = arguments.manifest.resolve()
    materialization = arguments.materialization_receipt.resolve()
    output_root = arguments.output_root.resolve()
    receipt = arguments.receipt.resolve()
    for path, expected in (
        (archive, arguments.expected_archive_sha256),
        (manifest, arguments.expected_manifest_sha256),
        (materialization, arguments.expected_materialization_receipt_sha256),
    ):
        if path.is_symlink() or not path.is_file() or _sha256(path) != expected:
            raise RuntimeError(f"Authenticated input differs: {path}")
    if receipt.exists() or receipt.is_symlink():
        raise FileExistsError(f"Refusing to overwrite receipt: {receipt}")

    selected_names = required_members(manifest)
    started = time.time()
    count, reused, extracted_bytes = extract_selected(
        archive,
        output_root,
        selected_names,
    )
    payload = {
        "authorization": {
            "development_decontamination": False,
            "stage_a_training": False,
        },
        "claim_boundary": (
            "Strict PINDER train-only canonical holo residue mappings only; "
            "sequence assembly and frozen-development decontamination remain pending."
        ),
        "counts": {
            "extracted_bytes": extracted_bytes,
            "required_holo_mapping_members": count,
            "reused_members": reused,
        },
        "elapsed_seconds": time.time() - started,
        "format_version": "abpairfm_pinder_holo_mapping_extraction_v1",
        "inputs": {
            "archive_sha256": arguments.expected_archive_sha256,
            "manifest_sha256": arguments.expected_manifest_sha256,
            "materialization_receipt_sha256": (
                arguments.expected_materialization_receipt_sha256
            ),
        },
        "status": "HOLO_MAPPINGS_EXTRACTED_NOT_DECONTAMINATED",
        "test_access": False,
    }
    _publish_create_only(receipt, payload)
    print(json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True))


if __name__ == "__main__":
    main()
