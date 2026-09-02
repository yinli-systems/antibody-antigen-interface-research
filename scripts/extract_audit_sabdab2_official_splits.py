#!/usr/bin/env python3
"""Safely extract and audit the four official SAbDab2 split tables."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import tempfile
from datetime import datetime, timezone


SOURCE_ID = "sabdab2_ml_zenodo_22019991"
EXPECTED_ARCHIVE_SHA256 = (
    "6f3aad7c62c2afb8f94075a38eca76a61b5558b5feef34cb4d976511012b06ce"
)
MEMBERS = {
    "splits_final/ab_split.csv": "ab_split.csv",
    "splits_final/ab_split_sd.csv": "ab_split_sd.csv",
    "splits_final/abag_split.csv": "abag_split.csv",
    "splits_final/abag_split_sd.csv": "abag_split_sd.csv",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def audit_csv(path: Path) -> dict[str, object]:
    split_field = "ab_ag_split" if path.name.startswith("abag_") else "ab_split"
    rows = 0
    instances: set[str] = set()
    pdb_ids: set[str] = set()
    by_split: dict[str, int] = {}
    instances_by_split: dict[str, set[str]] = {}
    pdbs_by_split: dict[str, set[str]] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        required = {"INSTANCE", "PDB_ID", split_field}
        if not required.issubset(fields):
            raise ValueError(f"{path.name}: missing fields {sorted(required - set(fields))}")
        for row in reader:
            rows += 1
            split = row[split_field]
            instance = row["INSTANCE"]
            pdb_id = row["PDB_ID"]
            by_split[split] = by_split.get(split, 0) + 1
            instances_by_split.setdefault(split, set()).add(instance)
            pdbs_by_split.setdefault(split, set()).add(pdb_id)
            instances.add(instance)
            pdb_ids.add(pdb_id)
    train_instances = instances_by_split.get("train", set())
    test_instances = instances_by_split.get("test", set())
    train_pdbs = pdbs_by_split.get("train", set())
    test_pdbs = pdbs_by_split.get("test", set())
    return {
        "field_count": len(fields),
        "fields": fields,
        "row_count": rows,
        "split_field": split_field,
        "split_counts": dict(sorted(by_split.items())),
        "unique_instance_count": len(instances),
        "duplicate_instance_rows": rows - len(instances),
        "unique_pdb_count": len(pdb_ids),
        "train_test_instance_overlap": len(train_instances & test_instances),
        "train_test_pdb_overlap": len(train_pdbs & test_pdbs),
    }


def extract_member(archive: tarfile.TarFile, member_name: str, output: Path) -> None:
    member = archive.getmember(member_name)
    if not member.isfile() or member.issym() or member.islnk():
        raise ValueError(f"archive member is not a regular file: {member_name}")
    source = archive.extractfile(member)
    if source is None:
        raise ValueError(f"archive member cannot be read: {member_name}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as handle:
        temporary = Path(handle.name)
        shutil.copyfileobj(source, handle, length=1024 * 1024)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        if output.exists():
            if sha256_file(output) != sha256_file(temporary):
                raise ValueError(f"existing output differs from archive: {output}")
        else:
            os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    archive_path = args.archive.resolve()
    output_dir = args.output_dir.resolve()
    existing_generated_at = None
    if args.receipt.exists():
        existing_receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
        existing_generated_at = existing_receipt.get("generated_at_utc")
    archive_sha256 = sha256_file(archive_path)
    if archive_sha256 != EXPECTED_ARCHIVE_SHA256:
        raise ValueError("archive SHA-256 differs from the pinned SAbDab2 release")

    with tarfile.open(archive_path, mode="r:gz") as archive:
        for member_name, output_name in sorted(MEMBERS.items()):
            extract_member(archive, member_name, output_dir / output_name)

    outputs = []
    for member_name, output_name in sorted(MEMBERS.items()):
        output = output_dir / output_name
        outputs.append(
            {
                "archive_member": member_name,
                "path": f"official_splits/{output_name}",
                "sha256": sha256_file(output),
                "size_bytes": output.stat().st_size,
                "audit": audit_csv(output),
            }
        )
    receipt = {
        "format_version": "sabdab2_official_split_extraction_receipt_v1",
        "source_id": SOURCE_ID,
        "status": "PASS_OFFICIAL_SPLITS_EXTRACTED_AND_AUDITED",
        "generated_at_utc": existing_generated_at
        or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "archive": {
            "path": "splits.tar.gz",
            "sha256": archive_sha256,
            "size_bytes": archive_path.stat().st_size,
        },
        "outputs": outputs,
        "safety": {
            "exact_member_allowlist": True,
            "regular_files_only": True,
            "symlinks_followed": False,
            "structures_read": False,
            "task_labels_read": False,
        },
    }
    receipt_bytes = canonical_bytes(receipt)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    if args.receipt.exists() and args.receipt.read_bytes() != receipt_bytes:
        raise FileExistsError(f"refusing to overwrite differing receipt: {args.receipt}")
    if not args.receipt.exists():
        args.receipt.write_bytes(receipt_bytes)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
