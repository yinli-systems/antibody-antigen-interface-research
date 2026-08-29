"""Immutable source-tree manifests for evidence-gated training jobs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .sabdab2_structure import canonical_json_bytes, sha256_file


SOURCE_MANIFEST_FORMAT = "framecontact_training_source_tree_v1"


def _eligible_files(root: Path, directories: Iterable[str]) -> list[Path]:
    files: list[Path] = []
    for directory in directories:
        base = root / directory
        if not base.is_dir() or base.is_symlink():
            raise ValueError(f"Source directory is missing or unsafe: {base}")
        for path in base.rglob("*"):
            if not path.is_file() or path.is_symlink():
                continue
            relative = path.relative_to(root)
            if "__pycache__" in relative.parts or path.suffix == ".pyc":
                continue
            files.append(path)
    return sorted(set(files), key=lambda value: value.relative_to(root).as_posix())


def build_source_manifest(
    root: Path,
    *,
    directories: tuple[str, ...] = ("scripts", "src", "protocols"),
    claim_boundary: str,
) -> dict[str, Any]:
    root = root.resolve()
    records = [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in _eligible_files(root, directories)
    ]
    tree_digest = hashlib.sha256(
        canonical_json_bytes(
            [{"path": row["path"], "sha256": row["sha256"]} for row in records]
        )
    ).hexdigest()
    return {
        "format_version": SOURCE_MANIFEST_FORMAT,
        "source_root": str(root),
        "included_directories": list(directories),
        "record_count": len(records),
        "tree_sha256": tree_digest,
        "records": records,
        "claim_boundary": claim_boundary,
    }


def verify_source_manifest(
    root: Path, manifest_path: Path, manifest_sha256: str
) -> dict[str, Any]:
    if sha256_file(manifest_path) != manifest_sha256:
        raise ValueError("Source manifest SHA-256 differs")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Cannot read source manifest") from error
    root = root.resolve()
    if (
        manifest.get("format_version") != SOURCE_MANIFEST_FORMAT
        or Path(str(manifest.get("source_root", ""))).resolve() != root
    ):
        raise ValueError("Source manifest root or format differs")
    directories = tuple(str(value) for value in manifest.get("included_directories", ()))
    expected_records = manifest.get("records")
    if not isinstance(expected_records, list):
        raise ValueError("Source manifest records are invalid")
    actual_paths = {
        path.relative_to(root).as_posix(): path
        for path in _eligible_files(root, directories)
    }
    expected_paths = {str(row.get("path")) for row in expected_records}
    if set(actual_paths) != expected_paths:
        raise ValueError("Source-tree file inventory differs")
    for row in expected_records:
        path = actual_paths[str(row["path"])]
        if path.stat().st_size != int(row["size_bytes"]):
            raise ValueError(f"Source file size differs: {row['path']}")
        if sha256_file(path) != row["sha256"]:
            raise ValueError(f"Source file SHA-256 differs: {row['path']}")
    observed_tree = hashlib.sha256(
        canonical_json_bytes(
            [
                {"path": row["path"], "sha256": row["sha256"]}
                for row in expected_records
            ]
        )
    ).hexdigest()
    if observed_tree != manifest.get("tree_sha256"):
        raise ValueError("Source-tree aggregate digest differs")
    if int(manifest.get("record_count", -1)) != len(expected_records):
        raise ValueError("Source manifest record count differs")
    return manifest


__all__ = [
    "SOURCE_MANIFEST_FORMAT",
    "build_source_manifest",
    "verify_source_manifest",
]
