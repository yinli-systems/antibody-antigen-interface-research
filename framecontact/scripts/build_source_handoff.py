#!/usr/bin/env python3
"""Build a deterministic, auditable source-only handoff archive.

The archive deliberately excludes interpreter/build caches and refuses to
package common secret-bearing filenames, symbolic links, or private-key
material. Every included file is recorded in a canonical JSON manifest.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import stat
import tarfile
from typing import Any


ARCHIVE_ROOT = "framecontact"
EXCLUDED_DIR_NAMES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "dist",
    "evaluation",
}
EXCLUDED_FILE_SUFFIXES = {".pyc", ".pyo"}
SECRET_FILE_NAMES = {
    ".env",
    ".env.local",
    "credentials",
    "credentials.json",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "id_rsa",
    "secrets.json",
}
SECRET_FILE_SUFFIXES = {".key", ".p12", ".pfx", ".pem"}
PRIVATE_KEY_MARKERS = (
    b"-----BEGIN PRIVATE KEY-----\n",
    b"-----BEGIN OPENSSH PRIVATE KEY-----\n",
    b"-----BEGIN RSA PRIVATE KEY-----\n",
    b"-----BEGIN EC PRIVATE KEY-----\n",
    b"-----BEGIN PRIVATE KEY-----\r\n",
    b"-----BEGIN OPENSSH PRIVATE KEY-----\r\n",
    b"-----BEGIN RSA PRIVATE KEY-----\r\n",
    b"-----BEGIN EC PRIVATE KEY-----\r\n",
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True).encode("utf-8")
        + b"\n"
    )


def is_excluded(relative_path: Path) -> bool:
    if any(
        part in EXCLUDED_DIR_NAMES or part.endswith(".egg-info")
        for part in relative_path.parts
    ):
        return True
    return relative_path.suffix.lower() in EXCLUDED_FILE_SUFFIXES


def collect_files(
    source_root: Path, output_dir: Path
) -> list[tuple[Path, Path, bytes]]:
    collected: list[tuple[Path, Path, bytes]] = []
    for absolute_path in sorted(source_root.rglob("*")):
        relative_path = absolute_path.relative_to(source_root)
        if is_excluded(relative_path):
            continue
        if absolute_path == output_dir or output_dir in absolute_path.parents:
            continue
        if absolute_path.is_symlink():
            raise ValueError(f"Refusing symbolic link: {relative_path}")
        if absolute_path.is_dir():
            continue
        if not absolute_path.is_file():
            raise ValueError(f"Refusing non-regular file: {relative_path}")

        lower_name = absolute_path.name.lower()
        if (
            lower_name in SECRET_FILE_NAMES
            or absolute_path.suffix.lower() in SECRET_FILE_SUFFIXES
        ):
            raise ValueError(f"Refusing secret-like filename: {relative_path}")

        payload = absolute_path.read_bytes()
        if any(marker in payload for marker in PRIVATE_KEY_MARKERS):
            raise ValueError(f"Refusing private-key material: {relative_path}")
        collected.append((absolute_path, relative_path, payload))
    if not collected:
        raise ValueError(f"No source files found under {source_root}")
    return collected


def normalized_file_info(
    archive_name: str, executable: bool, size: int
) -> tarfile.TarInfo:
    info = tarfile.TarInfo(archive_name)
    info.size = size
    info.mode = 0o755 if executable else 0o644
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = "root"
    info.gname = "root"
    return info


def build_archive(
    source_root: Path, output_dir: Path, archive_name: str
) -> tuple[Path, Path]:
    source_root = source_root.resolve()
    output_dir = output_dir.resolve()
    if output_dir == source_root or source_root in output_dir.parents:
        raise ValueError("--output-dir must be outside the source tree")
    output_dir.mkdir(parents=True, exist_ok=True)

    archive_path = output_dir / archive_name
    receipt_path = output_dir / f"{archive_name}.receipt.json"
    for target in (archive_path, receipt_path):
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite existing artifact: {target}")

    files = collect_files(source_root, output_dir)
    manifest = {
        "archive_format": "tar+gzip",
        "archive_root": ARCHIVE_ROOT,
        "deterministic_metadata": {
            "gzip_mtime": 0,
            "tar_gid": 0,
            "tar_mtime": 0,
            "tar_uid": 0,
        },
        "file_count": len(files),
        "files": [
            {
                "path": relative_path.as_posix(),
                "sha256": sha256_bytes(payload),
                "size_bytes": len(payload),
            }
            for _, relative_path, payload in files
        ],
        "scope": (
            "source, configs, tests, and documentation only; "
            "no datasets, checkpoints, or results"
        ),
        "schema_version": 1,
    }
    manifest_payload = canonical_json(manifest)

    with archive_path.open("xb") as raw_file:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=raw_file, mtime=0
        ) as gzip_file:
            with tarfile.open(
                fileobj=gzip_file, mode="w", format=tarfile.PAX_FORMAT
            ) as tar:
                for absolute_path, relative_path, payload in files:
                    executable = bool(
                        absolute_path.stat().st_mode & stat.S_IXUSR
                    )
                    archive_member = (
                        f"{ARCHIVE_ROOT}/{relative_path.as_posix()}"
                    )
                    info = normalized_file_info(
                        archive_member, executable, len(payload)
                    )
                    tar.addfile(info, io.BytesIO(payload))

                manifest_member = f"{ARCHIVE_ROOT}/SOURCE_MANIFEST.json"
                info = normalized_file_info(
                    manifest_member, False, len(manifest_payload)
                )
                tar.addfile(info, io.BytesIO(manifest_payload))

    receipt = {
        "archive": archive_path.name,
        "archive_sha256": sha256_bytes(archive_path.read_bytes()),
        "archive_size_bytes": archive_path.stat().st_size,
        "file_count": len(files),
        "manifest_sha256": sha256_bytes(manifest_payload),
        "schema_version": 1,
    }
    receipt_path.write_bytes(canonical_json(receipt))
    return archive_path, receipt_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--archive-name",
        default="framecontact-source.tar.gz",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    archive_path, receipt_path = build_archive(
        args.source_root,
        args.output_dir,
        args.archive_name,
    )
    print(archive_path)
    print(receipt_path)


if __name__ == "__main__":
    main()
