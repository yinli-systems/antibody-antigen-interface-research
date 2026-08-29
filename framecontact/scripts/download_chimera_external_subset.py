#!/usr/bin/env python3
"""Materialize a hash-verified CHIMERA split subset from a pinned revision."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen


DATASET_ID = "mansoorbaloch/chimera-bench"
PINNED_REVISION = "54d5a2a1e59a329db7a6c09184652ed225188940"
SPLIT_NAMES = ("temporal", "antigen_fold", "epitope_group")


def git_blob_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    digest.update(f"blob {path.stat().st_size}\0".encode("ascii"))
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_digest(entry: dict[str, Any]) -> tuple[str, str]:
    lfs = entry.get("lfs")
    if isinstance(lfs, dict) and isinstance(lfs.get("sha256"), str):
        return "sha256", lfs["sha256"]
    blob_id = entry.get("blobId")
    if not isinstance(blob_id, str) or len(blob_id) != 40:
        raise ValueError(f"artifact lacks a verifiable digest: {entry.get('rfilename')}")
    return "git_blob_sha1", blob_id


def observed_digest(path: Path, kind: str) -> str:
    if kind == "sha256":
        return sha256(path)
    if kind == "git_blob_sha1":
        return git_blob_sha1(path)
    raise ValueError(f"unknown digest kind: {kind}")


def _verify(path: Path, entry: dict[str, Any]) -> tuple[str, str]:
    expected_size = int(entry["size"])
    if path.stat().st_size != expected_size:
        raise ValueError(
            f"size mismatch for {entry['rfilename']}: "
            f"{path.stat().st_size} != {expected_size}"
        )
    kind, expected = expected_digest(entry)
    observed = observed_digest(path, kind)
    if observed != expected:
        raise ValueError(f"digest mismatch for {entry['rfilename']}")
    return kind, observed


def _download_one(
    root: Path, entry: dict[str, Any], timeout: int, retries: int = 5
) -> dict[str, Any]:
    relative = entry["rfilename"]
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        try:
            kind, observed = _verify(target, entry)
            return {
                "path": relative,
                "size": int(entry["size"]),
                "digest_kind": kind,
                "digest": observed,
                "state": "verified_existing",
            }
        except ValueError:
            pass
    target.unlink(missing_ok=True)
    url = (
        "https://huggingface.co/datasets/"
        f"{DATASET_ID}/resolve/{PINNED_REVISION}/{quote(relative, safe='/')}?download=true"
    )
    request = Request(url, headers={"User-Agent": "geometry-self-attention/0.2"})
    last_error: BaseException | None = None
    for attempt in range(retries):
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".part", dir=target.parent
        )
        try:
            with os.fdopen(fd, "wb") as destination, urlopen(
                request, timeout=timeout
            ) as source:
                while True:
                    block = source.read(1024 * 1024)
                    if not block:
                        break
                    destination.write(block)
                destination.flush()
                os.fsync(destination.fileno())
            temporary = Path(temporary_name)
            kind, observed = _verify(temporary, entry)
            os.replace(temporary, target)
            break
        except BaseException as error:
            last_error = error
            if attempt + 1 == retries:
                raise
            time.sleep(min(2**attempt, 8))
        finally:
            Path(temporary_name).unlink(missing_ok=True)
    else:  # pragma: no cover - loop either breaks or raises
        raise RuntimeError("download retries exhausted") from last_error
    return {
        "path": relative,
        "size": int(entry["size"]),
        "digest_kind": kind,
        "digest": observed,
        "state": "downloaded_and_verified",
    }


def _load_inputs(
    root: Path, metadata_path: Path, splits: list[str], subset: str
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, list[str]]]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("sha") != PINNED_REVISION:
        raise ValueError("authority metadata does not identify the pinned revision")
    index = {entry["rfilename"]: entry for entry in metadata.get("siblings", [])}
    split_ids: dict[str, list[str]] = {}
    complex_ids: set[str] = set()
    for split in splits:
        path = root / "splits" / f"{split}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        values = payload.get(subset)
        if not isinstance(values, list) or not values:
            raise ValueError(f"empty or malformed split subset: {split}.{subset}")
        identifiers = [str(value) for value in values]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError(f"duplicate IDs in {split}.{subset}")
        split_ids[f"{split}.{subset}"] = identifiers
        complex_ids.update(identifiers)
    paths = {f"complex_features/{identifier}.pt" for identifier in complex_ids}
    paths.update(f"splits/{split}.json" for split in splits)
    paths.update(
        f"structures/{identifier.split('_', 1)[0].lower()}.pdb"
        for identifier in complex_ids
    )
    missing = sorted(path for path in paths if path not in index)
    if missing:
        raise ValueError(f"authority metadata is missing {len(missing)} artifacts")
    return metadata, [index[path] for path in sorted(paths)], split_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--authority-metadata", type=Path, required=True)
    parser.add_argument(
        "--splits", nargs="+", choices=SPLIT_NAMES, default=["temporal", "antigen_fold"]
    )
    parser.add_argument("--subset", choices=("train", "val", "test"), default="test")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 32:
        raise SystemExit("--workers must be in [1, 32]")

    metadata, entries, split_ids = _load_inputs(
        args.root, args.authority_metadata, args.splits, args.subset
    )
    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(_download_one, args.root, entry, args.timeout): entry
            for entry in entries
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            record = future.result()
            records.append(record)
            if completed % 50 == 0 or completed == len(entries):
                print(f"verified {completed}/{len(entries)} artifacts", flush=True)

    manifest = {
        "schema_version": "chimera_external_materialization_manifest_v1",
        "dataset_id": DATASET_ID,
        "pinned_revision": PINNED_REVISION,
        "authority_metadata_sha256": sha256(args.authority_metadata),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "subset": args.subset,
        "splits": split_ids,
        "unique_complexes": len(set().union(*(set(v) for v in split_ids.values()))),
        "artifact_count": len(records),
        "total_bytes": sum(record["size"] for record in records),
        "artifacts": sorted(records, key=lambda item: item["path"]),
    }
    manifest_path = args.manifest or args.root / "materialization_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary.write_text(serialized, encoding="utf-8")
    os.replace(temporary, manifest_path)
    print(f"manifest={manifest_path}")
    print(f"manifest_sha256={sha256(manifest_path)}")


if __name__ == "__main__":
    main()
