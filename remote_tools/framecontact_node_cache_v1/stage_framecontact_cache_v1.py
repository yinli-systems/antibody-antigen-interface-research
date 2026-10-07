#!/usr/bin/env python3
"""Stage manifest-backed FrameContact caches onto node-local storage.

The staging operation is byte preserving.  A cache becomes visible only after
all manifest-listed artifacts have been rehashed successfully and the complete
tree has been atomically renamed into place.  Concurrent jobs on one node share
the same bundle under an advisory file lock.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def parse_named(values: list[str], option: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{option} requires NAME=VALUE: {value!r}")
        name, item = value.split("=", 1)
        if not name or not item or name in parsed:
            raise ValueError(f"invalid or duplicate {option}: {value!r}")
        parsed[name] = item
    return parsed


def safe_relative(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"unsafe relative path: {value!r}")
    return path


def tree_inventory(root: Path) -> tuple[list[str], int]:
    files: list[str] = []
    total = 0
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"cache tree contains symlink: {path}")
        if path.is_file():
            files.append(path.relative_to(root).as_posix())
            total += path.stat().st_size
    return files, total


def manifest_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("cache manifest must contain a non-empty records list")
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("cache manifest records must be objects")
    return records


def verify_staged_cache(
    *,
    source_root: Path,
    target_root: Path,
    manifest_relative: Path,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    source_files, source_bytes = tree_inventory(source_root)
    target_files, target_bytes = tree_inventory(target_root)
    if source_files != target_files or source_bytes != target_bytes:
        raise ValueError("staged cache tree inventory differs from source")

    target_manifest = target_root / manifest_relative
    if not target_manifest.is_file():
        raise ValueError(f"staged manifest is missing: {target_manifest}")
    actual_manifest_sha256 = sha256_file(target_manifest)
    if actual_manifest_sha256 != expected_manifest_sha256:
        raise ValueError("staged manifest SHA256 differs from frozen authority")

    payload = json.loads(target_manifest.read_text(encoding="utf-8"))
    records = manifest_records(payload)
    artifact_paths: set[str] = set()
    artifact_bytes = 0
    for record in records:
        relative = safe_relative(str(record.get("artifact_path", "")))
        relative_text = relative.as_posix()
        if relative_text in artifact_paths:
            raise ValueError(f"duplicate artifact path: {relative_text}")
        expected = str(record.get("artifact_sha256", ""))
        if len(expected) != 64:
            raise ValueError(f"invalid artifact SHA256 for {relative_text}")
        artifact = target_root / relative
        if not artifact.is_file():
            raise ValueError(f"staged artifact is missing: {artifact}")
        if sha256_file(artifact) != expected:
            raise ValueError(f"staged artifact SHA256 differs: {relative_text}")
        if "artifact_size_bytes" in record:
            if artifact.stat().st_size != int(record["artifact_size_bytes"]):
                raise ValueError(f"staged artifact size differs: {relative_text}")
        artifact_paths.add(relative_text)
        artifact_bytes += artifact.stat().st_size

    # Manifest-listed tensors dominate the tree.  Rehash every remaining file
    # against its source so receipts, split views, and metadata are also exact.
    unlisted_verified = 0
    for relative_text in target_files:
        if relative_text in artifact_paths or relative_text == manifest_relative.as_posix():
            continue
        if sha256_file(target_root / relative_text) != sha256_file(
            source_root / relative_text
        ):
            raise ValueError(f"staged auxiliary file differs: {relative_text}")
        unlisted_verified += 1

    return {
        "artifact_bytes": artifact_bytes,
        "artifact_count": len(artifact_paths),
        "file_count": len(target_files),
        "manifest_sha256": actual_manifest_sha256,
        "tree_bytes": target_bytes,
        "unlisted_file_count_verified": unlisted_verified,
    }


def load_valid_receipt(target: Path, spec_sha256: str) -> dict[str, Any] | None:
    receipt_path = target / ".STAGE_RECEIPT.json"
    if not receipt_path.is_file():
        return None
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("status") != "PASS_NODE_LOCAL_CACHE_VERIFIED"
        or receipt.get("spec_sha256") != spec_sha256
    ):
        return None
    for item in receipt.get("caches", []):
        cache_root = target / str(item["name"])
        manifest = cache_root / safe_relative(str(item["manifest_relative_path"]))
        if not manifest.is_file() or sha256_file(manifest) != item["manifest_sha256"]:
            return None
        verified = verify_staged_cache(
            source_root=Path(item["source_root"]),
            target_root=cache_root,
            manifest_relative=Path(item["manifest_relative_path"]),
            expected_manifest_sha256=item["manifest_sha256"],
        )
        if any(item.get(key) != value for key, value in verified.items()):
            return None
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument(
        "--manifest-relative", action="append", default=[], metavar="NAME=PATH"
    )
    parser.add_argument(
        "--manifest-sha256", action="append", default=[], metavar="NAME=SHA256"
    )
    parser.add_argument("--node-cache-root", type=Path, required=True)
    parser.add_argument("--durable-receipt", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    sources = parse_named(args.source_root, "--source-root")
    manifests = parse_named(args.manifest_relative, "--manifest-relative")
    manifest_shas = parse_named(args.manifest_sha256, "--manifest-sha256")
    if not sources or set(sources) != set(manifests) or set(sources) != set(manifest_shas):
        raise ValueError("source, manifest, and SHA256 names must match and be non-empty")

    spec = []
    required_bytes = 0
    for name in sorted(sources):
        source = Path(sources[name]).resolve(strict=True)
        if not source.is_dir():
            raise ValueError(f"source root is not a directory: {source}")
        manifest_relative = safe_relative(manifests[name])
        expected_sha = manifest_shas[name].lower()
        if len(expected_sha) != 64:
            raise ValueError(f"invalid manifest SHA256 for {name}")
        source_manifest = source / manifest_relative
        if not source_manifest.is_file() or sha256_file(source_manifest) != expected_sha:
            raise ValueError(f"source manifest authority differs for {name}")
        _, source_bytes = tree_inventory(source)
        required_bytes += source_bytes
        spec.append(
            {
                "manifest_relative_path": manifest_relative.as_posix(),
                "manifest_sha256": expected_sha,
                "name": name,
                "source_root": str(source),
                "source_tree_bytes": source_bytes,
            }
        )

    node_cache_root = args.node_cache_root.resolve()
    if node_cache_root == Path("/") or node_cache_root == Path.home().resolve():
        raise ValueError("refusing broad node cache root")
    node_cache_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    spec_sha256 = canonical_json_sha256(spec)
    target = node_cache_root / spec_sha256
    lock_path = node_cache_root / f"{spec_sha256}.lock"

    with lock_path.open("a+b") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        existing = load_valid_receipt(target, spec_sha256) if target.exists() else None
        reused = existing is not None
        if target.exists() and existing is None:
            raise ValueError(f"existing node cache is incomplete or invalid: {target}")
        if not reused:
            free_bytes = shutil.disk_usage(node_cache_root).free
            if free_bytes < required_bytes + max(required_bytes // 5, 1 << 30):
                raise OSError(
                    f"insufficient node-local space: need {required_bytes}, free {free_bytes}"
                )
            partial = node_cache_root / (
                f".{spec_sha256}.partial.{os.getpid()}."
                f"{os.environ.get('SLURM_JOB_ID', 'local')}"
            )
            if partial.exists():
                raise ValueError(f"unique partial path already exists: {partial}")
            partial.mkdir(mode=0o700)
            cache_receipts = []
            try:
                for item in spec:
                    source = Path(item["source_root"])
                    destination = partial / item["name"]
                    shutil.copytree(source, destination, symlinks=False)
                    verified = verify_staged_cache(
                        source_root=source,
                        target_root=destination,
                        manifest_relative=Path(item["manifest_relative_path"]),
                        expected_manifest_sha256=item["manifest_sha256"],
                    )
                    cache_receipts.append({**item, **verified})
                receipt = {
                    "caches": cache_receipts,
                    "format_version": "framecontact_node_local_cache_receipt_v1",
                    "scientific_protocol_changed": False,
                    "spec_sha256": spec_sha256,
                    "status": "PASS_NODE_LOCAL_CACHE_VERIFIED",
                    "test_or_temporal_access": False,
                }
                (partial / ".STAGE_RECEIPT.json").write_text(
                    json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                os.replace(partial, target)
            except BaseException:
                shutil.rmtree(partial, ignore_errors=True)
                raise
        receipt = load_valid_receipt(target, spec_sha256)
        if receipt is None:
            raise RuntimeError("node cache receipt failed post-rename verification")

    output = {
        **receipt,
        "reused_existing_cache": reused,
        "target_roots": {
            item["name"]: str(target / item["name"]) for item in spec
        },
    }
    if args.durable_receipt:
        args.durable_receipt.parent.mkdir(parents=True, exist_ok=True)
        args.durable_receipt.write_text(
            json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"FAIL_CLOSED_NODE_LOCAL_CACHE: {error}", file=sys.stderr)
        raise
