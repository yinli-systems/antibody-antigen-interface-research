#!/usr/bin/env python3
"""Retain only the newest verified ABPairFM DCP generations.

Large DCP payloads are removed only after their immutable result, progress, and
receipt records agree.  The small receipt and all campaign evidence outside the
checkpoint directory are preserved.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import stat
import tempfile


SEGMENT_RE = re.compile(r"segment-(\d{6})$")
PAYLOAD_RE = re.compile(r"__(\d+)_0\.distcp$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def verified_generation(root: Path, generation: Path) -> dict:
    match = SEGMENT_RE.fullmatch(generation.name)
    if match is None:
        raise RuntimeError(f"unexpected checkpoint directory: {generation}")
    index = int(match.group(1))
    receipt_path = generation / "receipt.json"
    result_path = root / "results" / f"{generation.name}.json"
    progress_path = root / "progress" / f"{generation.name}.json"
    for path in (receipt_path, result_path, progress_path):
        if not path.is_file():
            raise RuntimeError(f"missing verification artifact: {path}")

    receipt = load_json(receipt_path)
    result = load_json(result_path)
    progress = load_json(progress_path)
    content_sha = receipt.get("content_sha256")
    if (
        receipt.get("status") != "complete"
        or receipt.get("generation") != generation.name
        or result.get("status") != "PASS_CHECKPOINT_VERIFIED"
        or result.get("segment_index") != index
        or result.get("checkpoint", {}).get("generation") != generation.name
        or result.get("checkpoint", {}).get("content_sha256") != content_sha
        or progress.get("status") != "PASS_CHECKPOINT_VERIFIED"
        or progress.get("segment_index") != index
        or progress.get("checkpoint_content_sha256") != content_sha
        or result.get("test_access") is not False
        or progress.get("test_access") is not False
    ):
        raise RuntimeError(f"verification contract differs for {generation.name}")

    expected_files = receipt.get("files", {})
    if not isinstance(expected_files, dict) or not expected_files:
        raise RuntimeError(f"receipt has no payload manifest: {receipt_path}")
    actual_payload = sorted(
        path.name
        for path in generation.iterdir()
        if path.name == ".metadata" or PAYLOAD_RE.fullmatch(path.name)
    )
    if set(actual_payload) != set(expected_files):
        raise RuntimeError(f"payload names differ from receipt for {generation.name}")
    unknown = sorted(
        path.name
        for path in generation.iterdir()
        if path.name != "receipt.json"
        and path.name != ".metadata"
        and PAYLOAD_RE.fullmatch(path.name) is None
    )
    if unknown:
        raise RuntimeError(f"unrecognized checkpoint files in {generation}: {unknown}")

    bytes_total = 0
    for name in actual_payload:
        path = generation / name
        expected_size = expected_files[name].get("bytes")
        if not path.is_file() or path.stat().st_size != expected_size:
            raise RuntimeError(f"payload size differs from receipt: {path}")
        bytes_total += path.stat().st_size
    return {
        "index": index,
        "generation": generation.name,
        "content_sha256": content_sha,
        "receipt_sha256": sha256_file(receipt_path),
        "result_sha256": sha256_file(result_path),
        "progress_sha256": sha256_file(progress_path),
        "payload_files": actual_payload,
        "payload_bytes": bytes_total,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--keep-full", type=int, default=2)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    checkpoint_root = root / "checkpoints"
    if args.keep_full < 2:
        raise SystemExit("keep-full must be at least 2")
    if root.name != "abpairfm_3b_stage_a0_string25m_v1" or not checkpoint_root.is_dir():
        raise SystemExit(f"refusing unexpected campaign root: {root}")

    args.audit_root.mkdir(parents=True, exist_ok=True)
    lock_path = args.audit_root / "retention.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        full = []
        inflight = []
        for path in sorted(checkpoint_root.glob("segment-*")):
            if any(
                child.name == ".metadata" or PAYLOAD_RE.fullmatch(child.name)
                for child in path.iterdir()
            ):
                match = SEGMENT_RE.fullmatch(path.name)
                if match is None:
                    raise RuntimeError(f"unexpected checkpoint directory: {path}")
                index = int(match.group(1))
                required = (
                    path / "receipt.json",
                    root / "results" / f"{path.name}.json",
                    root / "progress" / f"{path.name}.json",
                )
                if not all(item.is_file() for item in required):
                    inflight.append({"index": index, "generation": path.name})
                    continue
                full.append(verified_generation(root, path))
        if inflight:
            if len(inflight) != 1 or not full or inflight[0]["index"] <= full[-1]["index"]:
                raise RuntimeError(f"unexpected unverified checkpoint generations: {inflight}")
        if len(full) <= args.keep_full:
            protected = full
            prune = []
        else:
            protected = full[-args.keep_full :]
            prune = full[: -args.keep_full]
        if prune and max(item["index"] for item in prune) >= min(
            item["index"] for item in protected
        ):
            raise RuntimeError("retention ordering invariant failed")

        now = dt.datetime.now(dt.timezone.utc)
        audit_path = args.audit_root / (
            f"retention-{now.strftime('%Y%m%dT%H%M%SZ')}-{os.getpid()}.json"
        )
        audit = {
            "format_version": "abpairfm_3b_checkpoint_retention_v1",
            "status": "PLANNED" if args.execute else "DRY_RUN_PASS",
            "campaign_root": str(root),
            "checkpoint_root": str(checkpoint_root),
            "keep_full": args.keep_full,
            "protected_full_generations": protected,
            "protected_inflight_generations": inflight,
            "pruned_generations": prune,
            "planned_freed_bytes": sum(item["payload_bytes"] for item in prune),
            "preserved_artifacts": ["receipt.json", "results", "progress", "audits"],
            "hostname": socket.gethostname(),
            "utc": now.isoformat(),
        }
        write_json_atomic(audit_path, audit)
        if args.execute:
            deleted_bytes = 0
            for item in prune:
                generation = checkpoint_root / item["generation"]
                os.chmod(generation, stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP)
                for name in item["payload_files"]:
                    path = generation / name
                    deleted_bytes += path.stat().st_size
                    path.unlink()
                os.chmod(
                    generation,
                    stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP,
                )
            audit["status"] = "PASS_RETENTION_EXECUTED"
            audit["deleted_bytes"] = deleted_bytes
            audit["remaining_full_generations"] = [
                item["generation"] for item in protected
            ]
            write_json_atomic(audit_path, audit)
        print(audit_path)
        print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()
