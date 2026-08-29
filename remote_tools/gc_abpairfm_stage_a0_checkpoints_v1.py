#!/usr/bin/env python3
"""Audit and garbage-collect verified ABPairFM Stage-A0 DCP generations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any


EXPECTED_ROOT = Path(
    "/ssd/scxi253/model_deployment/results/pretraining/"
    "abpairfm_3b_stage_a0_string25m_v1"
)
KEEP_SEGMENTS = (15, 31, 62, 72, 73)
EXPECTED_MILESTONES = {
    15: (252_633_502, "d6cb03ff651df3f01544f399fd7b5fa6f0ee681862f133ea2697f1afb56de598"),
    31: (505_708_186, "73fd08e73bf41118ece0e657059c5211ce5930c1c651e3cfd219554692edf08b"),
    62: (995_667_289, "0814c7cbbcc1b681af15d231d6c7162788fc9b20933f4086b112cac85f5828ba"),
    72: (1_153_744_101, "8f38ed8e99bb89ee849fd4212195f0be0bb9380437270359a4134ebfb6d35507"),
    73: (1_169_467_448, "38a0987985f119afb374f2062e407bc9c5006f3f7ad66711744086d9561810b1"),
}
SEGMENT_RE = re.compile(r"segment-(\d{6})$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()


def atomic_write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(canonical_bytes(payload))
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.iterdir() if item.is_file())


def collect(root: Path) -> list[dict[str, Any]]:
    checkpoints = root / "checkpoints"
    results = root / "results"
    if root != EXPECTED_ROOT or root.is_symlink() or checkpoints.is_symlink():
        raise ValueError(f"refusing unexpected or symlinked campaign root: {root}")
    rows: list[dict[str, Any]] = []
    for result_path in sorted(results.glob("segment-*.json")):
        match = SEGMENT_RE.fullmatch(result_path.stem)
        if match is None:
            continue
        segment = int(match.group(1))
        result = json.loads(result_path.read_text())
        checkpoint = checkpoints / f"segment-{segment:06d}"
        if (
            result.get("segment_index") != segment
            or result.get("status") != "PASS_CHECKPOINT_VERIFIED"
            or result.get("test_access") is not False
            or not checkpoint.is_dir()
            or checkpoint.is_symlink()
            or checkpoint.parent.resolve() != checkpoints.resolve()
        ):
            raise ValueError(f"segment {segment} is not an eligible verified DCP")
        receipt_path = checkpoint / "receipt.json"
        receipt = json.loads(receipt_path.read_text())
        content_sha = result.get("checkpoint", {}).get("content_sha256")
        if (
            receipt.get("status") != "complete"
            or receipt.get("content_sha256") != content_sha
            or receipt.get("training_state", {}).get("consumed_residue_tokens")
            != result.get("training_state", {}).get("consumed_residue_tokens")
        ):
            raise ValueError(f"segment {segment} result/receipt differs")
        for filename, metadata in receipt.get("files", {}).items():
            artifact = checkpoint / filename
            if (
                not artifact.is_file()
                or artifact.is_symlink()
                or artifact.stat().st_size != int(metadata["bytes"])
            ):
                raise ValueError(f"segment {segment} artifact size differs: {filename}")
        rows.append(
            {
                "segment": segment,
                "consumed_residue_tokens": int(
                    result["training_state"]["consumed_residue_tokens"]
                ),
                "dcp_content_sha256": content_sha,
                "result_sha256": sha256_file(result_path),
                "receipt_sha256": sha256_file(receipt_path),
                "checkpoint_bytes": directory_size(checkpoint),
                "action": "keep" if segment in KEEP_SEGMENTS else "delete",
            }
        )
    if [row["segment"] for row in rows] != list(range(74)):
        raise ValueError("verified checkpoint inventory is not exactly segments 0..73")
    for segment, (tokens, content_sha) in EXPECTED_MILESTONES.items():
        row = rows[segment]
        if (
            row["segment"] != segment
            or row["consumed_residue_tokens"] != tokens
            or row["dcp_content_sha256"] != content_sha
            or row["action"] != "keep"
        ):
            raise ValueError(f"frozen keep checkpoint differs: segment {segment}")
    return rows


def archive_receipts(root: Path, rows: list[dict[str, Any]]) -> Path:
    archive = root / "audits" / "checkpoint_receipts_pre_gc_v1"
    archive.mkdir(parents=True, exist_ok=True)
    for row in rows:
        segment = row["segment"]
        source = root / "checkpoints" / f"segment-{segment:06d}" / "receipt.json"
        target = archive / f"segment-{segment:06d}-receipt.json"
        if target.exists():
            if sha256_file(target) != row["receipt_sha256"]:
                raise ValueError(f"archived receipt differs: segment {segment}")
        else:
            shutil.copy2(source, target)
        if sha256_file(target) != row["receipt_sha256"]:
            raise ValueError(f"receipt archive failed: segment {segment}")
    return archive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, default=EXPECTED_ROOT)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    root = args.campaign_root.resolve()
    rows = collect(root)
    delete_rows = [row for row in rows if row["action"] == "delete"]
    plan = {
        "format_version": "abpairfm_stage_a0_checkpoint_gc_plan_v1",
        "campaign_root": str(root),
        "verified_segment_count": len(rows),
        "keep_segments": list(KEEP_SEGMENTS),
        "delete_segment_count": len(delete_rows),
        "planned_reclaim_bytes": sum(row["checkpoint_bytes"] for row in delete_rows),
        "receipt_archive_required": True,
        "large_payload_hashes_reused_from_verified_dcp_receipts": True,
        "test_access": False,
        "segments": rows,
    }
    if not args.execute:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return

    archive = archive_receipts(root, rows)
    plan_path = root / "audits" / "CHECKPOINT_GC_PLAN_V1.json"
    if plan_path.exists():
        if plan_path.read_bytes() != canonical_bytes(plan):
            raise ValueError("existing GC plan differs")
    else:
        atomic_write(plan_path, plan)

    deleted: list[int] = []
    for row in delete_rows:
        segment = row["segment"]
        target = root / "checkpoints" / f"segment-{segment:06d}"
        if target.exists():
            if (
                target.is_symlink()
                or target.parent.resolve() != (root / "checkpoints").resolve()
                or SEGMENT_RE.fullmatch(target.name) is None
                or segment in KEEP_SEGMENTS
            ):
                raise ValueError(f"refusing unsafe deletion target: {target}")
            # Completed DCP directories are deliberately frozen read-only. Restore
            # owner write permission only on an already authenticated delete target.
            target.chmod(0o700)
            shutil.rmtree(target)
        if target.exists():
            raise RuntimeError(f"checkpoint deletion failed: segment {segment}")
        deleted.append(segment)

    kept = []
    for segment in KEEP_SEGMENTS:
        checkpoint = root / "checkpoints" / f"segment-{segment:06d}"
        receipt = json.loads((checkpoint / "receipt.json").read_text())
        tokens, content_sha = EXPECTED_MILESTONES[segment]
        if (
            receipt.get("content_sha256") != content_sha
            or receipt.get("training_state", {}).get("consumed_residue_tokens") != tokens
        ):
            raise ValueError(f"kept checkpoint verification failed: segment {segment}")
        kept.append(
            {
                "segment": segment,
                "consumed_residue_tokens": tokens,
                "dcp_content_sha256": content_sha,
                "receipt_sha256": sha256_file(checkpoint / "receipt.json"),
                "checkpoint_bytes": directory_size(checkpoint),
            }
        )

    completion = {
        "format_version": "abpairfm_stage_a0_checkpoint_gc_completion_v1",
        "status": "PASS_GC_VERIFIED",
        "plan_sha256": sha256_file(plan_path),
        "receipt_archive": str(archive),
        "archived_receipt_count": len(list(archive.glob("segment-*-receipt.json"))),
        "deleted_segments": deleted,
        "deleted_segment_count": len(deleted),
        "reclaimed_checkpoint_bytes": sum(
            row["checkpoint_bytes"] for row in delete_rows
        ),
        "kept": kept,
        "results_progress_manifests_logs_deleted": False,
        "test_access": False,
    }
    completion_path = root / "audits" / "CHECKPOINT_GC_COMPLETION_V1.json"
    if completion_path.exists():
        if completion_path.read_bytes() != canonical_bytes(completion):
            raise ValueError("existing GC completion differs")
    else:
        atomic_write(completion_path, completion)
    print(json.dumps(completion, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
