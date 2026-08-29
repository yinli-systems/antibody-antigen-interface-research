"""Deterministic, fail-closed sharding for exhaustive Foldseek all-vs-all."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any

from .foldseek_normalize import normalize_foldseek_alignments
from .sabdab2_structure import (
    authenticate_file,
    canonical_json_bytes,
    sha256_file,
)


SHARD_PLAN_FORMAT = "framecontact_foldseek_shard_plan_v1"
SHARD_RECEIPT_FORMAT = "framecontact_foldseek_shard_receipt_v1"
_HASH = re.compile(r"[0-9a-f]{64}")


def prepare_foldseek_query_shards(
    *,
    export_root: Path,
    manifest_sha256: str,
    output_root: Path,
    shard_count: int,
) -> dict[str, Any]:
    """Create residue-count-balanced query shards using symlinks only."""
    if isinstance(shard_count, bool) or not isinstance(shard_count, int) or shard_count < 2:
        raise ValueError("shard_count must be an integer of at least two")
    manifest_path = export_root / "ANTIGEN_STRUCTURE_EXPORT_MANIFEST.csv"
    authenticated_manifest = authenticate_file(
        manifest_path, manifest_sha256, "antigen export manifest"
    )
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "antigen_structure_id",
        "foldseek_structure_id",
        "relative_path",
        "residue_count",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError("Antigen export manifest schema is incomplete")
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError("Refusing to overwrite Foldseek shard plan")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent)
    )
    try:
        assignments: list[list[dict[str, str]]] = [[] for _ in range(shard_count)]
        loads = [0] * shard_count
        for row in sorted(
            rows,
            key=lambda value: (
                -int(value["residue_count"]),
                value["antigen_structure_id"],
            ),
        ):
            shard_index = min(range(shard_count), key=lambda index: (loads[index], index))
            assignments[shard_index].append(row)
            loads[shard_index] += int(row["residue_count"])
        shards = []
        seen: set[str] = set()
        for shard_index, assignment in enumerate(assignments):
            structure_root = temporary / f"shard-{shard_index:03d}" / "structures"
            structure_root.mkdir(parents=True)
            identifiers = []
            for row in sorted(assignment, key=lambda value: value["antigen_structure_id"]):
                foldseek_id = row["foldseek_structure_id"]
                if foldseek_id in seen:
                    raise ValueError("Foldseek structure identifiers must be unique")
                seen.add(foldseek_id)
                source = (export_root / row["relative_path"]).resolve(strict=True)
                expected_name = f"{foldseek_id}{source.suffix}"
                if source.name != expected_name:
                    raise ValueError("Export relative path differs from Foldseek identifier")
                os.symlink(source, structure_root / expected_name)
                identifiers.append(foldseek_id)
            shards.append(
                {
                    "shard_index": shard_index,
                    "query_count": len(identifiers),
                    "query_residue_count": loads[shard_index],
                    "expected_raw_row_count": len(identifiers) * len(rows),
                    "foldseek_structure_ids": identifiers,
                }
            )
        if seen != {row["foldseek_structure_id"] for row in rows}:
            raise ValueError("Shard plan does not partition every exported structure")
        payload = {
            "format_version": SHARD_PLAN_FORMAT,
            "authority": {
                "export_manifest_sha256": authenticated_manifest,
                "task_labels_read": False,
            },
            "structure_count": len(rows),
            "shard_count": shard_count,
            "assignment_rule": (
                "deterministic greedy least-loaded bin by descending residue count, "
                "tie-broken by shard index"
            ),
            "shards": shards,
        }
        plan_path = temporary / "FOLDSEEK_SHARD_PLAN.json"
        plan_path.write_bytes(canonical_json_bytes(payload))
        os.replace(temporary, output_root)
        return {
            **payload,
            "plan_sha256": sha256_file(output_root / plan_path.name),
            "output_root": str(output_root),
        }
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _raw_identifier(value: str, known: set[str]) -> str:
    match = _HASH.search(value)
    if match is None or match.group(0) not in known:
        raise ValueError(f"Unknown Foldseek identifier: {value!r}")
    return match.group(0)


def finalize_foldseek_shard(
    *,
    plan_path: Path,
    plan_sha256: str,
    shard_index: int,
    raw_path: Path,
    raw_sha256: str,
    command_path: Path,
    receipt_path: Path,
    foldseek_version: str,
) -> dict[str, Any]:
    """Prove one query shard is a complete unique query-by-target rectangle."""
    authenticate_file(plan_path, plan_sha256, "Foldseek shard plan")
    authenticated_raw = authenticate_file(raw_path, raw_sha256, "Foldseek shard raw output")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("format_version") != SHARD_PLAN_FORMAT:
        raise ValueError("Unknown Foldseek shard plan format")
    shards = {int(value["shard_index"]): value for value in plan["shards"]}
    if shard_index not in shards:
        raise ValueError("Shard index is absent from the plan")
    shard = shards[shard_index]
    queries = tuple(shard["foldseek_structure_ids"])
    targets = {
        identifier
        for value in plan["shards"]
        for identifier in value["foldseek_structure_ids"]
    }
    query_index = {identifier: index for index, identifier in enumerate(queries)}
    target_index = {
        identifier: index for index, identifier in enumerate(sorted(targets))
    }
    expected = len(queries) * len(targets)
    observed = bytearray(expected)
    raw_rows = 0
    with raw_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 4:
                raise ValueError(f"Raw shard row {line_number} is malformed")
            query = _raw_identifier(fields[0], set(query_index))
            target = _raw_identifier(fields[1], targets)
            query_tm, target_tm = float(fields[2]), float(fields[3])
            if not (0 <= query_tm <= 1 and 0 <= target_tm <= 1):
                raise ValueError("Foldseek TM scores must lie in [0, 1]")
            flat_index = query_index[query] * len(targets) + target_index[target]
            if observed[flat_index]:
                raise ValueError("Foldseek shard contains a duplicate ordered pair")
            observed[flat_index] = 1
            raw_rows += 1
    if raw_rows != expected or 0 in observed:
        raise ValueError(
            f"Foldseek shard is incomplete: rows={raw_rows}, expected={expected}, "
            f"missing={observed.count(0)}"
        )
    if receipt_path.exists():
        raise FileExistsError("Refusing to overwrite Foldseek shard receipt")
    payload = {
        "format_version": SHARD_RECEIPT_FORMAT,
        "shard_index": shard_index,
        "query_count": len(queries),
        "target_count": len(targets),
        "raw_row_count": raw_rows,
        "complete_unique_rectangle": True,
        "raw_sha256": authenticated_raw,
        "plan_sha256": plan_sha256,
        "command_sha256": sha256_file(command_path),
        "foldseek_version": foldseek_version,
        "task_labels_read": False,
    }
    receipt_path.write_bytes(canonical_json_bytes(payload))
    return payload


def merge_and_normalize_foldseek_shards(
    *,
    shard_output_root: Path,
    plan_path: Path,
    plan_sha256: str,
    export_manifest_path: Path,
    export_manifest_sha256: str,
    raw_output_path: Path,
    normalized_output_path: Path,
    receipt_path: Path,
    foldseek_version: str,
) -> dict[str, Any]:
    """Authenticate, merge, and fully normalize all shard rectangles."""
    authenticate_file(plan_path, plan_sha256, "Foldseek shard plan")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("format_version") != SHARD_PLAN_FORMAT:
        raise ValueError("Unknown Foldseek shard plan format")
    if raw_output_path.exists():
        raise FileExistsError("Refusing to overwrite merged Foldseek raw output")
    raw_output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{raw_output_path.name}.", suffix=".tmp", dir=raw_output_path.parent
    )
    temporary_path = Path(temporary_name)
    command_hashes = []
    try:
        with os.fdopen(descriptor, "wb") as target:
            for shard in sorted(plan["shards"], key=lambda value: value["shard_index"]):
                root = shard_output_root / f"shard-{int(shard['shard_index']):03d}"
                raw_path = root / "foldseek.raw.tsv"
                receipt = json.loads(
                    (root / "FOLDSEEK_SHARD_RECEIPT.json").read_text(encoding="utf-8")
                )
                if (
                    receipt.get("format_version") != SHARD_RECEIPT_FORMAT
                    or receipt.get("complete_unique_rectangle") is not True
                    or int(receipt.get("shard_index", -1)) != int(shard["shard_index"])
                    or receipt.get("plan_sha256") != plan_sha256
                    or sha256_file(raw_path) != receipt.get("raw_sha256")
                ):
                    raise ValueError("Foldseek shard receipt authentication failed")
                command_hashes.append(receipt["command_sha256"])
                with raw_path.open("rb") as source:
                    shutil.copyfileobj(source, target, length=16 * 1024 * 1024)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary_path, raw_output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    raw_sha256 = sha256_file(raw_output_path)
    command = [
        "sharded-exhaustive-foldseek",
        f"shard_count={plan['shard_count']}",
        f"plan_sha256={plan_sha256}",
        *(f"shard_command_sha256={value}" for value in command_hashes),
    ]
    return normalize_foldseek_alignments(
        raw_path=raw_output_path,
        raw_sha256=raw_sha256,
        export_manifest_path=export_manifest_path,
        export_manifest_sha256=export_manifest_sha256,
        output_path=normalized_output_path,
        receipt_path=receipt_path,
        foldseek_version=foldseek_version,
        command=command,
    )


__all__ = [
    "SHARD_PLAN_FORMAT",
    "SHARD_RECEIPT_FORMAT",
    "finalize_foldseek_shard",
    "merge_and_normalize_foldseek_shards",
    "prepare_foldseek_query_shards",
]
