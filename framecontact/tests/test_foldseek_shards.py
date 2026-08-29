"""End-to-end contracts for sharded exhaustive Foldseek provenance."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest


def _authority(root: Path) -> tuple[Path, str, list[str]]:
    structures = root / "structures"
    structures.mkdir(parents=True)
    identifiers = [hashlib.sha256(f"structure-{index}".encode()).hexdigest() for index in range(4)]
    rows = []
    for index, identifier in enumerate(identifiers):
        path = structures / f"{identifier}.pdb"
        path.write_text("END\n", encoding="utf-8")
        rows.append(
            {
                "antigen_structure_id": f"case_{index}",
                "foldseek_structure_id": identifier,
                "relative_path": f"structures/{identifier}.pdb",
                "residue_count": str(10 + index),
            }
        )
    manifest = root / "ANTIGEN_STRUCTURE_EXPORT_MANIFEST.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return manifest, hashlib.sha256(manifest.read_bytes()).hexdigest(), identifiers


def test_sharded_rectangles_merge_to_a_complete_cartesian_receipt(tmp_path: Path) -> None:
    from interaction_first.foldseek_shards import (
        finalize_foldseek_shard,
        merge_and_normalize_foldseek_shards,
        prepare_foldseek_query_shards,
    )

    export_root = tmp_path / "export"
    manifest, manifest_sha, identifiers = _authority(export_root)
    plan_root = tmp_path / "plan"
    prepare_foldseek_query_shards(
        export_root=export_root,
        manifest_sha256=manifest_sha,
        output_root=plan_root,
        shard_count=2,
    )
    plan_path = plan_root / "FOLDSEEK_SHARD_PLAN.json"
    plan_sha = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    shard_output_root = tmp_path / "shard-outputs"
    for shard in plan["shards"]:
        index = int(shard["shard_index"])
        root = shard_output_root / f"shard-{index:03d}"
        root.mkdir(parents=True)
        raw = root / "foldseek.raw.tsv"
        with raw.open("w", encoding="utf-8") as handle:
            for query in shard["foldseek_structure_ids"]:
                for target in identifiers:
                    handle.write(f"{query}.pdb\t{target}.pdb\t0.4\t0.5\n")
        command = root / "COMMAND.txt"
        command.write_text(f"foldseek shard {index}\n", encoding="utf-8")
        raw_sha = hashlib.sha256(raw.read_bytes()).hexdigest()
        finalize_foldseek_shard(
            plan_path=plan_path,
            plan_sha256=plan_sha,
            shard_index=index,
            raw_path=raw,
            raw_sha256=raw_sha,
            command_path=command,
            receipt_path=root / "FOLDSEEK_SHARD_RECEIPT.json",
            foldseek_version="test-version",
        )
    merged = tmp_path / "merged" / "foldseek.raw.tsv"
    normalized = tmp_path / "merged" / "foldseek.normalized.tsv"
    receipt_path = tmp_path / "merged" / "FOLDSEEK_RECEIPT.json"
    receipt = merge_and_normalize_foldseek_shards(
        shard_output_root=shard_output_root,
        plan_path=plan_path,
        plan_sha256=plan_sha,
        export_manifest_path=manifest,
        export_manifest_sha256=manifest_sha,
        raw_output_path=merged,
        normalized_output_path=normalized,
        receipt_path=receipt_path,
        foldseek_version="test-version",
    )
    assert receipt["complete_ordered_cartesian_grid"] is True
    assert receipt["raw_row_count"] == 16
    assert sum(1 for _ in normalized.open(encoding="utf-8")) == 17


def test_shard_finalizer_rejects_a_duplicate_pair(tmp_path: Path) -> None:
    from interaction_first.foldseek_shards import (
        finalize_foldseek_shard,
        prepare_foldseek_query_shards,
    )

    export_root = tmp_path / "export"
    _, manifest_sha, identifiers = _authority(export_root)
    plan_root = tmp_path / "plan"
    prepare_foldseek_query_shards(
        export_root=export_root,
        manifest_sha256=manifest_sha,
        output_root=plan_root,
        shard_count=2,
    )
    plan_path = plan_root / "FOLDSEEK_SHARD_PLAN.json"
    plan_sha = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    query = plan["shards"][0]["foldseek_structure_ids"][0]
    raw = tmp_path / "bad.tsv"
    raw.write_text(
        f"{query}\t{identifiers[0]}\t0.1\t0.2\n"
        f"{query}\t{identifiers[0]}\t0.1\t0.2\n",
        encoding="utf-8",
    )
    command = tmp_path / "command.txt"
    command.write_text("test\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        finalize_foldseek_shard(
            plan_path=plan_path,
            plan_sha256=plan_sha,
            shard_index=0,
            raw_path=raw,
            raw_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),
            command_path=command,
            receipt_path=tmp_path / "receipt.json",
            foldseek_version="test",
        )
