#!/usr/bin/env python3
"""Freeze a complete RePaRank checkpoint-transfer evaluation plan.

This program reads only metadata and file inventories. It never reads contact
labels, pair scores, test/OOD labels, or model performance metrics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def path_inventory(root: Path, pattern: str) -> dict[str, Any]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    digest = hashlib.sha256()
    count = 0
    total = 0
    for path in sorted(root.glob(pattern)):
        if not path.is_file():
            continue
        stat = path.stat()
        digest.update(
            f"{path.relative_to(root).as_posix()}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode()
        )
        count += 1
        total += stat.st_size
    return {
        "path": str(root.resolve()),
        "glob": pattern,
        "file_count": count,
        "total_bytes": total,
        "path_size_mtime_inventory_sha256": digest.hexdigest(),
        "content_digest": False,
    }


def structure_inventory(manifest_path: Path) -> dict[str, Any]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("format_version") != "frontier_structure_units_v2":
        raise ValueError("unsupported structure authority")
    if payload.get("contact_labels_loaded") is not False or payload.get("test_loaded") is not False:
        raise ValueError("structure authority violates contact/test firewall")
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("empty structure authority")
    digest = hashlib.sha256()
    base = manifest_path.parent
    seen = set()
    for record in sorted(records, key=lambda item: str(item["sequence_sha256"])):
        key = str(record["sequence_sha256"])
        if key in seen:
            raise ValueError(f"duplicate structure sequence {key}")
        seen.add(key)
        pdb = base / str(record["pdb_path"])
        if not pdb.is_file():
            raise FileNotFoundError(pdb)
        expected = str(record["pdb_sha256"])
        if len(expected) != 64:
            raise ValueError(f"invalid PDB digest for {key}")
        stat = pdb.stat()
        digest.update(f"{key}\0{expected}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode())
    return {
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256(manifest_path),
        "record_count": len(records),
        "counts_by_side": payload.get("counts_by_side"),
        "coordinate_policy": payload.get("coordinate_policy"),
        "pdb_name_sha_size_mtime_inventory_sha256": digest.hexdigest(),
        "contact_labels_loaded": False,
        "test_loaded": False,
    }


def freeze(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = args.checkpoint.resolve()
    source_script = args.source_script.resolve()
    split_path = args.split_manifest.resolve()
    canonical = args.canonical_manifest.resolve()
    contact_dir = args.contact_dir.resolve()
    audit = args.checkpoint_audit.resolve()
    canary = args.canary_result.resolve()
    for path in (checkpoint, source_script, split_path, canonical, audit, canary):
        if not path.is_file():
            raise FileNotFoundError(path)

    checkpoint_audit = json.loads(audit.read_text(encoding="utf-8"))
    if checkpoint_audit.get("status") != "PASS_CHECKPOINT_AUDIT":
        raise ValueError("checkpoint audit did not pass")
    if checkpoint_audit.get("checkpoint_sha256") != sha256(checkpoint):
        raise ValueError("checkpoint audit does not bind the requested checkpoint")
    if checkpoint_audit.get("mode") != args.mode:
        raise ValueError("checkpoint mode differs from requested evaluation mode")
    if checkpoint_audit.get("model_state_loaded_strictly") is not True:
        raise ValueError("checkpoint state was not loaded strictly")

    canary_payload = json.loads(canary.read_text(encoding="utf-8"))
    expected_canary = (
        "PASS_SEQUENCE_CANARY_REPEAT" if args.mode == "sequence" else "PASS_STRUCTURE_CANARY"
    )
    if canary_payload.get("status") != expected_canary:
        raise ValueError(f"canary did not pass the required gate {expected_canary}")
    if canary_payload.get("test_labels_read") is not False:
        raise ValueError("canary reports test-label access")

    split = json.loads(split_path.read_text(encoding="utf-8"))
    splits = {
        name: set(map(str, split.get("splits", {}).get(name, [])))
        for name in ("train", "validation", "test")
    }
    if len(splits["validation"]) != 217:
        raise ValueError("validation size differs from the registered authority")
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = splits[left] & splits[right]
        if overlap:
            raise ValueError(f"split overlap {left}/{right}: {sorted(overlap)[:10]}")

    inputs = {
        str(checkpoint): sha256(checkpoint),
        str(source_script): sha256(source_script),
        str(split_path): sha256(split_path),
        str(canonical): sha256(canonical),
        str(audit): sha256(audit),
        str(canary): sha256(canary),
    }
    structure = None
    if args.mode == "sequence_plus_structure":
        if args.structure_manifest is None:
            raise ValueError("structure mode requires --structure-manifest")
        structure = structure_inventory(args.structure_manifest.resolve())
        inputs[str(args.structure_manifest.resolve())] = sha256(args.structure_manifest.resolve())
    elif args.structure_manifest is not None:
        raise ValueError("sequence mode must not name a structure authority")

    contact = path_inventory(contact_dir, "*.npz")
    if contact["file_count"] != 3663:
        raise ValueError("contact store count differs from the registered authority")

    return {
        "format_version": "reparank_same_grid_evaluation_plan_v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "evidence_role": "exposed pretrained checkpoint transfer diagnostic on development validation",
        "mode": args.mode,
        "source_commit": args.source_commit,
        "official_source_commit": args.official_source_commit,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_audit": str(audit),
        "canary_result": str(canary),
        "source_script": str(source_script),
        "split_manifest": str(split_path),
        "canonical_manifest": str(canonical),
        "contact_dir": str(contact_dir),
        "input_sha256": inputs,
        "contact_inventory": contact,
        "structure_authority": structure,
        "validation_instance_count": len(splits["validation"]),
        "primary_endpoint": "macro mean-per-complex non-interpolated pair AP",
        "score_definition": "negative Euclidean distance between official RePaRank complex embeddings",
        "candidate_grid": "exact existing complete antibody-by-antigen residue grid",
        "raw_pair_predictions": True,
        "checkpoint_training_overlap_audited": False,
        "test_authorized": False,
        "ood_authorized": False,
        "blind_claim_authorized": False,
        "sota_claim_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("sequence", "sequence_plus_structure"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-audit", type=Path, required=True)
    parser.add_argument("--canary-result", type=Path, required=True)
    parser.add_argument("--source-script", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--official-source-commit", required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--canonical-manifest", type=Path, required=True)
    parser.add_argument("--contact-dir", type=Path, required=True)
    parser.add_argument("--structure-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = freeze(args)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({
        "output": str(output),
        "sha256": sha256(output),
        "mode": payload["mode"],
        "validation_instance_count": payload["validation_instance_count"],
        "contact_files": payload["contact_inventory"]["file_count"],
        "structure_records": (
            payload["structure_authority"]["record_count"]
            if payload["structure_authority"] else None
        ),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
