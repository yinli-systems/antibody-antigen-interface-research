#!/usr/bin/env python3
"""Verify and freeze four PINDER-train S40/80 sequence relations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping


VIEWS = ("vh", "vl", "concatenated_antibody", "antigen")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def atomic_create(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run(protocol_path: Path, output_dir: Path) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("format_version") != "pinder_antibody_s40_relation_protocol_v1":
        raise ValueError("protocol format mismatch")
    if tuple(protocol.get("required_views", ())) != VIEWS:
        raise ValueError("protocol view order mismatch")
    if protocol.get("thresholds") != {
        "aligned_region_identity_minimum": 0.4,
        "minimum_reciprocal_coverage": 0.8,
        "threshold_equality_included": True,
    }:
        raise ValueError("protocol thresholds changed")

    relation_artifacts: dict[str, dict[str, Any]] = {}
    for view in VIEWS:
        receipt_path = output_dir / f"{view}.identity40_coverage80.receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("status") != "PASS_PINDER_ANTIBODY_S2_MMSEQS_RELATION":
            raise ValueError(f"{view} relation did not pass")
        if receipt.get("role") != view or receipt.get("thresholds") != {
            "identity": 0.4,
            "reciprocal_coverage": 0.8,
        }:
            raise ValueError(f"{view} relation policy mismatch")
        if receipt.get("test_access") is not False or receipt.get("training_authorized") is not False:
            raise ValueError(f"{view} relation crossed its authorization boundary")
        source = receipt.get("source", {}).get("fasta", {})
        if source.get("sha256") != protocol["source"]["fasta_sha256"][view]:
            raise ValueError(f"{view} FASTA authority mismatch")
        relation = receipt.get("canonical_relation", {})
        relation_path = Path(relation.get("path", ""))
        if not relation_path.is_file() or sha256_file(relation_path) != relation.get("sha256"):
            raise ValueError(f"{view} canonical relation SHA mismatch")
        relation_artifacts[view] = {
            "receipt_sha256": sha256_file(receipt_path),
            "relation_sha256": relation["sha256"],
            "canonical_row_count": receipt["canonical_row_count"],
        }

    result: dict[str, Any] = {
        "format_version": "pinder_antibody_s40_relation_result_v1",
        "status": "PASS_PINDER_TRAIN_FOUR_VIEW_S40_RELATIONS_MATERIALIZED",
        "protocol_sha256": sha256_file(protocol_path),
        "relations": relation_artifacts,
        "thresholds": protocol["thresholds"],
        "authorization": protocol["authorization"],
        "training_authorized": False,
        "model_selection_authorized": False,
        "required_next_gate": (
            "Build indivisible four-view similarity components, construct a split, "
            "then pass an independent zero-crossing audit."
        ),
        "claim_boundary": protocol["claim_boundary"],
    }
    result["scientific_payload_sha256"] = canonical_sha256(result)
    result_path = output_dir / "PINDER_ANTIBODY_S40_RELATION_RESULT_V1.json"
    atomic_create(result_path, result)
    receipt = {
        "format_version": "pinder_antibody_s40_relation_receipt_v1",
        "result_sha256": sha256_file(result_path),
        "scientific_payload_sha256": result["scientific_payload_sha256"],
        "training_authorized": False,
        "model_selection_authorized": False,
    }
    atomic_create(output_dir / "PINDER_ANTIBODY_S40_RELATION_RECEIPT_V1.json", receipt)
    return result


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()
    run(arguments.protocol.resolve(strict=True), arguments.output_dir.resolve(strict=True))
