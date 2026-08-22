#!/usr/bin/env python3
"""Freeze authenticated development pairs and reciprocal retrieval candidates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import torch


AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
PANEL_FORMAT = "framecontact_reciprocal_partner_panel_v2"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def checked_sequence(value: object, label: str, *, optional: bool = False) -> str:
    sequence = str(value).strip().upper()
    if not sequence and optional:
        return ""
    if not sequence or not set(sequence).issubset(AMINO_ACIDS):
        raise ValueError(f"{label} is empty or noncanonical")
    return sequence


def framed_antibody_sha256(heavy: str, light: str) -> str:
    """Match the label-independent AsEP partner-metadata authority."""
    return sha256_text(f"H\0{heavy}\0L\0{light}")


def canonical_pair_sha256(heavy: str, light: str, antigen: str) -> str:
    return sha256_text(f"H\0{heavy}\0L\0{light}\0AG\0{antigen}")


def load_json_authority(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    if path.is_symlink() or sha256_file(path) != expected_sha256:
        raise ValueError(f"{label} SHA-256 differs")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON object")
    return value


def load_panel(
    path: Path, expected_sha256: str, expected_ids: set[str], label: str
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    panel = load_json_authority(path, expected_sha256, label)
    if (
        panel.get("format_version") != PANEL_FORMAT
        or panel.get("role") != "validation"
        or panel.get("panel_size_per_direction") != 31
    ):
        raise ValueError(f"{label} schema differs")
    entries = panel.get("entries")
    if not isinstance(entries, list):
        raise ValueError(f"{label} entries are missing")
    indexed: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError(f"{label} entry is not an object")
        identifier = str(entry.get("instance_id", ""))
        if not identifier or identifier in indexed:
            raise ValueError(f"{label} contains an empty or duplicate instance")
        indexed[identifier] = entry
    if set(indexed) != expected_ids:
        raise ValueError(f"{label} instance set differs from validation authority")
    for identifier, entry in indexed.items():
        for key in ("wrong_antibody_instance_ids", "wrong_antigen_instance_ids"):
            candidates = entry.get(key)
            if (
                not isinstance(candidates, list)
                or len(candidates) != 31
                or len(set(candidates)) != 31
                or identifier in candidates
                or not set(candidates).issubset(expected_ids)
            ):
                raise ValueError(f"{label} invalid candidates for {identifier}: {key}")
    return indexed, panel


def pair_record(
    *,
    dataset: str,
    identifier: str,
    heavy: str,
    light: str,
    antigen: str,
    source_antibody_sha256: str,
    source_antigen_sha256: str,
    antibody_cluster_id: str,
    antigen_cluster_id: str,
    panel_entry: dict[str, Any],
) -> dict[str, Any]:
    return {
        "antibody_cluster_id": antibody_cluster_id,
        "antibody_endpoint_sequence": heavy + light,
        "antibody_endpoint_sha256": sha256_text(heavy + light),
        "antigen_cluster_id": antigen_cluster_id,
        "antigen_sequence": antigen,
        "antigen_sequence_sha256": sha256_text(antigen),
        "canonical_pair_sha256": canonical_pair_sha256(heavy, light, antigen),
        "dataset": dataset,
        "heavy_sequence": heavy,
        "heavy_sequence_sha256": sha256_text(heavy),
        "instance_id": identifier,
        "light_sequence": light,
        "light_sequence_sha256": sha256_text(light) if light else "",
        "role": "development_validation",
        "source_antibody_sequence_sha256": source_antibody_sha256,
        "source_antigen_sequence_sha256": source_antigen_sha256,
        "wrong_antibody_instance_ids": panel_entry["wrong_antibody_instance_ids"],
        "wrong_antigen_instance_ids": panel_entry["wrong_antigen_instance_ids"],
    }


def load_asep_pairs(
    root: Path,
    manifest_path: Path,
    manifest_sha256: str,
    panel_path: Path,
    panel_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = load_json_authority(
        manifest_path, manifest_sha256, "AsEP development manifest"
    )
    if manifest.get("checks", {}).get("official_test_members_materialized") is not False:
        raise ValueError("AsEP test boundary is not sealed")
    validation = {
        str(record["instance_id"]): record
        for record in manifest.get("records", [])
        if record.get("role") == "val"
    }
    if not validation:
        raise ValueError("AsEP validation authority is empty")
    panel, panel_payload = load_panel(
        panel_path, panel_sha256, set(validation), "AsEP validation panel"
    )
    records: list[dict[str, Any]] = []
    for identifier in sorted(validation):
        source = validation[identifier]
        artifact_path = root / source["artifact_path"]
        if artifact_path.is_symlink() or sha256_file(artifact_path) != source["artifact_sha256"]:
            raise ValueError(f"AsEP artifact SHA-256 differs: {identifier}")
        artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
        if artifact.get("instance_id") != identifier or artifact.get("role") != "val":
            raise ValueError(f"AsEP artifact identity differs: {identifier}")
        heavy = checked_sequence(artifact.get("heavy_sequence"), f"AsEP {identifier} H")
        light = checked_sequence(artifact.get("light_sequence"), f"AsEP {identifier} L")
        antigen = checked_sequence(
            artifact.get("antigen_sequence_full"), f"AsEP {identifier} antigen"
        )
        source_antibody = framed_antibody_sha256(heavy, light)
        source_antigen = sha256_text(antigen)
        entry = panel[identifier]
        if (
            entry.get("antibody_cluster_id") != f"exact:{source_antibody}"
            or entry.get("antigen_cluster_id") != f"exact:{source_antigen}"
        ):
            raise ValueError(f"AsEP panel endpoint identity differs: {identifier}")
        records.append(
            pair_record(
                dataset="asep",
                identifier=identifier,
                heavy=heavy,
                light=light,
                antigen=antigen,
                source_antibody_sha256=source_antibody,
                source_antigen_sha256=source_antigen,
                antibody_cluster_id=str(entry["antibody_cluster_id"]),
                antigen_cluster_id=str(entry["antigen_cluster_id"]),
                panel_entry=entry,
            )
        )
    return records, {
        "complexes": len(records),
        "missing_light_chain_complexes": 0,
        "panel_authority_sha256": sha256_text(
            json.dumps(panel_payload.get("authority", {}), sort_keys=True, separators=(",", ":"))
        ),
    }


def load_sabdab2_pairs(
    split_path: Path,
    split_sha256: str,
    canonical_path: Path,
    canonical_sha256: str,
    panel_path: Path,
    panel_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if split_path.is_symlink() or sha256_file(split_path) != split_sha256:
        raise ValueError("SAbDab2 split SHA-256 differs")
    if canonical_path.is_symlink() or sha256_file(canonical_path) != canonical_sha256:
        raise ValueError("SAbDab2 canonical authority SHA-256 differs")
    with split_path.open("rt", encoding="utf-8", newline="") as handle:
        validation = {
            str(row["instance_id"]): row
            for row in csv.DictReader(handle)
            if row.get("role") == "validation"
        }
    if not validation:
        raise ValueError("SAbDab2 validation authority is empty")
    panel, panel_payload = load_panel(
        panel_path, panel_sha256, set(validation), "SAbDab2 validation panel"
    )
    canonical: dict[str, dict[str, str]] = {}
    with canonical_path.open("rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            identifier = str(row["INSTANCE"])
            if identifier in validation:
                if identifier in canonical:
                    raise ValueError(f"Duplicate SAbDab2 canonical instance: {identifier}")
                canonical[identifier] = row
    if set(canonical) != set(validation):
        raise ValueError("SAbDab2 validation instances lack canonical sequences")
    records: list[dict[str, Any]] = []
    missing_light = 0
    for identifier in sorted(validation):
        source = validation[identifier]
        sequence = canonical[identifier]
        heavy = checked_sequence(sequence.get("vh_seq"), f"SAbDab2 {identifier} H")
        light = checked_sequence(
            sequence.get("vl_seq"), f"SAbDab2 {identifier} L", optional=True
        )
        antigen = checked_sequence(sequence.get("ag_seq"), f"SAbDab2 {identifier} antigen")
        if not light:
            missing_light += 1
        if (
            sha256_text(heavy + light) != source["antibody_sequence_sha256"]
            or sha256_text(antigen) != source["antigen_sequence_sha256"]
        ):
            raise ValueError(f"SAbDab2 endpoint digest differs: {identifier}")
        entry = panel[identifier]
        antibody_cluster = f"ab:{source['antibody_cluster_id']}"
        antigen_cluster = f"ag:{source['antigen_cluster_id']}"
        if (
            entry.get("antibody_cluster_id") != antibody_cluster
            or entry.get("antigen_cluster_id") != antigen_cluster
        ):
            raise ValueError(f"SAbDab2 panel cluster differs: {identifier}")
        records.append(
            pair_record(
                dataset="sabdab2",
                identifier=identifier,
                heavy=heavy,
                light=light,
                antigen=antigen,
                source_antibody_sha256=source["antibody_sequence_sha256"],
                source_antigen_sha256=source["antigen_sequence_sha256"],
                antibody_cluster_id=antibody_cluster,
                antigen_cluster_id=antigen_cluster,
                panel_entry=entry,
            )
        )
    return records, {
        "complexes": len(records),
        "missing_light_chain_complexes": missing_light,
        "panel_authority_sha256": sha256_text(
            json.dumps(panel_payload.get("authority", {}), sort_keys=True, separators=(",", ":"))
        ),
    }


def validate_decoy_endpoint_exclusion(records: list[dict[str, Any]]) -> None:
    by_dataset = {
        dataset: {record["instance_id"]: record for record in records if record["dataset"] == dataset}
        for dataset in {record["dataset"] for record in records}
    }
    for record in records:
        index = by_dataset[record["dataset"]]
        for identifier in record["wrong_antibody_instance_ids"]:
            if index[identifier]["source_antibody_sequence_sha256"] == record["source_antibody_sequence_sha256"]:
                raise ValueError(f"Wrong-antibody panel contains exact endpoint: {record['instance_id']}")
        for identifier in record["wrong_antigen_instance_ids"]:
            if index[identifier]["source_antigen_sequence_sha256"] == record["source_antigen_sequence_sha256"]:
                raise ValueError(f"Wrong-antigen panel contains exact endpoint: {record['instance_id']}")


def publish_create_only(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_path, path)
        path.chmod(0o444)
    except FileExistsError as error:
        raise FileExistsError(f"Refusing to overwrite development pair artifact: {path}") from error
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asep-root", type=Path, required=True)
    parser.add_argument("--asep-manifest", type=Path, required=True)
    parser.add_argument("--asep-manifest-sha256", required=True)
    parser.add_argument("--asep-panel", type=Path, required=True)
    parser.add_argument("--asep-panel-sha256", required=True)
    parser.add_argument("--sabdab2-split", type=Path, required=True)
    parser.add_argument("--sabdab2-split-sha256", required=True)
    parser.add_argument("--sabdab2-canonical", type=Path, required=True)
    parser.add_argument("--sabdab2-canonical-sha256", required=True)
    parser.add_argument("--sabdab2-panel", type=Path, required=True)
    parser.add_argument("--sabdab2-panel-sha256", required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    arguments = parser.parse_args()

    asep, asep_counts = load_asep_pairs(
        arguments.asep_root.resolve(),
        arguments.asep_manifest.resolve(),
        arguments.asep_manifest_sha256,
        arguments.asep_panel.resolve(),
        arguments.asep_panel_sha256,
    )
    sabdab2, sabdab2_counts = load_sabdab2_pairs(
        arguments.sabdab2_split.resolve(),
        arguments.sabdab2_split_sha256,
        arguments.sabdab2_canonical.resolve(),
        arguments.sabdab2_canonical_sha256,
        arguments.sabdab2_panel.resolve(),
        arguments.sabdab2_panel_sha256,
    )
    records = sorted(asep + sabdab2, key=lambda value: (value["dataset"], value["instance_id"]))
    validate_decoy_endpoint_exclusion(records)
    payload = b"".join(
        (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        for record in records
    )
    output = arguments.output_jsonl.resolve()
    publish_create_only(output, payload)
    receipt = {
        "authorization": {
            "development_pair_mlm": True,
            "development_retrieval": True,
            "development_contact_labels": False,
            "production_training": False,
            "test_access": False,
        },
        "claim_boundary": "Authenticated AsEP and SAbDab2 validation sequences, cognate pair mappings, and label-independent reciprocal decoys for development-only pair-MLM and retrieval. Contact labels, predictions, metrics, and every test role remain excluded.",
        "counts": {
            "asep": asep_counts,
            "sabdab2": sabdab2_counts,
            "records": len(records),
            "wrong_partner_candidates_per_direction_per_record": 31,
        },
        "format_version": "abpairfm_development_pair_panel_v1",
        "output_jsonl": {
            "bytes": output.stat().st_size,
            "sha256": sha256_file(output),
        },
        "source_sha256": {
            "asep_manifest": arguments.asep_manifest_sha256,
            "asep_panel": arguments.asep_panel_sha256,
            "sabdab2_canonical": arguments.sabdab2_canonical_sha256,
            "sabdab2_panel": arguments.sabdab2_panel_sha256,
            "sabdab2_split": arguments.sabdab2_split_sha256,
        },
        "status": "FROZEN_DEVELOPMENT_PAIR_PANEL",
        "test_access": False,
    }
    publish_create_only(
        arguments.receipt.resolve(),
        (json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"),
    )
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
