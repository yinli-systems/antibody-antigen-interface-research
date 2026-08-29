#!/usr/bin/env python3
"""Freeze exact AsEP and SAbDab2 validation sequences for train decontamination."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile

import torch


AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def checked_sequence(value: object, label: str) -> str:
    sequence = str(value).strip().upper()
    if not sequence or not set(sequence).issubset(AMINO_ACIDS):
        raise ValueError(f"{label} is empty or noncanonical")
    return sequence


def optional_sequence(value: object, label: str) -> str:
    sequence = str(value).strip().upper()
    if not sequence:
        return ""
    return checked_sequence(sequence, label)


def publish_create_only(path: Path, payload: bytes, mode: int = 0o444) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_path, path)
        path.chmod(mode)
    except FileExistsError as error:
        raise FileExistsError(f"Refusing to overwrite development artifact: {path}") from error
    finally:
        temporary_path.unlink(missing_ok=True)


def add_sequence(
    sequences: dict[str, tuple[str, str]], header: str, sequence: str
) -> None:
    digest = sequence_sha256(sequence)
    previous = sequences.get(digest)
    candidate = (header, sequence)
    if previous is None or candidate[0] < previous[0]:
        sequences[digest] = candidate
    elif previous[1] != sequence:
        raise AssertionError("SHA-256 sequence collision")


def load_asep(
    root: Path, manifest_path: Path, expected_sha256: str
) -> tuple[dict[str, tuple[str, str]], dict[str, int]]:
    if sha256_file(manifest_path) != expected_sha256:
        raise ValueError("AsEP development manifest SHA-256 differs")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("checks", {}).get("official_test_members_materialized") is not False:
        raise ValueError("AsEP test boundary is not sealed")
    sequences: dict[str, tuple[str, str]] = {}
    validation_records = [record for record in manifest["records"] if record["role"] == "val"]
    for record in validation_records:
        artifact_path = root / record["artifact_path"]
        if sha256_file(artifact_path) != record["artifact_sha256"]:
            raise ValueError(f"AsEP artifact SHA-256 differs: {record['instance_id']}")
        artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
        if artifact.get("role") != "val" or artifact.get("instance_id") != record["instance_id"]:
            raise ValueError(f"AsEP validation artifact identity differs: {record['instance_id']}")
        for partner, key in (
            ("heavy", "heavy_sequence"),
            ("light", "light_sequence"),
            ("antigen", "antigen_sequence_full"),
        ):
            sequence = checked_sequence(artifact[key], f"AsEP {record['instance_id']} {partner}")
            add_sequence(sequences, f"asep|{record['instance_id']}|{partner}", sequence)
    return sequences, {
        "complexes": len(validation_records),
        "sequence_observations": len(validation_records) * 3,
    }


def load_sabdab2(
    split_path: Path,
    expected_split_sha256: str,
    canonical_path: Path,
    expected_canonical_sha256: str,
) -> tuple[dict[str, tuple[str, str]], dict[str, int]]:
    if sha256_file(split_path) != expected_split_sha256:
        raise ValueError("SAbDab2 split manifest SHA-256 differs")
    if sha256_file(canonical_path) != expected_canonical_sha256:
        raise ValueError("SAbDab2 canonical sequence authority SHA-256 differs")
    with split_path.open("rt", encoding="utf-8", newline="") as handle:
        split_rows = {
            row["instance_id"]: row
            for row in csv.DictReader(handle)
            if row["role"] == "validation"
        }
    if not split_rows:
        raise ValueError("SAbDab2 validation split is empty")
    selected: dict[str, dict[str, str]] = {}
    with canonical_path.open("rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            identifier = row["INSTANCE"]
            if identifier in split_rows:
                if identifier in selected:
                    raise ValueError(f"Duplicate SAbDab2 canonical instance: {identifier}")
                selected[identifier] = row
    missing = sorted(set(split_rows) - set(selected))
    if missing:
        raise ValueError(f"SAbDab2 validation instances lack sequences: {missing[:5]}")

    sequences: dict[str, tuple[str, str]] = {}
    missing_light_count = 0
    sequence_observations = 0
    for identifier in sorted(selected):
        source = selected[identifier]
        split = split_rows[identifier]
        heavy = checked_sequence(source["vh_seq"], f"SAbDab2 {identifier} heavy")
        light = optional_sequence(source["vl_seq"], f"SAbDab2 {identifier} light")
        antigen = checked_sequence(source["ag_seq"], f"SAbDab2 {identifier} antigen")
        if sequence_sha256(heavy + light) != split["antibody_sequence_sha256"]:
            raise ValueError(f"SAbDab2 antibody sequence digest differs: {identifier}")
        if sequence_sha256(antigen) != split["antigen_sequence_sha256"]:
            raise ValueError(f"SAbDab2 antigen sequence digest differs: {identifier}")
        add_sequence(sequences, f"sabdab2|{identifier}|heavy", heavy)
        sequence_observations += 2
        if light:
            add_sequence(sequences, f"sabdab2|{identifier}|light", light)
            sequence_observations += 1
        else:
            missing_light_count += 1
        add_sequence(sequences, f"sabdab2|{identifier}|antigen", antigen)
    return sequences, {
        "complexes": len(selected),
        "missing_light_chain_complexes": missing_light_count,
        "sequence_observations": sequence_observations,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asep-root", type=Path, required=True)
    parser.add_argument("--asep-manifest", type=Path, required=True)
    parser.add_argument("--asep-manifest-sha256", required=True)
    parser.add_argument("--sabdab2-split", type=Path, required=True)
    parser.add_argument("--sabdab2-split-sha256", required=True)
    parser.add_argument("--sabdab2-canonical", type=Path, required=True)
    parser.add_argument("--sabdab2-canonical-sha256", required=True)
    parser.add_argument("--output-fasta", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    arguments = parser.parse_args()

    asep, asep_counts = load_asep(
        arguments.asep_root.resolve(),
        arguments.asep_manifest.resolve(),
        arguments.asep_manifest_sha256,
    )
    sabdab2, sabdab2_counts = load_sabdab2(
        arguments.sabdab2_split.resolve(),
        arguments.sabdab2_split_sha256,
        arguments.sabdab2_canonical.resolve(),
        arguments.sabdab2_canonical_sha256,
    )
    combined = dict(asep)
    for digest, record in sabdab2.items():
        previous = combined.get(digest)
        if previous is None or record[0] < previous[0]:
            combined[digest] = record
        elif previous[1] != record[1]:
            raise AssertionError("SHA-256 sequence collision across development datasets")
    fasta = "".join(
        f">{header}|sha256={digest}\n{sequence}\n"
        for digest, (header, sequence) in sorted(combined.items())
    ).encode("ascii")
    output = arguments.output_fasta.resolve()
    publish_create_only(output, fasta)
    receipt = {
        "authorization": {"training_decontamination": True, "test_access": False},
        "claim_boundary": "Exact validation sequences for training-corpus decontamination only. No task labels, predictions, metrics, or test sequences are included.",
        "counts": {
            "asep": asep_counts,
            "sabdab2": sabdab2_counts,
            "unique_exact_sequences": len(combined),
        },
        "format_version": "abpairfm_development_sequence_panel_v1",
        "output_fasta": {
            "bytes": output.stat().st_size,
            "sha256": sha256_file(output),
        },
        "source_sha256": {
            "asep_manifest": arguments.asep_manifest_sha256,
            "sabdab2_canonical": arguments.sabdab2_canonical_sha256,
            "sabdab2_split": arguments.sabdab2_split_sha256,
        },
        "status": "FROZEN_VALIDATION_SEQUENCE_PANEL",
        "test_access": False,
    }
    publish_create_only(
        arguments.receipt.resolve(),
        (json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n").encode(),
    )
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
