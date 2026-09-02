#!/usr/bin/env python3
"""Extract and audit only allowlisted PINDER predicted monomer structures."""

from __future__ import annotations

import argparse
from collections import OrderedDict
import csv
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
from typing import Mapping
import zipfile


AMINO_ACIDS = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "SEC": "U",
    "PYL": "O",
    "UNK": "X",
}
INVENTORY_COLUMNS = {
    "predicted_R_pdb",
    "predicted_L_pdb",
    "sequence_R",
    "sequence_L",
}
MANIFEST_COLUMNS = (
    "member",
    "pdb_sha256",
    "size_bytes",
    "sequence",
    "sequence_sha256",
    "residue_count",
    "physical_chain_id",
    "plddt_min",
    "plddt_max",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def create_only(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def inventory_members(path: Path) -> tuple[set[str], int]:
    members: set[str] = set()
    member_sequence_associations: set[tuple[str, str]] = set()
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or not INVENTORY_COLUMNS.issubset(reader.fieldnames):
            raise ValueError("Multimodal inventory schema differs")
        for row in reader:
            for side in ("R", "L"):
                member = row[f"predicted_{side}_pdb"]
                sequence = row[f"sequence_{side}"]
                if not member or not sequence:
                    raise ValueError("Both-predicted inventory contains an empty endpoint")
                if Path(member).name != member or not member.startswith("af__"):
                    raise ValueError(f"Unsafe predicted structure member: {member!r}")
                members.add(member)
                member_sequence_associations.add((member, sequence))
    if not members:
        raise ValueError("Predicted-structure allowlist is empty")
    return members, len(member_sequence_associations)


def parse_predicted_pdb(payload: bytes) -> dict[str, object]:
    try:
        lines = payload.decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError("Predicted PDB is not ASCII") from error
    residues: OrderedDict[tuple[str, str, str], dict[str, object]] = OrderedDict()
    for line in lines:
        if not line.startswith("ATOM  "):
            continue
        if len(line) < 66:
            raise ValueError("Truncated ATOM record")
        alternate = line[16]
        if alternate not in {" ", "A"}:
            continue
        atom = line[12:16].strip()
        if atom not in {"N", "CA", "C"}:
            continue
        residue_name = line[17:20].strip()
        chain_id = line[21]
        residue_id = line[22:26].strip()
        insertion = line[26]
        if residue_name not in AMINO_ACIDS or not residue_id:
            raise ValueError("Unsupported predicted-structure residue")
        key = (chain_id, residue_id, insertion)
        record = residues.setdefault(
            key,
            {"residue": AMINO_ACIDS[residue_name], "atoms": {}, "plddt": None},
        )
        if record["residue"] != AMINO_ACIDS[residue_name]:
            raise ValueError("Conflicting residue identity in predicted PDB")
        coordinates = tuple(float(line[start : start + 8]) for start in (30, 38, 46))
        if not all(math.isfinite(value) for value in coordinates):
            raise ValueError("Non-finite predicted coordinate")
        atoms = record["atoms"]
        assert isinstance(atoms, dict)
        if atom in atoms and atoms[atom] != coordinates:
            raise ValueError("Conflicting backbone atom alternate locations")
        atoms[atom] = coordinates
        if atom == "CA":
            confidence = float(line[60:66])
            if not math.isfinite(confidence) or not 0.0 <= confidence <= 100.0:
                raise ValueError("Predicted confidence is outside [0,100]")
            record["plddt"] = confidence

    if not residues:
        raise ValueError("Predicted PDB contains no backbone residues")
    chains = {key[0] for key in residues}
    if chains != {"A"}:
        raise ValueError("Predicted monomer structure must contain only physical chain A")
    residue_numbers: list[int] = []
    for _, residue_id, insertion in residues:
        if insertion != " ":
            raise ValueError("Predicted monomer structure contains an insertion code")
        try:
            residue_numbers.append(int(residue_id))
        except ValueError as error:
            raise ValueError("Predicted residue identifier is not an integer") from error
    if residue_numbers != list(range(1, len(residue_numbers) + 1)):
        raise ValueError("Predicted monomer residue identifiers are not consecutive")
    sequence_parts: list[str] = []
    confidences: list[float] = []
    for record in residues.values():
        atoms = record["atoms"]
        if not isinstance(atoms, dict) or set(atoms) != {"N", "CA", "C"}:
            raise ValueError("Predicted residue is missing N/CA/C backbone atoms")
        confidence = record["plddt"]
        if not isinstance(confidence, float):
            raise ValueError("Predicted residue is missing CA confidence")
        sequence_parts.append(str(record["residue"]))
        confidences.append(confidence)
    sequence = "".join(sequence_parts)
    return {
        "sequence": sequence,
        "residue_count": len(sequence),
        "physical_chain_id": next(iter(chains)),
        "plddt_min": min(confidences),
        "plddt_max": max(confidences),
    }


def manifest_bytes(rows: list[dict[str, object]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=MANIFEST_COLUMNS,
        delimiter="\t",
        lineterminator="\n",
        extrasaction="raise",
    )
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def materialize(arguments: argparse.Namespace) -> dict[str, object]:
    if sha256_file(arguments.inventory) != arguments.inventory_sha256:
        raise ValueError("Multimodal inventory SHA-256 differs")
    archive_size = arguments.structure_archive.stat().st_size
    if archive_size != arguments.structure_archive_size:
        raise ValueError("PINDER structure archive size differs")
    if sha256_file(arguments.structure_archive) != arguments.structure_archive_sha256:
        raise ValueError("PINDER structure archive SHA-256 differs")
    if arguments.output_root.exists() or arguments.output_result.exists():
        raise FileExistsError("Predicted structure cache output already exists")

    members, association_count = inventory_members(arguments.inventory)
    if len(members) != arguments.expected_structure_members:
        raise ValueError("Predicted structure allowlist count differs")
    staging = arguments.output_root.with_name(
        f".{arguments.output_root.name}.{os.getpid()}.staging"
    )
    if staging.exists():
        raise FileExistsError(staging)
    assets = staging / "pdb"
    assets.mkdir(parents=True)
    rows: list[dict[str, object]] = []
    total_bytes = 0
    try:
        with zipfile.ZipFile(arguments.structure_archive) as archive:
            for member in sorted(members):
                archive_name = f"pdbs/{member}"
                try:
                    info = archive.getinfo(archive_name)
                except KeyError as error:
                    raise ValueError(f"Allowlisted structure is absent: {member}") from error
                if info.is_dir() or Path(info.filename).name != member:
                    raise ValueError("Unsafe structure archive member")
                payload = archive.read(info)
                audit = parse_predicted_pdb(payload)
                output = assets / member
                create_only(output, payload)
                digest = hashlib.sha256(payload).hexdigest()
                total_bytes += len(payload)
                rows.append(
                    {
                        "member": member,
                        "pdb_sha256": digest,
                        "size_bytes": len(payload),
                        "sequence": audit["sequence"],
                        "sequence_sha256": hashlib.sha256(
                            str(audit["sequence"]).encode("ascii")
                        ).hexdigest(),
                        "residue_count": audit["residue_count"],
                        "physical_chain_id": audit["physical_chain_id"],
                        "plddt_min": f"{audit['plddt_min']:.6f}",
                        "plddt_max": f"{audit['plddt_max']:.6f}",
                    }
                )
        manifest = staging / "PREDICTED_STRUCTURE_CACHE_MANIFEST_V1.tsv"
        create_only(manifest, manifest_bytes(rows))
        manifest_sha = sha256_file(manifest)
        staging.rename(arguments.output_root)
    except BaseException:
        # Keep the staging directory as fail-closed evidence for manual inspection.
        raise

    payload = {
        "authorization": {
            "model_training": False,
            "structure_tokenization": True,
        },
        "claim_boundary": "Authenticated PINDER-Train AlphaFold monomer cache on its full UniProt residue axis only; mapping each holo construct to this axis remains separately gated, bound structures remain labels, and no validation/test structures were read.",
        "counts": {
            "allowlisted_structure_members": len(rows),
            "cache_bytes": total_bytes,
            "unique_member_holo_sequence_associations": association_count,
        },
        "format_version": "pinder_predicted_structure_cache_result_v1",
        "inputs": {
            "inventory_sha256": arguments.inventory_sha256,
            "structure_archive_sha256": arguments.structure_archive_sha256,
            "structure_archive_size": archive_size,
        },
        "outputs": {
            "manifest_sha256": manifest_sha,
            "holo_axis_alignment_status": "PENDING_SEPARATE_GATE",
        },
        "status": "PASS_AUTHENTICATED_PREDICTED_STRUCTURE_CACHE",
        "test_access": {
            "bound_coordinates_used_as_model_input": False,
            "pinder_test_structures_read": False,
            "pinder_validation_structures_read": False,
        },
    }
    create_only(arguments.output_result, canonical_json_bytes(payload))
    return payload


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--inventory-sha256", required=True)
    parser.add_argument("--structure-archive", type=Path, required=True)
    parser.add_argument("--structure-archive-sha256", required=True)
    parser.add_argument("--structure-archive-size", type=int, required=True)
    parser.add_argument("--expected-structure-members", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-result", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(materialize(parse_arguments()), sort_keys=True))
