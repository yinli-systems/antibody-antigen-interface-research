#!/usr/bin/env python3
"""Normalize PPIRef50K v2 into full-sequence pairs plus interface-patch contacts."""

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
import re
import subprocess
import tempfile
import zipfile

import numpy as np
from scipy.spatial import cKDTree


AA3 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEC": "U", "PYL": "O", "ASX": "B", "GLX": "Z",
}
INTERFACE_ID = re.compile(r"^([0-9A-Za-z]{4})_([^_]+)_([^_]+)$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_id(value: str) -> tuple[str, str, str]:
    match = INTERFACE_ID.fullmatch(value)
    if match is None or match.group(2) == match.group(3):
        raise ValueError(f"Malformed PPIRef interface id: {value!r}")
    return match.group(1).lower(), match.group(2), match.group(3)


def read_split(path: Path) -> list[tuple[str, str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("source") != "data/ppiref/ppi_6A" or set(payload.get("folds", {})) != {"whole"}:
        raise ValueError("PPIRef50K split schema differs")
    values = [canonical_id(value) for value in payload["folds"]["whole"]]
    if len(values) != 51_755 or len(set(values)) != len(values):
        raise ValueError("PPIRef50K split count or uniqueness differs")
    return sorted(values)


def target_full_sequences(
    path: Path, targets: set[tuple[str, str]]
) -> tuple[dict[tuple[str, str], str], dict[str, int]]:
    candidates: dict[tuple[str, str], str] = {}
    source_records = 0
    current: tuple[str, str] | None = None
    current_is_protein = False
    chunks: list[str] = []

    def emit() -> None:
        if current is None or not current_is_protein:
            return
        folded = (current[0], current[1].casefold())
        if folded not in target_folded:
            return
        sequence = "".join(chunks).upper()
        if not sequence or not sequence.isalpha():
            raise ValueError(f"Malformed wwPDB SEQRES sequence: {current}")
        previous = candidates.setdefault(current, sequence)
        if previous != sequence:
            raise ValueError(f"Conflicting wwPDB SEQRES records: {current}")

    target_folded: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for target in sorted(targets):
        target_folded.setdefault((target[0], target[1].casefold()), []).append(target)
    with gzip.open(path, "rt", encoding="ascii") as handle:
        for raw_line in handle:
            if raw_line.startswith(">"):
                emit()
                source_records += 1
                header = raw_line[1:].strip()
                token = header.split(maxsplit=1)[0]
                if "_" not in token:
                    raise ValueError(f"Malformed wwPDB SEQRES header: {header!r}")
                pdb_id, chain = token.split("_", 1)
                current = (pdb_id.lower(), chain)
                current_is_protein = "mol:protein" in header.lower()
                chunks = []
            else:
                chunks.append(raw_line.strip())
        emit()
    selected: dict[tuple[str, str], str] = {}
    candidate_folded: dict[tuple[str, str], list[str]] = {}
    for (pdb_id, chain), sequence in candidates.items():
        candidate_folded.setdefault((pdb_id, chain.casefold()), []).append(sequence)
    aliases = 0
    ambiguous = 0
    for target in sorted(targets):
        exact = candidates.get(target)
        if exact is not None:
            selected[target] = exact
            continue
        matches = candidate_folded.get((target[0], target[1].casefold()), [])
        if len(matches) == 1:
            selected[target] = matches[0]
            aliases += 1
        elif len(matches) > 1:
            ambiguous += 1
    return selected, {
        "source_seqres_records": source_records,
        "target_seqres_ambiguous": ambiguous,
        "target_seqres_casefold_aliases": aliases,
        "target_seqres_records": len(selected),
        "target_seqres_missing": len(targets - set(selected)),
    }


def archive_inventory(archive: zipfile.ZipFile) -> dict[tuple[str, str, str], str]:
    selected: dict[tuple[str, str, str], str] = {}
    for info in archive.infolist():
        if info.is_dir() or not info.filename.endswith(".pdb"):
            continue
        stem = Path(info.filename).stem
        try:
            key = canonical_id(stem)
        except ValueError:
            continue
        previous = selected.setdefault(key, info.filename)
        if previous != info.filename:
            raise ValueError(f"Duplicate PPIRef archive member for {key}")
    return selected


def resolve_archive_members(
    archive: zipfile.ZipFile, targets: list[tuple[str, str, str]]
) -> tuple[dict[tuple[str, str, str], dict[str, object]], dict[str, int]]:
    inventory = archive_inventory(archive)
    folded: dict[tuple[str, str, str], list[tuple[tuple[str, str, str], str]]] = {}
    for key, member in inventory.items():
        folded.setdefault((key[0], key[1].casefold(), key[2].casefold()), []).append((key, member))
    resolved: dict[tuple[str, str, str], dict[str, object]] = {}
    counts = {
        "archive_casefold_aliases": 0,
        "archive_exact_members": 0,
        "archive_unresolved_members": 0,
    }
    for target in targets:
        if target in inventory:
            archive_key = target
            member = inventory[target]
            method = "exact"
            counts["archive_exact_members"] += 1
        else:
            matches = folded.get((target[0], target[1].casefold(), target[2].casefold()), [])
            if len(matches) != 1:
                counts["archive_unresolved_members"] += 1
                continue
            archive_key, member = matches[0]
            method = "unique_same_order_casefold"
            counts["archive_casefold_aliases"] += 1
        resolved[target] = {
            "archive_key": archive_key,
            "archive_member": member,
            "resolution": method,
        }
    return resolved, counts


def _metadata_value(line: str, label: str) -> str | None:
    marker = f"REMARK   3  {label}:"
    if line.startswith(marker):
        value = line[len(marker):].strip()
        return value.split(" A", 1)[0]
    return None


def parse_interface_pdb(
    payload: bytes, chain1: str, chain2: str, radius: float
) -> dict[str, object]:
    residues: dict[str, OrderedDict[tuple[int, str], dict[str, object]]] = {}
    metadata: dict[str, str | float | None] = {
        "bsa": None, "resolution": None, "structure_method": None,
    }
    for line in payload.decode("ascii").splitlines():
        value = _metadata_value(line, "BSA")
        if value is not None:
            metadata["bsa"] = float(value)
        value = _metadata_value(line, "RESOLUTION")
        if value is not None:
            metadata["resolution"] = None if value.lower() == "none" else float(value)
        value = _metadata_value(line, "STRUCTURE METHOD")
        if value is not None:
            metadata["structure_method"] = value
        if not (line.startswith("ATOM  ") or line.startswith("HETATM")):
            continue
        if len(line) < 60:
            continue
        chain = line[21].strip()
        if not chain:
            continue
        element = line[76:78].strip().upper() if len(line) >= 78 else ""
        atom_name = line[12:16].strip().upper()
        if element in {"H", "D"} or (not element and atom_name.startswith("H")):
            continue
        try:
            residue_number = int(line[22:26])
            coordinate = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            occupancy = float(line[54:60])
        except ValueError as error:
            raise ValueError("Malformed coordinate in PPIRef PDB") from error
        key = (residue_number, line[26].strip())
        amino_acid = AA3.get(line[17:20].strip().upper(), "X")
        altloc = line[16]
        rank = (occupancy, altloc == " ", altloc == "A", altloc)
        record = residues.setdefault(chain, OrderedDict()).setdefault(
            key,
            {"shared_aa": None, "shared_atoms": {}, "variants": {}},
        )
        if altloc == " ":
            if record["shared_aa"] not in (None, amino_acid):
                raise ValueError(f"Conflicting shared residue identity for {chain}:{key}")
            record["shared_aa"] = amino_acid
            atoms = record["shared_atoms"]
        else:
            variant = record["variants"].setdefault(  # type: ignore[union-attr]
                altloc, {"aa": amino_acid, "atoms": {}, "occupancy": occupancy}
            )
            if variant["aa"] != amino_acid:
                raise ValueError(f"Conflicting altloc residue identity for {chain}:{key}:{altloc}")
            variant["occupancy"] = max(float(variant["occupancy"]), occupancy)
            atoms = variant["atoms"]
        previous = atoms.get(atom_name)  # type: ignore[union-attr]
        if previous is None or rank > previous[0]:
            atoms[atom_name] = (rank, coordinate)  # type: ignore[index]

    def resolve_chain(requested: str) -> str:
        if requested in residues:
            return requested
        matches = [value for value in residues if value.casefold() == requested.casefold()]
        if len(matches) != 1:
            raise ValueError(
                f"PPIRef interface chain cannot be resolved: requested={requested!r} "
                f"casefold_matches={matches} available={sorted(residues)}"
            )
        return matches[0]

    source_chain1 = resolve_chain(chain1)
    source_chain2 = resolve_chain(chain2)
    if source_chain1 == source_chain2:
        raise ValueError("PPIRef interface partners resolve to the same atom chain")

    sequences: dict[str, str] = {}
    atom_coordinates: dict[str, np.ndarray] = {}
    atom_residue_indices: dict[str, np.ndarray] = {}
    for chain in (source_chain1, source_chain2):
        records = list(residues[chain].values())
        if not records:
            raise ValueError(f"PPIRef interface chain has no heavy atoms: {chain}")
        coordinates: list[tuple[float, float, float]] = []
        indices: list[int] = []
        sequence: list[str] = []
        for residue_index, record in enumerate(records):
            variants = record["variants"]
            shared_atoms = record["shared_atoms"]
            if variants:
                selected_altloc, selected_variant = max(
                    variants.items(),
                    key=lambda item: (
                        float(item[1]["occupancy"]),
                        item[0] == "A",
                        item[0],
                    ),
                )
                amino_acid = str(selected_variant["aa"])
                selected_atoms = dict(shared_atoms)
                selected_atoms.update(selected_variant["atoms"])
            else:
                selected_altloc = " "
                amino_acid = str(record["shared_aa"] or "X")
                selected_atoms = shared_atoms
            if not selected_atoms:
                raise ValueError(f"PPIRef residue has no selected atoms: {chain}:{residue_index}:{selected_altloc}")
            sequence.append(amino_acid)
            selected_coordinates = [value[1] for value in selected_atoms.values()]
            coordinates.extend(selected_coordinates)
            indices.extend([residue_index] * len(selected_coordinates))
        sequences[chain] = "".join(sequence)
        atom_coordinates[chain] = np.asarray(coordinates, dtype=np.float32)
        atom_residue_indices[chain] = np.asarray(indices, dtype=np.int32)

    neighbours = cKDTree(atom_coordinates[source_chain2]).query_ball_point(
        atom_coordinates[source_chain1], r=radius
    )
    contacts: set[tuple[int, int]] = set()
    for atom1, matches in enumerate(neighbours):
        residue1 = int(atom_residue_indices[source_chain1][atom1])
        for atom2 in matches:
            contacts.add((residue1, int(atom_residue_indices[source_chain2][atom2])))
    return {
        "chain1_patch_sequence": sequences[source_chain1],
        "chain2_patch_sequence": sequences[source_chain2],
        "contact_pairs": sorted(contacts),
        "metadata": {
            **metadata,
            "source_atom_chain1": source_chain1,
            "source_atom_chain2": source_chain2,
        },
    }


def write_outputs(
    *,
    archive_path: Path,
    split: list[tuple[str, str, str]],
    resolved_members: dict[tuple[str, str, str], dict[str, object]],
    full_sequences: dict[tuple[str, str], str],
    manifest_output: Path,
    fasta_output: Path,
    minimum_contact_pairs: int,
    radius: float,
    threads: int,
) -> dict[str, object]:
    if threads < 1:
        raise ValueError("threads must be positive")
    for path in (manifest_output, fasta_output):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite normalized PPIRef output: {path}")
    manifest_descriptor, manifest_temporary = tempfile.mkstemp(
        prefix=f".{manifest_output.name}.", dir=manifest_output.parent
    )
    os.close(manifest_descriptor)
    manifest_temporary_path = Path(manifest_temporary)
    fasta_descriptor, fasta_temporary = tempfile.mkstemp(
        prefix=f".{fasta_output.name}.", dir=fasta_output.parent
    )
    fasta_temporary_path = Path(fasta_temporary)
    counts = {
        "admitted_interfaces": 0,
        "contact_pairs": 0,
        "dropped_below_minimum_contact_pairs": 0,
        "dropped_missing_archive_member": 0,
        "dropped_missing_full_sequence": 0,
        "full_sequence_residues": 0,
        "interface_patch_residues": 0,
        "source_split_interfaces": len(split),
    }
    endpoint_sequences: set[str] = set()
    exact_pairs: set[tuple[str, str]] = set()
    pdb_ids: set[str] = set()
    try:
        with zipfile.ZipFile(archive_path) as archive:
            with os.fdopen(fasta_descriptor, "wt", encoding="ascii", newline="") as fasta_handle:
                with manifest_temporary_path.open("wb") as compressed_handle:
                    compressor = subprocess.Popen(
                        ["zstd", "-q", f"-T{threads}", "-19", "-c"],
                        stdin=subprocess.PIPE,
                        stdout=compressed_handle,
                        stderr=subprocess.PIPE,
                        text=True,
                        encoding="ascii",
                    )
                    assert compressor.stdin is not None
                    for pdb_id, chain1, chain2 in split:
                        key = (pdb_id, chain1, chain2)
                        resolution = resolved_members.get(key)
                        if resolution is None:
                            counts["dropped_missing_archive_member"] += 1
                            continue
                        archive_key = resolution["archive_key"]
                        assert isinstance(archive_key, tuple)
                        _, archive_chain1, archive_chain2 = archive_key
                        member = resolution["archive_member"]
                        assert isinstance(member, str)
                        sequence1 = full_sequences.get((pdb_id, archive_chain1))
                        sequence2 = full_sequences.get((pdb_id, archive_chain2))
                        if sequence1 is None or sequence2 is None:
                            counts["dropped_missing_full_sequence"] += 1
                            continue
                        try:
                            interface = parse_interface_pdb(
                                archive.read(member), archive_chain1, archive_chain2, radius
                            )
                        except Exception as error:
                            raise ValueError(
                                f"Failed to normalize PPIRef interface {pdb_id}_{chain1}_{chain2} "
                                f"from {member}: {error}"
                            ) from error
                        contacts = interface["contact_pairs"]
                        if len(contacts) < minimum_contact_pairs:  # type: ignore[arg-type]
                            counts["dropped_below_minimum_contact_pairs"] += 1
                            continue
                        example_id = f"{pdb_id}_{chain1}_{chain2}"
                        record = {
                            "archive_member": member,
                            "archive_member_resolution": resolution["resolution"],
                            "chain1": archive_chain1,
                            "chain1_full_sequence": sequence1,
                            "chain1_patch_sequence": interface["chain1_patch_sequence"],
                            "chain2": archive_chain2,
                            "chain2_full_sequence": sequence2,
                            "chain2_patch_sequence": interface["chain2_patch_sequence"],
                            "contact_pairs": contacts,
                            "contact_radius_angstrom": radius,
                            "example_id": example_id,
                            "metadata": interface["metadata"],
                            "pdb_id": pdb_id,
                            "source_split_chain1": chain1,
                            "source_split_chain2": chain2,
                        }
                        compressor.stdin.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
                        fasta_handle.write(f">{example_id}|1\n{sequence1}\n>{example_id}|2\n{sequence2}\n")
                        counts["admitted_interfaces"] += 1
                        counts["contact_pairs"] += len(contacts)  # type: ignore[arg-type]
                        counts["full_sequence_residues"] += len(sequence1) + len(sequence2)
                        counts["interface_patch_residues"] += len(interface["chain1_patch_sequence"]) + len(interface["chain2_patch_sequence"])  # type: ignore[arg-type]
                        endpoint_sequences.update((sequence1, sequence2))
                        exact_pairs.add(tuple(sorted((sequence1, sequence2))))
                        pdb_ids.add(pdb_id)
                    compressor.stdin.close()
                    stderr = compressor.stderr.read() if compressor.stderr is not None else ""
                    return_code = compressor.wait()
                    if return_code:
                        raise RuntimeError(f"zstd failed with {return_code}: {stderr[-1000:]}")
                    compressed_handle.flush()
                    os.fsync(compressed_handle.fileno())
                fasta_handle.flush()
                os.fsync(fasta_handle.fileno())
        os.link(manifest_temporary_path, manifest_output)
        os.link(fasta_temporary_path, fasta_output)
        manifest_output.chmod(0o444)
        fasta_output.chmod(0o444)
    except FileExistsError as error:
        raise FileExistsError("Refusing to overwrite normalized PPIRef output") from error
    finally:
        manifest_temporary_path.unlink(missing_ok=True)
        fasta_temporary_path.unlink(missing_ok=True)
    counts["unique_exact_full_sequence_pairs"] = len(exact_pairs)
    counts["unique_full_endpoint_sequences"] = len(endpoint_sequences)
    counts["unique_pdb_ids"] = len(pdb_ids)
    classified = (
        counts["admitted_interfaces"]
        + counts["dropped_below_minimum_contact_pairs"]
        + counts["dropped_missing_archive_member"]
        + counts["dropped_missing_full_sequence"]
    )
    if counts["admitted_interfaces"] <= 0 or classified != counts["source_split_interfaces"]:
        raise ValueError("PPIRef normalization did not classify every source interface")
    return counts


def publish_receipt(path: Path, value: object) -> None:
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
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
        raise FileExistsError(f"Refusing to overwrite PPIRef normalization receipt: {path}") from error
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--aux-receipt", type=Path, required=True)
    parser.add_argument("--coverage-receipt", type=Path, required=True)
    parser.add_argument("--seqres", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--fasta-output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--minimum-contact-pairs", type=int, default=8)
    parser.add_argument("--contact-radius", type=float, default=6.0)
    parser.add_argument("--threads", type=int, default=6)
    arguments = parser.parse_args()
    if arguments.minimum_contact_pairs < 1 or not math.isclose(arguments.contact_radius, 6.0):
        raise ValueError("PPIRef contact gate differs from the frozen protocol")
    archive = arguments.archive.resolve()
    if sha256_file(archive) != arguments.archive_sha256:
        raise ValueError("PPIRef archive SHA-256 differs")
    aux_receipt_path = arguments.aux_receipt.resolve()
    aux_receipt = json.loads(aux_receipt_path.read_text(encoding="utf-8"))
    if aux_receipt.get("status") != "MATERIALIZED_NOT_NORMALIZED" or aux_receipt.get("test_access") is not False:
        raise ValueError("PPIRef auxiliary receipt is not eligible")
    split_path = arguments.split.resolve()
    seqres_path = arguments.seqres.resolve()
    for path in (split_path, seqres_path):
        expected = aux_receipt["files"][path.name]
        if path.stat().st_size != expected["bytes"] or sha256_file(path) != expected["sha256"]:
            raise ValueError(f"PPIRef auxiliary input differs: {path.name}")
    coverage_receipt_path = arguments.coverage_receipt.resolve()
    coverage_receipt = json.loads(coverage_receipt_path.read_text(encoding="utf-8"))
    if (
        coverage_receipt.get("format_version") != "abpairfm_ppiref50k_archive_coverage_v2"
        or coverage_receipt.get("authorization", {}).get("unique_same_order_casefold_recovery") is not True
        or coverage_receipt.get("counts", {}).get("absent_interfaces") != 0
        or coverage_receipt.get("counts", {}).get("reverse_order_only_interfaces") != 0
        or coverage_receipt.get("test_access") is not False
        or coverage_receipt.get("inputs", {}).get("archive_sha256") != arguments.archive_sha256
        or coverage_receipt.get("inputs", {}).get("split_sha256") != sha256_file(split_path)
    ):
        raise ValueError("PPIRef archive coverage receipt is not eligible")
    split = read_split(split_path)
    with zipfile.ZipFile(archive) as archive_handle:
        resolved_members, archive_counts = resolve_archive_members(archive_handle, split)
    targets: set[tuple[str, str]] = set()
    for value in resolved_members.values():
        archive_key = value["archive_key"]
        if not isinstance(archive_key, tuple) or len(archive_key) != 3:
            raise TypeError("Malformed resolved PPIRef archive key")
        targets.add((archive_key[0], archive_key[1]))
        targets.add((archive_key[0], archive_key[2]))
    full_sequences, seqres_counts = target_full_sequences(seqres_path, targets)
    manifest_output = arguments.manifest_output.resolve()
    fasta_output = arguments.fasta_output.resolve()
    counts = write_outputs(
        archive_path=archive,
        split=split,
        resolved_members=resolved_members,
        full_sequences=full_sequences,
        manifest_output=manifest_output,
        fasta_output=fasta_output,
        minimum_contact_pairs=arguments.minimum_contact_pairs,
        radius=arguments.contact_radius,
        threads=arguments.threads,
    )
    receipt = {
        "authorization": {"development_decontamination": True, "stage_a_training": False},
        "claim_boundary": "PPIRef50K full-sequence pair provenance plus 6-Angstrom interface-patch contact labels; development decontamination and cross-source deduplication remain pending.",
        "counts": {**archive_counts, **seqres_counts, **counts},
        "filters": {
            "minimum_contact_residue_pairs": arguments.minimum_contact_pairs,
            "ppiref_idist_threshold": 0.04,
            "source_interface_radius_angstrom": arguments.contact_radius,
        },
        "format_version": "abpairfm_ppiref50k_interfaces_v2",
        "inputs": {
            "archive_sha256": arguments.archive_sha256,
            "aux_materialization_receipt_sha256": sha256_file(aux_receipt_path),
            "coverage_receipt_sha256": sha256_file(coverage_receipt_path),
            "seqres_sha256": sha256_file(seqres_path),
            "split_sha256": sha256_file(split_path),
        },
        "outputs": {
            "full_sequence_fasta": {"bytes": fasta_output.stat().st_size, "sha256": sha256_file(fasta_output)},
            "interface_manifest": {"bytes": manifest_output.stat().st_size, "sha256": sha256_file(manifest_output)},
        },
        "status": "NORMALIZED_NOT_DECONTAMINATED_NOT_AUTHORIZED",
        "test_access": False,
    }
    publish_receipt(arguments.receipt.resolve(), receipt)
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
