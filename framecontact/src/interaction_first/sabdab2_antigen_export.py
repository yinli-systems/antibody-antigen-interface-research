"""Authenticated antigen-only backbone export for Foldseek."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tarfile
import tempfile
from typing import Any

import numpy as np

from .sabdab2_structure import (
    StructureCacheError,
    _bio_parser,
    _chain_backbone,
    _chain_lookup,
    _chain_names,
    _parse_numbering_list,
    _split_antigen_specs,
    authenticate_file,
    canonical_json_bytes,
    sha256_file,
)


ANTIGEN_EXPORT_FORMAT = "sabdab2_antigen_backbone_export_v1"
_THREE_LETTER = {
    one: three
    for one, three in zip(
        "ACDEFGHIKLMNPQRSTVWYX",
        (
            "ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE", "LYS",
            "LEU", "MET", "ASN", "PRO", "GLN", "ARG", "SER", "THR", "VAL",
            "TRP", "TYR", "UNK",
        ),
    )
}
_CHAIN_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"


def _metadata_index(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "INSTANCE" not in reader.fieldnames:
            raise StructureCacheError("Metadata lacks INSTANCE")
        rows = {}
        for row in reader:
            identifier = str(row.get("INSTANCE", "")).strip()
            if not identifier or identifier in rows:
                raise StructureCacheError("Metadata identifiers must be unique")
            rows[identifier] = {
                str(key): "" if value is None else str(value).strip()
                for key, value in row.items()
            }
        return rows


def _prepared_index(path: Path) -> dict[str, dict[str, Any]]:
    try:
        import pandas as pd
    except ModuleNotFoundError as error:  # pragma: no cover
        raise RuntimeError("pandas with a Parquet engine is required") from error
    frame = pd.read_parquet(path)
    required = {"instance_id", "ag_sequence_sha256", "ag_length"}
    if not required.issubset(frame.columns):
        raise StructureCacheError(
            f"Prepared authority lacks columns: {sorted(required - set(frame.columns))}"
        )
    if frame["instance_id"].isna().any() or frame["instance_id"].duplicated().any():
        raise StructureCacheError("Prepared identifiers must be unique")
    return {str(row["instance_id"]): row for row in frame.to_dict(orient="records")}


def _pdb_text(sequence: str, backbone: np.ndarray, chain_ids: np.ndarray) -> tuple[str, int]:
    if backbone.shape != (len(sequence), 3, 3) or chain_ids.shape != (len(sequence),):
        raise StructureCacheError("Antigen backbone axes do not agree")
    unique_chains = sorted(set(int(value) for value in chain_ids.tolist()))
    if len(unique_chains) > len(_CHAIN_ALPHABET):
        raise StructureCacheError("Too many antigen chains for PDB export")
    chain_names = {value: _CHAIN_ALPHABET[index] for index, value in enumerate(unique_chains)}
    residue_numbers = {value: 0 for value in unique_chains}
    atom_serial = 0
    valid_residues = 0
    lines: list[str] = []
    for index, amino_acid in enumerate(sequence):
        coordinates = backbone[index]
        chain_id = int(chain_ids[index])
        residue_numbers[chain_id] += 1
        residue_number = residue_numbers[chain_id]
        if residue_number > 9999:
            raise StructureCacheError("Antigen chain exceeds PDB residue-number capacity")
        if not np.isfinite(coordinates).all():
            continue
        valid_residues += 1
        residue_name = _THREE_LETTER.get(amino_acid, "UNK")
        for atom_name, element, xyz in zip(("N", "CA", "C"), ("N", "C", "C"), coordinates):
            atom_serial += 1
            if atom_serial > 99999:
                raise StructureCacheError("Antigen exceeds PDB atom-serial capacity")
            lines.append(
                f"ATOM  {atom_serial:5d} {atom_name:^4s} {residue_name:>3s} "
                f"{chain_names[chain_id]}{residue_number:4d}    "
                f"{float(xyz[0]):8.3f}{float(xyz[1]):8.3f}{float(xyz[2]):8.3f}"
                f"{1.00:6.2f}{0.00:6.2f}          {element:>2s}\n"
            )
    lines.append("END\n")
    return "".join(lines), valid_residues


def _antigen_backbone(
    identifier: str, metadata: dict[str, str], payload: bytes
) -> tuple[str, np.ndarray, np.ndarray]:
    structure = _bio_parser().get_structure(
        identifier, io.StringIO(payload.decode("utf-8", errors="strict"))
    )
    model = next(structure.get_models())
    chains = _chain_names(metadata.get("agchains"))
    specs = _split_antigen_specs(metadata.get("ag_index_list", ""), len(chains))
    if not chains:
        raise StructureCacheError("No antigen chains are declared")
    sequences, backbones, chain_ids = [], [], []
    for chain_index, (chain_name, raw_spec) in enumerate(zip(chains, specs)):
        sequence, backbone = _chain_backbone(
            _chain_lookup(model, chain_name), _parse_numbering_list(raw_spec)
        )
        sequences.append(sequence)
        backbones.append(backbone)
        chain_ids.append(np.full(len(sequence), chain_index, dtype=np.int64))
    return (
        "".join(sequences),
        np.concatenate(backbones, axis=0),
        np.concatenate(chain_ids, axis=0),
    )


def export_antigen_structures(
    *,
    archive_path: Path,
    archive_sha256: str,
    metadata_path: Path,
    metadata_sha256: str,
    prepared_path: Path,
    prepared_sha256: str,
    output_root: Path,
) -> dict[str, Any]:
    """Export one compact antigen-only PDB per prepared instance atomically."""
    authenticated = {
        "archive_sha256": authenticate_file(archive_path, archive_sha256, "structure archive"),
        "metadata_sha256": authenticate_file(metadata_path, metadata_sha256, "metadata"),
        "prepared_sha256": authenticate_file(prepared_path, prepared_sha256, "prepared cohort"),
    }
    metadata, prepared = _metadata_index(metadata_path), _prepared_index(prepared_path)
    missing_metadata = sorted(set(prepared) - set(metadata))
    if missing_metadata:
        raise StructureCacheError(
            f"Metadata lacks {len(missing_metadata)} prepared instances"
        )
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"Refusing to overwrite antigen export: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent))
    structures = temporary / "structures"
    structures.mkdir()
    records, exclusions, encountered = [], [], set()
    try:
        with tarfile.open(archive_path, "r|gz") as archive:
            for member in archive:
                if not member.isfile() or not member.name.endswith(".cif"):
                    continue
                identifier = Path(member.name).stem
                if identifier not in prepared:
                    continue
                encountered.add(identifier)
                binary = archive.extractfile(member)
                if binary is None:
                    exclusions.append({"instance_id": identifier, "reason": "unreadable_cif"})
                    continue
                try:
                    sequence, backbone, chain_ids = _antigen_backbone(
                        identifier, metadata[identifier], binary.read()
                    )
                    expected = prepared[identifier]
                    observed_sha = hashlib.sha256(sequence.encode("ascii")).hexdigest()
                    if observed_sha != str(expected["ag_sequence_sha256"]):
                        raise StructureCacheError("Antigen sequence digest differs")
                    if len(sequence) != int(expected["ag_length"]):
                        raise StructureCacheError("Antigen length differs")
                    text, valid_count = _pdb_text(sequence, backbone, chain_ids)
                    if valid_count < 3:
                        raise StructureCacheError("Fewer than three complete backbone residues")
                    foldseek_id = hashlib.sha256(identifier.encode("utf-8")).hexdigest()
                    path = structures / f"{foldseek_id}.pdb"
                    path.write_text(text, encoding="ascii", newline="")
                    records.append(
                        {
                            "antigen_structure_id": identifier,
                            "foldseek_structure_id": foldseek_id,
                            "relative_path": f"structures/{foldseek_id}.pdb",
                            "sha256": sha256_file(path),
                            "antigen_sequence_sha256": observed_sha,
                            "residue_count": len(sequence),
                            "valid_backbone_residue_count": valid_count,
                            "chain_count": len(set(chain_ids.tolist())),
                        }
                    )
                except Exception as error:
                    exclusions.append({"instance_id": identifier, "reason": str(error)})
        missing_cif = sorted(set(prepared) - encountered)
        exclusions.extend(
            {"instance_id": identifier, "reason": "missing_cif_member"}
            for identifier in missing_cif
        )
        records.sort(key=lambda row: row["antigen_structure_id"])
        exclusions.sort(key=lambda row: (row["instance_id"], row["reason"]))
        if exclusions or len(records) != len(prepared):
            preview = exclusions[:5]
            raise StructureCacheError(
                f"Antigen export excluded {len(exclusions)} instances: {preview}"
            )
        relative_paths = [row["relative_path"] for row in records]
        if len(set(relative_paths)) != len(relative_paths) or any(
            not (temporary / value).is_file() for value in relative_paths
        ):
            raise StructureCacheError("Antigen export paths are not unique regular files")
        manifest_path = temporary / "ANTIGEN_STRUCTURE_EXPORT_MANIFEST.csv"
        with manifest_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        receipt = {
            "format_version": ANTIGEN_EXPORT_FORMAT,
            "authority": {**authenticated, "task_labels_read": False},
            "structure_count": len(records),
            "structure_ids_sha256": hashlib.sha256(
                "".join(f"{row['antigen_structure_id']}\n" for row in records).encode()
            ).hexdigest(),
            "manifest_sha256": sha256_file(manifest_path),
            "atom_selection": ["N", "CA", "C"],
            "claim_boundary": "Antigen structure export only; no split or performance claim.",
        }
        (temporary / "ANTIGEN_STRUCTURE_EXPORT_RECEIPT.json").write_bytes(
            canonical_json_bytes(receipt)
        )
        os.replace(temporary, output_root)
        return {**receipt, "output_root": str(output_root)}
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


__all__ = ["ANTIGEN_EXPORT_FORMAT", "export_antigen_structures"]
