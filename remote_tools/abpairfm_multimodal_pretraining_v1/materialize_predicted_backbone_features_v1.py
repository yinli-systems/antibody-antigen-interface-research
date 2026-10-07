#!/usr/bin/env python3
"""Materialize flat 17D local-backbone features from predicted monomers."""

from __future__ import annotations

import argparse
from collections import OrderedDict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Mapping

import numpy as np


AMINO_ACIDS = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "SEC": "U", "PYL": "O", "UNK": "X",
}
FEATURE_DIMENSION = 17
BOND_NORMALIZATION = {
    "n_ca": (1.458, 0.020),
    "ca_c": (1.525, 0.020),
    "peptide_c_n": (1.329, 0.020),
    "adjacent_ca": (3.800, 0.100),
}
MANIFEST_COLUMNS = {
    "member", "pdb_sha256", "sequence", "sequence_sha256", "residue_count"
}
INDEX_COLUMNS = (
    "member", "sequence_sha256", "residue_offset", "residue_count", "feature_dimension"
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


def parse_backbone(path: Path) -> tuple[str, np.ndarray, np.ndarray]:
    residues: OrderedDict[tuple[str, int, str], dict[str, object]] = OrderedDict()
    with path.open("r", encoding="ascii") as handle:
        for line in handle:
            if not line.startswith("ATOM  "):
                continue
            if len(line) < 66:
                raise ValueError("Truncated ATOM record")
            if line[16] not in {" ", "A"}:
                continue
            atom = line[12:16].strip()
            if atom not in {"N", "CA", "C"}:
                continue
            residue_name = line[17:20].strip()
            if residue_name not in AMINO_ACIDS:
                raise ValueError("Unsupported predicted residue")
            try:
                residue_number = int(line[22:26])
            except ValueError as error:
                raise ValueError("Predicted residue number is not an integer") from error
            key = (line[21], residue_number, line[26])
            record = residues.setdefault(
                key, {"residue": AMINO_ACIDS[residue_name], "atoms": {}, "plddt": None}
            )
            if record["residue"] != AMINO_ACIDS[residue_name]:
                raise ValueError("Predicted residue identity differs")
            coordinates = np.asarray(
                [float(line[start : start + 8]) for start in (30, 38, 46)],
                dtype=np.float64,
            )
            if not np.isfinite(coordinates).all():
                raise ValueError("Predicted coordinate is non-finite")
            atoms = record["atoms"]
            assert isinstance(atoms, dict)
            atoms[atom] = coordinates
            if atom == "CA":
                confidence = float(line[60:66])
                if not math.isfinite(confidence) or not 0.0 <= confidence <= 100.0:
                    raise ValueError("Predicted confidence is outside [0,100]")
                record["plddt"] = confidence
    if not residues:
        raise ValueError("Predicted PDB contains no residues")
    keys = list(residues)
    if [key[0] for key in keys] != ["A"] * len(keys):
        raise ValueError("Predicted PDB must contain only chain A")
    if [key[1] for key in keys] != list(range(1, len(keys) + 1)):
        raise ValueError("Predicted residue axis is not consecutive")
    if any(key[2] != " " for key in keys):
        raise ValueError("Predicted residue axis contains an insertion code")
    sequence: list[str] = []
    coordinates: list[np.ndarray] = []
    confidences: list[float] = []
    for record in residues.values():
        atoms = record["atoms"]
        if not isinstance(atoms, dict) or set(atoms) != {"N", "CA", "C"}:
            raise ValueError("Predicted residue lacks N/CA/C")
        confidence = record["plddt"]
        if not isinstance(confidence, float):
            raise ValueError("Predicted residue lacks pLDDT")
        sequence.append(str(record["residue"]))
        coordinates.append(np.stack([atoms["N"], atoms["CA"], atoms["C"]]))
        confidences.append(confidence)
    return (
        "".join(sequence),
        np.stack(coordinates).astype(np.float64, copy=False),
        np.asarray(confidences, dtype=np.float64),
    )


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm < 1e-8:
        raise ValueError("Degenerate predicted backbone vector")
    return vector / norm


def _dihedral(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> float:
    b0 = -(b - a)
    b1 = _unit(c - b)
    b2 = d - c
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    if np.linalg.norm(v) < 1e-8 or np.linalg.norm(w) < 1e-8:
        return 0.0
    return float(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w)))


def _frame(n: np.ndarray, ca: np.ndarray, c: np.ndarray) -> np.ndarray:
    x = _unit(c - ca)
    n_direction = n - ca
    y = _unit(n_direction - np.dot(n_direction, x) * x)
    z = _unit(np.cross(x, y))
    return np.stack([x, y, z], axis=1)


def _normalized_distance(a: np.ndarray, b: np.ndarray, key: str) -> float:
    mean, scale = BOND_NORMALIZATION[key]
    return (float(np.linalg.norm(a - b)) - mean) / scale


def backbone_features(coordinates: np.ndarray, plddt: np.ndarray) -> np.ndarray:
    if coordinates.ndim != 3 or coordinates.shape[1:] != (3, 3):
        raise ValueError("Backbone coordinate tensor shape differs")
    if plddt.shape != (coordinates.shape[0],):
        raise ValueError("pLDDT axis differs")
    count = coordinates.shape[0]
    features = np.zeros((count, FEATURE_DIMENSION), dtype=np.float64)
    frames = [_frame(*coordinates[index]) for index in range(count)]
    for index in range(count):
        n, ca, c = coordinates[index]
        angles: list[float | None] = [None, None, None]
        if index > 0:
            angles[0] = _dihedral(coordinates[index - 1, 2], n, ca, c)
        if index + 1 < count:
            next_n, next_ca, _ = coordinates[index + 1]
            angles[1] = _dihedral(n, ca, c, next_n)
            angles[2] = _dihedral(ca, c, next_n, next_ca)
        for angle_index, angle in enumerate(angles):
            if angle is not None:
                features[index, 2 * angle_index] = math.sin(angle)
                features[index, 2 * angle_index + 1] = math.cos(angle)
        features[index, 6] = _normalized_distance(n, ca, "n_ca")
        features[index, 7] = _normalized_distance(ca, c, "ca_c")
        if index + 1 < count:
            next_n, next_ca, _ = coordinates[index + 1]
            features[index, 8] = _normalized_distance(c, next_n, "peptide_c_n")
            features[index, 9] = _normalized_distance(ca, next_ca, "adjacent_ca")
            relative = frames[index].T @ frames[index + 1]
            features[index, 10:16] = relative[:, :2].reshape(-1)
        features[index, 16] = plddt[index] / 100.0
    if not np.isfinite(features).all():
        raise ValueError("Backbone feature tensor is non-finite")
    return features.astype("<f4")


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or not MANIFEST_COLUMNS.issubset(reader.fieldnames):
            raise ValueError("Predicted-structure manifest schema differs")
        rows = list(reader)
    if not rows or len(rows) != len({row["member"] for row in rows}):
        raise ValueError("Predicted-structure manifest is empty or duplicated")
    return sorted(rows, key=lambda row: row["member"])


def run(arguments: argparse.Namespace) -> dict[str, object]:
    if sha256_file(arguments.structure_manifest) != arguments.structure_manifest_sha256:
        raise ValueError("Predicted-structure manifest SHA-256 differs")
    if arguments.output_features.exists() or arguments.output_index.exists() or arguments.output_result.exists():
        raise FileExistsError("Backbone feature output already exists")
    rows = load_manifest(arguments.structure_manifest)
    arguments.output_features.parent.mkdir(parents=True, exist_ok=True)
    feature_fd, feature_name = tempfile.mkstemp(
        prefix=f".{arguments.output_features.name}.", dir=arguments.output_features.parent
    )
    index_fd, index_name = tempfile.mkstemp(
        prefix=f".{arguments.output_index.name}.", dir=arguments.output_index.parent
    )
    feature_temporary = Path(feature_name)
    index_temporary = Path(index_name)
    residue_offset = 0
    try:
        with os.fdopen(feature_fd, "wb") as feature_handle, os.fdopen(
            index_fd, "w", encoding="utf-8", newline=""
        ) as index_handle:
            writer = csv.DictWriter(
                index_handle, fieldnames=INDEX_COLUMNS, delimiter="\t", lineterminator="\n"
            )
            writer.writeheader()
            for row in rows:
                pdb = arguments.structure_root / "pdb" / row["member"]
                if sha256_file(pdb) != row["pdb_sha256"]:
                    raise ValueError("Predicted PDB SHA-256 differs")
                sequence, coordinates, plddt = parse_backbone(pdb)
                sequence_sha = hashlib.sha256(sequence.encode("ascii")).hexdigest()
                if sequence != row["sequence"] or sequence_sha != row["sequence_sha256"]:
                    raise ValueError("Predicted PDB sequence authority differs")
                if len(sequence) != int(row["residue_count"]):
                    raise ValueError("Predicted PDB residue count differs")
                features = backbone_features(coordinates, plddt)
                feature_handle.write(features.tobytes(order="C"))
                writer.writerow(
                    {
                        "member": row["member"],
                        "sequence_sha256": sequence_sha,
                        "residue_offset": residue_offset,
                        "residue_count": len(sequence),
                        "feature_dimension": FEATURE_DIMENSION,
                    }
                )
                residue_offset += len(sequence)
            feature_handle.flush()
            os.fsync(feature_handle.fileno())
            index_handle.flush()
            os.fsync(index_handle.fileno())
        os.link(feature_temporary, arguments.output_features)
        os.link(index_temporary, arguments.output_index)
    finally:
        feature_temporary.unlink(missing_ok=True)
        index_temporary.unlink(missing_ok=True)
    payload: dict[str, object] = {
        "authorization": {"model_training": False, "structure_shard_materialization": True},
        "claim_boundary": "Local-backbone features from authenticated train-only predicted monomers; no experimental bound, validation, or test structure was read.",
        "counts": {
            "feature_dimension": FEATURE_DIMENSION,
            "predicted_structure_members": len(rows),
            "predicted_structure_residues": residue_offset,
        },
        "format_version": "pinder_predicted_backbone_features_result_v1",
        "inputs": {"structure_manifest_sha256": arguments.structure_manifest_sha256},
        "normalization": BOND_NORMALIZATION,
        "outputs": {
            "feature_binary_sha256": sha256_file(arguments.output_features),
            "feature_index_sha256": sha256_file(arguments.output_index),
        },
        "status": "PASS_AUTHENTICATED_PREDICTED_BACKBONE_FEATURES",
        "test_access": {
            "bound_coordinates_read": False,
            "pinder_test_structures_read": False,
            "pinder_validation_structures_read": False,
        },
    }
    create_only(arguments.output_result, canonical_json_bytes(payload))
    return payload


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--structure-root", type=Path, required=True)
    parser.add_argument("--structure-manifest", type=Path, required=True)
    parser.add_argument("--structure-manifest-sha256", required=True)
    parser.add_argument("--output-features", type=Path, required=True)
    parser.add_argument("--output-index", type=Path, required=True)
    parser.add_argument("--output-result", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_arguments()), sort_keys=True))
