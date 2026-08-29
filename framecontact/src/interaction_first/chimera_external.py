"""Fail-closed CHIMERA adapter for frozen external residue evaluation.

CHIMERA feature files are PyTorch zip archives whose payloads contain NumPy
arrays, strings, lists, and dictionaries.  This module intentionally does not
call unrestricted ``torch.load(..., weights_only=False)``.  It extracts the
single pickle member and permits only the minimal NumPy reconstruction globals
used by the pinned official release.
"""

from __future__ import annotations

import _codecs
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import pickle
import re
from typing import Any, Iterable, Mapping
import zipfile

import numpy as np
from scipy.spatial import cKDTree

try:
    from numpy._core import multiarray as _numpy_multiarray
except ImportError:  # NumPy < 2.0
    from numpy.core import multiarray as _numpy_multiarray


PROTOCOL_SCHEMA = "chimera_zero_shot_residue_interface_protocol_v1"
RECEIPT_SCHEMA = "chimera_zero_shot_frozen_protocol_receipt_v1"
DATASET_ID = "mansoorbaloch/chimera-bench"
PINNED_REVISION = "54d5a2a1e59a329db7a6c09184652ed225188940"
SPLIT_NAMES = ("epitope_group", "antigen_fold", "temporal")
SUBSETS = ("train", "val", "test")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SEQUENCE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWYXBZUOJ]+$")


class ChimeraExternalError(ValueError):
    """Raised when external data or its frozen authority fails validation."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_frozen_authority(protocol_path: Path, receipt_path: Path) -> dict[str, Any]:
    """Require a pre-access protocol whose six checkpoint hashes were verified."""
    for path, name in ((protocol_path, "protocol"), (receipt_path, "receipt")):
        if path.is_symlink() or not path.is_file():
            raise ChimeraExternalError(f"{name} must be a regular non-symlink file")
    try:
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ChimeraExternalError("cannot parse frozen external authority") from error
    if protocol.get("schema_version") != PROTOCOL_SCHEMA:
        raise ChimeraExternalError("unexpected external protocol schema")
    if receipt.get("schema_version") != RECEIPT_SCHEMA:
        raise ChimeraExternalError("unexpected external receipt schema")
    if receipt.get("protocol_sha256") != sha256_file(protocol_path):
        raise ChimeraExternalError("receipt does not bind this protocol")
    if receipt.get("checkpoint_digests_verified") is not True:
        raise ChimeraExternalError("remote checkpoint digests were not verified")
    if receipt.get("external_labels_or_model_scores_read_before_freeze") is not False:
        raise ChimeraExternalError("receipt does not certify pre-access freezing")
    if receipt.get("dataset_pinned_revision") != PINNED_REVISION:
        raise ChimeraExternalError("receipt dataset revision differs")
    dataset = protocol.get("dataset", {})
    if (
        dataset.get("dataset_id") != DATASET_ID
        or dataset.get("pinned_revision") != PINNED_REVISION
    ):
        raise ChimeraExternalError("protocol dataset authority differs")
    return protocol


def load_split(path: Path, *, expected_sha256: str | None = None) -> dict[str, tuple[str, ...]]:
    """Load one immutable CHIMERA split and reject overlap or schema drift."""
    if path.is_symlink() or not path.is_file():
        raise ChimeraExternalError("split must be a regular non-symlink file")
    if expected_sha256 is not None and sha256_file(path) != expected_sha256:
        raise ChimeraExternalError("split SHA-256 differs")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ChimeraExternalError("cannot parse CHIMERA split") from error
    if not isinstance(payload, dict) or set(payload) != set(SUBSETS):
        raise ChimeraExternalError("split must contain exactly train, val, and test")
    result: dict[str, tuple[str, ...]] = {}
    seen: set[str] = set()
    for role in SUBSETS:
        values = payload[role]
        if not isinstance(values, list) or not values:
            raise ChimeraExternalError(f"split role {role} is empty or malformed")
        identifiers = tuple(str(value) for value in values)
        if any(not value or "/" in value or "\\" in value for value in identifiers):
            raise ChimeraExternalError("split contains an unsafe complex ID")
        if len(set(identifiers)) != len(identifiers) or seen.intersection(identifiers):
            raise ChimeraExternalError("split roles overlap or contain duplicates")
        seen.update(identifiers)
        result[role] = identifiers
    return result


class _RestrictedNumpyUnpickler(pickle.Unpickler):
    _ALLOWED: dict[tuple[str, str], Any] = {
        ("numpy.core.multiarray", "_reconstruct"): _numpy_multiarray._reconstruct,
        ("numpy.core.multiarray", "scalar"): _numpy_multiarray.scalar,
        ("numpy._core.multiarray", "_reconstruct"): _numpy_multiarray._reconstruct,
        ("numpy._core.multiarray", "scalar"): _numpy_multiarray.scalar,
        ("numpy", "ndarray"): np.ndarray,
        ("numpy", "dtype"): np.dtype,
        ("_codecs", "encode"): _codecs.encode,
    }

    def find_class(self, module: str, name: str) -> Any:
        try:
            return self._ALLOWED[(module, name)]
        except KeyError as error:
            raise pickle.UnpicklingError(f"blocked pickle global {module}.{name}") from error


def load_feature_archive(path: Path, *, expected_sha256: str | None = None) -> dict[str, Any]:
    """Safely load and validate one pinned CHIMERA NumPy feature archive."""
    if path.is_symlink() or not path.is_file():
        raise ChimeraExternalError("feature archive must be a regular non-symlink file")
    if expected_sha256 is not None and sha256_file(path) != expected_sha256:
        raise ChimeraExternalError("feature archive SHA-256 differs")
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            pickle_members = [name for name in names if name.endswith("/data.pkl")]
            storage_members = [name for name in names if "/data/" in name]
            if len(pickle_members) != 1 or storage_members:
                raise ChimeraExternalError(
                    "feature archive is not the expected NumPy-only serialization"
                )
            raw = archive.read(pickle_members[0])
    except (OSError, zipfile.BadZipFile, KeyError) as error:
        raise ChimeraExternalError("cannot read CHIMERA feature archive") from error
    try:
        payload = _RestrictedNumpyUnpickler(io.BytesIO(raw)).load()
    except (pickle.UnpicklingError, EOFError, ValueError, TypeError) as error:
        raise ChimeraExternalError("feature pickle violates the restricted schema") from error
    if not isinstance(payload, dict):
        raise ChimeraExternalError("feature payload must be a dictionary")
    _validate_feature_payload(payload, expected_complex_id=path.stem)
    return payload


def _sequence(value: Any, name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ChimeraExternalError(f"{name} must be a string")
    value = value.strip().upper()
    if not value and allow_empty:
        return value
    if not value or not _SEQUENCE.fullmatch(value):
        raise ChimeraExternalError(f"{name} contains unsupported residues")
    return value


def _axis_arrays(payload: Mapping[str, Any], prefix: str, sequence: str) -> None:
    coords = payload.get(f"{prefix}_atom14_coords")
    mask = payload.get(f"{prefix}_atom14_mask")
    ca = payload.get(f"{prefix}_ca_coords")
    length = len(sequence)
    if not (
        isinstance(coords, np.ndarray)
        and coords.shape == (length, 14, 3)
        and coords.dtype == np.float32
        and isinstance(mask, np.ndarray)
        and mask.shape == (length, 14)
        and mask.dtype == np.bool_
        and isinstance(ca, np.ndarray)
        and ca.shape == (length, 3)
        and ca.dtype == np.float32
    ):
        raise ChimeraExternalError(f"{prefix} coordinate schema differs")
    if not np.isfinite(coords).all() or not np.isfinite(ca).all():
        raise ChimeraExternalError(f"{prefix} coordinates contain non-finite values")
    ca_valid = mask[:, 1]
    if not np.allclose(coords[ca_valid, 1], ca[ca_valid], rtol=0.0, atol=1e-4):
        raise ChimeraExternalError(f"{prefix} CA arrays disagree")


def _validate_annotation_list(value: Any, name: str, *, contact: bool = False) -> None:
    if not isinstance(value, list):
        raise ChimeraExternalError(f"{name} must be a list")
    width = 7 if contact else 3
    for index, item in enumerate(value):
        if not isinstance(item, tuple) or len(item) != width:
            raise ChimeraExternalError(f"{name}[{index}] has an invalid tuple schema")
        if contact:
            left = item[:3]
            right = item[3:6]
            distance = item[6]
            if not np.isfinite(float(distance)) or float(distance) > 4.5001:
                raise ChimeraExternalError(f"{name}[{index}] has an invalid distance")
            triples = (left, right)
        else:
            triples = (item,)
        for chain, residue_number, residue_name in triples:
            if (
                not isinstance(chain, str)
                or not chain
                or isinstance(residue_number, bool)
                or not isinstance(residue_number, int)
                or not isinstance(residue_name, str)
                or len(residue_name) != 3
            ):
                raise ChimeraExternalError(f"{name}[{index}] has invalid residue fields")


def _validate_feature_payload(payload: Mapping[str, Any], expected_complex_id: str) -> None:
    required = {
        "complex_id",
        "heavy_sequence",
        "light_sequence",
        "antigen_sequence",
        "heavy_atom14_coords",
        "heavy_atom14_mask",
        "heavy_ca_coords",
        "light_atom14_coords",
        "light_atom14_mask",
        "light_ca_coords",
        "antigen_atom14_coords",
        "antigen_atom14_mask",
        "antigen_ca_coords",
        "epitope_residues",
        "paratope_residues",
        "contact_pairs",
    }
    if not required.issubset(payload):
        raise ChimeraExternalError(
            f"feature payload is missing keys: {sorted(required - set(payload))}"
        )
    if payload["complex_id"] != expected_complex_id:
        raise ChimeraExternalError("complex ID differs from feature filename")
    heavy = _sequence(payload["heavy_sequence"], "heavy_sequence")
    light = _sequence(payload["light_sequence"], "light_sequence", allow_empty=True)
    antigen = _sequence(payload["antigen_sequence"], "antigen_sequence")
    _axis_arrays(payload, "heavy", heavy)
    _axis_arrays(payload, "light", light)
    _axis_arrays(payload, "antigen", antigen)
    _validate_annotation_list(payload["epitope_residues"], "epitope_residues")
    _validate_annotation_list(payload["paratope_residues"], "paratope_residues")
    _validate_annotation_list(payload["contact_pairs"], "contact_pairs", contact=True)


_AA3 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEC": "U", "PYL": "O", "ASX": "B", "GLX": "Z",
}


@dataclass(frozen=True)
class PdbResidue:
    chain: str
    number: int
    insertion_code: str
    name: str
    amino_acid: str
    ca: np.ndarray


@dataclass(frozen=True)
class PdbAtomResidue:
    """First-model PDB residue with the selected coordinates for every atom.

    CHIMERA's published contact tuples use an all-atom cutoff.  The atom list is
    retained only to disambiguate insertion-coded residues that collapse to the
    same ``(chain, number, residue name)`` annotation key; it is never used to
    generate new labels.
    """

    chain: str
    number: int
    insertion_code: str
    name: str
    amino_acid: str
    ca: np.ndarray
    atom_names: tuple[str, ...]
    atoms: np.ndarray


def parse_pdb_residues(path: Path) -> dict[str, tuple[PdbResidue, ...]]:
    """Read first-model polymer residues with one unambiguous CA atom each."""
    if path.is_symlink() or not path.is_file():
        raise ChimeraExternalError("PDB must be a regular non-symlink file")
    residues: dict[tuple[str, int, str, str], dict[str, Any]] = {}
    order: list[tuple[str, int, str, str]] = []
    try:
        lines = path.read_text(encoding="ascii", errors="strict").splitlines()
    except (OSError, UnicodeError) as error:
        raise ChimeraExternalError("cannot read PDB") from error
    for line in lines:
        if line.startswith("ENDMDL"):
            break
        if not line.startswith(("ATOM  ", "HETATM")) or len(line) < 54:
            continue
        name = line[17:20].strip().upper()
        if name not in _AA3:
            continue
        atom = line[12:16].strip().upper()
        if atom != "CA":
            continue
        altloc = line[16].strip()
        if altloc not in {"", "A", "1"}:
            continue
        chain = line[21].strip()
        if not chain:
            raise ChimeraExternalError("PDB contains an empty polymer chain ID")
        try:
            number = int(line[22:26])
            xyz = np.asarray(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                dtype=np.float32,
            )
            occupancy = float(line[54:60]) if len(line) >= 60 and line[54:60].strip() else 0.0
        except ValueError as error:
            raise ChimeraExternalError("PDB CA record is malformed") from error
        key = (chain, number, line[26].strip(), name)
        if key not in residues:
            order.append(key)
            residues[key] = {"ca": xyz, "occupancy": occupancy}
        elif occupancy > residues[key]["occupancy"]:
            residues[key] = {"ca": xyz, "occupancy": occupancy}
    by_chain: dict[str, list[PdbResidue]] = {}
    for chain, number, insertion, name in order:
        value = residues[(chain, number, insertion, name)]
        by_chain.setdefault(chain, []).append(
            PdbResidue(
                chain=chain,
                number=number,
                insertion_code=insertion,
                name=name,
                amino_acid=_AA3[name],
                ca=value["ca"],
            )
        )
    if not by_chain:
        raise ChimeraExternalError("PDB has no supported first-model CA residues")
    return {key: tuple(value) for key, value in by_chain.items()}


def parse_pdb_atom_residues(path: Path) -> dict[str, tuple[PdbAtomResidue, ...]]:
    """Read supported first-model residues and select one coordinate per atom.

    Blank/A/1 alternate locations are accepted, matching the conservative CA
    parser above.  For duplicate atom names, the highest-occupancy coordinate is
    selected deterministically.  Hydrogens are deliberately retained because
    the pinned CHIMERA structures and stored contact distances sometimes include
    them despite the dataset documentation's informal "all-heavy-atom" wording.
    """
    if path.is_symlink() or not path.is_file():
        raise ChimeraExternalError("PDB must be a regular non-symlink file")
    residues: dict[
        tuple[str, int, str, str], dict[str, tuple[np.ndarray, float]]
    ] = {}
    order: list[tuple[str, int, str, str]] = []
    try:
        lines = path.read_text(encoding="ascii", errors="strict").splitlines()
    except (OSError, UnicodeError) as error:
        raise ChimeraExternalError("cannot read PDB") from error
    for line in lines:
        if line.startswith("ENDMDL"):
            break
        if not line.startswith(("ATOM  ", "HETATM")) or len(line) < 54:
            continue
        residue_name = line[17:20].strip().upper()
        if residue_name not in _AA3:
            continue
        altloc = line[16].strip()
        if altloc not in {"", "A", "1"}:
            continue
        chain = line[21].strip()
        if not chain:
            raise ChimeraExternalError("PDB contains an empty polymer chain ID")
        atom_name = line[12:16].strip().upper()
        if not atom_name:
            raise ChimeraExternalError("PDB contains an empty atom name")
        try:
            number = int(line[22:26])
            xyz = np.asarray(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                dtype=np.float32,
            )
            occupancy = (
                float(line[54:60])
                if len(line) >= 60 and line[54:60].strip()
                else 0.0
            )
        except ValueError as error:
            raise ChimeraExternalError("PDB atom record is malformed") from error
        key = (chain, number, line[26].strip(), residue_name)
        if key not in residues:
            order.append(key)
            residues[key] = {}
        previous = residues[key].get(atom_name)
        if previous is None or occupancy > previous[1]:
            residues[key][atom_name] = (xyz, occupancy)

    by_chain: dict[str, list[PdbAtomResidue]] = {}
    for chain, number, insertion, residue_name in order:
        selected = residues[(chain, number, insertion, residue_name)]
        if "CA" not in selected:
            continue
        atom_names = tuple(sorted(selected))
        atoms = np.stack([selected[name][0] for name in atom_names], axis=0)
        if not np.isfinite(atoms).all():
            raise ChimeraExternalError("PDB atom coordinates contain non-finite values")
        by_chain.setdefault(chain, []).append(
            PdbAtomResidue(
                chain=chain,
                number=number,
                insertion_code=insertion,
                name=residue_name,
                amino_acid=_AA3[residue_name],
                ca=selected["CA"][0],
                atom_names=atom_names,
                atoms=atoms,
            )
        )
    if not by_chain:
        raise ChimeraExternalError("PDB has no supported first-model atom residues")
    return {key: tuple(value) for key, value in by_chain.items()}


def _coordinate_anchored_axis(
    chains: Mapping[str, tuple[PdbAtomResidue, ...]],
    chain: str,
    sequence: str,
    feature_ca: np.ndarray,
    feature_ca_valid: np.ndarray,
    name: str,
    *,
    tolerance: float = 1e-3,
) -> tuple[PdbAtomResidue | None, ...]:
    """Anchor PDB residue identities onto a feature axis using matching CAs.

    PDB-only terminal residues and feature-only unresolved residues are allowed.
    Every accepted anchor must be unique, residue-type consistent, one-to-one,
    and monotonic in PDB order.  Contact mapping later fails if an official
    positive tuple refers to an unanchored residue.
    """
    if chain not in chains:
        raise ChimeraExternalError(f"PDB is missing {name} chain {chain!r}")
    if feature_ca.shape != (len(sequence), 3) or feature_ca_valid.shape != (
        len(sequence),
    ):
        raise ChimeraExternalError(f"{name} feature CA axis is malformed")
    axis: list[PdbAtomResidue | None] = [None] * len(sequence)
    mapped_feature_indices: list[int] = []
    for residue in chains[chain]:
        distances = np.linalg.norm(feature_ca - residue.ca[None, :], axis=1)
        candidates = np.flatnonzero(
            feature_ca_valid
            & (distances <= tolerance)
            & np.asarray([letter == residue.amino_acid for letter in sequence])
        )
        if len(candidates) > 1:
            raise ChimeraExternalError(
                f"{name} PDB residue has multiple feature-axis CA matches"
            )
        if len(candidates) == 0:
            continue
        index = int(candidates[0])
        if axis[index] is not None:
            raise ChimeraExternalError(
                f"{name} feature residue has multiple PDB-axis CA matches"
            )
        axis[index] = residue
        mapped_feature_indices.append(index)
    if not mapped_feature_indices:
        raise ChimeraExternalError(f"{name} PDB has no coordinate-anchored residues")
    if any(
        right <= left
        for left, right in zip(mapped_feature_indices, mapped_feature_indices[1:])
    ):
        raise ChimeraExternalError(f"{name} coordinate anchors are not monotonic")
    return tuple(axis)


def _minimum_atom_distance(left: PdbAtomResidue, right: PdbAtomResidue) -> float:
    squared = np.sum(
        (left.atoms[:, None, :] - right.atoms[None, :, :]) ** 2,
        axis=-1,
    )
    return float(np.sqrt(max(float(squared.min()), 0.0)))


def _map_provided_contact_pairs(
    antibody_axis: tuple[PdbAtomResidue | None, ...],
    antigen_axis: tuple[PdbAtomResidue | None, ...],
    contact_pairs: Iterable[tuple[Any, ...]],
) -> tuple[tuple[tuple[int, int], ...], dict[str, int | float]]:
    """Map official contacts to feature indices without regenerating labels."""

    def lookup(
        axis: tuple[PdbAtomResidue | None, ...],
    ) -> dict[tuple[str, int, str], list[int]]:
        result: dict[tuple[str, int, str], list[int]] = {}
        for index, residue in enumerate(axis):
            if residue is not None:
                result.setdefault(
                    (residue.chain, residue.number, residue.name), []
                ).append(index)
        return result

    antibody_lookup = lookup(antibody_axis)
    antigen_lookup = lookup(antigen_axis)
    mapped: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    distance_disambiguated = 0
    unique_distance_mismatches = 0
    maximum_distance_error = 0.0
    for item in contact_pairs:
        left_key = (str(item[0]), int(item[1]), str(item[2]).upper())
        right_key = (str(item[3]), int(item[4]), str(item[5]).upper())
        stored_distance = float(item[6])
        candidates = [
            (left_index, right_index)
            for left_index in antibody_lookup.get(left_key, ())
            for right_index in antigen_lookup.get(right_key, ())
        ]
        if not candidates:
            raise ChimeraExternalError(
                "provided contact residue is absent from coordinate-anchored axes"
            )
        if len(candidates) > 1:
            distance_matches = []
            for left_index, right_index in candidates:
                observed = _minimum_atom_distance(
                    antibody_axis[left_index],  # type: ignore[arg-type]
                    antigen_axis[right_index],  # type: ignore[arg-type]
                )
                if round(observed, 2) == round(stored_distance, 2):
                    distance_matches.append((left_index, right_index))
            if len(distance_matches) != 1:
                raise ChimeraExternalError(
                    "ambiguous contact tuple was not uniquely resolved by stored distance"
                )
            pair = distance_matches[0]
            distance_disambiguated += 1
        else:
            pair = candidates[0]
            observed = _minimum_atom_distance(
                antibody_axis[pair[0]],  # type: ignore[arg-type]
                antigen_axis[pair[1]],  # type: ignore[arg-type]
            )
            error = abs(observed - stored_distance)
            maximum_distance_error = max(maximum_distance_error, error)
            if round(observed, 2) != round(stored_distance, 2):
                unique_distance_mismatches += 1
        if pair in seen:
            raise ChimeraExternalError("provided contacts collapse to a duplicate pair")
        seen.add(pair)
        mapped.append(pair)
    if not mapped:
        raise ChimeraExternalError("complex has no provided contact pair")
    return tuple(mapped), {
        "distance_disambiguated_pair_count": distance_disambiguated,
        "unique_pair_distance_mismatch_count": unique_distance_mismatches,
        "maximum_unique_pair_distance_error": maximum_distance_error,
    }


def _chain_axis(
    chains: Mapping[str, tuple[PdbResidue, ...]],
    chain: str,
    sequence: str,
    feature_ca: np.ndarray,
    name: str,
) -> tuple[PdbResidue, ...]:
    if chain not in chains:
        raise ChimeraExternalError(f"PDB is missing {name} chain {chain!r}")
    residues = chains[chain]
    observed_sequence = "".join(item.amino_acid for item in residues)
    if observed_sequence != sequence:
        raise ChimeraExternalError(f"{name} PDB sequence does not exactly match feature axis")
    pdb_ca = np.stack([item.ca for item in residues], axis=0)
    if pdb_ca.shape != feature_ca.shape or not np.allclose(
        pdb_ca, feature_ca, rtol=0.0, atol=1e-3
    ):
        raise ChimeraExternalError(f"{name} PDB and feature CA coordinates disagree")
    return residues


def _annotation_keys(value: Iterable[tuple[Any, ...]]) -> set[tuple[str, int, str]]:
    return {(str(item[0]), int(item[1]), str(item[2]).upper()) for item in value}


def _labels_for_axis(
    axes: Iterable[tuple[PdbResidue, ...]],
    positive_keys: set[tuple[str, int, str]],
    name: str,
) -> np.ndarray:
    residues = tuple(item for axis in axes for item in axis)
    lookup: dict[tuple[str, int, str], list[int]] = {}
    for index, residue in enumerate(residues):
        lookup.setdefault((residue.chain, residue.number, residue.name), []).append(index)
    unknown = sorted(positive_keys - set(lookup))
    ambiguous = sorted(key for key in positive_keys if len(lookup.get(key, ())) != 1)
    if unknown or ambiguous:
        raise ChimeraExternalError(
            f"{name} annotation mapping failed; unknown={unknown[:5]}, ambiguous={ambiguous[:5]}"
        )
    labels = np.zeros(len(residues), dtype=np.float32)
    for key in positive_keys:
        labels[lookup[key][0]] = 1.0
    return labels


def _backbone(payload: Mapping[str, Any], prefix: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coords = np.asarray(payload[f"{prefix}_atom14_coords"], dtype=np.float32)
    atom_mask = np.asarray(payload[f"{prefix}_atom14_mask"], dtype=np.bool_)
    valid = atom_mask[:, :3].all(axis=1)
    backbone = coords[:, :3].copy()
    backbone[~valid] = np.nan
    return backbone, valid, ~valid


def _native_atom14_contacts(
    antibody_coords: np.ndarray,
    antibody_mask: np.ndarray,
    antigen_coords: np.ndarray,
    antigen_mask: np.ndarray,
    *,
    cutoff: float,
) -> tuple[np.ndarray, np.ndarray, tuple[tuple[int, int, float], ...]]:
    """Compute contacts over the atoms present in the fixed-width feature arrays.

    This is not equivalent to CHIMERA's official all-atom annotations: the
    release arrays omit residue-specific distal atoms for multiple residue types.
    It remains only as a falsified diagnostic control.
    """
    if not np.isfinite(cutoff) or cutoff <= 0.0:
        raise ChimeraExternalError("contact cutoff must be finite and positive")
    if (
        antibody_coords.ndim != 3
        or antibody_coords.shape[1:] != (14, 3)
        or antibody_mask.shape != antibody_coords.shape[:2]
        or antigen_coords.ndim != 3
        or antigen_coords.shape[1:] != (14, 3)
        or antigen_mask.shape != antigen_coords.shape[:2]
    ):
        raise ChimeraExternalError("atom14 arrays do not share the native schema")

    antigen_atom_blocks = [
        antigen_coords[index, antigen_mask[index]]
        for index in range(len(antigen_coords))
        if bool(antigen_mask[index].any())
    ]
    if not antigen_atom_blocks:
        raise ChimeraExternalError("antigen has no valid atom14 coordinates")
    antigen_atom_residue = np.concatenate(
        [
            np.full(int(antigen_mask[index].sum()), index, dtype=np.int64)
            for index in range(len(antigen_coords))
            if bool(antigen_mask[index].any())
        ]
    )
    antigen_atoms = np.concatenate(antigen_atom_blocks, axis=0)
    tree = cKDTree(antigen_atoms)
    cutoff_squared = cutoff * cutoff
    epitope = np.zeros(len(antigen_coords), dtype=np.bool_)
    paratope = np.zeros(len(antibody_coords), dtype=np.bool_)
    pairs: list[tuple[int, int, float]] = []

    for antibody_index in range(len(antibody_coords)):
        valid_antibody_atoms = antibody_coords[
            antibody_index, antibody_mask[antibody_index]
        ]
        if not len(valid_antibody_atoms):
            continue
        atom_hits = tree.query_ball_point(valid_antibody_atoms, cutoff)
        flat_hits = sorted({atom for hits in atom_hits for atom in hits})
        if not flat_hits:
            continue
        antigen_indices = np.unique(antigen_atom_residue[flat_hits])
        for antigen_index_value in antigen_indices:
            antigen_index = int(antigen_index_value)
            valid_antigen_atoms = antigen_coords[
                antigen_index, antigen_mask[antigen_index]
            ]
            squared = np.sum(
                (
                    valid_antibody_atoms[:, None, :]
                    - valid_antigen_atoms[None, :, :]
                )
                ** 2,
                axis=-1,
            )
            minimum_squared = float(squared.min())
            if minimum_squared <= cutoff_squared + 1e-6:
                distance = float(np.sqrt(max(minimum_squared, 0.0)))
                pairs.append((antibody_index, antigen_index, distance))
                paratope[antibody_index] = True
                epitope[antigen_index] = True
    return epitope, paratope, tuple(pairs)


def _rounded_distance_counts(values: Iterable[float]) -> dict[float, int]:
    counts: dict[float, int] = {}
    for value in values:
        rounded = round(float(value), 2)
        counts[rounded] = counts.get(rounded, 0) + 1
    return counts


def adapt_complex_native_axis(
    feature: Mapping[str, Any],
    *,
    cutoff: float = 4.5,
) -> dict[str, Any]:
    """Adapt CHIMERA without a PDB-numbering round trip.

    This post-access diagnostic reconstructs labels from the incomplete fixed
    atom14 arrays. A release audit showed that it cannot reproduce most official
    contact sets, so it must not be used for benchmark scoring or to relabel a
    same-split rerun as a new confirmatory frozen test.
    """
    complex_id = str(feature.get("complex_id", ""))
    _validate_feature_payload(feature, expected_complex_id=complex_id)
    parts = complex_id.split("_")
    if len(parts) != 4 or any(not value for value in parts):
        raise ChimeraExternalError("complex ID must encode PDB, heavy, light, antigen chains")
    pdb_id, heavy_chain, light_chain, antigen_chain = parts
    heavy_sequence = _sequence(feature["heavy_sequence"], "heavy_sequence")
    light_sequence = _sequence(
        feature["light_sequence"], "light_sequence", allow_empty=True
    )
    antigen_sequence = _sequence(feature["antigen_sequence"], "antigen_sequence")

    heavy_coords = np.asarray(feature["heavy_atom14_coords"], dtype=np.float32)
    heavy_mask = np.asarray(feature["heavy_atom14_mask"], dtype=np.bool_)
    light_coords = np.asarray(feature["light_atom14_coords"], dtype=np.float32)
    light_mask = np.asarray(feature["light_atom14_mask"], dtype=np.bool_)
    antigen_coords = np.asarray(feature["antigen_atom14_coords"], dtype=np.float32)
    antigen_mask = np.asarray(feature["antigen_atom14_mask"], dtype=np.bool_)
    antibody_coords = np.concatenate((heavy_coords, light_coords), axis=0)
    antibody_atom_mask = np.concatenate((heavy_mask, light_mask), axis=0)
    epitope, paratope, native_pairs = _native_atom14_contacts(
        antibody_coords,
        antibody_atom_mask,
        antigen_coords,
        antigen_mask,
        cutoff=cutoff,
    )
    if not bool(epitope.any()) or not bool(paratope.any()):
        raise ChimeraExternalError("native atom14 axes contain no positive contact")

    epitope_keys = _annotation_keys(feature["epitope_residues"])
    paratope_keys = _annotation_keys(feature["paratope_residues"])
    contact_paratope = _annotation_keys(item[:3] for item in feature["contact_pairs"])
    contact_epitope = _annotation_keys(item[3:6] for item in feature["contact_pairs"])
    if epitope_keys != contact_epitope or paratope_keys != contact_paratope:
        raise ChimeraExternalError("provided annotations disagree with contact-pair residues")

    provided_pair_count = len(feature["contact_pairs"])
    if (
        provided_pair_count > len(native_pairs)
        or len(epitope_keys) > int(epitope.sum())
        or len(paratope_keys) > int(paratope.sum())
    ):
        raise ChimeraExternalError(
            "native atom14 contacts cannot account for provided annotations"
        )
    provided_distances = _rounded_distance_counts(
        float(item[6]) for item in feature["contact_pairs"]
    )
    native_distances = _rounded_distance_counts(item[2] for item in native_pairs)
    if any(
        native_distances.get(distance, 0) < count
        for distance, count in provided_distances.items()
    ):
        raise ChimeraExternalError(
            "native atom14 contact distances disagree with provided contacts"
        )

    heavy_backbone, heavy_valid, heavy_missing = _backbone(feature, "heavy")
    light_backbone, light_valid, light_missing = _backbone(feature, "light")
    antigen_backbone, antigen_valid, antigen_missing = _backbone(feature, "antigen")
    antibody_sequence = heavy_sequence + light_sequence
    return {
        "instance_id": complex_id,
        "antibody_sequence": antibody_sequence,
        "antigen_sequence": antigen_sequence,
        "antibody_backbone": np.concatenate((heavy_backbone, light_backbone), axis=0),
        "antigen_backbone": antigen_backbone,
        "antibody_chain_ids": np.concatenate(
            (
                np.zeros(len(heavy_sequence), dtype=np.int64),
                np.ones(len(light_sequence), dtype=np.int64),
            )
        ),
        "antigen_chain_ids": np.zeros(len(antigen_sequence), dtype=np.int64),
        "antibody_frame_valid_mask": np.concatenate((heavy_valid, light_valid)),
        "antigen_frame_valid_mask": antigen_valid,
        "antibody_missing_frame_mask": np.concatenate((heavy_missing, light_missing)),
        "antigen_missing_frame_mask": antigen_missing,
        "antibody_labels": paratope.astype(np.float32, copy=False),
        "antigen_labels": epitope.astype(np.float32, copy=False),
        "pdb_id": pdb_id.lower(),
        "heavy_chain": heavy_chain,
        "light_chain": light_chain,
        "antigen_chain": antigen_chain,
        "native_contact_pair_count": len(native_pairs),
        "provided_contact_pair_count": provided_pair_count,
    }


def adapt_complex_pdb_anchored(
    feature: Mapping[str, Any], pdb_path: Path
) -> dict[str, Any]:
    """Map official CHIMERA contact tuples onto native feature residue axes.

    Unlike :func:`adapt_complex_native_axis`, this adapter never recomputes
    contacts from the release's incomplete fixed atom14 arrays.  PDB CA
    coordinates establish residue identity; the stored contact tuples remain the
    sole label authority.  All-atom distances are consulted only when insertion
    codes were erased by the published annotation tuple schema.

    This method was developed after external-label access and therefore supports
    secondary coverage/robustness analysis only, not a new confirmatory claim on
    the already accessed CHIMERA splits.
    """
    complex_id = str(feature.get("complex_id", ""))
    _validate_feature_payload(feature, expected_complex_id=complex_id)
    parts = complex_id.split("_")
    if len(parts) != 4 or any(not value for value in parts):
        raise ChimeraExternalError("complex ID must encode PDB, heavy, light, antigen chains")
    pdb_id, heavy_chain, light_chain, antigen_chain = parts
    if pdb_path.stem.casefold() != pdb_id.casefold():
        raise ChimeraExternalError("PDB filename differs from complex ID")

    chains = parse_pdb_atom_residues(pdb_path)
    heavy_sequence = _sequence(feature["heavy_sequence"], "heavy_sequence")
    light_sequence = _sequence(
        feature["light_sequence"], "light_sequence", allow_empty=True
    )
    antigen_sequence = _sequence(feature["antigen_sequence"], "antigen_sequence")
    heavy_axis = _coordinate_anchored_axis(
        chains,
        heavy_chain,
        heavy_sequence,
        np.asarray(feature["heavy_ca_coords"], dtype=np.float32),
        np.asarray(feature["heavy_atom14_mask"], dtype=np.bool_)[:, 1],
        "heavy",
    )
    light_axis: tuple[PdbAtomResidue | None, ...] = ()
    if light_sequence:
        light_axis = _coordinate_anchored_axis(
            chains,
            light_chain,
            light_sequence,
            np.asarray(feature["light_ca_coords"], dtype=np.float32),
            np.asarray(feature["light_atom14_mask"], dtype=np.bool_)[:, 1],
            "light",
        )
    antigen_axis = _coordinate_anchored_axis(
        chains,
        antigen_chain,
        antigen_sequence,
        np.asarray(feature["antigen_ca_coords"], dtype=np.float32),
        np.asarray(feature["antigen_atom14_mask"], dtype=np.bool_)[:, 1],
        "antigen",
    )
    antibody_axis = heavy_axis + light_axis

    epitope_keys = _annotation_keys(feature["epitope_residues"])
    paratope_keys = _annotation_keys(feature["paratope_residues"])
    contact_paratope = _annotation_keys(item[:3] for item in feature["contact_pairs"])
    contact_epitope = _annotation_keys(item[3:6] for item in feature["contact_pairs"])
    if epitope_keys != contact_epitope or paratope_keys != contact_paratope:
        raise ChimeraExternalError("provided annotations disagree with contact-pair residues")

    mapped_pairs, diagnostics = _map_provided_contact_pairs(
        antibody_axis, antigen_axis, feature["contact_pairs"]
    )
    paratope = np.zeros(len(antibody_axis), dtype=np.float32)
    epitope = np.zeros(len(antigen_axis), dtype=np.float32)
    for antibody_index, antigen_index in mapped_pairs:
        paratope[antibody_index] = 1.0
        epitope[antigen_index] = 1.0

    mapped_paratope_keys = {
        (
            antibody_axis[index].chain,
            antibody_axis[index].number,
            antibody_axis[index].name,
        )
        for index in np.flatnonzero(paratope)
        if antibody_axis[index] is not None
    }
    mapped_epitope_keys = {
        (
            antigen_axis[index].chain,
            antigen_axis[index].number,
            antigen_axis[index].name,
        )
        for index in np.flatnonzero(epitope)
        if antigen_axis[index] is not None
    }
    if mapped_paratope_keys != paratope_keys or mapped_epitope_keys != epitope_keys:
        raise ChimeraExternalError("mapped contact marginals disagree with annotations")

    heavy_backbone, heavy_valid, heavy_missing = _backbone(feature, "heavy")
    light_backbone, light_valid, light_missing = _backbone(feature, "light")
    antigen_backbone, antigen_valid, antigen_missing = _backbone(feature, "antigen")
    antibody_sequence = heavy_sequence + light_sequence
    return {
        "instance_id": complex_id,
        "antibody_sequence": antibody_sequence,
        "antigen_sequence": antigen_sequence,
        "antibody_backbone": np.concatenate((heavy_backbone, light_backbone), axis=0),
        "antigen_backbone": antigen_backbone,
        "antibody_chain_ids": np.concatenate(
            (
                np.zeros(len(heavy_sequence), dtype=np.int64),
                np.ones(len(light_sequence), dtype=np.int64),
            )
        ),
        "antigen_chain_ids": np.zeros(len(antigen_sequence), dtype=np.int64),
        "antibody_frame_valid_mask": np.concatenate((heavy_valid, light_valid)),
        "antigen_frame_valid_mask": antigen_valid,
        "antibody_missing_frame_mask": np.concatenate((heavy_missing, light_missing)),
        "antigen_missing_frame_mask": antigen_missing,
        "antibody_labels": paratope,
        "antigen_labels": epitope,
        "pdb_id": pdb_id.lower(),
        "heavy_chain": heavy_chain,
        "light_chain": light_chain,
        "antigen_chain": antigen_chain,
        "provided_contact_pair_count": len(mapped_pairs),
        "coordinate_anchored_antibody_residues": sum(
            item is not None for item in antibody_axis
        ),
        "coordinate_anchored_antigen_residues": sum(
            item is not None for item in antigen_axis
        ),
        "feature_only_antibody_residues": sum(
            item is None for item in antibody_axis
        ),
        "feature_only_antigen_residues": sum(item is None for item in antigen_axis),
        **diagnostics,
    }


def adapt_complex(feature: Mapping[str, Any], pdb_path: Path) -> dict[str, Any]:
    """Map one official CHIMERA complex to the existing geometry batch contract."""
    complex_id = str(feature["complex_id"])
    parts = complex_id.split("_")
    if len(parts) != 4 or any(not value for value in parts):
        raise ChimeraExternalError("complex ID must encode PDB, heavy, light, antigen chains")
    pdb_id, heavy_chain, light_chain, antigen_chain = parts
    if pdb_path.stem.casefold() != pdb_id.casefold():
        raise ChimeraExternalError("PDB filename differs from complex ID")
    chains = parse_pdb_residues(pdb_path)
    heavy_sequence = _sequence(feature["heavy_sequence"], "heavy_sequence")
    light_sequence = _sequence(feature["light_sequence"], "light_sequence", allow_empty=True)
    antigen_sequence = _sequence(feature["antigen_sequence"], "antigen_sequence")
    heavy_axis = _chain_axis(
        chains, heavy_chain, heavy_sequence, feature["heavy_ca_coords"], "heavy"
    )
    light_axis: tuple[PdbResidue, ...] = ()
    if light_sequence:
        light_axis = _chain_axis(
            chains, light_chain, light_sequence, feature["light_ca_coords"], "light"
        )
    antigen_axis = _chain_axis(
        chains, antigen_chain, antigen_sequence, feature["antigen_ca_coords"], "antigen"
    )

    epitope_keys = _annotation_keys(feature["epitope_residues"])
    paratope_keys = _annotation_keys(feature["paratope_residues"])
    contact_paratope = _annotation_keys(item[:3] for item in feature["contact_pairs"])
    contact_epitope = _annotation_keys(item[3:6] for item in feature["contact_pairs"])
    if epitope_keys != contact_epitope or paratope_keys != contact_paratope:
        raise ChimeraExternalError("provided annotations disagree with contact-pair residues")

    heavy_backbone, heavy_valid, heavy_missing = _backbone(feature, "heavy")
    light_backbone, light_valid, light_missing = _backbone(feature, "light")
    antigen_backbone, antigen_valid, antigen_missing = _backbone(feature, "antigen")
    antibody_sequence = heavy_sequence + light_sequence
    antibody_backbone = np.concatenate((heavy_backbone, light_backbone), axis=0)
    antibody_valid = np.concatenate((heavy_valid, light_valid), axis=0)
    antibody_missing = np.concatenate((heavy_missing, light_missing), axis=0)
    antibody_chain_ids = np.concatenate(
        (
            np.zeros(len(heavy_sequence), dtype=np.int64),
            np.ones(len(light_sequence), dtype=np.int64),
        )
    )
    paratope_labels = _labels_for_axis(
        (heavy_axis, light_axis), paratope_keys, "paratope"
    )
    epitope_labels = _labels_for_axis((antigen_axis,), epitope_keys, "epitope")
    if paratope_labels.sum() <= 0 or epitope_labels.sum() <= 0:
        raise ChimeraExternalError("complex lacks a positive interface residue")
    return {
        "instance_id": complex_id,
        "antibody_sequence": antibody_sequence,
        "antigen_sequence": antigen_sequence,
        "antibody_backbone": antibody_backbone,
        "antigen_backbone": antigen_backbone,
        "antibody_chain_ids": antibody_chain_ids,
        "antigen_chain_ids": np.zeros(len(antigen_sequence), dtype=np.int64),
        "antibody_frame_valid_mask": antibody_valid,
        "antigen_frame_valid_mask": antigen_valid,
        "antibody_missing_frame_mask": antibody_missing,
        "antigen_missing_frame_mask": antigen_missing,
        "antibody_labels": paratope_labels,
        "antigen_labels": epitope_labels,
        "pdb_id": pdb_id.lower(),
        "heavy_chain": heavy_chain,
        "light_chain": light_chain,
        "antigen_chain": antigen_chain,
    }
