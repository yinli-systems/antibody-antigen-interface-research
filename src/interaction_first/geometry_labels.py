"""Prospective residue-pair geometry labels with fail-closed provenance.

This module is data agnostic and intentionally separate from the frozen
D0/R1/R2 protocol implementations. It only transforms explicit atom tables
provided by a caller. It does not read datasets, select models, or authorize
training or evaluation.

Pair tensors use antibody-by-antigen orientation:
``[antibody_residue, antigen_residue, ...]``.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np


FORMAT_VERSION = "pair_geometry_labels_v2"
ORIENTATION_SCHEMA_VERSION = "relative_pose_5d_periodic_bins_v1"
PRIMARY_CONTACT_CUTOFF_ANGSTROM = 4.5
SENSITIVITY_CONTACT_CUTOFFS_ANGSTROM = (5.0, 6.0, 8.0)
DEFAULT_DISTANCE_BIN_EDGES_ANGSTROM = (4.5, 5.0, 6.0, 8.0, 12.0, 16.0, 24.0)
ORIENTATION_AXIS_NAMES = (
    "antibody_direction_azimuth",
    "antibody_direction_polar",
    "antigen_direction_azimuth",
    "antigen_direction_polar",
    "relative_twist",
)
ORIENTATION_AXIS_IS_CYCLIC = (True, False, True, False, True)
ORIENTATION_BIN_COUNT = 24
ORIENTATION_BIN_EDGE_CONVENTION = (
    "internal_right_closed_thresholds_num_bins_minus_one"
)
LABEL_BUILDER_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

_MIN_BACKBONE_BOND_ANGSTROM = 0.8
_MAX_BACKBONE_BOND_ANGSTROM = 2.2
_MIN_FRAME_SINE_CONDITION = 1e-4
_MIN_DIRECTION_PROJECTION_CONDITION = 1e-8
_MAX_ABSOLUTE_COORDINATE_ANGSTROM = 1e6

_REQUIRED_PROVENANCE_FIELDS = frozenset(
    {
        "structure_id",
        "source_sha256",
        "assembly_id",
        "model_id",
        "parser_name",
        "parser_version",
        "coordinate_unit",
        "altloc_policy",
        "occupancy_policy",
        "assembly_generation_policy",
        "label_builder_source_sha256",
    }
)
_RESERVED_PROVENANCE_VALUES = {
    "label_format_version": FORMAT_VERSION,
    "orientation_schema_version": ORIENTATION_SCHEMA_VERSION,
    "pair_orientation": "antibody_by_antigen",
    "distance_unit": "angstrom",
    "primary_contact_definition": (
        "minimum_observed_residue_non_hydrogen_atom_distance_le_4.5_angstrom"
        "_with_positive_or_complete_pair_validity"
    ),
    "distance_definition": (
        "exact_minimum_non_hydrogen_atom_distance_only_when_both_residues"
        "_have_explicit_heavy_atom_completeness"
    ),
    "orientation_definition": (
        "right_handed_N_CA_C_frames_CA_directions_and_projected_x_axis_twist"
    ),
    "orientation_bin_edge_convention": ORIENTATION_BIN_EDGE_CONVENTION,
}
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_ELEMENT_PATTERN = re.compile(r"^[A-Z][a-z]?$")
_HYDROGEN_ELEMENTS = frozenset({"H", "D", "T"})
_KNOWN_ELEMENTS = frozenset(
    {
        "H",
        "He",
        "Li",
        "Be",
        "B",
        "C",
        "N",
        "O",
        "F",
        "Ne",
        "Na",
        "Mg",
        "Al",
        "Si",
        "P",
        "S",
        "Cl",
        "Ar",
        "K",
        "Ca",
        "Sc",
        "Ti",
        "V",
        "Cr",
        "Mn",
        "Fe",
        "Co",
        "Ni",
        "Cu",
        "Zn",
        "Ga",
        "Ge",
        "As",
        "Se",
        "Br",
        "Kr",
        "Rb",
        "Sr",
        "Y",
        "Zr",
        "Nb",
        "Mo",
        "Tc",
        "Ru",
        "Rh",
        "Pd",
        "Ag",
        "Cd",
        "In",
        "Sn",
        "Sb",
        "Te",
        "I",
        "Xe",
        "Cs",
        "Ba",
        "La",
        "Ce",
        "Pr",
        "Nd",
        "Pm",
        "Sm",
        "Eu",
        "Gd",
        "Tb",
        "Dy",
        "Ho",
        "Er",
        "Tm",
        "Yb",
        "Lu",
        "Hf",
        "Ta",
        "W",
        "Re",
        "Os",
        "Ir",
        "Pt",
        "Au",
        "Hg",
        "Tl",
        "Pb",
        "Bi",
        "Po",
        "At",
        "Rn",
        "Fr",
        "Ra",
        "Ac",
        "Th",
        "Pa",
        "U",
        "Np",
        "Pu",
        "Am",
        "Cm",
        "Bk",
        "Cf",
        "Es",
        "Fm",
        "Md",
        "No",
        "Lr",
        "Rf",
        "Db",
        "Sg",
        "Bh",
        "Hs",
        "Mt",
        "Ds",
        "Rg",
        "Cn",
        "Nh",
        "Fl",
        "Mc",
        "Lv",
        "Ts",
        "Og",
        "D",
        "T",
    }
)


@dataclass(frozen=True)
class ResidueAtomInput:
    """One role-specific, explicitly resolved residue atom table.

    Coordinates have shape ``[residue, atom_slot, 3]``. Atom names, atom
    elements, and the resolved mask have shape ``[residue, atom_slot]``.
    Chain and residue identifiers have shape ``[residue]``. Unresolved atom
    slots must have NaN coordinates and empty metadata. The caller must provide
    a per-residue heavy-atom-completeness assertion. ``False`` means unknown or
    incomplete and cannot authorize a negative or exact distance label.
    """

    coordinates: np.ndarray
    atom_names: np.ndarray
    atom_elements: np.ndarray
    atom_resolved_mask: np.ndarray
    heavy_atom_complete_mask: np.ndarray
    chain_ids: np.ndarray
    residue_ids: np.ndarray


@dataclass(frozen=True)
class PairGeometryLabels:
    """Immutable prospective label artifact and its provenance digest."""

    minimum_heavy_atom_distance: np.ndarray
    minimum_heavy_atom_distance_mask: np.ndarray
    pair_resolved_mask: np.ndarray
    pair_negative_distance_valid_mask: np.ndarray
    primary_contact_labels: np.ndarray
    primary_contact_mask: np.ndarray
    sensitivity_cutoffs: np.ndarray
    sensitivity_contact_labels: np.ndarray
    sensitivity_contact_masks: np.ndarray
    distance_bin_edges: np.ndarray
    distance_bin_count: np.ndarray
    primary_contact_distance_bin_count: np.ndarray
    distance_bin_indices: np.ndarray
    distance_bin_mask: np.ndarray
    antibody_heavy_atom_complete_mask: np.ndarray
    antigen_heavy_atom_complete_mask: np.ndarray
    antibody_frame_valid_mask: np.ndarray
    antigen_frame_valid_mask: np.ndarray
    orientation_valid_mask: np.ndarray
    orientation_component_valid_mask: np.ndarray
    orientation_axis_names: np.ndarray
    orientation_axis_is_cyclic: np.ndarray
    orientation_bin_edge_convention: np.ndarray
    orientation_bin_edges: np.ndarray
    orientation_bin_indices: np.ndarray
    orientation_bin_masks: np.ndarray
    antibody_to_antigen_direction: np.ndarray
    antigen_to_antibody_direction: np.ndarray
    relative_frame_rotation: np.ndarray
    orientation_angles: np.ndarray
    orientation_circular_features: np.ndarray
    provenance: Mapping[str, str]
    input_hashes: Mapping[str, str]
    output_hashes: Mapping[str, str]
    artifact_sha256: str


@dataclass(frozen=True)
class _ValidatedAtomTable:
    """Normalized atom table used only inside the label builder."""

    coordinates: np.ndarray
    atom_names: np.ndarray
    atom_elements: np.ndarray
    atom_resolved_mask: np.ndarray
    heavy_atom_complete_mask: np.ndarray
    chain_ids: np.ndarray
    residue_ids: np.ndarray
    heavy_atom_mask: np.ndarray
    residue_heavy_atom_mask: np.ndarray
    frames: np.ndarray
    frame_origins: np.ndarray
    frame_valid_mask: np.ndarray


def _canonical_json_bytes(value: Any) -> bytes:
    """Serialize JSON-compatible values deterministically."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    """Return the lowercase SHA-256 digest of bytes."""
    return hashlib.sha256(value).hexdigest()


def _hash_array(array: np.ndarray) -> str:
    """Hash one array using canonical metadata and deterministic contents."""
    value = np.asarray(array)
    metadata = {
        "dtype": value.dtype.str,
        "shape": list(value.shape),
    }
    digest = hashlib.sha256(_canonical_json_bytes(metadata))
    if value.dtype.kind == "U":
        digest.update(_canonical_json_bytes(value.tolist()))
    else:
        normalized = np.ascontiguousarray(value)
        if normalized.dtype.kind == "f" and bool(np.isnan(normalized).any()):
            normalized = normalized.copy()
            normalized[np.isnan(normalized)] = np.nan
        digest.update(normalized.tobytes(order="C"))
    return digest.hexdigest()


def _immutable_array(value: Any, dtype: Any | None = None) -> np.ndarray:
    """Return an array backed by immutable bytes.

    A normal owned NumPy array can have its write flag reopened by calling
    ``setflags(write=True)``. Reconstructing the array over a ``bytes`` buffer
    makes that operation fail, which is the integrity boundary required for a
    hash-bound artifact.
    """
    normalized = np.array(value, dtype=dtype, copy=True)
    original_shape = normalized.shape
    contiguous = np.ascontiguousarray(normalized)
    immutable_buffer = contiguous.tobytes(order="C")
    return np.frombuffer(
        immutable_buffer,
        dtype=contiguous.dtype,
        count=contiguous.size,
    ).reshape(original_shape)


def _validate_unicode_array(
    value: Any,
    *,
    name: str,
    ndim: int,
) -> np.ndarray:
    """Require a fixed-width Unicode array with a precise rank."""
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if value.dtype.kind != "U":
        raise ValueError(f"{name} must use a fixed-width Unicode dtype")
    if value.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional")
    return np.array(value, copy=True)


def _validate_provenance(provenance: Mapping[str, str]) -> Mapping[str, str]:
    """Validate required source identity without accepting mutable values."""
    if not isinstance(provenance, Mapping):
        raise TypeError("provenance must be a mapping")
    normalized: dict[str, str] = {}
    for key, value in provenance.items():
        if not isinstance(key, str) or not key or key.strip() != key:
            raise ValueError("provenance keys must be nonempty trimmed strings")
        if not isinstance(value, str) or not value or value.strip() != value:
            raise ValueError(
                f"provenance field {key!r} must be a nonempty trimmed string"
            )
        normalized[key] = value
    missing = sorted(_REQUIRED_PROVENANCE_FIELDS - set(normalized))
    if missing:
        raise ValueError(f"provenance is missing required fields: {missing}")
    for hash_field in ("source_sha256", "label_builder_source_sha256"):
        if not _SHA256_PATTERN.fullmatch(normalized[hash_field]):
            raise ValueError(
                f"provenance {hash_field} must be 64 lowercase hex characters"
            )
    if (
        normalized["label_builder_source_sha256"]
        != LABEL_BUILDER_SOURCE_SHA256
    ):
        raise ValueError(
            "provenance label_builder_source_sha256 does not match the "
            "executing label builder"
        )
    if normalized["coordinate_unit"] != "angstrom":
        raise ValueError("provenance coordinate_unit must equal 'angstrom'")
    for key, expected_value in _RESERVED_PROVENANCE_VALUES.items():
        if key in normalized and normalized[key] != expected_value:
            raise ValueError(
                f"provenance contains conflicting reserved field {key!r}"
            )
        normalized[key] = expected_value
    return MappingProxyType(dict(sorted(normalized.items())))


def _validate_distance_bin_edges(edges: Sequence[float]) -> np.ndarray:
    """Require positive, finite, strictly increasing distance-bin edges."""
    if isinstance(edges, (str, bytes)) or not isinstance(edges, Sequence):
        raise TypeError("distance_bin_edges must be a sequence")
    try:
        result = np.asarray(tuple(edges), dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("distance_bin_edges must contain finite numeric values") from error
    if result.ndim != 1 or result.size == 0:
        raise ValueError("distance_bin_edges must be a nonempty one-dimensional sequence")
    if not bool(np.isfinite(result).all()) or bool((result <= 0.0).any()):
        raise ValueError("distance_bin_edges must contain positive finite values")
    if bool((np.diff(result) <= 0.0).any()):
        raise ValueError("distance_bin_edges must be strictly increasing")
    if not bool(np.any(result == PRIMARY_CONTACT_CUTOFF_ANGSTROM)):
        raise ValueError(
            "distance_bin_edges must contain the exact 4.5 Å primary cutoff"
        )
    return result


def _build_residue_frames(
    coordinates: np.ndarray,
    atom_names: np.ndarray,
    resolved_mask: np.ndarray,
    *,
    role: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct right-handed N-CA-C frames for resolvable residues.

    The origin is CA. The x axis points from CA to C. The y axis is the
    component of CA-to-N orthogonal to x, and z is ``x cross y``. This is a
    deterministic local backbone frame. Missing backbone atoms yield an
    invalid frame mask; degenerate resolved backbone coordinates are rejected.
    """
    residue_count = int(coordinates.shape[0])
    frames = np.full((residue_count, 3, 3), np.nan, dtype=np.float64)
    origins = np.full((residue_count, 3), np.nan, dtype=np.float64)
    valid = np.zeros(residue_count, dtype=np.bool_)
    for residue_index in range(residue_count):
        name_to_index = {
            str(atom_names[residue_index, atom_index]): atom_index
            for atom_index in np.flatnonzero(resolved_mask[residue_index])
        }
        if not {"N", "CA", "C"}.issubset(name_to_index):
            continue
        n_coordinate = coordinates[residue_index, name_to_index["N"]]
        ca_coordinate = coordinates[residue_index, name_to_index["CA"]]
        c_coordinate = coordinates[residue_index, name_to_index["C"]]
        x_raw = c_coordinate - ca_coordinate
        n_raw = n_coordinate - ca_coordinate
        x_norm = float(np.linalg.norm(x_raw))
        n_norm = float(np.linalg.norm(n_raw))
        if not (
            _MIN_BACKBONE_BOND_ANGSTROM
            <= x_norm
            <= _MAX_BACKBONE_BOND_ANGSTROM
        ):
            raise ValueError(
                f"{role} residue has implausible CA-to-C bond length"
            )
        if not (
            _MIN_BACKBONE_BOND_ANGSTROM
            <= n_norm
            <= _MAX_BACKBONE_BOND_ANGSTROM
        ):
            raise ValueError(
                f"{role} residue has implausible N-to-CA bond length"
            )
        x_axis = x_raw / x_norm
        y_raw = n_raw - float(np.dot(n_raw, x_axis)) * x_axis
        y_norm = float(np.linalg.norm(y_raw))
        sine_condition = y_norm / n_norm
        if (
            not math.isfinite(sine_condition)
            or sine_condition <= _MIN_FRAME_SINE_CONDITION
        ):
            raise ValueError(
                f"{role} residue has ill-conditioned near-collinear N-CA-C "
                "frame geometry"
            )
        y_axis = y_raw / y_norm
        z_axis = np.cross(x_axis, y_axis)
        z_norm = float(np.linalg.norm(z_axis))
        if not math.isfinite(z_norm) or z_norm <= _MIN_FRAME_SINE_CONDITION:
            raise ValueError(f"{role} residue has degenerate backbone frame geometry")
        z_axis /= z_norm
        frame = np.stack((x_axis, y_axis, z_axis), axis=1)
        if not np.allclose(frame.T @ frame, np.eye(3), atol=1e-8, rtol=0.0):
            raise ValueError(f"{role} residue backbone frame is not orthonormal")
        if float(np.linalg.det(frame)) <= 0.0:
            raise ValueError(f"{role} residue backbone frame is not right handed")
        if not bool(np.isfinite(frame).all()):
            raise ValueError(f"{role} residue backbone frame contains nonfinite values")
        frames[residue_index] = frame
        origins[residue_index] = ca_coordinate
        valid[residue_index] = True
    return frames, origins, valid


def _validate_atom_table(value: ResidueAtomInput, *, role: str) -> _ValidatedAtomTable:
    """Validate and normalize one explicit atom table."""
    if not isinstance(value, ResidueAtomInput):
        raise TypeError(f"{role} must be a ResidueAtomInput")
    coordinates = value.coordinates
    if not isinstance(coordinates, np.ndarray):
        raise TypeError(f"{role}.coordinates must be a NumPy array")
    if coordinates.dtype.kind != "f":
        raise ValueError(f"{role}.coordinates must use a floating-point dtype")
    if coordinates.ndim != 3 or coordinates.shape[-1] != 3:
        raise ValueError(
            f"{role}.coordinates must have shape [residue, atom_slot, 3]"
        )
    if int(coordinates.shape[0]) == 0 or int(coordinates.shape[1]) == 0:
        raise ValueError(f"{role}.coordinates cannot contain an empty axis")
    coordinates = np.array(coordinates, dtype=np.float64, copy=True)

    atom_names = _validate_unicode_array(
        value.atom_names,
        name=f"{role}.atom_names",
        ndim=2,
    )
    atom_elements = _validate_unicode_array(
        value.atom_elements,
        name=f"{role}.atom_elements",
        ndim=2,
    )
    expected_atom_shape = coordinates.shape[:2]
    if atom_names.shape != expected_atom_shape:
        raise ValueError(f"{role}.atom_names must align with coordinates")
    if atom_elements.shape != expected_atom_shape:
        raise ValueError(f"{role}.atom_elements must align with coordinates")

    resolved_mask = value.atom_resolved_mask
    if not isinstance(resolved_mask, np.ndarray):
        raise TypeError(f"{role}.atom_resolved_mask must be a NumPy array")
    if resolved_mask.dtype != np.bool_ or resolved_mask.ndim != 2:
        raise ValueError(
            f"{role}.atom_resolved_mask must be a two-dimensional boolean array"
        )
    if resolved_mask.shape != expected_atom_shape:
        raise ValueError(f"{role}.atom_resolved_mask must align with coordinates")
    resolved_mask = np.array(resolved_mask, copy=True)

    heavy_atom_complete_mask = value.heavy_atom_complete_mask
    if not isinstance(heavy_atom_complete_mask, np.ndarray):
        raise TypeError(f"{role}.heavy_atom_complete_mask must be a NumPy array")
    residue_count = int(coordinates.shape[0])
    if (
        heavy_atom_complete_mask.dtype != np.bool_
        or heavy_atom_complete_mask.shape != (residue_count,)
    ):
        raise ValueError(
            f"{role}.heavy_atom_complete_mask must be a one-dimensional "
            "boolean array aligned with residues"
        )
    heavy_atom_complete_mask = np.array(heavy_atom_complete_mask, copy=True)

    chain_ids = _validate_unicode_array(
        value.chain_ids,
        name=f"{role}.chain_ids",
        ndim=1,
    )
    residue_ids = _validate_unicode_array(
        value.residue_ids,
        name=f"{role}.residue_ids",
        ndim=1,
    )
    if chain_ids.shape != (residue_count,) or residue_ids.shape != (residue_count,):
        raise ValueError(f"{role} chain and residue identifiers must align with residues")
    if any(not item or item.strip() != item for item in chain_ids.tolist()):
        raise ValueError(f"{role}.chain_ids must contain nonempty trimmed identifiers")
    if any(not item or item.strip() != item for item in residue_ids.tolist()):
        raise ValueError(f"{role}.residue_ids must contain nonempty trimmed identifiers")
    residue_keys = list(zip(chain_ids.tolist(), residue_ids.tolist(), strict=True))
    if len(set(residue_keys)) != len(residue_keys):
        raise ValueError(f"{role} contains duplicate chain/residue identifiers")

    if not bool(np.isfinite(coordinates[resolved_mask]).all()):
        raise ValueError(f"{role} resolved atom coordinates must be finite")
    if (
        coordinates[resolved_mask].size
        and float(np.max(np.abs(coordinates[resolved_mask])))
        > _MAX_ABSOLUTE_COORDINATE_ANGSTROM
    ):
        raise ValueError(
            f"{role} resolved atom coordinates exceed the supported physical range"
        )
    unresolved_coordinates = coordinates[~resolved_mask]
    if unresolved_coordinates.size and not bool(np.isnan(unresolved_coordinates).all()):
        raise ValueError(
            f"{role} unresolved atom coordinates must be NaN in every component"
        )
    if bool((atom_names[resolved_mask] == "").any()):
        raise ValueError(f"{role} resolved atoms must have nonempty names")
    if bool((atom_elements[resolved_mask] == "").any()):
        raise ValueError(f"{role} resolved atoms must have nonempty elements")
    if bool((atom_names[~resolved_mask] != "").any()):
        raise ValueError(f"{role} unresolved atoms must have empty names")
    if bool((atom_elements[~resolved_mask] != "").any()):
        raise ValueError(f"{role} unresolved atoms must have empty elements")

    for atom_name in atom_names[resolved_mask].tolist():
        if atom_name.strip() != atom_name or atom_name.upper() != atom_name:
            raise ValueError(f"{role} atom names must be trimmed uppercase identifiers")
    for element in atom_elements[resolved_mask].tolist():
        if (
            not _ELEMENT_PATTERN.fullmatch(element)
            or element not in _KNOWN_ELEMENTS
        ):
            raise ValueError(f"{role} contains an invalid atom element {element!r}")

    for residue_index in range(residue_count):
        names = atom_names[residue_index, resolved_mask[residue_index]].tolist()
        if len(names) != len(set(names)):
            raise ValueError(f"{role} contains duplicate atom names within a residue")
        metadata = {
            name: atom_elements[residue_index, atom_index]
            for atom_index, name in enumerate(atom_names[residue_index])
            if resolved_mask[residue_index, atom_index]
        }
        for atom_name, expected_element in (("N", "N"), ("CA", "C"), ("C", "C")):
            if atom_name in metadata and metadata[atom_name] != expected_element:
                raise ValueError(
                    f"{role} backbone atom {atom_name} has inconsistent element metadata"
                )

    heavy_atom_mask = resolved_mask & ~np.isin(
        atom_elements,
        tuple(_HYDROGEN_ELEMENTS),
    )
    residue_heavy_atom_mask = heavy_atom_mask.any(axis=1)
    if not bool(residue_heavy_atom_mask.any()):
        raise ValueError(f"{role} has no resolved non-hydrogen residue atoms")
    if bool((heavy_atom_complete_mask & ~residue_heavy_atom_mask).any()):
        raise ValueError(
            f"{role} cannot mark a residue heavy-atom complete without any "
            "resolved non-hydrogen atom"
        )

    frames, frame_origins, frame_valid_mask = _build_residue_frames(
        coordinates,
        atom_names,
        resolved_mask,
        role=role,
    )
    if not bool(frame_valid_mask.any()):
        raise ValueError(f"{role} has no valid resolved N-CA-C residue frame")
    return _ValidatedAtomTable(
        coordinates=coordinates,
        atom_names=atom_names,
        atom_elements=atom_elements,
        atom_resolved_mask=resolved_mask,
        heavy_atom_complete_mask=heavy_atom_complete_mask,
        chain_ids=chain_ids,
        residue_ids=residue_ids,
        heavy_atom_mask=heavy_atom_mask,
        residue_heavy_atom_mask=residue_heavy_atom_mask,
        frames=frames,
        frame_origins=frame_origins,
        frame_valid_mask=frame_valid_mask,
    )


def _minimum_heavy_atom_distances(
    antibody: _ValidatedAtomTable,
    antigen: _ValidatedAtomTable,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute observed minima and separate exact-negative validity.

    An observed heavy-atom contact is sufficient to establish a positive
    label. A non-contact or exact minimum-distance label is valid only when
    both residues carry explicit heavy-atom-completeness evidence.
    """
    antibody_count = int(antibody.coordinates.shape[0])
    antigen_count = int(antigen.coordinates.shape[0])
    distances = np.full((antibody_count, antigen_count), np.nan, dtype=np.float64)
    observed_pair_mask = (
        antibody.residue_heavy_atom_mask[:, None]
        & antigen.residue_heavy_atom_mask[None, :]
    )
    complete_pair_mask = (
        antibody.heavy_atom_complete_mask[:, None]
        & antigen.heavy_atom_complete_mask[None, :]
    )
    if bool((complete_pair_mask & ~observed_pair_mask).any()):
        raise RuntimeError("Internal complete-pair validation invariant failed")
    for antibody_index, antigen_index in np.argwhere(observed_pair_mask):
        antibody_coordinates = antibody.coordinates[
            antibody_index,
            antibody.heavy_atom_mask[antibody_index],
        ]
        antigen_coordinates = antigen.coordinates[
            antigen_index,
            antigen.heavy_atom_mask[antigen_index],
        ]
        differences = (
            antibody_coordinates[:, None, :] - antigen_coordinates[None, :, :]
        )
        try:
            with np.errstate(over="raise", invalid="raise"):
                squared_distances = np.einsum(
                    "ijk,ijk->ij",
                    differences,
                    differences,
                    optimize=True,
                )
        except FloatingPointError as error:
            raise ValueError(
                "Computed heavy-atom distances overflowed or became nonfinite"
            ) from error
        minimum_squared_distance = float(squared_distances.min())
        if not math.isfinite(minimum_squared_distance) or minimum_squared_distance < 0.0:
            raise ValueError("Computed heavy-atom distance is nonfinite or negative")
        minimum_distance = math.sqrt(minimum_squared_distance)
        if not math.isfinite(minimum_distance):
            raise ValueError("Computed heavy-atom distance is nonfinite")
        distances[antibody_index, antigen_index] = minimum_distance
    if not bool(observed_pair_mask.any()):
        raise ValueError("No resolved antibody-antigen heavy-atom pair exists")
    if not bool(np.isfinite(distances[observed_pair_mask]).all()):
        raise ValueError("Computed heavy-atom distance labels contain nonfinite values")
    return distances, observed_pair_mask, complete_pair_mask


def _canonical_cyclic_angle(angle: float) -> float:
    """Map an angle to the frozen half-open interval ``[-pi, pi)``."""
    result = (angle + math.pi) % (2.0 * math.pi) - math.pi
    if not math.isfinite(result):
        raise ValueError("Computed cyclic orientation angle is nonfinite")
    return result


def _orientation_schema_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return frozen axis names, cyclic flags, and internal thresholds.

    Each row contains ``ORIENTATION_BIN_COUNT - 1`` internal thresholds.
    ``searchsorted(..., side="left")`` therefore implements right-closed bins:
    bin zero ends at the first threshold and the last bin begins above the
    final threshold.
    """
    axis_names = np.asarray(ORIENTATION_AXIS_NAMES, dtype="<U40")
    cyclic_flags = np.asarray(ORIENTATION_AXIS_IS_CYCLIC, dtype=np.bool_)
    edges = np.empty(
        (len(ORIENTATION_AXIS_NAMES), ORIENTATION_BIN_COUNT - 1),
        dtype=np.float64,
    )
    for axis_index, is_cyclic in enumerate(ORIENTATION_AXIS_IS_CYCLIC):
        if is_cyclic:
            full_boundaries = np.linspace(
                -math.pi,
                math.pi,
                ORIENTATION_BIN_COUNT + 1,
                dtype=np.float64,
            )
        else:
            full_boundaries = np.linspace(
                0.0,
                math.pi,
                ORIENTATION_BIN_COUNT + 1,
                dtype=np.float64,
            )
        edges[axis_index] = full_boundaries[1:-1]
    return axis_names, cyclic_flags, edges


def _orientation_bin_index(
    angle: float,
    *,
    is_cyclic: bool,
    edges: np.ndarray,
) -> int:
    """Return a categorical bin index under the frozen orientation schema."""
    normalized_angle = (
        _canonical_cyclic_angle(angle) if is_cyclic else angle
    )
    if not math.isfinite(normalized_angle):
        raise ValueError("Computed orientation angle is nonfinite")
    if not is_cyclic and not 0.0 <= normalized_angle <= math.pi:
        raise ValueError("Computed polar orientation angle is outside [0, pi]")
    index = int(np.searchsorted(edges, normalized_angle, side="left"))
    if not 0 <= index < ORIENTATION_BIN_COUNT:
        raise RuntimeError("Internal orientation bin construction failed")
    return index


def _orientation_targets(
    antibody: _ValidatedAtomTable,
    antigen: _ValidatedAtomTable,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Build a masked, periodic-bin-ready five-DoF relative pose.

    The axes are antibody azimuth/polar, antigen azimuth/polar, and relative
    twist. Twist is the signed angle, about the antibody-to-antigen direction,
    between the two projected local x axes. Azimuth or twist components are
    masked when their defining projection is ill-conditioned. Every valid
    component has both a categorical bin and a sine/cosine representation.
    """
    shape = (
        int(antibody.coordinates.shape[0]),
        int(antigen.coordinates.shape[0]),
    )
    valid = (
        antibody.frame_valid_mask[:, None] & antigen.frame_valid_mask[None, :]
    )
    antibody_direction = np.full((*shape, 3), np.nan, dtype=np.float64)
    antigen_direction = np.full((*shape, 3), np.nan, dtype=np.float64)
    relative_rotation = np.full((*shape, 3, 3), np.nan, dtype=np.float64)
    angles = np.full((*shape, 5), np.nan, dtype=np.float64)
    component_valid = np.zeros((*shape, 5), dtype=np.bool_)
    circular_features = np.full((*shape, 5, 2), np.nan, dtype=np.float64)
    bin_indices = np.full((*shape, 5), -1, dtype=np.int64)
    _, cyclic_flags, bin_edges = _orientation_schema_arrays()
    for antibody_index, antigen_index in np.argwhere(valid):
        displacement = (
            antigen.frame_origins[antigen_index]
            - antibody.frame_origins[antibody_index]
        )
        displacement_norm = float(np.linalg.norm(displacement))
        if not math.isfinite(displacement_norm):
            raise ValueError("Computed CA-to-CA displacement is nonfinite")
        if displacement_norm <= _MIN_DIRECTION_PROJECTION_CONDITION:
            valid[antibody_index, antigen_index] = False
            continue
        unit_displacement = displacement / displacement_norm
        antibody_local = antibody.frames[antibody_index].T @ unit_displacement
        antigen_local = antigen.frames[antigen_index].T @ (-unit_displacement)
        if not (
            bool(np.isfinite(antibody_local).all())
            and bool(np.isfinite(antigen_local).all())
        ):
            raise ValueError("Computed local orientation direction is nonfinite")
        antibody_direction[antibody_index, antigen_index] = antibody_local
        antigen_direction[antibody_index, antigen_index] = antigen_local
        rotation = (
            antibody.frames[antibody_index].T @ antigen.frames[antigen_index]
        )
        if not bool(np.isfinite(rotation).all()):
            raise ValueError("Computed relative frame rotation is nonfinite")
        relative_rotation[antibody_index, antigen_index] = rotation

        pair_angles = np.full(5, np.nan, dtype=np.float64)
        pair_component_valid = np.zeros(5, dtype=np.bool_)
        antibody_azimuth_condition = float(np.hypot(*antibody_local[:2]))
        antigen_azimuth_condition = float(np.hypot(*antigen_local[:2]))
        if antibody_azimuth_condition > _MIN_DIRECTION_PROJECTION_CONDITION:
            pair_angles[0] = _canonical_cyclic_angle(
                math.atan2(float(antibody_local[1]), float(antibody_local[0]))
            )
            pair_component_valid[0] = True
        pair_angles[1] = math.acos(
            float(np.clip(antibody_local[2], -1.0, 1.0))
        )
        pair_component_valid[1] = True
        if antigen_azimuth_condition > _MIN_DIRECTION_PROJECTION_CONDITION:
            pair_angles[2] = _canonical_cyclic_angle(
                math.atan2(float(antigen_local[1]), float(antigen_local[0]))
            )
            pair_component_valid[2] = True
        pair_angles[3] = math.acos(
            float(np.clip(antigen_local[2], -1.0, 1.0))
        )
        pair_component_valid[3] = True

        antibody_x = antibody.frames[antibody_index, :, 0]
        antigen_x = antigen.frames[antigen_index, :, 0]
        antibody_x_projection = (
            antibody_x
            - float(np.dot(antibody_x, unit_displacement)) * unit_displacement
        )
        antigen_x_projection = (
            antigen_x
            - float(np.dot(antigen_x, unit_displacement)) * unit_displacement
        )
        antibody_projection_norm = float(np.linalg.norm(antibody_x_projection))
        antigen_projection_norm = float(np.linalg.norm(antigen_x_projection))
        if (
            antibody_projection_norm > _MIN_DIRECTION_PROJECTION_CONDITION
            and antigen_projection_norm > _MIN_DIRECTION_PROJECTION_CONDITION
        ):
            antibody_projected_unit = (
                antibody_x_projection / antibody_projection_norm
            )
            antigen_projected_unit = antigen_x_projection / antigen_projection_norm
            twist_sine = float(
                np.dot(
                    unit_displacement,
                    np.cross(antibody_projected_unit, antigen_projected_unit),
                )
            )
            twist_cosine = float(
                np.dot(antibody_projected_unit, antigen_projected_unit)
            )
            pair_angles[4] = _canonical_cyclic_angle(
                math.atan2(twist_sine, twist_cosine)
            )
            pair_component_valid[4] = True

        if not bool(np.isfinite(pair_angles[pair_component_valid]).all()):
            raise ValueError("Computed orientation components contain nonfinite values")
        angles[antibody_index, antigen_index] = pair_angles
        component_valid[antibody_index, antigen_index] = pair_component_valid
        for axis_index in np.flatnonzero(pair_component_valid):
            angle = float(pair_angles[axis_index])
            circular_features[
                antibody_index,
                antigen_index,
                axis_index,
            ] = (math.sin(angle), math.cos(angle))
            bin_indices[
                antibody_index,
                antigen_index,
                axis_index,
            ] = _orientation_bin_index(
                angle,
                is_cyclic=bool(cyclic_flags[axis_index]),
                edges=bin_edges[axis_index],
            )
    if not bool(valid.any()):
        raise ValueError("No antibody-antigen residue pair has a valid orientation")
    if bool((component_valid & ~valid[..., None]).any()):
        raise RuntimeError("Internal orientation component-mask invariant failed")
    return (
        valid,
        component_valid,
        antibody_direction,
        antigen_direction,
        relative_rotation,
        angles,
        circular_features,
        bin_indices,
    )


def _table_input_hashes(
    role: str,
    table: _ValidatedAtomTable,
) -> dict[str, str]:
    """Return hashes of every caller-controlled atom-table field."""
    return {
        f"{role}_coordinates_sha256": _hash_array(table.coordinates),
        f"{role}_atom_names_sha256": _hash_array(table.atom_names),
        f"{role}_atom_elements_sha256": _hash_array(table.atom_elements),
        f"{role}_atom_resolved_mask_sha256": _hash_array(
            table.atom_resolved_mask
        ),
        f"{role}_heavy_atom_complete_mask_sha256": _hash_array(
            table.heavy_atom_complete_mask
        ),
        f"{role}_chain_ids_sha256": _hash_array(table.chain_ids),
        f"{role}_residue_ids_sha256": _hash_array(table.residue_ids),
    }


def build_pair_geometry_labels(
    antibody: ResidueAtomInput,
    antigen: ResidueAtomInput,
    *,
    provenance: Mapping[str, str],
    distance_bin_edges: Sequence[float] = DEFAULT_DISTANCE_BIN_EDGES_ANGSTROM,
) -> PairGeometryLabels:
    """Build exact contacts, distance bins, and residue-frame orientations.

    Contacts use the minimum Euclidean distance between resolved non-hydrogen
    atoms in each residue pair. An observed distance within a cutoff proves a
    positive even for a partially observed residue. A negative or exact
    distance/bin label is valid only if both residues have explicit heavy-atom
    completeness evidence. The primary cutoff is inclusive at ``4.5 Å``;
    sensitivity cutoffs are inclusive at ``5 Å``, ``6 Å``, and ``8 Å``.

    Distance bins are right-closed: bin zero is ``[0, edge_0]``, intermediate
    bins are ``(edge_i, edge_(i+1)]``, and the final bin is above the last edge.
    """
    normalized_provenance = _validate_provenance(provenance)
    bin_edges = _validate_distance_bin_edges(distance_bin_edges)
    antibody_table = _validate_atom_table(antibody, role="antibody")
    antigen_table = _validate_atom_table(antigen, role="antigen")
    overlapping_chains = sorted(
        set(antibody_table.chain_ids.tolist())
        & set(antigen_table.chain_ids.tolist())
    )
    if overlapping_chains:
        raise ValueError(
            "Antibody and antigen chain identifiers overlap, making role mapping "
            f"ambiguous: {overlapping_chains}"
        )

    distances, observed_pair_mask, complete_pair_mask = _minimum_heavy_atom_distances(
        antibody_table,
        antigen_table,
    )
    primary_labels = np.zeros(observed_pair_mask.shape, dtype=np.bool_)
    primary_labels[observed_pair_mask] = (
        distances[observed_pair_mask] <= PRIMARY_CONTACT_CUTOFF_ANGSTROM
    )
    primary_mask = primary_labels | complete_pair_mask
    sensitivity_cutoffs = np.asarray(
        SENSITIVITY_CONTACT_CUTOFFS_ANGSTROM,
        dtype=np.float64,
    )
    sensitivity_labels = np.zeros(
        (len(sensitivity_cutoffs), *observed_pair_mask.shape),
        dtype=np.bool_,
    )
    sensitivity_masks = np.zeros_like(sensitivity_labels)
    for cutoff_index, cutoff in enumerate(sensitivity_cutoffs):
        sensitivity_labels[cutoff_index, observed_pair_mask] = (
            distances[observed_pair_mask] <= cutoff
        )
        sensitivity_masks[cutoff_index] = (
            sensitivity_labels[cutoff_index] | complete_pair_mask
        )

    distance_bin_indices = np.full(observed_pair_mask.shape, -1, dtype=np.int64)
    distance_bin_indices[complete_pair_mask] = np.searchsorted(
        bin_edges,
        distances[complete_pair_mask],
        side="left",
    )
    distance_bin_count = np.asarray(bin_edges.size + 1, dtype=np.int64)
    primary_contact_distance_bin_count = np.asarray(
        int(
            np.flatnonzero(
                bin_edges == PRIMARY_CONTACT_CUTOFF_ANGSTROM
            )[0]
        )
        + 1,
        dtype=np.int64,
    )
    (
        orientation_axis_names,
        orientation_axis_is_cyclic,
        orientation_bin_edges,
    ) = _orientation_schema_arrays()
    (
        orientation_mask,
        orientation_component_mask,
        antibody_direction,
        antigen_direction,
        relative_rotation,
        orientation_angles,
        orientation_circular_features,
        orientation_bin_indices,
    ) = _orientation_targets(antibody_table, antigen_table)
    orientation_bin_masks = orientation_component_mask.copy()

    input_hashes = {
        **_table_input_hashes("antibody", antibody_table),
        **_table_input_hashes("antigen", antigen_table),
        "distance_bin_edges_sha256": _hash_array(bin_edges),
        "provenance_sha256": _sha256_bytes(
            _canonical_json_bytes(dict(normalized_provenance))
        ),
    }
    output_arrays = {
        "minimum_heavy_atom_distance": distances,
        "minimum_heavy_atom_distance_mask": complete_pair_mask,
        "pair_resolved_mask": observed_pair_mask,
        "pair_negative_distance_valid_mask": complete_pair_mask,
        "primary_contact_labels": primary_labels,
        "primary_contact_mask": primary_mask,
        "sensitivity_cutoffs": sensitivity_cutoffs,
        "sensitivity_contact_labels": sensitivity_labels,
        "sensitivity_contact_masks": sensitivity_masks,
        "distance_bin_edges": bin_edges,
        "distance_bin_count": distance_bin_count,
        "primary_contact_distance_bin_count": (
            primary_contact_distance_bin_count
        ),
        "distance_bin_indices": distance_bin_indices,
        "distance_bin_mask": complete_pair_mask,
        "antibody_heavy_atom_complete_mask": (
            antibody_table.heavy_atom_complete_mask
        ),
        "antigen_heavy_atom_complete_mask": antigen_table.heavy_atom_complete_mask,
        "antibody_frame_valid_mask": antibody_table.frame_valid_mask,
        "antigen_frame_valid_mask": antigen_table.frame_valid_mask,
        "orientation_valid_mask": orientation_mask,
        "orientation_component_valid_mask": orientation_component_mask,
        "orientation_axis_names": orientation_axis_names,
        "orientation_axis_is_cyclic": orientation_axis_is_cyclic,
        "orientation_bin_edge_convention": np.asarray(
            ORIENTATION_BIN_EDGE_CONVENTION,
            dtype="<U64",
        ),
        "orientation_bin_edges": orientation_bin_edges,
        "orientation_bin_indices": orientation_bin_indices,
        "orientation_bin_masks": orientation_bin_masks,
        "antibody_to_antigen_direction": antibody_direction,
        "antigen_to_antibody_direction": antigen_direction,
        "relative_frame_rotation": relative_rotation,
        "orientation_angles": orientation_angles,
        "orientation_circular_features": orientation_circular_features,
    }
    output_hashes = {
        f"{name}_sha256": _hash_array(value)
        for name, value in sorted(output_arrays.items())
    }
    artifact_sha256 = _sha256_bytes(
        _canonical_json_bytes(
            {
                "format_version": FORMAT_VERSION,
                "input_hashes": dict(sorted(input_hashes.items())),
                "output_hashes": dict(sorted(output_hashes.items())),
                "provenance": dict(normalized_provenance),
            }
        )
    )
    if not _SHA256_PATTERN.fullmatch(artifact_sha256):
        raise RuntimeError("Internal artifact hash construction failed")

    immutable_outputs = {
        name: _immutable_array(value)
        for name, value in output_arrays.items()
    }
    return PairGeometryLabels(
        **immutable_outputs,
        provenance=normalized_provenance,
        input_hashes=MappingProxyType(dict(sorted(input_hashes.items()))),
        output_hashes=MappingProxyType(dict(sorted(output_hashes.items()))),
        artifact_sha256=artifact_sha256,
    )
