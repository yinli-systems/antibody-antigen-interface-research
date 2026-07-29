"""Immutable schema contract between pair-geometry labels, models, and losses.

This module does not choose scientific binning. Callers must provide every
distance and orientation edge explicitly. The adapter accepts a geometry-label
artifact only when its declared semantics exactly match this contract.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from collections.abc import Iterator
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA_FORMAT_VERSION = "pair_geometry_schema_v1"
PAIR_ORIENTATION = "antibody_by_antigen"
DISTANCE_UNIT = "angstrom"
ANGLE_UNIT = "radian"
DISTANCE_CLOSURE = "right_closed"


@dataclass(frozen=True)
class FrozenGeometryMetadata(Mapping[str, Any]):
    """A hashable, immutable mapping for canonical schema metadata."""

    entries: tuple[tuple[str, Any], ...]

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> "FrozenGeometryMetadata":
        """Freeze a mapping whose values are already immutable and hashable."""
        entries = tuple(sorted(value.items()))
        try:
            hash(entries)
        except TypeError as error:
            raise TypeError("Geometry metadata values must be hashable") from error
        return cls(entries=entries)

    def __getitem__(self, key: str) -> Any:
        for candidate, value in self.entries:
            if candidate == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (key for key, _ in self.entries)

    def __len__(self) -> int:
        return len(self.entries)


def _finite_positive(value: Any, name: str) -> float:
    """Return a positive finite float or reject it."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive finite scalar")
    try:
        converted = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a positive finite scalar") from error
    if not math.isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{name} must be a positive finite scalar")
    return converted


def _strict_edges(values: Sequence[float], name: str) -> tuple[float, ...]:
    """Normalize a nonempty, finite, strictly increasing edge sequence."""
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence")
    try:
        edges = tuple(float(value) for value in values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain finite numeric values") from error
    if not edges:
        raise ValueError(f"{name} must be nonempty")
    if not all(math.isfinite(value) for value in edges):
        raise ValueError(f"{name} must contain finite numeric values")
    if any(right <= left for left, right in zip(edges, edges[1:])):
        raise ValueError(f"{name} must be strictly increasing")
    return edges


def _trimmed_names(values: Sequence[str], name: str) -> tuple[str, ...]:
    """Normalize unique, nonempty, trimmed schema names."""
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence")
    names = tuple(values)
    if not names:
        raise ValueError(f"{name} must be nonempty")
    if any(
        not isinstance(value, str) or not value or value.strip() != value
        for value in names
    ):
        raise ValueError(f"{name} must contain nonempty trimmed strings")
    if len(set(names)) != len(names):
        raise ValueError(f"{name} must contain unique values")
    return names


@dataclass(frozen=True)
class PairGeometrySchema:
    """Hashable geometry semantics required by both predictions and targets."""

    distance_bin_edges_angstrom: tuple[float, ...]
    contact_cutoff_angstrom: float
    orientation_axis_names: tuple[str, ...]
    orientation_axis_is_cyclic: tuple[bool, ...]
    orientation_bin_edges_radians: tuple[tuple[float, ...], ...]
    label_format_version: str
    orientation_schema_version: str
    primary_contact_definition: str
    distance_definition: str
    orientation_definition: str
    orientation_bin_edge_convention: str
    schema_format_version: str = SCHEMA_FORMAT_VERSION
    pair_orientation: str = PAIR_ORIENTATION
    distance_unit: str = DISTANCE_UNIT
    angle_unit: str = ANGLE_UNIT
    distance_closure: str = DISTANCE_CLOSURE

    def __post_init__(self) -> None:
        """Normalize immutable fields and validate all semantic invariants."""
        distance_edges = _strict_edges(
            self.distance_bin_edges_angstrom,
            "distance_bin_edges_angstrom",
        )
        if any(value <= 0.0 for value in distance_edges):
            raise ValueError("distance_bin_edges_angstrom must contain positive values")
        contact_cutoff = _finite_positive(
            self.contact_cutoff_angstrom,
            "contact_cutoff_angstrom",
        )
        if contact_cutoff not in distance_edges:
            raise ValueError(
                "contact_cutoff_angstrom must exactly equal one distance-bin edge"
            )
        axis_names = _trimmed_names(
            self.orientation_axis_names,
            "orientation_axis_names",
        )
        if (
            isinstance(self.orientation_bin_edges_radians, (str, bytes))
            or not isinstance(self.orientation_bin_edges_radians, Sequence)
        ):
            raise TypeError("orientation_bin_edges_radians must be a sequence")
        orientation_edges = tuple(
            _strict_edges(values, f"orientation_bin_edges_radians[{index}]")
            for index, values in enumerate(self.orientation_bin_edges_radians)
        )
        if len(orientation_edges) != len(axis_names):
            raise ValueError(
                "orientation_bin_edges_radians must provide one edge sequence "
                "per orientation axis"
            )
        if any(len(values) < 1 for values in orientation_edges):
            raise ValueError(
                "Each orientation edge sequence must include at least two bins"
            )
        cyclic_flags = tuple(self.orientation_axis_is_cyclic)
        if len(cyclic_flags) != len(axis_names) or any(
            not isinstance(value, bool) for value in cyclic_flags
        ):
            raise ValueError(
                "orientation_axis_is_cyclic must provide one boolean per axis"
            )
        bin_counts = {len(values) + 1 for values in orientation_edges}
        if len(bin_counts) != 1:
            raise ValueError(
                "Every orientation axis must use the same number of bins"
            )
        for value, name in (
            (self.label_format_version, "label_format_version"),
            (self.orientation_schema_version, "orientation_schema_version"),
            (self.primary_contact_definition, "primary_contact_definition"),
            (self.distance_definition, "distance_definition"),
            (self.orientation_definition, "orientation_definition"),
            (
                self.orientation_bin_edge_convention,
                "orientation_bin_edge_convention",
            ),
        ):
            if not isinstance(value, str) or not value or value.strip() != value:
                raise ValueError(f"{name} must be a nonempty trimmed string")
        expected_constants = {
            "schema_format_version": SCHEMA_FORMAT_VERSION,
            "pair_orientation": PAIR_ORIENTATION,
            "distance_unit": DISTANCE_UNIT,
            "angle_unit": ANGLE_UNIT,
            "distance_closure": DISTANCE_CLOSURE,
        }
        for field_name, expected in expected_constants.items():
            if getattr(self, field_name) != expected:
                raise ValueError(f"{field_name} must equal {expected!r}")

        object.__setattr__(self, "distance_bin_edges_angstrom", distance_edges)
        object.__setattr__(self, "contact_cutoff_angstrom", contact_cutoff)
        object.__setattr__(self, "orientation_axis_names", axis_names)
        object.__setattr__(self, "orientation_axis_is_cyclic", cyclic_flags)
        object.__setattr__(self, "orientation_bin_edges_radians", orientation_edges)

    @property
    def num_distance_bins(self) -> int:
        """Return the distance class count implied by the right-closed edges."""
        return len(self.distance_bin_edges_angstrom) + 1

    @property
    def contact_distance_bin_count(self) -> int:
        """Return the exact prefix of bins whose distances are contacts."""
        return self.distance_bin_edges_angstrom.index(
            self.contact_cutoff_angstrom
        ) + 1

    @property
    def num_orientation_axes(self) -> int:
        """Return the number of explicitly named orientation axes."""
        return len(self.orientation_axis_names)

    @property
    def num_orientation_bins(self) -> int:
        """Return the shared categorical class count for orientation axes."""
        return len(self.orientation_bin_edges_radians[0]) + 1

    def canonical_metadata(self) -> FrozenGeometryMetadata:
        """Return a hashable, immutable, JSON-compatible representation."""
        metadata = {
            "schema_format_version": self.schema_format_version,
            "pair_orientation": self.pair_orientation,
            "distance_unit": self.distance_unit,
            "distance_closure": self.distance_closure,
            "distance_bin_edges_angstrom": self.distance_bin_edges_angstrom,
            "contact_cutoff_angstrom": self.contact_cutoff_angstrom,
            "contact_distance_bin_count": self.contact_distance_bin_count,
            "num_distance_bins": self.num_distance_bins,
            "angle_unit": self.angle_unit,
            "orientation_axis_names": self.orientation_axis_names,
            "orientation_axis_is_cyclic": self.orientation_axis_is_cyclic,
            "orientation_bin_edges_radians": self.orientation_bin_edges_radians,
            "num_orientation_axes": self.num_orientation_axes,
            "num_orientation_bins": self.num_orientation_bins,
            "label_format_version": self.label_format_version,
            "orientation_schema_version": self.orientation_schema_version,
            "primary_contact_definition": self.primary_contact_definition,
            "distance_definition": self.distance_definition,
            "orientation_definition": self.orientation_definition,
            "orientation_bin_edge_convention": (
                self.orientation_bin_edge_convention
            ),
        }
        return FrozenGeometryMetadata.from_mapping(metadata)

    @property
    def sha256(self) -> str:
        """Return the canonical schema digest."""
        payload = json.dumps(
            dict(self.canonical_metadata()),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class PairGeometryTargetArrays:
    """Read-only model targets adapted from one real geometry-label artifact."""

    pair_orientation: str
    geometry_schema: PairGeometrySchema
    contact_targets: np.ndarray
    contact_mask: np.ndarray
    distance_bin_targets: np.ndarray
    distance_mask: np.ndarray
    orientation_bin_targets: np.ndarray
    orientation_mask: np.ndarray


def schema_from_geometry_labels(
    labels: Any,
) -> PairGeometrySchema:
    """Read the complete frozen distance/orientation schema from an artifact."""
    from .geometry_labels import PairGeometryLabels, _hash_array

    if not isinstance(labels, PairGeometryLabels):
        raise TypeError("labels must be a PairGeometryLabels artifact")
    required = {
        "distance_bin_edges",
        "distance_bin_count",
        "primary_contact_distance_bin_count",
        "orientation_axis_names",
        "orientation_axis_is_cyclic",
        "orientation_bin_edge_convention",
        "orientation_bin_edges",
        "provenance",
    }
    missing = sorted(name for name in required if not hasattr(labels, name))
    if missing:
        raise TypeError(f"Geometry-label artifact is missing fields: {missing}")
    if not isinstance(labels.output_hashes, Mapping):
        raise TypeError("Geometry-label output_hashes must be a mapping")
    schema_array_names = (
        "distance_bin_edges",
        "distance_bin_count",
        "primary_contact_distance_bin_count",
        "orientation_axis_names",
        "orientation_axis_is_cyclic",
        "orientation_bin_edge_convention",
        "orientation_bin_edges",
    )
    for name in schema_array_names:
        expected_hash = labels.output_hashes.get(f"{name}_sha256")
        if expected_hash != _hash_array(np.asarray(getattr(labels, name))):
            raise ValueError(f"Geometry-label schema field {name!r} failed hash validation")
    provenance = labels.provenance
    if not isinstance(provenance, Mapping):
        raise TypeError("Geometry-label provenance must be a mapping")
    expected = {
        "pair_orientation": PAIR_ORIENTATION,
        "distance_unit": DISTANCE_UNIT,
    }
    for key, value in expected.items():
        if provenance.get(key) != value:
            raise ValueError(
                f"Geometry-label provenance {key!r} must equal {value!r}"
            )
    from .geometry_labels import (
        FORMAT_VERSION,
        ORIENTATION_BIN_EDGE_CONVENTION,
        ORIENTATION_SCHEMA_VERSION,
        PRIMARY_CONTACT_CUTOFF_ANGSTROM,
    )

    primary_definition = provenance.get("primary_contact_definition")
    distance_definition = provenance.get("distance_definition")
    orientation_definition = provenance.get("orientation_definition")
    label_format_version = provenance.get("label_format_version")
    orientation_schema_version = provenance.get("orientation_schema_version")
    orientation_bin_edge_convention = provenance.get(
        "orientation_bin_edge_convention"
    )
    expected_primary_definition = (
        "minimum_observed_residue_non_hydrogen_atom_distance_"
        f"le_{PRIMARY_CONTACT_CUTOFF_ANGSTROM:g}_angstrom"
        "_with_positive_or_complete_pair_validity"
    )
    if primary_definition != expected_primary_definition:
        raise ValueError("Unsupported primary contact definition")
    expected_distance_definition = (
        "exact_minimum_non_hydrogen_atom_distance_only_when_both_residues"
        "_have_explicit_heavy_atom_completeness"
    )
    if distance_definition != expected_distance_definition:
        raise ValueError("Unsupported distance definition")
    if orientation_bin_edge_convention != ORIENTATION_BIN_EDGE_CONVENTION:
        raise ValueError("Unsupported orientation bin-edge convention")
    if label_format_version != FORMAT_VERSION:
        raise ValueError(
            f"Geometry-label format must equal the supported {FORMAT_VERSION!r}"
        )
    if orientation_schema_version != ORIENTATION_SCHEMA_VERSION:
        raise ValueError(
            "Geometry-label orientation schema version is unsupported"
        )
    schema = PairGeometrySchema(
        distance_bin_edges_angstrom=tuple(
            float(value) for value in np.asarray(labels.distance_bin_edges).tolist()
        ),
        contact_cutoff_angstrom=PRIMARY_CONTACT_CUTOFF_ANGSTROM,
        orientation_axis_names=tuple(
            str(value) for value in np.asarray(labels.orientation_axis_names).tolist()
        ),
        orientation_axis_is_cyclic=tuple(
            bool(value)
            for value in np.asarray(labels.orientation_axis_is_cyclic).tolist()
        ),
        orientation_bin_edges_radians=tuple(
            tuple(float(edge) for edge in values)
            for values in np.asarray(labels.orientation_bin_edges).tolist()
        ),
        label_format_version=label_format_version,
        orientation_schema_version=orientation_schema_version,
        primary_contact_definition=primary_definition,
        distance_definition=distance_definition,
        orientation_definition=orientation_definition,
        orientation_bin_edge_convention=orientation_bin_edge_convention,
    )
    artifact_convention = str(
        np.asarray(labels.orientation_bin_edge_convention).item()
    )
    if artifact_convention != schema.orientation_bin_edge_convention:
        raise ValueError(
            "Geometry-label orientation edge convention is inconsistent"
        )
    if int(np.asarray(labels.distance_bin_count).item()) != schema.num_distance_bins:
        raise ValueError("Geometry-label distance-bin count is internally inconsistent")
    if int(
        np.asarray(labels.primary_contact_distance_bin_count).item()
    ) != schema.contact_distance_bin_count:
        raise ValueError(
            "Geometry-label contact-distance mapping is internally inconsistent"
        )
    return schema


def adapt_geometry_labels(
    labels: Any,
    schema: PairGeometrySchema,
) -> PairGeometryTargetArrays:
    """Convert one validated label artifact into immutable categorical arrays."""
    if not isinstance(schema, PairGeometrySchema):
        raise TypeError("schema must be PairGeometrySchema")
    rebound = schema_from_geometry_labels(labels)
    if rebound != schema or rebound.sha256 != schema.sha256:
        raise ValueError("Geometry-label artifact schema does not match target schema")
    from .geometry_labels import _hash_array

    target_array_names = (
        "primary_contact_labels",
        "primary_contact_mask",
        "distance_bin_indices",
        "distance_bin_mask",
        "orientation_bin_indices",
        "orientation_bin_masks",
    )
    for name in target_array_names:
        expected_hash = labels.output_hashes.get(f"{name}_sha256")
        if expected_hash != _hash_array(np.asarray(getattr(labels, name))):
            raise ValueError(f"Geometry-label target field {name!r} failed hash validation")

    contact_targets = np.asarray(labels.primary_contact_labels)
    contact_mask = np.asarray(labels.primary_contact_mask)
    distance_targets = np.asarray(labels.distance_bin_indices)
    distance_mask = np.asarray(labels.distance_bin_mask)
    orientation_targets = np.asarray(labels.orientation_bin_indices)
    orientation_mask = np.asarray(labels.orientation_bin_masks)
    if contact_targets.shape != distance_targets.shape:
        raise ValueError("Contact and distance label shapes must match")
    if orientation_targets.shape != (
        *contact_targets.shape,
        schema.num_orientation_axes,
    ):
        raise ValueError(
            "Orientation targets must match pair shape and schema axis count"
        )
    if orientation_mask.shape != orientation_targets.shape:
        raise ValueError("Orientation targets and masks must have equal shapes")
    if bool(
        (
            (orientation_targets[orientation_mask] < 0)
            | (orientation_targets[orientation_mask] >= schema.num_orientation_bins)
        ).any()
    ):
        raise ValueError("Geometry-label orientation indices violate their schema")
    shared = np.asarray(contact_mask) & np.asarray(distance_mask)
    expected_contact = (
        np.asarray(distance_targets)[shared]
        < schema.contact_distance_bin_count
    )
    if not np.array_equal(np.asarray(contact_targets)[shared], expected_contact):
        raise ValueError("Geometry-label contact and distance labels contradict")

    arrays = {
        "contact_targets": np.asarray(contact_targets, dtype=np.bool_),
        "contact_mask": np.asarray(contact_mask, dtype=np.bool_),
        "distance_bin_targets": np.asarray(distance_targets, dtype=np.int64),
        "distance_mask": np.asarray(distance_mask, dtype=np.bool_),
        "orientation_bin_targets": np.asarray(orientation_targets, dtype=np.int64),
        "orientation_mask": np.asarray(orientation_mask, dtype=np.bool_),
    }
    for value in arrays.values():
        value.setflags(write=False)
    return PairGeometryTargetArrays(
        pair_orientation=PAIR_ORIENTATION,
        geometry_schema=schema,
        **arrays,
    )
