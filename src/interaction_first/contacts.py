"""Validated sparse contact-map artifacts for the SAbDab2 benchmark."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


FORMAT_VERSION = "sabdab2_sparse_contact_v1"


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ModuleNotFoundError as error:
        raise RuntimeError("NumPy is required for contact-map artifacts") from error
    return np


@dataclass(frozen=True)
class SparseContactMap:
    """One contact map restricted to authoritative candidate residue pairs."""

    instance_id: str
    antibody_length: int
    antigen_length: int
    antibody_indices: Any
    antigen_indices: Any
    pair_antibody_local: Any
    pair_antigen_local: Any
    pair_labels: Any

    @classmethod
    def from_rows(
        cls,
        instance_id: str,
        antibody_length: int,
        antigen_length: int,
        antibody_indices: Iterable[int],
        antigen_indices: Iterable[int],
        labels: Iterable[int | bool],
    ) -> "SparseContactMap":
        """Build a canonical local-index representation from candidate rows."""
        np = _require_numpy()
        ab_global = np.asarray(list(antibody_indices), dtype=np.int64)
        ag_global = np.asarray(list(antigen_indices), dtype=np.int64)
        pair_labels = np.asarray(list(labels), dtype=np.uint8)
        if not instance_id:
            raise ValueError("instance_id cannot be empty")
        if antibody_length <= 0 or antigen_length <= 0:
            raise ValueError("Sequence lengths must be positive")
        if ab_global.ndim != 1 or ag_global.ndim != 1 or pair_labels.ndim != 1:
            raise ValueError("Candidate rows must be one-dimensional")
        if not (len(ab_global) == len(ag_global) == len(pair_labels)):
            raise ValueError("Candidate row columns have different lengths")
        if len(pair_labels) == 0:
            raise ValueError("A contact map must contain at least one candidate pair")
        if ab_global.min() < 0 or ab_global.max() >= antibody_length:
            raise ValueError("Antibody residue index is outside the sequence")
        if ag_global.min() < 0 or ag_global.max() >= antigen_length:
            raise ValueError("Antigen residue index is outside the sequence")
        if not np.isin(pair_labels, (0, 1)).all():
            raise ValueError("Contact labels must be binary")

        ab_unique = np.unique(ab_global)
        ag_unique = np.unique(ag_global)
        ab_local = np.searchsorted(ab_unique, ab_global)
        ag_local = np.searchsorted(ag_unique, ag_global)
        linear = ag_local * len(ab_unique) + ab_local
        if len(np.unique(linear)) != len(linear):
            raise ValueError("Candidate contact rows contain duplicate residue pairs")

        result = cls(
            instance_id=str(instance_id),
            antibody_length=int(antibody_length),
            antigen_length=int(antigen_length),
            antibody_indices=ab_unique,
            antigen_indices=ag_unique,
            pair_antibody_local=ab_local,
            pair_antigen_local=ag_local,
            pair_labels=pair_labels,
        )
        result.validate()
        return result

    @property
    def candidate_count(self) -> int:
        """Return the number of supervised candidate residue pairs."""
        return int(len(self.pair_labels))

    @property
    def positive_count(self) -> int:
        """Return the number of positive residue-pair contacts."""
        return int(self.pair_labels.sum())

    def validate(self) -> None:
        """Enforce index, shape, and binary-label invariants."""
        np = _require_numpy()
        arrays = (
            self.antibody_indices,
            self.antigen_indices,
            self.pair_antibody_local,
            self.pair_antigen_local,
            self.pair_labels,
        )
        if any(np.asarray(array).ndim != 1 for array in arrays):
            raise ValueError("All sparse contact arrays must be one-dimensional")
        if len(self.antibody_indices) == 0 or len(self.antigen_indices) == 0:
            raise ValueError("Both contact-map residue axes must be non-empty")
        if not (
            len(self.pair_antibody_local)
            == len(self.pair_antigen_local)
            == len(self.pair_labels)
        ):
            raise ValueError("Sparse pair arrays have inconsistent lengths")
        if not np.array_equal(self.antibody_indices, np.unique(self.antibody_indices)):
            raise ValueError("Antibody residue indices must be sorted and unique")
        if not np.array_equal(self.antigen_indices, np.unique(self.antigen_indices)):
            raise ValueError("Antigen residue indices must be sorted and unique")
        if self.antibody_indices.min() < 0 or self.antibody_indices.max() >= self.antibody_length:
            raise ValueError("Antibody residue index is outside the sequence")
        if self.antigen_indices.min() < 0 or self.antigen_indices.max() >= self.antigen_length:
            raise ValueError("Antigen residue index is outside the sequence")
        if self.pair_antibody_local.min() < 0 or self.pair_antibody_local.max() >= len(
            self.antibody_indices
        ):
            raise ValueError("Local antibody pair index is outside the contact axis")
        if self.pair_antigen_local.min() < 0 or self.pair_antigen_local.max() >= len(
            self.antigen_indices
        ):
            raise ValueError("Local antigen pair index is outside the contact axis")
        if not np.isin(self.pair_labels, (0, 1)).all():
            raise ValueError("Contact labels must be binary")
        linear = self.pair_antigen_local * len(self.antibody_indices) + self.pair_antibody_local
        if len(np.unique(linear)) != len(linear):
            raise ValueError("Sparse contact map contains duplicate local pairs")

    def dense(self) -> tuple[Any, Any]:
        """Return dense labels and the authoritative candidate mask."""
        np = _require_numpy()
        shape = (len(self.antigen_indices), len(self.antibody_indices))
        labels = np.zeros(shape, dtype=np.float32)
        mask = np.zeros(shape, dtype=np.bool_)
        labels[self.pair_antigen_local, self.pair_antibody_local] = self.pair_labels
        mask[self.pair_antigen_local, self.pair_antibody_local] = True
        return labels, mask

    def node_labels(self) -> tuple[Any, Any]:
        """Derive epitope and paratope labels from positive contact edges."""
        np = _require_numpy()
        epitope = np.zeros(len(self.antigen_indices), dtype=np.float32)
        paratope = np.zeros(len(self.antibody_indices), dtype=np.float32)
        positive = self.pair_labels.astype(bool)
        epitope[self.pair_antigen_local[positive]] = 1.0
        paratope[self.pair_antibody_local[positive]] = 1.0
        return epitope, paratope

    def payload(self) -> Mapping[str, Any]:
        """Return the strict serialized payload."""
        np = _require_numpy()
        return {
            "format_version": np.asarray(FORMAT_VERSION),
            "instance_id": np.asarray(self.instance_id),
            "antibody_length": np.asarray(self.antibody_length, dtype=np.int64),
            "antigen_length": np.asarray(self.antigen_length, dtype=np.int64),
            "antibody_indices": np.asarray(self.antibody_indices, dtype=np.int64),
            "antigen_indices": np.asarray(self.antigen_indices, dtype=np.int64),
            "pair_antibody_local": np.asarray(self.pair_antibody_local, dtype=np.int64),
            "pair_antigen_local": np.asarray(self.pair_antigen_local, dtype=np.int64),
            "pair_labels": np.asarray(self.pair_labels, dtype=np.uint8),
        }


def save_contact_map(contact_map: SparseContactMap, path: Path) -> None:
    """Write one validated compressed artifact atomically."""
    np = _require_numpy()
    contact_map.validate()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **contact_map.payload())
    os.replace(temporary, path)


def load_contact_map(path: Path, expected_instance_id: str | None = None) -> SparseContactMap:
    """Load and validate one contact-map artifact without pickle support."""
    np = _require_numpy()
    with np.load(path, allow_pickle=False) as artifact:
        required = {
            "format_version",
            "instance_id",
            "antibody_length",
            "antigen_length",
            "antibody_indices",
            "antigen_indices",
            "pair_antibody_local",
            "pair_antigen_local",
            "pair_labels",
        }
        if set(artifact.files) != required:
            raise ValueError(f"Unexpected contact artifact keys in {path}")
        if str(artifact["format_version"].item()) != FORMAT_VERSION:
            raise ValueError(f"Unsupported contact artifact format in {path}")
        result = SparseContactMap(
            instance_id=str(artifact["instance_id"].item()),
            antibody_length=int(artifact["antibody_length"].item()),
            antigen_length=int(artifact["antigen_length"].item()),
            antibody_indices=np.asarray(artifact["antibody_indices"], dtype=np.int64),
            antigen_indices=np.asarray(artifact["antigen_indices"], dtype=np.int64),
            pair_antibody_local=np.asarray(artifact["pair_antibody_local"], dtype=np.int64),
            pair_antigen_local=np.asarray(artifact["pair_antigen_local"], dtype=np.int64),
            pair_labels=np.asarray(artifact["pair_labels"], dtype=np.uint8),
        )
    result.validate()
    if expected_instance_id is not None and result.instance_id != expected_instance_id:
        raise ValueError(f"Contact artifact identity mismatch in {path}")
    return result
