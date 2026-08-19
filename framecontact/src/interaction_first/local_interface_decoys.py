"""Geometry-matched local interface motifs and hard decoys.

This module implements the negative hierarchy discussed for FrameContact:

* native local non-contact: the same antibody CDR patch and antigen surface
  patch as a true contact pair, but residue pairs that are not native contacts;
* matched local hard decoy: the same antibody CDR patch against a *different*
  antigen surface patch from the same complex, selected without model scores
  and matched on observable local geometry/surface descriptors.

The constructor never reads a frozen or official test split.  It is deliberately
NumPy-only so that an authenticated development cache can materialize masks on a
login node before a GPU run.  The labels used to exclude native epitope/contact
patches are training/development labels, not a test-time selection mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Iterable, Sequence

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ModuleNotFoundError:  # pragma: no cover
    torch = None
    Dataset = object


LOCAL_INTERFACE_DECOY_FORMAT = "framecontact_local_interface_decoy_v1"
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
DEFAULT_DISTANCE_BINS = (4.5, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0)


class LocalInterfaceDecoyError(ValueError):
    """Raised when a local motif/decoy cannot be constructed safely."""


@dataclass(frozen=True)
class PatchDescriptor:
    """Auditable features used to match a candidate local surface patch."""

    indices: tuple[int, ...]
    center_index: int
    surface_area: float
    residue_count: int
    mean_sasa: float
    curvature: float
    composition: tuple[float, ...]
    center_distance: float
    distance_bin: int


@dataclass(frozen=True)
class LocalMotifSample:
    """One positive motif and its two explicitly different negative tiers."""

    antibody_center: int
    antigen_center: int
    decoy_center: int
    antibody_patch: tuple[int, ...]
    antigen_patch: tuple[int, ...]
    decoy_patch: tuple[int, ...]
    native_contact_pairs: tuple[tuple[int, int], ...]
    native_local_noncontact_pairs: tuple[tuple[int, int], ...]
    positive_mask: np.ndarray
    native_local_noncontact_mask: np.ndarray
    matched_local_decoy_mask: np.ndarray
    positive_descriptor: PatchDescriptor
    decoy_descriptor: PatchDescriptor
    descriptor_distance: float


def _validate_coords(name: str, coords: np.ndarray) -> np.ndarray:
    value = np.asarray(coords, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 3:
        raise LocalInterfaceDecoyError(f"{name} must have shape [residue, 3]")
    if value.shape[0] == 0 or not np.isfinite(value).all():
        raise LocalInterfaceDecoyError(f"{name} must be nonempty and finite")
    return value


def _validate_mask(name: str, mask: np.ndarray, length: int) -> np.ndarray:
    value = np.asarray(mask, dtype=bool)
    if value.shape != (length,):
        raise LocalInterfaceDecoyError(f"{name} must have shape [{length}]")
    return value


def _shortest_path_distances(coords: np.ndarray, graph_radius: float) -> np.ndarray:
    """Return all-pairs geodesic distances on a residue proximity graph."""

    if not math.isfinite(graph_radius) or graph_radius <= 0:
        raise LocalInterfaceDecoyError("graph_radius must be finite and positive")
    euclidean = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    adjacency = euclidean <= graph_radius
    np.fill_diagonal(adjacency, False)
    result = np.full(euclidean.shape, np.inf, dtype=np.float64)
    np.fill_diagonal(result, 0.0)
    for source in range(coords.shape[0]):
        queue: list[tuple[float, int]] = [(0.0, source)]
        while queue:
            distance, node = heapq.heappop(queue)
            if distance != result[source, node]:
                continue
            neighbours = np.flatnonzero(adjacency[node])
            for target in neighbours:
                candidate = distance + float(euclidean[node, target])
                if candidate < result[source, target]:
                    result[source, target] = candidate
                    heapq.heappush(queue, (candidate, int(target)))
    return result


def _curvature_proxy(coords: np.ndarray, centre: int, *, neighbour_k: int) -> float:
    """A deterministic local curvature proxy from k-nearest CA distances.

    This is not a solvent surface curvature estimate.  It is intentionally
    named a proxy and is used only for matching decoys, never as a label.
    """

    distances = np.linalg.norm(coords - coords[centre], axis=1)
    distances = np.delete(distances, centre)
    if distances.size == 0:
        return 0.0
    nearest = np.sort(distances)[: max(1, min(neighbour_k, distances.size))]
    return float(np.std(nearest) / max(float(np.mean(nearest)), 1e-6))


def _composition(residues: Sequence[str] | None, indices: Sequence[int]) -> tuple[float, ...]:
    counts = np.zeros(len(AMINO_ACIDS), dtype=np.float64)
    if residues is not None:
        if len(residues) <= max(indices, default=-1):
            raise LocalInterfaceDecoyError("residue sequence is shorter than patch")
        for index in indices:
            letter = str(residues[index]).upper()
            if letter in AMINO_ACIDS:
                counts[AMINO_ACIDS.index(letter)] += 1.0
    total = counts.sum()
    return tuple((counts / total if total else counts).tolist())


def _distance_bin(distance: float, bins: Sequence[float]) -> int:
    values = tuple(float(value) for value in bins)
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise LocalInterfaceDecoyError("distance bins must be finite and positive")
    if any(right <= left for left, right in zip(values, values[1:])):
        raise LocalInterfaceDecoyError("distance bins must be strictly increasing")
    return int(np.searchsorted(np.asarray(values), distance, side="right"))


def extract_surface_patch(
    coords: np.ndarray,
    surface_mask: np.ndarray,
    centre_index: int,
    *,
    radius: float = 12.0,
    graph_radius: float = 8.0,
    sasa: Sequence[float] | None = None,
    residues: Sequence[str] | None = None,
    centre_distance: float = math.nan,
    distance_bins: Sequence[float] = DEFAULT_DISTANCE_BINS,
    curvature_neighbour_k: int = 6,
) -> PatchDescriptor:
    """Extract one antigen surface patch with no model-dependent routing."""

    coords = _validate_coords("coords", coords)
    surface = _validate_mask("surface_mask", surface_mask, coords.shape[0])
    if isinstance(centre_index, bool) or not 0 <= int(centre_index) < coords.shape[0]:
        raise LocalInterfaceDecoyError("centre_index is outside the residue axis")
    if not math.isfinite(radius) or radius <= 0:
        raise LocalInterfaceDecoyError("radius must be finite and positive")
    if isinstance(curvature_neighbour_k, bool) or curvature_neighbour_k <= 0:
        raise LocalInterfaceDecoyError("curvature_neighbour_k must be positive")
    if not surface[int(centre_index)]:
        raise LocalInterfaceDecoyError("patch centre must be on the surface")
    sasa_values = np.ones(coords.shape[0], dtype=np.float64)
    if sasa is not None:
        sasa_values = np.asarray(sasa, dtype=np.float64)
        if sasa_values.shape != (coords.shape[0],) or not np.isfinite(sasa_values).all():
            raise LocalInterfaceDecoyError("sasa must be finite and align with coords")
        sasa_values = np.maximum(sasa_values, 0.0)
    geodesic = _shortest_path_distances(coords, graph_radius)
    selected = np.flatnonzero(surface & (geodesic[int(centre_index)] <= radius))
    if selected.size == 0:
        raise LocalInterfaceDecoyError("surface patch is empty")
    mean_sasa = float(np.mean(sasa_values[selected]))
    composition = _composition(residues, selected.tolist())
    return PatchDescriptor(
        indices=tuple(int(index) for index in selected),
        center_index=int(centre_index),
        surface_area=float(np.sum(sasa_values[selected])),
        residue_count=int(selected.size),
        mean_sasa=mean_sasa,
        curvature=_curvature_proxy(coords, int(centre_index), neighbour_k=curvature_neighbour_k),
        composition=composition,
        center_distance=float(centre_distance),
        distance_bin=_distance_bin(float(centre_distance), distance_bins)
        if math.isfinite(centre_distance)
        else -1,
    )


def _descriptor_distance(left: PatchDescriptor, right: PatchDescriptor) -> float:
    """Compare scale-normalized geometry and surface chemistry descriptors."""

    scale = np.asarray(
        [
            max(left.residue_count, right.residue_count, 1),
            max(left.surface_area, right.surface_area, 1.0),
            max(abs(left.center_distance), abs(right.center_distance), 1.0),
            max(abs(left.mean_sasa), abs(right.mean_sasa), 1.0),
            1.0,
        ],
        dtype=np.float64,
    )
    scalar = np.asarray(
        [
            left.residue_count - right.residue_count,
            left.surface_area - right.surface_area,
            left.center_distance - right.center_distance,
            left.mean_sasa - right.mean_sasa,
            left.curvature - right.curvature,
        ],
        dtype=np.float64,
    )
    chemistry = np.asarray(left.composition) - np.asarray(right.composition)
    return float(np.linalg.norm(scalar / scale) + 0.5 * np.linalg.norm(chemistry, ord=1))


def build_local_motif_samples(
    antibody_coords: np.ndarray,
    antigen_coords: np.ndarray,
    contact_labels: np.ndarray,
    antibody_cdr_mask: np.ndarray,
    antigen_surface_mask: np.ndarray,
    antigen_epitope_mask: np.ndarray,
    *,
    patch_radius: float = 12.0,
    graph_radius: float = 8.0,
    antigen_sasa: Sequence[float] | None = None,
    antigen_residues: Sequence[str] | None = None,
    distance_bins: Sequence[float] = DEFAULT_DISTANCE_BINS,
    boundary_exclusion_angstrom: tuple[float, float] | None = (4.5, 6.0),
    max_positive_pairs: int | None = None,
) -> tuple[LocalMotifSample, ...]:
    """Build positive, native-local-negative, and matched-decoy masks.

    Candidate decoy centres must be surface residues outside the authenticated
    epitope mask and their entire local patch must contain no native contact.
    Candidates are ranked by the descriptor distance to the positive antigen
    patch.  Ties are resolved by residue index, making the artifact exactly
    reproducible across workers and epochs.
    """

    antibody = _validate_coords("antibody_coords", antibody_coords)
    antigen = _validate_coords("antigen_coords", antigen_coords)
    contact = np.asarray(contact_labels, dtype=bool)
    if contact.shape != (antibody.shape[0], antigen.shape[0]):
        raise LocalInterfaceDecoyError("contact_labels must align with both residue axes")
    cdr = _validate_mask("antibody_cdr_mask", antibody_cdr_mask, antibody.shape[0])
    surface = _validate_mask("antigen_surface_mask", antigen_surface_mask, antigen.shape[0])
    epitope = _validate_mask("antigen_epitope_mask", antigen_epitope_mask, antigen.shape[0])
    if isinstance(max_positive_pairs, bool) or (
        max_positive_pairs is not None and max_positive_pairs <= 0
    ):
        raise LocalInterfaceDecoyError("max_positive_pairs must be positive or None")
    if boundary_exclusion_angstrom is not None:
        if len(boundary_exclusion_angstrom) != 2:
            raise LocalInterfaceDecoyError(
                "boundary_exclusion_angstrom must contain two values"
            )
        boundary_low, boundary_high = (
            float(boundary_exclusion_angstrom[0]),
            float(boundary_exclusion_angstrom[1]),
        )
        if (
            not math.isfinite(boundary_low)
            or not math.isfinite(boundary_high)
            or boundary_low < 0
            or boundary_high <= boundary_low
        ):
            raise LocalInterfaceDecoyError(
                "boundary_exclusion_angstrom must be finite and increasing"
            )
    else:
        boundary_low = boundary_high = math.nan

    positive_pairs = np.argwhere(contact & cdr[:, None] & surface[None, :])
    if max_positive_pairs is not None:
        positive_pairs = positive_pairs[: int(max_positive_pairs)]
    if positive_pairs.size == 0:
        return ()
    graph_ab = _shortest_path_distances(antibody, graph_radius)
    candidate_centres = np.flatnonzero(surface & ~epitope)
    samples: list[LocalMotifSample] = []
    for antibody_index, antigen_index in positive_pairs.tolist():
        antibody_patch = tuple(
            int(index)
            for index in np.flatnonzero(cdr & (graph_ab[antibody_index] <= patch_radius))
        )
        antigen_positive = extract_surface_patch(
            antigen,
            surface,
            antigen_index,
            radius=patch_radius,
            graph_radius=graph_radius,
            sasa=antigen_sasa,
            residues=antigen_residues,
            centre_distance=float(np.linalg.norm(antibody[antibody_index] - antigen[antigen_index])),
            distance_bins=distance_bins,
        )
        native_pairs = tuple(
            (int(left), int(right))
            for left, right in np.argwhere(
                contact[np.asarray(antibody_patch)][:, np.asarray(antigen_positive.indices)]
            ).tolist()
        )
        native_pairs = tuple(
            (antibody_patch[left], antigen_positive.indices[right])
            for left, right in native_pairs
        )
        native_noncontact = tuple(
            (left, right)
            for left in antibody_patch
            for right in antigen_positive.indices
            if not contact[left, right]
        )
        positive_mask = np.zeros_like(contact, dtype=bool)
        native_noncontact_mask = np.zeros_like(contact, dtype=bool)
        for left, right in native_pairs:
            positive_mask[left, right] = True
        for left, right in native_noncontact:
            native_noncontact_mask[left, right] = True
        if not native_pairs:
            continue

        decoys: list[tuple[float, int, PatchDescriptor]] = []
        for candidate in candidate_centres.tolist():
            centre_distance = float(
                np.linalg.norm(antibody[antibody_index] - antigen[candidate])
            )
            if boundary_exclusion_angstrom is not None and (
                boundary_low < centre_distance < boundary_high
            ):
                continue
            try:
                descriptor = extract_surface_patch(
                    antigen,
                    surface,
                    int(candidate),
                    radius=patch_radius,
                    graph_radius=graph_radius,
                    sasa=antigen_sasa,
                    residues=antigen_residues,
                    centre_distance=centre_distance,
                    distance_bins=distance_bins,
                )
            except LocalInterfaceDecoyError:
                continue
            if epitope[np.asarray(descriptor.indices)].any():
                continue
            if contact[:, np.asarray(descriptor.indices)].any():
                continue
            decoys.append((_descriptor_distance(antigen_positive, descriptor), int(candidate), descriptor))
        if not decoys:
            continue
        descriptor_distance, _, decoy = min(decoys, key=lambda item: (item[0], item[1]))
        decoy_mask = np.zeros_like(contact, dtype=bool)
        decoy_mask[np.asarray(antibody_patch)[:, None], np.asarray(decoy.indices)[None, :]] = True
        samples.append(
            LocalMotifSample(
                antibody_center=int(antibody_index),
                antigen_center=int(antigen_index),
                decoy_center=decoy.center_index,
                antibody_patch=antibody_patch,
                antigen_patch=antigen_positive.indices,
                decoy_patch=decoy.indices,
                native_contact_pairs=native_pairs,
                native_local_noncontact_pairs=native_noncontact,
                positive_mask=positive_mask,
                native_local_noncontact_mask=native_noncontact_mask,
                matched_local_decoy_mask=decoy_mask,
                positive_descriptor=antigen_positive,
                decoy_descriptor=decoy,
                descriptor_distance=float(descriptor_distance),
            )
        )
    return tuple(samples)


def merge_local_motif_masks(
    samples: Iterable[LocalMotifSample],
    *,
    antibody_count: int,
    antigen_count: int,
) -> dict[str, np.ndarray]:
    """Union motif masks for one complex into collator-ready boolean arrays.

    Multiple true contact pairs can share a local patch.  Unioning is
    deterministic and keeps every eligible positive/decoy pair visible to the
    objective while preserving the three-tier names in the batch schema.
    """

    if isinstance(antibody_count, bool) or antibody_count <= 0:
        raise LocalInterfaceDecoyError("antibody_count must be positive")
    if isinstance(antigen_count, bool) or antigen_count <= 0:
        raise LocalInterfaceDecoyError("antigen_count must be positive")
    shape = (int(antibody_count), int(antigen_count))
    result = {
        "local_motif_positive_mask": np.zeros(shape, dtype=bool),
        "local_motif_native_noncontact_mask": np.zeros(shape, dtype=bool),
        "local_motif_matched_decoy_mask": np.zeros(shape, dtype=bool),
    }
    for sample in samples:
        if not isinstance(sample, LocalMotifSample):
            raise LocalInterfaceDecoyError("samples must contain LocalMotifSample values")
        for key, mask in (
            ("local_motif_positive_mask", sample.positive_mask),
            ("local_motif_native_noncontact_mask", sample.native_local_noncontact_mask),
            ("local_motif_matched_decoy_mask", sample.matched_local_decoy_mask),
        ):
            if mask.shape != shape:
                raise LocalInterfaceDecoyError("motif mask shape differs from requested axes")
            result[key] |= mask
    return result


class LocalMotifMaskDataset(Dataset):
    """Attach development-only motif masks to a pair-geometry dataset.

    The wrapper is intended for the *training* dataset only.  It keeps the
    original ``records`` attribute so the frozen partner-panel checks remain
    valid, and it does not alter validation/test examples or model selection.
    """

    def __init__(
        self,
        base_dataset: object,
        *,
        patch_radius: float = 12.0,
        graph_radius: float = 8.0,
        max_positive_pairs: int | None = None,
    ) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for LocalMotifMaskDataset")
        self.base_dataset = base_dataset
        self.records = base_dataset.records
        self.patch_radius = float(patch_radius)
        self.graph_radius = float(graph_radius)
        self.max_positive_pairs = max_positive_pairs
        self._cache: dict[int, dict[str, object]] = {}

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> dict[str, object]:
        if index in self._cache:
            return self._cache[index]
        item = dict(self.base_dataset[index])
        required = (
            "pair_labels",
            "antibody_cdr_mask",
            "antigen_surface_mask",
            "antibody_backbone",
            "antigen_backbone",
            "antigen_labels",
        )
        missing = [key for key in required if key not in item]
        if missing:
            raise LocalInterfaceDecoyError(
                "LocalMotifMaskDataset requires pair geometry and residue labels; missing "
                + ", ".join(missing)
            )
        samples = build_local_motif_samples(
            item["antibody_backbone"][:, 1, :].detach().cpu().numpy(),
            item["antigen_backbone"][:, 1, :].detach().cpu().numpy(),
            item["pair_labels"].detach().cpu().numpy(),
            item["antibody_cdr_mask"].detach().cpu().numpy(),
            item["antigen_surface_mask"].detach().cpu().numpy(),
            item["antigen_labels"].detach().cpu().numpy().astype(bool),
            patch_radius=self.patch_radius,
            graph_radius=self.graph_radius,
            max_positive_pairs=self.max_positive_pairs,
        )
        masks = merge_local_motif_masks(
            samples,
            antibody_count=int(item["pair_labels"].shape[0]),
            antigen_count=int(item["pair_labels"].shape[1]),
        )
        for key, value in masks.items():
            item[key] = torch.from_numpy(value)
        item["local_motif_sample_count"] = len(samples)
        self._cache[index] = item
        return item


__all__ = [
    "AMINO_ACIDS",
    "DEFAULT_DISTANCE_BINS",
    "LOCAL_INTERFACE_DECOY_FORMAT",
    "LocalInterfaceDecoyError",
    "LocalMotifSample",
    "LocalMotifMaskDataset",
    "PatchDescriptor",
    "build_local_motif_samples",
    "extract_surface_patch",
    "merge_local_motif_masks",
]
