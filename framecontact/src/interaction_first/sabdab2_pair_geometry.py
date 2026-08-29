"""Authenticated SAbDab2 distance/CDR/surface supervision for FrameContact.

The companion cache is intentionally separate from the immutable N/CA/C
structure cache.  It recovers the same minimum non-hydrogen-atom distance used
by the binary contact authority, stores only a capped distance matrix (the last
distogram bin is right-censored), and proves that ``distance < cutoff`` exactly
reconstructs every observed contact label before publishing an artifact.  A
label mismatch may be reconciled only inside an explicit coordinate-rounding
uncertainty band around the cutoff; every such adjustment is counted and the
largest adjustment is recorded in the authenticated manifest.

Antibody CDR masks are obtained from the release's explicit CDR sequences and
expanded by a registered within-chain flank.  Antigen surface masks are
computed on the isolated antigen with Shrake--Rupley relative SASA, so the
bound-state antibody cannot hide the cognate interface during surface routing.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

try:
    import torch
    from torch.utils.data import Dataset
except ModuleNotFoundError:  # pragma: no cover
    torch = None
    Dataset = object

from .sabdab2_contacts import Sabdab2ContactBatchCollator
from .sabdab2_structure import (
    StructureCacheError,
    _bio_parser,
    _canonical_sequence,
    _chain_backbone,
    _chain_lookup,
    _chain_names,
    _full_chain_specs,
    _parse_numbering_list,
    _project_specs_to_canonical,
    _residue_lookup,
    _residue_one_letter,
    _split_antigen_specs,
    _string,
    authenticate_file,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)


PAIR_GEOMETRY_CACHE_FORMAT = "sabdab2_framecontact_pair_geometry_cache_v1"
PAIR_GEOMETRY_ARTIFACT_FORMAT = "sabdab2_framecontact_pair_geometry_artifact_v1"
PAIR_GEOMETRY_RECEIPT_FORMAT = "sabdab2_framecontact_pair_geometry_receipt_v1"
PAIR_GEOMETRY_SELECTION_FORMAT = "sabdab2_framecontact_pair_geometry_selection_v1"

# The release's VH/VL numbering lists use IMGT positions for the explicit CDR
# strings.  Numbering is the authority; the strings are an independent check.
_IMGT_CDR_NUMBER_RANGES = ((27, 38), (56, 65), (105, 117))


# Maximum solvent-accessible areas from Tien et al. (2013), in A^2.
_MAX_ASA = {
    "A": 129.0,
    "R": 274.0,
    "N": 195.0,
    "D": 193.0,
    "C": 167.0,
    "Q": 225.0,
    "E": 223.0,
    "G": 104.0,
    "H": 224.0,
    "I": 197.0,
    "L": 201.0,
    "K": 236.0,
    "M": 224.0,
    "F": 240.0,
    "P": 159.0,
    "S": 155.0,
    "T": 172.0,
    "W": 285.0,
    "Y": 263.0,
    "V": 174.0,
    "X": 200.0,
}


def _require_torch() -> Any:
    if torch is None:
        raise RuntimeError("PyTorch is required for SAbDab2 pair geometry")
    return torch


def _csv_index(path: Path, key: str) -> dict[str, dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as error:
        raise StructureCacheError(f"Cannot read CSV authority: {path}") from error
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        identifier = _string(row.get(key))
        if not identifier or identifier in result:
            raise StructureCacheError(f"CSV {path} has invalid {key!r} values")
        result[identifier] = row
    return result


def freeze_pair_geometry_development_selection(
    *,
    split_path: Path,
    split_sha256: str,
    structure_cache_root: Path,
    structure_manifest_sha256: str,
    contact_cache_root: Path,
    contact_manifest_sha256: str,
    output_path: Path,
) -> dict[str, Any]:
    """Freeze an authenticated train+validation-only geometry selection."""
    authenticate_file(split_path, split_sha256, "SAbDab2 split manifest")
    structure_path = structure_cache_root / "STRUCTURE_CACHE_MANIFEST.json"
    contact_path = contact_cache_root / "CONTACT_CACHE_MANIFEST.json"
    authenticate_file(
        structure_path, structure_manifest_sha256, "structure cache manifest"
    )
    authenticate_file(contact_path, contact_manifest_sha256, "contact cache manifest")
    structure_manifest = json.loads(structure_path.read_text(encoding="utf-8"))
    contact_manifest = json.loads(contact_path.read_text(encoding="utf-8"))
    structure_ids = [
        str(row.get("instance_id", "")) for row in structure_manifest["records"]
    ]
    contact_ids = [
        str(row.get("instance_id", "")) for row in contact_manifest["records"]
    ]
    if (
        not structure_ids
        or len(set(structure_ids)) != len(structure_ids)
        or len(set(contact_ids)) != len(contact_ids)
        or set(structure_ids) != set(contact_ids)
    ):
        raise StructureCacheError("structure/contact cohort identifiers are invalid")
    split = _csv_index(split_path, "instance_id")
    if set(split) != set(structure_ids):
        raise StructureCacheError("split and structure/contact cohorts differ")
    roles = {identifier: _string(row.get("role")) for identifier, row in split.items()}
    if set(roles.values()) != {"train", "validation", "test"}:
        raise StructureCacheError("split roles must be exactly train/validation/test")
    selected = [
        {"instance_id": identifier, "role": roles[identifier]}
        for identifier in sorted(roles)
        if roles[identifier] in {"train", "validation"}
    ]
    if not selected:
        raise StructureCacheError("development selection is empty")
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite selection: {output_path}")
    payload = {
        "format_version": PAIR_GEOMETRY_SELECTION_FORMAT,
        "authority": {
            "split_manifest_sha256": split_sha256,
            "structure_manifest_sha256": structure_manifest_sha256,
            "contact_manifest_sha256": contact_manifest_sha256,
        },
        "included_roles": ["train", "validation"],
        "excluded_roles": ["test"],
        "counts": {
            "selected": len(selected),
            "train": sum(row["role"] == "train" for row in selected),
            "validation": sum(row["role"] == "validation" for row in selected),
            "excluded_test": sum(role == "test" for role in roles.values()),
        },
        "records": selected,
        "labels_or_model_scores_read": False,
        "claim_boundary": (
            "Identifier and role selection only; no residue label, pair label, model "
            "prediction, or metric was read."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_bytes(canonical_json_bytes(payload))
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {**payload, "selection_sha256": sha256_file(output_path)}


def _authenticated_selection(
    path: Path, expected_sha256: str
) -> tuple[list[str], dict[str, Any]]:
    observed_sha256 = authenticate_file(
        path, expected_sha256, "pair geometry development selection"
    )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise StructureCacheError("cannot parse pair geometry selection") from error
    if payload.get("format_version") != PAIR_GEOMETRY_SELECTION_FORMAT:
        raise StructureCacheError("unknown pair geometry selection format")
    if payload.get("included_roles") != ["train", "validation"]:
        raise StructureCacheError("pair geometry selection must include train/validation")
    if payload.get("excluded_roles") != ["test"]:
        raise StructureCacheError("pair geometry selection must exclude test")
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise StructureCacheError("pair geometry selection is empty")
    identifiers: list[str] = []
    roles: list[str] = []
    for row in records:
        if not isinstance(row, Mapping):
            raise StructureCacheError("pair geometry selection record is invalid")
        identifier = _string(row.get("instance_id"))
        role = _string(row.get("role"))
        if not identifier or role not in {"train", "validation"}:
            raise StructureCacheError("pair geometry selection record is invalid")
        identifiers.append(identifier)
        roles.append(role)
    if len(set(identifiers)) != len(identifiers):
        raise StructureCacheError("pair geometry selection has duplicate identifiers")
    counts = payload.get("counts", {})
    if (
        int(counts.get("selected", -1)) != len(identifiers)
        or int(counts.get("train", -1)) != roles.count("train")
        or int(counts.get("validation", -1)) != roles.count("validation")
        or int(counts.get("excluded_test", -1)) <= 0
    ):
        raise StructureCacheError("pair geometry selection counts differ")
    return identifiers, {
        "selection_manifest_sha256": observed_sha256,
        "selection_split_manifest_sha256": payload.get("authority", {}).get(
            "split_manifest_sha256"
        ),
        "selection_roles": ["train", "validation"],
        "selection_excluded_roles": ["test"],
    }


def _mapped_residues(chain: Any, specs: Any) -> tuple[Any | None, ...]:
    if specs is None:
        specs = _full_chain_specs(chain)
    lookup = _residue_lookup(chain)
    return tuple(lookup.get((spec.number, spec.insertion_code)) for spec in specs)


def _ordered_segment_mask(sequence: str, segments: Sequence[str]) -> list[bool]:
    """Require a unique ordered placement of the release's explicit CDR strings."""
    normalized = tuple(segment.strip().upper() for segment in segments if segment.strip())
    placements: list[tuple[tuple[int, int], ...]] = []

    def visit(index: int, cursor: int, selected: tuple[tuple[int, int], ...]) -> None:
        if len(placements) > 1:
            return
        if index == len(normalized):
            placements.append(selected)
            return
        segment = normalized[index]
        start = sequence.find(segment, cursor)
        while start >= 0:
            visit(index + 1, start + len(segment), (*selected, (start, start + len(segment))))
            start = sequence.find(segment, start + 1)

    visit(0, 0, ())
    if len(placements) != 1:
        raise StructureCacheError("CDR sequences do not have one unique ordered placement")
    mask = [False] * len(sequence)
    for start, stop in placements[0]:
        mask[start:stop] = [True] * (stop - start)
    return mask


def _numbered_cdr_mask(
    source_specs: Sequence[Any],
    projected_specs: Sequence[Any],
    explicit_segments: Sequence[str],
) -> list[bool]:
    """Map CDRs by IMGT number and authenticate them against release strings."""
    normalized = tuple(segment.strip().upper() for segment in explicit_segments)
    if len(normalized) != len(_IMGT_CDR_NUMBER_RANGES) or any(
        not segment for segment in normalized
    ):
        raise StructureCacheError("release CDR strings must contain CDR1/2/3")
    for (start, stop), expected in zip(
        _IMGT_CDR_NUMBER_RANGES, normalized, strict=True
    ):
        observed = "".join(
            spec.amino_acid for spec in source_specs if start <= spec.number <= stop
        )
        if observed != expected:
            raise StructureCacheError(
                "numbered CDR residues differ from the explicit release string"
            )
    mask = [
        any(start <= spec.number <= stop for start, stop in _IMGT_CDR_NUMBER_RANGES)
        for spec in projected_specs
    ]
    if not any(mask):
        raise StructureCacheError("numbered CDR mask is empty after axis projection")
    return mask


def _expand_mask(mask: Sequence[bool], flank: int) -> list[bool]:
    result = list(mask)
    for index, selected in enumerate(mask):
        if selected:
            result[max(0, index - flank) : min(len(mask), index + flank + 1)] = [
                True
            ] * (min(len(mask), index + flank + 1) - max(0, index - flank))
    return result


def _heavy_atom_coordinates(residue: Any | None) -> Any:
    import numpy as np

    if residue is None:
        return np.zeros((0, 3), dtype=np.float32)
    residue_conformers = (
        tuple(residue.disordered_get_list())
        if getattr(residue, "is_disordered", lambda: 0)() == 2
        and hasattr(residue, "disordered_get_list")
        else (residue,)
    )
    coordinates = []
    for conformer in residue_conformers:
        for selected_atom in conformer.get_atoms():
            atom_conformers = (
                tuple(selected_atom.disordered_get_list())
                if getattr(selected_atom, "is_disordered", lambda: 0)()
                and hasattr(selected_atom, "disordered_get_list")
                else (selected_atom,)
            )
            for atom in atom_conformers:
                element = str(getattr(atom, "element", "") or "").strip().upper()
                name = str(atom.get_name()).strip().upper()
                if element in {"H", "D"} or (
                    not element and name.startswith(("H", "D"))
                ):
                    continue
                value = np.asarray(atom.get_coord(), dtype=np.float32)
                if value.shape == (3,) and np.isfinite(value).all():
                    coordinates.append(value)
    return (
        np.stack(coordinates)
        if coordinates
        else np.zeros((0, 3), dtype=np.float32)
    )


def _capped_minimum_heavy_atom_distances(
    antibody_residues: Sequence[Any | None],
    antigen_residues: Sequence[Any | None],
    *,
    distance_cap: float,
) -> tuple[Any, Any]:
    import numpy as np
    from scipy.spatial import cKDTree

    antibody_atoms = tuple(_heavy_atom_coordinates(value) for value in antibody_residues)
    antigen_atoms = tuple(_heavy_atom_coordinates(value) for value in antigen_residues)
    antibody_observed = np.asarray([len(value) > 0 for value in antibody_atoms])
    antigen_observed = np.asarray([len(value) > 0 for value in antigen_atoms])
    pair_mask = antibody_observed[:, None] & antigen_observed[None, :]
    censored_value = np.nextafter(
        np.float32(distance_cap), np.float32(np.inf), dtype=np.float32
    )
    distances = np.full(pair_mask.shape, censored_value, dtype=np.float32)
    if not antigen_observed.any() or not antibody_observed.any():
        return distances, pair_mask
    antigen_coordinates = np.concatenate(
        [value for value in antigen_atoms if len(value)], axis=0
    )
    antigen_residue_index = np.concatenate(
        [
            np.full(len(value), index, dtype=np.int64)
            for index, value in enumerate(antigen_atoms)
            if len(value)
        ]
    )
    tree = cKDTree(antigen_coordinates)
    for antibody_index, coordinates in enumerate(antibody_atoms):
        if not len(coordinates):
            continue
        for coordinate in coordinates:
            neighbor_atoms = tree.query_ball_point(coordinate, distance_cap)
            if not neighbor_atoms:
                continue
            neighbor_atoms = np.asarray(neighbor_atoms, dtype=np.int64)
            local = np.linalg.norm(
                antigen_coordinates[neighbor_atoms] - coordinate[None, :], axis=-1
            )
            np.minimum.at(
                distances[antibody_index],
                antigen_residue_index[neighbor_atoms],
                local.astype(np.float32, copy=False),
            )
    return distances, pair_mask


def _reconcile_contact_boundary(
    distances: Any,
    pair_mask: Any,
    labels: Any,
    *,
    contact_cutoff: float,
    coordinate_quantization_tolerance: float,
) -> tuple[Any, dict[str, Any]]:
    """Snap only cutoff-ambiguous distances to the label-consistent side.

    PDB-style Cartesian coordinates are commonly rounded to three decimal
    places.  Rounding both atoms can move their reconstructed distance by as
    much as ``sqrt(3) * 0.001`` Angstrom.  The source contact labels may have
    been calculated before that rounding, so a sub-millangstrom disagreement
    at the exact cutoff is not evidence for a different contact definition.
    This function preserves the raw distance everywhere else and fails closed
    on any mismatch outside the registered tolerance.
    """
    import numpy as np

    distances = np.asarray(distances, dtype=np.float32)
    pair_mask = np.asarray(pair_mask, dtype=np.bool_)
    labels = np.asarray(labels, dtype=np.bool_)
    if not (distances.shape == pair_mask.shape == labels.shape):
        raise StructureCacheError("distance, mask, and contact shapes differ")
    if bool((labels & ~pair_mask).any()):
        raise StructureCacheError("contact label exists outside the observed pair mask")
    derived = distances < contact_cutoff
    mismatch = pair_mask & (derived != labels)
    delta = np.abs(distances.astype(np.float64) - float(contact_cutoff))
    outside = mismatch & (delta > coordinate_quantization_tolerance)
    if bool(outside.any()):
        observed = delta[outside]
        raise StructureCacheError(
            "heavy-atom distance does not reproduce contact labels outside the "
            "coordinate-quantization tolerance "
            f"(count={int(outside.sum())}, max_delta={float(observed.max()):.9g})"
        )
    reconciled = distances.copy()
    positive = mismatch & labels
    negative = mismatch & ~labels
    reconciled[positive] = np.nextafter(
        np.float32(contact_cutoff), np.float32(-np.inf), dtype=np.float32
    )
    reconciled[negative] = np.nextafter(
        np.float32(contact_cutoff), np.float32(np.inf), dtype=np.float32
    )
    if bool((pair_mask & ((reconciled < contact_cutoff) != labels)).any()):
        raise AssertionError("boundary reconciliation did not recover contact labels")
    mismatch_count = int(mismatch.sum())
    return reconciled, {
        "boundary_reconciliation_count": mismatch_count,
        "positive_contact_reconciliation_count": int(positive.sum()),
        "negative_contact_reconciliation_count": int(negative.sum()),
        "maximum_boundary_adjustment_angstrom": (
            0.0 if mismatch_count == 0 else float(delta[mismatch].max())
        ),
    }


def _antigen_surface_mask(
    model: Any,
    residue_axis: Sequence[tuple[str, Any | None]],
    antigen_chain_names: Sequence[str],
    *,
    relative_sasa_threshold: float,
) -> list[bool]:
    try:
        from Bio.PDB.SASA import ShrakeRupley
    except ModuleNotFoundError as error:
        raise RuntimeError("Biopython ShrakeRupley is required for surface masks") from error

    isolated = copy.deepcopy(model)
    antigen_names = set(antigen_chain_names)
    for chain in tuple(isolated.get_chains()):
        if chain.id not in antigen_names:
            chain.get_parent().detach_child(chain.id)
    ShrakeRupley().compute(isolated, level="R")
    copy_lookup = {
        (chain.id, residue.id): residue
        for chain in isolated.get_chains()
        for residue in chain
    }
    result = []
    for chain_name, residue in residue_axis:
        copied = None if residue is None else copy_lookup.get((chain_name, residue.id))
        if copied is None:
            result.append(False)
            continue
        amino_acid = _residue_one_letter(copied)
        relative = float(getattr(copied, "sasa", 0.0)) / _MAX_ASA.get(
            amino_acid, _MAX_ASA["X"]
        )
        result.append(relative >= relative_sasa_threshold)
    return result


def _axes(
    model: Any,
    metadata: Mapping[str, Any],
    canonical: Mapping[str, Any],
    *,
    cdr_flank: int,
) -> tuple[tuple[Any | None, ...], tuple[tuple[str, Any | None], ...], list[bool]]:
    antibody_residues: list[Any | None] = []
    antibody_cdr: list[bool] = []
    definitions = (
        ("Hchain", "VH_numbering_list", "vh_seq", ("CDRH1", "CDRH2", "CDRH3")),
        ("Lchain", "VL_numbering_list", "vl_seq", ("CDRL1", "CDRL2", "CDRL3")),
    )
    for chain_field, numbering_field, sequence_field, cdr_fields in definitions:
        expected = _canonical_sequence(canonical, sequence_field)
        chain_name = _string(metadata.get(chain_field))
        if not expected:
            continue
        chain = _chain_lookup(model, chain_name)
        parsed = _parse_numbering_list(_string(metadata.get(numbering_field)))
        if parsed is None:
            raise StructureCacheError("Antibody numbering may not use full-chain sentinel")
        projected, _ = _project_specs_to_canonical(parsed, expected)
        sequence, _ = _chain_backbone(chain, projected)
        antibody_residues.extend(_mapped_residues(chain, projected))
        exact_cdr = _numbered_cdr_mask(
            parsed,
            projected,
            [_string(metadata.get(key)) for key in cdr_fields],
        )
        antibody_cdr.extend(_expand_mask(exact_cdr, cdr_flank))
    antigen_axis: list[tuple[str, Any | None]] = []
    antigen_chains = _chain_names(metadata.get("agchains"))
    raw_specs = _split_antigen_specs(
        _string(metadata.get("ag_index_list")), len(antigen_chains)
    )
    antigen_sequence = []
    for chain_name, raw_spec in zip(antigen_chains, raw_specs, strict=True):
        chain = _chain_lookup(model, chain_name)
        parsed = _parse_numbering_list(raw_spec)
        sequence, _ = _chain_backbone(chain, parsed)
        residues = _mapped_residues(chain, parsed)
        antigen_axis.extend((chain_name, residue) for residue in residues)
        antigen_sequence.append(sequence)
    if "".join(antigen_sequence) != _canonical_sequence(canonical, "ag_seq"):
        raise StructureCacheError("Antigen pair-geometry axis differs from canonical")
    return tuple(antibody_residues), tuple(antigen_axis), antibody_cdr


def materialize_pair_geometry_cache(
    *,
    structure_cache_root: Path,
    structure_manifest_sha256: str,
    contact_cache_root: Path,
    contact_manifest_sha256: str,
    cif_root: Path,
    metadata_path: Path,
    metadata_sha256: str,
    canonical_manifest_path: Path,
    canonical_manifest_sha256: str,
    output_root: Path,
    contact_cutoff: float = 4.5,
    distance_cap: float = 20.0,
    cdr_flank: int = 2,
    antigen_relative_sasa_threshold: float = 0.2,
    coordinate_quantization_tolerance: float = 0.002,
    max_instances: int | None = None,
    instance_ids: Sequence[str] | None = None,
    selection_manifest_path: Path | None = None,
    selection_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Build an atomic, digest-bound pair-geometry supervision cache."""
    import numpy as np

    torch_module = _require_torch()
    if not 0 < contact_cutoff < distance_cap:
        raise ValueError("contact cutoff must be positive and below the distance cap")
    if isinstance(cdr_flank, bool) or not isinstance(cdr_flank, int) or cdr_flank < 0:
        raise ValueError("cdr_flank must be a nonnegative integer")
    if not 0 <= antigen_relative_sasa_threshold <= 1:
        raise ValueError("relative SASA threshold must lie in [0, 1]")
    if not (
        math.isfinite(coordinate_quantization_tolerance)
        and 0 <= coordinate_quantization_tolerance < contact_cutoff
    ):
        raise ValueError(
            "coordinate quantization tolerance must be finite, nonnegative, "
            "and below the contact cutoff"
        )
    if (selection_manifest_path is None) != (selection_manifest_sha256 is None):
        raise ValueError("selection manifest path and SHA-256 must be provided together")
    selector_count = sum(
        value is not None
        for value in (max_instances, instance_ids, selection_manifest_path)
    )
    if selector_count > 1:
        raise ValueError(
            "max_instances, instance_ids, and selection manifest are mutually exclusive"
        )
    if max_instances is not None and (
        isinstance(max_instances, bool)
        or not isinstance(max_instances, int)
        or max_instances <= 0
    ):
        raise ValueError("max_instances must be a positive integer")
    structure_manifest_path = structure_cache_root / "STRUCTURE_CACHE_MANIFEST.json"
    contact_manifest_path = contact_cache_root / "CONTACT_CACHE_MANIFEST.json"
    authenticate_file(
        structure_manifest_path, structure_manifest_sha256, "structure manifest"
    )
    authenticate_file(contact_manifest_path, contact_manifest_sha256, "contact manifest")
    authenticate_file(metadata_path, metadata_sha256, "SAbDab2 metadata")
    authenticate_file(
        canonical_manifest_path,
        canonical_manifest_sha256,
        "canonical manifest",
    )
    structure_manifest = json.loads(structure_manifest_path.read_text())
    contact_manifest = json.loads(contact_manifest_path.read_text())
    structure_records = {
        record["instance_id"]: record for record in structure_manifest["records"]
    }
    contact_records = {
        record["instance_id"]: record for record in contact_manifest["records"]
    }
    if set(structure_records) != set(contact_records):
        raise StructureCacheError("structure and contact cohort identifiers differ")
    identifiers = sorted(structure_records)
    selection_authority: dict[str, Any] = {}
    if selection_manifest_path is not None:
        selected, selection_authority = _authenticated_selection(
            selection_manifest_path, str(selection_manifest_sha256)
        )
        missing = sorted(set(selected).difference(structure_records))
        if missing:
            raise StructureCacheError(
                f"pair geometry cohort lacks selected instances: {missing[:5]}"
            )
        identifiers = sorted(selected)
    elif instance_ids is not None:
        selected = tuple(str(identifier).strip() for identifier in instance_ids)
        if not selected or any(not identifier for identifier in selected):
            raise ValueError("instance_ids must contain at least one nonempty identifier")
        if len(set(selected)) != len(selected):
            raise ValueError("instance_ids must not contain duplicates")
        missing = sorted(set(selected).difference(structure_records))
        if missing:
            raise StructureCacheError(
                f"pair geometry cohort lacks selected instances: {missing[:5]}"
            )
        identifiers = sorted(selected)
    elif max_instances is not None:
        identifiers = identifiers[:max_instances]
    metadata = _csv_index(metadata_path, "INSTANCE")
    canonical = _csv_index(canonical_manifest_path, "INSTANCE")
    if any(identifier not in metadata or identifier not in canonical for identifier in identifiers):
        raise StructureCacheError("pair geometry authorities lack a selected instance")
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"Refusing to overwrite pair geometry cache: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent))
    records = []
    total_observed = 0
    total_censored = 0
    total_boundary_reconciliations = 0
    maximum_boundary_adjustment = 0.0
    try:
        for identifier in identifiers:
            structure_record = structure_records[identifier]
            contact_record = contact_records[identifier]
            cif_path = cif_root / f"{identifier}.cif"
            if sha256_file(cif_path) != structure_record["source_cif_sha256"]:
                raise StructureCacheError(f"{identifier}: mmCIF digest differs")
            try:
                structure = _bio_parser().get_structure(identifier, str(cif_path))
                model = next(structure.get_models())
            except Exception as error:
                raise StructureCacheError(f"{identifier}: cannot parse mmCIF") from error
            try:
                antibody_axis, antigen_axis, cdr_mask = _axes(
                    model,
                    metadata[identifier],
                    canonical[identifier],
                    cdr_flank=cdr_flank,
                )
            except StructureCacheError as error:
                raise StructureCacheError(f"{identifier}: {error}") from error
            if (
                len(antibody_axis) != int(structure_record["antibody_length"])
                or len(antigen_axis) != int(structure_record["antigen_length"])
            ):
                raise StructureCacheError(f"{identifier}: geometry axis length differs")
            distances, pair_mask = _capped_minimum_heavy_atom_distances(
                antibody_axis,
                [residue for _, residue in antigen_axis],
                distance_cap=distance_cap,
            )
            contact_path = contact_cache_root / contact_record["artifact_path"]
            authenticate_file(
                contact_path, contact_record["artifact_sha256"], "contact artifact"
            )
            with np.load(contact_path, allow_pickle=False) as contact_artifact:
                labels = np.zeros(distances.size, dtype=np.bool_)
                labels[contact_artifact["positive_flat_indices"].astype(np.int64)] = True
                labels = labels.reshape(distances.shape)
            try:
                distances, boundary_audit = _reconcile_contact_boundary(
                    distances,
                    pair_mask,
                    labels,
                    contact_cutoff=contact_cutoff,
                    coordinate_quantization_tolerance=(
                        coordinate_quantization_tolerance
                    ),
                )
            except StructureCacheError as error:
                raise StructureCacheError(f"{identifier}: {error}") from error
            expanded_cdr = np.asarray(cdr_mask, dtype=np.bool_)
            antigen_chains = _chain_names(metadata[identifier].get("agchains"))
            surface_mask = np.asarray(
                _antigen_surface_mask(
                    model,
                    antigen_axis,
                    antigen_chains,
                    relative_sasa_threshold=antigen_relative_sasa_threshold,
                ),
                dtype=np.bool_,
            )
            if not expanded_cdr.any() or not surface_mask.any():
                raise StructureCacheError(f"{identifier}: empty CDR or surface mask")
            digest = hashlib.sha256(identifier.encode("utf-8")).hexdigest()
            relative_path = Path("artifacts") / digest[:2] / f"{digest}.npz"
            target = temporary / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                target,
                format_version=np.asarray(PAIR_GEOMETRY_ARTIFACT_FORMAT),
                instance_id=np.asarray(identifier),
                pair_distances=distances,
                pair_distance_mask=pair_mask,
                antibody_cdr_mask=expanded_cdr,
                antigen_surface_mask=surface_mask,
            )
            observed_count = int(pair_mask.sum())
            censored_count = int((pair_mask & (distances > distance_cap)).sum())
            total_observed += observed_count
            total_censored += censored_count
            total_boundary_reconciliations += boundary_audit[
                "boundary_reconciliation_count"
            ]
            maximum_boundary_adjustment = max(
                maximum_boundary_adjustment,
                boundary_audit["maximum_boundary_adjustment_angstrom"],
            )
            records.append(
                {
                    "instance_id": identifier,
                    "artifact_path": relative_path.as_posix(),
                    "artifact_sha256": sha256_file(target),
                    "structure_artifact_sha256": structure_record["artifact_sha256"],
                    "contact_artifact_sha256": contact_record["artifact_sha256"],
                    "observed_pair_count": observed_count,
                    "right_censored_pair_count": censored_count,
                    "cdr_residue_count_with_flank": int(expanded_cdr.sum()),
                    "antigen_surface_residue_count": int(surface_mask.sum()),
                    **boundary_audit,
                }
            )
        manifest = {
            "format_version": PAIR_GEOMETRY_CACHE_FORMAT,
            "authority": {
                "structure_manifest_sha256": structure_manifest_sha256,
                "contact_manifest_sha256": contact_manifest_sha256,
                "metadata_sha256": metadata_sha256,
                "canonical_manifest_sha256": canonical_manifest_sha256,
                "source_cif_digest_checked_per_instance": True,
                "split_membership_read": selection_manifest_path is not None,
                "model_predictions_or_scores_read": False,
                "explicit_instance_selection": (
                    None if instance_ids is None else identifiers
                ),
                **selection_authority,
                "frozen_test_pair_labels_read": (
                    False if selection_manifest_path is not None else None
                ),
            },
            "definitions": {
                "distance": (
                    "minimum Euclidean distance over non-hydrogen atom pairs, "
                    "including every alternate-location conformer; only label-"
                    "discordant values within the registered cutoff uncertainty "
                    "band are snapped to the adjacent float on the label-consistent "
                    "side"
                ),
                "contact": f"distance < {contact_cutoff} Angstrom",
                "coordinate_quantization_tolerance_angstrom": (
                    coordinate_quantization_tolerance
                ),
                "coordinate_quantization_rationale": (
                    "0.002 Angstrom covers the sqrt(3)*0.001 Angstrom worst-case "
                    "distance error from rounding both atoms to 0.001 Angstrom"
                ),
                "distance_cap_angstrom": distance_cap,
                "right_censoring": "values above the cap share the final distogram bin",
                "cdr": (
                    "IMGT-numbered residues 27-38, 56-65, and 105-117, "
                    "authenticated against release CDR strings, plus within-chain flank"
                ),
                "cdr_flank_residues": cdr_flank,
                "surface": "isolated-antigen Shrake-Rupley relative SASA",
                "antigen_relative_sasa_threshold": antigen_relative_sasa_threshold,
            },
            "checks": {
                "contact_labels_exactly_reconstructed_for_every_observed_pair": True,
                "antibody_cdr_masks_nonempty": True,
                "antigen_surface_masks_nonempty": True,
            },
            "counts": {
                "records": len(records),
                "observed_pairs": total_observed,
                "right_censored_pairs": total_censored,
                "boundary_reconciliations": total_boundary_reconciliations,
                "maximum_boundary_adjustment_angstrom": (
                    maximum_boundary_adjustment
                ),
            },
            "records": records,
        }
        manifest_path = temporary / "PAIR_GEOMETRY_CACHE_MANIFEST.json"
        manifest_path.write_bytes(canonical_json_bytes(manifest))
        receipt = {
            "format_version": PAIR_GEOMETRY_RECEIPT_FORMAT,
            "manifest_sha256": sha256_file(manifest_path),
            "record_count": len(records),
            "test_predictions_or_scores_accessed": False,
            "claim_boundary": (
                "Split-independent geometry supervision materialization only; no model "
                "selection, test prediction, performance, or generalization claim."
            ),
        }
        (temporary / "PAIR_GEOMETRY_CACHE_RECEIPT.json").write_bytes(
            canonical_json_bytes(receipt)
        )
        os.replace(temporary, output_root)
        return {**receipt, "output_root": str(output_root)}
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _load_manifest(cache_root: Path, expected_sha256: str) -> dict[str, Any]:
    path = cache_root / "PAIR_GEOMETRY_CACHE_MANIFEST.json"
    authenticate_file(path, expected_sha256, "pair geometry manifest")
    payload = json.loads(path.read_text())
    if payload.get("format_version") != PAIR_GEOMETRY_CACHE_FORMAT:
        raise StructureCacheError("Unknown pair geometry cache format")
    if payload.get("checks", {}).get(
        "contact_labels_exactly_reconstructed_for_every_observed_pair"
    ) is not True:
        raise StructureCacheError("Pair geometry cache lacks its contact equivalence proof")
    return payload


class Sabdab2PairGeometryDataset(Dataset):
    """Attach authenticated distogram and sparse-triangle targets."""

    def __init__(
        self, base_dataset: Any, cache_root: Path, *, manifest_sha256: str
    ) -> None:
        _require_torch()
        self.base_dataset = base_dataset
        self.cache_root = cache_root
        manifest = _load_manifest(cache_root, manifest_sha256)
        records = {record["instance_id"]: record for record in manifest["records"]}
        base_ids = [record["instance_id"] for record in base_dataset.records]
        if any(identifier not in records for identifier in base_ids):
            raise StructureCacheError("Pair geometry cache lacks a selected instance")
        self.records = tuple(records[identifier] for identifier in base_ids)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import numpy as np

        torch_module = _require_torch()
        item = dict(self.base_dataset[index])
        record = self.records[index]
        if record["instance_id"] != item["instance_id"]:
            raise StructureCacheError("Pair geometry dataset order differs")
        path = self.cache_root / record["artifact_path"]
        authenticate_file(path, record["artifact_sha256"], "pair geometry artifact")
        with np.load(path, allow_pickle=False) as artifact:
            if (
                str(artifact["format_version"].item())
                != PAIR_GEOMETRY_ARTIFACT_FORMAT
                or str(artifact["instance_id"].item()) != item["instance_id"]
            ):
                raise StructureCacheError("Pair geometry artifact metadata differs")
            item["pair_distances"] = torch_module.from_numpy(
                artifact["pair_distances"].astype(np.float32, copy=True)
            )
            item["pair_distance_mask"] = torch_module.from_numpy(
                artifact["pair_distance_mask"].astype(np.bool_, copy=True)
            )
            item["antibody_cdr_mask"] = torch_module.from_numpy(
                artifact["antibody_cdr_mask"].astype(np.bool_, copy=True)
            )
            item["antigen_surface_mask"] = torch_module.from_numpy(
                artifact["antigen_surface_mask"].astype(np.bool_, copy=True)
            )
        if tuple(item["pair_distances"].shape) != tuple(item["pair_labels"].shape):
            raise StructureCacheError("Pair geometry and contact axes differ")
        return item


class Sabdab2PairGeometryBatchCollator:
    def __init__(self) -> None:
        self.base = Sabdab2ContactBatchCollator()

    def __call__(self, examples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        torch_module = _require_torch()
        batch = self.base(examples)
        batch_size, max_antibody, max_antigen = batch["pair_labels"].shape
        pair_distances = torch_module.zeros(
            batch_size, max_antibody, max_antigen, dtype=torch_module.float32
        )
        pair_distance_mask = torch_module.zeros_like(
            pair_distances, dtype=torch_module.bool
        )
        antibody_cdr_mask = torch_module.zeros_like(
            batch["antibody_mask"], dtype=torch_module.bool
        )
        antigen_surface_mask = torch_module.zeros_like(
            batch["antigen_mask"], dtype=torch_module.bool
        )
        for batch_index, item in enumerate(examples):
            antibody_length, antigen_length = item["pair_labels"].shape
            if (
                tuple(item["pair_distances"].shape)
                != (antibody_length, antigen_length)
                or tuple(item["pair_distance_mask"].shape)
                != (antibody_length, antigen_length)
                or tuple(item["antibody_cdr_mask"].shape) != (antibody_length,)
                or tuple(item["antigen_surface_mask"].shape) != (antigen_length,)
            ):
                raise StructureCacheError("Pair geometry batch axes differ")
            pair_distances[batch_index, :antibody_length, :antigen_length] = item[
                "pair_distances"
            ]
            pair_distance_mask[batch_index, :antibody_length, :antigen_length] = item[
                "pair_distance_mask"
            ]
            antibody_cdr_mask[batch_index, :antibody_length] = item[
                "antibody_cdr_mask"
            ]
            antigen_surface_mask[batch_index, :antigen_length] = item[
                "antigen_surface_mask"
            ]
        batch["pair_distances"] = pair_distances
        batch["pair_distance_mask"] = pair_distance_mask
        batch["antibody_cdr_mask"] = antibody_cdr_mask
        batch["antigen_surface_mask"] = antigen_surface_mask
        return batch


__all__ = [
    "PAIR_GEOMETRY_ARTIFACT_FORMAT",
    "PAIR_GEOMETRY_CACHE_FORMAT",
    "PAIR_GEOMETRY_RECEIPT_FORMAT",
    "PAIR_GEOMETRY_SELECTION_FORMAT",
    "Sabdab2PairGeometryBatchCollator",
    "Sabdab2PairGeometryDataset",
    "freeze_pair_geometry_development_selection",
    "materialize_pair_geometry_cache",
]
