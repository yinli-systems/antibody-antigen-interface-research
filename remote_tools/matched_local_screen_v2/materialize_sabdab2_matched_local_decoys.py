#!/usr/bin/env python3
"""Materialize development-only matched-local antigen hard decoys.

The artifact is training supervision, not an evaluation result.  It uses true
contact labels only for instances whose frozen split role is ``train``.  Each
negative motif keeps the antibody CDR patch fixed and replaces the cognate
antigen surface patch with a non-epitope patch from the same antigen.  Matching
uses monomer-antigen SASA, residue-surface geodesics, a local curvature proxy,
amino-acid composition, and distance-to-epitope strata.
"""

from __future__ import annotations

import argparse
import copy
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import tarfile
import tempfile
from typing import Any, Iterable, Mapping

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(SOURCE_ROOT))

from interaction_first.sabdab2_partner_metadata import (  # noqa: E402
    SASA_POINTS_PER_ATOM,
    SASA_PROBE_RADIUS_ANGSTROM,
    TIEN_2013_THEORETICAL_MAX_ASA,
)
from interaction_first.sabdab2_structure import (  # noqa: E402
    StructureCacheError,
    _bio_parser,
    _chain_lookup,
    _chain_names,
    _full_chain_specs,
    _parse_numbering_list,
    _residue_lookup,
    _split_antigen_specs,
    canonical_json_bytes,
    sha256_file,
)


FORMAT_VERSION = "sabdab2_matched_local_decoy_manifest_v4"
ARTIFACT_FORMAT = "sabdab2_matched_local_decoy_artifact_v4"
RECEIPT_FORMAT = "sabdab2_matched_local_decoy_receipt_v4"
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AMINO_ACID_INDEX = {value: index for index, value in enumerate(AMINO_ACIDS)}
COMPOSITION_GROUPS = {
    "hydrophobic_aliphatic": "AILMV",
    "aromatic": "FWY",
    "polar_uncharged": "CNQST",
    "positive": "HKR",
    "negative": "DE",
    "conformational": "GP",
}
AMINO_ACID_GROUP_INDEX = {
    amino_acid: group_index
    for group_index, members in enumerate(COMPOSITION_GROUPS.values())
    for amino_acid in members
}
IMGT_CDR_INTERVALS = ((27, 38), (56, 65), (105, 117))


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _authenticate(path: Path, expected: str, label: str) -> str:
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"{label} SHA-256 differs: {observed}")
    return observed


def _csv_rows(payload: bytes) -> list[dict[str, str]]:
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8", errors="strict")))
    if reader.fieldnames is None:
        raise ValueError("CSV has no header")
    return [
        {str(key): "" if value is None else str(value).strip() for key, value in row.items()}
        for row in reader
    ]


def _selected_train_ids(split_path: Path) -> tuple[str, ...]:
    with split_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    identifiers = [
        str(row.get("instance_id", "")).strip()
        for row in rows
        if str(row.get("role", "")).strip() == "train"
        and str(row.get("screen_included", "true")).strip().casefold() != "false"
    ]
    if not identifiers or len(identifiers) != len(set(identifiers)):
        raise ValueError("Frozen train selection is empty or contains duplicates")
    return tuple(sorted(identifiers))


def _canonical_authority(
    archive_path: Path, selected: set[str]
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, Any]], dict[str, str]]:
    with tarfile.open(archive_path, "r:gz") as archive:
        manifest_payload = archive.extractfile("./manifest.csv").read()
        rows = _csv_rows(manifest_payload)
        by_id = {row["INSTANCE"]: row for row in rows if row.get("INSTANCE") in selected}
        mappings = {}
        mapping_hashes = {}
        for identifier in selected:
            member = f"./mapping/{identifier}.json"
            payload = archive.extractfile(member).read()
            mappings[identifier] = json.loads(payload)
            mapping_hashes[identifier] = _sha256_bytes(payload)
    if set(by_id) != selected or set(mappings) != selected:
        raise ValueError("Canonical archive lacks selected training instances")
    return by_id, mappings, {
        "manifest_sha256": _sha256_bytes(manifest_payload),
        "selected_mapping_sha256": hashlib.sha256(
            "".join(f"{key}\0{mapping_hashes[key]}\n" for key in sorted(mapping_hashes)).encode()
        ).hexdigest(),
    }


def _metadata_authority(archive_path: Path) -> tuple[dict[str, dict[str, str]], str]:
    with tarfile.open(archive_path, "r:gz") as archive:
        payload = archive.extractfile("splits_final/abag_split.csv").read()
    rows = _csv_rows(payload)
    by_id: dict[str, dict[str, str]] = {}
    for row in rows:
        identifier = row.get("INSTANCE", "")
        if not identifier or identifier in by_id:
            raise ValueError("Raw metadata INSTANCE values must be unique")
        by_id[identifier] = row
    return by_id, _sha256_bytes(payload)


def _manifest_index(root: Path, name: str, expected_sha256: str) -> dict[str, dict[str, Any]]:
    path = root / name
    _authenticate(path, expected_sha256, name)
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError(f"{name} has no records")
    index = {str(record["instance_id"]): record for record in records}
    if len(index) != len(records):
        raise ValueError(f"{name} has duplicate identifiers")
    return index


def _imgt_number(value: Any) -> int | None:
    match = re.match(r"^\s*(\d+)", str(value))
    return None if match is None else int(match.group(1))


def _cdr_mask(mapping: Mapping[str, Any], antibody_length: int) -> np.ndarray:
    positions = list(mapping.get("vh_imgt_positions", [])) + list(
        mapping.get("vl_imgt_positions", [])
    )
    if len(positions) != antibody_length:
        raise StructureCacheError("Canonical IMGT positions do not match antibody axis")
    numbers = [_imgt_number(value) for value in positions]
    return np.asarray(
        [
            number is not None
            and any(left <= number <= right for left, right in IMGT_CDR_INTERVALS)
            for number in numbers
        ],
        dtype=bool,
    )


def _antigen_sasa_profile(
    identifier: str,
    metadata: Mapping[str, Any],
    cif_payload: bytes,
    expected_sequence: str,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        from Bio.PDB.Chain import Chain
        from Bio.PDB.Model import Model
        from Bio.PDB.SASA import ShrakeRupley
    except ModuleNotFoundError as error:  # pragma: no cover
        raise RuntimeError("Biopython is required for matched-local SASA") from error

    structure = _bio_parser().get_structure(
        identifier, io.StringIO(cif_payload.decode("utf-8", errors="strict"))
    )
    source_model = next(structure.get_models())
    antigen_model = Model(0)
    chain_names = _chain_names(metadata.get("agchains"))
    raw_specs = _split_antigen_specs(
        str(metadata.get("ag_index_list", "")), len(chain_names)
    )
    if not chain_names:
        raise StructureCacheError("No antigen chains are declared")
    ordered: list[tuple[str, Any | None]] = []
    sequence: list[str] = []
    for chain_name, raw_spec in zip(chain_names, raw_specs):
        source_chain = _chain_lookup(source_model, chain_name)
        specs = _parse_numbering_list(raw_spec)
        if specs is None:
            specs = _full_chain_specs(source_chain)
        lookup = _residue_lookup(source_chain)
        antigen_chain = Chain(chain_name)
        antigen_model.add(antigen_chain)
        for spec in specs:
            sequence.append(spec.amino_acid)
            residue = lookup.get((spec.number, spec.insertion_code))
            if residue is None:
                ordered.append((spec.amino_acid, None))
                continue
            copied = copy.deepcopy(residue)
            antigen_chain.add(copied)
            ordered.append((spec.amino_acid, copied))
    if "".join(sequence) != expected_sequence:
        raise StructureCacheError(f"{identifier}: antigen sequence axis differs")
    ShrakeRupley(
        probe_radius=SASA_PROBE_RADIUS_ANGSTROM,
        n_points=SASA_POINTS_PER_ATOM,
    ).compute(antigen_model, level="R")
    sasa = np.full(len(ordered), np.nan, dtype=np.float32)
    rsa = np.full(len(ordered), np.nan, dtype=np.float32)
    for index, (amino_acid, residue) in enumerate(ordered):
        if residue is None:
            continue
        maximum = TIEN_2013_THEORETICAL_MAX_ASA.get(amino_acid)
        if maximum is None:
            raise StructureCacheError(f"No MaxASA for residue {amino_acid!r}")
        sasa[index] = float(residue.sasa)
        rsa[index] = float(residue.sasa) / float(maximum)
    return sasa, rsa


def _surface_geodesics(
    coordinates: np.ndarray,
    chain_ids: np.ndarray,
    surface_mask: np.ndarray,
    *,
    edge_radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import dijkstra
    except ModuleNotFoundError as error:  # pragma: no cover
        raise RuntimeError("SciPy is required for residue-surface geodesics") from error
    nodes = np.flatnonzero(surface_mask)
    if len(nodes) < 2:
        raise StructureCacheError("Fewer than two observed antigen surface residues")
    xyz = coordinates[nodes]
    pairwise = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
    adjacency = (pairwise <= edge_radius) & (pairwise > 0)
    local_by_global = {int(value): index for index, value in enumerate(nodes)}
    for global_index in nodes:
        for neighbor in (int(global_index) - 1, int(global_index) + 1):
            if (
                neighbor in local_by_global
                and chain_ids[neighbor] == chain_ids[global_index]
            ):
                left = local_by_global[int(global_index)]
                right = local_by_global[neighbor]
                adjacency[left, right] = True
                adjacency[right, left] = True
    weights = np.where(adjacency, pairwise, 0.0)
    distances = dijkstra(csr_matrix(weights), directed=False, return_predecessors=False)
    return nodes.astype(np.int64), np.asarray(distances, dtype=np.float32)


def _patch_features(
    center_local: int,
    nodes: np.ndarray,
    geodesics: np.ndarray,
    coordinates: np.ndarray,
    sequence: str,
    sasa: np.ndarray,
    rsa: np.ndarray,
    *,
    radius: float,
) -> dict[str, Any]:
    members = nodes[geodesics[center_local] <= radius]
    if len(members) == 0:
        raise StructureCacheError("Surface patch is empty")
    xyz = coordinates[members]
    centered = xyz - xyz.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(1, len(members) - 1)
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
    curvature = float(eigenvalues[0] / max(float(eigenvalues.sum()), 1e-8))
    composition = np.zeros(len(COMPOSITION_GROUPS), dtype=np.float32)
    for index in members:
        aa_index = AMINO_ACID_GROUP_INDEX.get(sequence[int(index)])
        if aa_index is not None:
            composition[aa_index] += 1.0
    composition /= max(float(composition.sum()), 1.0)
    center = int(nodes[center_local])
    return {
        "center": center,
        "members": members.astype(np.int64),
        "count": int(len(members)),
        "surface_area": float(np.nansum(sasa[members])),
        "mean_sasa": float(np.nanmean(sasa[members])),
        "center_sasa": float(sasa[center]),
        "center_rsa": float(rsa[center]),
        "curvature": curvature,
        "composition": composition,
    }


def _jensen_shannon(left: np.ndarray, right: np.ndarray) -> float:
    midpoint = 0.5 * (left + right)
    left_mask = left > 0
    right_mask = right > 0
    divergence = 0.5 * float(
        np.sum(left[left_mask] * np.log(left[left_mask] / midpoint[left_mask]))
    ) + 0.5 * float(
        np.sum(right[right_mask] * np.log(right[right_mask] / midpoint[right_mask]))
    )
    return math.sqrt(max(divergence, 0.0))


def _relative_difference(left: float, right: float) -> float:
    return abs(left - right) / max(abs(left), abs(right), 1e-8)


def _distance_bin(value: float, boundaries: tuple[float, ...]) -> int:
    return int(np.searchsorted(np.asarray(boundaries), value, side="right"))


def _matching_score(
    positive: Mapping[str, Any], negative: Mapping[str, Any]
) -> tuple[float, dict[str, float]]:
    diagnostics = {
        "count_relative_difference": _relative_difference(
            float(positive["count"]), float(negative["count"])
        ),
        "surface_area_relative_difference": _relative_difference(
            float(positive["surface_area"]), float(negative["surface_area"])
        ),
        "mean_sasa_relative_difference": _relative_difference(
            float(positive["mean_sasa"]), float(negative["mean_sasa"])
        ),
        "curvature_absolute_difference": abs(
            float(positive["curvature"]) - float(negative["curvature"])
        ),
        "composition_group_l1_distance": float(
            np.linalg.norm(
                np.asarray(positive["composition"])
                - np.asarray(negative["composition"]),
                ord=1,
            )
        ),
    }
    score = (
        diagnostics["count_relative_difference"]
        + diagnostics["surface_area_relative_difference"]
        + diagnostics["mean_sasa_relative_difference"]
        + diagnostics["curvature_absolute_difference"]
        + diagnostics["composition_group_l1_distance"]
    )
    return float(score), diagnostics


def _passes_match(diagnostics: Mapping[str, float], thresholds: Mapping[str, float]) -> bool:
    return all(float(diagnostics[key]) <= float(value) for key, value in thresholds.items())


def _cross_product_flat(
    antibody_indices: np.ndarray, antigen_indices: np.ndarray, antigen_length: int
) -> np.ndarray:
    return (
        antibody_indices[:, None] * int(antigen_length) + antigen_indices[None, :]
    ).reshape(-1).astype(np.int64)


def _load_positive_flat(contact_path: Path, expected_sha256: str) -> np.ndarray:
    _authenticate(contact_path, expected_sha256, "contact artifact")
    with np.load(contact_path, allow_pickle=False) as payload:
        return payload["positive_flat_indices"].astype(np.int64, copy=True)


def _worker(
    *,
    identifier: str,
    metadata: Mapping[str, Any],
    canonical: Mapping[str, str],
    mapping: Mapping[str, Any],
    cif_payload: bytes,
    structure_path: str,
    structure_sha256: str,
    contact_path: str,
    contact_sha256: str,
    output_path: str,
    configuration: Mapping[str, Any],
) -> dict[str, Any]:
    import torch

    structure_file = Path(structure_path)
    _authenticate(structure_file, structure_sha256, "structure artifact")
    structure = torch.load(
        structure_file, map_location="cpu", weights_only=False
    )
    antibody_backbone = structure["antibody_backbone"].numpy()
    antigen_backbone = structure["antigen_backbone"].numpy()
    antibody_valid = structure["antibody_frame_valid_mask"].numpy().astype(bool)
    antigen_valid = structure["antigen_frame_valid_mask"].numpy().astype(bool)
    antibody_chain_ids = structure["antibody_chain_ids"].numpy().astype(np.int64)
    antigen_chain_ids = structure["antigen_chain_ids"].numpy().astype(np.int64)
    antibody_coordinates = antibody_backbone[:, 1, :]
    antigen_coordinates = antigen_backbone[:, 1, :]
    antibody_length = len(antibody_coordinates)
    antigen_length = len(antigen_coordinates)
    if antibody_length != int(canonical["vh_len"]) + int(canonical["vl_len"]):
        raise StructureCacheError(f"{identifier}: canonical antibody length differs")
    sequence = canonical["ag_seq"]
    if len(sequence) != antigen_length:
        raise StructureCacheError(f"{identifier}: canonical antigen length differs")
    sasa, rsa = _antigen_sasa_profile(identifier, metadata, cif_payload, sequence)
    if len(sasa) != antigen_length:
        raise StructureCacheError(f"{identifier}: SASA axis differs")
    cdr_mask = _cdr_mask(mapping, antibody_length) & antibody_valid
    surface_mask = (
        antigen_valid
        & np.isfinite(sasa)
        & np.isfinite(rsa)
        & (rsa >= float(configuration["rsa_threshold"]))
    )
    cdr_nodes, cdr_geodesics = _surface_geodesics(
        antibody_coordinates,
        antibody_chain_ids,
        cdr_mask,
        edge_radius=float(configuration["cdr_edge_radius_angstrom"]),
    )
    cdr_local_by_global = {
        int(value): index for index, value in enumerate(cdr_nodes)
    }
    cdr_patch_cache = {
        int(center): cdr_nodes[
            cdr_geodesics[local]
            <= float(configuration["cdr_patch_radius_angstrom"])
        ].astype(np.int64)
        for local, center in enumerate(cdr_nodes)
    }
    surface_nodes, surface_geodesics = _surface_geodesics(
        antigen_coordinates,
        antigen_chain_ids,
        surface_mask,
        edge_radius=float(configuration["surface_edge_radius_angstrom"]),
    )
    local_by_global = {int(value): index for index, value in enumerate(surface_nodes)}
    positive_flat = _load_positive_flat(Path(contact_path), contact_sha256)
    positive_antibody = positive_flat // antigen_length
    positive_antigen = positive_flat % antigen_length
    epitope_mask = np.zeros(antigen_length, dtype=bool)
    epitope_mask[np.unique(positive_antigen)] = True
    epitope_surface_local = [
        local_by_global[int(value)]
        for value in np.flatnonzero(epitope_mask & surface_mask)
    ]
    if not epitope_surface_local:
        raise StructureCacheError(f"{identifier}: no epitope center lies on the surface")
    distance_to_epitope = np.min(surface_geodesics[epitope_surface_local], axis=0)
    patch_cache = {
        int(center): _patch_features(
            local,
            surface_nodes,
            surface_geodesics,
            antigen_coordinates,
            sequence,
            sasa,
            rsa,
            radius=float(configuration["antigen_patch_radius_angstrom"]),
        )
        for local, center in enumerate(surface_nodes)
    }
    negative_centers = []
    for local, center in enumerate(surface_nodes):
        center = int(center)
        if epitope_mask[center] or not math.isfinite(float(distance_to_epitope[local])):
            continue
        patch = patch_cache[center]
        if bool(epitope_mask[patch["members"]].any()):
            continue
        negative_centers.append((center, float(distance_to_epitope[local])))

    thresholds = configuration["matching_thresholds"]
    boundaries = tuple(float(value) for value in configuration["distance_bin_boundaries_angstrom"])
    candidates_by_positive: dict[int, list[dict[str, Any]]] = {}
    nearest_unfiltered_by_positive: dict[int, dict[str, Any]] = {}
    for positive_center in sorted(set(int(value) for value in positive_antigen)):
        if positive_center not in patch_cache:
            continue
        positive_patch = patch_cache[positive_center]
        candidates = []
        unfiltered_candidates = []
        for negative_center, geodesic_distance in negative_centers:
            negative_patch = patch_cache[negative_center]
            score, diagnostics = _matching_score(positive_patch, negative_patch)
            candidate = {
                "negative_center": negative_center,
                "distance_bin": _distance_bin(geodesic_distance, boundaries),
                "distance_to_epitope": geodesic_distance,
                "score": score,
                "diagnostics": diagnostics,
            }
            unfiltered_candidates.append(candidate)
            if not _passes_match(diagnostics, thresholds):
                continue
            candidates.append(candidate)
        unfiltered_candidates.sort(
            key=lambda value: (
                value["score"], value["distance_bin"], value["negative_center"]
            )
        )
        if unfiltered_candidates:
            nearest_unfiltered_by_positive[positive_center] = unfiltered_candidates[0]
        candidates.sort(
            key=lambda value: (
                value["score"], value["distance_bin"], value["negative_center"]
            )
        )
        selected = []
        used_bins = set()
        for candidate in candidates:
            if candidate["distance_bin"] in used_bins:
                continue
            selected.append(candidate)
            used_bins.add(candidate["distance_bin"])
            if len(selected) >= int(configuration["max_decoys_per_positive"]):
                break
        if len(selected) < int(configuration["max_decoys_per_positive"]):
            for candidate in candidates:
                if candidate in selected:
                    continue
                selected.append(candidate)
                if len(selected) >= int(configuration["max_decoys_per_positive"]):
                    break
        candidates_by_positive[positive_center] = selected

    relations = []
    eligible_pairs = 0
    covered_pairs = 0
    for antibody_center, antigen_center, flat_index in zip(
        positive_antibody, positive_antigen, positive_flat
    ):
        antibody_center = int(antibody_center)
        antigen_center = int(antigen_center)
        if not cdr_mask[antibody_center] or antigen_center not in patch_cache:
            continue
        eligible_pairs += 1
        selected = candidates_by_positive.get(antigen_center, [])
        if not selected:
            continue
        covered_pairs += 1
        if antibody_center not in cdr_local_by_global:
            continue
        antibody_patch = cdr_patch_cache[antibody_center]
        positive_patch = patch_cache[antigen_center]
        for candidate in selected:
            negative_patch = patch_cache[int(candidate["negative_center"])]
            relation = {
                "positive_center_flat": int(flat_index),
                "negative_center_flat": antibody_center * antigen_length
                + int(candidate["negative_center"]),
                "positive_pairs": _cross_product_flat(
                    antibody_patch, positive_patch["members"], antigen_length
                ),
                "negative_pairs": _cross_product_flat(
                    antibody_patch, negative_patch["members"], antigen_length
                ),
                **candidate,
            }
            relation["tie_break"] = hashlib.sha256(
                f"{identifier}\0{flat_index}\0{candidate['negative_center']}".encode()
            ).hexdigest()
            relations.append(relation)
    relations.sort(key=lambda value: (value["tie_break"], value["score"]))
    relations = relations[: int(configuration["max_decoys_per_complex"])]

    positive_offsets = [0]
    negative_offsets = [0]
    positive_values: list[np.ndarray] = []
    negative_values: list[np.ndarray] = []
    for relation in relations:
        positive_values.append(relation["positive_pairs"])
        negative_values.append(relation["negative_pairs"])
        positive_offsets.append(positive_offsets[-1] + len(relation["positive_pairs"]))
        negative_offsets.append(negative_offsets[-1] + len(relation["negative_pairs"]))
    diagnostic_keys = tuple(configuration["matching_thresholds"])
    nearest_positive_centers = sorted(nearest_unfiltered_by_positive)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        format_version=np.asarray(ARTIFACT_FORMAT),
        instance_id=np.asarray(identifier),
        antibody_length=np.asarray(antibody_length, dtype=np.int64),
        antigen_length=np.asarray(antigen_length, dtype=np.int64),
        decoy_count=np.asarray(len(relations), dtype=np.int64),
        positive_center_flat_indices=np.asarray(
            [value["positive_center_flat"] for value in relations], dtype=np.int64
        ),
        negative_center_flat_indices=np.asarray(
            [value["negative_center_flat"] for value in relations], dtype=np.int64
        ),
        positive_motif_offsets=np.asarray(positive_offsets, dtype=np.int64),
        positive_motif_flat_indices=(
            np.concatenate(positive_values) if positive_values else np.asarray([], dtype=np.int64)
        ),
        negative_motif_offsets=np.asarray(negative_offsets, dtype=np.int64),
        negative_motif_flat_indices=(
            np.concatenate(negative_values) if negative_values else np.asarray([], dtype=np.int64)
        ),
        distance_bins=np.asarray(
            [value["distance_bin"] for value in relations], dtype=np.int64
        ),
        distance_to_epitope_angstrom=np.asarray(
            [value["distance_to_epitope"] for value in relations], dtype=np.float32
        ),
        matching_scores=np.asarray(
            [value["score"] for value in relations], dtype=np.float32
        ),
        matching_diagnostics=np.asarray(
            [
                [value["diagnostics"][key] for key in diagnostic_keys]
                for value in relations
            ],
            dtype=np.float32,
        ).reshape(len(relations), len(diagnostic_keys)),
        nearest_unfiltered_positive_antigen_indices=np.asarray(
            nearest_positive_centers, dtype=np.int64
        ),
        nearest_unfiltered_negative_antigen_indices=np.asarray(
            [
                nearest_unfiltered_by_positive[value]["negative_center"]
                for value in nearest_positive_centers
            ],
            dtype=np.int64,
        ),
        nearest_unfiltered_matching_scores=np.asarray(
            [
                nearest_unfiltered_by_positive[value]["score"]
                for value in nearest_positive_centers
            ],
            dtype=np.float32,
        ),
        nearest_unfiltered_matching_diagnostics=np.asarray(
            [
                [
                    nearest_unfiltered_by_positive[value]["diagnostics"][key]
                    for key in diagnostic_keys
                ]
                for value in nearest_positive_centers
            ],
            dtype=np.float32,
        ).reshape(len(nearest_positive_centers), len(diagnostic_keys)),
    )
    return {
        "instance_id": identifier,
        "artifact_sha256": sha256_file(target),
        "artifact_size_bytes": target.stat().st_size,
        "decoy_count": len(relations),
        "true_contact_pair_count": int(len(positive_flat)),
        "eligible_cdr_surface_contact_pair_count": eligible_pairs,
        "covered_cdr_surface_contact_pair_count": covered_pairs,
        "surface_residue_count": int(surface_mask.sum()),
        "epitope_surface_residue_count": len(epitope_surface_local),
        "native_non_epitope_patch_center_count": len(negative_centers),
    }


def materialize(arguments: argparse.Namespace) -> dict[str, Any]:
    authority = {
        "raw_archive_sha256": _authenticate(
            arguments.raw_archive, arguments.raw_archive_sha256, "raw archive"
        ),
        "canonical_archive_sha256": _authenticate(
            arguments.canonical_archive,
            arguments.canonical_archive_sha256,
            "canonical archive",
        ),
        "split_sha256": _authenticate(arguments.split, arguments.split_sha256, "split"),
        "contact_manifest_sha256": _authenticate(
            arguments.contact_root / "CONTACT_CACHE_MANIFEST.json",
            arguments.contact_manifest_sha256,
            "contact manifest",
        ),
        "structure_manifest_sha256": _authenticate(
            arguments.structure_root / "STRUCTURE_CACHE_MANIFEST.json",
            arguments.structure_manifest_sha256,
            "structure manifest",
        ),
        "task_labels_read_for_roles": ["train"],
        "validation_labels_read": False,
        "test_labels_read": False,
        "model_predictions_read": False,
    }
    selected_ids = _selected_train_ids(arguments.split)
    selected = set(selected_ids)
    canonical, mappings, canonical_authority = _canonical_authority(
        arguments.canonical_archive, selected
    )
    metadata, metadata_sha256 = _metadata_authority(arguments.raw_archive)
    if metadata_sha256 != arguments.metadata_sha256:
        raise ValueError("Raw metadata member SHA-256 differs")
    if not selected.issubset(metadata):
        raise ValueError("Raw metadata lacks selected training instances")
    contact_records = _manifest_index(
        arguments.contact_root,
        "CONTACT_CACHE_MANIFEST.json",
        arguments.contact_manifest_sha256,
    )
    structure_records = _manifest_index(
        arguments.structure_root,
        "STRUCTURE_CACHE_MANIFEST.json",
        arguments.structure_manifest_sha256,
    )
    if not selected.issubset(contact_records) or not selected.issubset(structure_records):
        raise ValueError("Training selection is absent from structure/contact caches")
    if arguments.output_root.exists() or arguments.output_root.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {arguments.output_root}")
    arguments.output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{arguments.output_root.name}.", dir=arguments.output_root.parent
        )
    )
    configuration = {
        "rsa_threshold": arguments.rsa_threshold,
        "surface_edge_radius_angstrom": arguments.surface_edge_radius,
        "antigen_patch_radius_angstrom": arguments.antigen_patch_radius,
        "cdr_edge_radius_angstrom": arguments.cdr_edge_radius,
        "cdr_patch_radius_angstrom": arguments.cdr_patch_radius,
        "distance_bin_boundaries_angstrom": list(arguments.distance_bins),
        "composition_groups": COMPOSITION_GROUPS,
        "max_decoys_per_positive": arguments.max_decoys_per_positive,
        "max_decoys_per_complex": arguments.max_decoys_per_complex,
        "matching_thresholds": {
            "count_relative_difference": arguments.max_count_relative_difference,
            "surface_area_relative_difference": arguments.max_surface_area_relative_difference,
            "mean_sasa_relative_difference": arguments.max_mean_sasa_relative_difference,
            "curvature_absolute_difference": arguments.max_curvature_absolute_difference,
            "composition_group_l1_distance": (
                arguments.max_composition_group_l1_distance
            ),
        },
    }
    executor = ProcessPoolExecutor(max_workers=arguments.workers)
    pending: dict[Any, str] = {}
    records: list[dict[str, Any]] = []
    encountered: set[str] = set()

    def collect_one() -> None:
        completed, _ = wait(pending, return_when=FIRST_COMPLETED)
        for future in completed:
            identifier = pending.pop(future)
            try:
                records.append(future.result())
            except BaseException as error:
                raise RuntimeError(f"Matched-local worker failed for {identifier}") from error

    try:
        with tarfile.open(arguments.raw_archive, "r|gz") as archive:
            for member in archive:
                if not member.isfile() or not member.name.endswith(".cif"):
                    continue
                identifier = Path(member.name).stem
                if identifier not in selected:
                    continue
                binary = archive.extractfile(member)
                if binary is None:
                    raise ValueError(f"Cannot extract {member.name}")
                encountered.add(identifier)
                structure_record = structure_records[identifier]
                contact_record = contact_records[identifier]
                identifier_sha = hashlib.sha256(identifier.encode()).hexdigest()
                relative = Path("artifacts") / identifier_sha[:2] / f"{identifier_sha}.npz"
                future = executor.submit(
                    _worker,
                    identifier=identifier,
                    metadata=metadata[identifier],
                    canonical=canonical[identifier],
                    mapping=mappings[identifier],
                    cif_payload=binary.read(),
                    structure_path=str(
                        arguments.structure_root / structure_record["artifact_path"]
                    ),
                    structure_sha256=structure_record["artifact_sha256"],
                    contact_path=str(arguments.contact_root / contact_record["artifact_path"]),
                    contact_sha256=contact_record["artifact_sha256"],
                    output_path=str(temporary / relative),
                    configuration=configuration,
                )
                pending[future] = identifier
                if len(pending) >= arguments.workers * 2:
                    collect_one()
        while pending:
            collect_one()
        if encountered != selected:
            missing = sorted(selected - encountered)
            raise ValueError(f"Raw archive lacks selected instances: {missing[:5]}")
        records.sort(key=lambda value: value["instance_id"])
        for record in records:
            identifier_sha = hashlib.sha256(record["instance_id"].encode()).hexdigest()
            record["artifact_path"] = (
                Path("artifacts") / identifier_sha[:2] / f"{identifier_sha}.npz"
            ).as_posix()
        eligible = sum(value["eligible_cdr_surface_contact_pair_count"] for value in records)
        covered = sum(value["covered_cdr_surface_contact_pair_count"] for value in records)
        decoy_counts = np.asarray([value["decoy_count"] for value in records], dtype=float)
        native_counts = np.asarray(
            [value["native_non_epitope_patch_center_count"] for value in records],
            dtype=float,
        )
        manifest = {
            "format_version": FORMAT_VERSION,
            "authority": {
                **authority,
                "raw_metadata_member_sha256": metadata_sha256,
                **canonical_authority,
            },
            "configuration": configuration,
            "counts": {
                "train_instances": len(records),
                "instances_with_decoys": int((decoy_counts > 0).sum()),
                "eligible_cdr_surface_contact_pairs": eligible,
                "covered_cdr_surface_contact_pairs": covered,
                "strict_pair_coverage": None if eligible == 0 else covered / eligible,
                "decoys": int(decoy_counts.sum()),
                "median_decoys_per_instance": float(np.median(decoy_counts)),
                "median_native_non_epitope_patch_centers": float(np.median(native_counts)),
            },
            "diagnostic_columns": list(configuration["matching_thresholds"]),
            "records": records,
            "claim_boundary": (
                "Development-training supervision only. The curvature field is a local "
                "residue-surface PCA proxy, not a molecular-mesh curvature. Validation "
                "and test labels, model predictions, and performance outcomes were not read."
            ),
        }
        manifest_path = temporary / "MATCHED_LOCAL_DECOY_MANIFEST.json"
        manifest_path.write_bytes(canonical_json_bytes(manifest))
        receipt = {
            "format_version": RECEIPT_FORMAT,
            "manifest_sha256": sha256_file(manifest_path),
            "train_instance_count": len(records),
            "strict_pair_coverage": manifest["counts"]["strict_pair_coverage"],
            "minimum_strict_pair_coverage": arguments.minimum_strict_pair_coverage,
            "status": (
                "PASS"
                if eligible > 0
                and covered / eligible >= arguments.minimum_strict_pair_coverage
                else "FAIL_INSUFFICIENT_COVERAGE"
            ),
            "validation_labels_read": False,
            "test_labels_read": False,
            "model_predictions_read": False,
        }
        (temporary / "MATCHED_LOCAL_DECOY_RECEIPT.json").write_bytes(
            canonical_json_bytes(receipt)
        )
        os.replace(temporary, arguments.output_root)
        if receipt["status"] != "PASS":
            raise RuntimeError(
                "Matched-local strict pair coverage is below the frozen minimum: "
                f"{manifest['counts']['strict_pair_coverage']} < "
                f"{arguments.minimum_strict_pair_coverage}"
            )
        return {**receipt, "output_root": str(arguments.output_root)}
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-archive", type=Path, required=True)
    parser.add_argument("--raw-archive-sha256", required=True)
    parser.add_argument("--metadata-sha256", required=True)
    parser.add_argument("--canonical-archive", type=Path, required=True)
    parser.add_argument("--canonical-archive-sha256", required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--split-sha256", required=True)
    parser.add_argument("--contact-root", type=Path, required=True)
    parser.add_argument("--contact-manifest-sha256", required=True)
    parser.add_argument("--structure-root", type=Path, required=True)
    parser.add_argument("--structure-manifest-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--rsa-threshold", type=float, default=0.20)
    parser.add_argument("--surface-edge-radius", type=float, default=8.0)
    parser.add_argument("--antigen-patch-radius", type=float, default=12.0)
    parser.add_argument("--cdr-edge-radius", type=float, default=10.0)
    parser.add_argument("--cdr-patch-radius", type=float, default=8.0)
    parser.add_argument("--distance-bins", type=float, nargs="+", default=(8, 16, 24, 32))
    parser.add_argument("--max-decoys-per-positive", type=int, default=4)
    parser.add_argument("--max-decoys-per-complex", type=int, default=32)
    parser.add_argument("--max-count-relative-difference", type=float, default=0.30)
    parser.add_argument("--max-surface-area-relative-difference", type=float, default=0.30)
    parser.add_argument("--max-mean-sasa-relative-difference", type=float, default=0.30)
    parser.add_argument("--max-curvature-absolute-difference", type=float, default=0.35)
    parser.add_argument(
        "--max-composition-group-l1-distance", type=float, default=0.60
    )
    parser.add_argument("--minimum-strict-pair-coverage", type=float, default=0.95)
    arguments = parser.parse_args()
    if arguments.workers <= 0:
        parser.error("workers must be positive")
    if arguments.max_decoys_per_positive <= 0 or arguments.max_decoys_per_complex <= 0:
        parser.error("decoy caps must be positive")
    if not 0 <= arguments.minimum_strict_pair_coverage <= 1:
        parser.error("minimum strict pair coverage must be between zero and one")
    return arguments


def main() -> None:
    print(json.dumps(materialize(_arguments()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
