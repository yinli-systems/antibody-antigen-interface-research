"""Fail-closed AsEP graph adapter for development-only schema validation.

The official interim graph archive is a PyTorch serialization containing NumPy
arrays.  This module never falls back to unrestricted pickle loading.  It
allows only the four NumPy reconstruction globals present in the authenticated
release plus the three concrete scalar dtypes required by its schema, and it
rejects any other pickle global before deserialization.

The archive auditor opens graph tensors only for an authenticated official
``train`` or ``val`` role.  Official test graph tensors and labels remain
sealed until a separate frozen-opening receipt exists.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import tempfile
from typing import Any, Mapping

import numpy as np
import torch

from .asep_authority import INNER_AUDIT_FORMAT, SPLIT_MANIFEST_FORMAT
from .sabdab2_structure import authenticate_file, canonical_json_bytes, sha256_file


DEVELOPMENT_GRAPH_AUDIT_FORMAT = "framecontact_asep_development_graph_audit_v1"
DEVELOPMENT_STRUCTURE_AUDIT_FORMAT = "framecontact_asep_development_structure_audit_v1"
DEVELOPMENT_CACHE_FORMAT = "framecontact_asep_development_cache_v1"
DEVELOPMENT_ARTIFACT_FORMAT = "framecontact_asep_development_artifact_v1"
PARTNER_METADATA_FORMAT = "framecontact_asep_partner_metadata_v1"
_SEQUENCE = re.compile(r"[ACDEFGHIKLMNPQRSTVWYBXZOU]+")
_EXPECTED_GLOBALS = {
    "numpy.core.multiarray._reconstruct",
    "numpy.core.multiarray.scalar",
    "numpy._core.multiarray._reconstruct",
    "numpy._core.multiarray.scalar",
    "numpy.dtype",
    "numpy.ndarray",
}
_GRAPH_KEYS = {"Nb", "Ng", "abdbid", "seqres", "mapping", "embedding", "edges", "stats"}
_AA3 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEC": "U", "PYL": "O", "ASX": "B", "GLX": "Z",
}
_BACKBONE_ATOMS = ("N", "CA", "C")


class AsepExternalError(ValueError):
    """Raised when authenticated AsEP data violates the frozen adapter schema."""


def _json(path: Path, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AsepExternalError(f"Cannot parse {name}") from error
    if not isinstance(payload, dict):
        raise AsepExternalError(f"{name} must be a JSON object")
    return payload


def _sequence(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise AsepExternalError(f"{name} must be a string")
    value = value.strip().upper()
    if not value or _SEQUENCE.fullmatch(value) is None:
        raise AsepExternalError(f"{name} contains unsupported residues")
    return value


def _numpy_safe_globals() -> list[Any]:
    multiarray = np._core.multiarray
    return [
        (multiarray._reconstruct, "numpy.core.multiarray._reconstruct"),
        (multiarray._reconstruct, "numpy._core.multiarray._reconstruct"),
        (multiarray.scalar, "numpy.core.multiarray.scalar"),
        (multiarray.scalar, "numpy._core.multiarray.scalar"),
        (np.dtype, "numpy.dtype"),
        (np.ndarray, "numpy.ndarray"),
        np.dtypes.Int64DType,
        np.dtypes.Float64DType,
        np.dtypes.BoolDType,
    ]


def load_asep_graph_bytes(
    payload: bytes,
    *,
    expected_instance_id: str,
    maximum_bytes: int = 64 * 1024 * 1024,
) -> dict[str, Any]:
    """Load one official graph with a bounded, explicit weights-only allowlist."""
    if not isinstance(payload, bytes) or not payload or len(payload) > maximum_bytes:
        raise AsepExternalError("AsEP graph member size is outside the allowed range")
    try:
        unsafe = set(
            torch.serialization.get_unsafe_globals_in_checkpoint(io.BytesIO(payload))
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise AsepExternalError("Cannot statically inspect AsEP graph serialization") from error
    if not unsafe.issubset(_EXPECTED_GLOBALS):
        raise AsepExternalError(
            f"AsEP graph contains blocked pickle globals: {sorted(unsafe - _EXPECTED_GLOBALS)}"
        )
    try:
        with torch.serialization.safe_globals(_numpy_safe_globals()):
            graph = torch.load(
                io.BytesIO(payload), map_location="cpu", weights_only=True
            )
    except Exception as error:
        raise AsepExternalError("AsEP graph violates the restricted tensor loader") from error
    if not isinstance(graph, dict):
        raise AsepExternalError("AsEP graph payload must be a dictionary")
    normalized = normalize_asep_graph(graph, expected_instance_id=expected_instance_id)
    normalized["member_sha256"] = hashlib.sha256(payload).hexdigest()
    normalized["restricted_pickle_globals"] = tuple(sorted(unsafe))
    return normalized


def _binary_mapping(value: Any, length: int, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.dtype != np.int64 or value.shape != (length,):
        raise AsepExternalError(f"{name} must be an int64 array aligned to sequence")
    if not np.isin(value, (0, 1)).all():
        raise AsepExternalError(f"{name} must be binary")
    return value.astype(np.bool_, copy=False)


def _embedding(value: Any, length: int, width: int, name: str) -> torch.Tensor:
    if (
        not torch.is_tensor(value)
        or value.layout != torch.strided
        or value.dtype != torch.float32
        or tuple(value.shape) != (length, width)
        or not bool(torch.isfinite(value).all())
    ):
        raise AsepExternalError(f"{name} embedding schema differs")
    return value.detach().cpu().contiguous()


def _sparse_binary_matrix(
    value: Any,
    shape: tuple[int, int],
    name: str,
) -> torch.Tensor:
    if (
        not torch.is_tensor(value)
        or value.layout != torch.sparse_coo
        or value.dtype != torch.int8
        or tuple(value.shape) != shape
    ):
        raise AsepExternalError(f"{name} must be an int8 sparse COO matrix")
    try:
        matrix = value.detach().cpu().coalesce()
    except RuntimeError as error:
        raise AsepExternalError(f"{name} cannot be coalesced") from error
    indices = matrix.indices()
    values = matrix.values()
    if indices.ndim != 2 or indices.shape[0] != 2 or values.ndim != 1:
        raise AsepExternalError(f"{name} sparse storage is malformed")
    if indices.shape[1] != values.shape[0] or not bool((values == 1).all()):
        raise AsepExternalError(f"{name} sparse values must be unique binary edges")
    if indices.numel() and (
        int(indices.min()) < 0
        or int(indices[0].max()) >= shape[0]
        or int(indices[1].max()) >= shape[1]
    ):
        raise AsepExternalError(f"{name} contains out-of-bounds edges")
    return matrix


def _integer_scalar(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise AsepExternalError(f"{name} must be an integer scalar")
    result = int(value)
    if result < 0:
        raise AsepExternalError(f"{name} must be nonnegative")
    return result


def normalize_asep_graph(
    graph: Mapping[str, Any], *, expected_instance_id: str
) -> dict[str, Any]:
    """Validate an official interim graph and expose model-aligned reduced axes."""
    if set(graph) != _GRAPH_KEYS:
        raise AsepExternalError("AsEP graph top-level keys differ from the official schema")
    if graph.get("abdbid") != expected_instance_id:
        raise AsepExternalError("AsEP graph ID differs from its authenticated member name")
    seqres = graph["seqres"]
    if not isinstance(seqres, dict) or set(seqres) != {"ab", "ag"}:
        raise AsepExternalError("AsEP sequence dictionary schema differs")
    if not isinstance(seqres["ab"], dict) or set(seqres["ab"]) != {"H", "L"}:
        raise AsepExternalError("AsEP antibody sequence must contain H and L")
    if not isinstance(seqres["ag"], dict) or len(seqres["ag"]) != 1:
        raise AsepExternalError("AsEP antigen sequence must contain exactly one chain")
    heavy = _sequence(seqres["ab"]["H"], "antibody H")
    light = _sequence(seqres["ab"]["L"], "antibody L")
    antigen_chain, antigen_raw = next(iter(seqres["ag"].items()))
    if not isinstance(antigen_chain, str) or not antigen_chain or len(antigen_chain) != 1:
        raise AsepExternalError("AsEP antigen chain label is invalid")
    antigen = _sequence(antigen_raw, "antigen")
    antibody = heavy + light

    mapping = graph["mapping"]
    if (
        not isinstance(mapping, dict)
        or set(mapping) != {"ab", "ag"}
        or not isinstance(mapping["ab"], dict)
        or set(mapping["ab"]) != {"seqres2cdr"}
        or not isinstance(mapping["ag"], dict)
        or set(mapping["ag"]) != {"seqres2surf", "seqres2epitope"}
    ):
        raise AsepExternalError("AsEP residue mapping schema differs")
    cdr = _binary_mapping(mapping["ab"]["seqres2cdr"], len(antibody), "seqres2cdr")
    surface = _binary_mapping(mapping["ag"]["seqres2surf"], len(antigen), "seqres2surf")
    epitope_full = _binary_mapping(
        mapping["ag"]["seqres2epitope"], len(antigen), "seqres2epitope"
    )
    if bool((epitope_full & ~surface).any()):
        raise AsepExternalError("AsEP epitope mapping is not a subset of surface")
    antibody_nodes = int(cdr.sum())
    antigen_nodes = int(surface.sum())
    if antibody_nodes <= 0 or antigen_nodes <= 0:
        raise AsepExternalError("AsEP reduced graph axes must be nonempty")
    if _integer_scalar(graph["Nb"], "Nb") != antibody_nodes or _integer_scalar(
        graph["Ng"], "Ng"
    ) != antigen_nodes:
        raise AsepExternalError("AsEP node counts differ from residue mappings")

    embeddings = graph["embedding"]
    if (
        not isinstance(embeddings, dict)
        or set(embeddings) != {"ab", "ag"}
        or not isinstance(embeddings["ab"], dict)
        or set(embeddings["ab"]) != {"igfold", "esm2"}
        or not isinstance(embeddings["ag"], dict)
        or set(embeddings["ag"]) != {"esm2"}
    ):
        raise AsepExternalError("AsEP embedding dictionary schema differs")
    antibody_esm = _embedding(embeddings["ab"]["esm2"], len(antibody), 480, "antibody ESM2")
    antibody_igfold = _embedding(
        embeddings["ab"]["igfold"], len(antibody), 512, "antibody IgFold"
    )
    antigen_esm = _embedding(embeddings["ag"]["esm2"], len(antigen), 480, "antigen ESM2")

    edges = graph["edges"]
    if not isinstance(edges, dict) or set(edges) != {"ab", "ag", "bipartite"}:
        raise AsepExternalError("AsEP edge dictionary schema differs")
    antibody_edges = _sparse_binary_matrix(
        edges["ab"], (antibody_nodes, antibody_nodes), "antibody edges"
    )
    antigen_edges = _sparse_binary_matrix(
        edges["ag"], (antigen_nodes, antigen_nodes), "antigen edges"
    )
    contacts = _sparse_binary_matrix(
        edges["bipartite"], (antibody_nodes, antigen_nodes), "bipartite contacts"
    )
    if contacts._nnz() <= 0:
        raise AsepExternalError("AsEP graph contains no positive bipartite contact")
    contact_indices = contacts.indices()
    paratope = torch.zeros(antibody_nodes, dtype=torch.bool)
    epitope = torch.zeros(antigen_nodes, dtype=torch.bool)
    paratope[contact_indices[0]] = True
    epitope[contact_indices[1]] = True
    official_epitope = torch.from_numpy(epitope_full[surface].copy())
    if not torch.equal(epitope, official_epitope):
        raise AsepExternalError("AsEP bipartite contacts do not reproduce epitope mapping")

    stats = graph["stats"]
    if not isinstance(stats, dict) or set(stats) != {
        "cdr",
        "surf",
        "epitope",
        "epitope2surf_ratio",
    }:
        raise AsepExternalError("AsEP stats schema differs")
    if (
        _integer_scalar(stats["cdr"], "stats.cdr") != antibody_nodes
        or _integer_scalar(stats["surf"], "stats.surf") != antigen_nodes
        or _integer_scalar(stats["epitope"], "stats.epitope") != int(epitope.sum())
    ):
        raise AsepExternalError("AsEP stats counts differ from graph tensors")
    ratio = float(stats["epitope2surf_ratio"])
    exact_ratio = float(epitope.sum()) / antigen_nodes
    # The release stores this redundant statistic at six decimal places.  A
    # half-unit quantization bound validates it against the exact integer
    # counts without requiring one particular tie-breaking convention.
    ratio_matches = math.isclose(
        ratio, exact_ratio, rel_tol=0.0, abs_tol=5.000001e-7
    )
    if not math.isfinite(ratio) or not ratio_matches:
        raise AsepExternalError("AsEP epitope ratio differs from graph tensors")

    antibody_letters = np.asarray(list(antibody))[cdr]
    antigen_letters = np.asarray(list(antigen))[surface]
    full_antibody_chain_ids = np.concatenate(
        (np.zeros(len(heavy), dtype=np.int64), np.ones(len(light), dtype=np.int64))
    )
    return {
        "instance_id": expected_instance_id,
        "heavy_sequence": heavy,
        "light_sequence": light,
        "antigen_sequence_full": antigen,
        "antigen_chain": antigen_chain,
        "antibody_sequence": "".join(antibody_letters.tolist()),
        "antigen_sequence": "".join(antigen_letters.tolist()),
        "antibody_full_selection": torch.from_numpy(cdr.copy()),
        "antigen_full_selection": torch.from_numpy(surface.copy()),
        "antibody_chain_ids": torch.from_numpy(full_antibody_chain_ids[cdr].copy()),
        "antigen_chain_ids": torch.zeros(antigen_nodes, dtype=torch.long),
        "antibody_esm2_35m": antibody_esm[cdr].contiguous(),
        "antibody_igfold": antibody_igfold[cdr].contiguous(),
        "antigen_esm2_35m": antigen_esm[surface].contiguous(),
        "antibody_edges": antibody_edges,
        "antigen_edges": antigen_edges,
        "pair_positive_indices": contact_indices.clone(),
        "antibody_labels": paratope.to(torch.float32),
        "antigen_labels": epitope.to(torch.float32),
        "antibody_length": antibody_nodes,
        "antigen_length": antigen_nodes,
        "positive_pair_count": int(contacts._nnz()),
        "paratope_positive_count": int(paratope.sum()),
        "epitope_positive_count": int(epitope.sum()),
    }


@dataclass(frozen=True)
class _ObservedResidue:
    amino_acid: str
    backbone: np.ndarray


def _parse_pdb_axes(
    payload: bytes,
) -> tuple[dict[str, str], dict[str, tuple[_ObservedResidue, ...]]]:
    if not isinstance(payload, bytes) or not payload or len(payload) > 64 * 1024 * 1024:
        raise AsepExternalError("AsEP PDB member size is outside the allowed range")
    try:
        lines = payload.decode("ascii", errors="strict").splitlines()
    except UnicodeError as error:
        raise AsepExternalError("AsEP PDB is not strict ASCII") from error
    seqres_blocks: dict[str, list[dict[str, Any]]] = {}
    residues: dict[tuple[str, int, str, str], dict[str, tuple[np.ndarray, float]]] = {}
    order: list[tuple[str, int, str, str]] = []
    first_model_started = False
    for line in lines:
        if line.startswith("SEQRES"):
            if len(line) < 20:
                raise AsepExternalError("AsEP PDB contains a malformed SEQRES record")
            chain = line[11].strip()
            if not chain:
                raise AsepExternalError("AsEP PDB SEQRES chain is empty")
            try:
                serial = int(line[7:10])
                declared = int(line[13:17])
            except ValueError as error:
                raise AsepExternalError("AsEP PDB SEQRES numbering is malformed") from error
            blocks = seqres_blocks.setdefault(chain, [])
            if serial == 1:
                blocks.append({"declared": declared, "serial": 1, "tokens": []})
            elif (
                not blocks
                or serial != int(blocks[-1]["serial"]) + 1
                or declared != int(blocks[-1]["declared"])
            ):
                raise AsepExternalError("AsEP PDB SEQRES block is not contiguous")
            blocks[-1]["serial"] = serial
            blocks[-1]["tokens"].extend(line[19:70].split())
            continue
        if line.startswith("MODEL"):
            if first_model_started:
                break
            first_model_started = True
            continue
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
        atom_name = line[12:16].strip().upper()
        if not chain or not atom_name:
            raise AsepExternalError("AsEP PDB atom chain/name is empty")
        try:
            residue_number = int(line[22:26])
            coordinate = np.asarray(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                dtype=np.float32,
            )
            occupancy = (
                float(line[54:60])
                if len(line) >= 60 and line[54:60].strip()
                else 0.0
            )
        except ValueError as error:
            raise AsepExternalError("AsEP PDB atom record is malformed") from error
        if not np.isfinite(coordinate).all() or not math.isfinite(occupancy):
            raise AsepExternalError("AsEP PDB atom record is non-finite")
        key = (chain, residue_number, line[26].strip(), residue_name)
        if key not in residues:
            residues[key] = {}
            order.append(key)
        previous = residues[key].get(atom_name)
        if previous is None or occupancy > previous[1]:
            residues[key][atom_name] = (coordinate, occupancy)
    seqres: dict[str, str] = {}
    for chain, blocks in seqres_blocks.items():
        protein_sequences: list[str] = []
        for block in blocks:
            tokens = block["tokens"]
            if len(tokens) != int(block["declared"]):
                raise AsepExternalError("AsEP PDB SEQRES length differs from declaration")
            supported_tokens = [
                token.upper() for token in tokens if token.upper() in _AA3
            ]
            if not supported_tokens:
                # Curated complex PDBs may retain ligand-only SEQRES blocks.
                continue
            # Some protein blocks append a non-polymer token such as PO4.  Only
            # amino-acid tokens define the residue axis; the caller then proves
            # exact equality to the authenticated graph sequence.
            protein_sequences.append("".join(_AA3[token] for token in supported_tokens))
        if len(protein_sequences) > 1:
            raise AsepExternalError("AsEP PDB chain has multiple protein SEQRES blocks")
        if protein_sequences:
            seqres[chain] = protein_sequences[0]
    observed: dict[str, list[_ObservedResidue]] = {}
    for chain, residue_number, insertion_code, residue_name in order:
        atoms = residues[(chain, residue_number, insertion_code, residue_name)]
        backbone = np.full((3, 3), np.nan, dtype=np.float32)
        for index, atom_name in enumerate(_BACKBONE_ATOMS):
            if atom_name in atoms:
                backbone[index] = atoms[atom_name][0]
        observed.setdefault(chain, []).append(
            _ObservedResidue(_AA3[residue_name], backbone)
        )
    if not seqres or not observed:
        raise AsepExternalError("AsEP PDB lacks SEQRES or observed polymer residues")
    return seqres, {chain: tuple(values) for chain, values in observed.items()}


def extract_asep_label_free_structure_metadata(
    pdb_payload: bytes,
    *,
    expected_instance_id: str,
) -> dict[str, Any]:
    """Extract test-safe full-chain metadata without opening graph labels.

    This parser only reads the authenticated PDB input feature.  It never opens
    the serialized AsEP graph, bipartite contacts, epitope mapping, or model
    predictions.  The PDB ``SEQRES`` axis can be an ordered deletion-only view
    of the graph sequence, so the resulting digests are named explicitly as
    PDB-SEQRES digests rather than silently treated as graph-sequence digests.
    """
    match = re.fullmatch(r"([0-9a-z]{4})_[0-9]+P", expected_instance_id)
    if match is None:
        raise AsepExternalError("AsEP instance identifier is malformed")
    seqres, observed = _parse_pdb_axes(pdb_payload)
    if "H" not in seqres or "L" not in seqres:
        raise AsepExternalError("AsEP PDB lacks antibody H/L SEQRES chains")
    antigen_chains = sorted(set(seqres) - {"H", "L"})
    if len(antigen_chains) != 1:
        raise AsepExternalError("AsEP PDB must contain one protein antigen chain")
    antigen_chain = antigen_chains[0]
    if set(seqres) != {"H", "L", antigen_chain} or not set(seqres).issubset(observed):
        raise AsepExternalError("AsEP PDB label-free chain axes are incomplete")
    heavy = seqres["H"]
    light = seqres["L"]
    antigen = seqres[antigen_chain]
    antibody_digest = hashlib.sha256(
        f"H\0{heavy}\0L\0{light}".encode("ascii")
    ).hexdigest()
    antigen_digest = hashlib.sha256(antigen.encode("ascii")).hexdigest()
    return {
        "instance_id": expected_instance_id,
        "pdb_id": match.group(1),
        "antigen_chain": antigen_chain,
        "heavy_sequence": heavy,
        "light_sequence": light,
        "antigen_sequence": antigen,
        "antibody_pdb_seqres_sha256": antibody_digest,
        "antigen_pdb_seqres_sha256": antigen_digest,
        "antibody_pdb_seqres_length": len(heavy) + len(light),
        "antigen_pdb_seqres_length": len(antigen),
        "antibody_observed_residue_count": len(observed["H"]) + len(observed["L"]),
        "antigen_observed_residue_count": len(observed[antigen_chain]),
    }


def extract_asep_antigen_pdb_bytes(
    pdb_payload: bytes,
    *,
    antigen_chain: str,
) -> bytes:
    """Return a deterministic first-model, antigen-only PDB for Foldseek."""
    if not isinstance(antigen_chain, str) or len(antigen_chain) != 1:
        raise AsepExternalError("AsEP antigen chain must be one character")
    try:
        lines = pdb_payload.decode("ascii", errors="strict").splitlines()
    except UnicodeError as error:
        raise AsepExternalError("AsEP PDB is not strict ASCII") from error
    selected: list[str] = []
    model_started = False
    for line in lines:
        if line.startswith("MODEL"):
            if model_started:
                break
            model_started = True
            continue
        if line.startswith("ENDMDL"):
            break
        if (
            line.startswith("ATOM  ")
            and len(line) >= 54
            and line[21].strip() == antigen_chain
        ):
            selected.append(line.rstrip())
    if not selected or not any(line[12:16].strip() == "CA" for line in selected):
        raise AsepExternalError("AsEP antigen-only PDB has no CA-bearing ATOM records")
    return ("\n".join((*selected, "TER", "END")) + "\n").encode("ascii")


def _align_observed_backbone(
    sequence: str,
    observed: tuple[_ObservedResidue, ...],
    *,
    maximum_optimal_alignments: int = 1024,
) -> tuple[np.ndarray, dict[str, int]]:
    try:
        from Bio.Align import PairwiseAligner
    except ModuleNotFoundError as error:  # pragma: no cover
        raise RuntimeError("Biopython is required for AsEP structure alignment") from error
    observed_sequence = "".join(residue.amino_acid for residue in observed)
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2.0
    aligner.mismatch_score = -100.0
    aligner.open_gap_score = -2.0
    aligner.extend_gap_score = -0.1
    alignments = aligner.align(sequence, observed_sequence)
    try:
        alignment_count = len(alignments)
    except OverflowError as error:
        raise AsepExternalError("AsEP PDB alignment count overflowed") from error
    if alignment_count <= 0 or alignment_count > maximum_optimal_alignments:
        raise AsepExternalError("AsEP PDB has too many optimal sequence alignments")
    mappings: list[tuple[int | None, ...]] = []
    for alignment in alignments:
        mapping: list[int | None] = [None] * len(sequence)
        mapped_observed: set[int] = set()
        target_blocks, query_blocks = alignment.aligned
        for target_block, query_block in zip(target_blocks, query_blocks):
            target_start, target_end = (int(value) for value in target_block)
            query_start, query_end = (int(value) for value in query_block)
            if target_end - target_start != query_end - query_start:
                raise AsepExternalError("AsEP PDB alignment block lengths differ")
            for target_index, query_index in zip(
                range(target_start, target_end), range(query_start, query_end)
            ):
                if sequence[target_index] != observed_sequence[query_index]:
                    raise AsepExternalError("AsEP PDB observed sequence differs from SEQRES")
                mapping[target_index] = query_index
                mapped_observed.add(query_index)
        if mapped_observed != set(range(len(observed))):
            raise AsepExternalError("AsEP PDB alignment does not cover observed residues")
        mappings.append(tuple(mapping))
    backbone = np.full((len(sequence), 3, 3), np.nan, dtype=np.float32)
    ambiguous = 0
    mapped = 0
    for sequence_index in range(len(sequence)):
        candidates = {mapping[sequence_index] for mapping in mappings}
        if len(candidates) != 1:
            ambiguous += 1
            continue
        observed_index = next(iter(candidates))
        if observed_index is None:
            continue
        backbone[sequence_index] = observed[observed_index].backbone
        mapped += 1
    return backbone, {
        "optimal_alignment_count": alignment_count,
        "ambiguous_seqres_position_count": ambiguous,
        "mapped_seqres_position_count": mapped,
    }


def adapt_asep_structure_bytes(
    pdb_payload: bytes,
    graph: Mapping[str, Any],
) -> dict[str, Any]:
    """Align authenticated PDB N/CA/C frames to the graph's reduced residue axes."""
    required = {
        "instance_id",
        "heavy_sequence",
        "light_sequence",
        "antigen_sequence_full",
        "antigen_chain",
        "antibody_full_selection",
        "antigen_full_selection",
    }
    if not required.issubset(graph):
        raise AsepExternalError("Normalized AsEP graph lacks structure-axis fields")
    heavy = _sequence(graph["heavy_sequence"], "antibody H")
    light = _sequence(graph["light_sequence"], "antibody L")
    antigen = _sequence(graph["antigen_sequence_full"], "antigen")
    antigen_chain = graph["antigen_chain"]
    if antigen_chain in {"H", "L"}:
        raise AsepExternalError("AsEP antigen chain collides with antibody chain")
    seqres, observed = _parse_pdb_axes(pdb_payload)
    required_chains = {"H", "L", antigen_chain}
    if set(seqres) != required_chains or not required_chains.issubset(observed):
        raise AsepExternalError("AsEP PDB polymer chains differ from graph chains")
    expected = {"H": heavy, "L": light, antigen_chain: antigen}
    def ordered_subsequence(candidate: str, authority: str) -> bool:
        iterator = iter(authority)
        return all(any(letter == value for value in iterator) for letter in candidate)

    if any(
        not ordered_subsequence(seqres[chain], sequence)
        for chain, sequence in expected.items()
    ):
        raise AsepExternalError(
            "AsEP PDB SEQRES differs from graph sequence by more than deletions"
        )
    seqres_deletions = {
        chain: len(sequence) - len(seqres[chain])
        for chain, sequence in expected.items()
    }
    full_backbones: dict[str, np.ndarray] = {}
    alignment: dict[str, dict[str, int]] = {}
    for chain in ("H", "L", antigen_chain):
        full_backbones[chain], alignment[chain] = _align_observed_backbone(
            expected[chain], observed[chain]
        )
    antibody_full = np.concatenate((full_backbones["H"], full_backbones["L"]), axis=0)
    antibody_selection = graph["antibody_full_selection"]
    antigen_selection = graph["antigen_full_selection"]
    if (
        not torch.is_tensor(antibody_selection)
        or antibody_selection.dtype != torch.bool
        or tuple(antibody_selection.shape) != (len(heavy) + len(light),)
        or not torch.is_tensor(antigen_selection)
        or antigen_selection.dtype != torch.bool
        or tuple(antigen_selection.shape) != (len(antigen),)
    ):
        raise AsepExternalError("AsEP graph selections do not align with full sequences")
    antibody_backbone = torch.from_numpy(
        antibody_full[antibody_selection.numpy()].copy()
    )
    antigen_backbone = torch.from_numpy(
        full_backbones[antigen_chain][antigen_selection.numpy()].copy()
    )
    antibody_valid = torch.isfinite(antibody_backbone).all(dim=(-1, -2))
    antigen_valid = torch.isfinite(antigen_backbone).all(dim=(-1, -2))
    if not bool(antibody_valid.any()) or not bool(antigen_valid.any()):
        raise AsepExternalError("AsEP reduced axes contain no complete backbone frame")
    selected_ambiguous = {
        "antibody": int(
            sum(alignment[chain]["ambiguous_seqres_position_count"] for chain in ("H", "L"))
        ),
        "antigen": int(alignment[antigen_chain]["ambiguous_seqres_position_count"]),
    }
    return {
        "antibody_backbone": antibody_backbone,
        "antigen_backbone": antigen_backbone,
        "antibody_frame_valid_mask": antibody_valid,
        "antigen_frame_valid_mask": antigen_valid,
        "antibody_missing_frame_mask": ~antibody_valid,
        "antigen_missing_frame_mask": ~antigen_valid,
        "alignment": alignment,
        "pdb_seqres_deletion_count": seqres_deletions,
        "full_axis_ambiguous_position_count": selected_ambiguous,
    }


def _load_split_roles(
    split_root: Path,
    receipt_sha256: str,
    split_method: str,
) -> tuple[dict[str, str], dict[str, Any], str]:
    receipt_path = split_root / "ASEP_SPLIT_MANIFEST_RECEIPT.json"
    authenticate_file(receipt_path, receipt_sha256, "AsEP split receipt")
    receipt = _json(receipt_path, "AsEP split receipt")
    if (
        receipt.get("format_version") != SPLIT_MANIFEST_FORMAT
        or receipt.get("authority", {}).get("task_labels_read") is not False
        or receipt.get("authority", {}).get("model_predictions_or_scores_read") is not False
        or split_method not in {"epitope_group", "epitope_ratio"}
    ):
        raise AsepExternalError("AsEP split authority is not eligible")
    entry = receipt.get("manifests", {}).get(split_method, {})
    manifest_path = split_root / str(entry.get("path", ""))
    manifest_sha = authenticate_file(
        manifest_path, str(entry.get("sha256", "")), "AsEP split manifest"
    )
    roles: dict[str, str] = {}
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["dataset_index", "instance_id", "role"]:
            raise AsepExternalError("AsEP split manifest columns differ")
        for expected_index, row in enumerate(reader):
            if int(row["dataset_index"]) != expected_index:
                raise AsepExternalError("AsEP split manifest index axis is not canonical")
            identifier = row["instance_id"]
            role = row["role"]
            if not identifier or identifier in roles or role not in {"train", "val", "test"}:
                raise AsepExternalError("AsEP split manifest row is invalid")
            roles[identifier] = role
    if len(roles) != int(receipt.get("instance_count", -1)):
        raise AsepExternalError("AsEP split manifest count differs")
    observed_counts = {
        role: sum(value == role for value in roles.values())
        for role in ("train", "val", "test")
    }
    if observed_counts != entry.get("role_counts"):
        raise AsepExternalError("AsEP split manifest role counts differ from receipt")
    return roles, receipt, manifest_sha


def audit_asep_development_graphs(
    *,
    authority_root: Path,
    inner_audit_path: Path,
    inner_audit_sha256: str,
    split_root: Path,
    split_receipt_sha256: str,
    split_method: str,
    output_root: Path,
    maximum_member_bytes: int = 64 * 1024 * 1024,
) -> dict[str, Any]:
    """Stream and validate every official train/val graph, leaving test sealed."""
    authenticate_file(inner_audit_path, inner_audit_sha256, "AsEP inner audit")
    inner = _json(inner_audit_path, "AsEP inner audit")
    if (
        inner.get("format_version") != INNER_AUDIT_FORMAT
        or inner.get("authority", {}).get("task_labels_read") is not False
        or inner.get("checks", {}).get("split_payload_opened") is not False
    ):
        raise AsepExternalError("AsEP inner audit is not eligible")
    roles, split_receipt, manifest_sha = _load_split_roles(
        split_root, split_receipt_sha256, split_method
    )
    index_by_id = {identifier: index for index, identifier in enumerate(roles)}
    selected = {identifier for identifier, role in roles.items() if role in {"train", "val"}}
    if not selected or len(selected) >= len(roles):
        raise AsepExternalError("AsEP development selection is invalid")
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"Refusing to overwrite AsEP graph audit: {output_root}")
    graph_archive = authority_root / "payload/asepv1.2.0/asepv1_interim_graphs.tar.gz"
    archive_sha = authenticate_file(
        graph_archive,
        str(inner.get("graph_archive", {}).get("sha256", "")),
        "AsEP graph archive",
    )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent))
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    global_names: set[str] = set()
    try:
        with tarfile.open(graph_archive, "r|gz") as archive:
            for member in archive:
                path = PurePosixPath(member.name)
                if member.isdir():
                    continue
                if (
                    not member.isfile()
                    or len(path.parts) != 2
                    or path.parts[0] != "asepv1_interim_graphs"
                    or path.suffix != ".pt"
                ):
                    raise AsepExternalError("AsEP graph tar member schema differs")
                identifier = path.stem
                if identifier not in selected:
                    continue
                if identifier in seen:
                    raise AsepExternalError("AsEP graph tar repeats a selected member")
                if member.size <= 0 or member.size > maximum_member_bytes:
                    raise AsepExternalError("AsEP graph member size is outside the limit")
                handle = archive.extractfile(member)
                if handle is None:
                    raise AsepExternalError("Cannot read selected AsEP graph member")
                raw = handle.read(maximum_member_bytes + 1)
                if len(raw) != member.size:
                    raise AsepExternalError("AsEP graph member length differs from tar metadata")
                try:
                    graph = load_asep_graph_bytes(
                        raw,
                        expected_instance_id=identifier,
                        maximum_bytes=maximum_member_bytes,
                    )
                except AsepExternalError as error:
                    raise AsepExternalError(
                        f"AsEP development graph {identifier!r} failed: {error}"
                    ) from error
                seen.add(identifier)
                global_names.update(graph["restricted_pickle_globals"])
                records.append(
                    {
                        "dataset_index": index_by_id[identifier],
                        "instance_id": identifier,
                        "role": roles[identifier],
                        "member_sha256": graph["member_sha256"],
                        "antibody_length": graph["antibody_length"],
                        "antigen_length": graph["antigen_length"],
                        "positive_pair_count": graph["positive_pair_count"],
                        "paratope_positive_count": graph["paratope_positive_count"],
                        "epitope_positive_count": graph["epitope_positive_count"],
                    }
                )
        if seen != selected:
            missing = sorted(selected - seen)
            raise AsepExternalError(f"AsEP graph archive lacks selected member {missing[:1]}")
        records.sort(key=lambda value: int(value["dataset_index"]))
        manifest_path = temporary / "ASEP_DEVELOPMENT_GRAPH_MANIFEST.csv"
        with manifest_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=tuple(records[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(records)
        role_counts = {
            role: sum(record["role"] == role for record in records)
            for role in ("train", "val")
        }
        receipt = {
            "format_version": DEVELOPMENT_GRAPH_AUDIT_FORMAT,
            "authority": {
                "inner_audit_sha256": inner_audit_sha256,
                "graph_archive_sha256": archive_sha,
                "split_receipt_sha256": split_receipt_sha256,
                "split_manifest_sha256": manifest_sha,
                "split_method": split_method,
            },
            "development_instance_count": len(records),
            "role_counts": role_counts,
            "manifest_sha256": sha256_file(manifest_path),
            "restricted_pickle_globals": sorted(global_names),
            "checks": {
                "torch_load_weights_only": True,
                "only_minimal_numpy_globals_allowlisted": True,
                "all_development_graphs_validated": True,
                "pair_contacts_reproduce_residue_marginals": True,
                "official_test_graph_tensors_opened": False,
                "official_test_labels_read": False,
                "model_predictions_or_scores_read": False,
            },
            "label_access": {
                "development_train_and_validation_labels_read": True,
                "purpose": "adapter schema validation only",
            },
            "claim_boundary": (
                "Authenticated train/validation graph-schema audit only. No official "
                "test graph tensor, prediction, score, or performance metric was opened."
            ),
        }
        (temporary / "ASEP_DEVELOPMENT_GRAPH_AUDIT.json").write_bytes(
            canonical_json_bytes(receipt)
        )
        os.replace(temporary, output_root)
        return {**receipt, "output_root": str(output_root)}
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def audit_asep_development_structures(
    *,
    authority_root: Path,
    inner_audit_path: Path,
    inner_audit_sha256: str,
    graph_audit_root: Path,
    graph_audit_receipt_sha256: str,
    output_root: Path,
    maximum_graph_member_bytes: int = 64 * 1024 * 1024,
    maximum_pdb_member_bytes: int = 64 * 1024 * 1024,
) -> dict[str, Any]:
    """Audit N/CA/C mapping for all authenticated development graph axes."""
    authenticate_file(inner_audit_path, inner_audit_sha256, "AsEP inner audit")
    inner = _json(inner_audit_path, "AsEP inner audit")
    if inner.get("format_version") != INNER_AUDIT_FORMAT:
        raise AsepExternalError("AsEP inner audit format differs")
    graph_receipt_path = graph_audit_root / "ASEP_DEVELOPMENT_GRAPH_AUDIT.json"
    authenticate_file(
        graph_receipt_path,
        graph_audit_receipt_sha256,
        "AsEP development graph audit",
    )
    graph_receipt = _json(graph_receipt_path, "AsEP development graph audit")
    if (
        graph_receipt.get("format_version") != DEVELOPMENT_GRAPH_AUDIT_FORMAT
        or graph_receipt.get("authority", {}).get("inner_audit_sha256")
        != inner_audit_sha256
        or graph_receipt.get("checks", {}).get("official_test_graph_tensors_opened")
        is not False
        or graph_receipt.get("checks", {}).get("official_test_labels_read") is not False
    ):
        raise AsepExternalError("AsEP development graph audit is not eligible")
    graph_manifest_path = graph_audit_root / "ASEP_DEVELOPMENT_GRAPH_MANIFEST.csv"
    authenticate_file(
        graph_manifest_path,
        str(graph_receipt.get("manifest_sha256", "")),
        "AsEP development graph manifest",
    )
    selected: dict[str, str] = {}
    with graph_manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"dataset_index", "instance_id", "role"}
        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            raise AsepExternalError("AsEP development graph manifest columns differ")
        for row in reader:
            identifier = row["instance_id"]
            role = row["role"]
            if not identifier or identifier in selected or role not in {"train", "val"}:
                raise AsepExternalError("AsEP development graph manifest row is invalid")
            selected[identifier] = role
    if len(selected) != int(graph_receipt.get("development_instance_count", -1)):
        raise AsepExternalError("AsEP development graph manifest count differs")
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"Refusing to overwrite AsEP structure audit: {output_root}")
    payload_root = authority_root / "payload/asepv1.2.0"
    graph_archive = payload_root / "asepv1_interim_graphs.tar.gz"
    structure_archive = payload_root / "structures.tar.gz"
    graph_archive_sha = authenticate_file(
        graph_archive,
        str(inner.get("graph_archive", {}).get("sha256", "")),
        "AsEP graph archive",
    )
    structure_archive_sha = authenticate_file(
        structure_archive,
        str(inner.get("structure_archive", {}).get("sha256", "")),
        "AsEP structure archive",
    )
    axes: dict[str, dict[str, Any]] = {}
    with tarfile.open(graph_archive, "r|gz") as archive:
        for member in archive:
            identifier = PurePosixPath(member.name).stem
            if identifier not in selected:
                continue
            if not member.isfile() or identifier in axes:
                raise AsepExternalError("AsEP development graph member is invalid")
            if member.size <= 0 or member.size > maximum_graph_member_bytes:
                raise AsepExternalError("AsEP development graph member exceeds limit")
            handle = archive.extractfile(member)
            if handle is None:
                raise AsepExternalError("Cannot read AsEP development graph member")
            raw = handle.read(maximum_graph_member_bytes + 1)
            try:
                graph = load_asep_graph_bytes(
                    raw,
                    expected_instance_id=identifier,
                    maximum_bytes=maximum_graph_member_bytes,
                )
            except AsepExternalError as error:
                raise AsepExternalError(
                    f"AsEP development graph {identifier!r} failed: {error}"
                ) from error
            axes[identifier] = {
                key: graph[key]
                for key in (
                    "instance_id",
                    "heavy_sequence",
                    "light_sequence",
                    "antigen_sequence_full",
                    "antigen_chain",
                    "antibody_full_selection",
                    "antigen_full_selection",
                )
            }
    if set(axes) != set(selected):
        raise AsepExternalError("AsEP graph axes do not cover development manifest")

    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent))
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    try:
        with tarfile.open(structure_archive, "r|gz") as archive:
            for member in archive:
                identifier = PurePosixPath(member.name).stem
                if identifier not in selected:
                    continue
                if not member.isfile() or identifier in seen:
                    raise AsepExternalError("AsEP development structure member is invalid")
                if member.size <= 0 or member.size > maximum_pdb_member_bytes:
                    raise AsepExternalError("AsEP development PDB member exceeds limit")
                handle = archive.extractfile(member)
                if handle is None:
                    raise AsepExternalError("Cannot read AsEP development PDB member")
                raw = handle.read(maximum_pdb_member_bytes + 1)
                if len(raw) != member.size:
                    raise AsepExternalError("AsEP development PDB member length differs")
                try:
                    adapted = adapt_asep_structure_bytes(raw, axes[identifier])
                except AsepExternalError as error:
                    raise AsepExternalError(
                        f"AsEP development structure {identifier!r} failed: {error}"
                    ) from error
                seen.add(identifier)
                ab_valid = int(adapted["antibody_frame_valid_mask"].sum())
                ag_valid = int(adapted["antigen_frame_valid_mask"].sum())
                ab_length = int(adapted["antibody_frame_valid_mask"].numel())
                ag_length = int(adapted["antigen_frame_valid_mask"].numel())
                antigen_chain = axes[identifier]["antigen_chain"]
                records.append(
                    {
                        "instance_id": identifier,
                        "role": selected[identifier],
                        "pdb_member_sha256": hashlib.sha256(raw).hexdigest(),
                        "antibody_length": ab_length,
                        "antigen_length": ag_length,
                        "antibody_valid_frame_count": ab_valid,
                        "antigen_valid_frame_count": ag_valid,
                        "antibody_missing_frame_count": ab_length - ab_valid,
                        "antigen_missing_frame_count": ag_length - ag_valid,
                        "heavy_optimal_alignment_count": adapted["alignment"]["H"][
                            "optimal_alignment_count"
                        ],
                        "light_optimal_alignment_count": adapted["alignment"]["L"][
                            "optimal_alignment_count"
                        ],
                        "antigen_optimal_alignment_count": adapted["alignment"][
                            antigen_chain
                        ]["optimal_alignment_count"],
                        "pdb_seqres_deletion_count_heavy": adapted[
                            "pdb_seqres_deletion_count"
                        ]["H"],
                        "pdb_seqres_deletion_count_light": adapted[
                            "pdb_seqres_deletion_count"
                        ]["L"],
                        "pdb_seqres_deletion_count_antigen": adapted[
                            "pdb_seqres_deletion_count"
                        ][antigen_chain],
                        "full_axis_ambiguous_antibody_positions": adapted[
                            "full_axis_ambiguous_position_count"
                        ]["antibody"],
                        "full_axis_ambiguous_antigen_positions": adapted[
                            "full_axis_ambiguous_position_count"
                        ]["antigen"],
                    }
                )
        if seen != set(selected):
            missing = sorted(set(selected) - seen)
            raise AsepExternalError(
                f"AsEP structure archive lacks development member {missing[:1]}"
            )
        order = {identifier: index for index, identifier in enumerate(selected)}
        records.sort(key=lambda value: order[str(value["instance_id"])])
        manifest_path = temporary / "ASEP_DEVELOPMENT_STRUCTURE_MANIFEST.csv"
        with manifest_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=tuple(records[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(records)
        totals = {
            key: sum(int(record[key]) for record in records)
            for key in (
                "antibody_length",
                "antigen_length",
                "antibody_valid_frame_count",
                "antigen_valid_frame_count",
                "antibody_missing_frame_count",
                "antigen_missing_frame_count",
                "full_axis_ambiguous_antibody_positions",
                "full_axis_ambiguous_antigen_positions",
            )
        }
        receipt = {
            "format_version": DEVELOPMENT_STRUCTURE_AUDIT_FORMAT,
            "authority": {
                "inner_audit_sha256": inner_audit_sha256,
                "graph_audit_receipt_sha256": graph_audit_receipt_sha256,
                "graph_archive_sha256": graph_archive_sha,
                "structure_archive_sha256": structure_archive_sha,
            },
            "development_instance_count": len(records),
            "manifest_sha256": sha256_file(manifest_path),
            "totals": totals,
            "complexes_with_any_antibody_missing_frame": sum(
                int(record["antibody_missing_frame_count"]) > 0 for record in records
            ),
            "complexes_with_any_antigen_missing_frame": sum(
                int(record["antigen_missing_frame_count"]) > 0 for record in records
            ),
            "checks": {
                "pdb_seqres_is_ordered_deletion_only_view_of_graph_sequences": True,
                "ambiguous_optimal_alignments_masked_not_guessed": True,
                "n_ca_c_frame_mask_explicit": True,
                "official_test_graph_tensors_opened": False,
                "official_test_labels_read": False,
                "official_test_pdb_members_opened": False,
                "model_predictions_or_scores_read": False,
            },
            "claim_boundary": (
                "Authenticated train/validation structure-axis audit only. Missing "
                "or alignment-ambiguous N/CA/C frames are masked. No test member, "
                "prediction, score, or performance metric was opened."
            ),
        }
        (temporary / "ASEP_DEVELOPMENT_STRUCTURE_AUDIT.json").write_bytes(
            canonical_json_bytes(receipt)
        )
        os.replace(temporary, output_root)
        return {**receipt, "output_root": str(output_root)}
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def materialize_asep_development_cache(
    *,
    authority_root: Path,
    inner_audit_path: Path,
    inner_audit_sha256: str,
    graph_audit_root: Path,
    graph_audit_receipt_sha256: str,
    structure_audit_root: Path,
    structure_audit_receipt_sha256: str,
    output_root: Path,
    maximum_member_bytes: int = 64 * 1024 * 1024,
) -> dict[str, Any]:
    """Materialize audited AsEP train/val tensors into an atomic runnable cache."""
    authenticate_file(inner_audit_path, inner_audit_sha256, "AsEP inner audit")
    inner = _json(inner_audit_path, "AsEP inner audit")
    graph_receipt_path = graph_audit_root / "ASEP_DEVELOPMENT_GRAPH_AUDIT.json"
    structure_receipt_path = (
        structure_audit_root / "ASEP_DEVELOPMENT_STRUCTURE_AUDIT.json"
    )
    authenticate_file(
        graph_receipt_path,
        graph_audit_receipt_sha256,
        "AsEP development graph audit",
    )
    authenticate_file(
        structure_receipt_path,
        structure_audit_receipt_sha256,
        "AsEP development structure audit",
    )
    graph_receipt = _json(graph_receipt_path, "AsEP development graph audit")
    structure_receipt = _json(
        structure_receipt_path, "AsEP development structure audit"
    )
    if (
        inner.get("format_version") != INNER_AUDIT_FORMAT
        or graph_receipt.get("format_version") != DEVELOPMENT_GRAPH_AUDIT_FORMAT
        or structure_receipt.get("format_version")
        != DEVELOPMENT_STRUCTURE_AUDIT_FORMAT
        or graph_receipt.get("authority", {}).get("inner_audit_sha256")
        != inner_audit_sha256
        or structure_receipt.get("authority", {}).get("inner_audit_sha256")
        != inner_audit_sha256
        or structure_receipt.get("authority", {}).get(
            "graph_audit_receipt_sha256"
        )
        != graph_audit_receipt_sha256
        or graph_receipt.get("checks", {}).get("official_test_graph_tensors_opened")
        is not False
        or structure_receipt.get("checks", {}).get("official_test_pdb_members_opened")
        is not False
    ):
        raise AsepExternalError("AsEP development audits are not eligible for cache use")
    structure_manifest_path = (
        structure_audit_root / "ASEP_DEVELOPMENT_STRUCTURE_MANIFEST.csv"
    )
    authenticate_file(
        structure_manifest_path,
        str(structure_receipt.get("manifest_sha256", "")),
        "AsEP development structure manifest",
    )
    selected: dict[str, str] = {}
    with structure_manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {"instance_id", "role"}.issubset(
            reader.fieldnames
        ):
            raise AsepExternalError("AsEP structure manifest columns differ")
        for row in reader:
            identifier = row["instance_id"]
            role = row["role"]
            if not identifier or identifier in selected or role not in {"train", "val"}:
                raise AsepExternalError("AsEP structure manifest row is invalid")
            selected[identifier] = role
    if len(selected) != int(structure_receipt.get("development_instance_count", -1)):
        raise AsepExternalError("AsEP structure manifest count differs")
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"Refusing to overwrite AsEP cache: {output_root}")
    payload_root = authority_root / "payload/asepv1.2.0"
    graph_archive = payload_root / "asepv1_interim_graphs.tar.gz"
    structure_archive = payload_root / "structures.tar.gz"
    graph_sha = authenticate_file(
        graph_archive,
        str(inner.get("graph_archive", {}).get("sha256", "")),
        "AsEP graph archive",
    )
    structure_sha = authenticate_file(
        structure_archive,
        str(inner.get("structure_archive", {}).get("sha256", "")),
        "AsEP structure archive",
    )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent))
    staged_root = temporary / ".graph-stage"
    staged_root.mkdir()
    records: list[dict[str, Any]] = []
    try:
        staged: set[str] = set()
        with tarfile.open(graph_archive, "r|gz") as archive:
            for member in archive:
                identifier = PurePosixPath(member.name).stem
                if identifier not in selected:
                    continue
                if not member.isfile() or identifier in staged:
                    raise AsepExternalError("AsEP cache graph member is invalid")
                if member.size <= 0 or member.size > maximum_member_bytes:
                    raise AsepExternalError("AsEP cache graph member exceeds limit")
                handle = archive.extractfile(member)
                if handle is None:
                    raise AsepExternalError("Cannot read AsEP cache graph member")
                raw = handle.read(maximum_member_bytes + 1)
                graph = load_asep_graph_bytes(
                    raw,
                    expected_instance_id=identifier,
                    maximum_bytes=maximum_member_bytes,
                )
                stage = {
                    key: graph[key]
                    for key in (
                        "instance_id",
                        "heavy_sequence",
                        "light_sequence",
                        "antigen_sequence_full",
                        "antigen_chain",
                        "antibody_sequence",
                        "antigen_sequence",
                        "antibody_full_selection",
                        "antigen_full_selection",
                        "antibody_chain_ids",
                        "antigen_chain_ids",
                        "antibody_esm2_35m",
                        "antibody_igfold",
                        "antigen_esm2_35m",
                        "pair_positive_indices",
                        "antibody_labels",
                        "antigen_labels",
                    )
                }
                torch.save(stage, staged_root / f"{identifier}.pt")
                staged.add(identifier)
        if staged != set(selected):
            raise AsepExternalError("AsEP cache graph stage is incomplete")

        materialized: set[str] = set()
        with tarfile.open(structure_archive, "r|gz") as archive:
            for member in archive:
                identifier = PurePosixPath(member.name).stem
                if identifier not in selected:
                    continue
                if not member.isfile() or identifier in materialized:
                    raise AsepExternalError("AsEP cache structure member is invalid")
                if member.size <= 0 or member.size > maximum_member_bytes:
                    raise AsepExternalError("AsEP cache structure member exceeds limit")
                handle = archive.extractfile(member)
                if handle is None:
                    raise AsepExternalError("Cannot read AsEP cache structure member")
                pdb_payload = handle.read(maximum_member_bytes + 1)
                graph = torch.load(
                    staged_root / f"{identifier}.pt",
                    map_location="cpu",
                    weights_only=True,
                )
                structure = adapt_asep_structure_bytes(pdb_payload, graph)
                artifact = {
                    "format_version": DEVELOPMENT_ARTIFACT_FORMAT,
                    "instance_id": identifier,
                    "role": selected[identifier],
                    **graph,
                    "antibody_backbone": structure["antibody_backbone"],
                    "antigen_backbone": structure["antigen_backbone"],
                    "antibody_frame_valid_mask": structure[
                        "antibody_frame_valid_mask"
                    ],
                    "antigen_frame_valid_mask": structure["antigen_frame_valid_mask"],
                    "antibody_missing_frame_mask": structure[
                        "antibody_missing_frame_mask"
                    ],
                    "antigen_missing_frame_mask": structure[
                        "antigen_missing_frame_mask"
                    ],
                }
                relative = Path("artifacts") / identifier[:2] / f"{identifier}.pt"
                artifact_path = temporary / relative
                artifact_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(artifact, artifact_path)
                records.append(
                    {
                        "instance_id": identifier,
                        "role": selected[identifier],
                        "artifact_path": relative.as_posix(),
                        "artifact_sha256": sha256_file(artifact_path),
                        "antibody_length": len(graph["antibody_sequence"]),
                        "antigen_length": len(graph["antigen_sequence"]),
                        "positive_pair_count": int(graph["pair_positive_indices"].shape[1]),
                    }
                )
                materialized.add(identifier)
        if materialized != set(selected):
            raise AsepExternalError("AsEP development cache is incomplete")
        shutil.rmtree(staged_root)
        order = {identifier: index for index, identifier in enumerate(selected)}
        records.sort(key=lambda value: order[str(value["instance_id"])])
        manifest = {
            "format_version": DEVELOPMENT_CACHE_FORMAT,
            "authority": {
                "inner_audit_sha256": inner_audit_sha256,
                "graph_audit_receipt_sha256": graph_audit_receipt_sha256,
                "structure_audit_receipt_sha256": structure_audit_receipt_sha256,
                "graph_archive_sha256": graph_sha,
                "structure_archive_sha256": structure_sha,
            },
            "feature_modes": {
                "official_esm2_35m": {
                    "antibody_dim": 480,
                    "antigen_dim": 480,
                },
                "official_igfold_esm2": {
                    "antibody_dim": 512,
                    "antigen_dim": 480,
                },
            },
            "record_count": len(records),
            "role_counts": {
                role: sum(record["role"] == role for record in records)
                for role in ("train", "val")
            },
            "records": records,
            "checks": {
                "official_test_members_materialized": False,
                "contact_marginals_validated": True,
                "missing_frames_explicit": True,
                "artifacts_weights_only_loadable": True,
            },
            "claim_boundary": (
                "Runnable AsEP train/validation cache only. Official small-ESM/IgFold "
                "features are baseline inputs, not the 1280-d ESM model used by the "
                "SAbDab2 experiment. No test result is authorized."
            ),
        }
        manifest_path = temporary / "ASEP_DEVELOPMENT_CACHE_MANIFEST.json"
        manifest_path.write_bytes(canonical_json_bytes(manifest))
        receipt = {
            "format_version": DEVELOPMENT_CACHE_FORMAT,
            "manifest_sha256": sha256_file(manifest_path),
            "record_count": len(records),
            "role_counts": manifest["role_counts"],
            "official_test_members_materialized": False,
            "model_predictions_or_scores_read": False,
        }
        (temporary / "ASEP_DEVELOPMENT_CACHE_RECEIPT.json").write_bytes(
            canonical_json_bytes(receipt)
        )
        os.replace(temporary, output_root)
        return {**receipt, "output_root": str(output_root)}
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


class AsepDevelopmentDataset(torch.utils.data.Dataset):
    """Load an authenticated AsEP train/val cache on complete pair grids."""

    def __init__(
        self,
        cache_root: Path,
        *,
        manifest_sha256: str,
        split_role: str,
        feature_mode: str = "official_esm2_35m",
    ) -> None:
        if split_role not in {"train", "val"}:
            raise ValueError("AsEP development split_role must be train or val")
        manifest_path = cache_root / "ASEP_DEVELOPMENT_CACHE_MANIFEST.json"
        authenticate_file(manifest_path, manifest_sha256, "AsEP cache manifest")
        manifest = _json(manifest_path, "AsEP cache manifest")
        if (
            manifest.get("format_version") != DEVELOPMENT_CACHE_FORMAT
            or manifest.get("checks", {}).get("official_test_members_materialized")
            is not False
            or feature_mode not in manifest.get("feature_modes", {})
        ):
            raise AsepExternalError("AsEP development cache is not eligible")
        records = [
            record for record in manifest.get("records", []) if record.get("role") == split_role
        ]
        if not records:
            raise AsepExternalError("AsEP development dataset selection is empty")
        self.cache_root = cache_root
        self.records = tuple(records)
        self.feature_mode = feature_mode

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        path = self.cache_root / record["artifact_path"]
        authenticate_file(path, record["artifact_sha256"], "AsEP cache artifact")
        artifact = torch.load(path, map_location="cpu", weights_only=True)
        if (
            artifact.get("format_version") != DEVELOPMENT_ARTIFACT_FORMAT
            or artifact.get("instance_id") != record["instance_id"]
            or artifact.get("role") != record["role"]
        ):
            raise AsepExternalError("AsEP cache artifact metadata differs")
        antibody_features = (
            artifact["antibody_esm2_35m"]
            if self.feature_mode == "official_esm2_35m"
            else artifact["antibody_igfold"]
        )
        antigen_features = artifact["antigen_esm2_35m"]
        antibody_length = int(antibody_features.shape[0])
        antigen_length = int(antigen_features.shape[0])
        pair_labels = torch.zeros(
            antibody_length, antigen_length, dtype=torch.float32
        )
        positive = artifact["pair_positive_indices"]
        if (
            not torch.is_tensor(positive)
            or positive.dtype != torch.long
            or positive.ndim != 2
            or positive.shape[0] != 2
        ):
            raise AsepExternalError("AsEP cache contact indices are malformed")
        pair_labels[positive[0], positive[1]] = 1.0
        if not torch.equal(
            pair_labels.bool().any(dim=1).to(torch.float32),
            artifact["antibody_labels"],
        ) or not torch.equal(
            pair_labels.bool().any(dim=0).to(torch.float32),
            artifact["antigen_labels"],
        ):
            raise AsepExternalError("AsEP cache contacts do not reproduce marginals")
        return {
            "instance_id": artifact["instance_id"],
            "antibody_features": antibody_features,
            "antigen_features": antigen_features,
            "antibody_backbone": artifact["antibody_backbone"],
            "antigen_backbone": artifact["antigen_backbone"],
            "antibody_chain_ids": artifact["antibody_chain_ids"],
            "antigen_chain_ids": artifact["antigen_chain_ids"],
            "antibody_frame_valid_mask": artifact["antibody_frame_valid_mask"],
            "antigen_frame_valid_mask": artifact["antigen_frame_valid_mask"],
            "antibody_missing_frame_mask": artifact["antibody_missing_frame_mask"],
            "antigen_missing_frame_mask": artifact["antigen_missing_frame_mask"],
            "antibody_labels": artifact["antibody_labels"],
            "antigen_labels": artifact["antigen_labels"],
            "pair_labels": pair_labels,
            "pair_candidate_mask": torch.ones_like(pair_labels, dtype=torch.bool),
        }


def materialize_asep_partner_metadata(
    *,
    cache_root: Path,
    cache_manifest_sha256: str,
    output_root: Path,
) -> dict[str, Any]:
    """Build label-independent exact-sequence metadata for reciprocal panels."""
    manifest_path = cache_root / "ASEP_DEVELOPMENT_CACHE_MANIFEST.json"
    authenticate_file(manifest_path, cache_manifest_sha256, "AsEP cache manifest")
    manifest = _json(manifest_path, "AsEP cache manifest")
    if (
        manifest.get("format_version") != DEVELOPMENT_CACHE_FORMAT
        or manifest.get("checks", {}).get("official_test_members_materialized")
        is not False
    ):
        raise AsepExternalError("AsEP cache is not eligible for partner metadata")
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"Refusing to overwrite AsEP partner metadata: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent))
    rows: list[dict[str, Any]] = []
    try:
        for record in manifest.get("records", []):
            path = cache_root / record["artifact_path"]
            authenticate_file(path, record["artifact_sha256"], "AsEP cache artifact")
            artifact = torch.load(path, map_location="cpu", weights_only=True)
            if (
                artifact.get("format_version") != DEVELOPMENT_ARTIFACT_FORMAT
                or artifact.get("instance_id") != record["instance_id"]
                or artifact.get("role") != record["role"]
            ):
                raise AsepExternalError("AsEP artifact differs during metadata build")
            # Do not inspect antibody_labels, antigen_labels, or contacts here.
            heavy = _sequence(artifact["heavy_sequence"], "antibody H")
            light = _sequence(artifact["light_sequence"], "antibody L")
            antigen = _sequence(artifact["antigen_sequence_full"], "antigen")
            antibody_digest = hashlib.sha256(
                f"H\0{heavy}\0L\0{light}".encode("ascii")
            ).hexdigest()
            antigen_digest = hashlib.sha256(antigen.encode("ascii")).hexdigest()
            role = "validation" if record["role"] == "val" else record["role"]
            rows.append(
                {
                    "instance_id": record["instance_id"],
                    "role": role,
                    "antigen_cluster_id": f"exact:{antigen_digest}",
                    "antigen_sequence_sha256": antigen_digest,
                    "antigen_length": len(antigen),
                    "surface_residue_count": int(
                        artifact["antigen_full_selection"].sum()
                    ),
                    "antigen_class": "protein",
                    "antibody_cluster_id": f"exact:{antibody_digest}",
                    "antibody_sequence_sha256": antibody_digest,
                    "antibody_length": len(heavy) + len(light),
                    "cdr_residue_count": int(
                        artifact["antibody_full_selection"].sum()
                    ),
                    "antibody_class": "paired_HL",
                    "documented_cognate_group_id": "",
                }
            )
        if len(rows) != int(manifest.get("record_count", -1)):
            raise AsepExternalError("AsEP partner metadata count differs")
        metadata_path = temporary / "ASEP_PARTNER_METADATA.csv"
        with metadata_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        receipt = {
            "format_version": PARTNER_METADATA_FORMAT,
            "cache_manifest_sha256": cache_manifest_sha256,
            "metadata_sha256": sha256_file(metadata_path),
            "record_count": len(rows),
            "role_counts": {
                role: sum(row["role"] == role for row in rows)
                for role in ("train", "validation")
            },
            "sequence_cluster_definition": "exact full chain sequence SHA-256",
            "source_artifacts_contain_task_labels": True,
            "task_label_or_contact_keys_accessed_by_metadata_builder": False,
            "metadata_contains_task_label_columns": False,
            "official_test_members_present": False,
            "claim_boundary": (
                "Panel-matching metadata only. Exact-sequence IDs and CDR/surface "
                "axis sizes are used; no label/contact value participates in candidate ranking."
            ),
        }
        (temporary / "ASEP_PARTNER_METADATA_RECEIPT.json").write_bytes(
            canonical_json_bytes(receipt)
        )
        os.replace(temporary, output_root)
        return {**receipt, "output_root": str(output_root)}
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


__all__ = [
    "AsepExternalError",
    "AsepDevelopmentDataset",
    "DEVELOPMENT_ARTIFACT_FORMAT",
    "DEVELOPMENT_CACHE_FORMAT",
    "DEVELOPMENT_GRAPH_AUDIT_FORMAT",
    "DEVELOPMENT_STRUCTURE_AUDIT_FORMAT",
    "PARTNER_METADATA_FORMAT",
    "adapt_asep_structure_bytes",
    "audit_asep_development_graphs",
    "audit_asep_development_structures",
    "load_asep_graph_bytes",
    "extract_asep_antigen_pdb_bytes",
    "extract_asep_label_free_structure_metadata",
    "materialize_asep_development_cache",
    "materialize_asep_partner_metadata",
    "normalize_asep_graph",
]
