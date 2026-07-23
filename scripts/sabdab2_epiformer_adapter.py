#!/usr/bin/env python3
"""Run and package EpiFormer predictions for the frozen SAbDab2 benchmark.

The adapter has two commands:

``infer``
    Runs a resumable shard of structure inference and projects surface-residue
    probabilities onto the benchmark's canonical antigen axis.

``assemble``
    Joins all canonical residue scores to the authoritative finite candidate
    pairs, validates the fixed prediction contract, and writes summary metrics.

The implementation intentionally keeps inference and assembly separate. A GPU
failure can therefore be retried without rebuilding a large Parquet file, and
the final prediction table is always derived directly from the frozen candidate
rows instead of reconstructed indexes.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import re
import sys
import tempfile
import time
import traceback
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from Bio import Align
from Bio.PDB import MMCIFParser, PDBIO, PDBParser, Select
from Bio.PDB.Model import Model
from Bio.PDB.Structure import Structure
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)


AA3_TO_AA1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "MSE": "M",
}

IMGT_CDR_RANGES = ((27, 38), (56, 65), (105, 117))

PREDICTION_SCHEMA = pa.schema(
    [
        pa.field("instance_id", pa.string(), nullable=False),
        pa.field("ab_res_idx", pa.int64(), nullable=False),
        pa.field("ag_res_idx", pa.int64(), nullable=False),
        pa.field("score", pa.float32(), nullable=False),
    ]
)


@dataclass(frozen=True)
class BenchmarkLayout:
    """Resolved paths for one materialized SAbDab2 delivery."""

    root: Path
    frozen: Path
    structures: Path
    mappings: Path

    @classmethod
    def from_root(cls, root: Path) -> "BenchmarkLayout":
        root = root.resolve()
        data_root = root / "data/nfs/yangwuyu"
        return cls(
            root=root,
            frozen=data_root / "baseline-eval/frozen/sabdab2",
            structures=data_root / "baseline-eval/datasets/pdb",
            mappings=data_root / "sabdab2_bipartite_contacts/canonical/mapping",
        )

    def validate(self) -> None:
        required = [
            self.frozen / "pairs.parquet",
            self.frozen / "instances.parquet",
            self.frozen / "splits.json",
            self.frozen / "MANIFEST.json",
            self.structures,
            self.mappings,
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing benchmark inputs: {missing}")


@dataclass(frozen=True)
class ResidueRecord:
    """A standard residue with a stable structural identity."""

    chain_id: str
    residue_id: int
    insertion_code: str
    amino_acid: str

    @property
    def key(self) -> tuple[str, int, str]:
        return (self.chain_id, self.residue_id, self.insertion_code)


@dataclass(frozen=True)
class ProjectionAudit:
    """Mapping diagnostics retained for every completed test instance."""

    instance_id: str
    mapping_method: str
    canonical_length: int
    source_residue_count: int
    model_surface_residue_count: int
    antibody_cdr_residue_count: int
    mapped_surface_residue_count: int
    mapped_surface_fraction: float
    unique_canonical_surface_count: int
    filled_with_minimum_count: int
    source_mapping_coverage: float
    source_mapping_identity: float
    model_minimum_score: float
    model_maximum_score: float
    elapsed_seconds: float


@dataclass(frozen=True)
class ResolvedChains:
    """Label-chain structure and resolved model component assignments."""

    structure: Any
    antigen_chain_ids: tuple[str, ...]
    antibody_components: tuple[tuple[str, str], ...]


class ChainSelect(Select):
    """Select an explicit chain set while preserving structure coordinates."""

    def __init__(self, chain_ids: Sequence[str]) -> None:
        self.chain_ids = frozenset(chain_ids)

    def accept_chain(self, chain: Any) -> bool:
        return chain.id in self.chain_ids


def parse_chain_ids(value: Any) -> list[str]:
    """Parse comma, plus, pipe, semicolon, or whitespace-separated chain IDs."""
    tokens = [token for token in re.split(r"[,+;|\s]+", str(value).strip()) if token]
    if not tokens:
        raise ValueError(f"No chain IDs could be parsed from {value!r}")
    if len(tokens) != len(set(tokens)):
        raise ValueError(f"Duplicate chain IDs are not allowed: {tokens}")
    return tokens


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON through an adjacent temporary file and an atomic rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_write_npz(path: Path, **arrays: np.ndarray) -> None:
    """Write a compressed NumPy archive atomically without suffix rewriting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file using bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_split(layout: BenchmarkLayout, split_name: str) -> list[str]:
    """Load and validate one fixed split from the benchmark contract."""
    payload = json.loads((layout.frozen / "splits.json").read_text(encoding="utf-8-sig"))
    try:
        instance_ids = payload["sabdab2"][split_name]
    except KeyError as error:
        raise KeyError(f"Split {split_name!r} is not available") from error
    if len(instance_ids) != len(set(instance_ids)):
        raise ValueError(f"Split {split_name!r} contains duplicate instance IDs")
    return [str(instance_id) for instance_id in instance_ids]


def load_instances(layout: BenchmarkLayout) -> pd.DataFrame:
    """Load instance metadata and enforce one row per instance."""
    columns = [
        "instance_id",
        "ab_chain_ids",
        "ag_chain_ids",
        "ab_seq",
        "ag_seq",
        "pdb_path",
    ]
    instances = pd.read_parquet(layout.frozen / "instances.parquet", columns=columns)
    if instances["instance_id"].isna().any():
        raise ValueError("instances.parquet contains a null instance_id")
    if instances["instance_id"].duplicated().any():
        duplicates = instances.loc[
            instances["instance_id"].duplicated(), "instance_id"
        ].head(5)
        raise ValueError(f"instances.parquet contains duplicate IDs: {duplicates.tolist()}")
    return instances.set_index("instance_id", verify_integrity=True)


def write_selected_chains(
    cif_path: Path,
    output_path: Path,
    chain_ids: Sequence[str],
) -> None:
    """Extract selected mmCIF chains into the PDB input expected by EpiFormer."""
    structure = MMCIFParser(QUIET=True).get_structure("complex", str(cif_path))
    available = {chain.id for chain in structure.get_chains()}
    missing = set(chain_ids) - available
    if missing:
        raise ValueError(
            f"Missing chains {sorted(missing)} in {cif_path.name}; "
            f"available={sorted(available)}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = PDBIO()
    writer.set_structure(structure)
    writer.save(str(output_path), ChainSelect(chain_ids))


def read_pdb_residues(pdb_path: Path) -> list[ResidueRecord]:
    """Read standard CA-bearing residues in the exact PDB traversal order."""
    structure = PDBParser(QUIET=True).get_structure("protein", str(pdb_path))
    records: list[ResidueRecord] = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] != " " or "CA" not in residue:
                    continue
                records.append(
                    ResidueRecord(
                        chain_id=str(chain.id),
                        residue_id=int(residue.id[1]),
                        insertion_code=str(residue.id[2]),
                        amino_acid=AA3_TO_AA1.get(str(residue.resname).upper(), "X"),
                    )
                )
    if not records:
        raise ValueError(f"No standard CA-bearing residues found in {pdb_path}")
    keys = [record.key for record in records]
    if len(keys) != len(set(keys)):
        raise ValueError(f"Residue identities are not unique in {pdb_path}")
    return records


def structure_residues_for_chains(
    structure: Any,
    chain_ids: Sequence[str],
) -> list[ResidueRecord]:
    """Collect standard CA-bearing residues for selected chains from one structure."""
    wanted = set(chain_ids)
    available = {chain.id for chain in structure.get_chains()}
    missing = wanted - available
    if missing:
        raise ValueError(
            f"Missing structure chains {sorted(missing)}; available={sorted(available)}"
        )
    records: list[ResidueRecord] = []
    for chain in structure.get_chains():
        if chain.id not in wanted:
            continue
        for residue in chain:
            if residue.id[0] != " " or "CA" not in residue:
                continue
            records.append(
                ResidueRecord(
                    chain_id=str(chain.id),
                    residue_id=int(residue.id[1]),
                    insertion_code=str(residue.id[2]),
                    amino_acid=AA3_TO_AA1.get(str(residue.resname).upper(), "X"),
                )
            )
    return records


def _as_string_list(value: Any) -> list[str]:
    """Normalize an mmCIF scalar or list field to a list of strings."""
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _logical_chain_candidates(
    logical_chain_id: str,
    available_labels: Sequence[str],
    auth_to_labels: Mapping[str, Sequence[str]],
) -> list[str]:
    """Find label-chain candidates for a benchmark logical chain identifier."""
    candidates: list[str] = []

    def add(values: Iterable[str]) -> None:
        for value in values:
            if value in available_labels and value not in candidates:
                candidates.append(value)

    add([logical_chain_id])
    add(auth_to_labels.get(logical_chain_id, []))
    base_match = re.match(r"^(.+?)(\d+)$", logical_chain_id)
    if base_match is not None:
        add(auth_to_labels.get(base_match.group(1), []))
    return candidates


def _best_chain_for_sequence(
    canonical_sequence: str,
    candidates: Sequence[str],
    residues_by_chain: Mapping[str, Sequence[ResidueRecord]],
    excluded: set[str],
) -> str:
    """Choose the candidate whose structure sequence best covers a component."""
    scored: list[tuple[float, float, str]] = []
    for chain_id in candidates:
        if chain_id in excluded or chain_id not in residues_by_chain:
            continue
        try:
            _, _, coverage, identity = canonical_to_source_residue_map(
                residues_by_chain[chain_id],
                canonical_sequence,
                [],
            )
        except ValueError:
            continue
        scored.append((coverage, identity, chain_id))
    if not scored:
        raise ValueError("No physical chain aligns reliably to a canonical component")
    scored.sort(reverse=True)
    return scored[0][2]


def resolve_instance_chains(
    cif_path: Path,
    row: pd.Series,
    mapping_payload: Mapping[str, Any],
) -> ResolvedChains:
    """Resolve benchmark logical IDs to unique mmCIF label-chain IDs."""
    parser = MMCIFParser(QUIET=True, auth_chains=False)
    structure = parser.get_structure("complex", str(cif_path))
    available_labels = [str(chain.id) for chain in structure.get_chains()]
    mmcif = parser._mmcif_dict
    label_values = _as_string_list(mmcif["_atom_site.label_asym_id"])
    auth_values = _as_string_list(mmcif["_atom_site.auth_asym_id"])
    if len(label_values) != len(auth_values):
        raise ValueError("mmCIF label/auth chain arrays have different lengths")
    auth_to_labels: dict[str, list[str]] = {}
    for label_chain, auth_chain in zip(label_values, auth_values):
        labels = auth_to_labels.setdefault(auth_chain, [])
        if label_chain in available_labels and label_chain not in labels:
            labels.append(label_chain)

    all_residues = structure_residues_for_chains(structure, available_labels)
    residues_by_chain: dict[str, list[ResidueRecord]] = {}
    for residue in all_residues:
        residues_by_chain.setdefault(residue.chain_id, []).append(residue)

    canonical_antibody = str(row["ab_seq"])
    vh_length = len(mapping_payload.get("vh_imgt_positions", []))
    vl_length = len(mapping_payload.get("vl_imgt_positions", []))
    if vh_length + vl_length != len(canonical_antibody):
        raise ValueError("Canonical antibody sequence does not match VH/VL metadata")
    components: list[tuple[str, str, str]] = [
        (str(mapping_payload.get("Hchain", "")), "H", canonical_antibody[:vh_length])
    ]
    light_chain = str(mapping_payload.get("Lchain", ""))
    if light_chain and light_chain != "+":
        components.append((light_chain, "L", canonical_antibody[vh_length:]))

    used: set[str] = set()
    physical_components: list[tuple[str, str]] = []
    for logical_chain, chain_type, canonical_sequence in components:
        candidates = _logical_chain_candidates(
            logical_chain,
            available_labels,
            auth_to_labels,
        )
        if not candidates:
            candidates = list(available_labels)
        try:
            physical_chain = _best_chain_for_sequence(
                canonical_sequence,
                candidates,
                residues_by_chain,
                used,
            )
        except ValueError:
            try:
                physical_chain = _best_chain_for_sequence(
                    canonical_sequence,
                    available_labels,
                    residues_by_chain,
                    used,
                )
            except ValueError:
                physical_chain = _best_chain_for_sequence(
                    canonical_sequence,
                    available_labels,
                    residues_by_chain,
                    set(),
                )
        physical_components.append((physical_chain, chain_type))
        used.add(physical_chain)

    logical_antigen_chains = parse_chain_ids(row["ag_chain_ids"])
    antigen_candidate_sets: list[list[str]] = []
    for logical_chain in logical_antigen_chains:
        candidates = [
            chain_id
            for chain_id in _logical_chain_candidates(
                logical_chain,
                available_labels,
                auth_to_labels,
            )
            if chain_id not in used
        ]
        if not candidates:
            candidates = [chain_id for chain_id in available_labels if chain_id not in used]
        antigen_candidate_sets.append(candidates)

    canonical_antigen = str(row["ag_seq"])
    scored_antigen_assignments: list[tuple[float, float, tuple[str, ...]]] = []
    for assignment in product(*antigen_candidate_sets):
        if len(assignment) != len(set(assignment)):
            continue
        residues = [
            residue
            for chain_id in assignment
            for residue in residues_by_chain.get(chain_id, [])
        ]
        if not residues:
            continue
        try:
            _, coverage, identity = alignment_index_map(residues, canonical_antigen)
        except ValueError:
            continue
        if identity >= 0.80 and coverage >= 0.80:
            scored_antigen_assignments.append(
                (coverage, identity, tuple(str(chain_id) for chain_id in assignment))
            )
    if not scored_antigen_assignments:
        raise ValueError("No physical antigen chain assignment matches the canonical sequence")
    scored_antigen_assignments.sort(reverse=True)
    antigen_chain_ids = scored_antigen_assignments[0][2]
    if set(antigen_chain_ids) & {chain_id for chain_id, _ in physical_components}:
        raise ValueError("Resolved antigen and antibody components overlap")
    return ResolvedChains(
        structure=structure,
        antigen_chain_ids=antigen_chain_ids,
        antibody_components=tuple(physical_components),
    )


def write_resolved_chains(
    structure: Any,
    output_path: Path,
    chain_assignments: Sequence[tuple[str, str]],
) -> None:
    """Write selected label chains in the requested biological component order."""
    chain_lookup = {str(chain.id): chain for chain in structure.get_chains()}
    selected_structure = Structure("selected")
    selected_model = Model(0)
    selected_structure.add(selected_model)
    for chain_id, alias in chain_assignments:
        if chain_id not in chain_lookup:
            raise ValueError(f"Resolved chain {chain_id!r} is absent from the structure")
        alias = str(alias)
        if len(alias) != 1:
            raise ValueError(f"PDB chain alias must contain one character: {alias!r}")
        selected_chain = deepcopy(chain_lookup[chain_id])
        selected_chain.detach_parent()
        selected_chain.id = alias
        selected_model.add(selected_chain)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = PDBIO()
    writer.set_structure(selected_structure)
    writer.save(str(output_path))


def alignment_index_map(
    source: Sequence[ResidueRecord],
    canonical_sequence: str,
) -> tuple[dict[tuple[str, int, str], int], float, float]:
    """Align structure residues to a canonical sequence and return index mapping."""
    source_sequence = "".join(record.amino_acid for record in source)
    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -8.0
    aligner.extend_gap_score = -0.5
    alignment = aligner.align(source_sequence, canonical_sequence)[0]

    source_to_canonical: dict[int, int] = {}
    for source_block, canonical_block in zip(*alignment.aligned):
        source_start, source_end = map(int, source_block)
        canonical_start, canonical_end = map(int, canonical_block)
        if source_end - source_start != canonical_end - canonical_start:
            raise RuntimeError("Aligned ungapped blocks must have equal lengths")
        for offset in range(source_end - source_start):
            source_to_canonical[source_start + offset] = canonical_start + offset

    if not source_to_canonical:
        raise ValueError("Structure-to-canonical alignment produced no mapped residues")
    matches = sum(
        source_sequence[source_index] == canonical_sequence[canonical_index]
        for source_index, canonical_index in source_to_canonical.items()
    )
    identity = matches / len(source_to_canonical)
    coverage = len(set(source_to_canonical.values())) / len(canonical_sequence)
    mapping = {
        source[source_index].key: canonical_index
        for source_index, canonical_index in source_to_canonical.items()
    }
    return mapping, coverage, identity


def canonical_index_map(
    source: Sequence[ResidueRecord],
    canonical_sequence: str,
    mapping_payload: Mapping[str, Any],
) -> tuple[dict[tuple[str, int, str], int], str, float, float]:
    """Prefer the delivered canonical mapping and fall back to sequence alignment."""
    delivered = mapping_payload.get("ag_to_agseq")
    if isinstance(delivered, list) and len(delivered) == len(source):
        mapping: dict[tuple[str, int, str], int] = {}
        comparable = 0
        matches = 0
        for record, raw_index in zip(source, delivered):
            if raw_index is None:
                continue
            canonical_index = int(raw_index)
            if not 0 <= canonical_index < len(canonical_sequence):
                continue
            mapping[record.key] = canonical_index
            if record.amino_acid != "X":
                comparable += 1
                matches += int(record.amino_acid == canonical_sequence[canonical_index])
        identity = matches / comparable if comparable else 0.0
        coverage = len(set(mapping.values())) / len(canonical_sequence)
        if mapping and identity >= 0.90:
            return mapping, "delivered_ag_to_agseq", coverage, identity

    mapping, coverage, identity = alignment_index_map(source, canonical_sequence)
    if identity < 0.80:
        raise ValueError(
            f"Structure-to-canonical sequence identity is too low: {identity:.4f}"
        )
    return mapping, "global_sequence_alignment", coverage, identity


def antibody_chain_type_map(
    mapping_payload: Mapping[str, Any],
    antibody_chains: Sequence[str],
) -> dict[str, str]:
    """Build an explicit heavy/light map from the delivered canonical metadata."""
    chain_types: dict[str, str] = {}
    heavy_chain = str(mapping_payload.get("Hchain", "")).strip()
    light_chain = str(mapping_payload.get("Lchain", "")).strip()
    if heavy_chain and heavy_chain != "+":
        chain_types[heavy_chain] = "H"
    if light_chain and light_chain != "+":
        if light_chain in chain_types:
            raise ValueError(f"Chain {light_chain!r} is marked as both heavy and light")
        chain_types[light_chain] = "L"
    missing = sorted(set(antibody_chains) - set(chain_types))
    extra = sorted(set(chain_types) - set(antibody_chains))
    if missing or extra:
        raise ValueError(
            "Canonical antibody chain metadata does not match the instance chains: "
            f"missing_types={missing}, unexpected_types={extra}"
        )
    return chain_types


def canonical_to_source_residue_map(
    source_residues: Sequence[ResidueRecord],
    canonical_sequence: str,
    delivered_source_indices: Sequence[Any],
) -> tuple[dict[int, int], str, float, float]:
    """Map canonical antibody positions to structure residues with validation."""
    if len(delivered_source_indices) == len(canonical_sequence):
        delivered_map: dict[int, int] = {}
        comparable = 0
        matches = 0
        valid = True
        for canonical_index, raw_source_index in enumerate(delivered_source_indices):
            if raw_source_index is None:
                continue
            source_index = int(raw_source_index)
            if not 0 <= source_index < len(source_residues):
                valid = False
                break
            delivered_map[canonical_index] = source_index
            source_amino_acid = source_residues[source_index].amino_acid
            canonical_amino_acid = canonical_sequence[canonical_index]
            if source_amino_acid != "X" and canonical_amino_acid != "X":
                comparable += 1
                matches += int(source_amino_acid == canonical_amino_acid)
        identity = matches / comparable if comparable else 0.0
        coverage = len(delivered_map) / len(canonical_sequence)
        if valid and identity >= 0.90 and coverage >= 0.80:
            return delivered_map, "delivered_vhvl_to_source", coverage, identity

    source_sequence = "".join(residue.amino_acid for residue in source_residues)
    aligner = Align.PairwiseAligner()
    aligner.mode = "local"
    aligner.match_score = 2.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -2.0
    aligner.extend_gap_score = -0.2
    alignment = aligner.align(source_sequence, canonical_sequence)[0]
    canonical_to_source: dict[int, int] = {}
    for source_block, canonical_block in zip(*alignment.aligned):
        source_start, source_end = map(int, source_block)
        canonical_start, canonical_end = map(int, canonical_block)
        if source_end - source_start != canonical_end - canonical_start:
            raise RuntimeError("Aligned ungapped blocks must have equal lengths")
        for offset in range(source_end - source_start):
            canonical_to_source[canonical_start + offset] = source_start + offset

    if not canonical_to_source:
        raise ValueError("Antibody sequence alignment produced no mapped residues")
    comparable = 0
    matches = 0
    for canonical_index, source_index in canonical_to_source.items():
        source_amino_acid = source_residues[source_index].amino_acid
        canonical_amino_acid = canonical_sequence[canonical_index]
        if source_amino_acid != "X" and canonical_amino_acid != "X":
            comparable += 1
            matches += int(source_amino_acid == canonical_amino_acid)
    identity = matches / comparable if comparable else 0.0
    coverage = len(canonical_to_source) / len(canonical_sequence)
    if identity < 0.80 or coverage < 0.80:
        raise ValueError(
            "Canonical antibody sequence does not align reliably to the structure: "
            f"identity={identity:.4f}, coverage={coverage:.4f}"
        )
    return canonical_to_source, "local_sequence_alignment", coverage, identity


def benchmark_cdr_residue_keys(
    antibody_residues: Sequence[ResidueRecord],
    mapping_payload: Mapping[str, Any],
    canonical_antibody_sequence: str,
    chain_types: Mapping[str, str] | Sequence[tuple[str, str]],
) -> list[tuple[str, int, str]]:
    """Select CDR residues from delivered IMGT and source-sequence mappings."""
    residues_by_chain: dict[str, list[ResidueRecord]] = {}
    for residue in antibody_residues:
        residues_by_chain.setdefault(residue.chain_id, []).append(residue)

    vh_positions = mapping_payload.get("vh_imgt_positions", [])
    vl_positions = mapping_payload.get("vl_imgt_positions", [])
    vh_length = len(vh_positions)
    vl_length = len(vl_positions)
    if vh_length + vl_length != len(canonical_antibody_sequence):
        raise ValueError(
            "Canonical antibody sequence length does not match delivered VH/VL positions: "
            f"sequence={len(canonical_antibody_sequence)}, vh={vh_length}, vl={vl_length}"
        )

    selected: list[tuple[str, int, str]] = []
    chain_components = (
        list(chain_types.items()) if isinstance(chain_types, Mapping) else list(chain_types)
    )
    for chain_id, chain_type in chain_components:
        prefix = "vh" if chain_type == "H" else "vl"
        sequence_key = "hseq" if chain_type == "H" else "lseq"
        positions = mapping_payload.get(f"{prefix}_imgt_positions", [])
        source_indices = mapping_payload.get(f"{prefix}_to_{sequence_key}", [])
        if len(positions) != len(source_indices):
            raise ValueError(
                f"IMGT/source mapping length mismatch for chain {chain_id!r}: "
                f"positions={len(positions)}, source_indices={len(source_indices)}"
            )
        if chain_id not in residues_by_chain:
            raise ValueError(f"No structure residues found for antibody chain {chain_id!r}")
        chain_residues = residues_by_chain[chain_id]
        sequence_offset = 0 if chain_type == "H" else vh_length
        canonical_chain_sequence = canonical_antibody_sequence[
            sequence_offset : sequence_offset + len(positions)
        ]
        canonical_to_source, _, _, _ = canonical_to_source_residue_map(
            chain_residues,
            canonical_chain_sequence,
            source_indices,
        )
        mapped_cdr_positions = 0
        expected_cdr_positions = 0
        for canonical_index, position in enumerate(positions):
            match = re.match(r"^(\d+)", str(position))
            if match is None:
                raise ValueError(f"Invalid IMGT position {position!r} for chain {chain_id!r}")
            imgt_number = int(match.group(1))
            is_cdr = any(start <= imgt_number <= end for start, end in IMGT_CDR_RANGES)
            expected_cdr_positions += int(is_cdr)
            source_index = canonical_to_source.get(canonical_index)
            if source_index is None:
                continue
            source_residue = chain_residues[source_index]
            if is_cdr:
                selected.append(source_residue.key)
                mapped_cdr_positions += 1
        if expected_cdr_positions == 0 or mapped_cdr_positions / expected_cdr_positions < 0.80:
            raise ValueError(
                f"Insufficient mapped CDR coverage for chain {chain_id!r}: "
                f"{mapped_cdr_positions}/{expected_cdr_positions}"
            )

    if not selected:
        raise ValueError("Delivered IMGT mapping selected no antibody CDR residues")
    if len(selected) != len(set(selected)):
        raise ValueError("Delivered IMGT mapping selected duplicate CDR residues")
    return selected


def project_surface_scores(
    instance_id: str,
    canonical_sequence: str,
    source_residues: Sequence[ResidueRecord],
    mapping_payload: Mapping[str, Any],
    inference_result: Mapping[str, Any],
    elapsed_seconds: float,
) -> tuple[np.ndarray, np.ndarray, ProjectionAudit]:
    """Project native EpiFormer surface scores onto the canonical antigen axis."""
    index_map, method, coverage, identity = canonical_index_map(
        source_residues,
        canonical_sequence,
        mapping_payload,
    )
    details = inference_result["residue_details"]
    if not details:
        raise ValueError(f"EpiFormer returned no surface scores for {instance_id}")

    canonical_scores = np.full(len(canonical_sequence), np.nan, dtype=np.float32)
    observation_count = np.zeros(len(canonical_sequence), dtype=np.int32)
    mapped_native_scores: list[float] = []
    mapped_surface_count = 0

    for detail in details:
        key = (
            str(detail["chain_id"]),
            int(detail["residue_id"]),
            str(detail.get("insertion_code", " ")),
        )
        canonical_index = index_map.get(key)
        if canonical_index is None:
            continue
        score = float(detail["probability"])
        if not math.isfinite(score):
            raise ValueError(f"Non-finite native score for {instance_id} at {key}")
        mapped_surface_count += 1
        mapped_native_scores.append(score)
        if observation_count[canonical_index] == 0:
            canonical_scores[canonical_index] = score
        else:
            canonical_scores[canonical_index] = max(canonical_scores[canonical_index], score)
        observation_count[canonical_index] += 1

    if not mapped_native_scores:
        raise ValueError(f"No EpiFormer surface residue mapped for {instance_id}")
    mapped_surface_fraction = mapped_surface_count / len(details)
    if mapped_surface_fraction < 0.95:
        raise ValueError(
            f"Only {mapped_surface_count}/{len(details)} surface residues mapped for "
            f"{instance_id}; refusing a potentially misaligned projection"
        )
    minimum_score = float(min(mapped_native_scores))
    maximum_score = float(max(mapped_native_scores))
    observed_mask = observation_count > 0
    canonical_scores[~observed_mask] = minimum_score

    if not np.isfinite(canonical_scores).all():
        raise ValueError(f"Canonical projection contains non-finite scores for {instance_id}")
    audit = ProjectionAudit(
        instance_id=instance_id,
        mapping_method=method,
        canonical_length=len(canonical_sequence),
        source_residue_count=len(source_residues),
        model_surface_residue_count=len(details),
        antibody_cdr_residue_count=int(inference_result.get("n_antibody_residues", 0)),
        mapped_surface_residue_count=mapped_surface_count,
        mapped_surface_fraction=float(mapped_surface_fraction),
        unique_canonical_surface_count=int(observed_mask.sum()),
        filled_with_minimum_count=int((~observed_mask).sum()),
        source_mapping_coverage=float(coverage),
        source_mapping_identity=float(identity),
        model_minimum_score=minimum_score,
        model_maximum_score=maximum_score,
        elapsed_seconds=float(elapsed_seconds),
    )
    return canonical_scores, observed_mask, audit


def validate_score_archive(
    score_path: Path,
    audit_path: Path,
    canonical_length: int,
) -> bool:
    """Return whether a resumable per-instance result is complete and consistent."""
    if not score_path.is_file() or not audit_path.is_file():
        return False
    try:
        with np.load(score_path, allow_pickle=False) as archive:
            scores = archive["canonical_scores"]
            observed = archive["observed_mask"]
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False
    return bool(
        scores.shape == (canonical_length,)
        and observed.shape == (canonical_length,)
        and np.isfinite(scores).all()
        and audit.get("canonical_length") == canonical_length
    )


def infer_one_instance(
    *,
    instance_id: str,
    row: pd.Series,
    layout: BenchmarkLayout,
    output_root: Path,
    checkpoint: Path,
    resources: Any,
    run_inference: Any,
    sasa_threshold: float,
) -> ProjectionAudit:
    """Run one structure through EpiFormer and persist its canonical scores."""
    score_path = output_root / "residue_scores" / f"{instance_id}.npz"
    audit_path = output_root / "audits" / f"{instance_id}.json"
    canonical_sequence = str(row["ag_seq"])
    if validate_score_archive(score_path, audit_path, len(canonical_sequence)):
        return ProjectionAudit(**json.loads(audit_path.read_text(encoding="utf-8")))

    cif_path = layout.structures / Path(str(row["pdb_path"])).name
    mapping_path = layout.mappings / f"{instance_id}.json"
    if not cif_path.is_file():
        raise FileNotFoundError(f"Structure file not found: {cif_path}")
    if not mapping_path.is_file():
        raise FileNotFoundError(f"Canonical mapping not found: {mapping_path}")

    mapping_payload = json.loads(mapping_path.read_text(encoding="utf-8"))
    resolved = resolve_instance_chains(cif_path, row, mapping_payload)
    physical_antibody_components = list(resolved.antibody_components)
    antigen_assignments = [
        (chain_id, chr(ord("A") + index))
        for index, chain_id in enumerate(resolved.antigen_chain_ids)
    ]
    if len(antigen_assignments) > 26:
        raise ValueError("More than 26 antigen chains cannot be represented in PDB")
    antibody_assignments = [
        (chain_id, chain_type)
        for chain_id, chain_type in physical_antibody_components
    ]
    antibody_chain_types = {chain_type: chain_type for _, chain_type in antibody_assignments}
    temp_parent = output_root / "tmp"
    temp_parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    with tempfile.TemporaryDirectory(prefix=f"{instance_id}-", dir=temp_parent) as temp_dir:
        temp_dir_path = Path(temp_dir)
        antigen_pdb = temp_dir_path / "antigen.pdb"
        antibody_pdb = temp_dir_path / "antibody.pdb"
        write_resolved_chains(
            resolved.structure,
            antigen_pdb,
            antigen_assignments,
        )
        write_resolved_chains(
            resolved.structure,
            antibody_pdb,
            antibody_assignments,
        )
        source_residues = read_pdb_residues(antigen_pdb)
        antibody_residues = read_pdb_residues(antibody_pdb)
        cdr_residue_keys = benchmark_cdr_residue_keys(
            antibody_residues,
            mapping_payload,
            str(row["ab_seq"]),
            antibody_chain_types,
        )

        result = run_inference(
            antigen_pdb=str(antigen_pdb),
            antibody_pdb=str(antibody_pdb),
            checkpoint_path=str(checkpoint),
            threshold=0.3,
            device=str(resources.device),
            skip_filtering=False,
            sasa_threshold=sasa_threshold,
            resources=resources,
            antibody_chain_types=antibody_chain_types,
            antibody_cdr_residues=cdr_residue_keys,
        )
        scores, observed_mask, audit = project_surface_scores(
            instance_id=instance_id,
            canonical_sequence=canonical_sequence,
            source_residues=source_residues,
            mapping_payload=mapping_payload,
            inference_result=result,
            elapsed_seconds=time.monotonic() - started,
        )

    atomic_write_npz(
        score_path,
        canonical_scores=scores.astype(np.float32, copy=False),
        observed_mask=observed_mask.astype(np.bool_, copy=False),
    )
    atomic_write_json(audit_path, asdict(audit))
    return audit


def run_inference_shard(args: argparse.Namespace) -> int:
    """Execute one deterministic and resumable shard of the frozen split."""
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, --num-shards)")

    layout = BenchmarkLayout.from_root(args.benchmark_root)
    layout.validate()
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    split_ids = load_split(layout, args.split)
    instances = load_instances(layout)
    missing = sorted(set(split_ids) - set(instances.index))
    if missing:
        raise ValueError(f"Split IDs missing from instances.parquet: {missing[:5]}")

    if args.instance_id is not None:
        if args.instance_id not in split_ids:
            raise ValueError(f"{args.instance_id!r} is not in split {args.split!r}")
        shard_ids = [args.instance_id]
    else:
        shard_ids = split_ids[args.shard_index :: args.num_shards]
        if args.max_instances is not None:
            shard_ids = shard_ids[: args.max_instances]
    args.output_root.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(args.epiformer_code.resolve()))
    from inference import load_inference_resources, run_inference

    print(
        json.dumps(
            {
                "event": "shard_start",
                "shard_index": args.shard_index,
                "num_shards": args.num_shards,
                "instance_count": len(shard_ids),
                "checkpoint": str(checkpoint),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    resources = load_inference_resources(str(checkpoint), args.device)
    completed = 0
    failures: list[dict[str, str]] = []
    shard_started = time.monotonic()

    for position, instance_id in enumerate(shard_ids, start=1):
        try:
            audit = infer_one_instance(
                instance_id=instance_id,
                row=instances.loc[instance_id],
                layout=layout,
                output_root=args.output_root,
                checkpoint=checkpoint,
                resources=resources,
                run_inference=run_inference,
                sasa_threshold=args.sasa_threshold,
            )
            completed += 1
            print(
                json.dumps(
                    {
                        "event": "instance_complete",
                        "position": position,
                        "shard_total": len(shard_ids),
                        **asdict(audit),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        except Exception as error:
            failure = {
                "instance_id": instance_id,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
            failures.append(failure)
            atomic_write_json(
                args.output_root / "failures" / f"{instance_id}.json",
                failure,
            )
            print(json.dumps({"event": "instance_failed", **failure}), file=sys.stderr)
            if args.fail_fast:
                break

    summary = {
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "assigned": len(shard_ids),
        "completed": completed,
        "failed": len(failures),
        "elapsed_seconds": time.monotonic() - shard_started,
        "failed_instance_ids": [item["instance_id"] for item in failures],
    }
    atomic_write_json(
        args.output_root / "shards" / f"shard-{args.shard_index:03d}.json",
        summary,
    )
    print(json.dumps({"event": "shard_complete", **summary}, sort_keys=True), flush=True)
    return 2 if failures else 0


def preflight_benchmark(args: argparse.Namespace) -> int:
    """Validate all structure and canonical mappings before GPU inference."""
    layout = BenchmarkLayout.from_root(args.benchmark_root)
    layout.validate()
    split_ids = load_split(layout, args.split)
    instances = load_instances(layout)
    cdr_counts: Counter[int] = Counter()
    antigen_mapping_methods: Counter[str] = Counter()
    antigen_identities: list[float] = []
    antigen_coverages: list[float] = []
    failures: list[dict[str, str]] = []

    for position, instance_id in enumerate(split_ids, start=1):
        try:
            row = instances.loc[instance_id]
            cif_path = layout.structures / Path(str(row["pdb_path"])).name
            mapping_path = layout.mappings / f"{instance_id}.json"
            mapping_payload = json.loads(mapping_path.read_text(encoding="utf-8"))
            resolved = resolve_instance_chains(cif_path, row, mapping_payload)
            chain_components = list(resolved.antibody_components)
            antigen_residues = structure_residues_for_chains(
                resolved.structure,
                resolved.antigen_chain_ids,
            )
            antibody_residues = structure_residues_for_chains(
                resolved.structure,
                [chain_id for chain_id, _ in chain_components],
            )
            _, method, coverage, identity = canonical_index_map(
                antigen_residues,
                str(row["ag_seq"]),
                mapping_payload,
            )
            cdr_keys = benchmark_cdr_residue_keys(
                antibody_residues,
                mapping_payload,
                str(row["ab_seq"]),
                chain_components,
            )
            antigen_mapping_methods[method] += 1
            antigen_identities.append(identity)
            antigen_coverages.append(coverage)
            cdr_counts[len(cdr_keys)] += 1
            if position % 25 == 0 or position == len(split_ids):
                print(
                    json.dumps(
                        {
                            "event": "preflight_progress",
                            "validated": position,
                            "total": len(split_ids),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            del resolved
            if position % 25 == 0:
                gc.collect()
        except Exception as error:
            failures.append(
                {
                    "instance_id": instance_id,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                }
            )

    report = {
        "benchmark": "sabdab2",
        "split": args.split,
        "instance_count": len(split_ids),
        "validated_count": len(split_ids) - len(failures),
        "failure_count": len(failures),
        "failures": failures,
        "antigen_mapping_method_counts": {
            method: int(count) for method, count in antigen_mapping_methods.items()
        },
        "minimum_antigen_mapping_identity": (
            float(min(antigen_identities)) if antigen_identities else None
        ),
        "minimum_antigen_mapping_coverage": (
            float(min(antigen_coverages)) if antigen_coverages else None
        ),
        "cdr_residue_count_distribution": {
            str(count): int(frequency) for count, frequency in sorted(cdr_counts.items())
        },
    }
    atomic_write_json(args.output_root / "preflight.json", report)
    print(
        json.dumps(
            {
                "event": "preflight_complete",
                "validated": report["validated_count"],
                "failed": report["failure_count"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 2 if failures else 0


def row_group_instance_index(
    parquet_file: pq.ParquetFile,
    wanted_ids: set[str],
) -> dict[str, int]:
    """Index one-instance row groups without scanning pair payload columns."""
    field_index = parquet_file.schema_arrow.get_field_index("instance_id")
    if field_index < 0:
        raise ValueError("pairs.parquet has no instance_id column")
    index: dict[str, int] = {}
    unresolved: list[int] = []

    for row_group_index in range(parquet_file.num_row_groups):
        column = parquet_file.metadata.row_group(row_group_index).column(field_index)
        statistics = column.statistics
        instance_id: str | None = None
        if statistics is not None and statistics.has_min_max:
            minimum = statistics.min
            maximum = statistics.max
            if isinstance(minimum, bytes):
                minimum = minimum.decode("utf-8")
            if isinstance(maximum, bytes):
                maximum = maximum.decode("utf-8")
            if minimum == maximum:
                instance_id = str(minimum)
        if instance_id is None:
            unresolved.append(row_group_index)
            continue
        if instance_id in wanted_ids:
            if instance_id in index:
                raise ValueError(f"Multiple row groups found for {instance_id}")
            index[instance_id] = row_group_index

    for row_group_index in unresolved:
        values = parquet_file.read_row_group(
            row_group_index,
            columns=["instance_id"],
        ).column("instance_id").unique().to_pylist()
        if len(values) != 1:
            raise ValueError(
                f"Pair row group {row_group_index} spans multiple instances: {values[:5]}"
            )
        instance_id = str(values[0])
        if instance_id in wanted_ids:
            if instance_id in index:
                raise ValueError(f"Multiple row groups found for {instance_id}")
            index[instance_id] = row_group_index

    missing = sorted(wanted_ids - set(index))
    if missing:
        raise ValueError(f"No pair row group found for split instances: {missing[:5]}")
    return index


def safe_binary_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
) -> dict[str, float | None]:
    """Compute binary ranking metrics with explicit degenerate-case handling."""
    labels = np.asarray(labels, dtype=np.bool_)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.shape != scores.shape or labels.ndim != 1:
        raise ValueError("Metric labels and scores must be aligned one-dimensional arrays")
    if not np.isfinite(scores).all():
        raise ValueError("Metric scores contain non-finite values")
    if labels.size == 0 or np.unique(labels).size < 2:
        return {"auroc": None, "auprc": None}
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
    }


def summarize_values(
    values: Iterable[float | None],
) -> dict[str, float | int | None]:
    """Summarize finite values while retaining the valid sample count."""
    array = np.asarray([value for value in values if value is not None], dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {"count": 0, "mean": None, "std": None}
    return {
        "count": int(finite.size),
        "mean": float(finite.mean()),
        "std": float(finite.std(ddof=0)),
    }


def assemble_predictions(args: argparse.Namespace) -> int:
    """Assemble exact candidate predictions and compute epitope projection metrics."""
    layout = BenchmarkLayout.from_root(args.benchmark_root)
    layout.validate()
    split_ids = load_split(layout, args.split)
    if args.max_instances is not None:
        if args.max_instances <= 0:
            raise ValueError("--max-instances must be positive")
        split_ids = split_ids[: args.max_instances]
    instances = load_instances(layout)
    pairs_path = layout.frozen / "pairs.parquet"
    pair_file = pq.ParquetFile(pairs_path)
    row_groups = row_group_instance_index(pair_file, set(split_ids))

    missing_scores = [
        instance_id
        for instance_id in split_ids
        if not validate_score_archive(
            args.output_root / "residue_scores" / f"{instance_id}.npz",
            args.output_root / "audits" / f"{instance_id}.json",
            len(str(instances.loc[instance_id, "ag_seq"])),
        )
    ]
    if missing_scores:
        raise ValueError(
            f"Cannot assemble predictions; {len(missing_scores)} instances are missing or "
            f"invalid. First examples: {missing_scores[:5]}"
        )

    predictions_path = args.output_root / "predictions.parquet"
    partial_path = predictions_path.with_name(f".{predictions_path.name}.partial")
    partial_path.unlink(missing_ok=True)
    writer = pq.ParquetWriter(
        partial_path,
        PREDICTION_SCHEMA,
        compression="zstd",
        compression_level=6,
        use_dictionary=["instance_id"],
        write_statistics=True,
    )

    global_labels: list[np.ndarray] = []
    global_scores: list[np.ndarray] = []
    per_instance_metrics: list[dict[str, Any]] = []
    total_rows = 0
    try:
        for position, instance_id in enumerate(split_ids, start=1):
            score_path = args.output_root / "residue_scores" / f"{instance_id}.npz"
            with np.load(score_path, allow_pickle=False) as archive:
                canonical_scores = archive["canonical_scores"].astype(np.float32, copy=False)

            pair_table = pair_file.read_row_group(
                row_groups[instance_id],
                columns=["instance_id", "ab_res_idx", "ag_res_idx", "label"],
            )
            pair_instances = pair_table.column("instance_id").unique().to_pylist()
            if pair_instances != [instance_id]:
                raise ValueError(
                    f"Unexpected pair row group identity for {instance_id}: {pair_instances}"
                )
            ab_indices = pair_table.column("ab_res_idx").to_numpy(zero_copy_only=False)
            ag_indices = pair_table.column("ag_res_idx").to_numpy(zero_copy_only=False)
            pair_labels = pair_table.column("label").to_numpy(zero_copy_only=False)
            if ag_indices.size == 0:
                raise ValueError(f"No finite candidate pairs found for {instance_id}")
            if int(ag_indices.min()) < 0 or int(ag_indices.max()) >= len(canonical_scores):
                raise IndexError(f"Candidate antigen index is out of range for {instance_id}")

            pair_scores = canonical_scores[ag_indices]
            if not np.isfinite(pair_scores).all():
                raise ValueError(f"Non-finite projected pair score for {instance_id}")
            output_table = pa.Table.from_arrays(
                [
                    pair_table.column("instance_id"),
                    pa.array(ab_indices, type=pa.int64()),
                    pa.array(ag_indices, type=pa.int64()),
                    pa.array(pair_scores, type=pa.float32()),
                ],
                schema=PREDICTION_SCHEMA,
            )
            writer.write_table(output_table)
            total_rows += len(output_table)

            candidate_antigen_indices = np.unique(ag_indices)
            epitope_labels = np.zeros(len(canonical_scores), dtype=np.bool_)
            np.logical_or.at(epitope_labels, ag_indices, pair_labels)
            marginal_labels = epitope_labels[candidate_antigen_indices]
            marginal_scores = canonical_scores[candidate_antigen_indices]
            rank_metrics = safe_binary_metrics(marginal_labels, marginal_scores)
            threshold_predictions = marginal_scores > args.threshold
            metrics = {
                "instance_id": instance_id,
                "candidate_antigen_residues": int(candidate_antigen_indices.size),
                "positive_epitope_residues": int(marginal_labels.sum()),
                **rank_metrics,
                "f1_at_threshold": float(
                    f1_score(marginal_labels, threshold_predictions, zero_division=0)
                ),
                "mcc_at_threshold": float(
                    matthews_corrcoef(marginal_labels, threshold_predictions)
                ),
            }
            per_instance_metrics.append(metrics)
            global_labels.append(marginal_labels)
            global_scores.append(marginal_scores)
            print(
                json.dumps(
                    {
                        "event": "instance_assembled",
                        "position": position,
                        "total": len(split_ids),
                        "instance_id": instance_id,
                        "pair_rows": len(output_table),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    except Exception:
        writer.close()
        partial_path.unlink(missing_ok=True)
        raise
    else:
        writer.close()

    written_file = pq.ParquetFile(partial_path)
    if written_file.metadata.num_rows != total_rows:
        partial_path.unlink(missing_ok=True)
        raise ValueError(
            f"Prediction row-count mismatch: wrote {written_file.metadata.num_rows}, "
            f"expected {total_rows}"
        )
    if written_file.schema_arrow != PREDICTION_SCHEMA:
        partial_path.unlink(missing_ok=True)
        raise ValueError(f"Unexpected prediction schema: {written_file.schema_arrow}")
    os.replace(partial_path, predictions_path)

    concatenated_labels = np.concatenate(global_labels)
    concatenated_scores = np.concatenate(global_scores)
    micro = safe_binary_metrics(concatenated_labels, concatenated_scores)
    micro_predictions = concatenated_scores > args.threshold
    micro.update(
        {
            "f1_at_threshold": float(
                f1_score(concatenated_labels, micro_predictions, zero_division=0)
            ),
            "mcc_at_threshold": float(
                matthews_corrcoef(concatenated_labels, micro_predictions)
            ),
            "residue_count": int(concatenated_labels.size),
            "positive_count": int(concatenated_labels.sum()),
        }
    )
    macro = {
        metric: summarize_values(item[metric] for item in per_instance_metrics)
        for metric in ("auroc", "auprc", "f1_at_threshold", "mcc_at_threshold")
    }
    score_payload = {
        "benchmark": "sabdab2",
        "split": args.split,
        "projection": "epitope",
        "threshold": args.threshold,
        "instance_count": len(split_ids),
        "is_partial_split": args.max_instances is not None,
        "prediction_row_count": total_rows,
        "prediction_schema": str(PREDICTION_SCHEMA),
        "micro": micro,
        "macro_per_instance": macro,
        "per_instance": per_instance_metrics,
        "note": (
            "EpiFormer is an epitope-residue model. Candidate-pair scores are the "
            "frozen epitope projection score[ag_res_idx] and do not represent a "
            "native pair-interaction output."
        ),
    }
    atomic_write_json(args.output_root / "score.json", score_payload)

    audits = [
        json.loads(
            (args.output_root / "audits" / f"{instance_id}.json").read_text(
                encoding="utf-8"
            )
        )
        for instance_id in split_ids
    ]
    checkpoint = args.checkpoint.resolve()
    provenance = {
        "method": "EpiFormer",
        "benchmark": "sabdab2",
        "split": args.split,
        "is_partial_split": args.max_instances is not None,
        "projection": "epitope",
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "benchmark_manifest_sha256": sha256_file(layout.frozen / "MANIFEST.json"),
        "sasa_threshold": args.sasa_threshold,
        "classification_threshold": args.threshold,
        "mapping_method_counts": {
            str(method): int(count)
            for method, count in pd.Series(
                [audit["mapping_method"] for audit in audits]
            ).value_counts().items()
        },
        "mean_mapping_identity": float(
            np.mean([audit["source_mapping_identity"] for audit in audits])
        ),
        "mean_mapping_coverage": float(
            np.mean([audit["source_mapping_coverage"] for audit in audits])
        ),
        "prediction_file": str(predictions_path),
        "prediction_sha256": sha256_file(predictions_path),
    }
    atomic_write_json(args.output_root / "provenance.json", provenance)
    print(json.dumps({"event": "assembly_complete", **micro}, sort_keys=True), flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface shared by GPU and assembly jobs."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--benchmark-root", type=Path, required=True)
    preflight.add_argument("--output-root", type=Path, required=True)
    preflight.add_argument("--split", default="test")
    preflight.set_defaults(handler=preflight_benchmark)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--benchmark-root", type=Path, required=True)
    common.add_argument("--output-root", type=Path, required=True)
    common.add_argument("--checkpoint", type=Path, required=True)
    common.add_argument("--split", default="test")
    common.add_argument("--sasa-threshold", type=float, default=5.0)

    infer = subparsers.add_parser("infer", parents=[common])
    infer.add_argument("--epiformer-code", type=Path, required=True)
    infer.add_argument("--device", default="cuda:0")
    infer.add_argument("--shard-index", type=int, required=True)
    infer.add_argument("--num-shards", type=int, required=True)
    infer.add_argument("--max-instances", type=int)
    infer.add_argument("--instance-id")
    infer.add_argument("--fail-fast", action="store_true")
    infer.set_defaults(handler=run_inference_shard)

    assemble = subparsers.add_parser("assemble", parents=[common])
    assemble.add_argument("--threshold", type=float, default=0.3)
    assemble.add_argument("--max-instances", type=int)
    assemble.set_defaults(handler=assemble_predictions)
    return parser


def main() -> int:
    """Dispatch the selected adapter command."""
    args = build_parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
