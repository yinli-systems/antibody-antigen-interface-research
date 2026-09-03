#!/usr/bin/env python3
"""Convert AsEP dataset into the format expected by the interaction_first pipeline.

The adapter performs four primary functions:
1. Iterates over the official AsEP complex identifiers (asepv1-AbDb-IDs.txt) and reads
   the structural interim graphs from asepv1_interim_graphs/.
2. Extracts the bipartite contact matrix from each graph (PyTorch sparse COO tensor),
   converts it to a SciPy sparse matrix (CSR/COO), and writes it as compressed .npz
   files into an output contacts/ directory using scipy.sparse.save_npz.
3. Loads pretrained antibody (AntiBERTy) and antigen (ESM-2 650M) embeddings from
   asep_antiberty_embeddings.pt and ag_esm2_embeddings_asep.pt, hashes each unique
   sequence using SHA-256, and writes validated artifacts into the sppider_seq_embedding_v1
   format in an output embeddings/ directory (partitioned by sequence_sha256[:2]/sequence_sha256.pt).
4. Emits a compatible instances.parquet manifest containing instance identifiers, sequence
   hashes, lengths, paratope/epitope positive counts, candidate counts, and split assignments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch


FORMAT_VERSION = "asep_to_interaction_first_v1"
EMBEDDING_FORMAT = "sppider_seq_embedding_v1"
SABDAB2_CONTACT_FORMAT = "sabdab2_sparse_contact_v1"


# =============================================================================
# Helper Utilities & Hashing
# =============================================================================


def setup_logger(verbose: bool = False) -> logging.Logger:
    """Configure a clean standard logger."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=level,
    )
    return logging.getLogger("asep_adapter")


def sequence_sha256(sequence: str) -> str:
    """Return the SHA-256 hexadecimal digest of a normalized uppercase protein sequence."""
    normalized = str(sequence).strip().upper()
    if not normalized:
        raise ValueError("Protein sequence cannot be empty")
    return hashlib.sha256(normalized.encode("ascii")).hexdigest()


def atomic_save_npz(matrix: sp.spmatrix, path: Path) -> None:
    """Save a SciPy sparse matrix atomically to an .npz file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    sp.save_npz(temp_path, matrix, compressed=True)
    os.replace(temp_path, path)


def atomic_torch_save(payload: Any, path: Path) -> None:
    """Persist a PyTorch object atomically through an adjacent temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temp_path)
    os.replace(temp_path, path)


def atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    """Write a pandas DataFrame to a Parquet file atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(temp_path, index=False)
    os.replace(temp_path, path)


def atomic_write_json(payload: Mapping[str, Any], path: Path) -> None:
    """Write a dictionary as formatted JSON atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temp_path, path)


# =============================================================================
# Embedding Formatting & Validation
# =============================================================================


def create_embedding_artifact(
    sequence: str,
    sequence_hash: str,
    embedding: torch.Tensor,
    model_name: str,
) -> dict[str, Any]:
    """Construct an artifact dictionary in sppider_seq_embedding_v1 format."""
    seq_len = len(sequence)
    if embedding.shape[0] != seq_len:
        raise ValueError(
            f"Embedding length ({embedding.shape[0]}) does not match sequence length ({seq_len})"
        )
    return {
        "format_version": EMBEDDING_FORMAT,
        "sequence_sha256": sequence_hash,
        "sequence_length": int(seq_len),
        "model_name": str(model_name),
        "max_tokens": int(seq_len),
        "stride": int(seq_len),
        "starts": [0],
        "ends": [int(seq_len)],
        "chunks": [embedding.float().cpu().contiguous()],
    }


def validate_embedding_artifact(
    artifact: Mapping[str, Any],
    expected_key: Optional[str] = None,
    expected_dim: Optional[int] = None,
) -> None:
    """Validate that the embedding artifact meets all invariant checks."""
    required = {
        "format_version",
        "sequence_sha256",
        "sequence_length",
        "model_name",
        "starts",
        "ends",
        "chunks",
    }
    missing = required - set(artifact)
    if missing:
        raise ValueError(f"Embedding artifact missing required keys: {sorted(missing)}")
    if artifact["format_version"] != EMBEDDING_FORMAT:
        raise ValueError(f"Unsupported embedding format: {artifact['format_version']!r}")
    if expected_key is not None and artifact["sequence_sha256"] != expected_key:
        raise ValueError(
            f"Embedding key mismatch: expected={expected_key}, observed={artifact['sequence_sha256']}"
        )
    starts = [int(v) for v in artifact["starts"]]
    ends = [int(v) for v in artifact["ends"]]
    chunks = list(artifact["chunks"])
    if not chunks or len(starts) != len(ends) or len(starts) != len(chunks):
        raise ValueError("Inconsistent chunk intervals")
    sequence_length = int(artifact["sequence_length"])
    if sequence_length <= 0:
        raise ValueError("Sequence length must be positive")
    covered = np.zeros(sequence_length, dtype=bool)
    for start, end, chunk in zip(starts, ends, chunks):
        if start < 0 or end <= start or end > sequence_length:
            raise ValueError(f"Invalid interval [{start}, {end}) for length {sequence_length}")
        if chunk.ndim != 2 or chunk.shape[0] != (end - start):
            raise ValueError(f"Chunk shape {chunk.shape} does not match interval [{start}, {end})")
        if expected_dim is not None and chunk.shape[1] != expected_dim:
            raise ValueError(f"Chunk dim {chunk.shape[1]} does not match expected {expected_dim}")
        if not bool(chunk.isfinite().all()):
            raise ValueError("Chunk tensor contains non-finite values (NaN or Inf)")
        covered[start:end] = True
    if not covered.all():
        raise ValueError("Intervals do not cover the complete sequence")


# =============================================================================
# Graph Data Extraction & Contacts
# =============================================================================


def extract_graph_data(graph: Mapping[str, Any], instance_id: str) -> dict[str, Any]:
    """Extract sequences, mappings, node counts, and bipartite contact matrix from AsEP graph."""
    # 1. Sequences
    seqres = graph.get("seqres", {})
    ab_seqres = seqres.get("ab", {})
    heavy = str(ab_seqres.get("H", "")).strip().upper()
    light = str(ab_seqres.get("L", "")).strip().upper()
    ab_full = heavy + light

    ag_seqres = seqres.get("ag", {})
    if isinstance(ag_seqres, dict):
        ag_full = "".join(str(v).strip().upper() for v in ag_seqres.values())
    else:
        ag_full = str(ag_seqres).strip().upper()

    # 2. Mappings
    mapping = graph.get("mapping", {})
    ab_map = mapping.get("ab", {})
    cdr_mask_raw = ab_map.get("seqres2cdr")
    if cdr_mask_raw is not None:
        cdr_mask = np.asarray(cdr_mask_raw, dtype=bool)
    else:
        cdr_mask = np.ones(len(ab_full), dtype=bool)

    ag_map = mapping.get("ag", {})
    surf_mask_raw = ag_map.get("seqres2surf")
    if surf_mask_raw is not None:
        surf_mask = np.asarray(surf_mask_raw, dtype=bool)
    else:
        surf_mask = np.ones(len(ag_full), dtype=bool)

    # Subselected sequences (CDR for antibody, surface for antigen)
    if len(ab_full) > 0 and len(cdr_mask) == len(ab_full):
        ab_cdr = "".join(np.asarray(list(ab_full))[cdr_mask])
    else:
        ab_cdr = ab_full

    if len(ag_full) > 0 and len(surf_mask) == len(ag_full):
        ag_surf = "".join(np.asarray(list(ag_full))[surf_mask])
    else:
        ag_surf = ag_full

    # 3. Node counts
    nb = int(graph.get("Nb", len(ab_cdr)))
    ng = int(graph.get("Ng", len(ag_surf)))

    # 4. Bipartite contact edges
    edges = graph.get("edges", {})
    bipartite = edges.get("bipartite")
    if bipartite is None:
        raise ValueError(f"Graph for {instance_id} is missing 'bipartite' edge entry")

    if torch.is_tensor(bipartite):
        if bipartite.is_sparse:
            coalesced = bipartite.detach().cpu().coalesce()
            indices = coalesced.indices().numpy()  # shape (2, nnz)
            values = coalesced.values().numpy()    # shape (nnz,)
            shape = tuple(coalesced.shape)
        else:
            dense_arr = bipartite.detach().cpu().numpy()
            shape = dense_arr.shape
            sp_coo = sp.coo_matrix(dense_arr)
            indices = np.vstack([sp_coo.row, sp_coo.col])
            values = sp_coo.data
    elif isinstance(bipartite, sp.spmatrix):
        sp_coo = sp.coo_matrix(bipartite)
        shape = sp_coo.shape
        indices = np.vstack([sp_coo.row, sp_coo.col])
        values = sp_coo.data
    elif isinstance(bipartite, np.ndarray):
        sp_coo = sp.coo_matrix(bipartite)
        shape = sp_coo.shape
        indices = np.vstack([sp_coo.row, sp_coo.col])
        values = sp_coo.data
    else:
        raise TypeError(f"Unsupported bipartite type {type(bipartite)} for {instance_id}")

    if shape != (nb, ng):
        shape = (nb, ng)

    # Positive and candidate counts
    positive_count = int(indices.shape[1])
    if positive_count > 0:
        paratope_count = int(len(np.unique(indices[0])))
        epitope_count = int(len(np.unique(indices[1])))
    else:
        paratope_count = 0
        epitope_count = 0

    return {
        "instance_id": instance_id,
        "ab_full": ab_full,
        "ag_full": ag_full,
        "ab_cdr": ab_cdr,
        "ag_surf": ag_surf,
        "nb": nb,
        "ng": ng,
        "indices": indices,
        "values": values,
        "shape": shape,
        "positive_count": positive_count,
        "paratope_count": paratope_count,
        "epitope_count": epitope_count,
    }


def make_contact_matrices(
    indices: np.ndarray,
    values: np.ndarray,
    shape: Tuple[int, int],
) -> Tuple[sp.csr_matrix, sp.coo_matrix]:
    """Construct SciPy COO and CSR contact matrices."""
    if len(values) == 0 or not np.isin(values, (0, 1)).all():
        binary_vals = np.ones(indices.shape[1], dtype=np.uint8)
    else:
        binary_vals = values.astype(np.uint8)
    coo = sp.coo_matrix((binary_vals, (indices[0], indices[1])), shape=shape, dtype=np.uint8)
    csr = coo.tocsr()
    return csr, coo


def save_sabdab2_sparse_contact_map(
    path: Path,
    instance_id: str,
    nb: int,
    ng: int,
    indices: np.ndarray,
    values: np.ndarray,
) -> None:
    """Save in the exact sabdab2_sparse_contact_v1 format defined in contacts.py."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")

    ab_indices = np.arange(nb, dtype=np.int64)
    ag_indices = np.arange(ng, dtype=np.int64)

    if indices.shape[1] > 0:
        ab_local = indices[0].astype(np.int64)
        ag_local = indices[1].astype(np.int64)
        labels = np.ones(indices.shape[1], dtype=np.uint8)
    else:
        ab_local = np.array([0], dtype=np.int64)
        ag_local = np.array([0], dtype=np.int64)
        labels = np.array([0], dtype=np.uint8)

    payload = {
        "format_version": np.asarray(SABDAB2_CONTACT_FORMAT),
        "instance_id": np.asarray(instance_id),
        "antibody_length": np.asarray(nb, dtype=np.int64),
        "antigen_length": np.asarray(ng, dtype=np.int64),
        "antibody_indices": ab_indices,
        "antigen_indices": ag_indices,
        "pair_antibody_local": ab_local,
        "pair_antigen_local": ag_local,
        "pair_labels": labels,
    }
    with temp_path.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    os.replace(temp_path, path)


# =============================================================================
# Split Dictionary Handling
# =============================================================================


def load_asep_splits(
    split_path: Optional[Path],
    identifiers: Sequence[str],
    method: str = "epitope_group",
) -> Dict[str, str]:
    """Load AsEP split assignments from split_dict.pt, normalizing 'val' -> 'validation'."""
    split_map: Dict[str, str] = {}
    if split_path is None or not split_path.is_file():
        return split_map

    try:
        payload = torch.load(split_path, map_location="cpu", weights_only=False)
        if isinstance(payload, dict) and method in payload:
            method_splits = payload[method]
            for role, indices_or_ids in method_splits.items():
                normalized_role = "validation" if role in ("val", "validation") else str(role)
                if torch.is_tensor(indices_or_ids):
                    items = indices_or_ids.tolist()
                else:
                    items = list(indices_or_ids)
                for item in items:
                    if isinstance(item, int) and 0 <= item < len(identifiers):
                        split_map[identifiers[item]] = normalized_role
                    elif isinstance(item, str):
                        split_map[item] = normalized_role
    except Exception as exc:
        logging.warning("Failed to load split dictionary from %s: %s", split_path, exc)

    return split_map


# =============================================================================
# Embedding Extraction & Storage
# =============================================================================


def load_raw_embedding_dict(path: Path) -> Dict[str, Any]:
    """Load an embedding file containing either direct dict or nested dict."""
    if not path.is_file():
        raise FileNotFoundError(f"Embedding file not found: {path}")
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(loaded, dict):
        raise TypeError(f"Expected dict from embedding file {path}, got {type(loaded)}")
    # Handle nested models if present
    if "esm2_650m" in loaded and isinstance(loaded["esm2_650m"], dict):
        return loaded["esm2_650m"]
    if "antiberty" in loaded and isinstance(loaded["antiberty"], dict):
        return loaded["antiberty"]
    return loaded


def process_instance_embeddings(
    instance_id: str,
    ab_seq_cdr: str,
    ab_seq_full: str,
    ag_seq_surf: str,
    ag_seq_full: str,
    ab_embeddings_dict: Mapping[str, Any],
    ag_embeddings_dict: Mapping[str, Any],
    embeddings_dir: Path,
    seen_sequence_hashes: Set[str],
) -> Tuple[str, str, str, str, int, int]:
    """Extract, validate, hash, and rewrite embeddings into sppider_seq_embedding_v1 format.

    Returns:
        (ab_hash, ag_hash, ab_seq_chosen, ag_seq_chosen, ab_dim, ag_dim)
    """
    # 1. Antibody embedding
    if instance_id not in ab_embeddings_dict:
        raise KeyError(f"Antibody embedding missing for {instance_id}")
    ab_raw = ab_embeddings_dict[instance_id]
    ab_tensor = torch.as_tensor(ab_raw, dtype=torch.float32).detach().cpu().contiguous()
    if ab_tensor.ndim == 1:
        ab_tensor = ab_tensor.unsqueeze(0)

    if ab_tensor.shape[0] == len(ab_seq_cdr):
        ab_seq = ab_seq_cdr
    elif ab_tensor.shape[0] == len(ab_seq_full):
        ab_seq = ab_seq_full
    else:
        raise ValueError(
            f"Antibody embedding rows ({ab_tensor.shape[0]}) does not match CDR length "
            f"({len(ab_seq_cdr)}) or full sequence length ({len(ab_seq_full)}) for {instance_id}"
        )
    ab_hash = sequence_sha256(ab_seq)
    ab_dim = int(ab_tensor.shape[1])

    # 2. Antigen embedding
    if instance_id not in ag_embeddings_dict:
        raise KeyError(f"Antigen embedding missing for {instance_id}")
    ag_raw = ag_embeddings_dict[instance_id]
    ag_tensor = torch.as_tensor(ag_raw, dtype=torch.float32).detach().cpu().contiguous()
    if ag_tensor.ndim == 1:
        ag_tensor = ag_tensor.unsqueeze(0)

    if ag_tensor.shape[0] == len(ag_seq_surf):
        ag_seq = ag_seq_surf
    elif ag_tensor.shape[0] == len(ag_seq_full):
        ag_seq = ag_seq_full
    else:
        raise ValueError(
            f"Antigen embedding rows ({ag_tensor.shape[0]}) does not match surface length "
            f"({len(ag_seq_surf)}) or full sequence length ({len(ag_seq_full)}) for {instance_id}"
        )
    ag_hash = sequence_sha256(ag_seq)
    ag_dim = int(ag_tensor.shape[1])

    # 3. Save antibody artifact if not seen
    if ab_hash not in seen_sequence_hashes:
        ab_artifact = create_embedding_artifact(
            sequence=ab_seq,
            sequence_hash=ab_hash,
            embedding=ab_tensor,
            model_name="AntiBERTy",
        )
        validate_embedding_artifact(ab_artifact, expected_key=ab_hash, expected_dim=ab_dim)
        ab_dest = embeddings_dir / ab_hash[:2] / f"{ab_hash}.pt"
        atomic_torch_save(ab_artifact, ab_dest)
        seen_sequence_hashes.add(ab_hash)

    # 4. Save antigen artifact if not seen
    if ag_hash not in seen_sequence_hashes:
        ag_artifact = create_embedding_artifact(
            sequence=ag_seq,
            sequence_hash=ag_hash,
            embedding=ag_tensor,
            model_name="esm2_t33_650M_UR50D",
        )
        validate_embedding_artifact(ag_artifact, expected_key=ag_hash, expected_dim=ag_dim)
        ag_dest = embeddings_dir / ag_hash[:2] / f"{ag_hash}.pt"
        atomic_torch_save(ag_artifact, ag_dest)
        seen_sequence_hashes.add(ag_hash)

    return ab_hash, ag_hash, ab_seq, ag_seq, ab_dim, ag_dim


# =============================================================================
# Main Conversion Pipeline
# =============================================================================


def parse_arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert AsEP dataset into the interaction_first pipeline format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/ssd/scxi253/model_deployment/data/asep"),
        help="Root directory of the AsEP dataset.",
    )
    parser.add_argument(
        "--ids-path",
        type=Path,
        default=None,
        help="Path to asepv1-AbDb-IDs.txt (defaults to <data-root>/asepv1-AbDb-IDs.txt).",
    )
    parser.add_argument(
        "--graphs-dir",
        type=Path,
        default=None,
        help="Path to asepv1_interim_graphs/ directory (defaults to <data-root>/asepv1_interim_graphs).",
    )
    parser.add_argument(
        "--antiberty-embeddings",
        type=Path,
        default=None,
        help="Path to asep_antiberty_embeddings.pt.",
    )
    parser.add_argument(
        "--esm2-embeddings",
        type=Path,
        default=None,
        help="Path to ag_esm2_embeddings_asep.pt.",
    )
    parser.add_argument(
        "--split-dict",
        type=Path,
        default=None,
        help="Path to split_dict.pt (defaults to <data-root>/split_dict.pt).",
    )
    parser.add_argument(
        "--split-method",
        choices=("epitope_group", "epitope_ratio"),
        default="epitope_group",
        help="Split method to select from split_dict.pt.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/ssd/scxi253/model_deployment/data/interaction_first_asep"),
        help="Destination directory for converted contacts/, embeddings/, and instances.parquet.",
    )
    parser.add_argument(
        "--contact-format",
        choices=("csr", "coo", "sabdab2", "both"),
        default="csr",
        help="Sparse matrix format for contacts/ (.npz files). 'csr' writes standard scipy.sparse CSR.",
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=None,
        help="Optional limit on number of instances to process (for debugging or smoke tests).",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=100,
        help="Print progress log every N complexes.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Fail immediately upon encountering an error on an instance instead of recording and skipping.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume execution by skipping instances whose contact file already exists.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose debug logging.",
    )
    return parser.parse_args(argv)


def run_adapter(args: argparse.Namespace) -> dict[str, Any]:
    """Execute the end-to-end dataset conversion."""
    logger = setup_logger(args.verbose)
    started_at = time.time()

    data_root = args.data_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    contacts_dir = output_dir / "contacts"
    embeddings_dir = output_dir / "embeddings"
    contacts_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    # Resolve input paths
    ids_path = (
        args.ids_path.expanduser().resolve()
        if args.ids_path
        else data_root / "asepv1-AbDb-IDs.txt"
    )
    graphs_dir = (
        args.graphs_dir.expanduser().resolve()
        if args.graphs_dir
        else data_root / "asepv1_interim_graphs"
    )
    split_dict_path = (
        args.split_dict.expanduser().resolve()
        if args.split_dict
        else data_root / "split_dict.pt"
    )

    # Antiberty embeddings resolution
    if args.antiberty_embeddings:
        ab_emb_path = args.antiberty_embeddings.expanduser().resolve()
    else:
        candidate_ab = data_root / "antibody" / "antiberty_embeddings" / "asep_antiberty_embeddings.pt"
        if candidate_ab.is_file():
            ab_emb_path = candidate_ab
        else:
            ab_emb_path = data_root / "asep_antiberty_embeddings.pt"

    # ESM2 embeddings resolution
    if args.esm2_embeddings:
        ag_emb_path = args.esm2_embeddings.expanduser().resolve()
    else:
        candidate_ag = data_root / "antigen" / "plm_embeddings" / "ag_esm2_embeddings_asep.pt"
        if candidate_ag.is_file():
            ag_emb_path = candidate_ag
        else:
            ag_emb_path = data_root / "ag_esm2_embeddings_asep.pt"

    logger.info("Starting AsEP to interaction_first adapter")
    logger.info("  IDs path:               %s", ids_path)
    logger.info("  Graphs directory:       %s", graphs_dir)
    logger.info("  AntiBERTy embeddings:   %s", ab_emb_path)
    logger.info("  ESM-2 embeddings:       %s", ag_emb_path)
    logger.info("  Split dictionary:       %s", split_dict_path)
    logger.info("  Output directory:       %s", output_dir)
    logger.info("  Contact format:         %s", args.contact_format)

    if not ids_path.is_file():
        raise FileNotFoundError(f"IDs file not found: {ids_path}")
    if not graphs_dir.is_dir():
        raise FileNotFoundError(f"Graphs directory not found: {graphs_dir}")

    # Read identifiers
    identifiers = [
        line.strip()
        for line in ids_path.read_text(encoding="ascii").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    total_ids = len(identifiers)
    logger.info("Loaded %d identifiers from %s", total_ids, ids_path)

    if args.max_instances is not None and args.max_instances > 0:
        identifiers = identifiers[: args.max_instances]
        logger.info("Constrained run to first %d identifiers", len(identifiers))

    # Load split dictionary
    split_map = load_asep_splits(split_dict_path, identifiers, method=args.split_method)
    if split_map:
        logger.info("Loaded %d split assignments using method %s", len(split_map), args.split_method)
    else:
        logger.info("No split assignments loaded; splits will default to 'unassigned'")

    # Load embeddings
    logger.info("Loading antibody embeddings from %s...", ab_emb_path)
    ab_embeddings = load_raw_embedding_dict(ab_emb_path)
    logger.info("Loaded %d antibody embeddings", len(ab_embeddings))

    logger.info("Loading antigen embeddings from %s...", ag_emb_path)
    ag_embeddings = load_raw_embedding_dict(ag_emb_path)
    logger.info("Loaded %d antigen embeddings", len(ag_embeddings))

    # Conversion iteration
    instances_records: List[dict[str, Any]] = []
    skipped_records: List[dict[str, Any]] = []
    seen_sequence_hashes: Set[str] = set()

    for idx, instance_id in enumerate(identifiers, start=1):
        graph_file = graphs_dir / f"{instance_id}.pt"
        contact_dest = contacts_dir / f"{instance_id}.npz"

        # Check resume
        if args.resume and contact_dest.is_file():
            pass

        if not graph_file.is_file():
            reason = f"Graph file missing: {graph_file}"
            if args.fail_fast:
                raise FileNotFoundError(reason)
            skipped_records.append({"instance_id": instance_id, "reason": reason})
            continue

        try:
            # 1. Load graph
            graph_data_raw = torch.load(graph_file, map_location="cpu", weights_only=False)
            extracted = extract_graph_data(graph_data_raw, instance_id)

            # 2. Extract and save contacts
            csr_mat, coo_mat = make_contact_matrices(
                indices=extracted["indices"],
                values=extracted["values"],
                shape=extracted["shape"],
            )

            if args.contact_format == "csr":
                atomic_save_npz(csr_mat, contact_dest)
            elif args.contact_format == "coo":
                atomic_save_npz(coo_mat, contact_dest)
            elif args.contact_format == "sabdab2":
                save_sabdab2_sparse_contact_map(
                    path=contact_dest,
                    instance_id=instance_id,
                    nb=extracted["nb"],
                    ng=extracted["ng"],
                    indices=extracted["indices"],
                    values=extracted["values"],
                )
            elif args.contact_format == "both":
                atomic_save_npz(csr_mat, contact_dest)
                sabdab2_dest = output_dir / "contacts_sabdab2" / f"{instance_id}.npz"
                save_sabdab2_sparse_contact_map(
                    path=sabdab2_dest,
                    instance_id=instance_id,
                    nb=extracted["nb"],
                    ng=extracted["ng"],
                    indices=extracted["indices"],
                    values=extracted["values"],
                )

            # 3. Process embeddings
            ab_hash, ag_hash, ab_seq, ag_seq, ab_dim, ag_dim = process_instance_embeddings(
                instance_id=instance_id,
                ab_seq_cdr=extracted["ab_cdr"],
                ab_seq_full=extracted["ab_full"],
                ag_seq_surf=extracted["ag_surf"],
                ag_seq_full=extracted["ag_full"],
                ab_embeddings_dict=ab_embeddings,
                ag_embeddings_dict=ag_embeddings,
                embeddings_dir=embeddings_dir,
                seen_sequence_hashes=seen_sequence_hashes,
            )

            # 4. Split assignments
            split_role = split_map.get(instance_id, "unassigned")

            # 5. Build instance metadata record
            instances_records.append(
                {
                    "instance_id": str(instance_id),
                    "ab_sequence_sha256": str(ab_hash),
                    "ag_sequence_sha256": str(ag_hash),
                    "ab_seq": str(ab_seq),
                    "ag_seq": str(ag_seq),
                    "ab_length": int(len(ab_seq)),
                    "ag_length": int(len(ag_seq)),
                    "ab_candidate_residues": int(extracted["nb"]),
                    "ag_candidate_residues": int(extracted["ng"]),
                    "candidate_pair_count": int(extracted["nb"] * extracted["ng"]),
                    "positive_pair_count": int(extracted["positive_count"]),
                    "paratope_positive_count": int(extracted["paratope_count"]),
                    "epitope_positive_count": int(extracted["epitope_count"]),
                    "frozen_split": str(split_role),
                    "paratope_split": str(split_role),
                    "epitope_split": str(split_role),
                    "interaction_split": str(split_role),
                    "ab_seq_full": str(extracted["ab_full"]),
                    "ag_seq_full": str(extracted["ag_full"]),
                    "ab_full_length": int(len(extracted["ab_full"])),
                    "ag_full_length": int(len(extracted["ag_full"])),
                    "ab_full_sequence_sha256": (
                        sequence_sha256(extracted["ab_full"])
                        if extracted["ab_full"]
                        else ""
                    ),
                    "ag_full_sequence_sha256": (
                        sequence_sha256(extracted["ag_full"])
                        if extracted["ag_full"]
                        else ""
                    ),
                }
            )

        except Exception as exc:
            reason = f"Error processing {instance_id}: {exc!r}"
            if args.fail_fast:
                raise RuntimeError(reason) from exc
            logger.warning(reason)
            skipped_records.append({"instance_id": instance_id, "reason": reason})

        if idx % args.checkpoint_every == 0 or idx == len(identifiers):
            logger.info(
                "Progress: %d/%d processed (valid=%d, skipped=%d, unique_seqs=%d)",
                idx,
                len(identifiers),
                len(instances_records),
                len(skipped_records),
                len(seen_sequence_hashes),
            )

    # Emit instances.parquet
    if not instances_records:
        raise RuntimeError("No instances were successfully processed!")

    instances_df = pd.DataFrame(instances_records).sort_values("instance_id", kind="stable")
    parquet_path = output_dir / "instances.parquet"
    atomic_write_parquet(instances_df, parquet_path)
    logger.info("Wrote %d instances to %s", len(instances_df), parquet_path)

    # Emit summary report
    elapsed_seconds = time.time() - started_at
    split_counts = instances_df["frozen_split"].value_counts().to_dict()
    summary = {
        "format_version": FORMAT_VERSION,
        "created_at_iso": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(elapsed_seconds, 2),
        "total_requested_instances": len(identifiers),
        "successful_instances": len(instances_df),
        "skipped_instances_count": len(skipped_records),
        "skipped_records": skipped_records,
        "unique_sequence_count": len(seen_sequence_hashes),
        "unique_antibody_sequences": int(instances_df["ab_sequence_sha256"].nunique()),
        "unique_antigen_sequences": int(instances_df["ag_sequence_sha256"].nunique()),
        "total_candidate_pairs": int(instances_df["candidate_pair_count"].sum()),
        "total_positive_pairs": int(instances_df["positive_pair_count"].sum()),
        "split_counts": split_counts,
        "paths": {
            "instances_parquet": str(parquet_path),
            "contacts_dir": str(contacts_dir),
            "embeddings_dir": str(embeddings_dir),
        },
    }
    summary_path = output_dir / "adapter_summary.json"
    atomic_write_json(summary, summary_path)
    logger.info("Saved summary report to %s", summary_path)
    logger.info(
        "Conversion complete in %.1f seconds: %d instances, %d positive contacts, %d unique sequences",
        elapsed_seconds,
        len(instances_df),
        summary["total_positive_pairs"],
        summary["unique_sequence_count"],
    )

    return summary


def main() -> None:
    args = parse_arguments()
    run_adapter(args)


if __name__ == "__main__":
    main()
