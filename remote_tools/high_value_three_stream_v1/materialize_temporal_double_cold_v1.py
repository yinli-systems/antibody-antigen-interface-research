#!/usr/bin/env python3
"""Open temporal labels once and materialize immutable FrameContact inputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch


CONTACT_CUTOFF_ANGSTROM = 4.5
STRUCTURE_ARTIFACT_FORMAT = "sabdab2_intra_chain_structure_cohort_v2"
STRUCTURE_CACHE_FORMAT = "sabdab2_intra_chain_structure_cohort_cache_v2"
CONTACT_ARTIFACT_FORMAT = "sabdab2_framecontact_positive_flat_v1"
CONTACT_CACHE_FORMAT = "sabdab2_framecontact_cohort_cache_v2"


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha(path: Path, expected: str, label: str) -> None:
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"{label} SHA-256 differs: {observed}")


def atomic_write(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(payload)
    os.replace(temporary, path)


def observed_chain(model: object, chain_id: str) -> list[dict[str, Any]]:
    from Bio.PDB.Polypeptide import is_aa
    from Bio.SeqUtils import seq1

    try:
        chain = model[chain_id]
    except KeyError as error:
        raise ValueError(f"mmCIF lacks declared chain {chain_id!r}") from error
    rows: list[dict[str, Any]] = []
    for residue in chain:
        if not is_aa(residue, standard=False):
            continue
        amino_acid = seq1(
            residue.get_resname(), custom_map={"MSE": "M"}, undef_code="X"
        ).upper()
        amino_acid = amino_acid if amino_acid in "ACDEFGHIKLMNPQRSTVWY" else "X"
        backbone = np.full((3, 3), np.nan, dtype=np.float32)
        if all(name in residue for name in ("N", "CA", "C")):
            candidate = np.stack(
                [residue[name].coord.astype(np.float32) for name in ("N", "CA", "C")]
            )
            if np.isfinite(candidate).all():
                backbone = candidate
        heavy_atoms: list[np.ndarray] = []
        for atom in residue.get_atoms():
            element = str(getattr(atom, "element", "")).strip().upper()
            if element == "H" or atom.get_name().upper().startswith("H"):
                continue
            coordinate = atom.coord.astype(np.float32)
            if np.isfinite(coordinate).all():
                heavy_atoms.append(coordinate)
        rows.append(
            {
                "amino_acid": amino_acid,
                "backbone": backbone,
                "heavy_atoms": (
                    np.stack(heavy_atoms).astype(np.float32)
                    if heavy_atoms
                    else np.empty((0, 3), dtype=np.float32)
                ),
            }
        )
    if not rows:
        raise ValueError(f"chain {chain_id!r} has no protein residues")
    return rows


def parse_instance(row: dict[str, Any]) -> dict[str, Any]:
    from Bio.PDB import MMCIFParser

    cif_path = Path(row["cif_path"])
    require_sha(cif_path, row["cif_sha256"], f"{row['instance_id']} mmCIF")
    model = next(
        MMCIFParser(QUIET=True)
        .get_structure(str(row["instance_id"]), cif_path)
        .get_models()
    )
    antibody_chain_ids = [str(value) for value in row["antibody_chain_ids"]]
    antigen_chain_ids = [str(value) for value in row["antigen_chain_ids"]]
    antibody_parts = [observed_chain(model, chain) for chain in antibody_chain_ids]
    antigen_parts = [observed_chain(model, chain) for chain in antigen_chain_ids]
    antibody = [residue for chain in antibody_parts for residue in chain]
    antigen = [residue for chain in antigen_parts for residue in chain]
    ab_sequence = "".join(residue["amino_acid"] for residue in antibody)
    ag_sequence = "".join(residue["amino_acid"] for residue in antigen)
    if (
        ab_sequence != row["ab_sequence"]
        or ag_sequence != row["ag_sequence"]
        or sha256_bytes(ab_sequence.encode("ascii")) != row["ab_sequence_sha256"]
        or sha256_bytes(ag_sequence.encode("ascii")) != row["ag_sequence_sha256"]
    ):
        raise ValueError(f"{row['instance_id']}: sequence changed after label-free freeze")
    return {
        "instance_id": str(row["instance_id"]),
        "pdb_id": str(row["pdb_id"]),
        "source_cif_sha256": str(row["cif_sha256"]),
        "ab_sequence": ab_sequence,
        "ag_sequence": ag_sequence,
        "ab_backbone": np.stack([residue["backbone"] for residue in antibody]),
        "ag_backbone": np.stack([residue["backbone"] for residue in antigen]),
        "ab_heavy_atoms": [residue["heavy_atoms"] for residue in antibody],
        "ag_heavy_atoms": [residue["heavy_atoms"] for residue in antigen],
        "ab_chain_ids": np.concatenate(
            [np.full(len(chain), index, dtype=np.int64) for index, chain in enumerate(antibody_parts)]
        ),
        "ag_chain_ids": np.concatenate(
            [np.full(len(chain), index, dtype=np.int64) for index, chain in enumerate(antigen_parts)]
        ),
    }


def frame_valid(backbone: np.ndarray) -> np.ndarray:
    finite = np.isfinite(backbone).all(axis=(-1, -2))
    safe = np.nan_to_num(backbone, nan=0.0)
    n_atom, ca_atom, c_atom = safe[:, 0], safe[:, 1], safe[:, 2]
    x_raw = c_atom - ca_atom
    x_norm = np.linalg.norm(x_raw, axis=-1)
    x_axis = x_raw / np.maximum(x_norm[:, None], 1e-8)
    n_raw = n_atom - ca_atom
    y_raw = n_raw - (n_raw * x_axis).sum(axis=-1, keepdims=True) * x_axis
    y_norm = np.linalg.norm(y_raw, axis=-1)
    return finite & (x_norm > 1e-5) & (y_norm > 1e-5)


def positive_contacts(
    antibody_atoms: list[np.ndarray], antigen_atoms: list[np.ndarray]
) -> np.ndarray:
    nonempty_antigen = [atoms for atoms in antigen_atoms if len(atoms)]
    if not nonempty_antigen:
        return np.empty(0, dtype=np.int64)
    antigen_flat = np.concatenate(nonempty_antigen, axis=0)
    antigen_owner = np.concatenate(
        [
            np.full(len(atoms), index, dtype=np.int64)
            for index, atoms in enumerate(antigen_atoms)
            if len(atoms)
        ]
    )
    positives: list[int] = []
    threshold2 = CONTACT_CUTOFF_ANGSTROM**2
    antigen_length = len(antigen_atoms)
    for antibody_index, atoms in enumerate(antibody_atoms):
        if not len(atoms):
            continue
        distances2 = ((atoms[:, None, :] - antigen_flat[None, :, :]) ** 2).sum(axis=2)
        atom_hits = np.flatnonzero((distances2 <= threshold2).any(axis=0))
        for antigen_index in np.unique(antigen_owner[atom_hits]):
            positives.append(antibody_index * antigen_length + int(antigen_index))
    return np.asarray(sorted(positives), dtype=np.int64)


def save_structure(root: Path, instance: dict[str, Any]) -> dict[str, object]:
    identifier = instance["instance_id"]
    identifier_sha = sha256_bytes(identifier.encode("utf-8"))
    relative = Path("artifacts") / identifier_sha[:2] / f"{identifier_sha}.pt"
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    ab_valid = frame_valid(instance["ab_backbone"])
    ag_valid = frame_valid(instance["ag_backbone"])
    if not ab_valid.any() or not ag_valid.any():
        raise ValueError(f"{identifier}: one partner has no valid backbone frame")
    artifact = {
        "format_version": STRUCTURE_ARTIFACT_FORMAT,
        "instance_id": identifier,
        "source_cif_sha256": instance["source_cif_sha256"],
        "antibody_sequence_sha256": sha256_bytes(instance["ab_sequence"].encode("ascii")),
        "antigen_sequence_sha256": sha256_bytes(instance["ag_sequence"].encode("ascii")),
        "antibody_backbone": torch.from_numpy(instance["ab_backbone"]).to(torch.float32),
        "antigen_backbone": torch.from_numpy(instance["ag_backbone"]).to(torch.float32),
        "antibody_chain_ids": torch.from_numpy(instance["ab_chain_ids"]),
        "antigen_chain_ids": torch.from_numpy(instance["ag_chain_ids"]),
        "antibody_frame_valid_mask": torch.from_numpy(ab_valid),
        "antigen_frame_valid_mask": torch.from_numpy(ag_valid),
        "antibody_missing_frame_mask": torch.from_numpy(~ab_valid),
        "antigen_missing_frame_mask": torch.from_numpy(~ag_valid),
        "antibody_axis_projections": [],
    }
    torch.save(artifact, target)
    return {
        "instance_id": identifier,
        "pdb_id": instance["pdb_id"],
        "artifact_path": relative.as_posix(),
        "artifact_sha256": sha256_file(target),
        "artifact_size_bytes": target.stat().st_size,
        "source_cif_sha256": instance["source_cif_sha256"],
        "antibody_sequence_sha256": artifact["antibody_sequence_sha256"],
        "antigen_sequence_sha256": artifact["antigen_sequence_sha256"],
        "antibody_length": len(instance["ab_sequence"]),
        "antigen_length": len(instance["ag_sequence"]),
        "antibody_valid_frame_count": int(ab_valid.sum()),
        "antigen_valid_frame_count": int(ag_valid.sum()),
        "antibody_axis_projection_count": 0,
        "antibody_axis_projections": [],
    }


def save_contact(
    root: Path,
    identifier: str,
    antibody_length: int,
    antigen_length: int,
    positives: np.ndarray,
) -> dict[str, object]:
    identifier_sha = sha256_bytes(identifier.encode("utf-8"))
    relative = Path("artifacts") / identifier_sha[:2] / f"{identifier_sha}.npz"
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        format_version=np.asarray(CONTACT_ARTIFACT_FORMAT),
        instance_id=np.asarray(identifier),
        antibody_length=np.asarray(antibody_length, dtype=np.int64),
        antigen_length=np.asarray(antigen_length, dtype=np.int64),
        candidate_pair_count=np.asarray(antibody_length * antigen_length, dtype=np.int64),
        positive_flat_indices=positives,
    )
    return {
        "instance_id": identifier,
        "antibody_length": antibody_length,
        "antigen_length": antigen_length,
        "candidate_pair_count": antibody_length * antigen_length,
        "positive_pair_count": int(len(positives)),
        "artifact_path": relative.as_posix(),
        "artifact_sha256": sha256_file(target),
    }


def materialize_embeddings(
    sequences: dict[str, str], root: Path, model_root: Path, source_root: Path
) -> dict[str, object]:
    sys.path.insert(0, str(source_root / "src"))
    from interaction_first.esm_embeddings import (  # type: ignore
        EMBED_DIM,
        ESM_MODEL_NAME,
        ESM_MODEL_REVISION,
        ESM_MODEL_WEIGHT_SHA256,
        atomic_torch_save,
        embed_sequence,
        embedding_path,
        validate_embedding_artifact,
    )
    from transformers import EsmModel, EsmTokenizer

    require_sha(model_root / "model.safetensors", ESM_MODEL_WEIGHT_SHA256, "ESM-2 weight")
    tokenizer = EsmTokenizer.from_pretrained(model_root, local_files_only=True)
    model = EsmModel.from_pretrained(model_root, local_files_only=True).cuda().eval()
    if int(model.config.hidden_size) != EMBED_DIM:
        raise ValueError("ESM-2 dimension differs")
    for parameter in model.parameters():
        parameter.requires_grad = False
    records = []
    for index, digest in enumerate(sorted(sequences), start=1):
        path = embedding_path(root, digest)
        if path.exists():
            raise FileExistsError(path)
        artifact = embed_sequence(sequences[digest], tokenizer, model, torch.device("cuda"))
        validate_embedding_artifact(artifact, expected_key=digest)
        atomic_torch_save(artifact, path)
        records.append(
            {
                "sequence_sha256": digest,
                "sequence_length": len(sequences[digest]),
                "artifact_path": path.relative_to(root).as_posix(),
                "artifact_sha256": sha256_file(path),
            }
        )
        print(json.dumps({"phase": "esm2", "done": index, "total": len(sequences)}), flush=True)
    return {
        "schema_version": "temporal_double_cold_esm_embedding_manifest_v1",
        "model_name": ESM_MODEL_NAME,
        "model_revision": ESM_MODEL_REVISION,
        "model_weight_sha256": ESM_MODEL_WEIGHT_SHA256,
        "embedding_dimension": EMBED_DIM,
        "selected_sequence_count": len(sequences),
        "complete": True,
        "records": records,
        "task_labels_read_for_embedding": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("SLURM_JOB_PARTITION") != "gpu_4090":
        raise RuntimeError("gpu_4090 is required")
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 4090":
        raise RuntimeError("one RTX 4090 is required")
    require_sha(args.protocol, args.protocol_sha256, "measurement protocol")
    protocol = json.loads(args.protocol.read_text())
    if (
        protocol.get("status") != "FROZEN_AFTER_LABEL_FREE_GATE_BEFORE_SINGLE_LABEL_OPENING"
        or protocol.get("opening", {}).get("opening_count_before_materialization") != 0
        or protocol.get("opening", {}).get("maximum") != 1
        or protocol.get("execution", {}).get("materializer_sha256") != sha256_file(Path(__file__))
    ):
        raise ValueError("temporal measurement authority differs")
    label_free_path = Path(protocol["label_free_manifest"])
    require_sha(label_free_path, protocol["label_free_manifest_sha256"], "label-free manifest")
    label_free = json.loads(label_free_path.read_text())
    selected = set(protocol["cohort"]["instance_ids"])
    rows = [row for row in label_free["instances"] if row["instance_id"] in selected]
    if (
        len(rows) != protocol["cohort"]["instances"]
        or any(not row.get("double_cold") for row in rows)
        or args.output_root.exists()
        or args.output_root.is_symlink()
    ):
        raise ValueError("temporal selected cohort differs or output exists")

    temporary = args.output_root.with_name(args.output_root.name + f".tmp.{os.getpid()}")
    temporary.mkdir(parents=True)
    structure_records: list[dict[str, object]] = []
    contact_records: list[dict[str, object]] = []
    sequences: dict[str, str] = {}
    pdb_by_instance: dict[str, str] = {}
    for index, row in enumerate(sorted(rows, key=lambda value: value["instance_id"]), start=1):
        instance = parse_instance(row)
        identifier = instance["instance_id"]
        ab_sha = sha256_bytes(instance["ab_sequence"].encode("ascii"))
        ag_sha = sha256_bytes(instance["ag_sequence"].encode("ascii"))
        sequences.setdefault(ab_sha, instance["ab_sequence"])
        sequences.setdefault(ag_sha, instance["ag_sequence"])
        pdb_by_instance[identifier] = instance["pdb_id"]
        positives = positive_contacts(instance["ab_heavy_atoms"], instance["ag_heavy_atoms"])
        structure_records.append(save_structure(temporary / "structures", instance))
        contact_records.append(
            save_contact(
                temporary / "contacts",
                identifier,
                len(instance["ab_sequence"]),
                len(instance["ag_sequence"]),
                positives,
            )
        )
        print(
            json.dumps(
                {
                    "phase": "single_opening_materialization",
                    "done": index,
                    "total": len(rows),
                    "instance": identifier,
                    "positive_pairs": int(len(positives)),
                }
            ),
            flush=True,
        )

    split_path = temporary / "TEMPORAL_DOUBLE_COLD_SPLIT.csv"
    with split_path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["instance_id", "role"])
        writer.writeheader()
        for identifier in sorted(selected):
            writer.writerow({"instance_id": identifier, "role": "test"})
    structure_manifest = {
        "format_version": STRUCTURE_CACHE_FORMAT,
        "authority": {
            "measurement_protocol_sha256": args.protocol_sha256,
            "label_free_manifest_sha256": protocol["label_free_manifest_sha256"],
            "bound_complex_derived_intra_chain_backbone_geometry": True,
            "cross_chain_coordinates_are_not_model_inputs": True,
        },
        "atom_order": ["N", "CA", "C"],
        "counts": {"records": len(structure_records), "exclusions": 0},
        "records": structure_records,
        "exclusions": [],
    }
    atomic_write(temporary / "structures" / "STRUCTURE_CACHE_MANIFEST.json", canonical_bytes(structure_manifest))
    contact_manifest = {
        "format_version": CONTACT_CACHE_FORMAT,
        "authority": {
            "structure_manifest_sha256": sha256_file(temporary / "structures" / "STRUCTURE_CACHE_MANIFEST.json"),
            "measurement_protocol_sha256": args.protocol_sha256,
            "contact_definition": "minimum inter-residue non-hydrogen atom distance <= 4.5 angstrom",
            "candidate_grid_invariant": "complete_cartesian_product_exactly_once",
            "single_temporal_label_opening": True,
        },
        "counts": {
            "records": len(contact_records),
            "candidate_pairs": sum(int(row["candidate_pair_count"]) for row in contact_records),
            "positive_pairs": sum(int(row["positive_pair_count"]) for row in contact_records),
        },
        "records": contact_records,
    }
    atomic_write(temporary / "contacts" / "CONTACT_CACHE_MANIFEST.json", canonical_bytes(contact_manifest))

    random.seed(20260827)
    np.random.seed(20260827)
    torch.manual_seed(20260827)
    torch.cuda.manual_seed_all(20260827)
    torch.use_deterministic_algorithms(True, warn_only=True)
    embedding_manifest = materialize_embeddings(
        sequences, temporary / "embeddings", args.model_root, args.source_root
    )
    atomic_write(
        temporary / "embeddings" / "TEMPORAL_DOUBLE_COLD_ESM_EMBEDDING_MANIFEST.json",
        canonical_bytes(embedding_manifest),
    )
    data_manifest = {
        "format_version": "framecontact_temporal_double_cold_data_manifest_v1",
        "status": "PASS_SINGLE_OPENING_TEMPORAL_DATA_MATERIALIZED",
        "protocol_sha256": args.protocol_sha256,
        "label_free_manifest_sha256": protocol["label_free_manifest_sha256"],
        "split_sha256": sha256_file(split_path),
        "structure_manifest_sha256": sha256_file(temporary / "structures" / "STRUCTURE_CACHE_MANIFEST.json"),
        "contact_manifest_sha256": sha256_file(temporary / "contacts" / "CONTACT_CACHE_MANIFEST.json"),
        "embedding_manifest_sha256": sha256_file(temporary / "embeddings" / "TEMPORAL_DOUBLE_COLD_ESM_EMBEDDING_MANIFEST.json"),
        "counts": {
            "instances": len(rows),
            "unique_pdbs": len(set(pdb_by_instance.values())),
            "unique_sequences": len(sequences),
            "zero_positive_contact_instances": sum(int(row["positive_pair_count"]) == 0 for row in contact_records),
        },
        "pdb_by_instance": pdb_by_instance,
        "opening": {
            "opening_count": 1,
            "required_interventions_complete": True,
            "selection_performed_on_temporal": False,
            "calibration_refit": False,
        },
        "audit": {
            "old_exposed_test_access": False,
            "task_labels_derived_only_after_measurement_protocol_freeze": True,
            "cross_chain_coordinates_are_not_model_inputs": True,
        },
    }
    atomic_write(temporary / "TEMPORAL_DOUBLE_COLD_DATA_MANIFEST_V1.json", canonical_bytes(data_manifest))
    os.replace(temporary, args.output_root)
    result = args.output_root / "TEMPORAL_DOUBLE_COLD_DATA_MANIFEST_V1.json"
    receipt = {
        "format_version": "framecontact_temporal_double_cold_data_receipt_v1",
        "status": data_manifest["status"],
        "manifest_sha256": sha256_file(result),
        "protocol_sha256": args.protocol_sha256,
        "opening_count": 1,
        "selection_performed": False,
        "old_exposed_test_access": False,
    }
    atomic_write(
        args.output_root / "TEMPORAL_DOUBLE_COLD_DATA_RECEIPT_V1.json",
        canonical_bytes(receipt),
    )
    print(
        json.dumps(
            {
                **receipt,
                "receipt_sha256": sha256_file(args.output_root / "TEMPORAL_DOUBLE_COLD_DATA_RECEIPT_V1.json"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
