#!/usr/bin/env python3
"""Materialize fixed common-axis bound/unbound geometry and bound contacts."""

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


CONDITIONS = {
    "bound_bound": ("bound", "bound"),
    "unbound_ab_bound_ag": ("unbound", "bound"),
    "bound_ab_unbound_ag": ("bound", "unbound"),
    "unbound_unbound": ("unbound", "unbound"),
}
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


def protein_residues(path: Path) -> dict[str, list[dict[str, Any]]]:
    from Bio.PDB import PDBParser
    from Bio.PDB.Polypeptide import is_aa
    from Bio.SeqUtils import seq1

    model = next(PDBParser(QUIET=True).get_structure(path.stem, path).get_models())
    result: dict[str, list[dict[str, Any]]] = {}
    for chain in model:
        rows: list[dict[str, Any]] = []
        for residue in chain:
            if not is_aa(residue, standard=False):
                continue
            amino_acid = seq1(
                residue.get_resname(), custom_map={"MSE": "M"}, undef_code="X"
            ).upper()
            amino_acid = amino_acid if amino_acid in "ACDEFGHIKLMNPQRSTVWY" else "X"
            backbone = None
            if all(name in residue for name in ("N", "CA", "C")):
                candidate = np.stack(
                    [residue[name].coord.astype(np.float32) for name in ("N", "CA", "C")]
                )
                if np.isfinite(candidate).all():
                    backbone = candidate
            heavy_atoms = []
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
                    "heavy_atoms": np.stack(heavy_atoms) if heavy_atoms else np.empty((0, 3), np.float32),
                }
            )
        if rows:
            result[str(chain.id)] = rows
    return result


def common_axis(
    case: dict[str, Any],
    partner: str,
) -> dict[str, Any]:
    if partner not in {"antibody", "antigen"}:
        raise ValueError(partner)
    bound_role = f"{partner}_bound"
    unbound_role = f"{partner}_unbound"
    bound = protein_residues(Path(case["files"][bound_role]["path"]))
    unbound = protein_residues(Path(case["files"][unbound_role]["path"]))
    bound_chain_order = [row["chain_id"] for row in case["files"][bound_role]["chains"]]
    order = {chain: index for index, chain in enumerate(bound_chain_order)}
    assignments = sorted(
        case[f"{partner}_mapping"]["assignments"],
        key=lambda row: order[str(row["bound_chain_id"])],
    )
    sequence: list[str] = []
    chain_ids: list[int] = []
    bound_backbone: list[np.ndarray] = []
    unbound_backbone: list[np.ndarray] = []
    bound_heavy_atoms: list[np.ndarray] = []
    mapping_receipt: list[dict[str, object]] = []
    for chain_index, assignment in enumerate(assignments):
        bound_chain = str(assignment["bound_chain_id"])
        unbound_chain = str(assignment["unbound_chain_id"])
        if bound_chain not in bound or unbound_chain not in unbound:
            raise ValueError(f"{case['case_id']}: mapped chain is absent")
        retained = 0
        for bound_index, unbound_index in assignment["bound_to_unbound_index_pairs"]:
            left = bound[bound_chain][int(bound_index)]
            right = unbound[unbound_chain][int(unbound_index)]
            if (
                left["amino_acid"] != right["amino_acid"]
                or left["amino_acid"] == "X"
                or left["backbone"] is None
                or right["backbone"] is None
                or len(left["heavy_atoms"]) == 0
            ):
                continue
            sequence.append(left["amino_acid"])
            chain_ids.append(chain_index)
            bound_backbone.append(left["backbone"])
            unbound_backbone.append(right["backbone"])
            bound_heavy_atoms.append(left["heavy_atoms"])
            retained += 1
        mapping_receipt.append(
            {
                "bound_chain_id": bound_chain,
                "unbound_chain_id": unbound_chain,
                "retained_exact_residue_pairs_with_complete_backbone": retained,
            }
        )
    if not sequence:
        raise ValueError(f"{case['case_id']}: empty common {partner} axis")
    return {
        "sequence": "".join(sequence),
        "chain_ids": np.asarray(chain_ids, dtype=np.int64),
        "bound_backbone": np.stack(bound_backbone).astype(np.float32),
        "unbound_backbone": np.stack(unbound_backbone).astype(np.float32),
        "bound_heavy_atoms": bound_heavy_atoms,
        "mapping_receipt": mapping_receipt,
    }


def positive_contacts(
    antibody_atoms: list[np.ndarray], antigen_atoms: list[np.ndarray]
) -> np.ndarray:
    antigen_flat = np.concatenate(antigen_atoms, axis=0)
    antigen_owner = np.concatenate(
        [np.full(len(atoms), index, dtype=np.int64) for index, atoms in enumerate(antigen_atoms)]
    )
    positives: list[int] = []
    threshold2 = CONTACT_CUTOFF_ANGSTROM ** 2
    antigen_length = len(antigen_atoms)
    for antibody_index, atoms in enumerate(antibody_atoms):
        distances2 = ((atoms[:, None, :] - antigen_flat[None, :, :]) ** 2).sum(axis=2)
        atom_hits = np.flatnonzero((distances2 <= threshold2).any(axis=0))
        for antigen_index in np.unique(antigen_owner[atom_hits]):
            positives.append(antibody_index * antigen_length + int(antigen_index))
    return np.asarray(sorted(positives), dtype=np.int64)


def save_structure(
    root: Path,
    case_id: str,
    condition: str,
    antibody: dict[str, Any],
    antigen: dict[str, Any],
    source_digest: str,
) -> dict[str, object]:
    ab_sequence = antibody["sequence"]
    ag_sequence = antigen["sequence"]
    ab_sha = sha256_bytes(ab_sequence.encode("ascii"))
    ag_sha = sha256_bytes(ag_sequence.encode("ascii"))
    ab_state, ag_state = CONDITIONS[condition]
    ab_backbone = torch.from_numpy(antibody[f"{ab_state}_backbone"]).to(torch.float32)
    ag_backbone = torch.from_numpy(antigen[f"{ag_state}_backbone"]).to(torch.float32)
    identifier_sha = sha256_bytes(case_id.encode("utf-8"))
    relative = Path("artifacts") / identifier_sha[:2] / f"{identifier_sha}.pt"
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "format_version": STRUCTURE_ARTIFACT_FORMAT,
        "instance_id": case_id,
        "source_cif_sha256": source_digest,
        "antibody_sequence_sha256": ab_sha,
        "antigen_sequence_sha256": ag_sha,
        "antibody_backbone": ab_backbone,
        "antigen_backbone": ag_backbone,
        "antibody_chain_ids": torch.from_numpy(antibody["chain_ids"]),
        "antigen_chain_ids": torch.from_numpy(antigen["chain_ids"]),
        "antibody_frame_valid_mask": torch.ones(len(ab_sequence), dtype=torch.bool),
        "antigen_frame_valid_mask": torch.ones(len(ag_sequence), dtype=torch.bool),
        "antibody_missing_frame_mask": torch.zeros(len(ab_sequence), dtype=torch.bool),
        "antigen_missing_frame_mask": torch.zeros(len(ag_sequence), dtype=torch.bool),
        "antibody_axis_projections": [],
    }
    torch.save(artifact, target)
    return {
        "instance_id": case_id,
        "artifact_path": relative.as_posix(),
        "artifact_sha256": sha256_file(target),
        "artifact_size_bytes": target.stat().st_size,
        "source_cif_sha256": source_digest,
        "antibody_sequence_sha256": ab_sha,
        "antigen_sequence_sha256": ag_sha,
        "antibody_length": len(ab_sequence),
        "antigen_length": len(ag_sequence),
        "antibody_valid_frame_count": len(ab_sequence),
        "antigen_valid_frame_count": len(ag_sequence),
        "antibody_axis_projection_count": 0,
        "antibody_axis_projections": [],
    }


def save_contact(
    root: Path,
    case_id: str,
    antibody_length: int,
    antigen_length: int,
    positives: np.ndarray,
) -> dict[str, object]:
    identifier_sha = sha256_bytes(case_id.encode("utf-8"))
    relative = Path("artifacts") / identifier_sha[:2] / f"{identifier_sha}.npz"
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        format_version=np.asarray(CONTACT_ARTIFACT_FORMAT),
        instance_id=np.asarray(case_id),
        antibody_length=np.asarray(antibody_length, dtype=np.int64),
        antigen_length=np.asarray(antigen_length, dtype=np.int64),
        candidate_pair_count=np.asarray(antibody_length * antigen_length, dtype=np.int64),
        positive_flat_indices=positives,
    )
    return {
        "instance_id": case_id,
        "antibody_length": antibody_length,
        "antigen_length": antigen_length,
        "candidate_pair_count": antibody_length * antigen_length,
        "positive_pair_count": int(len(positives)),
        "artifact_path": relative.as_posix(),
        "artifact_sha256": sha256_file(target),
    }


def materialize_embeddings(
    sequences: dict[str, str],
    root: Path,
    model_root: Path,
    source_root: Path,
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

    weight_path = model_root / "model.safetensors"
    require_sha(weight_path, ESM_MODEL_WEIGHT_SHA256, "ESM-2 weight")
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
        if index % 10 == 0 or index == len(sequences):
            print(json.dumps({"phase": "esm2_embedding", "done": index, "total": len(sequences)}), flush=True)
    return {
        "schema_version": "bound_unbound_esm_embedding_manifest_v1",
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
    parser.add_argument("--label-free-manifest", type=Path, required=True)
    parser.add_argument("--label-free-manifest-sha256", required=True)
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
    require_sha(args.label_free_manifest, args.label_free_manifest_sha256, "label-free manifest")
    require_sha(args.protocol, args.protocol_sha256, "measurement protocol")
    protocol = json.loads(args.protocol.read_text())
    if (
        protocol.get("format_version") != "framecontact_bound_unbound_measurement_protocol_v1"
        or protocol.get("status") != "FROZEN_AFTER_LABEL_FREE_GATE_BEFORE_CONTACTS"
        or protocol.get("execution", {}).get("materializer_sha256") != sha256_file(Path(__file__))
    ):
        raise ValueError("measurement protocol differs")
    if args.output_root.exists() or args.output_root.is_symlink():
        raise FileExistsError(args.output_root)
    label_free = json.loads(args.label_free_manifest.read_text())
    if (
        label_free.get("status") != "PASS_BOUND_UNBOUND_LABEL_FREE_FREEZE"
        or label_free.get("audit", {}).get("contact_labels_computed") is not False
        or label_free.get("counts", {}).get("eligible_cases") != 52
    ):
        raise ValueError("label-free gate differs")
    eligible = set(label_free["eligible_case_ids"])
    cases = [case for case in label_free["cases"] if case["case_id"] in eligible]
    temporary = args.output_root.with_name(args.output_root.name + f".tmp.{os.getpid()}")
    temporary.mkdir(parents=True)
    sequences: dict[str, str] = {}
    condition_records: dict[str, list[dict[str, object]]] = {key: [] for key in CONDITIONS}
    contact_records: dict[str, list[dict[str, object]]] = {key: [] for key in CONDITIONS}
    common_axis_records = []
    for index, case in enumerate(sorted(cases, key=lambda row: row["case_id"]), start=1):
        antibody = common_axis(case, "antibody")
        antigen = common_axis(case, "antigen")
        ab_sha = sha256_bytes(antibody["sequence"].encode("ascii"))
        ag_sha = sha256_bytes(antigen["sequence"].encode("ascii"))
        sequences.setdefault(ab_sha, antibody["sequence"])
        sequences.setdefault(ag_sha, antigen["sequence"])
        positives = positive_contacts(antibody["bound_heavy_atoms"], antigen["bound_heavy_atoms"])
        common_axis_records.append(
            {
                "case_id": case["case_id"],
                "antibody_length": len(antibody["sequence"]),
                "antigen_length": len(antigen["sequence"]),
                "antibody_sequence_sha256": ab_sha,
                "antigen_sequence_sha256": ag_sha,
                "positive_pair_count": int(len(positives)),
                "antibody_mapping": antibody["mapping_receipt"],
                "antigen_mapping": antigen["mapping_receipt"],
            }
        )
        for condition in CONDITIONS:
            structure_root = temporary / "structures" / condition
            source_digest = sha256_bytes(
                canonical_bytes(
                    {
                        "case_id": case["case_id"],
                        "condition": condition,
                        "file_sha256": {role: value["sha256"] for role, value in case["files"].items()},
                    }
                )
            )
            structure_record = save_structure(
                structure_root, case["case_id"], condition, antibody, antigen, source_digest
            )
            condition_records[condition].append(structure_record)
            contact_records[condition].append(
                save_contact(
                    temporary / "contacts" / condition,
                    case["case_id"],
                    len(antibody["sequence"]),
                    len(antigen["sequence"]),
                    positives,
                )
            )
        print(json.dumps({"phase": "common_axis_and_contacts", "done": index, "total": len(cases), "case": case["case_id"], "positive_pairs": int(len(positives))}), flush=True)

    split_path = temporary / "BOUND_UNBOUND_SPLIT.csv"
    with split_path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["instance_id", "role"])
        writer.writeheader()
        for case in sorted(eligible):
            writer.writerow({"instance_id": case, "role": "test"})
    for condition in CONDITIONS:
        structure_root = temporary / "structures" / condition
        structure_manifest = {
            "format_version": STRUCTURE_CACHE_FORMAT,
            "authority": {
                "label_free_manifest_sha256": args.label_free_manifest_sha256,
                "measurement_protocol_sha256": args.protocol_sha256,
                "condition": condition,
                "common_axis_exact_sequence_and_complete_backbone_intersection": True,
            },
            "atom_order": ["N", "CA", "C"],
            "counts": {"records": len(condition_records[condition]), "exclusions": 0},
            "records": condition_records[condition],
            "exclusions": [],
        }
        atomic_write(structure_root / "STRUCTURE_CACHE_MANIFEST.json", canonical_bytes(structure_manifest))
        contact_root = temporary / "contacts" / condition
        contact_manifest = {
            "format_version": CONTACT_CACHE_FORMAT,
            "authority": {
                "structure_manifest_sha256": sha256_file(structure_root / "STRUCTURE_CACHE_MANIFEST.json"),
                "measurement_protocol_sha256": args.protocol_sha256,
                "bound_state_heavy_atom_cutoff_angstrom": CONTACT_CUTOFF_ANGSTROM,
                "candidate_grid_invariant": "complete_cartesian_product_exactly_once",
            },
            "counts": {
                "records": len(contact_records[condition]),
                "candidate_pairs": sum(int(row["candidate_pair_count"]) for row in contact_records[condition]),
                "positive_pairs": sum(int(row["positive_pair_count"]) for row in contact_records[condition]),
            },
            "records": contact_records[condition],
        }
        atomic_write(contact_root / "CONTACT_CACHE_MANIFEST.json", canonical_bytes(contact_manifest))

    random.seed(20260828)
    np.random.seed(20260828)
    torch.manual_seed(20260828)
    torch.cuda.manual_seed_all(20260828)
    torch.use_deterministic_algorithms(True, warn_only=True)
    embedding_manifest = materialize_embeddings(
        sequences,
        temporary / "embeddings",
        args.model_root,
        args.source_root,
    )
    atomic_write(
        temporary / "embeddings" / "BOUND_UNBOUND_ESM_EMBEDDING_MANIFEST.json",
        canonical_bytes(embedding_manifest),
    )
    data_manifest = {
        "format_version": "framecontact_bound_unbound_measurement_data_manifest_v1",
        "status": "PASS_BOUND_UNBOUND_MEASUREMENT_DATA_MATERIALIZED",
        "protocol_sha256": args.protocol_sha256,
        "label_free_manifest_sha256": args.label_free_manifest_sha256,
        "split_sha256": sha256_file(split_path),
        "embedding_manifest_sha256": sha256_file(temporary / "embeddings" / "BOUND_UNBOUND_ESM_EMBEDDING_MANIFEST.json"),
        "conditions": {
            condition: {
                "structure_manifest_sha256": sha256_file(temporary / "structures" / condition / "STRUCTURE_CACHE_MANIFEST.json"),
                "contact_manifest_sha256": sha256_file(temporary / "contacts" / condition / "CONTACT_CACHE_MANIFEST.json"),
            }
            for condition in CONDITIONS
        },
        "counts": {
            "cases": len(cases),
            "unique_sequences": len(sequences),
            "zero_positive_contact_cases": sum(row["positive_pair_count"] == 0 for row in common_axis_records),
        },
        "common_axes": common_axis_records,
        "audit": {
            "contact_labels_derived_only_from_bound_coordinates": True,
            "same_contact_labels_used_for_all_geometry_conditions": True,
            "old_exposed_test_access": False,
            "temporal_holdout_access": False,
            "selection_performed": False,
        },
    }
    atomic_write(temporary / "BOUND_UNBOUND_MEASUREMENT_DATA_MANIFEST_V1.json", canonical_bytes(data_manifest))
    os.replace(temporary, args.output_root)
    result = args.output_root / "BOUND_UNBOUND_MEASUREMENT_DATA_MANIFEST_V1.json"
    receipt = {
        "format_version": "framecontact_bound_unbound_measurement_data_receipt_v1",
        "status": data_manifest["status"],
        "manifest_sha256": sha256_file(result),
        "protocol_sha256": args.protocol_sha256,
        "case_count": len(cases),
        "old_exposed_test_access": False,
        "temporal_holdout_access": False,
        "selection_performed": False,
    }
    atomic_write(args.output_root / "BOUND_UNBOUND_MEASUREMENT_DATA_RECEIPT_V1.json", canonical_bytes(receipt))
    print(json.dumps({**receipt, "receipt_sha256": sha256_file(args.output_root / "BOUND_UNBOUND_MEASUREMENT_DATA_RECEIPT_V1.json")}, sort_keys=True))


if __name__ == "__main__":
    main()
