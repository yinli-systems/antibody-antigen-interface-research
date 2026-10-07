#!/usr/bin/env python3
"""Materialize SABLE labels once after the frozen bound-label authorization."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np


EXPECTED_EXTERNAL_PLAN_SHA256 = (
    "d97f0b94863fe9227d80c5675e944091809245b8fab22a89cda81148a8de172a"
)
EXPECTED_INPUT_PLAN_SHA256 = (
    "c82c12f08c87b42d21aee31cc555373ffbc4211415d9aff302444baefc971ad6"
)
EXPECTED_METADATA_SHA256 = (
    "d6b16885c7ab44bbe2a38b3fc02ab085adff73479c1a119e3786831f464b20b7"
)
EXPECTED_ENTRIES = 97
EXPECTED_PDB_IDS = 88
MINIMUM_ANALYZABLE_ENTRIES = 50
MINIMUM_ANALYZABLE_PDB_IDS = 40
CONTACT_CUTOFF_ANGSTROM = 4.5
AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")


class LabelError(ValueError):
    """Raised when a frozen authority or bound-label mapping differs."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def canonical_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise LabelError(f"JSON root is not an object: {path}")
    return value


def normalize(sequence: str) -> str:
    return "".join(str(sequence).split()).upper()


def verify_authorization(path: Path, receipt_path: Path) -> tuple[str, str]:
    authorization = load_json(path)
    receipt = load_json(receipt_path)
    authorization_sha256 = sha256_file(path)
    if (
        authorization.get("status") != "AUTHORIZED_ONE_TIME_BOUND_LABEL_OPENING"
        or authorization.get("external_execution_plan_sha256")
        != EXPECTED_EXTERNAL_PLAN_SHA256
        or authorization.get("maximum_label_openings") != 1
        or authorization.get("bound_label_materialization_authorized") is not True
        or authorization.get("one_time_evaluation_authorized") is not True
        or authorization.get("feedback_to_model_or_protocol") is not False
        or authorization.get("performance_threshold_applied") is not False
        or authorization.get("selection_or_tuning") is not False
        or receipt.get("authorization_sha256") != authorization_sha256
        or receipt.get("all_label_free_gates_verified") is not True
    ):
        raise LabelError("one-time bound-label authorization differs")
    return authorization_sha256, sha256_file(receipt_path)


def verify_download(root: Path, authorization_sha256: str) -> tuple[dict[str, Any], str]:
    manifest_path = root / "AUTHORIZED_BOUND_STRUCTURE_MANIFEST.json"
    receipt_path = root / "AUTHORIZED_BOUND_STRUCTURE_RECEIPT.json"
    manifest = load_json(manifest_path)
    receipt = load_json(receipt_path)
    manifest_sha256 = sha256_file(manifest_path)
    if (
        manifest.get("status") != "PASS_AUTHORIZED_BOUND_STRUCTURES_DOWNLOADED_ONCE"
        or manifest.get("authority", {}).get("authorization_sha256")
        != authorization_sha256
        or manifest.get("authority", {}).get("label_opening_count_after_download") != 1
        or manifest.get("authority", {}).get("labels_or_performance_read_during_download")
        is not False
        or manifest.get("authority", {}).get("selection_or_tuning") is not False
        or manifest.get("counts", {}).get("structures") != EXPECTED_ENTRIES
        or manifest.get("counts", {}).get("unique_pdb_ids") != EXPECTED_PDB_IDS
        or receipt.get("manifest_sha256") != manifest_sha256
        or receipt.get("all_archive_sha256_verified") is not True
        or receipt.get("all_selected_structure_sha256_recorded") is not True
        or receipt.get("opening_count") != 1
    ):
        raise LabelError("authorized bound-structure download authority differs")
    records = manifest.get("records", [])
    if len(records) != EXPECTED_ENTRIES:
        raise LabelError("authorized bound-structure record count differs")
    for record in records:
        path = root / str(record["structure_path"])
        if not path.is_file() or path.is_symlink():
            raise LabelError(f"missing selected bound structure: {path}")
        if sha256_file(path) != record["structure_sha256"]:
            raise LabelError(f"selected bound structure SHA differs: {path}")
    return manifest, manifest_sha256


def metadata_rows(metadata: Path, input_plan: Path) -> dict[str, dict[str, Any]]:
    if sha256_file(metadata) != EXPECTED_METADATA_SHA256:
        raise LabelError("SABLE metadata SHA256 differs")
    if sha256_file(input_plan) != EXPECTED_INPUT_PLAN_SHA256:
        raise LabelError("SABLE AF3 input plan SHA256 differs")
    plan = load_json(input_plan)
    selected = {str(row["entry_id"]) for row in plan.get("entry_bindings", [])}
    if len(selected) != EXPECTED_ENTRIES:
        raise LabelError("selected entry cardinality differs")
    rows: dict[str, dict[str, Any]] = {}
    with metadata.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            entry_id = str(raw["entry_id"])
            if entry_id not in selected:
                continue
            heavy = normalize(raw["H_sequence"])
            light = normalize(raw["L_sequence"])
            antigen_sequences = tuple(
                normalize(value) for value in raw["antigen_sequence"].split("|")
            )
            antigen_chain_ids = tuple(
                value.strip() for value in raw["antigen_chain_id"].split("|")
            )
            if (
                not heavy
                or set(heavy) - AMINO_ACIDS
                or (light and set(light) - AMINO_ACIDS)
                or not antigen_sequences
                or any(not value or set(value) - AMINO_ACIDS for value in antigen_sequences)
                or len(antigen_sequences) != len(antigen_chain_ids)
            ):
                raise LabelError(f"selected metadata sequence differs: {entry_id}")
            antibody_chain_ids = (str(raw["H_chain_id"]).strip(),)
            antibody_sequences = (heavy,)
            if light:
                light_id = str(raw["L_chain_id"]).strip()
                if not light_id:
                    raise LabelError(f"selected light chain lacks an ID: {entry_id}")
                antibody_chain_ids += (light_id,)
                antibody_sequences += (light,)
            if any(not value for value in antibody_chain_ids + antigen_chain_ids):
                raise LabelError(f"selected chain ID is empty: {entry_id}")
            rows[entry_id] = {
                "entry_id": entry_id,
                "pdb_id": str(raw["pdb_id"]).lower(),
                "structure_path": str(raw["structure_path"]),
                "antibody_chain_ids": antibody_chain_ids,
                "antibody_sequences": antibody_sequences,
                "antigen_chain_ids": antigen_chain_ids,
                "antigen_sequences": antigen_sequences,
            }
    if set(rows) != selected or len({row["pdb_id"] for row in rows.values()}) != EXPECTED_PDB_IDS:
        raise LabelError("selected metadata cohort differs")
    return rows


def observed_chain(model: Any, chain_id: str) -> list[dict[str, Any]]:
    from Bio.PDB.Polypeptide import is_aa
    from Bio.SeqUtils import seq1

    try:
        chain = model[chain_id]
    except KeyError as error:
        raise LabelError(f"bound structure lacks declared chain {chain_id!r}") from error
    rows: list[dict[str, Any]] = []
    for residue in chain:
        if not is_aa(residue, standard=False):
            continue
        amino_acid = seq1(
            residue.get_resname(), custom_map={"MSE": "M"}, undef_code="X"
        ).upper()
        heavy_atoms: list[np.ndarray] = []
        for atom in residue.get_atoms():
            element = str(getattr(atom, "element", "")).strip().upper()
            if element == "H" or atom.get_name().upper().startswith("H"):
                continue
            coordinate = np.asarray(atom.coord, dtype=np.float32)
            if coordinate.shape == (3,) and np.isfinite(coordinate).all():
                heavy_atoms.append(coordinate)
        rows.append(
            {
                "amino_acid": amino_acid,
                "heavy_atoms": (
                    np.stack(heavy_atoms).astype(np.float32)
                    if heavy_atoms
                    else np.empty((0, 3), dtype=np.float32)
                ),
            }
        )
    if not rows:
        raise LabelError(f"bound structure chain {chain_id!r} has no amino acids")
    return rows


def exact_order_preserving_mapping(
    observed: list[dict[str, Any]], target: str, label: str
) -> dict[int, int]:
    from Bio.Align import PairwiseAligner

    observed_sequence = "".join(row["amino_acid"] for row in observed)
    aligner = PairwiseAligner(mode="global")
    aligner.match_score = 2.0
    aligner.mismatch_score = -5.0
    aligner.open_gap_score = -11.0
    aligner.extend_gap_score = -1.0
    alignment = aligner.align(observed_sequence, target)[0]
    mapping: dict[int, int] = {}
    for (observed_start, observed_end), (target_start, target_end) in zip(
        alignment.aligned[0], alignment.aligned[1]
    ):
        if observed_end - observed_start != target_end - target_start:
            raise LabelError(f"aligned block length differs: {label}")
        for observed_index, target_index in zip(
            range(int(observed_start), int(observed_end)),
            range(int(target_start), int(target_end)),
        ):
            if observed_sequence[observed_index] != target[target_index]:
                raise LabelError(
                    f"non-identical mapped amino acid in {label}: "
                    f"{observed_sequence[observed_index]}/{target[target_index]}"
                )
            mapping[observed_index] = target_index
    if not mapping:
        raise LabelError(f"no exact order-preserving residue mapping: {label}")
    return mapping


def partner_axis(
    model: Any,
    chain_ids: tuple[str, ...],
    sequences: tuple[str, ...],
    label: str,
) -> tuple[str, np.ndarray, list[np.ndarray]]:
    sequence = "".join(sequences)
    observed_mask = np.zeros(len(sequence), dtype=np.bool_)
    heavy_atoms = [np.empty((0, 3), dtype=np.float32) for _ in sequence]
    offset = 0
    for chain_id, chain_sequence in zip(chain_ids, sequences):
        observed = observed_chain(model, chain_id)
        mapping = exact_order_preserving_mapping(
            observed, chain_sequence, f"{label} chain {chain_id}"
        )
        for observed_index, target_index in mapping.items():
            atoms = observed[observed_index]["heavy_atoms"]
            if len(atoms):
                axis_index = offset + target_index
                observed_mask[axis_index] = True
                heavy_atoms[axis_index] = atoms
        offset += len(chain_sequence)
    return sequence, observed_mask, heavy_atoms


def positive_contacts(
    antibody_atoms: list[np.ndarray], antigen_atoms: list[np.ndarray]
) -> np.ndarray:
    available_antigen = [atoms for atoms in antigen_atoms if len(atoms)]
    if not available_antigen:
        return np.empty(0, dtype=np.int64)
    antigen_flat = np.concatenate(available_antigen, axis=0)
    antigen_owner = np.concatenate(
        [
            np.full(len(atoms), index, dtype=np.int64)
            for index, atoms in enumerate(antigen_atoms)
            if len(atoms)
        ]
    )
    positives: list[int] = []
    threshold_squared = CONTACT_CUTOFF_ANGSTROM**2
    antigen_length = len(antigen_atoms)
    for antibody_index, atoms in enumerate(antibody_atoms):
        if not len(atoms):
            continue
        squared = ((atoms[:, None, :] - antigen_flat[None, :, :]) ** 2).sum(axis=2)
        atom_hits = np.flatnonzero((squared <= threshold_squared).any(axis=0))
        for antigen_index in np.unique(antigen_owner[atom_hits]):
            positives.append(antibody_index * antigen_length + int(antigen_index))
    return np.asarray(sorted(positives), dtype=np.int64)


def materialize_entry(
    row: dict[str, Any], structure_path: Path, artifact_root: Path
) -> dict[str, Any]:
    from Bio.PDB import PDBParser

    model = next(
        PDBParser(QUIET=True)
        .get_structure(str(row["entry_id"]), str(structure_path))
        .get_models()
    )
    antibody_sequence, antibody_mask, antibody_atoms = partner_axis(
        model,
        row["antibody_chain_ids"],
        row["antibody_sequences"],
        f"{row['entry_id']} antibody",
    )
    antigen_sequence, antigen_mask, antigen_atoms = partner_axis(
        model,
        row["antigen_chain_ids"],
        row["antigen_sequences"],
        f"{row['entry_id']} antigen",
    )
    if not bool(antibody_mask.any()) or not bool(antigen_mask.any()):
        raise LabelError(f"one partner is unmappable: {row['entry_id']}")
    positives = positive_contacts(antibody_atoms, antigen_atoms)
    identifier_sha256 = hashlib.sha256(row["entry_id"].encode("utf-8")).hexdigest()
    relative = Path("artifacts") / identifier_sha256[:2] / f"{identifier_sha256}.npz"
    target = artifact_root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        format_version=np.asarray("sable_v0_2_0_bound_contact_labels_v1"),
        entry_id=np.asarray(row["entry_id"]),
        antibody_length=np.asarray(len(antibody_sequence), dtype=np.int64),
        antigen_length=np.asarray(len(antigen_sequence), dtype=np.int64),
        antibody_observed_label_mask=antibody_mask,
        antigen_observed_label_mask=antigen_mask,
        positive_flat_indices=positives,
    )
    candidate_count = int(antibody_mask.sum()) * int(antigen_mask.sum())
    if candidate_count <= 0 or bool(
        np.any(~np.outer(antibody_mask, antigen_mask).reshape(-1)[positives])
    ):
        raise LabelError(f"contact labels violate the observed mask: {row['entry_id']}")
    return {
        "entry_id": row["entry_id"],
        "pdb_id": row["pdb_id"],
        "artifact_path": relative.as_posix(),
        "artifact_sha256": sha256_file(target),
        "antibody_sequence_sha256": sequence_sha256(antibody_sequence),
        "antigen_sequence_sha256": sequence_sha256(antigen_sequence),
        "antibody_length": len(antibody_sequence),
        "antigen_length": len(antigen_sequence),
        "observed_antibody_residues": int(antibody_mask.sum()),
        "observed_antigen_residues": int(antigen_mask.sum()),
        "observed_candidate_pairs": candidate_count,
        "positive_pairs": int(len(positives)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--authorization-receipt", type=Path, required=True)
    parser.add_argument("--external-plan", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--input-plan", type=Path, required=True)
    parser.add_argument("--download-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists() or args.output_root.is_symlink():
        raise FileExistsError(args.output_root)
    if sha256_file(args.external_plan) != EXPECTED_EXTERNAL_PLAN_SHA256:
        raise LabelError("external execution plan SHA256 differs")
    plan = load_json(args.external_plan)
    if (
        plan.get("status") != "FROZEN_POST_GATE_BEFORE_BOUND_LABEL_OPENING"
        or plan.get("opening", {}).get("maximum_label_openings") != 1
        or plan.get("cohort_authority", {}).get("minimum_analyzable_entries_after_mapping")
        != MINIMUM_ANALYZABLE_ENTRIES
        or plan.get("cohort_authority", {}).get(
            "minimum_analyzable_unique_pdb_ids_after_mapping"
        )
        != MINIMUM_ANALYZABLE_PDB_IDS
        or plan.get("selection_or_tuning") is not False
    ):
        raise LabelError("external execution plan differs")
    authorization_sha256, authorization_receipt_sha256 = verify_authorization(
        args.authorization, args.authorization_receipt
    )
    download, download_manifest_sha256 = verify_download(
        args.download_root, authorization_sha256
    )
    rows = metadata_rows(args.metadata, args.input_plan)
    download_by_entry = {row["entry_id"]: row for row in download["records"]}
    if set(download_by_entry) != set(rows):
        raise LabelError("downloaded and selected entry IDs differ")

    args.output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=f".{args.output_root.name}.tmp-", dir=args.output_root.parent)
    )
    records: list[dict[str, Any]] = []
    exclusions: list[dict[str, str]] = []
    try:
        for index, entry_id in enumerate(sorted(rows), start=1):
            row = rows[entry_id]
            structure_path = args.download_root / download_by_entry[entry_id]["structure_path"]
            try:
                record = materialize_entry(row, structure_path, temporary_root)
            except LabelError as error:
                if "non-identical mapped amino acid" in str(error):
                    raise
                exclusions.append({"entry_id": entry_id, "reason": str(error)})
            else:
                records.append(record)
            print(
                json.dumps(
                    {
                        "phase": "single_bound_label_opening",
                        "done": index,
                        "total": len(rows),
                        "entry_id": entry_id,
                        "analyzable": len(records),
                        "excluded": len(exclusions),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        analyzable_pdb_ids = {row["pdb_id"] for row in records}
        passed = (
            len(records) >= MINIMUM_ANALYZABLE_ENTRIES
            and len(analyzable_pdb_ids) >= MINIMUM_ANALYZABLE_PDB_IDS
        )
        status = (
            "PASS_SINGLE_BOUND_LABEL_OPENING_AND_MAPPING_GATE"
            if passed
            else "FAIL_CLOSED_SINGLE_BOUND_LABEL_MAPPING_GATE"
        )
        payload = {
            "format_version": "sable_v0_2_0_bound_label_manifest_v1",
            "status": status,
            "authority": {
                "authorization_sha256": authorization_sha256,
                "authorization_receipt_sha256": authorization_receipt_sha256,
                "external_execution_plan_sha256": EXPECTED_EXTERNAL_PLAN_SHA256,
                "input_plan_sha256": EXPECTED_INPUT_PLAN_SHA256,
                "metadata_sha256": EXPECTED_METADATA_SHA256,
                "download_manifest_sha256": download_manifest_sha256,
                "opening_count": 1,
                "contact_definition": (
                    "minimum cross-partner non-hydrogen atom distance <= 4.5 angstrom"
                ),
                "sequence_mapping": (
                    "deterministic chain-specific global order-preserving exact-amino-acid mapping"
                ),
                "unobserved_or_unmappable_residues_assigned_negative": False,
                "experimental_bound_coordinates_used_as_model_input": False,
                "experimental_bound_coordinates_used_for_labels_only": True,
                "selection_or_tuning": False,
                "feedback_to_model_or_protocol": False,
                "old_sabdab2_test_loaded": False,
                "temporal_holdout_loaded": False,
            },
            "mapping_gate": {
                "minimum_analyzable_entries": MINIMUM_ANALYZABLE_ENTRIES,
                "minimum_analyzable_unique_pdb_ids": MINIMUM_ANALYZABLE_PDB_IDS,
                "observed_analyzable_entries": len(records),
                "observed_analyzable_unique_pdb_ids": len(analyzable_pdb_ids),
                "passed": passed,
                "performance_read_before_gate": False,
            },
            "counts": {
                "selected_entries": len(rows),
                "analyzable_entries": len(records),
                "excluded_entries": len(exclusions),
                "analyzable_unique_pdb_ids": len(analyzable_pdb_ids),
                "observed_candidate_pairs": sum(
                    int(row["observed_candidate_pairs"]) for row in records
                ),
                "positive_pairs": sum(int(row["positive_pairs"]) for row in records),
            },
            "records": records,
            "exclusions": exclusions,
        }
        manifest_path = temporary_root / "SABLE_BOUND_LABEL_MANIFEST.json"
        manifest_path.write_bytes(canonical_bytes(payload))
        receipt = {
            "format_version": "sable_v0_2_0_bound_label_receipt_v1",
            "status": status,
            "manifest_sha256": sha256_file(manifest_path),
            "authorization_sha256": authorization_sha256,
            "opening_count": 1,
            "mapping_gate_passed": passed,
            "analyzable_entry_count": len(records),
            "analyzable_unique_pdb_id_count": len(analyzable_pdb_ids),
            "performance_read": False,
            "selection_or_tuning": False,
        }
        receipt_path = temporary_root / "SABLE_BOUND_LABEL_RECEIPT.json"
        receipt_path.write_bytes(canonical_bytes(receipt))
        os.replace(temporary_root, args.output_root)
    except BaseException:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise
    print(
        json.dumps(
            {**receipt, "receipt_sha256": sha256_file(args.output_root / receipt_path.name)},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
