#!/usr/bin/env python3
"""Materialize train-only PINDER residue-pair labels on resolved PDB axes.

The bound coordinates are used only to construct supervision.  Model inputs are
the two resolved amino-acid strings; no coordinate, distance, or structure
feature is published in the training records.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import csv
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import statistics
import subprocess
import tempfile
from typing import Iterable, Sequence
import zipfile

import numpy as np
from scipy.spatial import cKDTree


AA3 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEC": "U", "PYL": "O", "ASX": "B", "GLX": "Z",
}
PRIMARY_CONTACT_RADIUS = 4.5
AUXILIARY_CONTACT_RADIUS = 6.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sql_path(path: Path) -> str:
    return str(path).replace("'", "''")


def _safe_member(pdb_name: str) -> str:
    path = PurePosixPath(pdb_name)
    if path.name != pdb_name or path.suffix != ".pdb":
        raise ValueError(f"Invalid PINDER holo filename: {pdb_name!r}")
    return f"pdbs/{pdb_name}"


def parse_holo_endpoint(payload: bytes) -> tuple[str, np.ndarray, np.ndarray]:
    """Return the resolved sequence and selected heavy-atom coordinates."""

    residues: OrderedDict[tuple[str, int, str], dict[str, object]] = OrderedDict()
    saw_model = False
    selected_model = False
    for line in payload.decode("ascii").splitlines():
        if line.startswith("MODEL "):
            if not saw_model:
                saw_model = True
                selected_model = True
            else:
                selected_model = False
            continue
        if line.startswith("ENDMDL"):
            if selected_model:
                break
            continue
        if saw_model and not selected_model:
            continue
        if not line.startswith("ATOM  ") or len(line) < 60:
            continue
        chain = line[21].strip()
        if not chain:
            raise ValueError("PINDER holo endpoint has a blank atom-chain id")
        atom_name = line[12:16].strip().upper()
        element = line[76:78].strip().upper() if len(line) >= 78 else ""
        if element in {"H", "D"} or (not element and atom_name.startswith(("H", "D"))):
            continue
        try:
            residue_number = int(line[22:26])
            coordinate = (
                float(line[30:38]),
                float(line[38:46]),
                float(line[46:54]),
            )
            occupancy = float(line[54:60])
        except ValueError as error:
            raise ValueError("Malformed PINDER holo coordinate") from error
        amino_acid = AA3.get(line[17:20].strip().upper())
        if amino_acid is None:
            raise ValueError(f"Unsupported PINDER residue: {line[17:20].strip()!r}")
        key = (chain, residue_number, line[26].strip())
        record = residues.setdefault(
            key,
            {
                "shared_aa": None,
                "shared_atoms": {},
                "ignored_shared_residue_variants": set(),
                "variants": {},
            },
        )
        altloc = line[16]
        rank = (occupancy, altloc == " ", altloc == "A", altloc)
        if altloc == " ":
            if record["shared_aa"] not in (None, amino_acid):
                # Some PINDER endpoint files flatten mutually exclusive residue
                # identities into consecutive blank-altloc ATOM blocks. Keep the
                # first complete identity deterministically; never mix its atoms
                # with the later residue variant.
                ignored = record["ignored_shared_residue_variants"]
                assert isinstance(ignored, set)
                ignored.add(amino_acid)
                continue
            record["shared_aa"] = amino_acid
            atoms = record["shared_atoms"]
        else:
            variants = record["variants"]
            assert isinstance(variants, dict)
            variant = variants.setdefault(
                altloc,
                {"aa": amino_acid, "atoms": {}, "occupancy": occupancy},
            )
            if variant["aa"] != amino_acid:
                raise ValueError(f"Conflicting alternate residue identity: {key}")
            variant["occupancy"] = max(float(variant["occupancy"]), occupancy)
            atoms = variant["atoms"]
        assert isinstance(atoms, dict)
        previous = atoms.get(atom_name)
        if previous is None or rank > previous[0]:
            atoms[atom_name] = (rank, coordinate)

    if not residues:
        raise ValueError("PINDER holo endpoint contains no selected heavy atoms")
    chains = {key[0] for key in residues}
    if len(chains) != 1:
        raise ValueError(f"PINDER holo endpoint is not a single chain: {sorted(chains)}")
    sequence: list[str] = []
    coordinates: list[tuple[float, float, float]] = []
    atom_residue_indices: list[int] = []
    for residue_index, (key, record) in enumerate(residues.items()):
        variants = record["variants"]
        shared_atoms = record["shared_atoms"]
        assert isinstance(variants, dict) and isinstance(shared_atoms, dict)
        if variants:
            altloc, selected = max(
                variants.items(),
                key=lambda item: (
                    float(item[1]["occupancy"]),
                    item[0] == "A",
                    item[0],
                ),
            )
            amino_acid = str(selected["aa"])
            atoms = dict(shared_atoms)
            atoms.update(selected["atoms"])
        else:
            altloc = " "
            amino_acid = str(record["shared_aa"] or "")
            atoms = shared_atoms
        if not amino_acid or not atoms:
            raise ValueError(f"Incomplete PINDER residue after altloc selection: {key}:{altloc}")
        sequence.append(amino_acid)
        for _, coordinate in atoms.values():
            coordinates.append(coordinate)
            atom_residue_indices.append(residue_index)
    return (
        "".join(sequence),
        np.asarray(coordinates, dtype=np.float32),
        np.asarray(atom_residue_indices, dtype=np.int32),
    )


def residue_contacts(
    left_coordinates: np.ndarray,
    left_residue_indices: np.ndarray,
    right_coordinates: np.ndarray,
    right_residue_indices: np.ndarray,
    radius: float,
) -> list[tuple[int, int]]:
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("Contact radius must be finite and positive")
    neighbours = cKDTree(right_coordinates).query_ball_point(left_coordinates, r=radius)
    pairs = {
        (int(left_residue_indices[left_atom]), int(right_residue_indices[right_atom]))
        for left_atom, right_atoms in enumerate(neighbours)
        for right_atom in right_atoms
    }
    return sorted(pairs)


def _allocate_crop_budgets(left_length: int, right_length: int, maximum: int) -> tuple[int, int]:
    if min(left_length, right_length) < 1 or maximum < 16:
        raise ValueError("Contact crop dimensions are invalid")
    if left_length + right_length <= maximum:
        return left_length, right_length
    left_minimum = min(left_length, 64)
    right_minimum = min(right_length, 64)
    if left_minimum + right_minimum > maximum:
        raise ValueError("Contact crop cannot retain both endpoints")
    left_budget = round(maximum * left_length / (left_length + right_length))
    left_budget = max(left_minimum, min(left_length, left_budget))
    left_budget = min(left_budget, maximum - right_minimum)
    right_budget = min(right_length, maximum - left_budget)
    left_budget = min(left_length, maximum - right_budget)
    if left_budget + right_budget != maximum:
        raise ValueError("Contact crop did not consume its residue budget")
    return left_budget, right_budget


def _best_window(length: int, budget: int, indices: Iterable[int]) -> int:
    if budget >= length:
        return 0
    counts = [0] * (length + 1)
    observed = sorted(int(index) for index in indices)
    if not observed or observed[0] < 0 or observed[-1] >= length:
        raise ValueError("Contact crop indices are absent or out of bounds")
    for index in observed:
        counts[index + 1] += 1
    for index in range(length):
        counts[index + 1] += counts[index]
    centre = statistics.median(observed)
    candidates = []
    for start in range(length - budget + 1):
        retained = counts[start + budget] - counts[start]
        centre_distance = abs((start + (budget - 1) / 2) - centre)
        candidates.append((-retained, centre_distance, start))
    return min(candidates)[2]


def crop_contact_patch(
    left_sequence: str,
    right_sequence: str,
    primary_pairs: Sequence[tuple[int, int]],
    auxiliary_pairs: Sequence[tuple[int, int]],
    *,
    maximum_residues: int,
) -> tuple[str, str, list[tuple[int, int]], list[tuple[int, int]], dict[str, int]]:
    if not primary_pairs or not set(primary_pairs).issubset(auxiliary_pairs):
        raise ValueError("Primary contacts must be a non-empty subset of auxiliary contacts")
    left_budget, right_budget = _allocate_crop_budgets(
        len(left_sequence), len(right_sequence), maximum_residues
    )
    left_start = _best_window(
        len(left_sequence), left_budget, (left for left, _ in primary_pairs)
    )
    right_start = _best_window(
        len(right_sequence), right_budget, (right for _, right in primary_pairs)
    )

    def remap(pairs: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
        return sorted(
            (left - left_start, right - right_start)
            for left, right in pairs
            if left_start <= left < left_start + left_budget
            and right_start <= right < right_start + right_budget
        )

    cropped_primary = remap(primary_pairs)
    cropped_auxiliary = remap(auxiliary_pairs)
    if not cropped_primary:
        anchor_left, anchor_right = primary_pairs[0]
        left_start = min(max(0, anchor_left - left_budget // 2), len(left_sequence) - left_budget)
        right_start = min(max(0, anchor_right - right_budget // 2), len(right_sequence) - right_budget)
        cropped_primary = remap(primary_pairs)
        cropped_auxiliary = remap(auxiliary_pairs)
    if not cropped_primary or not set(cropped_primary).issubset(cropped_auxiliary):
        raise ValueError("Interface-preserving crop lost every primary contact")
    return (
        left_sequence[left_start : left_start + left_budget],
        right_sequence[right_start : right_start + right_budget],
        cropped_primary,
        cropped_auxiliary,
        {
            "left_start": left_start,
            "right_start": right_start,
            "left_resolved_length": len(left_sequence),
            "right_resolved_length": len(right_sequence),
        },
    )


def endpoint_cluster_query(curriculum: Path, index: Path, output: Path) -> str:
    return f"""
    COPY (
      SELECT c.id, i.cluster_id_R, i.cluster_id_L, i.split
      FROM read_csv('{_sql_path(curriculum)}', delim='\\t', header=true) c
      JOIN read_parquet('{_sql_path(index)}') i USING (id)
      ORDER BY c.id
    ) TO '{_sql_path(output)}' (FORMAT CSV, HEADER TRUE, DELIMITER '\\t');
    """


def _load_endpoint_clusters(
    duckdb: Path, curriculum: Path, index: Path, temporary_root: Path
) -> dict[str, tuple[str, str]]:
    output = temporary_root / "endpoint_clusters.tsv"
    result = subprocess.run(
        [str(duckdb), "-batch", "-c", endpoint_cluster_query(curriculum, index, output)],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(f"PINDER endpoint-cluster join failed: {result.stderr[-2000:]}")
    clusters: dict[str, tuple[str, str]] = {}
    with output.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != ("id", "cluster_id_R", "cluster_id_L", "split"):
            raise RuntimeError("PINDER endpoint-cluster join schema differs")
        for row in reader:
            if row["split"] != "train" or not row["cluster_id_R"] or not row["cluster_id_L"]:
                raise RuntimeError("PINDER endpoint-cluster authority differs")
            if row["id"] in clusters:
                raise RuntimeError("PINDER endpoint-cluster join is not one-to-one")
            clusters[row["id"]] = (row["cluster_id_R"], row["cluster_id_L"])
    return clusters


def _publish_create_only(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_path, path)
        path.chmod(0o444)
    except FileExistsError as error:
        raise FileExistsError(f"Refusing to overwrite output: {path}") from error
    finally:
        temporary_path.unlink(missing_ok=True)


def materialize(
    curriculum: Path,
    pdb_archive: Path,
    endpoint_clusters: dict[str, tuple[str, str]],
    maximum_residues: int,
) -> tuple[bytes, dict[str, object]]:
    records: list[dict[str, object]] = []
    primary_contact_count = 0
    auxiliary_contact_count = 0
    cropped_systems = 0
    resolved_residue_count = 0
    seen_ids: set[str] = set()
    with gzip.open(curriculum, "rt", encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source, delimiter="\t"))
    if not rows:
        raise RuntimeError("PINDER curriculum is empty")
    with zipfile.ZipFile(pdb_archive) as archive:
        inventory = {item.filename: item for item in archive.infolist()}
        for row in rows:
            identifier = row["id"]
            if row["split"] != "train" or identifier in seen_ids:
                raise RuntimeError("PINDER curriculum is not unique train-only data")
            seen_ids.add(identifier)
            if identifier not in endpoint_clusters:
                raise RuntimeError(f"PINDER endpoint clusters are absent: {identifier}")
            payloads = []
            for column in ("holo_R_pdb", "holo_L_pdb"):
                member = _safe_member(row[column])
                if member not in inventory:
                    raise RuntimeError(f"PINDER structure member is absent: {member}")
                payloads.append(archive.read(member))
            try:
                left_sequence, left_coordinates, left_atom_residues = parse_holo_endpoint(payloads[0])
                right_sequence, right_coordinates, right_atom_residues = parse_holo_endpoint(payloads[1])
            except ValueError as error:
                raise ValueError(f"Invalid PINDER endpoint for {identifier}") from error
            primary_pairs = residue_contacts(
                left_coordinates,
                left_atom_residues,
                right_coordinates,
                right_atom_residues,
                PRIMARY_CONTACT_RADIUS,
            )
            auxiliary_pairs = residue_contacts(
                left_coordinates,
                left_atom_residues,
                right_coordinates,
                right_atom_residues,
                AUXILIARY_CONTACT_RADIUS,
            )
            if not primary_pairs:
                raise RuntimeError(f"PINDER system has no 4.5-A contact: {identifier}")
            original_total = len(left_sequence) + len(right_sequence)
            (
                left_patch,
                right_patch,
                primary_pairs,
                auxiliary_pairs,
                crop,
            ) = crop_contact_patch(
                left_sequence,
                right_sequence,
                primary_pairs,
                auxiliary_pairs,
                maximum_residues=maximum_residues,
            )
            cropped_systems += int(original_total > maximum_residues)
            primary_contact_count += len(primary_pairs)
            auxiliary_contact_count += len(auxiliary_pairs)
            resolved_residue_count += len(left_patch) + len(right_patch)
            left_cluster, right_cluster = endpoint_clusters[identifier]
            records.append(
                {
                    "auxiliary_contact_pairs_6A": auxiliary_pairs,
                    "cluster_id": row["cluster_id"],
                    "contact_pairs": primary_pairs,
                    "contact_radius_angstrom": PRIMARY_CONTACT_RADIUS,
                    "contact_supervision": True,
                    "crop": crop,
                    "endpoint_cluster1": left_cluster,
                    "endpoint_cluster2": right_cluster,
                    "endpoint_exact1_sha256": row["sequence_sha256_R"],
                    "endpoint_exact2_sha256": row["sequence_sha256_L"],
                    "example_id": identifier,
                    "full_pair_hash_sha256": row["exact_sequence_pair_sha256"],
                    "known_positive_group": row["pdb_id"],
                    "pdb_id": row["pdb_id"],
                    "retrieval_supervision": True,
                    "sequence1": left_patch,
                    "sequence2": right_patch,
                    "source": "pinder_train_bound_labels",
                    "system_weight": float(row["system_weight"]),
                }
            )
    raw = "".join(
        json.dumps(record, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n"
        for record in records
    ).encode("ascii")
    compressed = io.BytesIO()
    with gzip.GzipFile(fileobj=compressed, mode="wb", compresslevel=9, mtime=0) as handle:
        handle.write(raw)
    counts: dict[str, object] = {
        "auxiliary_6A_contact_pairs": auxiliary_contact_count,
        "cropped_systems": cropped_systems,
        "primary_4p5A_contact_pairs": primary_contact_count,
        "published_residues": resolved_residue_count,
        "systems": len(records),
        "unique_endpoint_cluster_R": len({record["endpoint_cluster1"] for record in records}),
        "unique_endpoint_cluster_L": len({record["endpoint_cluster2"] for record in records}),
        "unique_pdbs": len({record["pdb_id"] for record in records}),
    }
    return compressed.getvalue(), counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--curriculum", type=Path, required=True)
    parser.add_argument("--expected-curriculum-sha256", required=True)
    parser.add_argument("--curriculum-receipt", type=Path, required=True)
    parser.add_argument("--expected-curriculum-receipt-sha256", required=True)
    parser.add_argument("--pdb-archive", type=Path, required=True)
    parser.add_argument("--pdb-materialization-receipt", type=Path, required=True)
    parser.add_argument("--expected-pdb-materialization-receipt-sha256", required=True)
    parser.add_argument("--index-parquet", type=Path, required=True)
    parser.add_argument("--expected-index-sha256", required=True)
    parser.add_argument("--duckdb", type=Path, required=True)
    parser.add_argument("--expected-duckdb-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--maximum-residues", type=int, default=1020)
    arguments = parser.parse_args()

    authenticated = (
        (arguments.curriculum, arguments.expected_curriculum_sha256),
        (arguments.curriculum_receipt, arguments.expected_curriculum_receipt_sha256),
        (
            arguments.pdb_materialization_receipt,
            arguments.expected_pdb_materialization_receipt_sha256,
        ),
        (arguments.index_parquet, arguments.expected_index_sha256),
        (arguments.duckdb, arguments.expected_duckdb_sha256),
    )
    for path, expected in authenticated:
        if path.is_symlink() or not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"Authenticated input differs: {path}")
    if arguments.pdb_archive.is_symlink() or not arguments.pdb_archive.is_file():
        raise RuntimeError("PINDER structure archive is absent")
    materialization = json.loads(
        arguments.pdb_materialization_receipt.read_text(encoding="utf-8")
    )
    if (
        materialization.get("status") != "MATERIALIZED"
        or materialization.get("archive", {}).get("sha256")
        != "125bc49d515803576c19123afdc72f4f59e84d64522866f99d089c53b7d27912"
        or materialization.get("archive", {}).get("size_bytes")
        != arguments.pdb_archive.stat().st_size
        or materialization.get("test_access", {}).get("pinder_test_rows_used") is not False
        or materialization.get("test_access", {}).get("pinder_test_structure_bytes_accessed") is not False
    ):
        raise RuntimeError("PINDER structure materialization authority differs")
    curriculum_receipt = json.loads(
        arguments.curriculum_receipt.read_text(encoding="utf-8")
    )
    if (
        curriculum_receipt.get("status")
        != "PASS_TRAIN_ONLY_NESTED_CLUSTER_BALANCED_CURRICULUM"
        or curriculum_receipt.get("checks", {}).get("all_rows_train") is not True
        or curriculum_receipt.get("checks", {}).get("test_rows_read") is not False
        or curriculum_receipt.get("stages", {}).get("B-canary", {}).get("sha256")
        != arguments.expected_curriculum_sha256
    ):
        raise RuntimeError("PINDER curriculum authority differs")
    if arguments.maximum_residues != 1020:
        raise ValueError("PINDER contact label token budget differs from protocol")
    for output in (arguments.output, arguments.receipt):
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"Refusing to overwrite output: {output}")

    with tempfile.TemporaryDirectory(prefix="pinder-contact-labels-", dir="/dev/shm") as temporary:
        endpoint_clusters = _load_endpoint_clusters(
            arguments.duckdb,
            arguments.curriculum,
            arguments.index_parquet,
            Path(temporary),
        )
        payload, counts = materialize(
            arguments.curriculum,
            arguments.pdb_archive,
            endpoint_clusters,
            arguments.maximum_residues,
        )
    _publish_create_only(arguments.output, payload)
    receipt = {
        "authorization": {"B_canary_training": True, "B32K_training": False},
        "checks": {
            "bound_coordinates_published_as_model_input": False,
            "contact_indices_match_published_resolved_sequence_axes": True,
            "endpoint_clusters_joined_from_train_index": True,
            "pinder_test_rows_read": False,
            "pinder_test_structures_read": False,
        },
        "claim_boundary": (
            "Create-only B-canary supervision from authenticated PINDER-Train bound "
            "coordinates. Coordinates are labels only and are not model inputs."
        ),
        "contact_definition": {
            "auxiliary": "minimum inter-chain non-hydrogen atom distance <= 6.0 Angstrom",
            "primary": "minimum inter-chain non-hydrogen atom distance <= 4.5 Angstrom",
        },
        "counts": counts,
        "format_version": "abpairfm_pinder_contact_labels_v1",
        "inputs": {
            "curriculum_receipt_sha256": arguments.expected_curriculum_receipt_sha256,
            "curriculum_sha256": arguments.expected_curriculum_sha256,
            "duckdb_sha256": arguments.expected_duckdb_sha256,
            "index_sha256": arguments.expected_index_sha256,
            "pdb_archive_sha256": materialization["archive"]["sha256"],
            "pdb_materialization_receipt_sha256": (
                arguments.expected_pdb_materialization_receipt_sha256
            ),
        },
        "output": {
            "bytes": arguments.output.stat().st_size,
            "path": arguments.output.name,
            "sha256": sha256_file(arguments.output),
        },
        "status": "PASS_B_CANARY_PAIR_LABELS_MODEL_INPUT_SEQUENCE_ONLY",
        "test_access": False,
    }
    serialized = (
        json.dumps(receipt, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode("ascii")
    _publish_create_only(arguments.receipt, serialized)
    print(serialized.decode("ascii"), end="")


if __name__ == "__main__":
    main()
