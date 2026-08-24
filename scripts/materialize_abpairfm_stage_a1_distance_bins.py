#!/usr/bin/env python3
"""Add train-only heavy-atom distance-bin targets to the Stage-A1 manifest."""

from __future__ import annotations

import argparse
import base64
from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Iterable
import zipfile

import numpy as np
from scipy.spatial.distance import cdist


AA3 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEC": "U", "PYL": "O", "ASX": "B", "GLX": "Z",
}
DEFAULT_DISTANCE_BIN_EDGES = (2.0, 4.5, 6.0, 8.0, 12.0, 16.0, 24.0)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def zstd_lines(path: Path) -> Iterable[str]:
    process = subprocess.Popen(
        ["zstd", "-q", "-dc", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    assert process.stdout is not None
    try:
        yield from process.stdout
    finally:
        process.stdout.close()
        error = process.stderr.read() if process.stderr is not None else ""
        return_code = process.wait()
        if return_code:
            raise RuntimeError(f"zstd failed with {return_code}: {error[-1000:]}")


def _publish_zstd_create_only(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as raw:
            process = subprocess.Popen(
                ["zstd", "-q", "-8", "-c"],
                stdin=subprocess.PIPE,
                stdout=raw,
                stderr=subprocess.PIPE,
            )
            assert process.stdin is not None
            for row in rows:
                process.stdin.write(
                    (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode(
                        "utf-8"
                    )
                )
            process.stdin.close()
            error = process.stderr.read() if process.stderr is not None else b""
            return_code = process.wait()
            if return_code:
                raise RuntimeError(
                    f"zstd compression failed with {return_code}: "
                    f"{error[-1000:].decode('utf-8', errors='replace')}"
                )
            raw.flush()
            os.fsync(raw.fileno())
        os.link(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _publish_json_create_only(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(rendered)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-manifest", type=Path, required=True)
    parser.add_argument("--expected-pilot-manifest-sha256", required=True)
    parser.add_argument("--ppiref-manifest", type=Path, required=True)
    parser.add_argument("--expected-ppiref-manifest-sha256", required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--expected-archive-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--minimum-combined-tokens", type=int, default=65)
    parser.add_argument("--maximum-combined-tokens", type=int, default=128)
    return parser.parse_args()


def _heavy_atom_residues(
    raw_pdb: bytes, chain1: str, chain2: str
) -> tuple[str, list[np.ndarray], str, list[np.ndarray]]:
    residues: dict[str, OrderedDict[tuple[int, str], dict[str, Any]]] = {}
    active_model = True
    saw_model = False
    for raw_line in raw_pdb.decode("ascii", errors="replace").splitlines():
        if raw_line.startswith("MODEL"):
            if saw_model:
                active_model = False
            else:
                saw_model = True
            continue
        if raw_line.startswith("ENDMDL") and active_model:
            active_model = False
            continue
        if not active_model or not raw_line.startswith("ATOM") or len(raw_line) < 54:
            continue
        line = raw_line.ljust(80)
        chain = line[21].strip()
        if not chain:
            continue
        atom_name = line[12:16].strip()
        element = line[76:78].strip().upper() or atom_name[:1].upper()
        if element == "H" or atom_name.upper().startswith("H"):
            continue
        try:
            residue_number = int(line[22:26])
            coordinate = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            occupancy = float(line[54:60] or "0")
        except ValueError as error:
            raise ValueError("Malformed PPIRef heavy-atom record") from error
        key = (residue_number, line[26].strip())
        amino_acid = AA3.get(line[17:20].strip().upper(), "X")
        altloc = line[16]
        rank = (occupancy, altloc == " ", altloc == "A", altloc)
        record = residues.setdefault(chain, OrderedDict()).setdefault(
            key,
            {"shared_aa": None, "shared_atoms": {}, "variants": {}},
        )
        if altloc == " ":
            record["shared_aa"] = amino_acid
            atoms = record["shared_atoms"]
        else:
            variant = record["variants"].setdefault(
                altloc, {"aa": amino_acid, "atoms": {}, "occupancy": occupancy}
            )
            variant["occupancy"] = max(float(variant["occupancy"]), occupancy)
            atoms = variant["atoms"]
        previous = atoms.get(atom_name)
        if previous is None or rank > previous[0]:
            atoms[atom_name] = (rank, coordinate)

    def resolve(requested: str) -> str:
        if requested in residues:
            return requested
        matches = [chain for chain in residues if chain.casefold() == requested.casefold()]
        if len(matches) != 1:
            raise ValueError(f"PPIRef chain cannot be resolved: {requested!r}")
        return matches[0]

    def select(chain: str) -> tuple[str, list[np.ndarray]]:
        sequence: list[str] = []
        coordinates: list[np.ndarray] = []
        for record in residues[chain].values():
            variants = record["variants"]
            shared_atoms = record["shared_atoms"]
            if variants:
                _, selected = max(
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
                amino_acid = str(record["shared_aa"] or "X")
                atoms = shared_atoms
            if not atoms:
                raise ValueError("PPIRef residue has no selected heavy atom")
            sequence.append(amino_acid)
            coordinates.append(
                np.asarray([value[1] for value in atoms.values()], dtype=np.float32)
            )
        return "".join(sequence), coordinates

    source_chain1 = resolve(chain1)
    source_chain2 = resolve(chain2)
    if source_chain1 == source_chain2:
        raise ValueError("PPIRef partners resolve to the same atom chain")
    sequence1, coordinates1 = select(source_chain1)
    sequence2, coordinates2 = select(source_chain2)
    return sequence1, coordinates1, sequence2, coordinates2


def minimum_heavy_atom_distance_matrix(
    raw_pdb: bytes, chain1: str, chain2: str
) -> tuple[str, str, np.ndarray]:
    sequence1, residues1, sequence2, residues2 = _heavy_atom_residues(
        raw_pdb, chain1, chain2
    )
    coordinates1 = np.concatenate(residues1, axis=0)
    coordinates2 = np.concatenate(residues2, axis=0)
    indices1 = np.repeat(np.arange(len(residues1)), [len(value) for value in residues1])
    indices2 = np.repeat(np.arange(len(residues2)), [len(value) for value in residues2])
    atom_distances = cdist(coordinates1, coordinates2)
    flat_indices = indices1[:, None] * len(residues2) + indices2[None, :]
    matrix = np.full(len(residues1) * len(residues2), np.inf, dtype=np.float64)
    np.minimum.at(matrix, flat_indices.ravel(), atom_distances.ravel())
    matrix = matrix.reshape(len(residues1), len(residues2))
    if not bool(np.isfinite(matrix).all()):
        raise RuntimeError("PPIRef heavy-atom distance matrix is incomplete")
    return sequence1, sequence2, matrix


def main() -> None:
    arguments = _arguments()
    if arguments.output.exists() or arguments.receipt.exists():
        raise FileExistsError("Stage-A1 distance-bin outputs are create-only")
    inputs = {
        "archive": (arguments.archive, arguments.expected_archive_sha256),
        "pilot_manifest": (
            arguments.pilot_manifest,
            arguments.expected_pilot_manifest_sha256,
        ),
        "ppiref_manifest": (
            arguments.ppiref_manifest,
            arguments.expected_ppiref_manifest_sha256,
        ),
    }
    for name, (path, expected) in inputs.items():
        if path.is_symlink() or not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"Stage-A1 distance-bin input differs: {name}")

    rows = [json.loads(line) for line in zstd_lines(arguments.pilot_manifest)]
    selected_ids = {
        str(row["example_id"])
        for row in rows
        if row.get("source") == "ppiref50k"
        and row.get("contact_supervision") is True
        and arguments.minimum_combined_tokens
        <= len(str(row["sequence1"])) + len(str(row["sequence2"])) + 4
        <= arguments.maximum_combined_tokens
    }
    ppiref = {}
    for line in zstd_lines(arguments.ppiref_manifest):
        value = json.loads(line)
        identifier = str(value["example_id"])
        if identifier in selected_ids:
            ppiref[identifier] = value
    if set(ppiref) != selected_ids:
        raise RuntimeError("Stage-A1 distance-bin source join is incomplete")

    bin_edges = np.asarray(DEFAULT_DISTANCE_BIN_EDGES, dtype=np.float64)
    bin_counts = np.zeros(len(bin_edges) + 1, dtype=np.int64)
    six_angstrom_pairs = 0
    four_point_five_angstrom_pairs = 0
    selected = 0
    with zipfile.ZipFile(arguments.archive) as archive:
        for row in rows:
            identifier = str(row["example_id"])
            source = ppiref.get(identifier)
            if source is None:
                continue
            raw_pdb = archive.read(str(source["archive_member"]))
            metadata = source.get("metadata", {})
            sequence1, sequence2, distances = minimum_heavy_atom_distance_matrix(
                raw_pdb,
                str(metadata["source_atom_chain1"]),
                str(metadata["source_atom_chain2"]),
            )
            if sequence1 != row["sequence1"] or sequence2 != row["sequence2"]:
                raise RuntimeError(f"Stage-A1 distance-bin sequence differs: {identifier}")
            observed_six = set(zip(*np.nonzero(distances <= 6.0)))
            expected_six = {tuple(pair) for pair in row["contact_pairs"]}
            if observed_six != expected_six:
                raise RuntimeError(f"Stage-A1 six-A contact replay differs: {identifier}")
            bins = np.searchsorted(bin_edges, distances, side="left").astype(np.uint8)
            counts = np.bincount(bins.ravel(), minlength=len(bin_counts))
            bin_counts += counts
            six_angstrom_pairs += int((distances <= 6.0).sum())
            four_point_five_angstrom_pairs += int((distances <= 4.5).sum())
            row["distance_bin_edges_angstrom"] = list(DEFAULT_DISTANCE_BIN_EDGES)
            row["distance_bin_shape"] = list(bins.shape)
            row["distance_bin_u8_base64"] = base64.b64encode(bins.tobytes()).decode(
                "ascii"
            )
            row["distance_definition"] = "minimum_interchain_heavy_atom_distance"
            selected += 1
            if selected % 100 == 0:
                print(f"materialized={selected}/{len(selected_ids)}", flush=True)

    if selected != len(selected_ids):
        raise RuntimeError("Stage-A1 distance-bin coverage differs")
    _publish_zstd_create_only(arguments.output, rows)
    output_sha256 = sha256_file(arguments.output)
    receipt = {
        "authorization": {
            "development_evaluation": False,
            "pilot_training": True,
            "production_training": False,
            "test_access": False,
        },
        "counts": {
            "distance_bin_records": selected,
            "distance_bins": bin_counts.tolist(),
            "four_point_five_angstrom_pairs": four_point_five_angstrom_pairs,
            "records": len(rows),
            "retrieval_records_with_endpoint_cluster_ids": sum(
                bool(row.get("retrieval_supervision"))
                and bool(row.get("endpoint_cluster1"))
                and bool(row.get("endpoint_cluster2"))
                for row in rows
            ),
            "retrieval_supervised_records": sum(
                bool(row.get("retrieval_supervision")) for row in rows
            ),
            "six_angstrom_pairs": six_angstrom_pairs,
        },
        "cross_source_exact_pair_overlap_after_filter": 0,
        "distance_bin_edges_angstrom": list(DEFAULT_DISTANCE_BIN_EDGES),
        "format_version": "abpairfm_stage_a1_distance_bin_manifest_v1",
        "inputs": {
            name: {"path": str(path.resolve()), "sha256": expected}
            for name, (path, expected) in inputs.items()
        },
        "output": {
            "bytes": arguments.output.stat().st_size,
            "path": str(arguments.output.resolve()),
            "sha256": output_sha256,
        },
        "status": "PASS_STAGE_A1_DISTANCE_BIN_TRAIN_ONLY_MATERIALIZATION",
        "test_access": False,
    }
    _publish_json_create_only(arguments.receipt, receipt)


if __name__ == "__main__":
    main()
