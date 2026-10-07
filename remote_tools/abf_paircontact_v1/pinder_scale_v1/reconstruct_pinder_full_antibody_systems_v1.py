#!/usr/bin/env python3
"""Reconstruct complete train-only antibody systems from SAbDab--PINDER authority.

This gate reads metadata and sequences only.  It deliberately does not read
coordinates, contact labels, validation/test rows, checkpoints, or scores, and
it does not authorize training or model selection.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import string
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd
import pyarrow.parquet as pq


PROTOCOL_FORMAT_VERSION = "pinder_full_antibody_system_reconstruction_protocol_v1"
RESULT_FORMAT_VERSION = "pinder_full_antibody_system_reconstruction_result_v1"
MANIFEST_NAME = "PINDER_FULL_ANTIBODY_SYSTEMS_V1.tsv.gz"
RESULT_NAME = "PINDER_FULL_ANTIBODY_SYSTEM_RECONSTRUCTION_RESULT_V1.json"
RECEIPT_NAME = "PINDER_FULL_ANTIBODY_SYSTEM_RECONSTRUCTION_RECEIPT_V1.json"
TOPOLOGIES = ("fab", "heavy_only_or_vhh", "scfv")

SABDAB_COLUMNS = (
    "pdb_id",
    "Hchain",
    "Lchain",
    "model",
    "antigen_chain",
    "scfv",
    "pinder_Hchain_ids",
    "pinder_Hchain_monomers",
    "pinder_Lchain_ids",
    "pinder_Lchain_monomers",
    "pinder_antigen_chain_ids",
    "pinder_antigen_chain_monomers",
)
INDEX_COLUMNS = (
    "split",
    "id",
    "pdb_id",
    "cluster_id",
    "cluster_id_R",
    "cluster_id_L",
)
INVENTORY_REQUIRED_COLUMNS = {
    "id",
    "pdb_id",
    "antibody_monomer",
    "antigen_monomer",
    "source_split",
    "label_structure_kind",
    "model_input_modality",
}
MANIFEST_COLUMNS = (
    "system_id",
    "system_key_sha256",
    "pdb_id",
    "topology",
    "heavy_monomer",
    "light_monomer",
    "fused_monomer",
    "antigen_monomer",
    "heavy_sequence",
    "light_sequence",
    "fused_sequence",
    "antigen_sequence",
    "heavy_sequence_sha256",
    "light_sequence_sha256",
    "fused_sequence_sha256",
    "antigen_sequence_sha256",
    "antibody_axis_sequence_sha256",
    "support_hl_edge_ids",
    "support_heavy_antigen_edge_ids",
    "support_light_antigen_edge_ids",
    "support_fused_antigen_edge_ids",
    "support_antibody_antigen_edge_count",
    "support_interface_cluster_ids",
    "source_sabdab_row_sha256s",
    "source_sabdab_row_count",
    "source_split",
    "antigen_scope",
    "label_structure_kind",
    "model_input_modality",
)


class ReconstructionError(ValueError):
    """Raised when a frozen authority or reconstruction invariant differs."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def atomic_create_bytes(path: Path, content: bytes) -> None:
    target = path.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".staged", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError as error:
            raise FileExistsError(f"refusing to overwrite {target}") from error
        directory_descriptor = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_create_json(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_create_bytes(
        path,
        (
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8"),
    )


def deterministic_gzip(content: bytes) -> bytes:
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", filename="", mtime=0) as handle:
        handle.write(content)
    return buffer.getvalue()


def tsv_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=list(MANIFEST_COLUMNS),
        delimiter="\t",
        lineterminator="\n",
        extrasaction="raise",
    )
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def load_protocol(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "authorization",
        "claim_boundary",
        "deduplication_key",
        "expected_counts",
        "format_version",
        "pinder_release",
        "reconstruction",
        "source_sha256",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ReconstructionError("protocol schema differs")
    if payload["format_version"] != PROTOCOL_FORMAT_VERSION:
        raise ReconstructionError("protocol format_version differs")
    expected_authorization = {
        "contact_labels_read": False,
        "model_predictions_read": False,
        "model_selection_authorized": False,
        "structures_read": False,
        "test_rows_read": False,
        "train_metadata_and_sequences_only": True,
        "training_authorized": False,
        "validation_rows_read": False,
    }
    if payload["authorization"] != expected_authorization:
        raise ReconstructionError("protocol authorization firewall differs")
    expected_deduplication = [
        "topology",
        "heavy_monomer",
        "light_monomer",
        "fused_monomer",
        "antigen_monomer",
    ]
    if payload["deduplication_key"] != expected_deduplication:
        raise ReconstructionError("deduplication key differs")
    return payload


def require_sha(path: Path, expected: str, label: str) -> str:
    observed = sha256_file(path)
    if observed != expected:
        raise ReconstructionError(f"{label} SHA mismatch: {observed} != {expected}")
    return observed


def nonempty(value: Any) -> bool:
    return not pd.isna(value) and bool(str(value).strip())


def semicolon_tokens(value: Any) -> set[str]:
    if not nonempty(value):
        return set()
    return {token.strip() for token in str(value).split(";") if token.strip()}


def role_monomers(value: Any) -> set[str]:
    monomers: set[str] = set()
    for token in semicolon_tokens(value):
        if not token.endswith(("-R", "-L")):
            raise ReconstructionError(f"invalid role monomer token: {token}")
        monomer = token[:-2]
        if "--" in monomer or "__" not in monomer:
            raise ReconstructionError(f"invalid PINDER monomer: {monomer}")
        monomers.add(monomer)
    return monomers


def edge_endpoints(edge_id: str) -> tuple[str, str]:
    endpoints = edge_id.split("--")
    if len(endpoints) != 2 or not all(endpoints):
        raise ReconstructionError(f"invalid PINDER dimer id: {edge_id}")
    return endpoints[0], endpoints[1]


def edge_connects(edge_id: str, first: str, second: str) -> bool:
    left, right = edge_endpoints(edge_id)
    return (left == first and right == second) or (left == second and right == first)


def monomer_pdb_id(monomer: str) -> str:
    if "__" not in monomer:
        raise ReconstructionError(f"invalid monomer without PDB prefix: {monomer}")
    return monomer.split("__", 1)[0].lower()


def stable_row_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    if hasattr(value, "item"):
        value = value.item()
    return value


def sabdab_row_sha256(row: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {column: stable_row_value(row[column]) for column in SABDAB_COLUMNS}
    )


def classify_topology(row: Mapping[str, Any]) -> str | None:
    if bool(stable_row_value(row["scfv"]) or False):
        return "scfv"
    has_heavy = nonempty(row["Hchain"])
    has_light = nonempty(row["Lchain"])
    if has_heavy and has_light:
        return "fab"
    if has_heavy and not has_light:
        return "heavy_only_or_vhh"
    return None


def read_inventory(path: Path) -> dict[str, dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows or not INVENTORY_REQUIRED_COLUMNS.issubset(rows[0]):
        raise ReconstructionError("inventory manifest schema differs")
    by_id: dict[str, dict[str, str]] = {}
    for row in rows:
        edge_id = row["id"]
        if edge_id in by_id:
            raise ReconstructionError(f"duplicate inventory edge: {edge_id}")
        if (
            row["source_split"] != "train"
            or row["model_input_modality"] != "sequence_only"
            or row["label_structure_kind"] != "experimental_holo_dimer"
        ):
            raise ReconstructionError("inventory claim boundary differs")
        if not edge_connects(
            edge_id, row["antibody_monomer"], row["antigen_monomer"]
        ):
            raise ReconstructionError(f"inventory role endpoints differ: {edge_id}")
        by_id[edge_id] = row
    if len(by_id) != 35_366:
        raise ReconstructionError(f"expected 35,366 inventory edges, found {len(by_id)}")
    return by_id


def read_sabdab(path: Path) -> pd.DataFrame:
    frame = pq.read_table(path, columns=list(SABDAB_COLUMNS)).to_pandas()
    if len(frame) != 18_204:
        raise ReconstructionError(f"expected 18,204 SAbDab rows, found {len(frame)}")
    return frame


def read_mapped_index(
    path: Path, mapped_edge_ids: set[str]
) -> tuple[dict[str, dict[str, str]], dict[str, int], int]:
    by_id: dict[str, dict[str, str]] = {}
    split_counts: Counter[str] = Counter()
    total_rows = 0
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=131_072, columns=list(INDEX_COLUMNS)):
        for row in batch.to_pylist():
            total_rows += 1
            edge_id = str(row["id"])
            if edge_id not in mapped_edge_ids:
                continue
            if edge_id in by_id:
                raise ReconstructionError(f"duplicate PINDER index id: {edge_id}")
            normalized = {key: str(row[key]) for key in INDEX_COLUMNS}
            by_id[edge_id] = normalized
            split_counts[normalized["split"]] += 1
    missing = mapped_edge_ids - set(by_id)
    if missing:
        raise ReconstructionError(f"SAbDab mapping references missing edge: {min(missing)}")
    return by_id, dict(sorted(split_counts.items())), total_rows


def validate_inventory_against_index(
    inventory: Mapping[str, Mapping[str, str]],
    index_by_id: Mapping[str, Mapping[str, str]],
) -> None:
    for edge_id, row in inventory.items():
        if edge_id not in index_by_id:
            raise ReconstructionError(f"inventory edge absent from SAbDab mappings: {edge_id}")
        index_row = index_by_id[edge_id]
        if index_row["split"] != "train":
            raise ReconstructionError(f"inventory edge is not train: {edge_id}")
        if index_row["pdb_id"].lower() != row["pdb_id"].lower():
            raise ReconstructionError(f"inventory/index PDB differs: {edge_id}")


def empty_system(key: tuple[str, str, str, str, str]) -> dict[str, Any]:
    topology, heavy, light, fused, antigen = key
    pdb_ids = {
        monomer_pdb_id(value) for value in (heavy, light, fused, antigen) if value
    }
    if len(pdb_ids) != 1:
        raise ReconstructionError(f"system spans PDB IDs: {key}")
    return {
        "topology": topology,
        "heavy_monomer": heavy,
        "light_monomer": light,
        "fused_monomer": fused,
        "antigen_monomer": antigen,
        "pdb_id": next(iter(pdb_ids)),
        "support_hl_edge_ids": set(),
        "support_heavy_antigen_edge_ids": set(),
        "support_light_antigen_edge_ids": set(),
        "support_fused_antigen_edge_ids": set(),
        "source_sabdab_row_sha256s": set(),
    }


def add_system(
    systems: dict[tuple[str, str, str, str, str], dict[str, Any]],
    key: tuple[str, str, str, str, str],
    source_row_sha256: str,
    hl_edges: Iterable[str] = (),
    heavy_edges: Iterable[str] = (),
    light_edges: Iterable[str] = (),
    fused_edges: Iterable[str] = (),
) -> None:
    system = systems.get(key)
    if system is None:
        system = empty_system(key)
        systems[key] = system
    system["support_hl_edge_ids"].update(hl_edges)
    system["support_heavy_antigen_edge_ids"].update(heavy_edges)
    system["support_light_antigen_edge_ids"].update(light_edges)
    system["support_fused_antigen_edge_ids"].update(fused_edges)
    system["source_sabdab_row_sha256s"].add(source_row_sha256)


def strict_support_edges(
    role_ids: set[str],
    antigen_ids: set[str],
    train_ids: set[str],
    inventory: Mapping[str, Mapping[str, str]],
    antibody_monomer: str,
    antigen_monomer: str,
) -> tuple[str, ...]:
    selected: list[str] = []
    for edge_id in role_ids & antigen_ids & train_ids:
        if edge_id not in inventory:
            continue
        if not edge_connects(edge_id, antibody_monomer, antigen_monomer):
            continue
        row = inventory[edge_id]
        if (
            row["antibody_monomer"] != antibody_monomer
            or row["antigen_monomer"] != antigen_monomer
        ):
            raise ReconstructionError(f"inventory role direction differs: {edge_id}")
        selected.append(edge_id)
    return tuple(sorted(selected))


def reconstruct_systems(
    sabdab: pd.DataFrame,
    index_by_id: Mapping[str, Mapping[str, str]],
    inventory: Mapping[str, Mapping[str, str]],
) -> tuple[dict[tuple[str, str, str, str, str], dict[str, Any]], dict[str, Any]]:
    train_ids = {
        edge_id for edge_id, row in index_by_id.items() if row["split"] == "train"
    }
    systems: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    exclusions: Counter[str] = Counter()
    row_topologies: Counter[str] = Counter()
    raw_system_count = 0

    for row in sabdab.to_dict(orient="records"):
        topology = classify_topology(row)
        row_topologies[topology or "unsupported"] += 1
        if topology is None:
            exclusions["unsupported_light_only_or_no_antibody_row"] += 1
            continue

        heavy_monomers = role_monomers(row["pinder_Hchain_monomers"])
        light_monomers = role_monomers(row["pinder_Lchain_monomers"])
        antigen_monomers = role_monomers(row["pinder_antigen_chain_monomers"])
        heavy_ids = semicolon_tokens(row["pinder_Hchain_ids"]) & train_ids
        light_ids = semicolon_tokens(row["pinder_Lchain_ids"]) & train_ids
        antigen_ids = semicolon_tokens(row["pinder_antigen_chain_ids"]) & train_ids
        row_sha256 = sabdab_row_sha256(row)
        row_pdb_id = str(row["pdb_id"]).lower()

        if not antigen_monomers:
            exclusions[f"{topology}_missing_antigen_monomers"] += 1
            continue
        if (heavy_monomers | light_monomers) & antigen_monomers:
            exclusions[f"{topology}_antibody_antigen_role_overlap"] += 1
            continue
        all_row_monomers = heavy_monomers | light_monomers | antigen_monomers
        if any(monomer_pdb_id(monomer) != row_pdb_id for monomer in all_row_monomers):
            raise ReconstructionError("SAbDab row maps across PDB IDs")

        emitted = 0
        if topology == "fab":
            if not heavy_monomers or not light_monomers:
                exclusions["fab_missing_heavy_or_light_monomer"] += 1
                continue
            hl_supports: list[tuple[str, str, str]] = []
            for edge_id in sorted(heavy_ids & light_ids):
                left, right = edge_endpoints(edge_id)
                if left in heavy_monomers and right in light_monomers:
                    hl_supports.append((edge_id, left, right))
                elif right in heavy_monomers and left in light_monomers:
                    hl_supports.append((edge_id, right, left))
            if not hl_supports:
                exclusions["fab_missing_train_hl_support"] += 1
                continue
            for hl_edge, heavy, light in hl_supports:
                for antigen in sorted(antigen_monomers):
                    heavy_edges = strict_support_edges(
                        heavy_ids,
                        antigen_ids,
                        train_ids,
                        inventory,
                        heavy,
                        antigen,
                    )
                    light_edges = strict_support_edges(
                        light_ids,
                        antigen_ids,
                        train_ids,
                        inventory,
                        light,
                        antigen,
                    )
                    if not heavy_edges and not light_edges:
                        continue
                    add_system(
                        systems,
                        (topology, heavy, light, "", antigen),
                        row_sha256,
                        hl_edges=(hl_edge,),
                        heavy_edges=heavy_edges,
                        light_edges=light_edges,
                    )
                    emitted += 1
        elif topology == "heavy_only_or_vhh":
            if not heavy_monomers:
                exclusions["heavy_only_or_vhh_missing_heavy_monomer"] += 1
                continue
            for heavy in sorted(heavy_monomers):
                for antigen in sorted(antigen_monomers):
                    heavy_edges = strict_support_edges(
                        heavy_ids,
                        antigen_ids,
                        train_ids,
                        inventory,
                        heavy,
                        antigen,
                    )
                    if not heavy_edges:
                        continue
                    add_system(
                        systems,
                        (topology, heavy, "", "", antigen),
                        row_sha256,
                        heavy_edges=heavy_edges,
                    )
                    emitted += 1
        else:
            fused_monomers = heavy_monomers | light_monomers
            if not fused_monomers:
                exclusions["scfv_missing_fused_monomer"] += 1
                continue
            fused_role_ids = heavy_ids | light_ids
            for fused in sorted(fused_monomers):
                for antigen in sorted(antigen_monomers):
                    fused_edges = strict_support_edges(
                        fused_role_ids,
                        antigen_ids,
                        train_ids,
                        inventory,
                        fused,
                        antigen,
                    )
                    if not fused_edges:
                        continue
                    add_system(
                        systems,
                        (topology, "", "", fused, antigen),
                        row_sha256,
                        fused_edges=fused_edges,
                    )
                    emitted += 1

        if emitted == 0:
            exclusions[f"{topology}_missing_strict_train_antigen_support"] += 1
        raw_system_count += emitted

    for key, system in systems.items():
        topology = key[0]
        antigen_edges = (
            system["support_heavy_antigen_edge_ids"]
            | system["support_light_antigen_edge_ids"]
            | system["support_fused_antigen_edge_ids"]
        )
        if not antigen_edges:
            raise ReconstructionError(f"system lacks antibody-antigen support: {key}")
        if topology == "fab" and len(system["support_hl_edge_ids"]) != 1:
            raise ReconstructionError(f"Fab lacks one exact H-L support: {key}")
        if topology != "fab" and system["support_hl_edge_ids"]:
            raise ReconstructionError(f"non-Fab unexpectedly has H-L support: {key}")

    diagnostics = {
        "raw_system_occurrence_count": raw_system_count,
        "duplicate_occurrences_collapsed": raw_system_count - len(systems),
        "row_topology_counts": dict(sorted(row_topologies.items())),
        "exclusion_counts": dict(sorted(exclusions.items())),
    }
    return systems, diagnostics


def sequence_monomer_key(pdb_key: str, pdb_kind: str) -> str | None:
    suffix = {"receptor": "-R.pdb", "ligand": "-L.pdb"}.get(pdb_kind)
    if suffix is None:
        return None
    if not pdb_key.endswith(suffix):
        raise ReconstructionError(f"sequence key/kind suffix differs: {pdb_key}")
    return pdb_key[: -len(suffix)]


def load_used_sequences(path: Path, used_monomers: set[str]) -> dict[str, str]:
    sequences: dict[str, str] = {}
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(
        batch_size=131_072, columns=["pdb", "sequence", "pdb_kind"]
    ):
        for row in batch.to_pylist():
            key = sequence_monomer_key(str(row["pdb"]), str(row["pdb_kind"]))
            if key is None or key not in used_monomers:
                continue
            sequence = str(row["sequence"]).upper()
            if not sequence or not set(sequence).issubset(set(string.ascii_uppercase)):
                raise ReconstructionError(f"invalid monomer sequence: {key}")
            prior = sequences.setdefault(key, sequence)
            if prior != sequence:
                raise ReconstructionError(f"conflicting sequences for monomer: {key}")
    missing = used_monomers - set(sequences)
    if missing:
        raise ReconstructionError(f"missing monomer sequence: {min(missing)}")
    return sequences


def sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def antibody_axis_sha256(topology: str, heavy: str, light: str, fused: str) -> str:
    if topology == "fab":
        payload = {"heavy": heavy, "light": light, "topology": topology}
    elif topology == "heavy_only_or_vhh":
        payload = {"heavy": heavy, "topology": topology}
    else:
        payload = {"fused": fused, "topology": topology}
    return canonical_sha256(payload)


def manifest_rows(
    systems: Mapping[tuple[str, str, str, str, str], Mapping[str, Any]],
    sequences: Mapping[str, str],
    index_by_id: Mapping[str, Mapping[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_system_ids: set[str] = set()
    for key in sorted(systems):
        system = systems[key]
        topology, heavy_monomer, light_monomer, fused_monomer, antigen_monomer = key
        key_payload = {
            "antigen_monomer": antigen_monomer,
            "fused_monomer": fused_monomer,
            "heavy_monomer": heavy_monomer,
            "light_monomer": light_monomer,
            "topology": topology,
        }
        key_sha256 = canonical_sha256(key_payload)
        system_id = f"pfs-{key_sha256[:24]}"
        if system_id in seen_system_ids:
            raise ReconstructionError(f"truncated system-id collision: {system_id}")
        seen_system_ids.add(system_id)

        heavy_sequence = sequences[heavy_monomer] if heavy_monomer else ""
        light_sequence = sequences[light_monomer] if light_monomer else ""
        fused_sequence = sequences[fused_monomer] if fused_monomer else ""
        antigen_sequence = sequences[antigen_monomer]
        hl_edges = sorted(system["support_hl_edge_ids"])
        heavy_edges = sorted(system["support_heavy_antigen_edge_ids"])
        light_edges = sorted(system["support_light_antigen_edge_ids"])
        fused_edges = sorted(system["support_fused_antigen_edge_ids"])
        antigen_edges = heavy_edges + light_edges + fused_edges
        interface_clusters = sorted(
            {index_by_id[edge_id]["cluster_id"] for edge_id in antigen_edges}
        )
        source_rows = sorted(system["source_sabdab_row_sha256s"])
        rows.append(
            {
                "system_id": system_id,
                "system_key_sha256": key_sha256,
                "pdb_id": system["pdb_id"],
                "topology": topology,
                "heavy_monomer": heavy_monomer,
                "light_monomer": light_monomer,
                "fused_monomer": fused_monomer,
                "antigen_monomer": antigen_monomer,
                "heavy_sequence": heavy_sequence,
                "light_sequence": light_sequence,
                "fused_sequence": fused_sequence,
                "antigen_sequence": antigen_sequence,
                "heavy_sequence_sha256": (
                    sequence_sha256(heavy_sequence) if heavy_sequence else ""
                ),
                "light_sequence_sha256": (
                    sequence_sha256(light_sequence) if light_sequence else ""
                ),
                "fused_sequence_sha256": (
                    sequence_sha256(fused_sequence) if fused_sequence else ""
                ),
                "antigen_sequence_sha256": sequence_sha256(antigen_sequence),
                "antibody_axis_sequence_sha256": antibody_axis_sha256(
                    topology, heavy_sequence, light_sequence, fused_sequence
                ),
                "support_hl_edge_ids": ";".join(hl_edges),
                "support_heavy_antigen_edge_ids": ";".join(heavy_edges),
                "support_light_antigen_edge_ids": ";".join(light_edges),
                "support_fused_antigen_edge_ids": ";".join(fused_edges),
                "support_antibody_antigen_edge_count": len(antigen_edges),
                "support_interface_cluster_ids": ";".join(interface_clusters),
                "source_sabdab_row_sha256s": ";".join(source_rows),
                "source_sabdab_row_count": len(source_rows),
                "source_split": "train",
                "antigen_scope": "single_pinder_protein_chain",
                "label_structure_kind": "experimental_holo_dimers_not_read_at_gate",
                "model_input_modality": "sequence_only",
            }
        )
    return rows


def summarize_manifest(rows: list[dict[str, Any]]) -> dict[str, Any]:
    topology_counts = Counter(row["topology"] for row in rows)
    fab_support = Counter()
    for row in rows:
        if row["topology"] == "fab":
            support = (
                bool(row["support_heavy_antigen_edge_ids"]),
                bool(row["support_light_antigen_edge_ids"]),
            )
            label = {
                (True, True): "heavy_and_light",
                (True, False): "heavy_only",
                (False, True): "light_only",
            }[support]
            fab_support[label] += 1
    exact_pairs = Counter(
        (row["antibody_axis_sequence_sha256"], row["antigen_sequence_sha256"])
        for row in rows
    )
    multiplicities = sorted(exact_pairs.values())
    standard_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
    unique_sequences = {
        sequence
        for row in rows
        for sequence in (
            row["heavy_sequence"],
            row["light_sequence"],
            row["fused_sequence"],
            row["antigen_sequence"],
        )
        if sequence
    }
    return {
        "system_count": len(rows),
        "topology_counts": dict(sorted(topology_counts.items())),
        "unique_pdb_count": len({row["pdb_id"] for row in rows}),
        "unique_antibody_axis_sequence_count": len(
            {row["antibody_axis_sequence_sha256"] for row in rows}
        ),
        "unique_antigen_sequence_count": len(
            {row["antigen_sequence_sha256"] for row in rows}
        ),
        "unique_exact_sequence_pair_count": len(exact_pairs),
        "exact_pair_multiplicity_median": (
            float(pd.Series(multiplicities).median()) if multiplicities else 0.0
        ),
        "exact_pair_multiplicity_p95": (
            float(pd.Series(multiplicities).quantile(0.95)) if multiplicities else 0.0
        ),
        "exact_pair_multiplicity_max": max(multiplicities, default=0),
        "fab_antigen_support_patterns": dict(sorted(fab_support.items())),
        "unique_support_antibody_antigen_edge_count": len(
            {
                edge_id
                for row in rows
                for field in (
                    "support_heavy_antigen_edge_ids",
                    "support_light_antigen_edge_ids",
                    "support_fused_antigen_edge_ids",
                )
                for edge_id in str(row[field]).split(";")
                if edge_id
            }
        ),
        "unique_support_hl_edge_count": len(
            {
                edge_id
                for row in rows
                for edge_id in str(row["support_hl_edge_ids"]).split(";")
                if edge_id
            }
        ),
        "unique_sequence_count": len(unique_sequences),
        "unique_sequence_with_noncanonical_residue_count": sum(
            not set(sequence).issubset(standard_amino_acids)
            for sequence in unique_sequences
        ),
    }


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    protocol = load_protocol(arguments.protocol_json)
    input_paths = {
        "index_parquet": arguments.index_parquet,
        "inventory_manifest": arguments.inventory_manifest,
        "sabdab_metadata_parquet": arguments.sabdab_metadata_parquet,
        "sequence_database_parquet": arguments.sequence_database_parquet,
    }
    observed_sha = {
        label: require_sha(path, protocol["source_sha256"][label], label)
        for label, path in input_paths.items()
    }
    observed_sha["protocol_json"] = sha256_file(arguments.protocol_json)
    observed_sha["reconstruction_source"] = sha256_file(Path(__file__))

    inventory = read_inventory(arguments.inventory_manifest)
    sabdab = read_sabdab(arguments.sabdab_metadata_parquet)
    mapped_edge_ids = set().union(
        *(
            semicolon_tokens(value)
            for column in (
                "pinder_Hchain_ids",
                "pinder_Lchain_ids",
                "pinder_antigen_chain_ids",
            )
            for value in sabdab[column]
        )
    )
    index_by_id, mapped_split_counts, index_row_count = read_mapped_index(
        arguments.index_parquet, mapped_edge_ids
    )
    validate_inventory_against_index(inventory, index_by_id)
    systems, reconstruction_diagnostics = reconstruct_systems(
        sabdab, index_by_id, inventory
    )
    used_monomers = {
        monomer
        for key in systems
        for monomer in key[1:]
        if monomer
    }
    sequences = load_used_sequences(arguments.sequence_database_parquet, used_monomers)
    rows = manifest_rows(systems, sequences, index_by_id)
    summary = summarize_manifest(rows)

    expected_counts = protocol["expected_counts"]
    observed_expected = {
        "fab": summary["topology_counts"].get("fab", 0),
        "heavy_only_or_vhh": summary["topology_counts"].get(
            "heavy_only_or_vhh", 0
        ),
        "scfv": summary["topology_counts"].get("scfv", 0),
        "total_unique_systems": summary["system_count"],
    }
    if observed_expected != expected_counts:
        raise ReconstructionError(
            f"reconstructed counts differ: {observed_expected} != {expected_counts}"
        )
    if any(row["source_split"] != "train" for row in rows):
        raise ReconstructionError("non-train system emitted")
    if len({row["system_id"] for row in rows}) != len(rows):
        raise ReconstructionError("system IDs are not unique")

    output_dir = arguments.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / MANIFEST_NAME
    atomic_create_bytes(manifest_path, deterministic_gzip(tsv_bytes(rows)))
    manifest_sha256 = sha256_file(manifest_path)

    result: dict[str, Any] = {
        "format_version": RESULT_FORMAT_VERSION,
        "status": "PASS_PINDER_TRAIN_FULL_ANTIBODY_SYSTEM_RECONSTRUCTION",
        "pinder_release": protocol["pinder_release"],
        "source_sha256": observed_sha,
        "firewall": protocol["authorization"],
        "checks": {
            "all_antibody_antigen_support_edges_in_role_exclusive_inventory": True,
            "all_support_edges_train_split": True,
            "all_used_monomer_sequences_resolved_without_conflict": True,
            "antibody_antigen_roles_disjoint": True,
            "fab_hl_support_required": True,
            "physical_copy_key_is_pinder_monomer_not_sequence_identity": True,
            "system_ids_unique": True,
            "test_rows_read": False,
            "validation_rows_read": False,
        },
        "index_authority": {
            "index_row_count": index_row_count,
            "sabdab_mapped_edge_count": len(mapped_edge_ids),
            "sabdab_mapped_edge_split_counts": mapped_split_counts,
            "strict_train_inventory_edge_count": len(inventory),
        },
        "reconstruction_diagnostics": reconstruction_diagnostics,
        "system_summary": summary,
        "artifact": {
            "filename": manifest_path.name,
            "row_count": len(rows),
            "sha256": manifest_sha256,
        },
        "claim_boundary": protocol["claim_boundary"],
        "training_authorized": False,
        "model_selection_authorized": False,
        "required_next_gate": (
            "Freeze one system-level quality/deduplication/multiplicity policy, then "
            "construct and independently audit an antibody-and-antigen two-axis split."
        ),
    }
    result["scientific_payload_sha256"] = canonical_sha256(result)
    result_path = output_dir / RESULT_NAME
    atomic_create_json(result_path, result)
    receipt = {
        "format_version": "pinder_full_antibody_system_reconstruction_receipt_v1",
        "status": result["status"],
        "result": {"filename": result_path.name, "sha256": sha256_file(result_path)},
        "manifest": result["artifact"],
        "scientific_payload_sha256": result["scientific_payload_sha256"],
        "training_authorized": False,
        "model_selection_authorized": False,
        "test_access": False,
    }
    atomic_create_json(output_dir / RECEIPT_NAME, receipt)
    return result


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-json", type=Path, required=True)
    parser.add_argument("--index-parquet", type=Path, required=True)
    parser.add_argument("--sabdab-metadata-parquet", type=Path, required=True)
    parser.add_argument("--sequence-database-parquet", type=Path, required=True)
    parser.add_argument("--inventory-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_arguments())
