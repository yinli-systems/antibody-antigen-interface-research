#!/usr/bin/env python3
"""Freeze train-only PINDER Ab--Ag sequence clusters and quality statistics.

This is a metadata/sequence audit. It never reads validation/test rows, contact
labels, model checkpoints, predictions, or scores, and it does not authorize
training or model selection.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd


FORMAT_VERSION = "pinder_antibody_sequence_cluster_quality_audit_v1"
PROTOCOL_FORMAT_VERSION = "pinder_antibody_sequence_cluster_protocol_v1"
THRESHOLDS = (30, 50, 70, 90)
EDGE_FIELDS = (
    "query",
    "target",
    "fident",
    "qcov",
    "tcov",
    "alnlen",
    "evalue",
    "bits",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
        os.link(temporary, target)
        directory_descriptor = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_create_json(path: Path, payload: Mapping[str, Any]) -> None:
    content = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    atomic_create_bytes(path, content)


def deterministic_gzip(content: bytes) -> bytes:
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", filename="", mtime=0) as handle:
        handle.write(content)
    return buffer.getvalue()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def require_exact_sha(path: Path, expected: str, label: str) -> str:
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"{label} SHA mismatch: {observed} != {expected}")
    return observed


def load_protocol(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    required = {
        "format_version",
        "pinder_release",
        "mmseqs_release",
        "mmseqs_binary_sha256",
        "minimum_sequence_identity_percent",
        "reciprocal_coverage_percent",
        "cluster_identity_thresholds_percent",
        "edge_format",
        "input_sha256",
        "quality_filters",
        "authorization",
    }
    if set(payload) != required:
        raise ValueError(
            f"protocol schema mismatch: missing={sorted(required - set(payload))}, "
            f"unexpected={sorted(set(payload) - required)}"
        )
    if payload["format_version"] != PROTOCOL_FORMAT_VERSION:
        raise ValueError("unsupported protocol format_version")
    if payload["cluster_identity_thresholds_percent"] != list(THRESHOLDS):
        raise ValueError("cluster thresholds must be [30, 50, 70, 90]")
    if payload["edge_format"] != list(EDGE_FIELDS):
        raise ValueError("edge_format does not match frozen MMseqs columns")
    if float(payload["minimum_sequence_identity_percent"]) != 30.0:
        raise ValueError("minimum sequence identity must be 30 percent")
    if float(payload["reciprocal_coverage_percent"]) != 80.0:
        raise ValueError("reciprocal coverage must be 80 percent")
    expected_quality_filters = {
        "method": "X-RAY DIFFRACTION",
        "resolution_max_angstrom": 3.5,
        "minimum_chain_length": 40,
        "minimum_atom_types": 3,
        "interface_atom_gaps_4A": 0,
        "max_coordinate_variance": 0.98,
        "components_per_chain": 1,
        "bio_subset_label": "BIO",
    }
    if payload["quality_filters"] != expected_quality_filters:
        raise ValueError("quality_filters do not match the frozen PINDER-like policy")
    authorization = payload["authorization"]
    expected_authorization = {
        "train_metadata_only": True,
        "structures_read": False,
        "contact_labels_read": False,
        "validation_rows_read": False,
        "test_rows_read": False,
        "model_predictions_read": False,
        "training_authorized": False,
        "model_selection_authorized": False,
    }
    if authorization != expected_authorization:
        raise ValueError("protocol authorization firewall mismatch")
    return payload


def read_fasta(path: Path) -> dict[str, str]:
    sequences: dict[str, str] = {}
    current: str | None = None
    chunks: list[str] = []

    def publish() -> None:
        if current is None:
            return
        sequence = "".join(chunks).upper()
        if not sequence:
            raise ValueError(f"empty FASTA sequence for {current}")
        if hashlib.sha256(sequence.encode("ascii")).hexdigest() != current:
            raise ValueError(f"FASTA identifier is not the sequence SHA256: {current}")
        if current in sequences:
            raise ValueError(f"duplicate FASTA identifier: {current}")
        sequences[current] = sequence

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                publish()
                current = line[1:].split()[0]
                chunks = []
            else:
                if current is None:
                    raise ValueError("FASTA sequence encountered before header")
                chunks.append(line)
    publish()
    return sequences


class UnionFind:
    def __init__(self, nodes: Iterable[str]) -> None:
        self.parent = {node: node for node in nodes}
        self.size = {node: 1 for node in nodes}

    def find(self, node: str) -> str:
        parent = self.parent[node]
        while parent != self.parent[parent]:
            parent = self.parent[parent]
        while node != parent:
            next_node = self.parent[node]
            self.parent[node] = parent
            node = next_node
        return parent

    def union(self, first: str, second: str) -> None:
        left, right = self.find(first), self.find(second)
        if left == right:
            return
        if (self.size[left], left) < (self.size[right], right):
            left, right = right, left
        self.parent[right] = left
        self.size[left] += self.size[right]

    def components(self) -> dict[str, tuple[str, ...]]:
        groups: dict[str, list[str]] = defaultdict(list)
        for node in sorted(self.parent):
            groups[self.find(node)].append(node)
        result: dict[str, tuple[str, ...]] = {}
        for members in groups.values():
            ordered = tuple(sorted(members))
            component_id = hashlib.sha256("\n".join(ordered).encode("ascii")).hexdigest()
            result[component_id] = ordered
        return dict(sorted(result.items()))


def fraction_as_percent(value: str, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite {label}")
    # MMseqs convertalis emits fident/qcov/tcov as fractions in [0, 1].
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{label} outside [0, 1]: {number}")
    return round(100.0 * number, 6)


def load_edges(
    path: Path,
    nodes: set[str],
    minimum_identity: float,
    minimum_coverage: float,
) -> tuple[list[tuple[str, str, float]], dict[str, Any]]:
    undirected: dict[tuple[str, str], float] = {}
    directional_rows = 0
    self_rows = 0
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for line_number, row in enumerate(reader, start=1):
            if len(row) != len(EDGE_FIELDS):
                raise ValueError(f"{path}:{line_number} has {len(row)} columns")
            query, target = row[0], row[1]
            if query not in nodes or target not in nodes:
                raise ValueError(f"unauthorized edge endpoint at {path}:{line_number}")
            identity = fraction_as_percent(row[2], "fident")
            qcov = fraction_as_percent(row[3], "qcov")
            tcov = fraction_as_percent(row[4], "tcov")
            alnlen = int(row[5])
            evalue, bits = float(row[6]), float(row[7])
            if alnlen <= 0 or not all(math.isfinite(x) for x in (evalue, bits)):
                raise ValueError(f"invalid alignment statistics at {path}:{line_number}")
            if identity + 1e-9 < minimum_identity:
                raise ValueError(f"edge below identity threshold at {path}:{line_number}")
            if min(qcov, tcov) + 1e-9 < minimum_coverage:
                raise ValueError(f"edge below reciprocal coverage at {path}:{line_number}")
            directional_rows += 1
            if query == target:
                self_rows += 1
                continue
            pair = tuple(sorted((query, target)))
            undirected[pair] = max(identity, undirected.get(pair, -1.0))
    edges = [(left, right, identity) for (left, right), identity in undirected.items()]
    edges.sort(key=lambda item: (item[0], item[1]))
    return edges, {
        "directional_row_count": directional_rows,
        "self_row_count": self_rows,
        "unique_undirected_nonself_edge_count": len(edges),
    }


def cluster_role(
    role: str,
    sequences: Mapping[str, str],
    edges: list[tuple[str, str, float]],
) -> tuple[dict[int, dict[str, str]], dict[str, Any]]:
    assignments: dict[int, dict[str, str]] = {}
    summaries: dict[str, Any] = {}
    for threshold in THRESHOLDS:
        union_find = UnionFind(sequences)
        accepted = 0
        for left, right, identity in edges:
            if identity + 1e-9 >= threshold:
                union_find.union(left, right)
                accepted += 1
        components = union_find.components()
        node_assignment = {
            member: component_id
            for component_id, members in components.items()
            for member in members
        }
        if set(node_assignment) != set(sequences):
            raise RuntimeError(f"{role} threshold {threshold} has incomplete accounting")
        sizes = sorted((len(members) for members in components.values()), reverse=True)
        assignments[threshold] = node_assignment
        summaries[str(threshold)] = {
            "accepted_undirected_edge_count": accepted,
            "component_count": len(components),
            "singleton_component_count": sum(size == 1 for size in sizes),
            "largest_component_size": sizes[0],
            "median_component_size": float(pd.Series(sizes).median()),
        }

    for lower, higher in zip(THRESHOLDS, THRESHOLDS[1:]):
        lower_assignment = assignments[lower]
        higher_assignment = assignments[higher]
        contained: dict[str, set[str]] = defaultdict(set)
        for node in sequences:
            contained[higher_assignment[node]].add(lower_assignment[node])
        if any(len(parent_components) != 1 for parent_components in contained.values()):
            raise RuntimeError(
                f"{role} threshold {higher} is not a refinement of threshold {lower}"
            )
    return assignments, summaries


def cluster_role_from_edge_file(
    role: str,
    sequences: Mapping[str, str],
    path: Path,
    minimum_identity: float,
    minimum_coverage: float,
) -> tuple[dict[int, dict[str, str]], dict[str, Any], dict[str, Any]]:
    """Build all nested components in one streaming pass over a large edge TSV."""
    nodes = set(sequences)
    union_finds = {threshold: UnionFind(nodes) for threshold in THRESHOLDS}
    accepted_directional = Counter({threshold: 0 for threshold in THRESHOLDS})
    directional_rows = 0
    self_rows = 0
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for line_number, row in enumerate(reader, start=1):
            if len(row) != len(EDGE_FIELDS):
                raise ValueError(f"{path}:{line_number} has {len(row)} columns")
            query, target = row[0], row[1]
            if query not in nodes or target not in nodes:
                raise ValueError(f"unauthorized edge endpoint at {path}:{line_number}")
            identity = fraction_as_percent(row[2], "fident")
            qcov = fraction_as_percent(row[3], "qcov")
            tcov = fraction_as_percent(row[4], "tcov")
            alnlen = int(row[5])
            evalue, bits = float(row[6]), float(row[7])
            if alnlen <= 0 or not all(math.isfinite(x) for x in (evalue, bits)):
                raise ValueError(f"invalid alignment statistics at {path}:{line_number}")
            if identity + 1e-9 < minimum_identity:
                raise ValueError(f"edge below identity threshold at {path}:{line_number}")
            if min(qcov, tcov) + 1e-9 < minimum_coverage:
                raise ValueError(f"edge below reciprocal coverage at {path}:{line_number}")
            directional_rows += 1
            if query == target:
                self_rows += 1
                continue
            for threshold, union_find in union_finds.items():
                if identity + 1e-9 >= threshold:
                    union_find.union(query, target)
                    accepted_directional[threshold] += 1

    assignments: dict[int, dict[str, str]] = {}
    summaries: dict[str, Any] = {}
    for threshold in THRESHOLDS:
        components = union_finds[threshold].components()
        node_assignment = {
            member: component_id
            for component_id, members in components.items()
            for member in members
        }
        if set(node_assignment) != nodes:
            raise RuntimeError(f"{role} threshold {threshold} has incomplete accounting")
        sizes = sorted((len(members) for members in components.values()), reverse=True)
        assignments[threshold] = node_assignment
        summaries[str(threshold)] = {
            "accepted_directional_nonself_edge_count": accepted_directional[threshold],
            "component_count": len(components),
            "singleton_component_count": sum(size == 1 for size in sizes),
            "largest_component_size": sizes[0],
            "median_component_size": float(pd.Series(sizes).median()),
        }

    for lower, higher in zip(THRESHOLDS, THRESHOLDS[1:]):
        contained: dict[str, set[str]] = defaultdict(set)
        for node in nodes:
            contained[assignments[higher][node]].add(assignments[lower][node])
        if any(len(parent_components) != 1 for parent_components in contained.values()):
            raise RuntimeError(
                f"{role} threshold {higher} is not a refinement of threshold {lower}"
            )
    edge_stats = {
        "directional_row_count": directional_rows,
        "self_row_count": self_rows,
        "directional_nonself_row_count": directional_rows - self_rows,
    }
    return assignments, summaries, edge_stats


def read_sequence_manifest(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    required = {
        "id",
        "pdb_id",
        "antibody_monomer",
        "antigen_monomer",
        "interface_cluster_id",
        "source_split",
        "label_structure_kind",
        "model_input_modality",
        "antibody_sequence_sha256",
        "antigen_sequence_sha256",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError("sequence manifest schema mismatch")
    if any(row["source_split"] != "train" for row in rows):
        raise ValueError("sequence manifest contains non-train rows")
    if any(row["model_input_modality"] != "sequence_only" for row in rows):
        raise ValueError("unexpected model_input_modality")
    return rows


def official_like_xray_quality(metadata: pd.DataFrame) -> pd.Series:
    resolution = pd.to_numeric(metadata["resolution"], errors="coerce")
    length1 = pd.to_numeric(metadata["length1"], errors="coerce")
    length2 = pd.to_numeric(metadata["length2"], errors="coerce")
    atom_types = pd.to_numeric(metadata["num_atom_types"], errors="coerce")
    gaps = pd.to_numeric(metadata["interface_atom_gaps_4A"], errors="coerce")
    max_var1 = pd.to_numeric(metadata["max_var_1"], errors="coerce")
    max_var2 = pd.to_numeric(metadata["max_var_2"], errors="coerce")
    components1 = pd.to_numeric(metadata["number_of_components_1"], errors="coerce")
    components2 = pd.to_numeric(metadata["number_of_components_2"], errors="coerce")
    return (
        metadata["method"].eq("X-RAY DIFFRACTION")
        & resolution.le(3.5)
        & length1.ge(40)
        & length2.ge(40)
        & atom_types.ge(3)
        & gaps.eq(0)
        & max_var1.le(0.98)
        & max_var2.le(0.98)
        & components1.eq(1)
        & components2.eq(1)
    )


def subset_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    sequence_pairs = [
        (row["antibody_sequence_sha256"], row["antigen_sequence_sha256"])
        for row in rows
    ]
    pdb_sequence_pairs = [
        (row["pdb_id"], row["antibody_sequence_sha256"], row["antigen_sequence_sha256"])
        for row in rows
    ]
    multiplicities = Counter(sequence_pairs)
    values = sorted(multiplicities.values())
    return {
        "row_count": len(rows),
        "unique_pdb_count": len({row["pdb_id"] for row in rows}),
        "unique_interface_cluster_count": len(
            {row["interface_cluster_id"] for row in rows}
        ),
        "unique_exact_sequence_pair_count": len(set(sequence_pairs)),
        "unique_pdb_sequence_pair_count": len(set(pdb_sequence_pairs)),
        "unique_antibody_sequence_count": len({pair[0] for pair in sequence_pairs}),
        "unique_antigen_sequence_count": len({pair[1] for pair in sequence_pairs}),
        "exact_pair_multiplicity_median": float(pd.Series(values).median()) if values else 0.0,
        "exact_pair_multiplicity_p95": float(pd.Series(values).quantile(0.95)) if values else 0.0,
        "exact_pair_multiplicity_max": max(values, default=0),
    }


def tsv_bytes(fieldnames: list[str], rows: Iterable[Mapping[str, Any]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=fieldnames,
        delimiter="\t",
        lineterminator="\n",
        extrasaction="raise",
    )
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    protocol = load_protocol(arguments.protocol_json)
    paths = {
        "inventory_manifest": arguments.inventory_manifest,
        "sequence_manifest": arguments.sequence_manifest,
        "metadata_parquet": arguments.metadata_parquet,
        "antibody_fasta": arguments.antibody_fasta,
        "antigen_fasta": arguments.antigen_fasta,
        "antibody_edges_tsv": arguments.antibody_edges_tsv,
        "antigen_edges_tsv": arguments.antigen_edges_tsv,
    }
    observed_sha = {
        label: require_exact_sha(path, protocol["input_sha256"][label], label)
        for label, path in paths.items()
    }
    observed_sha["protocol_json"] = sha256_file(arguments.protocol_json)
    observed_sha["audit_source"] = sha256_file(Path(__file__))

    antibody_sequences = read_fasta(arguments.antibody_fasta)
    antigen_sequences = read_fasta(arguments.antigen_fasta)
    if set(antibody_sequences) & set(antigen_sequences):
        raise ValueError("antibody and antigen sequence roles overlap")
    (
        antibody_assignments,
        antibody_clusters,
        antibody_edge_stats,
    ) = cluster_role_from_edge_file(
        "antibody",
        antibody_sequences,
        arguments.antibody_edges_tsv,
        30.0,
        80.0,
    )
    (
        antigen_assignments,
        antigen_clusters,
        antigen_edge_stats,
    ) = cluster_role_from_edge_file(
        "antigen",
        antigen_sequences,
        arguments.antigen_edges_tsv,
        30.0,
        80.0,
    )

    rows = read_sequence_manifest(arguments.sequence_manifest)
    if len(rows) != 35_366:
        raise ValueError(f"expected 35,366 train rows, observed {len(rows)}")
    if any(row["antibody_sequence_sha256"] not in antibody_sequences for row in rows):
        raise ValueError("sequence manifest references missing antibody sequence")
    if any(row["antigen_sequence_sha256"] not in antigen_sequences for row in rows):
        raise ValueError("sequence manifest references missing antigen sequence")

    metadata = pd.read_parquet(arguments.metadata_parquet)
    if metadata["id"].duplicated().any():
        raise ValueError("PINDER metadata id is not unique")
    metadata = metadata.set_index("id", drop=False)
    row_ids = {row["id"] for row in rows}
    if not row_ids.issubset(set(metadata.index)):
        raise ValueError("train inventory ids missing from PINDER metadata")
    selected_metadata = metadata.loc[sorted(row_ids)].copy()
    quality_mask = official_like_xray_quality(selected_metadata)
    xray_ids = set(selected_metadata.loc[quality_mask, "id"])
    xray_bio_ids = set(
        selected_metadata.loc[quality_mask & selected_metadata["label"].eq("BIO"), "id"]
    )

    enriched_rows: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: item["id"]):
        enriched: dict[str, Any] = {
            "id": row["id"],
            "pdb_id": row["pdb_id"],
            "antibody_sequence_sha256": row["antibody_sequence_sha256"],
            "antigen_sequence_sha256": row["antigen_sequence_sha256"],
            "official_like_xray_quality": row["id"] in xray_ids,
            "official_like_xray_bio_quality": row["id"] in xray_bio_ids,
        }
        for threshold in THRESHOLDS:
            enriched[f"antibody_component_{threshold}"] = antibody_assignments[threshold][
                row["antibody_sequence_sha256"]
            ]
            enriched[f"antigen_component_{threshold}"] = antigen_assignments[threshold][
                row["antigen_sequence_sha256"]
            ]
        enriched_rows.append(enriched)

    all_rows = rows
    xray_rows = [row for row in rows if row["id"] in xray_ids]
    xray_bio_rows = [row for row in rows if row["id"] in xray_bio_ids]
    pair_component_counts: dict[str, Any] = {}
    for threshold in THRESHOLDS:
        pair_component_counts[str(threshold)] = {
            "all_train_unique_component_pairs": len(
                {
                    (
                        antibody_assignments[threshold][row["antibody_sequence_sha256"]],
                        antigen_assignments[threshold][row["antigen_sequence_sha256"]],
                    )
                    for row in all_rows
                }
            ),
            "official_like_xray_unique_component_pairs": len(
                {
                    (
                        antibody_assignments[threshold][row["antibody_sequence_sha256"]],
                        antigen_assignments[threshold][row["antigen_sequence_sha256"]],
                    )
                    for row in xray_rows
                }
            ),
            "official_like_xray_bio_unique_component_pairs": len(
                {
                    (
                        antibody_assignments[threshold][row["antibody_sequence_sha256"]],
                        antigen_assignments[threshold][row["antigen_sequence_sha256"]],
                    )
                    for row in xray_bio_rows
                }
            ),
        }

    assignment_fields = [
        "id",
        "pdb_id",
        "antibody_sequence_sha256",
        "antigen_sequence_sha256",
        "official_like_xray_quality",
        "official_like_xray_bio_quality",
    ] + [
        f"{role}_component_{threshold}"
        for threshold in THRESHOLDS
        for role in ("antibody", "antigen")
    ]
    output_dir = arguments.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    assignment_path = output_dir / "PINDER_ANTIBODY_SEQUENCE_CLUSTER_ASSIGNMENTS_V1.tsv.gz"
    atomic_create_bytes(
        assignment_path,
        deterministic_gzip(tsv_bytes(assignment_fields, enriched_rows)),
    )

    result: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "status": "PASS_PINDER_ANTIBODY_SEQUENCE_CLUSTER_AND_QUALITY_AUDIT",
        "pinder_release": protocol["pinder_release"],
        "mmseqs_release": protocol["mmseqs_release"],
        "input_sha256": observed_sha,
        "firewall": protocol["authorization"],
        "sequence_inventory": {
            "antibody_unique_sequence_count": len(antibody_sequences),
            "antigen_unique_sequence_count": len(antigen_sequences),
            "role_overlap_count": 0,
        },
        "edge_inventory": {
            "antibody": antibody_edge_stats,
            "antigen": antigen_edge_stats,
        },
        "component_summary": {
            "antibody": antibody_clusters,
            "antigen": antigen_clusters,
        },
        "quality_subset_summary": {
            "all_train_role_exclusive": subset_summary(all_rows),
            "official_like_xray": subset_summary(xray_rows),
            "official_like_xray_bio": subset_summary(xray_bio_rows),
        },
        "unique_component_pair_summary": pair_component_counts,
        "threshold_partitions_are_nested": True,
        "artifacts": {
            "cluster_assignments": {
                "filename": assignment_path.name,
                "sha256": sha256_file(assignment_path),
                "row_count": len(enriched_rows),
            }
        },
        "authorized_for_training": False,
        "authorized_for_model_selection": False,
        "required_next_gate": (
            "freeze one quality and deduplication policy, build a two-axis "
            "similarity-disconnected train/development split, and independently audit "
            "all cross-fold antibody and antigen similarities before training"
        ),
    }
    result["scientific_payload_sha256"] = canonical_sha256(result)
    result_path = output_dir / "PINDER_ANTIBODY_SEQUENCE_CLUSTER_RESULT_V1.json"
    atomic_create_json(result_path, result)
    receipt = {
        "format_version": "pinder_antibody_sequence_cluster_receipt_v1",
        "status": result["status"],
        "result": {"filename": result_path.name, "sha256": sha256_file(result_path)},
        "cluster_assignments": result["artifacts"]["cluster_assignments"],
        "scientific_payload_sha256": result["scientific_payload_sha256"],
        "training_authorized": False,
        "model_selection_authorized": False,
    }
    atomic_create_json(
        output_dir / "PINDER_ANTIBODY_SEQUENCE_CLUSTER_RECEIPT_V1.json", receipt
    )
    return result


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-json", type=Path, required=True)
    parser.add_argument("--inventory-manifest", type=Path, required=True)
    parser.add_argument("--sequence-manifest", type=Path, required=True)
    parser.add_argument("--metadata-parquet", type=Path, required=True)
    parser.add_argument("--antibody-fasta", type=Path, required=True)
    parser.add_argument("--antigen-fasta", type=Path, required=True)
    parser.add_argument("--antibody-edges-tsv", type=Path, required=True)
    parser.add_argument("--antigen-edges-tsv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_arguments())
