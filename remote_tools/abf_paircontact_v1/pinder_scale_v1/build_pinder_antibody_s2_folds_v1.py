#!/usr/bin/env python3
"""Solve a PDB- and exact-pair-indivisible provisional antibody S2 split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import scipy
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix, lil_matrix


FORMAT_VERSION = "pinder_antibody_provisional_s2_fold_assignment_v1"
FOLD_COUNT = 3


class SplitError(RuntimeError):
    """Raised when a split input, optimization, or ledger invariant fails."""


class UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        smaller, larger = sorted((left_root, right_root))
        self.parent[larger] = smaller


@dataclass(frozen=True)
class SourceRow:
    system_id: str
    pdb_id: str
    topology: str
    antibody_digest: str
    antigen_digest: str


@dataclass(frozen=True)
class Component:
    component_id: str
    pair_ids: tuple[str, ...]
    pdb_ids: tuple[str, ...]
    system_ids: tuple[str, ...]
    antibody_clusters: tuple[int, ...]
    antigen_clusters: tuple[int, ...]


@dataclass(frozen=True)
class VariableIndex:
    antibody_count: int
    antigen_count: int
    component_count: int
    fold_count: int

    @property
    def antigen_offset(self) -> int:
        return self.antibody_count * self.fold_count

    @property
    def component_offset(self) -> int:
        return (self.antibody_count + self.antigen_count) * self.fold_count

    @property
    def count(self) -> int:
        return (
            self.antibody_count + self.antigen_count + self.component_count
        ) * self.fold_count

    def antibody(self, index: int, fold: int) -> int:
        return index * self.fold_count + fold

    def antigen(self, index: int, fold: int) -> int:
        return self.antigen_offset + index * self.fold_count + fold

    def component(self, index: int, fold: int) -> int:
        return self.component_offset + index * self.fold_count + fold


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def membership_sha256(values: Iterable[str]) -> str:
    payload = "".join(f"{value}\n" for value in sorted(set(values))).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def require_sha(path: Path, expected: str, label: str) -> dict[str, Any]:
    observed = sha256_file(path)
    if observed != expected:
        raise SplitError(f"{label} SHA mismatch: expected {expected}, observed {observed}")
    return {"path": str(path.resolve()), "sha256": observed, "size_bytes": path.stat().st_size}


def normalize_fraction(value: str) -> float:
    result = float(value)
    if result > 1.0:
        result /= 100.0
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise SplitError("similarity metric is not a finite fraction")
    return result


def read_sources(path: Path) -> list[SourceRow]:
    rows: list[SourceRow] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"INSTANCE", "pdb_id", "topology", "vh_seq", "vl_seq", "ag_seq"}
        if not required.issubset(reader.fieldnames or []):
            raise SplitError("canonical manifest schema mismatch")
        for raw in reader:
            antibody = raw["vh_seq"] + raw["vl_seq"]
            rows.append(
                SourceRow(
                    system_id=raw["INSTANCE"],
                    pdb_id=raw["pdb_id"],
                    topology=raw["topology"],
                    antibody_digest=sequence_sha256(antibody),
                    antigen_digest=sequence_sha256(raw["ag_seq"]),
                )
            )
    if len(rows) != 4_070 or len({row.system_id for row in rows}) != len(rows):
        raise SplitError("canonical source count or uniqueness mismatch")
    return sorted(rows, key=lambda row: row.system_id)


def load_clusters(
    relation_path: Path,
    digests: set[str],
    identity_threshold: float,
    coverage_threshold: float,
) -> tuple[dict[str, str], dict[str, Any]]:
    union_find = UnionFind(digests)
    total_rows = 0
    qualified_nonself: set[tuple[str, str]] = set()
    observed: set[str] = set()
    with relation_path.open("r", encoding="ascii") as handle:
        for line_number, line in enumerate(handle, start=1):
            total_rows += 1
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 5:
                raise SplitError(f"similarity row {line_number} does not have five fields")
            query, target = fields[:2]
            if query not in digests or target not in digests:
                raise SplitError("similarity relation references an unknown digest")
            observed.update((query, target))
            identity, qcov, tcov = map(normalize_fraction, fields[2:])
            if (
                identity + 1e-12 >= identity_threshold
                and qcov + 1e-12 >= coverage_threshold
                and tcov + 1e-12 >= coverage_threshold
                and query != target
            ):
                qualified_nonself.add(tuple(sorted((query, target))))
    if observed != digests:
        raise SplitError("similarity relation does not cover every authorized digest")
    for left, right in sorted(qualified_nonself):
        union_find.union(left, right)
    roots = {digest: union_find.find(digest) for digest in sorted(digests)}
    sizes = sorted(Counter(roots.values()).values(), reverse=True)
    return roots, {
        "entity_count": len(digests),
        "relation_row_count": total_rows,
        "qualified_unique_nonself_edge_count": len(qualified_nonself),
        "cluster_count": len(sizes),
        "singleton_cluster_count": sum(size == 1 for size in sizes),
        "largest_cluster_entity_count": sizes[0],
        "largest_cluster_entity_fraction": sizes[0] / len(digests),
        "top_cluster_entity_counts": sizes[:20],
    }


def exact_pair_id(antibody_digest: str, antigen_digest: str) -> str:
    return sequence_sha256(f"{antibody_digest}\0{antigen_digest}")


def interaction_components(
    rows: Sequence[SourceRow],
    antibody_cluster_by_digest: Mapping[str, str],
    antigen_cluster_by_digest: Mapping[str, str],
) -> tuple[list[Component], dict[str, tuple[str, str]], dict[str, str]]:
    pair_by_id = {
        exact_pair_id(row.antibody_digest, row.antigen_digest): (
            row.antibody_digest,
            row.antigen_digest,
        )
        for row in rows
    }
    nodes = {f"pair:{pair_id}" for pair_id in pair_by_id} | {
        f"pdb:{row.pdb_id}" for row in rows
    }
    union_find = UnionFind(nodes)
    for row in rows:
        pair_id = exact_pair_id(row.antibody_digest, row.antigen_digest)
        union_find.union(f"pair:{pair_id}", f"pdb:{row.pdb_id}")
    roots = {node: union_find.find(node) for node in sorted(nodes)}
    component_pairs: defaultdict[str, set[str]] = defaultdict(set)
    component_pdbs: defaultdict[str, set[str]] = defaultdict(set)
    component_systems: defaultdict[str, set[str]] = defaultdict(set)
    for node, root in roots.items():
        kind, value = node.split(":", 1)
        (component_pairs if kind == "pair" else component_pdbs)[root].add(value)
    component_by_system: dict[str, str] = {}
    for row in rows:
        pair_id = exact_pair_id(row.antibody_digest, row.antigen_digest)
        root = roots[f"pair:{pair_id}"]
        component_systems[root].add(row.system_id)
        component_by_system[row.system_id] = root

    antibody_clusters = tuple(sorted(set(antibody_cluster_by_digest.values())))
    antigen_clusters = tuple(sorted(set(antigen_cluster_by_digest.values())))
    antibody_index = {cluster: index for index, cluster in enumerate(antibody_clusters)}
    antigen_index = {cluster: index for index, cluster in enumerate(antigen_clusters)}
    components: list[Component] = []
    for root in sorted(component_pairs):
        pair_ids = tuple(sorted(component_pairs[root]))
        components.append(
            Component(
                component_id=hashlib.sha256(root.encode("ascii")).hexdigest(),
                pair_ids=pair_ids,
                pdb_ids=tuple(sorted(component_pdbs[root])),
                system_ids=tuple(sorted(component_systems[root])),
                antibody_clusters=tuple(
                    sorted(
                        {
                            antibody_index[antibody_cluster_by_digest[pair_by_id[pair][0]]]
                            for pair in pair_ids
                        }
                    )
                ),
                antigen_clusters=tuple(
                    sorted(
                        {
                            antigen_index[antigen_cluster_by_digest[pair_by_id[pair][1]]]
                            for pair in pair_ids
                        }
                    )
                ),
            )
        )
    return components, pair_by_id, component_by_system


def solve(
    rows: Sequence[SourceRow],
    antibody_roots: Mapping[str, str],
    antigen_roots: Mapping[str, str],
    components: Sequence[Component],
    tolerance: float,
    time_limit: float,
) -> tuple[np.ndarray, dict[str, Any], tuple[str, ...], tuple[str, ...]]:
    antibody_clusters = tuple(sorted(set(antibody_roots.values())))
    antigen_clusters = tuple(sorted(set(antigen_roots.values())))
    antibody_index = {cluster: index for index, cluster in enumerate(antibody_clusters)}
    antigen_index = {cluster: index for index, cluster in enumerate(antigen_clusters)}
    exact_pairs = {(row.antibody_digest, row.antigen_digest) for row in rows}
    antibody_degree = Counter(antibody_index[antibody_roots[a]] for a, _ in exact_pairs)
    antigen_degree = Counter(antigen_index[antigen_roots[g]] for _, g in exact_pairs)
    index = VariableIndex(
        len(antibody_clusters), len(antigen_clusters), len(components), FOLD_COUNT
    )
    constraint_rows: list[list[tuple[int, float]]] = []
    lower: list[float] = []
    upper: list[float] = []

    def add(values: list[tuple[int, float]], low: float, high: float) -> None:
        constraint_rows.append(values); lower.append(low); upper.append(high)

    for antibody in range(index.antibody_count):
        add([(index.antibody(antibody, fold), 1.0) for fold in range(FOLD_COUNT)], 1.0, 1.0)
    for antigen in range(index.antigen_count):
        add([(index.antigen(antigen, fold), 1.0) for fold in range(FOLD_COUNT)], 1.0, 1.0)
    for component_number, component in enumerate(components):
        for fold in range(FOLD_COUNT):
            retained = index.component(component_number, fold)
            for antibody in component.antibody_clusters:
                add([(retained, 1.0), (index.antibody(antibody, fold), -1.0)], -np.inf, 0.0)
            for antigen in component.antigen_clusters:
                add([(retained, 1.0), (index.antigen(antigen, fold), -1.0)], -np.inf, 0.0)
    target = len(exact_pairs) / FOLD_COUNT
    for fold in range(FOLD_COUNT):
        add(
            [(index.antibody(cluster, fold), float(antibody_degree[cluster])) for cluster in range(index.antibody_count)],
            (1.0 - tolerance) * target,
            (1.0 + tolerance) * target,
        )
        add(
            [(index.antigen(cluster, fold), float(antigen_degree[cluster])) for cluster in range(index.antigen_count)],
            (1.0 - tolerance) * target,
            (1.0 + tolerance) * target,
        )
    matrix = lil_matrix((len(constraint_rows), index.count), dtype=float)
    for row_number, values in enumerate(constraint_rows):
        for column, coefficient in values:
            matrix[row_number, column] = coefficient
    constraint = LinearConstraint(
        csr_matrix(matrix), np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    )
    objective = np.zeros(index.count, dtype=float)
    for component_number, component in enumerate(components):
        for fold in range(FOLD_COUNT):
            objective[index.component(component_number, fold)] = -len(component.pair_ids)
    variable_lower = np.zeros(index.count, dtype=float)
    variable_upper = np.ones(index.count, dtype=float)
    largest_antibody = min(
        range(index.antibody_count),
        key=lambda cluster: (-antibody_degree[cluster], antibody_clusters[cluster]),
    )
    variable_lower[index.antibody(largest_antibody, 0)] = 1.0
    variable_upper[index.antibody(largest_antibody, 0)] = 1.0
    result = milp(
        objective,
        integrality=np.ones(index.count, dtype=int),
        bounds=Bounds(variable_lower, variable_upper),
        constraints=constraint,
        options={"time_limit": time_limit, "mip_rel_gap": 0.0, "presolve": True},
    )
    if result.status != 0 or result.x is None or abs(float(result.mip_gap)) > 1e-12:
        raise SplitError(
            f"MILP did not prove an optimum: status={result.status}, gap={getattr(result, 'mip_gap', None)}, message={result.message}"
        )
    return np.asarray(result.x), {
        "status": int(result.status),
        "message": str(result.message),
        "mip_gap": float(result.mip_gap),
        "mip_node_count": int(result.mip_node_count),
        "optimal_retained_unique_exact_pair_count": int(round(-float(result.fun))),
        "variable_count": index.count,
        "constraint_count": len(constraint_rows),
    }, antibody_clusters, antigen_clusters


def assigned_fold(values: Sequence[float]) -> int:
    rounded = [int(round(float(value))) for value in values]
    if sum(rounded) != 1:
        raise SplitError("axis assignment is not one-hot")
    return rounded.index(1)


def atomic_create_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded); handle.flush(); os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    protocol = json.loads(arguments.protocol_json.read_text(encoding="utf-8"))
    if protocol.get("format_version") != "pinder_antibody_s2_split_protocol_v1":
        raise SplitError("split protocol format mismatch")
    inputs = {
        "canonical_manifest": require_sha(arguments.canonical_manifest, protocol["source_sha256"]["canonical_manifest"], "canonical_manifest"),
        "s2_asset_result": require_sha(arguments.s2_asset_result, protocol["source_sha256"]["s2_asset_result"], "s2_asset_result"),
        "antibody_relation": require_sha(arguments.antibody_relation, protocol["source_sha256"]["antibody_relation"], "antibody_relation"),
        "antibody_relation_receipt": require_sha(arguments.antibody_relation_receipt, protocol["source_sha256"]["antibody_relation_receipt"], "antibody_relation_receipt"),
        "antigen_relation": require_sha(arguments.antigen_relation, protocol["source_sha256"]["antigen_relation"], "antigen_relation"),
        "antigen_relation_receipt": require_sha(arguments.antigen_relation_receipt, protocol["source_sha256"]["antigen_relation_receipt"], "antigen_relation_receipt"),
        "protocol": {"path": str(arguments.protocol_json.resolve()), "sha256": sha256_file(arguments.protocol_json), "size_bytes": arguments.protocol_json.stat().st_size},
        "builder_source": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__)), "size_bytes": Path(__file__).stat().st_size},
    }
    rows = read_sources(arguments.canonical_manifest)
    expected = protocol["expected_input_counts"]
    observed = {
        "canonical_systems": len(rows),
        "unique_concatenated_antibodies": len({row.antibody_digest for row in rows}),
        "unique_antigens": len({row.antigen_digest for row in rows}),
        "unique_exact_pairs": len({(row.antibody_digest, row.antigen_digest) for row in rows}),
        "unique_pdbs": len({row.pdb_id for row in rows}),
    }
    if observed != expected:
        raise SplitError(f"frozen input counts differ: expected {expected}, observed {observed}")
    identity = protocol["similarity"]["aligned_region_identity_minimum"]
    coverage = protocol["similarity"]["minimum_reciprocal_coverage"]
    antibody_roots, antibody_summary = load_clusters(arguments.antibody_relation, {row.antibody_digest for row in rows}, identity, coverage)
    antigen_roots, antigen_summary = load_clusters(arguments.antigen_relation, {row.antigen_digest for row in rows}, identity, coverage)
    components, pair_by_id, component_by_system = interaction_components(rows, antibody_roots, antigen_roots)
    solution, solver_result, antibody_clusters, antigen_clusters = solve(
        rows,
        antibody_roots,
        antigen_roots,
        components,
        protocol["balance"]["axis_unique_pair_degree_tolerance"],
        protocol["determinism"]["time_limit_seconds"],
    )
    index = VariableIndex(len(antibody_clusters), len(antigen_clusters), len(components), FOLD_COUNT)
    antibody_fold_by_root = {
        root: assigned_fold([solution[index.antibody(number, fold)] for fold in range(FOLD_COUNT)])
        for number, root in enumerate(antibody_clusters)
    }
    antigen_fold_by_root = {
        root: assigned_fold([solution[index.antigen(number, fold)] for fold in range(FOLD_COUNT)])
        for number, root in enumerate(antigen_clusters)
    }
    retained_component_fold: dict[str, int] = {}
    for number, component in enumerate(components):
        active = [fold for fold in range(FOLD_COUNT) if solution[index.component(number, fold)] > 0.5]
        if len(active) > 1:
            raise SplitError("interaction component retained in multiple folds")
        if active:
            retained_component_fold[component.component_id] = active[0]
    component_by_root = {
        root: component for root, component in zip(sorted({component_by_system[row.system_id] for row in rows}), components)
    }
    # interaction_components emits components in sorted root order.
    root_to_id = {root: component_by_root[root].component_id for root in component_by_root}
    retained_by_fold: defaultdict[int, list[SourceRow]] = defaultdict(list)
    discarded: list[SourceRow] = []
    for row in rows:
        component_id = root_to_id[component_by_system[row.system_id]]
        fold = retained_component_fold.get(component_id)
        (discarded if fold is None else retained_by_fold[fold]).append(row)
    retained_folds: list[dict[str, Any]] = []
    for fold in range(FOLD_COUNT):
        fold_rows = sorted(retained_by_fold[fold], key=lambda row: row.system_id)
        pair_ids = sorted({exact_pair_id(row.antibody_digest, row.antigen_digest) for row in fold_rows})
        pdb_ids = sorted({row.pdb_id for row in fold_rows})
        system_ids = [row.system_id for row in fold_rows]
        retained_folds.append({
            "fold_index": fold,
            "retained_system_count": len(system_ids),
            "retained_unique_exact_pair_count": len(pair_ids),
            "retained_unique_pdb_count": len(pdb_ids),
            "retained_system_ids": system_ids,
            "retained_system_membership_sha256": membership_sha256(system_ids),
            "retained_exact_pair_ids": pair_ids,
            "retained_exact_pair_membership_sha256": membership_sha256(pair_ids),
            "retained_pdb_ids": pdb_ids,
            "retained_pdb_membership_sha256": membership_sha256(pdb_ids),
        })
    discarded_ids = sorted(row.system_id for row in discarded)
    discarded_pairs = sorted({exact_pair_id(row.antibody_digest, row.antigen_digest) for row in discarded})
    discarded_pdbs = sorted({row.pdb_id for row in discarded})
    retained_ids = {value for fold in retained_folds for value in fold["retained_system_ids"]}
    if retained_ids & set(discarded_ids) or retained_ids | set(discarded_ids) != {row.system_id for row in rows}:
        raise SplitError("retained/discarded system ledger is not disjoint and exhaustive")
    retained_pair_sets = [set(fold["retained_exact_pair_ids"]) for fold in retained_folds]
    retained_pdb_sets = [set(fold["retained_pdb_ids"]) for fold in retained_folds]
    if any(retained_pair_sets[i] & retained_pair_sets[j] for i in range(FOLD_COUNT) for j in range(i + 1, FOLD_COUNT)):
        raise SplitError("an exact pair spans retained folds")
    if any(retained_pdb_sets[i] & retained_pdb_sets[j] for i in range(FOLD_COUNT) for j in range(i + 1, FOLD_COUNT)):
        raise SplitError("a PDB spans retained folds")
    cluster_assignments = {
        "antibody": [
            {"sequence_sha256": digest, "cluster_representative_sequence_sha256": root, "fold_index": antibody_fold_by_root[root]}
            for digest, root in sorted(antibody_roots.items())
        ],
        "antigen": [
            {"sequence_sha256": digest, "cluster_representative_sequence_sha256": root, "fold_index": antigen_fold_by_root[root]}
            for digest, root in sorted(antigen_roots.items())
        ],
    }
    payload: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "status": "PASS_PROVISIONAL_PDB_AND_EXACT_PAIR_INDIVISIBLE_S2_80_ASSIGNMENT",
        "approval_status": "UNAPPROVED_PENDING_EXACT_FOUR_VIEW_NEEDLEMAN_WUNSCH_AUDIT",
        "inputs": inputs,
        "protocol": protocol,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "platform": platform.platform()},
        "input_counts": observed,
        "similarity_summary": {"antibody": antibody_summary, "antigen": antigen_summary},
        "interaction_component_summary": {
            "component_count": len(components),
            "largest_component_unique_exact_pair_count": max(len(component.pair_ids) for component in components),
            "largest_component_pdb_count": max(len(component.pdb_ids) for component in components),
            "largest_component_system_count": max(len(component.system_ids) for component in components),
        },
        "solver_result": solver_result,
        "cluster_assignments": cluster_assignments,
        "retained_folds": retained_folds,
        "discarded_ledger": {
            "discard_policy": "discard_complete_exact_pair_and_pdb_connected_component",
            "discarded_system_count": len(discarded_ids),
            "discarded_unique_exact_pair_count": len(discarded_pairs),
            "discarded_unique_pdb_count": len(discarded_pdbs),
            "discarded_system_ids": discarded_ids,
            "discarded_system_membership_sha256": membership_sha256(discarded_ids),
            "discarded_exact_pair_ids": discarded_pairs,
            "discarded_exact_pair_membership_sha256": membership_sha256(discarded_pairs),
            "discarded_pdb_ids": discarded_pdbs,
            "discarded_pdb_membership_sha256": membership_sha256(discarded_pdbs),
        },
        "checks": {
            "all_input_systems_accounted_once": True,
            "exact_pairs_indivisible_across_retained_folds": True,
            "pdbs_indivisible_across_retained_folds": True,
            "test_rows_read": False,
            "validation_rows_read": False,
        },
        "authorized_for_training": False,
        "authorized_for_model_selection": False,
        "required_next_gate": "independent exact four-view Needleman-Wunsch zero-crossing audit at 80% identity and 80% reciprocal coverage",
        "claim_boundary": protocol["claim_boundary"],
    }
    payload["scientific_payload_sha256"] = canonical_sha256(payload)
    atomic_create_json(arguments.output, payload)
    return payload


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-json", type=Path, required=True)
    parser.add_argument("--canonical-manifest", type=Path, required=True)
    parser.add_argument("--s2-asset-result", type=Path, required=True)
    parser.add_argument("--antibody-relation", type=Path, required=True)
    parser.add_argument("--antibody-relation-receipt", type=Path, required=True)
    parser.add_argument("--antigen-relation", type=Path, required=True)
    parser.add_argument("--antigen-relation-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_arguments())
