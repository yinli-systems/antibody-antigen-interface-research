#!/usr/bin/env python3
"""Diagnose strict no-discard S40/80 four-view component feasibility."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


VIEWS = ("vh", "vl", "concatenated_antibody", "antigen")


class DisjointSet:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}
        self.size = {value: 1 for value in self.parent}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        while parent != self.parent[parent]:
            parent = self.parent[parent]
        while value != parent:
            next_value = self.parent[value]
            self.parent[value] = parent
            value = next_value
        return parent

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if self.size[left_root] < self.size[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        self.size[left_root] += self.size[right_root]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sequence_digest(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def exact_pair_digest(antibody_digest: str, antigen_digest: str) -> str:
    return hashlib.sha256(
        f"{antibody_digest}\0{antigen_digest}".encode("ascii")
    ).hexdigest()


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def atomic_create(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_systems(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"INSTANCE", "pdb_id", "vh_seq", "vl_seq", "ag_seq"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("canonical manifest schema mismatch")
    if len(rows) != len({row["INSTANCE"] for row in rows}):
        raise ValueError("canonical manifest system ids are not unique")
    return rows


def _membership_maps(rows: Iterable[Mapping[str, str]]):
    maps: dict[str, dict[str, set[str]]] = {
        view: defaultdict(set) for view in VIEWS
    }
    exact_pairs: set[tuple[str, str]] = set()
    pdbs: set[str] = set()
    for row in rows:
        vh = row["vh_seq"]
        vl = row["vl_seq"]
        antigen = row["ag_seq"]
        antibody_digest = sequence_digest(vh + vl)
        antigen_digest = sequence_digest(antigen)
        maps["vh"][sequence_digest(vh)].add(antibody_digest)
        if vl:
            maps["vl"][sequence_digest(vl)].add(antibody_digest)
        maps["concatenated_antibody"][antibody_digest].add(antibody_digest)
        maps["antigen"][antigen_digest].add(antigen_digest)
        exact_pairs.add((antibody_digest, antigen_digest))
        pdbs.add(row["pdb_id"].lower())
    return maps, exact_pairs, pdbs


def union_relation(
    path: Path,
    membership: Mapping[str, set[str]],
    disjoint_set: DisjointSet,
    *,
    expected_sha256: str,
    expected_rows: int,
) -> int:
    if sha256_file(path) != expected_sha256:
        raise ValueError("relation SHA mismatch")
    row_count = 0
    with path.open("r", encoding="ascii") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 5:
                raise ValueError("relation row schema mismatch")
            left, right = fields[:2]
            if left not in membership or right not in membership:
                raise ValueError("relation references an unknown sequence digest")
            left_entities = membership[left]
            right_entities = membership[right]
            anchor = min(left_entities)
            for entity in (*left_entities, *right_entities):
                disjoint_set.union(anchor, entity)
            row_count += 1
    if row_count != expected_rows:
        raise ValueError("relation row count mismatch")
    return row_count


def subset_sum_feasibility(
    component_degrees: Iterable[int],
    minimum: int,
    maximum: int,
    *,
    fix_largest_to_train: bool,
) -> dict[str, Any]:
    degrees = sorted(component_degrees, reverse=True)
    if not degrees or any(degree < 1 for degree in degrees):
        raise ValueError("component degrees must be positive")
    candidates = degrees[1:] if fix_largest_to_train else degrees
    reachable = 1
    for degree in candidates:
        reachable |= reachable << degree
    feasible = [value for value in range(minimum, maximum + 1) if (reachable >> value) & 1]
    below = [value for value in range(minimum) if (reachable >> value) & 1]
    above = [
        value
        for value in range(maximum + 1, sum(candidates) + 1)
        if (reachable >> value) & 1
    ]
    return {
        "component_count": len(degrees),
        "largest_component_unique_pair_degree": degrees[0],
        "largest_component_fixed_to_train": fix_largest_to_train,
        "feasible_subset_sum_exists": bool(feasible),
        "feasible_subset_sum_count": len(feasible),
        "minimum_feasible_subset_sum": min(feasible) if feasible else None,
        "maximum_feasible_subset_sum": max(feasible) if feasible else None,
        "nearest_reachable_below_minimum": max(below) if below else None,
        "nearest_reachable_above_maximum": min(above) if above else None,
    }


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    relation_result = json.loads(arguments.relation_result.read_text(encoding="utf-8"))
    if sha256_file(arguments.relation_result) != arguments.relation_result_sha256:
        raise ValueError("relation result SHA mismatch")
    if relation_result.get("status") != "PASS_PINDER_TRAIN_FOUR_VIEW_S40_RELATIONS_MATERIALIZED":
        raise ValueError("four-view relation result did not pass")
    if relation_result.get("training_authorized") is not False:
        raise ValueError("relation result unexpectedly authorized training")
    if sha256_file(arguments.canonical_manifest) != arguments.canonical_manifest_sha256:
        raise ValueError("canonical manifest SHA mismatch")
    rows = load_systems(arguments.canonical_manifest)
    maps, exact_pairs, pdbs = _membership_maps(rows)
    antibodies = {antibody for antibody, _ in exact_pairs}
    antigens = {antigen for _, antigen in exact_pairs}
    antibody_dsu = DisjointSet(antibodies)
    antigen_dsu = DisjointSet(antigens)
    relation_paths = {
        "vh": arguments.vh_relation,
        "vl": arguments.vl_relation,
        "concatenated_antibody": arguments.concatenated_antibody_relation,
        "antigen": arguments.antigen_relation,
    }
    consumed: dict[str, int] = {}
    for view in VIEWS:
        contract = relation_result["relations"][view]
        consumed[view] = union_relation(
            relation_paths[view],
            maps[view],
            antigen_dsu if view == "antigen" else antibody_dsu,
            expected_sha256=contract["relation_sha256"],
            expected_rows=contract["canonical_row_count"],
        )

    antibody_roots = {entity: antibody_dsu.find(entity) for entity in antibodies}
    antigen_roots = {entity: antigen_dsu.find(entity) for entity in antigens}
    antibody_degrees = Counter(antibody_roots[antibody] for antibody, _ in exact_pairs)
    antigen_degrees = Counter(antigen_roots[antigen] for _, antigen in exact_pairs)
    minimum = math.ceil(0.15 * len(exact_pairs))
    maximum = math.floor(0.25 * len(exact_pairs))
    axes = {
        "antibody": subset_sum_feasibility(
            antibody_degrees.values(), minimum, maximum, fix_largest_to_train=True
        ),
        "antigen": subset_sum_feasibility(
            antigen_degrees.values(), minimum, maximum, fix_largest_to_train=False
        ),
    }
    infeasible = any(not axis["feasible_subset_sum_exists"] for axis in axes.values())
    payload: dict[str, Any] = {
        "format_version": "pinder_antibody_s40_component_diagnostic_v1",
        "status": (
            "FAIL_S40_STRICT_NO_DISCARD_SPLIT_INFEASIBLE"
            if infeasible
            else "PASS_S40_AXIS_SUBSET_FEASIBLE_NOT_SPLIT_AUTHORIZED"
        ),
        "authorized_for_training": False,
        "authorized_for_model_selection": False,
        "inputs": {
            "canonical_manifest_sha256": arguments.canonical_manifest_sha256,
            "relation_result_sha256": arguments.relation_result_sha256,
            "relation_rows_consumed": consumed,
        },
        "counts": {
            "systems": len(rows),
            "unique_exact_pairs": len(exact_pairs),
            "unique_pdbs": len(pdbs),
            "antibody_components": len(antibody_degrees),
            "antigen_components": len(antigen_degrees),
        },
        "validation_unique_pair_degree_bounds": {"minimum": minimum, "maximum": maximum},
        "axis_results": axes,
        "proof_boundary": (
            "This is a strict no-discard component diagnostic under the frozen "
            "MMseqs2 40/80 relations. It does not rule out a separately frozen "
            "vertex-separator protocol that discards complete bridge systems."
        ),
        "monotonicity": (
            "If an axis is already subset-sum infeasible under these qualifying "
            "edges, adding missed qualifying edges can only merge components and "
            "cannot create a previously unavailable subset of the current components."
        ),
        "data_firewall": {
            "source_pool": "PINDER_train_only",
            "structures_read": False,
            "task_labels_read": False,
            "predictions_read": False,
            "test_rows_read": False,
        },
        "required_next_action": (
            "If strict closure is infeasible, preregister a minimum-discard vertex "
            "separator and independently audit zero four-view crossings."
        ),
    }
    payload["scientific_payload_sha256"] = canonical_sha256(payload)
    atomic_create(arguments.output, payload)
    return payload


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-manifest", type=Path, required=True)
    parser.add_argument("--canonical-manifest-sha256", required=True)
    parser.add_argument("--relation-result", type=Path, required=True)
    parser.add_argument("--relation-result-sha256", required=True)
    for view in VIEWS:
        parser.add_argument(f"--{view.replace('_', '-')}-relation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_arguments())
    print(json.dumps(result, indent=2, sort_keys=True))
