#!/usr/bin/env python3
"""Construct a strict no-discard antibody-CDR/antigen S40 split when feasible."""

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
        if (self.size[left_root], left_root) < (self.size[right_root], right_root):
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        self.size[left_root] += self.size[right_root]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def pair_digest(antibody: str, antigen: str) -> str:
    return digest(f"{antibody}\0{antigen}")


def membership_sha256(values: Iterable[str]) -> str:
    return hashlib.sha256(
        "".join(f"{value}\n" for value in sorted(set(values))).encode("ascii")
    ).hexdigest()


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    ).hexdigest()


def atomic_create(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def read_rows(canonical_path: Path, cdr_path: Path) -> list[dict[str, str]]:
    with canonical_path.open("r", encoding="utf-8", newline="") as handle:
        canonical = {row["INSTANCE"]: row for row in csv.DictReader(handle)}
    with cdr_path.open("r", encoding="utf-8", newline="") as handle:
        cdr = {row["INSTANCE"]: row for row in csv.DictReader(handle)}
    if not canonical or set(canonical) != set(cdr):
        raise ValueError("canonical/CDR instance identities differ")
    rows = []
    for instance in sorted(canonical):
        source, cdr_row = canonical[instance], cdr[instance]
        combined = cdr_row["combined_cdr_sequence"]
        antibody = digest(combined)
        if antibody != cdr_row["combined_cdr_sequence_sha256"]:
            raise ValueError("CDR sequence digest differs")
        antigen = digest(source["ag_seq"])
        rows.append(
            {
                "system_id": instance,
                "pdb_id": source["pdb_id"].lower(),
                "antibody": antibody,
                "antigen": antigen,
                "pair_id": pair_digest(antibody, antigen),
            }
        )
    return rows


def ingest_relation(path: Path, known: set[str], dsu: DisjointSet) -> int:
    count = 0
    with path.open("r", encoding="ascii") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 5:
                raise ValueError("canonical relation row does not have five fields")
            left, right = fields[:2]
            if left not in known or right not in known:
                raise ValueError("relation references unknown sequence digest")
            identity, qcov, tcov = map(float, fields[2:])
            if identity + 1e-12 < 0.4 or min(qcov, tcov) + 1e-12 < 0.8:
                raise ValueError("relation row is below frozen threshold")
            dsu.union(left, right)
            count += 1
    return count


def choose_validation_components(
    degrees: list[tuple[str, int]], minimum: int, maximum: int, target: int
) -> set[str] | None:
    """Deterministic subset-sum witness nearest the frozen 20 percent target."""
    predecessors: dict[int, tuple[int, str] | None] = {0: None}
    for component, degree in sorted(degrees, key=lambda value: (value[1], value[0])):
        updates: dict[int, tuple[int, str]] = {}
        for total in sorted(predecessors, reverse=True):
            candidate = total + degree
            if candidate <= maximum and candidate not in predecessors and candidate not in updates:
                updates[candidate] = (total, component)
        predecessors.update(updates)
    feasible = [value for value in predecessors if minimum <= value <= maximum]
    if not feasible:
        return None
    selected_total = min(feasible, key=lambda value: (abs(value - target), -value))
    selected: set[str] = set()
    current = selected_total
    while current:
        predecessor = predecessors[current]
        if predecessor is None:
            raise RuntimeError("subset witness ended early")
        previous, component = predecessor
        selected.add(component)
        current = previous
    return selected


def ledger(bucket: str, rows: list[dict[str, str]]) -> dict[str, Any]:
    systems = sorted(row["system_id"] for row in rows)
    pairs = sorted({row["pair_id"] for row in rows})
    pdbs = sorted({row["pdb_id"] for row in rows})
    return {
        "bucket": bucket,
        "system_ids": systems,
        "system_count": len(systems),
        "system_membership_sha256": membership_sha256(systems),
        "exact_pair_ids": pairs,
        "unique_exact_pair_count": len(pairs),
        "exact_pair_membership_sha256": membership_sha256(pairs),
        "pdb_ids": pdbs,
        "unique_pdb_count": len(pdbs),
        "pdb_membership_sha256": membership_sha256(pdbs),
    }


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    protocol = json.loads(arguments.protocol.read_text(encoding="utf-8"))
    if protocol.get("format_version") != "pinder_antibody_cdr_antigen_s40_protocol_v1":
        raise ValueError("protocol format differs")
    source = protocol["source"]
    if sha256_file(arguments.canonical_manifest) != source["canonical_manifest_sha256"]:
        raise ValueError("canonical manifest SHA mismatch")
    if sha256_file(arguments.s40_relation_result) != source["s40_antigen_relation_result_sha256"]:
        raise ValueError("S40 relation result SHA mismatch")
    cdr_result = json.loads(arguments.cdr_result.read_text(encoding="utf-8"))
    if cdr_result.get("status") != "PASS_PINDER_TRAIN_FIXED_IMGT_CDR_VIEWS":
        raise ValueError("CDR materialization did not pass")
    if sha256_file(arguments.cdr_csv) != cdr_result["artifacts"]["csv_sha256"]:
        raise ValueError("CDR CSV SHA mismatch")
    cdr_receipt = json.loads(arguments.cdr_relation_receipt.read_text(encoding="utf-8"))
    antigen_result = json.loads(arguments.s40_relation_result.read_text(encoding="utf-8"))
    if cdr_receipt.get("role") != "antibody_cdr" or cdr_receipt.get("thresholds") != {
        "identity": 0.4,
        "reciprocal_coverage": 0.8,
    }:
        raise ValueError("CDR relation policy differs")
    if sha256_file(arguments.cdr_relation) != cdr_receipt["canonical_relation"]["sha256"]:
        raise ValueError("CDR relation SHA mismatch")
    antigen_contract = antigen_result["relations"]["antigen"]
    if sha256_file(arguments.antigen_relation) != antigen_contract["relation_sha256"]:
        raise ValueError("antigen relation SHA mismatch")

    rows = read_rows(arguments.canonical_manifest, arguments.cdr_csv)
    antibodies = {row["antibody"] for row in rows}
    antigens = {row["antigen"] for row in rows}
    antibody_dsu, antigen_dsu = DisjointSet(antibodies), DisjointSet(antigens)
    relation_counts = {
        "antibody_cdr": ingest_relation(arguments.cdr_relation, antibodies, antibody_dsu),
        "antigen": ingest_relation(arguments.antigen_relation, antigens, antigen_dsu),
    }

    system_dsu = DisjointSet(row["system_id"] for row in rows)
    by_antibody_component: defaultdict[str, list[str]] = defaultdict(list)
    by_antigen_component: defaultdict[str, list[str]] = defaultdict(list)
    by_pair: defaultdict[str, list[str]] = defaultdict(list)
    by_pdb: defaultdict[str, list[str]] = defaultdict(list)
    for row in rows:
        by_antibody_component[antibody_dsu.find(row["antibody"])].append(row["system_id"])
        by_antigen_component[antigen_dsu.find(row["antigen"])].append(row["system_id"])
        by_pair[row["pair_id"]].append(row["system_id"])
        by_pdb[row["pdb_id"]].append(row["system_id"])
    for groups in (by_antibody_component, by_antigen_component, by_pair, by_pdb):
        for members in groups.values():
            anchor = members[0]
            for member in members[1:]:
                system_dsu.union(anchor, member)

    by_system_component: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_system_component[system_dsu.find(row["system_id"])].append(row)
    component_degrees = [
        (component, len({row["pair_id"] for row in component_rows}))
        for component, component_rows in by_system_component.items()
    ]
    unique_pairs = len({row["pair_id"] for row in rows})
    minimum = math.ceil(
        protocol["retention_policy"]["validation_fraction_of_retained_unique_exact_pairs"]["minimum"]
        * unique_pairs
    )
    maximum = math.floor(
        protocol["retention_policy"]["validation_fraction_of_retained_unique_exact_pairs"]["maximum"]
        * unique_pairs
    )
    selected = choose_validation_components(
        component_degrees, minimum, maximum, round(0.2 * unique_pairs)
    )
    feasible = selected is not None
    bucket_rows: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    if selected is not None:
        for component, component_rows in by_system_component.items():
            bucket_rows["validation" if component in selected else "train"].extend(component_rows)
    largest = max(component_degrees, key=lambda value: (value[1], value[0]))
    payload: dict[str, Any] = {
        "format_version": "pinder_antibody_cdr_antigen_s40_diagnostic_v1",
        "status": (
            "PASS_PROVISIONAL_CDR_ANTIGEN_S40_NO_DISCARD_ASSIGNMENT"
            if feasible
            else "FAIL_CDR_ANTIGEN_S40_NO_DISCARD_SPLIT_INFEASIBLE"
        ),
        "approval_status": (
            "UNAPPROVED_PENDING_EXACT_ASYMMETRIC_ZERO_CROSSING_AUDIT"
            if feasible
            else "NOT_APPROVED_REQUIRES_FROZEN_SEPARATOR"
        ),
        "authorized_for_training": False,
        "authorized_for_model_selection": False,
        "inputs": {
            "protocol_sha256": sha256_file(arguments.protocol),
            "canonical_manifest_sha256": source["canonical_manifest_sha256"],
            "cdr_result_sha256": sha256_file(arguments.cdr_result),
            "cdr_relation_receipt_sha256": sha256_file(arguments.cdr_relation_receipt),
            "s40_relation_result_sha256": source["s40_antigen_relation_result_sha256"],
            "relation_rows_consumed": relation_counts,
        },
        "counts": {
            "systems": len(rows),
            "unique_exact_pairs": unique_pairs,
            "unique_pdbs": len({row["pdb_id"] for row in rows}),
            "antibody_cdr_components": len({antibody_dsu.find(value) for value in antibodies}),
            "antigen_components": len({antigen_dsu.find(value) for value in antigens}),
            "joint_indivisible_components": len(by_system_component),
            "largest_joint_component_unique_exact_pairs": largest[1],
        },
        "validation_unique_pair_bounds": {"minimum": minimum, "maximum": maximum},
        "data_firewall": protocol["authorization"],
        "required_next_gate": (
            "independent exact CDR-S40 and antigen-S40 train-validation zero-crossing audit"
            if feasible
            else "run the preregistered minimum-discard separator with retention at least 80 percent"
        ),
        "claim_boundary": protocol["claim_boundary"],
    }
    if feasible:
        payload["bucket_ledgers"] = {
            "train": ledger("train", bucket_rows["train"]),
            "validation": ledger("validation", bucket_rows["validation"]),
            "discarded": ledger("discarded", []),
        }
    payload["scientific_payload_sha256"] = canonical_sha256(payload)
    atomic_create(arguments.output, payload)
    return payload


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--canonical-manifest", type=Path, required=True)
    parser.add_argument("--cdr-csv", type=Path, required=True)
    parser.add_argument("--cdr-result", type=Path, required=True)
    parser.add_argument("--cdr-relation", type=Path, required=True)
    parser.add_argument("--cdr-relation-receipt", type=Path, required=True)
    parser.add_argument("--antigen-relation", type=Path, required=True)
    parser.add_argument("--s40-relation-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_arguments())
    print(json.dumps(result, indent=2, sort_keys=True))
