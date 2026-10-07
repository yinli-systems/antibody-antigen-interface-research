#!/usr/bin/env python3
"""Build one provisional four-view, two-axis train/validation S2 split.

This replacement is used only after the frozen balanced three-fold design has
been proven infeasible.  It does not relax the exact 80/80 sequence rule.  The
largest known antibody component is deterministically kept on the train axis;
the validation axis is constrained to 15--25% of original exact-pair degree on
both antibody and antigen sides.  PDB/exact-pair components incompatible with
the selected axis assignments are discarded whole.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import scipy
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix, lil_matrix

import audit_pinder_antibody_s2_exact_four_view_v1 as audit
import build_pinder_antibody_s2_folds_v1 as base
import prove_pinder_antibody_s2_three_fold_infeasibility_v1 as proof


FORMAT_VERSION = "pinder_antibody_s2_train_validation_assignment_v1"
PROTOCOL_FORMAT_VERSION = "pinder_antibody_s2_train_validation_protocol_v1"
BUCKETS = ("train", "validation")


class SplitError(RuntimeError):
    """Raised when an input, solver result, or ledger invariant is invalid."""


@dataclass(frozen=True)
class VariableIndex:
    antibody_count: int
    antigen_count: int
    component_count: int

    @property
    def antigen_offset(self) -> int:
        return self.antibody_count

    @property
    def train_component_offset(self) -> int:
        return self.antibody_count + self.antigen_count

    @property
    def validation_component_offset(self) -> int:
        return self.train_component_offset + self.component_count

    @property
    def count(self) -> int:
        return self.validation_component_offset + self.component_count

    def antibody_validation(self, index: int) -> int:
        return index

    def antigen_validation(self, index: int) -> int:
        return self.antigen_offset + index

    def component_train(self, index: int) -> int:
        return self.train_component_offset + index

    def component_validation(self, index: int) -> int:
        return self.validation_component_offset + index


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = audit.strict_json(path)
    if protocol.get("format_version") != PROTOCOL_FORMAT_VERSION:
        raise SplitError("protocol format mismatch")
    if protocol.get("identity_threshold") != 0.8:
        raise SplitError("identity threshold must remain 0.8")
    if protocol.get("reciprocal_coverage_threshold") != 0.8:
        raise SplitError("coverage threshold must remain 0.8")
    if protocol.get("bucket_names") != list(BUCKETS):
        raise SplitError("protocol bucket names mismatch")
    balance = protocol.get("validation_axis_degree_fraction")
    if balance != {"minimum": 0.15, "maximum": 0.25}:
        raise SplitError("validation axis-degree interval changed")
    if protocol.get("largest_antibody_component_bucket") != "train":
        raise SplitError("largest antibody component must remain in train")
    if protocol.get("threshold_relaxation_after_results") is not False:
        raise SplitError("post-result threshold relaxation is forbidden")
    return protocol


def _component_roots(
    rows: Sequence[audit.SourceRow],
    parent_assignment: Mapping[str, Any],
    edges_by_view: Mapping[str, set[tuple[str, str]]],
) -> tuple[dict[str, str], dict[str, str], set[tuple[str, str]], dict[str, Any]]:
    parent_antibody, parent_antigen = proof.load_parent_cluster_roots(
        parent_assignment,
        {row.antibody_digest for row in rows},
        {row.antigen_digest for row in rows},
    )
    return proof.build_augmented_components(
        rows, parent_antibody, parent_antigen, edges_by_view
    )


def _base_rows(rows: Sequence[audit.SourceRow]) -> list[base.SourceRow]:
    return [
        base.SourceRow(
            system_id=row.system_id,
            pdb_id=row.pdb_id,
            topology="not_used_by_split_solver",
            antibody_digest=row.antibody_digest,
            antigen_digest=row.antigen_digest,
        )
        for row in rows
    ]


def solve_binary(
    rows: Sequence[base.SourceRow],
    antibody_roots: Mapping[str, str],
    antigen_roots: Mapping[str, str],
    components: Sequence[base.Component],
    minimum_validation_fraction: float,
    maximum_validation_fraction: float,
    time_limit_seconds: float,
) -> tuple[np.ndarray, dict[str, Any], tuple[str, ...], tuple[str, ...]]:
    antibody_clusters = tuple(sorted(set(antibody_roots.values())))
    antigen_clusters = tuple(sorted(set(antigen_roots.values())))
    antibody_index = {root: index for index, root in enumerate(antibody_clusters)}
    antigen_index = {root: index for index, root in enumerate(antigen_clusters)}
    exact_pairs = {(row.antibody_digest, row.antigen_digest) for row in rows}
    antibody_degree = Counter(
        antibody_index[antibody_roots[antibody]] for antibody, _ in exact_pairs
    )
    antigen_degree = Counter(
        antigen_index[antigen_roots[antigen]] for _, antigen in exact_pairs
    )
    index = VariableIndex(
        antibody_count=len(antibody_clusters),
        antigen_count=len(antigen_clusters),
        component_count=len(components),
    )

    constraint_rows: list[list[tuple[int, float]]] = []
    lower: list[float] = []
    upper: list[float] = []

    def add(values: list[tuple[int, float]], low: float, high: float) -> None:
        constraint_rows.append(values)
        lower.append(low)
        upper.append(high)

    for component_number, component in enumerate(components):
        train = index.component_train(component_number)
        validation = index.component_validation(component_number)
        add([(train, 1.0), (validation, 1.0)], -np.inf, 1.0)
        for antibody in component.antibody_clusters:
            axis = index.antibody_validation(antibody)
            add([(validation, 1.0), (axis, -1.0)], -np.inf, 0.0)
            add([(train, 1.0), (axis, 1.0)], -np.inf, 1.0)
        for antigen in component.antigen_clusters:
            axis = index.antigen_validation(antigen)
            add([(validation, 1.0), (axis, -1.0)], -np.inf, 0.0)
            add([(train, 1.0), (axis, 1.0)], -np.inf, 1.0)

    total = len(exact_pairs)
    minimum = math.ceil(minimum_validation_fraction * total)
    maximum = math.floor(maximum_validation_fraction * total)
    add(
        [
            (index.antibody_validation(cluster), float(antibody_degree[cluster]))
            for cluster in range(index.antibody_count)
        ],
        minimum,
        maximum,
    )
    add(
        [
            (index.antigen_validation(cluster), float(antigen_degree[cluster]))
            for cluster in range(index.antigen_count)
        ],
        minimum,
        maximum,
    )

    matrix = lil_matrix((len(constraint_rows), index.count), dtype=float)
    for row_number, values in enumerate(constraint_rows):
        for column, coefficient in values:
            matrix[row_number, column] = coefficient
    objective = np.zeros(index.count, dtype=float)
    for component_number, component in enumerate(components):
        weight = float(len(component.pair_ids))
        objective[index.component_train(component_number)] = -weight
        objective[index.component_validation(component_number)] = -weight

    variable_lower = np.zeros(index.count, dtype=float)
    variable_upper = np.ones(index.count, dtype=float)
    largest_antibody = min(
        range(index.antibody_count),
        key=lambda cluster: (-antibody_degree[cluster], antibody_clusters[cluster]),
    )
    variable_upper[index.antibody_validation(largest_antibody)] = 0.0
    result = milp(
        objective,
        integrality=np.ones(index.count, dtype=int),
        bounds=Bounds(variable_lower, variable_upper),
        constraints=LinearConstraint(
            csr_matrix(matrix),
            np.asarray(lower, dtype=float),
            np.asarray(upper, dtype=float),
        ),
        options={
            "time_limit": time_limit_seconds,
            "mip_rel_gap": 0.0,
            "presolve": True,
        },
    )
    if result.status != 0 or result.x is None or abs(float(result.mip_gap)) > 1e-12:
        raise SplitError(
            "binary MILP did not prove an optimum: "
            f"status={result.status}, gap={getattr(result, 'mip_gap', None)}, "
            f"message={result.message}"
        )
    return (
        np.asarray(result.x),
        {
            "status": int(result.status),
            "message": str(result.message),
            "mip_gap": float(result.mip_gap),
            "mip_node_count": int(result.mip_node_count),
            "optimal_retained_unique_exact_pair_count": int(round(-float(result.fun))),
            "variable_count": index.count,
            "constraint_count": len(constraint_rows),
            "validation_axis_degree_integer_bounds": {
                "minimum": minimum,
                "maximum": maximum,
            },
            "largest_antibody_component_fixed_to_train": True,
        },
        antibody_clusters,
        antigen_clusters,
    )


def _bucket_ledger(
    bucket: str, rows: Sequence[base.SourceRow]
) -> dict[str, Any]:
    system_ids = sorted(row.system_id for row in rows)
    pair_ids = sorted(
        {base.exact_pair_id(row.antibody_digest, row.antigen_digest) for row in rows}
    )
    pdb_ids = sorted({row.pdb_id for row in rows})
    return {
        "bucket": bucket,
        "system_count": len(system_ids),
        "unique_exact_pair_count": len(pair_ids),
        "unique_pdb_count": len(pdb_ids),
        "system_ids": system_ids,
        "system_membership_sha256": audit.membership_sha256(system_ids),
        "exact_pair_ids": pair_ids,
        "exact_pair_membership_sha256": audit.membership_sha256(pair_ids),
        "pdb_ids": pdb_ids,
        "pdb_membership_sha256": audit.membership_sha256(pdb_ids),
    }


def _atomic_create(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    protocol_reference = audit.verify_file(
        arguments.protocol, arguments.protocol_sha256, "train-validation protocol"
    )
    protocol = load_protocol(Path(protocol_reference["path"]))
    source_reference = audit.verify_file(
        Path(__file__), arguments.source_sha256, "split builder source"
    )
    canonical_reference = audit.verify_file(
        arguments.canonical_manifest,
        protocol["source_sha256"]["canonical_manifest"],
        "canonical manifest",
    )
    assignment_reference = audit.verify_file(
        arguments.parent_assignment,
        protocol["source_sha256"]["parent_assignment"],
        "parent assignment",
    )
    audit_reference = audit.verify_file(
        arguments.parent_audit_result,
        protocol["source_sha256"]["parent_audit_result"],
        "parent audit result",
    )
    proof_reference = audit.verify_file(
        arguments.infeasibility_receipt,
        protocol["source_sha256"]["infeasibility_receipt"],
        "three-fold infeasibility receipt",
    )
    proof_payload = audit.strict_json(Path(proof_reference["path"]))
    if proof_payload.get("valid_infeasibility_proof") is not True:
        raise SplitError("three-fold infeasibility proof is invalid")
    if proof_payload.get("strict_balanced_three_fold_design_feasible") is not False:
        raise SplitError("three-fold design was not proven infeasible")

    rows = audit.load_source_rows(Path(canonical_reference["path"]))
    parent_assignment = audit.strict_json(Path(assignment_reference["path"]))
    audit.validate_assignment(
        parent_assignment,
        rows,
        protocol["source_sha256"]["parent_assignment_scientific_payload"],
    )
    parent_audit = audit.strict_json(Path(audit_reference["path"]))
    contracts = parent_audit.get("crossing_artifacts")
    if not isinstance(contracts, Mapping) or set(contracts) != set(
        proof.REQUIRED_VIEWS
    ):
        raise SplitError("parent audit crossing contracts are incomplete")
    edge_paths = {
        "vh": arguments.vh_edges,
        "vl": arguments.vl_edges,
        "concatenated_antibody": arguments.concatenated_antibody_edges,
        "antigen": arguments.antigen_edges,
    }
    view_digests = {
        "vh": {audit.sequence_sha256(row.vh) for row in rows},
        "vl": {audit.sequence_sha256(row.vl) for row in rows if row.vl},
        "concatenated_antibody": {row.antibody_digest for row in rows},
        "antigen": {row.antigen_digest for row in rows},
    }
    edges_by_view: dict[str, set[tuple[str, str]]] = {}
    edge_references: dict[str, dict[str, Any]] = {}
    for view in proof.REQUIRED_VIEWS:
        contract = contracts[view]
        edges, reference = proof.load_exact_edges(
            edge_paths[view],
            view,
            protocol["source_sha256"][f"{view}_crossing_edges"],
            int(contract["edge_count"]),
            view_digests[view],
        )
        if reference["sha256"] != contract["sha256"]:
            raise SplitError(f"{view} edge SHA differs from parent audit")
        edges_by_view[view] = edges
        edge_references[view] = reference

    antibody_roots, antigen_roots, exact_pairs, closure_evidence = _component_roots(
        rows, parent_assignment, edges_by_view
    )
    solver_rows = _base_rows(rows)
    components, _, component_by_system = base.interaction_components(
        solver_rows, antibody_roots, antigen_roots
    )
    balance = protocol["validation_axis_degree_fraction"]
    solution, solver_result, antibody_clusters, antigen_clusters = solve_binary(
        solver_rows,
        antibody_roots,
        antigen_roots,
        components,
        balance["minimum"],
        balance["maximum"],
        protocol["solver"]["time_limit_seconds"],
    )
    index = VariableIndex(
        len(antibody_clusters), len(antigen_clusters), len(components)
    )
    component_bucket: dict[str, str] = {}
    for number, component in enumerate(components):
        train = solution[index.component_train(number)] > 0.5
        validation = solution[index.component_validation(number)] > 0.5
        if train and validation:
            raise SplitError("component retained in both buckets")
        if train:
            component_bucket[component.component_id] = "train"
        elif validation:
            component_bucket[component.component_id] = "validation"

    roots = sorted({component_by_system[row.system_id] for row in solver_rows})
    if len(roots) != len(components):
        raise SplitError("interaction component root count mismatch")
    id_by_root = {
        root: component.component_id for root, component in zip(roots, components)
    }
    bucket_rows: defaultdict[str, list[base.SourceRow]] = defaultdict(list)
    for row in solver_rows:
        component_id = id_by_root[component_by_system[row.system_id]]
        bucket_rows[component_bucket.get(component_id, "discarded")].append(row)
    ledgers = {
        bucket: _bucket_ledger(bucket, bucket_rows[bucket])
        for bucket in (*BUCKETS, "discarded")
    }
    all_systems = {row.system_id for row in solver_rows}
    bucket_sets = [set(ledgers[bucket]["system_ids"]) for bucket in (*BUCKETS, "discarded")]
    if any(bucket_sets[left] & bucket_sets[right] for left in range(3) for right in range(left + 1, 3)):
        raise SplitError("bucket system ledgers overlap")
    if set().union(*bucket_sets) != all_systems:
        raise SplitError("bucket system ledgers are not exhaustive")
    for key in ("exact_pair_ids", "pdb_ids"):
        sets = [set(ledgers[bucket][key]) for bucket in (*BUCKETS, "discarded")]
        if any(sets[left] & sets[right] for left in range(3) for right in range(left + 1, 3)):
            raise SplitError(f"{key} are not bucket-indivisible")

    antibody_index = {root: number for number, root in enumerate(antibody_clusters)}
    antigen_index = {root: number for number, root in enumerate(antigen_clusters)}
    antibody_assignments = {
        digest: (
            "validation"
            if solution[index.antibody_validation(antibody_index[root])] > 0.5
            else "train"
        )
        for digest, root in sorted(antibody_roots.items())
    }
    antigen_assignments = {
        digest: (
            "validation"
            if solution[index.antigen_validation(antigen_index[root])] > 0.5
            else "train"
        )
        for digest, root in sorted(antigen_roots.items())
    }
    pair_axis_counts = {
        "antibody_validation_degree": sum(
            antibody_assignments[antibody] == "validation"
            for antibody, _ in exact_pairs
        ),
        "antigen_validation_degree": sum(
            antigen_assignments[antigen] == "validation"
            for _, antigen in exact_pairs
        ),
    }
    integer_bounds = solver_result["validation_axis_degree_integer_bounds"]
    if not all(
        integer_bounds["minimum"] <= value <= integer_bounds["maximum"]
        for value in pair_axis_counts.values()
    ):
        raise SplitError("validation axis degree is outside frozen bounds")

    immutable_references = {
        "protocol": protocol_reference,
        "builder_source": source_reference,
        "canonical_manifest": canonical_reference,
        "parent_assignment": assignment_reference,
        "parent_exact_audit_result": audit_reference,
        "three_fold_infeasibility_receipt": proof_reference,
        **{f"{view}_crossing_edges": edge_references[view] for view in proof.REQUIRED_VIEWS},
    }
    for label, reference in immutable_references.items():
        if audit.sha256_file(Path(reference["path"])) != reference["sha256"]:
            raise SplitError(f"{label} changed during split construction")

    payload: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "status": "PASS_PROVISIONAL_TRAIN_VALIDATION_S2_80_ASSIGNMENT",
        "approval_status": "UNAPPROVED_PENDING_EXACT_FOUR_VIEW_ZERO_CROSSING_AUDIT",
        "authorized_for_training": False,
        "authorized_for_model_selection": False,
        "inputs": immutable_references,
        "protocol": protocol,
        "closure_evidence": closure_evidence,
        "input_counts": {
            "canonical_systems": len(rows),
            "unique_concatenated_antibodies": len(antibody_roots),
            "unique_antigens": len(antigen_roots),
            "unique_exact_pairs": len(exact_pairs),
            "unique_pdbs": len({row.pdb_id for row in rows}),
        },
        "known_component_counts": {
            "antibody": len(set(antibody_roots.values())),
            "antigen": len(set(antigen_roots.values())),
            "pdb_exact_pair_components": len(components),
        },
        "solver_result": solver_result,
        "axis_degree_counts": pair_axis_counts,
        "entity_assignments": {
            "antibody": [
                {
                    "sequence_sha256": digest,
                    "component_root_sha256": antibody_roots[digest],
                    "bucket": bucket,
                }
                for digest, bucket in sorted(antibody_assignments.items())
            ],
            "antigen": [
                {
                    "sequence_sha256": digest,
                    "component_root_sha256": antigen_roots[digest],
                    "bucket": bucket,
                }
                for digest, bucket in sorted(antigen_assignments.items())
            ],
        },
        "bucket_ledgers": ledgers,
        "checks": {
            "all_source_systems_accounted_once": True,
            "exact_pairs_indivisible_across_all_buckets": True,
            "pdbs_indivisible_across_all_buckets": True,
            "largest_antibody_component_fixed_to_train": True,
            "validation_axis_degrees_within_frozen_bounds": True,
            "validation_rows_read": False,
            "test_rows_read": False,
            "task_labels_read": False,
            "structures_read": False,
            "predictions_or_scores_read": False,
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
        },
        "required_next_gate": (
            "independent exact four-view Needleman-Wunsch zero-crossing audit; "
            "if crossings remain, augment only from the new digest-only edge ledger "
            "and rebuild under this unchanged protocol"
        ),
        "claim_boundary": protocol["claim_boundary"],
    }
    payload["scientific_payload_sha256"] = canonical_sha256(payload)
    _atomic_create(arguments.output, audit.canonical_json_bytes(payload))
    return payload


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--canonical-manifest", type=Path, required=True)
    parser.add_argument("--parent-assignment", type=Path, required=True)
    parser.add_argument("--parent-audit-result", type=Path, required=True)
    parser.add_argument("--infeasibility-receipt", type=Path, required=True)
    parser.add_argument("--vh-edges", type=Path, required=True)
    parser.add_argument("--vl-edges", type=Path, required=True)
    parser.add_argument("--concatenated-antibody-edges", type=Path, required=True)
    parser.add_argument("--antigen-edges", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    for name in ("protocol_sha256", "source_sha256"):
        value = getattr(arguments, name)
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            parser.error(f"{name} is not a lowercase SHA-256")
    return arguments


if __name__ == "__main__":
    result = run(parse_arguments())
    print(
        json.dumps(
            {
                "status": result["status"],
                "train_systems": result["bucket_ledgers"]["train"]["system_count"],
                "validation_systems": result["bucket_ledgers"]["validation"]["system_count"],
                "discarded_systems": result["bucket_ledgers"]["discarded"]["system_count"],
                "retained_exact_pairs": result["solver_result"][
                    "optimal_retained_unique_exact_pair_count"
                ],
                "scientific_payload_sha256": result["scientific_payload_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
