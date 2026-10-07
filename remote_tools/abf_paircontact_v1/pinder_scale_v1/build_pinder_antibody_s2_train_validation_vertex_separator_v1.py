#!/usr/bin/env python3
"""Build a leakage-safe PINDER split by discarding 80/80 bridge entities."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import platform
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import scipy
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

import audit_pinder_antibody_s2_exact_four_view_v1 as audit
import audit_pinder_antibody_s2_full_four_view_relation_v1 as full_relation
import build_pinder_antibody_s2_train_validation_full_closure_v1 as full_closure
import build_pinder_antibody_s2_train_validation_v1 as split_builder
import build_pinder_antibody_s2_folds_v1 as fold_builder
import prove_pinder_antibody_s2_three_fold_infeasibility_v1 as closure


PROTOCOL_FORMAT_VERSION = "pinder_antibody_s2_vertex_separator_protocol_v1"
ASSIGNMENT_FORMAT_VERSION = "pinder_antibody_s2_train_validation_assignment_v1"
REQUIRED_VIEWS = full_relation.REQUIRED_VIEWS
BUCKETS = ("train", "validation")


class SeparatorError(RuntimeError):
    """Raised when a separator input, optimization, or invariant fails."""


@dataclass(frozen=True)
class VariableIndex:
    antibody_count: int
    antigen_count: int
    component_count: int

    @property
    def antibody_validation_offset(self) -> int:
        return self.antibody_count

    @property
    def antigen_train_offset(self) -> int:
        return 2 * self.antibody_count

    @property
    def antigen_validation_offset(self) -> int:
        return self.antigen_train_offset + self.antigen_count

    @property
    def component_train_offset(self) -> int:
        return self.antigen_validation_offset + self.antigen_count

    @property
    def component_validation_offset(self) -> int:
        return self.component_train_offset + self.component_count

    @property
    def count(self) -> int:
        return self.component_validation_offset + self.component_count

    def antibody_train(self, number: int) -> int:
        return number

    def antibody_validation(self, number: int) -> int:
        return self.antibody_validation_offset + number

    def antigen_train(self, number: int) -> int:
        return self.antigen_train_offset + number

    def antigen_validation(self, number: int) -> int:
        return self.antigen_validation_offset + number

    def component_train(self, number: int) -> int:
        return self.component_train_offset + number

    def component_validation(self, number: int) -> int:
        return self.component_validation_offset + number


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = audit.strict_json(path)
    if protocol.get("format_version") != PROTOCOL_FORMAT_VERSION:
        raise SeparatorError("vertex-separator protocol format mismatch")
    if protocol.get("identity_threshold") != 0.8:
        raise SeparatorError("identity threshold must remain 0.8")
    if protocol.get("reciprocal_coverage_threshold") != 0.8:
        raise SeparatorError("coverage threshold must remain 0.8")
    if protocol.get("validation_fraction") != {"minimum": 0.15, "maximum": 0.25}:
        raise SeparatorError("validation interval changed")
    if protocol.get("threshold_relaxation_after_results") is not False:
        raise SeparatorError("post-result threshold relaxation is forbidden")
    if protocol.get("bridge_entity_policy") != "discard_allowed":
        raise SeparatorError("bridge entities must be discardable")
    sources = protocol.get("source_sha256")
    required = {
        "canonical_manifest",
        "full_relation_result",
        "full_closure_infeasibility_receipt",
        *(f"{view}_complete_edges" for view in REQUIRED_VIEWS),
    }
    if not isinstance(sources, Mapping) or set(sources) != required:
        raise SeparatorError("protocol source bindings are incomplete")
    return protocol


def validate_infeasibility_receipt(
    receipt: Mapping[str, Any],
    canonical_manifest_sha256: str,
    full_relation_result_sha256: str,
) -> None:
    if receipt.get("status") != "FAIL_FULL_CLOSURE_BINARY_SPLIT_INFEASIBLE":
        raise SeparatorError("bound infeasibility receipt did not fail closed")
    if receipt.get("authorized_for_training") is not False:
        raise SeparatorError("infeasibility receipt unexpectedly authorized training")
    if receipt.get("authorized_for_model_selection") is not False:
        raise SeparatorError("infeasibility receipt unexpectedly authorized selection")
    inputs = receipt.get("inputs")
    if not isinstance(inputs, Mapping):
        raise SeparatorError("infeasibility receipt inputs are missing")
    if inputs.get("canonical_manifest", {}).get("sha256") != canonical_manifest_sha256:
        raise SeparatorError("infeasibility receipt canonical SHA mismatch")
    if inputs.get("full_relation_result", {}).get("sha256") != full_relation_result_sha256:
        raise SeparatorError("infeasibility receipt relation SHA mismatch")
    antibody = receipt.get("axis_results", {}).get("antibody", {})
    proof = receipt.get("proof", {})
    if antibody.get("feasible_subset_sum_exists") is not False:
        raise SeparatorError("infeasibility receipt lacks the antibody-axis witness")
    if proof.get("at_least_one_axis_has_no_feasible_validation_subset") is not True:
        raise SeparatorError("infeasibility receipt proof flag is missing")


def build_direct_forbidden_edges(
    rows: Sequence[audit.SourceRow],
    edges_by_view: Mapping[str, set[tuple[str, str]]],
) -> tuple[set[tuple[str, str]], set[tuple[str, str]], dict[str, Any]]:
    concat_by_chain: dict[str, defaultdict[str, set[str]]] = {
        "vh": defaultdict(set),
        "vl": defaultdict(set),
    }
    for row in rows:
        concat_by_chain["vh"][audit.sequence_sha256(row.vh)].add(
            row.antibody_digest
        )
        if row.vl:
            concat_by_chain["vl"][audit.sequence_sha256(row.vl)].add(
                row.antibody_digest
            )

    antibody_edges: set[tuple[str, str]] = set()
    lifted_counts: dict[str, int] = {}
    for view in ("vh", "vl"):
        mapping = concat_by_chain[view]
        before = len(antibody_edges)
        for members in mapping.values():
            antibody_edges.update(
                tuple(sorted(edge))
                for edge in itertools.combinations(sorted(members), 2)
            )
        for left, right in edges_by_view[view]:
            left_entities, right_entities = mapping.get(left), mapping.get(right)
            if not left_entities or not right_entities:
                raise SeparatorError(f"{view} edge cannot be lifted")
            for left_entity in left_entities:
                for right_entity in right_entities:
                    if left_entity != right_entity:
                        antibody_edges.add(tuple(sorted((left_entity, right_entity))))
        lifted_counts[view] = len(antibody_edges) - before
    before_concat = len(antibody_edges)
    antibody_edges.update(edges_by_view["concatenated_antibody"])
    lifted_counts["concatenated_antibody"] = len(antibody_edges) - before_concat
    antigen_edges = set(edges_by_view["antigen"])
    return antibody_edges, antigen_edges, {
        "direct_forbidden_antibody_edge_count": len(antibody_edges),
        "direct_forbidden_antigen_edge_count": len(antigen_edges),
        "new_unique_antibody_edges_by_ingestion_stage": lifted_counts,
        "transitive_closure_used_as_a_cross_split_constraint": False,
        "bridge_entities_may_be_discarded": True,
    }


def solve_separator(
    rows: Sequence[fold_builder.SourceRow],
    components: Sequence[fold_builder.Component],
    pair_by_id: Mapping[str, tuple[str, str]],
    antibody_edges: set[tuple[str, str]],
    antigen_edges: set[tuple[str, str]],
    minimum_fraction: float,
    maximum_fraction: float,
    time_limit_seconds: float,
) -> tuple[np.ndarray, dict[str, Any], tuple[str, ...], tuple[str, ...]]:
    antibodies = tuple(sorted({row.antibody_digest for row in rows}))
    antigens = tuple(sorted({row.antigen_digest for row in rows}))
    antibody_index = {value: number for number, value in enumerate(antibodies)}
    antigen_index = {value: number for number, value in enumerate(antigens)}
    index = VariableIndex(len(antibodies), len(antigens), len(components))
    exact_pairs = {(row.antibody_digest, row.antigen_digest) for row in rows}
    antibody_degree = Counter(antibody for antibody, _ in exact_pairs)
    antigen_degree = Counter(antigen for _, antigen in exact_pairs)
    total = len(exact_pairs)
    minimum, maximum = math.ceil(minimum_fraction * total), math.floor(
        maximum_fraction * total
    )

    matrix_rows: list[int] = []
    matrix_columns: list[int] = []
    matrix_values: list[float] = []
    lower: list[float] = []
    upper: list[float] = []

    def add(values: Sequence[tuple[int, float]], low: float, high: float) -> None:
        row_number = len(lower)
        lower.append(low)
        upper.append(high)
        for column, coefficient in values:
            matrix_rows.append(row_number)
            matrix_columns.append(column)
            matrix_values.append(coefficient)

    for number in range(len(antibodies)):
        add(
            [
                (index.antibody_train(number), 1.0),
                (index.antibody_validation(number), 1.0),
            ],
            -np.inf,
            1.0,
        )
    for number in range(len(antigens)):
        add(
            [
                (index.antigen_train(number), 1.0),
                (index.antigen_validation(number), 1.0),
            ],
            -np.inf,
            1.0,
        )

    for left, right in sorted(antibody_edges):
        left_number, right_number = antibody_index[left], antibody_index[right]
        add(
            [
                (index.antibody_train(left_number), 1.0),
                (index.antibody_validation(right_number), 1.0),
            ],
            -np.inf,
            1.0,
        )
        add(
            [
                (index.antibody_validation(left_number), 1.0),
                (index.antibody_train(right_number), 1.0),
            ],
            -np.inf,
            1.0,
        )
    for left, right in sorted(antigen_edges):
        left_number, right_number = antigen_index[left], antigen_index[right]
        add(
            [
                (index.antigen_train(left_number), 1.0),
                (index.antigen_validation(right_number), 1.0),
            ],
            -np.inf,
            1.0,
        )
        add(
            [
                (index.antigen_validation(left_number), 1.0),
                (index.antigen_train(right_number), 1.0),
            ],
            -np.inf,
            1.0,
        )

    for component_number, component in enumerate(components):
        train = index.component_train(component_number)
        validation = index.component_validation(component_number)
        add([(train, 1.0), (validation, 1.0)], -np.inf, 1.0)
        component_pairs = [pair_by_id[pair_id] for pair_id in component.pair_ids]
        component_antibodies = {pair[0] for pair in component_pairs}
        component_antigens = {pair[1] for pair in component_pairs}
        for antibody in sorted(component_antibodies):
            number = antibody_index[antibody]
            add(
                [(train, 1.0), (index.antibody_train(number), -1.0)],
                -np.inf,
                0.0,
            )
            add(
                [
                    (validation, 1.0),
                    (index.antibody_validation(number), -1.0),
                ],
                -np.inf,
                0.0,
            )
        for antigen in sorted(component_antigens):
            number = antigen_index[antigen]
            add(
                [(train, 1.0), (index.antigen_train(number), -1.0)],
                -np.inf,
                0.0,
            )
            add(
                [
                    (validation, 1.0),
                    (index.antigen_validation(number), -1.0),
                ],
                -np.inf,
                0.0,
            )

    add(
        [
            (index.antibody_validation(antibody_index[value]), float(degree))
            for value, degree in sorted(antibody_degree.items())
        ],
        minimum,
        maximum,
    )
    add(
        [
            (index.antigen_validation(antigen_index[value]), float(degree))
            for value, degree in sorted(antigen_degree.items())
        ],
        minimum,
        maximum,
    )
    add(
        [
            (index.component_validation(number), float(len(component.pair_ids)))
            for number, component in enumerate(components)
        ],
        minimum,
        maximum,
    )

    matrix = coo_matrix(
        (matrix_values, (matrix_rows, matrix_columns)),
        shape=(len(lower), index.count),
        dtype=float,
    ).tocsr()
    objective = np.zeros(index.count, dtype=float)
    primary_factor = total + 1
    for number, component in enumerate(components):
        weight = float(len(component.pair_ids))
        objective[index.component_train(number)] = -primary_factor * weight
        objective[index.component_validation(number)] = -(primary_factor + 1) * weight
    result = milp(
        objective,
        integrality=np.ones(index.count, dtype=int),
        bounds=Bounds(np.zeros(index.count), np.ones(index.count)),
        constraints=LinearConstraint(
            matrix, np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
        ),
        options={
            "time_limit": time_limit_seconds,
            "mip_rel_gap": 0.0,
            "presolve": True,
        },
    )
    if result.status != 0 or result.x is None or abs(float(result.mip_gap)) > 1e-12:
        raise SeparatorError(
            "vertex-separator MILP did not prove an optimum: "
            f"status={result.status}, gap={getattr(result, 'mip_gap', None)}, "
            f"message={result.message}"
        )
    solution = np.asarray(result.x)
    retained_train = sum(
        len(component.pair_ids)
        for number, component in enumerate(components)
        if solution[index.component_train(number)] > 0.5
    )
    retained_validation = sum(
        len(component.pair_ids)
        for number, component in enumerate(components)
        if solution[index.component_validation(number)] > 0.5
    )
    return solution, {
        "status": int(result.status),
        "message": str(result.message),
        "mip_gap": float(result.mip_gap),
        "mip_node_count": int(result.mip_node_count),
        "variable_count": index.count,
        "constraint_count": len(lower),
        "primary_objective": "maximize retained unique exact pairs",
        "secondary_objective": "maximize validation exact pairs at fixed total retention",
        "optimal_retained_unique_exact_pair_count": retained_train + retained_validation,
        "retained_train_unique_exact_pair_count": retained_train,
        "retained_validation_unique_exact_pair_count": retained_validation,
        "validation_integer_bounds": {"minimum": minimum, "maximum": maximum},
    }, antibodies, antigens


def build_ledgers(
    rows: Sequence[fold_builder.SourceRow],
    component_by_system: Mapping[str, str],
    components: Sequence[fold_builder.Component],
    solution: np.ndarray,
    index: VariableIndex,
) -> dict[str, dict[str, Any]]:
    roots = sorted({component_by_system[row.system_id] for row in rows})
    if len(roots) != len(components):
        raise SeparatorError("interaction component root count mismatch")
    id_by_root = {
        root: component.component_id for root, component in zip(roots, components)
    }
    component_bucket: dict[str, str] = {}
    for number, component in enumerate(components):
        train = solution[index.component_train(number)] > 0.5
        validation = solution[index.component_validation(number)] > 0.5
        if train and validation:
            raise SeparatorError("component retained in both buckets")
        if train:
            component_bucket[component.component_id] = "train"
        elif validation:
            component_bucket[component.component_id] = "validation"
    bucket_rows: defaultdict[str, list[fold_builder.SourceRow]] = defaultdict(list)
    for row in rows:
        component_id = id_by_root[component_by_system[row.system_id]]
        bucket_rows[component_bucket.get(component_id, "discarded")].append(row)
    return {
        bucket: split_builder._bucket_ledger(bucket, bucket_rows[bucket])
        for bucket in (*BUCKETS, "discarded")
    }


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    protocol_reference = audit.verify_file(
        arguments.protocol, arguments.protocol_sha256, "vertex-separator protocol"
    )
    protocol = load_protocol(Path(protocol_reference["path"]))
    source_reference = audit.verify_file(
        Path(__file__), arguments.source_sha256, "vertex-separator source"
    )
    sources = protocol["source_sha256"]
    canonical_reference = audit.verify_file(
        arguments.canonical_manifest,
        sources["canonical_manifest"],
        "canonical manifest",
    )
    relation_reference = audit.verify_file(
        arguments.full_relation_result,
        sources["full_relation_result"],
        "full relation result",
    )
    infeasibility_reference = audit.verify_file(
        arguments.infeasibility_receipt,
        sources["full_closure_infeasibility_receipt"],
        "full-closure infeasibility receipt",
    )
    relation = audit.strict_json(Path(relation_reference["path"]))
    contracts = full_closure.validate_full_relation_binding(
        relation, canonical_reference["sha256"]
    )
    infeasibility = audit.strict_json(Path(infeasibility_reference["path"]))
    validate_infeasibility_receipt(
        infeasibility,
        canonical_reference["sha256"],
        relation_reference["sha256"],
    )
    rows = audit.load_source_rows(Path(canonical_reference["path"]))
    view_digests = {
        "vh": {audit.sequence_sha256(row.vh) for row in rows},
        "vl": {audit.sequence_sha256(row.vl) for row in rows if row.vl},
        "concatenated_antibody": {row.antibody_digest for row in rows},
        "antigen": {row.antigen_digest for row in rows},
    }
    edge_paths = {
        "vh": arguments.vh_edges,
        "vl": arguments.vl_edges,
        "concatenated_antibody": arguments.concatenated_antibody_edges,
        "antigen": arguments.antigen_edges,
    }
    edges_by_view: dict[str, set[tuple[str, str]]] = {}
    edge_references: dict[str, dict[str, Any]] = {}
    for view in REQUIRED_VIEWS:
        contract = contracts[view]
        edges, reference = closure.load_exact_edges(
            edge_paths[view],
            view,
            sources[f"{view}_complete_edges"],
            int(contract["edge_count"]),
            view_digests[view],
        )
        if reference["sha256"] != contract["sha256"]:
            raise SeparatorError(f"{view} edge SHA differs from relation result")
        edges_by_view[view] = edges
        edge_references[view] = reference
    antibody_edges, antigen_edges, edge_evidence = build_direct_forbidden_edges(
        rows, edges_by_view
    )

    solver_rows = split_builder._base_rows(rows)
    identity_antibody = {row.antibody_digest: row.antibody_digest for row in rows}
    identity_antigen = {row.antigen_digest: row.antigen_digest for row in rows}
    components, pair_by_id, component_by_system = fold_builder.interaction_components(
        solver_rows, identity_antibody, identity_antigen
    )
    balance = protocol["validation_fraction"]
    solution, solver_result, antibodies, antigens = solve_separator(
        solver_rows,
        components,
        pair_by_id,
        antibody_edges,
        antigen_edges,
        balance["minimum"],
        balance["maximum"],
        protocol["solver"]["time_limit_seconds"],
    )
    index = VariableIndex(len(antibodies), len(antigens), len(components))
    ledgers = build_ledgers(
        solver_rows, component_by_system, components, solution, index
    )
    bucket_sets = [
        set(ledgers[bucket]["system_ids"])
        for bucket in (*BUCKETS, "discarded")
    ]
    if any(
        bucket_sets[left] & bucket_sets[right]
        for left in range(3)
        for right in range(left + 1, 3)
    ) or set().union(*bucket_sets) != {row.system_id for row in solver_rows}:
        raise SeparatorError("bucket ledgers are not disjoint and exhaustive")
    for key in ("exact_pair_ids", "pdb_ids"):
        sets = [set(ledgers[bucket][key]) for bucket in (*BUCKETS, "discarded")]
        if any(
            sets[left] & sets[right]
            for left in range(3)
            for right in range(left + 1, 3)
        ):
            raise SeparatorError(f"{key} are not bucket-indivisible")

    antibody_assignments = {
        value: (
            "train"
            if solution[index.antibody_train(number)] > 0.5
            else "validation"
            if solution[index.antibody_validation(number)] > 0.5
            else "discarded"
        )
        for number, value in enumerate(antibodies)
    }
    antigen_assignments = {
        value: (
            "train"
            if solution[index.antigen_train(number)] > 0.5
            else "validation"
            if solution[index.antigen_validation(number)] > 0.5
            else "discarded"
        )
        for number, value in enumerate(antigens)
    }
    direct_crossings = {
        "antibody": sum(
            {antibody_assignments[left], antibody_assignments[right]}
            == {"train", "validation"}
            for left, right in antibody_edges
        ),
        "antigen": sum(
            {antigen_assignments[left], antigen_assignments[right]}
            == {"train", "validation"}
            for left, right in antigen_edges
        ),
    }
    if any(direct_crossings.values()):
        raise SeparatorError("solver assignment violates a direct 80/80 edge")

    exact_pairs = {(row.antibody_digest, row.antigen_digest) for row in rows}
    axis_validation_degree = {
        "antibody": sum(
            antibody_assignments[antibody] == "validation"
            for antibody, _ in exact_pairs
        ),
        "antigen": sum(
            antigen_assignments[antigen] == "validation"
            for _, antigen in exact_pairs
        ),
    }
    payload: dict[str, Any] = {
        "format_version": ASSIGNMENT_FORMAT_VERSION,
        "closure_iteration": "complete_relation_vertex_separator_v1",
        "status": "PASS_PROVISIONAL_TRAIN_VALIDATION_S2_80_ASSIGNMENT",
        "approval_status": "UNAPPROVED_PENDING_EXACT_FOUR_VIEW_ZERO_CROSSING_AUDIT",
        "authorized_for_training": False,
        "authorized_for_model_selection": False,
        "inputs": {
            "protocol": protocol_reference,
            "builder_source": source_reference,
            "canonical_manifest": canonical_reference,
            "full_relation_result": relation_reference,
            "full_closure_infeasibility_receipt": infeasibility_reference,
            **{
                f"{view}_complete_edges": edge_references[view]
                for view in REQUIRED_VIEWS
            },
        },
        "protocol": protocol,
        "input_counts": {
            "canonical_systems": len(rows),
            "unique_concatenated_antibodies": len(antibodies),
            "unique_antigens": len(antigens),
            "unique_exact_pairs": len(exact_pairs),
            "unique_pdbs": len({row.pdb_id for row in rows}),
        },
        "direct_relation_evidence": edge_evidence,
        "solver_result": solver_result,
        "axis_validation_degree": axis_validation_degree,
        "entity_assignment_counts": {
            "antibody": dict(Counter(antibody_assignments.values())),
            "antigen": dict(Counter(antigen_assignments.values())),
        },
        "entity_assignments": {
            "antibody": [
                {"sequence_sha256": value, "bucket": bucket}
                for value, bucket in sorted(antibody_assignments.items())
            ],
            "antigen": [
                {"sequence_sha256": value, "bucket": bucket}
                for value, bucket in sorted(antigen_assignments.items())
            ],
        },
        "bucket_ledgers": ledgers,
        "checks": {
            "all_source_systems_accounted_once": True,
            "exact_pairs_indivisible_across_all_buckets": True,
            "pdbs_indivisible_across_all_buckets": True,
            "complete_direct_four_view_relation_constrained": True,
            "direct_antibody_crossing_count": 0,
            "direct_antigen_crossing_count": 0,
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
        "required_next_gate": "independent exact four-view Needleman-Wunsch zero-crossing audit",
        "claim_boundary": protocol["claim_boundary"],
    }
    payload["scientific_payload_sha256"] = split_builder.canonical_sha256(payload)
    split_builder._atomic_create(arguments.output, audit.canonical_json_bytes(payload))
    return payload


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--canonical-manifest", type=Path, required=True)
    parser.add_argument("--full-relation-result", type=Path, required=True)
    parser.add_argument("--infeasibility-receipt", type=Path, required=True)
    parser.add_argument("--vh-edges", type=Path, required=True)
    parser.add_argument("--vl-edges", type=Path, required=True)
    parser.add_argument("--concatenated-antibody-edges", type=Path, required=True)
    parser.add_argument("--antigen-edges", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    result = run(parse_arguments())
    print(
        json.dumps(
            {
                "status": result["status"],
                "train_systems": result["bucket_ledgers"]["train"]["system_count"],
                "validation_systems": result["bucket_ledgers"]["validation"]["system_count"],
                "discarded_systems": result["bucket_ledgers"]["discarded"]["system_count"],
                "solver_result": result["solver_result"],
                "scientific_payload_sha256": result["scientific_payload_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
