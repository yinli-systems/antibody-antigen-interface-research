#!/usr/bin/env python3
"""Solve a train-only approximate S2 retention feasibility problem.

The optimization maximizes retained unique exact sequence pairs while both
entity axes remain within the declared pair-degree tolerance. Fold-level pair
and duplicate source-row balance are reported descriptively and never override
the sequence-pair objective. The MMseqs2 clusters used here are a design
diagnostic; final folds still require exact four-view NW auditing.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import scipy
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix, lil_matrix

import analyze_train_only_s2_feasibility as base


FORMAT_VERSION = "sabdab2_train_only_s2_milp_feasibility_v1"


@dataclass(frozen=True)
class S2Problem:
    """One cluster-level bipartite retention problem."""

    antibody_clusters: tuple[str, ...]
    antigen_clusters: tuple[str, ...]
    edges: tuple[tuple[int, int], ...]
    unique_pair_weights: tuple[int, ...]
    source_row_weights: tuple[int, ...]
    antibody_pair_degrees: tuple[int, ...]
    antigen_pair_degrees: tuple[int, ...]
    total_unique_pairs: int
    total_source_rows: int


@dataclass(frozen=True)
class VariableIndex:
    """Contiguous binary-variable layout for one fixed fold count."""

    antibody_count: int
    antigen_count: int
    edge_count: int
    fold_count: int

    @property
    def antibody_offset(self) -> int:
        """Return the antibody-assignment offset."""
        return 0

    @property
    def antigen_offset(self) -> int:
        """Return the antigen-assignment offset."""
        return self.antibody_count * self.fold_count

    @property
    def edge_offset(self) -> int:
        """Return the retained-edge offset."""
        return (self.antibody_count + self.antigen_count) * self.fold_count

    @property
    def binary_count(self) -> int:
        """Return the total number of binary assignment variables."""
        return (
            self.antibody_count + self.antigen_count + self.edge_count
        ) * self.fold_count

    def antibody(self, cluster: int, fold: int) -> int:
        """Return one antibody assignment index."""
        return self.antibody_offset + cluster * self.fold_count + fold

    def antigen(self, cluster: int, fold: int) -> int:
        """Return one antigen assignment index."""
        return self.antigen_offset + cluster * self.fold_count + fold

    def edge(self, edge: int, fold: int) -> int:
        """Return one retained-edge index."""
        return self.edge_offset + edge * self.fold_count + fold


def build_problem(
    records: Sequence[tuple[str, str, str]],
    antibody_cluster_by_digest: Mapping[str, str],
    antigen_cluster_by_digest: Mapping[str, str],
) -> S2Problem:
    """Aggregate exact pairs and duplicate source rows on cluster-pair edges."""
    antibody_clusters = tuple(sorted(set(antibody_cluster_by_digest.values())))
    antigen_clusters = tuple(sorted(set(antigen_cluster_by_digest.values())))
    antibody_index = {value: index for index, value in enumerate(antibody_clusters)}
    antigen_index = {value: index for index, value in enumerate(antigen_clusters)}

    exact_pairs = {(antibody, antigen) for _, antibody, antigen in records}
    unique_pair_weights: Counter[tuple[int, int]] = Counter()
    source_row_weights: Counter[tuple[int, int]] = Counter()
    for antibody, antigen in exact_pairs:
        edge = (
            antibody_index[antibody_cluster_by_digest[antibody]],
            antigen_index[antigen_cluster_by_digest[antigen]],
        )
        unique_pair_weights[edge] += 1
    for _, antibody, antigen in records:
        edge = (
            antibody_index[antibody_cluster_by_digest[antibody]],
            antigen_index[antigen_cluster_by_digest[antigen]],
        )
        source_row_weights[edge] += 1

    edges = tuple(sorted(unique_pair_weights))
    antibody_degrees = [0] * len(antibody_clusters)
    antigen_degrees = [0] * len(antigen_clusters)
    for edge in edges:
        weight = unique_pair_weights[edge]
        antibody_degrees[edge[0]] += weight
        antigen_degrees[edge[1]] += weight
    return S2Problem(
        antibody_clusters=antibody_clusters,
        antigen_clusters=antigen_clusters,
        edges=edges,
        unique_pair_weights=tuple(unique_pair_weights[edge] for edge in edges),
        source_row_weights=tuple(source_row_weights[edge] for edge in edges),
        antibody_pair_degrees=tuple(antibody_degrees),
        antigen_pair_degrees=tuple(antigen_degrees),
        total_unique_pairs=len(exact_pairs),
        total_source_rows=len(records),
    )


def base_constraints(
    problem: S2Problem,
    index: VariableIndex,
    axis_balance_tolerance: float,
) -> tuple[list[list[tuple[int, float]]], list[float], list[float]]:
    """Build assignment, AND-linearization, and two-axis balance constraints."""
    rows: list[list[tuple[int, float]]] = []
    lower: list[float] = []
    upper: list[float] = []

    def add(values: list[tuple[int, float]], low: float, high: float) -> None:
        rows.append(values)
        lower.append(low)
        upper.append(high)

    for antibody in range(index.antibody_count):
        add(
            [(index.antibody(antibody, fold), 1.0) for fold in range(index.fold_count)],
            1.0,
            1.0,
        )
    for antigen in range(index.antigen_count):
        add(
            [(index.antigen(antigen, fold), 1.0) for fold in range(index.fold_count)],
            1.0,
            1.0,
        )
    for edge, (antibody, antigen) in enumerate(problem.edges):
        for fold in range(index.fold_count):
            retained = index.edge(edge, fold)
            add([(retained, 1.0), (index.antibody(antibody, fold), -1.0)], -np.inf, 0.0)
            add([(retained, 1.0), (index.antigen(antigen, fold), -1.0)], -np.inf, 0.0)
            add(
                [
                    (index.antibody(antibody, fold), 1.0),
                    (index.antigen(antigen, fold), 1.0),
                    (retained, -1.0),
                ],
                -np.inf,
                1.0,
            )

    target = problem.total_unique_pairs / index.fold_count
    balance_lower = (1.0 - axis_balance_tolerance) * target
    balance_upper = (1.0 + axis_balance_tolerance) * target
    for fold in range(index.fold_count):
        add(
            [
                (index.antibody(cluster, fold), float(weight))
                for cluster, weight in enumerate(problem.antibody_pair_degrees)
            ],
            balance_lower,
            balance_upper,
        )
        add(
            [
                (index.antigen(cluster, fold), float(weight))
                for cluster, weight in enumerate(problem.antigen_pair_degrees)
            ],
            balance_lower,
            balance_upper,
        )
    return rows, lower, upper


def sparse_constraints(
    rows: Sequence[Sequence[tuple[int, float]]],
    lower: Sequence[float],
    upper: Sequence[float],
    variable_count: int,
) -> LinearConstraint:
    """Convert row-wise sparse coefficients to one SciPy constraint object."""
    matrix = lil_matrix((len(rows), variable_count), dtype=float)
    for row_index, values in enumerate(rows):
        for column, coefficient in values:
            matrix[row_index, column] = coefficient
    return LinearConstraint(
        csr_matrix(matrix), np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    )


def solver_options(time_limit_seconds: float) -> dict[str, Any]:
    """Return exact-MIP options shared by all lexicographic stages."""
    return {
        "time_limit": float(time_limit_seconds),
        "mip_rel_gap": 0.0,
        "presolve": True,
        "disp": False,
    }


def solve_stage(
    objective: np.ndarray,
    integrality: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    constraint: LinearConstraint,
    time_limit_seconds: float,
    label: str,
) -> Any:
    """Solve one MIP stage and require a proven zero-gap optimum."""
    result = milp(
        objective,
        integrality=integrality,
        bounds=Bounds(lower_bounds, upper_bounds),
        constraints=constraint,
        options=solver_options(time_limit_seconds),
    )
    if (
        result.status != 0
        or result.x is None
        or result.fun is None
        or getattr(result, "mip_gap", None) is None
        or abs(float(result.mip_gap)) > 1e-12
    ):
        raise RuntimeError(
            f"{label} did not reach a proven optimum: "
            f"status={result.status}, message={result.message}, "
            f"mip_gap={getattr(result, 'mip_gap', None)}"
        )
    return result


def retained_expression(
    problem: S2Problem,
    index: VariableIndex,
    fold: int | None,
    weights: Sequence[int],
) -> list[tuple[int, float]]:
    """Return a sparse retained-interaction expression."""
    folds = range(index.fold_count) if fold is None else (fold,)
    return [
        (index.edge(edge, current_fold), float(weights[edge]))
        for edge in range(index.edge_count)
        for current_fold in folds
    ]


def optimize(
    problem: S2Problem,
    fold_count: int,
    axis_balance_tolerance: float,
    time_limit_seconds: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Maximize retained exact pairs under the two-axis balance contract."""
    index = VariableIndex(
        len(problem.antibody_clusters),
        len(problem.antigen_clusters),
        len(problem.edges),
        fold_count,
    )
    rows, lower, upper = base_constraints(problem, index, axis_balance_tolerance)
    binary_count = index.binary_count
    binary_lower = np.zeros(binary_count, dtype=float)
    binary_upper = np.ones(binary_count, dtype=float)
    integrality = np.ones(binary_count, dtype=int)

    largest_antibody = min(
        range(index.antibody_count),
        key=lambda cluster: (
            -problem.antibody_pair_degrees[cluster],
            problem.antibody_clusters[cluster],
        ),
    )
    binary_lower[index.antibody(largest_antibody, 0)] = 1.0
    binary_upper[index.antibody(largest_antibody, 0)] = 1.0

    objective_one = np.zeros(binary_count, dtype=float)
    for column, weight in retained_expression(
        problem, index, None, problem.unique_pair_weights
    ):
        objective_one[column] = -weight
    stage_one = solve_stage(
        objective_one,
        integrality,
        binary_lower,
        binary_upper,
        sparse_constraints(rows, lower, upper, binary_count),
        time_limit_seconds,
        "retained-pair maximization",
    )
    retained_optimum = int(round(-float(stage_one.fun)))
    stages = {
        "stage_1": {
            "objective": "maximize_retained_unique_exact_pairs",
            "optimal_retained_unique_exact_pair_count": retained_optimum,
            "mip_gap": float(stage_one.mip_gap),
            "mip_node_count": int(stage_one.mip_node_count),
        },
    }
    return np.asarray(stage_one.x[:binary_count]), stages


def summarize_solution(
    problem: S2Problem,
    solution: np.ndarray,
    fold_count: int,
    axis_balance_tolerance: float,
) -> dict[str, Any]:
    """Recompute every reported count independently from binary assignments."""
    index = VariableIndex(
        len(problem.antibody_clusters),
        len(problem.antigen_clusters),
        len(problem.edges),
        fold_count,
    )

    def assigned_fold(columns: Sequence[int], label: str) -> int:
        values = [float(solution[column]) for column in columns]
        rounded = [int(round(value)) for value in values]
        if sum(rounded) != 1 or any(abs(value - round(value)) > 1e-7 for value in values):
            raise RuntimeError(f"{label} assignment is not one-hot integral")
        return rounded.index(1)

    antibody_fold = [
        assigned_fold(
            [index.antibody(cluster, fold) for fold in range(fold_count)],
            f"antibody cluster {cluster}",
        )
        for cluster in range(index.antibody_count)
    ]
    antigen_fold = [
        assigned_fold(
            [index.antigen(cluster, fold) for fold in range(fold_count)],
            f"antigen cluster {cluster}",
        )
        for cluster in range(index.antigen_count)
    ]
    retained_pairs = [0] * fold_count
    retained_rows = [0] * fold_count
    discarded_pairs = 0
    discarded_rows = 0
    for edge, (antibody, antigen) in enumerate(problem.edges):
        if antibody_fold[antibody] == antigen_fold[antigen]:
            fold = antibody_fold[antibody]
            retained_pairs[fold] += problem.unique_pair_weights[edge]
            retained_rows[fold] += problem.source_row_weights[edge]
        else:
            discarded_pairs += problem.unique_pair_weights[edge]
            discarded_rows += problem.source_row_weights[edge]
    antibody_degrees = [0] * fold_count
    antigen_degrees = [0] * fold_count
    antibody_cluster_counts = [0] * fold_count
    antigen_cluster_counts = [0] * fold_count
    for cluster, fold in enumerate(antibody_fold):
        antibody_degrees[fold] += problem.antibody_pair_degrees[cluster]
        antibody_cluster_counts[fold] += 1
    for cluster, fold in enumerate(antigen_fold):
        antigen_degrees[fold] += problem.antigen_pair_degrees[cluster]
        antigen_cluster_counts[fold] += 1
    target = problem.total_unique_pairs / fold_count
    balance_lower = (1.0 - axis_balance_tolerance) * target
    balance_upper = (1.0 + axis_balance_tolerance) * target
    if not all(
        balance_lower - 1e-9 <= value <= balance_upper + 1e-9
        for value in [*antibody_degrees, *antigen_degrees]
    ):
        raise RuntimeError("Recomputed axis degree violates the balance contract")
    return {
        "retained_unique_exact_pair_count": sum(retained_pairs),
        "discarded_unique_exact_pair_count": discarded_pairs,
        "retained_unique_exact_pair_fraction": (
            sum(retained_pairs) / problem.total_unique_pairs
        ),
        "retained_source_row_count": sum(retained_rows),
        "discarded_source_row_count": discarded_rows,
        "retained_source_row_fraction": sum(retained_rows) / problem.total_source_rows,
        "folds": [
            {
                "fold_index": fold,
                "retained_unique_exact_pair_count": retained_pairs[fold],
                "retained_source_row_count": retained_rows[fold],
                "assigned_antibody_cluster_count": antibody_cluster_counts[fold],
                "assigned_antigen_cluster_count": antigen_cluster_counts[fold],
                "assigned_antibody_unique_pair_degree": antibody_degrees[fold],
                "assigned_antigen_unique_pair_degree": antigen_degrees[fold],
            }
            for fold in range(fold_count)
        ],
        "retained_unique_exact_pair_count_range": max(retained_pairs) - min(retained_pairs),
        "retained_source_row_count_range": max(retained_rows) - min(retained_rows),
        "axis_balance_bounds": {
            "target": target,
            "lower": balance_lower,
            "upper": balance_upper,
            "all_fold_axis_degrees_within_bounds": True,
        },
        "cross_fold_interactions_retained": 0,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority-split", type=Path, required=True)
    parser.add_argument("--authority-sha256", required=True)
    parser.add_argument("--canonical-manifest", type=Path, required=True)
    parser.add_argument("--canonical-manifest-sha256", required=True)
    parser.add_argument("--antibody-similarity-tsv", type=Path, required=True)
    parser.add_argument("--antibody-similarity-sha256", required=True)
    parser.add_argument("--antigen-similarity-tsv", type=Path, required=True)
    parser.add_argument("--antigen-similarity-sha256", required=True)
    parser.add_argument("--expected-base-source-sha256", required=True)
    parser.add_argument("--identity-threshold", type=float, default=0.80)
    parser.add_argument("--coverage-threshold", type=float, default=0.80)
    parser.add_argument("--fold-count", type=int, default=3)
    parser.add_argument("--axis-balance-tolerance", type=float, default=0.05)
    parser.add_argument("--time-limit-seconds", type=float, default=300.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.fold_count != 3:
        parser.error("This diagnostic requires exactly three folds")
    for name in ("identity_threshold", "coverage_threshold", "axis_balance_tolerance"):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            parser.error(f"--{name.replace('_', '-')} must be within [0, 1]")
    if args.time_limit_seconds <= 0:
        parser.error("--time-limit-seconds must be positive")
    return args


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Verify inputs, solve the diagnostic, and create an immutable receipt."""
    base_source = base.verify_file(
        Path(base.__file__),
        args.expected_base_source_sha256,
        "base feasibility source",
    )
    input_references = {
        "authority_split": base.verify_file(
            args.authority_split, args.authority_sha256, "authority split"
        ),
        "canonical_sequence_manifest": base.verify_file(
            args.canonical_manifest,
            args.canonical_manifest_sha256,
            "canonical sequence manifest",
        ),
        "antibody_mmseqs2_similarity": base.verify_file(
            args.antibody_similarity_tsv,
            args.antibody_similarity_sha256,
            "antibody similarity TSV",
        ),
        "antigen_mmseqs2_similarity": base.verify_file(
            args.antigen_similarity_tsv,
            args.antigen_similarity_sha256,
            "antigen similarity TSV",
        ),
        "base_feasibility_source": base_source,
    }
    train_ids, sealed_ids, authority = base.load_authority(
        Path(input_references["authority_split"]["path"])
    )
    records, firewall = base.load_authorized_pairs(
        Path(input_references["canonical_sequence_manifest"]["path"]),
        train_ids,
        sealed_ids,
    )
    antibody_digests = {record[1] for record in records}
    antigen_digests = {record[2] for record in records}
    antibody_edges, antibody_similarity = base.load_similarity_relation(
        Path(input_references["antibody_mmseqs2_similarity"]["path"]),
        antibody_digests,
        args.identity_threshold,
        args.coverage_threshold,
    )
    antigen_edges, antigen_similarity = base.load_similarity_relation(
        Path(input_references["antigen_mmseqs2_similarity"]["path"]),
        antigen_digests,
        args.identity_threshold,
        args.coverage_threshold,
    )
    antibody_clusters, antibody_cluster_summary = base.cluster_entities(
        antibody_digests, antibody_edges
    )
    antigen_clusters, antigen_cluster_summary = base.cluster_entities(
        antigen_digests, antigen_edges
    )
    problem = build_problem(records, antibody_clusters, antigen_clusters)
    solution, stages = optimize(
        problem,
        args.fold_count,
        args.axis_balance_tolerance,
        args.time_limit_seconds,
    )
    summary = summarize_solution(
        problem, solution, args.fold_count, args.axis_balance_tolerance
    )
    payload: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "stage": "train_only_approximate_s2_milp_feasibility",
        "intended_use": "train_only_split_design_diagnostic",
        "valid": True,
        "inputs": input_references,
        "parameters": {
            "identity_threshold": args.identity_threshold,
            "reciprocal_coverage_threshold": args.coverage_threshold,
            "fold_count": args.fold_count,
            "axis_balance_tolerance": args.axis_balance_tolerance,
            "similarity_backend": "precomputed_mmseqs2",
            "primary_balance_and_retention_unit": "unique_exact_sequence_pair",
            "time_limit_seconds_per_stage": args.time_limit_seconds,
        },
        "authority": authority,
        "data_firewall": firewall,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "solver": "scipy.optimize.milp_HiGHS",
        },
        "similarity_evidence": {
            "antibody": antibody_similarity,
            "antigen": antigen_similarity,
        },
        "entity_clusters": {
            "antibody": antibody_cluster_summary,
            "antigen": antigen_cluster_summary,
        },
        "optimization_problem": {
            "antibody_cluster_count": len(problem.antibody_clusters),
            "antigen_cluster_count": len(problem.antigen_clusters),
            "cluster_pair_edge_count": len(problem.edges),
            "total_unique_exact_pair_count": problem.total_unique_pairs,
            "total_source_row_count": problem.total_source_rows,
            "assignment_exported": False,
        },
        "solver_result": stages["stage_1"],
        "solution_summary": summary,
        "decision": {
            "approximate_three_fold_s2_is_feasible": True,
            "whole_bipartite_component_assignment_required": False,
            "cross_fold_interactions_must_be_discarded": True,
            "final_fold_freeze_authorized": False,
        },
        "limitations": [
            "This is an authority-train-only split-design diagnostic, not model evaluation.",
            "MMseqs2 relations are approximate; final folds require exact Needleman-Wunsch audit closure.",
            "Only concatenated-antibody and antigen relations are optimized here; final S2 clustering must also include VH and VL relations.",
            "The solver assignment is intentionally not exported because the approximate folds are not authorized for model selection.",
            "No authority-validation or frozen-test sequence, label, embedding, prediction, checkpoint, or score was materialized.",
        ],
    }
    payload["scientific_payload_sha256"] = base.canonical_sha256(payload)
    base.atomic_create_json(args.output, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    """Run the command-line entrypoint."""
    payload = run(parse_args(argv))
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
