#!/usr/bin/env python3
"""Build an auditable, provisional train-only two-axis S2 fold assignment.

The builder assigns concatenated-antibody and antigen similarity clusters to
three folds with the previously validated retention-maximizing MILP. Positive
edges whose two endpoint clusters receive different folds are explicitly
discarded. The resulting artifact is deliberately not approved for training or
model selection until a separate exact four-view Needleman--Wunsch audit passes.
"""

from __future__ import annotations

import argparse
import errno
import json
import os
import platform
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import scipy

import analyze_train_only_s2_feasibility as base
import optimize_train_only_s2_feasibility as optimizer


FORMAT_VERSION = "sabdab2_train_only_two_axis_s2_fold_assignment_v1"
CONFIG_FORMAT_VERSION = "sabdab2_train_only_two_axis_s2_builder_config_v1"
REQUIRED_SEQUENCE_VIEWS = ("vh", "vl", "concatenated_antibody", "antigen")


def _require_exact_keys(
    payload: Mapping[str, Any], required: set[str], label: str
) -> None:
    """Require an exact mapping schema."""
    observed = set(payload)
    if observed != required:
        missing = sorted(required - observed)
        unexpected = sorted(observed - required)
        raise ValueError(
            f"{label} has an invalid schema: missing={missing}, "
            f"unexpected={unexpected}"
        )


def _finite_number(value: Any, label: str) -> float:
    """Return one finite non-boolean JSON number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a JSON number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def load_config(path: Path) -> dict[str, Any]:
    """Load and validate the exact immutable builder configuration."""
    payload = base.strict_json_object(path, "S2 builder config")
    required = {
        "format_version",
        "identity_threshold",
        "reciprocal_coverage_threshold",
        "fold_count",
        "axis_balance_tolerance",
        "time_limit_seconds",
        "similarity_backend",
        "optimized_sequence_views",
        "deterministic_policy",
    }
    _require_exact_keys(payload, required, "S2 builder config")
    if payload["format_version"] != CONFIG_FORMAT_VERSION:
        raise ValueError("Unsupported S2 builder config format_version")
    if payload["similarity_backend"] != "precomputed_mmseqs2":
        raise ValueError("similarity_backend must be precomputed_mmseqs2")
    if payload["optimized_sequence_views"] != [
        "concatenated_antibody",
        "antigen",
    ]:
        raise ValueError(
            "optimized_sequence_views must be concatenated_antibody and antigen"
        )
    if payload["deterministic_policy"] != (
        "sorted_inputs_fixed_highs_then_canonical_fold_relabel_v1"
    ):
        raise ValueError("Unsupported deterministic_policy")
    fold_count = payload["fold_count"]
    if isinstance(fold_count, bool) or not isinstance(fold_count, int):
        raise ValueError("fold_count must be an integer")
    if fold_count != 3:
        raise ValueError("This builder requires exactly three folds")
    identity = _finite_number(payload["identity_threshold"], "identity_threshold")
    coverage = _finite_number(
        payload["reciprocal_coverage_threshold"],
        "reciprocal_coverage_threshold",
    )
    tolerance = _finite_number(
        payload["axis_balance_tolerance"], "axis_balance_tolerance"
    )
    time_limit = _finite_number(payload["time_limit_seconds"], "time_limit_seconds")
    for value, label in (
        (identity, "identity_threshold"),
        (coverage, "reciprocal_coverage_threshold"),
        (tolerance, "axis_balance_tolerance"),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{label} must be within [0, 1]")
    if time_limit <= 0.0:
        raise ValueError("time_limit_seconds must be positive")
    return {
        **payload,
        "identity_threshold": identity,
        "reciprocal_coverage_threshold": coverage,
        "axis_balance_tolerance": tolerance,
        "time_limit_seconds": time_limit,
    }


def atomic_create_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish one complete JSON file atomically without overwrite semantics."""
    target = path.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".staged", dir=target.parent
    )
    temporary = Path(temporary_name)
    published = False
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target)
            published = True
        except FileExistsError:
            raise
        except OSError as error:
            if error.errno == errno.EEXIST:
                raise FileExistsError(target) from error
            raise
        directory_descriptor = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)
        if not published and target.exists():
            # A concurrent publisher owns an existing target and must be preserved.
            pass


def _one_hot_fold(values: Sequence[float], label: str) -> int:
    """Decode one strict one-hot assignment from a proven MIP solution."""
    rounded = [int(round(float(value))) for value in values]
    if sum(rounded) != 1 or any(
        abs(float(value) - round(float(value))) > 1e-7 for value in values
    ):
        raise RuntimeError(f"{label} is not one-hot integral")
    return rounded.index(1)


def decode_cluster_folds(
    problem: optimizer.S2Problem,
    solution: np.ndarray,
    fold_count: int,
) -> tuple[list[int], list[int]]:
    """Decode antibody and antigen cluster folds from the binary solution."""
    index = optimizer.VariableIndex(
        len(problem.antibody_clusters),
        len(problem.antigen_clusters),
        len(problem.edges),
        fold_count,
    )
    antibody_folds = [
        _one_hot_fold(
            [solution[index.antibody(cluster, fold)] for fold in range(fold_count)],
            f"antibody cluster {cluster}",
        )
        for cluster in range(index.antibody_count)
    ]
    antigen_folds = [
        _one_hot_fold(
            [solution[index.antigen(cluster, fold)] for fold in range(fold_count)],
            f"antigen cluster {cluster}",
        )
        for cluster in range(index.antigen_count)
    ]
    return antibody_folds, antigen_folds


def canonicalize_fold_labels(
    problem: optimizer.S2Problem,
    antibody_folds: Sequence[int],
    antigen_folds: Sequence[int],
    fold_count: int,
) -> tuple[list[int], list[int], dict[int, int]]:
    """Remove arbitrary solver fold-label permutations deterministically."""
    signatures: list[tuple[tuple[str, ...], tuple[str, ...], int]] = []
    for fold in range(fold_count):
        antibodies = tuple(
            cluster
            for cluster, assigned in zip(problem.antibody_clusters, antibody_folds)
            if assigned == fold
        )
        antigens = tuple(
            cluster
            for cluster, assigned in zip(problem.antigen_clusters, antigen_folds)
            if assigned == fold
        )
        signatures.append((antibodies, antigens, fold))
    old_to_new = {
        signature[2]: new_fold
        for new_fold, signature in enumerate(sorted(signatures))
    }
    return (
        [old_to_new[fold] for fold in antibody_folds],
        [old_to_new[fold] for fold in antigen_folds],
        old_to_new,
    )


def encode_solution(
    problem: optimizer.S2Problem,
    antibody_folds: Sequence[int],
    antigen_folds: Sequence[int],
    fold_count: int,
) -> np.ndarray:
    """Encode canonical assignments and their retained-edge AND variables."""
    index = optimizer.VariableIndex(
        len(problem.antibody_clusters),
        len(problem.antigen_clusters),
        len(problem.edges),
        fold_count,
    )
    solution = np.zeros(index.binary_count, dtype=float)
    for cluster, fold in enumerate(antibody_folds):
        solution[index.antibody(cluster, fold)] = 1.0
    for cluster, fold in enumerate(antigen_folds):
        solution[index.antigen(cluster, fold)] = 1.0
    for edge, (antibody, antigen) in enumerate(problem.edges):
        if antibody_folds[antibody] == antigen_folds[antigen]:
            solution[index.edge(edge, antibody_folds[antibody])] = 1.0
    return solution


def _stable_cluster_ids(
    cluster_roots: Iterable[str], prefix: str
) -> dict[str, str]:
    """Map lexical cluster representatives to compact deterministic labels."""
    return {
        root: f"{prefix}_{index:06d}"
        for index, root in enumerate(sorted(set(cluster_roots)))
    }


def _edge_id(antibody_digest: str, antigen_digest: str) -> str:
    """Return a collision-resistant identifier for one exact sequence pair."""
    return base.sequence_sha256(f"{antibody_digest}\0{antigen_digest}")


def build_exports(
    records: Sequence[tuple[str, str, str]],
    problem: optimizer.S2Problem,
    antibody_cluster_by_digest: Mapping[str, str],
    antigen_cluster_by_digest: Mapping[str, str],
    antibody_folds: Sequence[int],
    antigen_folds: Sequence[int],
    fold_count: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Build cluster assignments, retained folds, and complete discard ledgers."""
    antibody_cluster_index = {
        cluster: index for index, cluster in enumerate(problem.antibody_clusters)
    }
    antigen_cluster_index = {
        cluster: index for index, cluster in enumerate(problem.antigen_clusters)
    }
    antibody_cluster_ids = _stable_cluster_ids(
        problem.antibody_clusters, "antibody_cluster"
    )
    antigen_cluster_ids = _stable_cluster_ids(problem.antigen_clusters, "antigen_cluster")

    cluster_assignments = {
        "antibody": [
            {
                "sequence_sha256": digest,
                "cluster_id": antibody_cluster_ids[root],
                "cluster_representative_sequence_sha256": root,
                "fold_index": antibody_folds[antibody_cluster_index[root]],
            }
            for digest, root in sorted(antibody_cluster_by_digest.items())
        ],
        "antigen": [
            {
                "sequence_sha256": digest,
                "cluster_id": antigen_cluster_ids[root],
                "cluster_representative_sequence_sha256": root,
                "fold_index": antigen_folds[antigen_cluster_index[root]],
            }
            for digest, root in sorted(antigen_cluster_by_digest.items())
        ],
    }
    antibody_fold_by_digest = {
        row["sequence_sha256"]: row["fold_index"]
        for row in cluster_assignments["antibody"]
    }
    antigen_fold_by_digest = {
        row["sequence_sha256"]: row["fold_index"]
        for row in cluster_assignments["antigen"]
    }
    source_ids_by_pair: defaultdict[tuple[str, str], list[str]] = defaultdict(list)
    for instance_id, antibody, antigen in sorted(records):
        source_ids_by_pair[(antibody, antigen)].append(instance_id)

    retained_by_fold: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    discarded_edges: list[dict[str, Any]] = []
    discarded_source_rows: list[dict[str, Any]] = []
    retained_source_ids_by_fold: defaultdict[int, list[str]] = defaultdict(list)
    retained_pair_ids_by_fold: defaultdict[int, list[str]] = defaultdict(list)
    for (antibody, antigen), source_ids in sorted(source_ids_by_pair.items()):
        antibody_fold = antibody_fold_by_digest[antibody]
        antigen_fold = antigen_fold_by_digest[antigen]
        edge_id = _edge_id(antibody, antigen)
        antibody_root = antibody_cluster_by_digest[antibody]
        antigen_root = antigen_cluster_by_digest[antigen]
        common = {
            "exact_pair_id_sha256": edge_id,
            "antibody_sequence_sha256": antibody,
            "antigen_sequence_sha256": antigen,
            "antibody_cluster_id": antibody_cluster_ids[antibody_root],
            "antigen_cluster_id": antigen_cluster_ids[antigen_root],
            "antibody_fold_index": antibody_fold,
            "antigen_fold_index": antigen_fold,
            "source_instance_ids": sorted(source_ids),
            "source_row_count": len(source_ids),
        }
        if antibody_fold == antigen_fold:
            retained_by_fold[antibody_fold].append(
                {**common, "fold_index": antibody_fold}
            )
            retained_pair_ids_by_fold[antibody_fold].append(edge_id)
            retained_source_ids_by_fold[antibody_fold].extend(source_ids)
        else:
            discarded_edges.append(
                {**common, "discard_reason": "cross_fold_cluster_assignment"}
            )
            discarded_source_rows.extend(
                {
                    "source_instance_id": instance_id,
                    "exact_pair_id_sha256": edge_id,
                    "antibody_fold_index": antibody_fold,
                    "antigen_fold_index": antigen_fold,
                    "discard_reason": "cross_fold_cluster_assignment",
                }
                for instance_id in sorted(source_ids)
            )

    retained_folds = []
    for fold in range(fold_count):
        exact_pairs = sorted(
            retained_by_fold[fold], key=lambda row: row["exact_pair_id_sha256"]
        )
        source_ids = sorted(retained_source_ids_by_fold[fold])
        pair_ids = sorted(retained_pair_ids_by_fold[fold])
        retained_folds.append(
            {
                "fold_index": fold,
                "retained_unique_exact_pair_count": len(exact_pairs),
                "retained_source_row_count": len(source_ids),
                "retained_exact_pairs": exact_pairs,
                "retained_exact_pair_membership_sha256": base.membership_sha256(
                    pair_ids
                ),
                "retained_source_instance_ids": source_ids,
                "retained_source_membership_sha256": base.membership_sha256(source_ids),
            }
        )
    discarded_edges.sort(key=lambda row: row["exact_pair_id_sha256"])
    discarded_source_rows.sort(
        key=lambda row: (row["source_instance_id"], row["exact_pair_id_sha256"])
    )
    discarded_ledger = {
        "discard_policy": "discard_every_positive_edge_whose_endpoint_folds_differ",
        "discarded_unique_exact_pair_count": len(discarded_edges),
        "discarded_source_row_count": len(discarded_source_rows),
        "discarded_cross_fold_edges": discarded_edges,
        "discarded_source_rows": discarded_source_rows,
        "discarded_exact_pair_membership_sha256": base.membership_sha256(
            row["exact_pair_id_sha256"] for row in discarded_edges
        ),
        "discarded_source_membership_sha256": base.membership_sha256(
            row["source_instance_id"] for row in discarded_source_rows
        ),
    }

    retained_source_ids = sorted(
        instance_id
        for fold in retained_folds
        for instance_id in fold["retained_source_instance_ids"]
    )
    discarded_source_ids = sorted(
        row["source_instance_id"] for row in discarded_source_rows
    )
    all_source_ids = sorted(record[0] for record in records)
    retained_pair_ids = sorted(
        pair["exact_pair_id_sha256"]
        for fold in retained_folds
        for pair in fold["retained_exact_pairs"]
    )
    discarded_pair_ids = sorted(
        row["exact_pair_id_sha256"] for row in discarded_edges
    )
    all_pair_ids = sorted(
        _edge_id(antibody, antigen) for antibody, antigen in source_ids_by_pair
    )
    if set(retained_source_ids).intersection(discarded_source_ids):
        raise RuntimeError("Retained and discarded source-row ledgers overlap")
    if sorted(retained_source_ids + discarded_source_ids) != all_source_ids:
        raise RuntimeError("Source-row ledger is not exhaustive")
    if set(retained_pair_ids).intersection(discarded_pair_ids):
        raise RuntimeError("Retained and discarded exact-pair ledgers overlap")
    if sorted(retained_pair_ids + discarded_pair_ids) != all_pair_ids:
        raise RuntimeError("Exact-pair edge ledger is not exhaustive")
    accounting = {
        "valid": True,
        "all_authority_train_source_row_count": len(all_source_ids),
        "retained_source_row_count": len(retained_source_ids),
        "discarded_source_row_count": len(discarded_source_ids),
        "all_unique_exact_pair_count": len(all_pair_ids),
        "retained_unique_exact_pair_count": len(retained_pair_ids),
        "discarded_unique_exact_pair_count": len(discarded_pair_ids),
        "retained_and_discarded_source_rows_disjoint": True,
        "retained_and_discarded_source_rows_exhaustive": True,
        "retained_and_discarded_exact_pairs_disjoint": True,
        "retained_and_discarded_exact_pairs_exhaustive": True,
        "all_source_membership_sha256": base.membership_sha256(all_source_ids),
        "all_exact_pair_membership_sha256": base.membership_sha256(all_pair_ids),
    }
    return cluster_assignments, retained_folds, discarded_ledger, accounting


def _verify_inputs_unchanged(references: Mapping[str, Mapping[str, Any]]) -> None:
    """Rehash every bound input and source immediately before publication."""
    for label, reference in references.items():
        observed = base.sha256_file(Path(str(reference["path"])))
        if observed != reference["sha256"]:
            raise RuntimeError(f"Bound file changed during execution: {label}")


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
    parser.add_argument("--config-json", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--expected-builder-source-sha256", required=True)
    parser.add_argument("--expected-base-source-sha256", required=True)
    parser.add_argument("--expected-optimizer-source-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Build and atomically publish one provisional train-only S2 artifact."""
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
            "antibody MMseqs2 similarity TSV",
        ),
        "antigen_mmseqs2_similarity": base.verify_file(
            args.antigen_similarity_tsv,
            args.antigen_similarity_sha256,
            "antigen MMseqs2 similarity TSV",
        ),
        "builder_config": base.verify_file(
            args.config_json, args.config_sha256, "S2 builder config"
        ),
    }
    source_references = {
        "builder": base.verify_file(
            Path(__file__),
            args.expected_builder_source_sha256,
            "S2 builder source",
        ),
        "train_only_firewall_and_clustering": base.verify_file(
            Path(base.__file__),
            args.expected_base_source_sha256,
            "train-only firewall and clustering source",
        ),
        "milp_optimizer": base.verify_file(
            Path(optimizer.__file__),
            args.expected_optimizer_source_sha256,
            "S2 MILP optimizer source",
        ),
    }
    config = load_config(Path(input_references["builder_config"]["path"]))
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
        config["identity_threshold"],
        config["reciprocal_coverage_threshold"],
    )
    antigen_edges, antigen_similarity = base.load_similarity_relation(
        Path(input_references["antigen_mmseqs2_similarity"]["path"]),
        antigen_digests,
        config["identity_threshold"],
        config["reciprocal_coverage_threshold"],
    )
    antibody_clusters, antibody_cluster_summary = base.cluster_entities(
        antibody_digests, antibody_edges
    )
    antigen_clusters, antigen_cluster_summary = base.cluster_entities(
        antigen_digests, antigen_edges
    )
    problem = optimizer.build_problem(records, antibody_clusters, antigen_clusters)
    raw_solution, stages = optimizer.optimize(
        problem,
        config["fold_count"],
        config["axis_balance_tolerance"],
        config["time_limit_seconds"],
    )
    antibody_folds, antigen_folds = decode_cluster_folds(
        problem, raw_solution, config["fold_count"]
    )
    antibody_folds, antigen_folds, fold_relabel = canonicalize_fold_labels(
        problem,
        antibody_folds,
        antigen_folds,
        config["fold_count"],
    )
    canonical_solution = encode_solution(
        problem, antibody_folds, antigen_folds, config["fold_count"]
    )
    solution_summary = optimizer.summarize_solution(
        problem,
        canonical_solution,
        config["fold_count"],
        config["axis_balance_tolerance"],
    )
    (
        cluster_assignments,
        retained_folds,
        discarded_ledger,
        accounting,
    ) = build_exports(
        records,
        problem,
        antibody_clusters,
        antigen_clusters,
        antibody_folds,
        antigen_folds,
        config["fold_count"],
    )
    if accounting["retained_unique_exact_pair_count"] != solution_summary[
        "retained_unique_exact_pair_count"
    ]:
        raise RuntimeError("Retained exact-pair accounting disagrees with the MILP")
    if accounting["discarded_unique_exact_pair_count"] != solution_summary[
        "discarded_unique_exact_pair_count"
    ]:
        raise RuntimeError("Discarded exact-pair accounting disagrees with the MILP")
    if accounting["retained_source_row_count"] != solution_summary[
        "retained_source_row_count"
    ]:
        raise RuntimeError("Retained source-row accounting disagrees with the MILP")
    if accounting["discarded_source_row_count"] != solution_summary[
        "discarded_source_row_count"
    ]:
        raise RuntimeError("Discarded source-row accounting disagrees with the MILP")

    artifact_bindings = {
        "cluster_assignments_sha256": base.canonical_sha256(cluster_assignments),
        "retained_folds_sha256": base.canonical_sha256(
            {"retained_folds": retained_folds}
        ),
        "discarded_cross_fold_ledger_sha256": base.canonical_sha256(
            discarded_ledger
        ),
        "accounting_sha256": base.canonical_sha256(accounting),
    }
    payload: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "stage": "provisional_train_only_two_axis_s2_fold_assignment",
        "intended_use": "exact_four_view_audit_input_only",
        "approval_status": "unapproved_pending_exact_four_view_nw_audit",
        "valid_as_provisional_assignment": True,
        "authorized_for_training": False,
        "authorized_for_model_selection": False,
        "inputs": input_references,
        "source_files": source_references,
        "config": config,
        "config_payload_sha256": base.canonical_sha256(config),
        "authority": authority,
        "data_firewall": {
            **firewall,
            "authorized_source_split": "train",
            "validation_sequence_fields_parsed": 0,
            "test_sequence_fields_parsed": 0,
            "validation_or_test_labels_read": 0,
            "validation_or_test_predictions_read": 0,
            "validation_or_test_scores_read": 0,
            "accepted_label_prediction_checkpoint_or_score_inputs": False,
        },
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
        "entity_cluster_summary": {
            "antibody": antibody_cluster_summary,
            "antigen": antigen_cluster_summary,
        },
        "optimization_problem": {
            "antibody_cluster_count": len(problem.antibody_clusters),
            "antigen_cluster_count": len(problem.antigen_clusters),
            "cluster_pair_edge_count": len(problem.edges),
            "total_unique_exact_pair_count": problem.total_unique_pairs,
            "total_source_row_count": problem.total_source_rows,
            "objective": "maximize_retained_unique_exact_sequence_pairs",
            "constraints": (
                "one_fold_per_axis_cluster_AND_edge_retention_"
                "plus_two_axis_unique_pair_degree_balance"
            ),
        },
        "solver_result": stages["stage_1"],
        "determinism": {
            "policy": config["deterministic_policy"],
            "input_records_sorted": True,
            "similarity_edges_sorted": True,
            "cluster_representatives_lexical": True,
            "largest_antibody_cluster_pinned_by_optimizer": True,
            "solver_fold_labels_canonicalized": True,
            "old_to_canonical_fold_index": {
                str(old): new for old, new in sorted(fold_relabel.items())
            },
            "runtime_is_content_bound": True,
        },
        "solution_summary": solution_summary,
        "cluster_assignments": cluster_assignments,
        "retained_folds": retained_folds,
        "discarded_cross_fold_ledger": discarded_ledger,
        "ledger_accounting": accounting,
        "artifact_section_bindings": artifact_bindings,
        "required_downstream_gate": {
            "gate_name": "exact_four_view_needleman_wunsch_cross_fold_audit",
            "required_sequence_views": list(REQUIRED_SEQUENCE_VIEWS),
            "required_identity_threshold": config["identity_threshold"],
            "required_reciprocal_coverage_threshold": config[
                "reciprocal_coverage_threshold"
            ],
            "required_result": (
                "zero_cross_fold_pairs_at_or_above_threshold_in_every_view"
            ),
            "must_bind_this_artifact_file_sha256": True,
            "must_bind_this_artifact_scientific_payload_sha256": True,
            "approval_may_only_be_granted_by_separate_create_only_audit_receipt": True,
            "current_gate_passed": False,
        },
        "limitations": [
            "This artifact uses authority-train rows only and is not model evaluation.",
            "The optimized similarity views are concatenated antibody and antigen only.",
            "MMseqs2 threshold relations are approximate and may miss qualifying pairs.",
            "Exact VH, VL, concatenated-antibody, and antigen Needleman--Wunsch auditing is mandatory before any training or model-selection use.",
            "No authority-validation or frozen-test sequence, label, embedding, prediction, checkpoint, or score was materialized.",
        ],
    }
    payload["scientific_payload_sha256"] = base.canonical_sha256(payload)
    _verify_inputs_unchanged({**input_references, **source_references})
    atomic_create_json(args.output, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    """Run the command-line entrypoint."""
    payload = run(parse_args(argv))
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
