#!/usr/bin/env python3
"""Rebuild a provisional PINDER train/validation split from one failed audit.

This is the closure-iteration companion to
``build_pinder_antibody_s2_train_validation_v1.py``.  It consumes the previous
provisional assignment plus the digest-only crossing ledgers emitted by its
independent exact four-view audit, merges those relations into indivisible
components, and reruns the unchanged binary MILP.  It never reads structures,
labels, model outputs, validation data, or test data, and it never relaxes the
inclusive 80/80 rule.
"""

from __future__ import annotations

import argparse
import json
import platform
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import scipy

import audit_pinder_antibody_s2_exact_four_view_v1 as audit
import audit_pinder_antibody_s2_train_validation_exact_v1 as exact_audit
import build_pinder_antibody_s2_folds_v1 as fold_builder
import build_pinder_antibody_s2_train_validation_v1 as initial_builder
import prove_pinder_antibody_s2_three_fold_infeasibility_v1 as closure


PROTOCOL_FORMAT_VERSION = (
    "pinder_antibody_s2_train_validation_iteration_protocol_v2"
)
REQUIRED_VIEWS = exact_audit.REQUIRED_VIEWS
BUCKETS = initial_builder.BUCKETS


class IterationError(RuntimeError):
    """Raised when an iteration input, binding, or solver invariant is invalid."""


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = audit.strict_json(path)
    if protocol.get("format_version") != PROTOCOL_FORMAT_VERSION:
        raise IterationError("iteration protocol format mismatch")
    iteration = protocol.get("closure_iteration")
    if not isinstance(iteration, int) or isinstance(iteration, bool) or iteration < 2:
        raise IterationError("closure iteration must be an integer >= 2")
    if protocol.get("identity_threshold") != 0.8:
        raise IterationError("identity threshold must remain 0.8")
    if protocol.get("reciprocal_coverage_threshold") != 0.8:
        raise IterationError("coverage threshold must remain 0.8")
    if protocol.get("bucket_names") != list(BUCKETS):
        raise IterationError("bucket names changed")
    if protocol.get("validation_axis_degree_fraction") != {
        "minimum": 0.15,
        "maximum": 0.25,
    }:
        raise IterationError("validation axis-degree interval changed")
    if protocol.get("largest_antibody_component_bucket") != "train":
        raise IterationError("largest antibody component must remain in train")
    if protocol.get("threshold_relaxation_after_results") is not False:
        raise IterationError("post-result threshold relaxation is forbidden")
    sources = protocol.get("source_sha256")
    required_sources = {
        "canonical_manifest",
        "previous_assignment",
        "previous_assignment_scientific_payload",
        "previous_audit_result",
        *(f"{view}_crossing_edges" for view in REQUIRED_VIEWS),
    }
    if not isinstance(sources, Mapping) or set(sources) != required_sources:
        raise IterationError("iteration source bindings are incomplete")
    return protocol


def load_previous_component_roots(
    assignment: Mapping[str, Any],
    antibody_entities: set[str],
    antigen_entities: set[str],
) -> tuple[dict[str, str], dict[str, str]]:
    assignments = assignment.get("entity_assignments")
    if not isinstance(assignments, Mapping):
        raise IterationError("previous entity assignments are missing")

    def load_axis(axis: str, expected: set[str]) -> dict[str, str]:
        records = assignments.get(axis)
        if not isinstance(records, list):
            raise IterationError(f"previous {axis} assignments are missing")
        roots: dict[str, str] = {}
        for record in records:
            if not isinstance(record, Mapping):
                raise IterationError(f"previous {axis} assignment is not an object")
            digest = record.get("sequence_sha256")
            root = record.get("component_root_sha256")
            bucket = record.get("bucket")
            if not isinstance(digest, str) or not isinstance(root, str):
                raise IterationError(f"previous {axis} component binding is invalid")
            if bucket not in BUCKETS:
                raise IterationError(f"previous {axis} bucket is invalid")
            if digest in roots:
                raise IterationError(f"duplicate previous {axis} assignment")
            roots[digest] = root
        if set(roots) != expected:
            raise IterationError(f"previous {axis} assignment inventory mismatch")
        return roots

    return (
        load_axis("antibody", antibody_entities),
        load_axis("antigen", antigen_entities),
    )


def validate_failed_audit_binding(
    audit_result: Mapping[str, Any],
    previous_assignment_sha256: str,
    previous_assignment_scientific_sha256: str,
    canonical_manifest_sha256: str,
) -> Mapping[str, Any]:
    if audit_result.get("status") != (
        "FAIL_EXACT_FOUR_VIEW_TRAIN_VALIDATION_S2_80_CROSSINGS_DETECTED"
    ):
        raise IterationError("previous exact audit is not a failed boundary audit")
    if audit_result.get("approval_status") != "NOT_APPROVED_REBUILD_WITH_NEW_EXACT_EDGES":
        raise IterationError("previous exact audit decision is not rebuild")
    if audit_result.get("authorized_for_training") is not False:
        raise IterationError("failed audit unexpectedly authorized training")
    if audit_result.get("authorized_for_model_selection") is not False:
        raise IterationError("failed audit unexpectedly authorized selection")
    if audit_result.get("assignment_scientific_payload_sha256") != (
        previous_assignment_scientific_sha256
    ):
        raise IterationError("failed audit binds another scientific assignment")
    inputs = audit_result.get("inputs")
    if not isinstance(inputs, Mapping):
        raise IterationError("failed audit input bindings are missing")
    if inputs.get("assignment", {}).get("sha256") != previous_assignment_sha256:
        raise IterationError("failed audit assignment SHA mismatch")
    if inputs.get("canonical_manifest", {}).get("sha256") != canonical_manifest_sha256:
        raise IterationError("failed audit canonical manifest SHA mismatch")
    contracts = audit_result.get("crossing_artifacts")
    if not isinstance(contracts, Mapping) or set(contracts) != set(REQUIRED_VIEWS):
        raise IterationError("failed audit does not bind four crossing ledgers")
    total = audit_result.get("exhaustive_accounting", {}).get(
        "total_threshold_crossing_edge_count"
    )
    if not isinstance(total, int) or isinstance(total, bool) or total <= 0:
        raise IterationError("failed audit has no positive crossing count")
    return contracts


def _build_ledgers(
    rows: Sequence[fold_builder.SourceRow],
    component_by_system: Mapping[str, str],
    components: Sequence[fold_builder.Component],
    solution: np.ndarray,
    index: initial_builder.VariableIndex,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    component_bucket: dict[str, str] = {}
    for number, component in enumerate(components):
        train = solution[index.component_train(number)] > 0.5
        validation = solution[index.component_validation(number)] > 0.5
        if train and validation:
            raise IterationError("component retained in both buckets")
        if train:
            component_bucket[component.component_id] = "train"
        elif validation:
            component_bucket[component.component_id] = "validation"

    roots = sorted({component_by_system[row.system_id] for row in rows})
    if len(roots) != len(components):
        raise IterationError("interaction component root count mismatch")
    id_by_root = {
        root: component.component_id for root, component in zip(roots, components)
    }
    bucket_rows: defaultdict[str, list[fold_builder.SourceRow]] = defaultdict(list)
    for row in rows:
        component_id = id_by_root[component_by_system[row.system_id]]
        bucket_rows[component_bucket.get(component_id, "discarded")].append(row)
    ledgers = {
        bucket: initial_builder._bucket_ledger(bucket, bucket_rows[bucket])
        for bucket in (*BUCKETS, "discarded")
    }
    return ledgers, component_bucket


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    protocol_reference = audit.verify_file(
        arguments.protocol, arguments.protocol_sha256, "iteration protocol"
    )
    protocol = load_protocol(Path(protocol_reference["path"]))
    source_reference = audit.verify_file(
        Path(__file__), arguments.source_sha256, "iteration builder source"
    )
    sources = protocol["source_sha256"]
    canonical_reference = audit.verify_file(
        arguments.canonical_manifest,
        sources["canonical_manifest"],
        "canonical manifest",
    )
    previous_assignment_reference = audit.verify_file(
        arguments.previous_assignment,
        sources["previous_assignment"],
        "previous assignment",
    )
    previous_audit_reference = audit.verify_file(
        arguments.previous_audit_result,
        sources["previous_audit_result"],
        "previous exact audit",
    )

    rows = audit.load_source_rows(Path(canonical_reference["path"]))
    previous_assignment = audit.strict_json(Path(previous_assignment_reference["path"]))
    bucket_ids, _ = exact_audit.validate_assignment(
        previous_assignment,
        rows,
        sources["previous_assignment_scientific_payload"],
    )
    if not bucket_ids["train"] or not bucket_ids["validation"]:
        raise IterationError("previous assignment has an empty retained bucket")
    previous_audit = audit.strict_json(Path(previous_audit_reference["path"]))
    contracts = validate_failed_audit_binding(
        previous_audit,
        previous_assignment_reference["sha256"],
        sources["previous_assignment_scientific_payload"],
        canonical_reference["sha256"],
    )

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
            sources[f"{view}_crossing_edges"],
            int(contract["edge_count"]),
            view_digests[view],
        )
        if reference["sha256"] != contract["sha256"]:
            raise IterationError(f"{view} edge SHA differs from failed audit")
        edges_by_view[view] = edges
        edge_references[view] = reference
    if sum(len(edges) for edges in edges_by_view.values()) != previous_audit[
        "exhaustive_accounting"
    ]["total_threshold_crossing_edge_count"]:
        raise IterationError("four-view crossing total mismatch")

    antibody_entities = {row.antibody_digest for row in rows}
    antigen_entities = {row.antigen_digest for row in rows}
    previous_antibody_roots, previous_antigen_roots = load_previous_component_roots(
        previous_assignment, antibody_entities, antigen_entities
    )
    antibody_roots, antigen_roots, exact_pairs, closure_evidence = (
        closure.build_augmented_components(
            rows,
            previous_antibody_roots,
            previous_antigen_roots,
            edges_by_view,
        )
    )
    solver_rows = initial_builder._base_rows(rows)
    components, _, component_by_system = fold_builder.interaction_components(
        solver_rows, antibody_roots, antigen_roots
    )
    balance = protocol["validation_axis_degree_fraction"]
    solution, solver_result, antibody_clusters, antigen_clusters = (
        initial_builder.solve_binary(
            solver_rows,
            antibody_roots,
            antigen_roots,
            components,
            balance["minimum"],
            balance["maximum"],
            protocol["solver"]["time_limit_seconds"],
        )
    )
    index = initial_builder.VariableIndex(
        len(antibody_clusters), len(antigen_clusters), len(components)
    )
    ledgers, _ = _build_ledgers(
        solver_rows, component_by_system, components, solution, index
    )
    all_systems = {row.system_id for row in solver_rows}
    bucket_sets = [
        set(ledgers[bucket]["system_ids"])
        for bucket in (*BUCKETS, "discarded")
    ]
    if any(
        bucket_sets[left] & bucket_sets[right]
        for left in range(3)
        for right in range(left + 1, 3)
    ):
        raise IterationError("bucket system ledgers overlap")
    if set().union(*bucket_sets) != all_systems:
        raise IterationError("bucket system ledgers are not exhaustive")
    for key in ("exact_pair_ids", "pdb_ids"):
        sets = [set(ledgers[bucket][key]) for bucket in (*BUCKETS, "discarded")]
        if any(
            sets[left] & sets[right]
            for left in range(3)
            for right in range(left + 1, 3)
        ):
            raise IterationError(f"{key} are not bucket-indivisible")

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
    axis_degree_counts = {
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
        for value in axis_degree_counts.values()
    ):
        raise IterationError("validation axis degree is outside frozen bounds")

    immutable_references = {
        "protocol": protocol_reference,
        "builder_source": source_reference,
        "canonical_manifest": canonical_reference,
        "previous_assignment": previous_assignment_reference,
        "previous_exact_audit_result": previous_audit_reference,
        **{
            f"{view}_crossing_edges": edge_references[view]
            for view in REQUIRED_VIEWS
        },
    }
    for label, reference in immutable_references.items():
        if audit.sha256_file(Path(reference["path"])) != reference["sha256"]:
            raise IterationError(f"{label} changed during split construction")

    payload: dict[str, Any] = {
        "format_version": initial_builder.FORMAT_VERSION,
        "closure_iteration": protocol["closure_iteration"],
        "status": "PASS_PROVISIONAL_TRAIN_VALIDATION_S2_80_ASSIGNMENT",
        "approval_status": "UNAPPROVED_PENDING_EXACT_FOUR_VIEW_ZERO_CROSSING_AUDIT",
        "authorized_for_training": False,
        "authorized_for_model_selection": False,
        "inputs": immutable_references,
        "protocol": protocol,
        "closure_evidence": {
            **closure_evidence,
            "previous_assignment_scientific_payload_sha256": sources[
                "previous_assignment_scientific_payload"
            ],
            "previous_audit_scientific_payload_sha256": previous_audit.get(
                "scientific_payload_sha256"
            ),
        },
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
        "axis_degree_counts": axis_degree_counts,
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
            "new_crossing_edges_merged_before_reassignment": True,
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
            "if crossings remain, repeat this authenticated closure iteration "
            "without changing the 80/80 rule or reading labels/scores"
        ),
        "claim_boundary": protocol["claim_boundary"],
    }
    payload["scientific_payload_sha256"] = initial_builder.canonical_sha256(payload)
    initial_builder._atomic_create(
        arguments.output, audit.canonical_json_bytes(payload)
    )
    return payload


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--canonical-manifest", type=Path, required=True)
    parser.add_argument("--previous-assignment", type=Path, required=True)
    parser.add_argument("--previous-audit-result", type=Path, required=True)
    parser.add_argument("--vh-edges", type=Path, required=True)
    parser.add_argument("--vl-edges", type=Path, required=True)
    parser.add_argument("--concatenated-antibody-edges", type=Path, required=True)
    parser.add_argument("--antigen-edges", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    for name in ("protocol_sha256", "source_sha256"):
        value = getattr(arguments, name)
        if len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            parser.error(f"{name} is not a lowercase SHA-256")
    return arguments


if __name__ == "__main__":
    result = run(parse_arguments())
    print(
        json.dumps(
            {
                "status": result["status"],
                "closure_iteration": result["closure_iteration"],
                "train_systems": result["bucket_ledgers"]["train"]["system_count"],
                "validation_systems": result["bucket_ledgers"]["validation"][
                    "system_count"
                ],
                "discarded_systems": result["bucket_ledgers"]["discarded"][
                    "system_count"
                ],
                "retained_exact_pairs": result["solver_result"][
                    "optimal_retained_unique_exact_pair_count"
                ],
                "scientific_payload_sha256": result["scientific_payload_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
