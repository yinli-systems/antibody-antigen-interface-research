#!/usr/bin/env python3
"""Build a provisional PINDER split from the complete four-view 80/80 closure."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import scipy

import audit_pinder_antibody_s2_exact_four_view_v1 as audit
import audit_pinder_antibody_s2_full_four_view_relation_v1 as full_relation
import build_pinder_antibody_s2_folds_v1 as fold_builder
import build_pinder_antibody_s2_train_validation_iteration_v2 as iteration
import build_pinder_antibody_s2_train_validation_v1 as initial_builder
import prove_pinder_antibody_s2_three_fold_infeasibility_v1 as closure


PROTOCOL_FORMAT_VERSION = "pinder_antibody_s2_train_validation_full_closure_protocol_v1"
REQUIRED_VIEWS = full_relation.REQUIRED_VIEWS
BUCKETS = initial_builder.BUCKETS


class FullClosureError(RuntimeError):
    """Raised when complete-closure provenance or split invariants fail."""


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = audit.strict_json(path)
    if protocol.get("format_version") != PROTOCOL_FORMAT_VERSION:
        raise FullClosureError("full-closure protocol format mismatch")
    if protocol.get("identity_threshold") != 0.8:
        raise FullClosureError("identity threshold must remain 0.8")
    if protocol.get("reciprocal_coverage_threshold") != 0.8:
        raise FullClosureError("coverage threshold must remain 0.8")
    if protocol.get("bucket_names") != list(BUCKETS):
        raise FullClosureError("bucket names changed")
    if protocol.get("validation_axis_degree_fraction") != {
        "minimum": 0.15,
        "maximum": 0.25,
    }:
        raise FullClosureError("validation axis-degree interval changed")
    if protocol.get("largest_antibody_component_bucket") != "train":
        raise FullClosureError("largest antibody component must remain in train")
    if protocol.get("threshold_relaxation_after_results") is not False:
        raise FullClosureError("post-result threshold relaxation is forbidden")
    sources = protocol.get("source_sha256")
    required_sources = {
        "canonical_manifest",
        "full_relation_result",
        *(f"{view}_complete_edges" for view in REQUIRED_VIEWS),
    }
    if not isinstance(sources, Mapping) or set(sources) != required_sources:
        raise FullClosureError("full-closure source bindings are incomplete")
    return protocol


def validate_full_relation_binding(
    result: Mapping[str, Any], canonical_manifest_sha256: str
) -> Mapping[str, Any]:
    if result.get("format_version") != full_relation.FORMAT_VERSION:
        raise FullClosureError("full relation format mismatch")
    if result.get("status") != (
        "PASS_COMPLETE_EXACT_FOUR_VIEW_S2_80_RELATION_MATERIALIZED"
    ):
        raise FullClosureError("full relation did not pass completeness gate")
    if result.get("authorized_for_training") is not False:
        raise FullClosureError("relation artifact unexpectedly authorized training")
    if result.get("authorized_for_model_selection") is not False:
        raise FullClosureError("relation artifact unexpectedly authorized selection")
    checks = result.get("checks")
    if not isinstance(checks, Mapping) or any(
        checks.get(name) is not True
        for name in (
            "all_four_views_complete",
            "all_unordered_unique_sequence_pairs_covered",
            "canonical_edge_orientation_is_digest_ascending",
        )
    ):
        raise FullClosureError("full relation completeness checks are missing")
    inputs = result.get("inputs")
    if not isinstance(inputs, Mapping) or inputs.get("canonical_manifest", {}).get(
        "sha256"
    ) != canonical_manifest_sha256:
        raise FullClosureError("full relation canonical manifest SHA mismatch")
    contracts = result.get("crossing_artifacts")
    if not isinstance(contracts, Mapping) or set(contracts) != set(REQUIRED_VIEWS):
        raise FullClosureError("full relation does not bind four edge ledgers")
    results = result.get("results")
    if not isinstance(results, Mapping) or set(results) != set(REQUIRED_VIEWS):
        raise FullClosureError("full relation does not contain four view results")
    for view in REQUIRED_VIEWS:
        if results[view].get("all_unordered_pairs_covered") is not True:
            raise FullClosureError(f"{view} full relation is incomplete")
        if contracts[view].get("edge_count") != results[view].get(
            "qualifying_unordered_edge_count"
        ):
            raise FullClosureError(f"{view} relation edge count mismatch")
    return contracts


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    protocol_reference = audit.verify_file(
        arguments.protocol, arguments.protocol_sha256, "full-closure protocol"
    )
    protocol = load_protocol(Path(protocol_reference["path"]))
    source_reference = audit.verify_file(
        Path(__file__), arguments.source_sha256, "full-closure builder source"
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
    relation = audit.strict_json(Path(relation_reference["path"]))
    contracts = validate_full_relation_binding(
        relation, canonical_reference["sha256"]
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
            raise FullClosureError(f"{view} edge SHA differs from relation result")
        edges_by_view[view] = edges
        edge_references[view] = reference

    antibody_entities = {row.antibody_digest for row in rows}
    antigen_entities = {row.antigen_digest for row in rows}
    antibody_roots, antigen_roots, exact_pairs, closure_evidence = (
        closure.build_augmented_components(
            rows,
            {digest: digest for digest in antibody_entities},
            {digest: digest for digest in antigen_entities},
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
    ledgers, _ = iteration._build_ledgers(
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
        raise FullClosureError("bucket system ledgers overlap")
    if set().union(*bucket_sets) != all_systems:
        raise FullClosureError("bucket system ledgers are not exhaustive")
    for key in ("exact_pair_ids", "pdb_ids"):
        sets = [set(ledgers[bucket][key]) for bucket in (*BUCKETS, "discarded")]
        if any(
            sets[left] & sets[right]
            for left in range(3)
            for right in range(left + 1, 3)
        ):
            raise FullClosureError(f"{key} are not bucket-indivisible")

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
        raise FullClosureError("validation axis degree is outside frozen bounds")

    immutable_references = {
        "protocol": protocol_reference,
        "builder_source": source_reference,
        "canonical_manifest": canonical_reference,
        "full_relation_result": relation_reference,
        **{
            f"{view}_complete_edges": edge_references[view]
            for view in REQUIRED_VIEWS
        },
    }
    for label, reference in immutable_references.items():
        if audit.sha256_file(Path(reference["path"])) != reference["sha256"]:
            raise FullClosureError(f"{label} changed during split construction")

    payload: dict[str, Any] = {
        "format_version": initial_builder.FORMAT_VERSION,
        "closure_iteration": "complete_relation_v1",
        "status": "PASS_PROVISIONAL_TRAIN_VALIDATION_S2_80_ASSIGNMENT",
        "approval_status": "UNAPPROVED_PENDING_EXACT_FOUR_VIEW_ZERO_CROSSING_AUDIT",
        "authorized_for_training": False,
        "authorized_for_model_selection": False,
        "inputs": immutable_references,
        "protocol": protocol,
        "closure_evidence": {
            **closure_evidence,
            "complete_relation_scientific_payload_sha256": relation.get(
                "scientific_payload_sha256"
            ),
            "complete_relation_guarantees_all_known_80_80_edges_indivisible": True,
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
            "complete_four_view_relation_merged_before_reassignment": True,
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
            "independent exact four-view Needleman-Wunsch zero-crossing audit"
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
    parser.add_argument("--full-relation-result", type=Path, required=True)
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
