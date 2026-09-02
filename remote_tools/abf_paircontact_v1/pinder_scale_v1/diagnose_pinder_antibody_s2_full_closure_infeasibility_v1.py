#!/usr/bin/env python3
"""Produce a solver-independent proof for the complete-closure split failure."""

from __future__ import annotations

import argparse
import json
import math
import platform
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import audit_pinder_antibody_s2_exact_four_view_v1 as audit
import audit_pinder_antibody_s2_full_four_view_relation_v1 as full_relation
import build_pinder_antibody_s2_train_validation_full_closure_v1 as builder
import build_pinder_antibody_s2_train_validation_v1 as split_builder
import prove_pinder_antibody_s2_three_fold_infeasibility_v1 as closure


FORMAT_VERSION = "pinder_antibody_s2_full_closure_infeasibility_receipt_v1"
REQUIRED_VIEWS = full_relation.REQUIRED_VIEWS


class DiagnosticError(RuntimeError):
    """Raised when a diagnostic input or proof invariant is invalid."""


def subset_sum_summary(
    roots: Mapping[str, str],
    exact_pairs: set[tuple[str, str]],
    axis: str,
    minimum: int,
    maximum: int,
    fix_largest_to_train: bool,
) -> dict[str, Any]:
    if axis not in {"antibody", "antigen"}:
        raise DiagnosticError("unknown axis")
    members: defaultdict[str, set[str]] = defaultdict(set)
    for entity, root in roots.items():
        members[root].add(entity)
    degrees = Counter(
        roots[antibody if axis == "antibody" else antigen]
        for antibody, antigen in exact_pairs
    )
    if not members or sum(degrees.values()) != len(exact_pairs):
        raise DiagnosticError(f"{axis} component-degree inventory is invalid")
    ranked = sorted(members, key=lambda root: (-degrees[root], root))
    fixed_root = ranked[0] if fix_largest_to_train else None
    candidates = [root for root in ranked if root != fixed_root]
    reachable = 1
    for root in candidates:
        reachable |= reachable << degrees[root]
    reachable_values = [
        value
        for value in range(len(exact_pairs) + 1)
        if (reachable >> value) & 1
    ]
    feasible_values = [
        value for value in reachable_values if minimum <= value <= maximum
    ]
    below = [value for value in reachable_values if value < minimum]
    above = [value for value in reachable_values if value > maximum]
    largest = ranked[0]
    return {
        "axis": axis,
        "entity_count": len(roots),
        "component_count": len(members),
        "singleton_component_count": sum(len(group) == 1 for group in members.values()),
        "largest_pair_degree_component": {
            "entity_count": len(members[largest]),
            "entity_fraction": len(members[largest]) / len(roots),
            "unique_exact_pair_degree": degrees[largest],
            "unique_exact_pair_degree_fraction": degrees[largest] / len(exact_pairs),
            "entity_membership_sha256": audit.membership_sha256(members[largest]),
            "fixed_to_train": largest == fixed_root,
        },
        "top_component_unique_exact_pair_degrees": [
            {"degree": degrees[root], "entity_count": len(members[root])}
            for root in ranked[:20]
        ],
        "validation_degree_bounds": {"minimum": minimum, "maximum": maximum},
        "feasible_subset_sum_exists": bool(feasible_values),
        "feasible_subset_sum_count": len(feasible_values),
        "minimum_feasible_subset_sum": min(feasible_values) if feasible_values else None,
        "maximum_feasible_subset_sum": max(feasible_values) if feasible_values else None,
        "nearest_reachable_below_minimum": max(below) if below else None,
        "nearest_reachable_above_maximum": min(above) if above else None,
        "reachable_subset_sum_count": len(reachable_values),
        "raw_sequences_or_system_ids_included": False,
    }


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    source_reference = audit.verify_file(
        Path(__file__), arguments.source_sha256, "diagnostic source"
    )
    canonical_reference = audit.verify_file(
        arguments.canonical_manifest,
        arguments.canonical_manifest_sha256,
        "canonical manifest",
    )
    relation_reference = audit.verify_file(
        arguments.full_relation_result,
        arguments.full_relation_result_sha256,
        "full relation result",
    )
    relation = audit.strict_json(Path(relation_reference["path"]))
    contracts = builder.validate_full_relation_binding(
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
            contract["sha256"],
            int(contract["edge_count"]),
            view_digests[view],
        )
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
    total = len(exact_pairs)
    minimum, maximum = math.ceil(0.15 * total), math.floor(0.25 * total)
    axis_results = {
        "antibody": subset_sum_summary(
            antibody_roots, exact_pairs, "antibody", minimum, maximum, True
        ),
        "antigen": subset_sum_summary(
            antigen_roots, exact_pairs, "antigen", minimum, maximum, False
        ),
    }
    if all(result["feasible_subset_sum_exists"] for result in axis_results.values()):
        raise DiagnosticError(
            "axis subset sums are independently feasible; this diagnostic does not prove infeasibility"
        )
    payload: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "status": "FAIL_FULL_CLOSURE_BINARY_SPLIT_INFEASIBLE",
        "authorized_for_training": False,
        "authorized_for_model_selection": False,
        "inputs": {
            "diagnostic_source": source_reference,
            "canonical_manifest": canonical_reference,
            "full_relation_result": relation_reference,
            **{
                f"{view}_complete_edges": edge_references[view]
                for view in REQUIRED_VIEWS
            },
        },
        "frozen_constraints": {
            "identity_threshold": 0.8,
            "reciprocal_coverage_threshold": 0.8,
            "validation_axis_degree_fraction": {"minimum": 0.15, "maximum": 0.25},
            "largest_antibody_component_bucket": "train",
        },
        "input_counts": {
            "systems": len(rows),
            "unique_exact_pairs": total,
            "unique_pdbs": len({row.pdb_id for row in rows}),
        },
        "closure_evidence": closure_evidence,
        "axis_results": axis_results,
        "proof": {
            "method": "exact integer subset-sum enumeration over complete 80/80 component degrees",
            "logic": "Each complete-relation component is indivisible. If no subset of component degrees lies in the frozen validation interval on either axis, no binary assignment can satisfy the frozen split constraints.",
            "at_least_one_axis_has_no_feasible_validation_subset": True,
        },
        "data_firewall": {
            "source_pool": "PINDER_train_only",
            "external_validation_rows_read": False,
            "test_rows_read": False,
            "task_labels_read": False,
            "structures_read": False,
            "predictions_or_scores_read": False,
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "required_next_action": "Freeze this infeasibility result and design a new leakage-safe split protocol without treating the rejected 15-25 percent design as authorized.",
        "claim_boundary": "This receipt proves infeasibility of the frozen binary 15-25 percent axis-degree design under the complete exact four-view 80/80 closure. It does not authorize training, threshold relaxation, test access, or any performance claim.",
    }
    payload["scientific_payload_sha256"] = split_builder.canonical_sha256(payload)
    split_builder._atomic_create(arguments.output, audit.canonical_json_bytes(payload))
    return payload


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--canonical-manifest", type=Path, required=True)
    parser.add_argument("--canonical-manifest-sha256", required=True)
    parser.add_argument("--full-relation-result", type=Path, required=True)
    parser.add_argument("--full-relation-result-sha256", required=True)
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
                "scientific_payload_sha256": result["scientific_payload_sha256"],
                "axis_results": result["axis_results"],
            },
            indent=2,
            sort_keys=True,
        )
    )
