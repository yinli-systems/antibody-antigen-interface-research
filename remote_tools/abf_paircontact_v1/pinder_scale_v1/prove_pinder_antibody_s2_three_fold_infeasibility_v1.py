#!/usr/bin/env python3
"""Prove the frozen balanced three-fold PINDER-antibody S2 design infeasible.

The proof uses only train-authority sequences and the digest-only crossing edges
published by the independent exact four-view audit.  Those edges are a lower
bound on the complete 80/80 relation: adding further exact edges can merge
components but cannot reduce their unique-pair degree.  Consequently, one
known antibody component whose degree exceeds the frozen per-fold upper bound
is a sufficient, solver-free infeasibility witness.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import audit_pinder_antibody_s2_exact_four_view_v1 as audit


FORMAT_VERSION = "pinder_antibody_s2_three_fold_infeasibility_receipt_v1"
REQUIRED_VIEWS = ("vh", "vl", "concatenated_antibody", "antigen")


class ProofError(RuntimeError):
    """Raised when a proof input or invariant is invalid."""


class UnionFind:
    """Deterministic lexical-root disjoint-set structure."""

    def __init__(self, values: Iterable[str]) -> None:
        identifiers = sorted(set(values))
        if not identifiers:
            raise ProofError("empty union-find inventory")
        self.parent = {value: value for value in identifiers}

    def find(self, value: str) -> str:
        if value not in self.parent:
            raise ProofError(f"unknown union-find identifier: {value}")
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        smaller, larger = sorted((left_root, right_root))
        self.parent[larger] = smaller


def _union_group(union_find: UnionFind, values: Iterable[str]) -> None:
    members = sorted(set(values))
    if not members:
        raise ProofError("cannot union an empty group")
    for member in members[1:]:
        union_find.union(members[0], member)


def _strict_json_line(line: str, label: str) -> dict[str, Any]:
    def reject_duplicate(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for key, value in pairs:
            if key in payload:
                raise ProofError(f"duplicate JSON key in {label}: {key}")
            payload[key] = value
        return payload

    def reject_constant(value: str) -> None:
        raise ProofError(f"non-finite JSON constant in {label}: {value}")

    payload = json.loads(
        line,
        object_pairs_hook=reject_duplicate,
        parse_constant=reject_constant,
    )
    if not isinstance(payload, dict):
        raise ProofError(f"{label} is not an object")
    return payload


def load_parent_cluster_roots(
    assignment: Mapping[str, Any],
    antibody_entities: set[str],
    antigen_entities: set[str],
) -> tuple[dict[str, str], dict[str, str]]:
    assignments = assignment.get("cluster_assignments")
    if not isinstance(assignments, Mapping):
        raise ProofError("parent cluster assignments are missing")

    def load_axis(axis: str, expected: set[str]) -> dict[str, str]:
        rows = assignments.get(axis)
        if not isinstance(rows, list):
            raise ProofError(f"parent {axis} assignments are missing")
        result: dict[str, str] = {}
        for row in rows:
            if not isinstance(row, Mapping):
                raise ProofError(f"parent {axis} assignment is not an object")
            digest = row.get("sequence_sha256")
            root = row.get("cluster_representative_sequence_sha256")
            if not isinstance(digest, str) or not isinstance(root, str):
                raise ProofError(f"parent {axis} assignment digest is invalid")
            if digest in result:
                raise ProofError(f"duplicate parent {axis} assignment")
            result[digest] = root
        if set(result) != expected:
            raise ProofError(f"parent {axis} assignment inventory mismatch")
        return result

    return (
        load_axis("antibody", antibody_entities),
        load_axis("antigen", antigen_entities),
    )


def load_exact_edges(
    path: Path,
    view: str,
    expected_sha256: str,
    expected_count: int,
    allowed_digests: set[str],
) -> tuple[set[tuple[str, str]], dict[str, Any]]:
    reference = audit.verify_file(path, expected_sha256, f"{view} crossing edges")
    edges: set[tuple[str, str]] = set()
    with Path(reference["path"]).open("r", encoding="ascii") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = _strict_json_line(line, f"{view} edge {line_number}")
            if row.get("format_version") != audit.EDGE_FORMAT_VERSION:
                raise ProofError(f"{view} edge format mismatch")
            if row.get("sequence_view") != view:
                raise ProofError(f"{view} edge declares another view")
            if row.get("locked_threshold_rule_met") is not True:
                raise ProofError(f"{view} edge did not meet the locked threshold")
            if row.get("raw_sequences_or_system_ids_included") is not False:
                raise ProofError(f"{view} edge is not digest-only")
            left = row.get("left_sequence_sha256")
            right = row.get("right_sequence_sha256")
            if left not in allowed_digests or right not in allowed_digests:
                raise ProofError(f"{view} edge endpoint is absent from train authority")
            alignment = row.get("alignment")
            if not isinstance(alignment, Mapping):
                raise ProofError(f"{view} edge alignment is missing")
            paired = alignment.get("aligned_residue_pair_count")
            identical = alignment.get("identical_residue_pair_count")
            left_length = row.get("left_length")
            right_length = row.get("right_length")
            if not all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in (paired, identical, left_length, right_length)
            ):
                raise ProofError(f"{view} edge integer metrics are invalid")
            if not (
                5 * identical >= 4 * paired
                and 5 * paired >= 4 * left_length
                and 5 * paired >= 4 * right_length
            ):
                raise ProofError(f"{view} edge fails the inclusive 80/80 rule")
            edge = tuple(sorted((left, right)))
            if edge in edges:
                raise ProofError(f"duplicate {view} crossing edge")
            edges.add(edge)
    if len(edges) != expected_count:
        raise ProofError(
            f"{view} edge count mismatch: expected={expected_count}, observed={len(edges)}"
        )
    return edges, {**reference, "edge_count": len(edges), "sequence_view": view}


def build_augmented_components(
    rows: Sequence[audit.SourceRow],
    parent_antibody_root: Mapping[str, str],
    parent_antigen_root: Mapping[str, str],
    edges_by_view: Mapping[str, set[tuple[str, str]]],
) -> tuple[dict[str, str], dict[str, str], set[tuple[str, str]], dict[str, Any]]:
    antibody_entities = {row.antibody_digest for row in rows}
    antigen_entities = {row.antigen_digest for row in rows}
    antibody_union = UnionFind(antibody_entities)
    antigen_union = UnionFind(antigen_entities)

    base_antibody_groups: defaultdict[str, set[str]] = defaultdict(set)
    base_antigen_groups: defaultdict[str, set[str]] = defaultdict(set)
    for digest, root in parent_antibody_root.items():
        base_antibody_groups[root].add(digest)
    for digest, root in parent_antigen_root.items():
        base_antigen_groups[root].add(digest)
    for members in base_antibody_groups.values():
        _union_group(antibody_union, members)
    for members in base_antigen_groups.values():
        _union_group(antigen_union, members)

    concat_by_vh: defaultdict[str, set[str]] = defaultdict(set)
    concat_by_vl: defaultdict[str, set[str]] = defaultdict(set)
    for row in rows:
        concat_by_vh[audit.sequence_sha256(row.vh)].add(row.antibody_digest)
        if row.vl:
            concat_by_vl[audit.sequence_sha256(row.vl)].add(row.antibody_digest)
    for mapping in (concat_by_vh, concat_by_vl):
        for members in mapping.values():
            _union_group(antibody_union, members)

    for view, mapping in (("vh", concat_by_vh), ("vl", concat_by_vl)):
        for left, right in sorted(edges_by_view[view]):
            left_entities = mapping.get(left)
            right_entities = mapping.get(right)
            if not left_entities or not right_entities:
                raise ProofError(f"{view} edge cannot be lifted to antibody entities")
            _union_group(antibody_union, left_entities)
            _union_group(antibody_union, right_entities)
            antibody_union.union(min(left_entities), min(right_entities))
    for left, right in sorted(edges_by_view["concatenated_antibody"]):
        antibody_union.union(left, right)
    for left, right in sorted(edges_by_view["antigen"]):
        antigen_union.union(left, right)

    antibody_roots = {
        digest: antibody_union.find(digest) for digest in sorted(antibody_entities)
    }
    antigen_roots = {
        digest: antigen_union.find(digest) for digest in sorted(antigen_entities)
    }
    exact_pairs = {(row.antibody_digest, row.antigen_digest) for row in rows}
    evidence = {
        "base_component_count": {
            "antibody": len(base_antibody_groups),
            "antigen": len(base_antigen_groups),
        },
        "exact_chain_digest_group_count": {
            "vh": len(concat_by_vh),
            "nonempty_vl": len(concat_by_vl),
        },
        "ingested_crossing_edge_count": {
            view: len(edges_by_view[view]) for view in REQUIRED_VIEWS
        },
        "known_edges_are_a_lower_bound_on_complete_exact_relation": True,
    }
    return antibody_roots, antigen_roots, exact_pairs, evidence


def summarize_axis(
    roots: Mapping[str, str],
    exact_pairs: set[tuple[str, str]],
    axis: str,
) -> tuple[dict[str, Any], set[str], int]:
    members: defaultdict[str, set[str]] = defaultdict(set)
    for entity, root in roots.items():
        members[root].add(entity)
    degrees: Counter[str] = Counter()
    for antibody, antigen in exact_pairs:
        entity = antibody if axis == "antibody" else antigen
        degrees[roots[entity]] += 1
    ranked = sorted(members, key=lambda root: (-degrees[root], -len(members[root]), root))
    witness = ranked[0]
    return (
        {
            "axis": axis,
            "entity_count": len(roots),
            "component_count": len(members),
            "singleton_component_count": sum(len(group) == 1 for group in members.values()),
            "largest_pair_degree_component": {
                "entity_count": len(members[witness]),
                "entity_fraction": len(members[witness]) / len(roots),
                "unique_exact_pair_degree": degrees[witness],
                "unique_exact_pair_degree_fraction": degrees[witness] / len(exact_pairs),
                "entity_membership_sha256": audit.membership_sha256(members[witness]),
            },
            "top_component_unique_exact_pair_degrees": [
                {"degree": degrees[root], "entity_count": len(members[root])}
                for root in ranked[:20]
            ],
            "raw_sequences_or_system_ids_included": False,
        },
        members[witness],
        degrees[witness],
    )


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
    source_reference = audit.verify_file(
        Path(__file__), arguments.source_sha256, "proof source"
    )
    canonical_reference = audit.verify_file(
        arguments.canonical_manifest,
        arguments.canonical_manifest_sha256,
        "canonical manifest",
    )
    assignment_reference = audit.verify_file(
        arguments.assignment, arguments.assignment_sha256, "parent assignment"
    )
    audit_reference = audit.verify_file(
        arguments.audit_result, arguments.audit_result_sha256, "parent exact audit"
    )
    rows = audit.load_source_rows(Path(canonical_reference["path"]))
    assignment = audit.strict_json(Path(assignment_reference["path"]))
    audit.validate_assignment(
        assignment, rows, arguments.assignment_scientific_sha256
    )
    parent_audit = audit.strict_json(Path(audit_reference["path"]))
    if parent_audit.get("status") != "FAIL_EXACT_FOUR_VIEW_S2_80_CROSSINGS_DETECTED":
        raise ProofError("parent exact audit is not the required failed audit")
    if parent_audit.get("authorized_for_training") is not False:
        raise ProofError("parent exact audit unexpectedly authorized training")
    if parent_audit.get("assignment_scientific_payload_sha256") != (
        arguments.assignment_scientific_sha256
    ):
        raise ProofError("parent exact audit binds another assignment")
    if parent_audit.get("inputs", {}).get("assignment", {}).get("sha256") != (
        arguments.assignment_sha256
    ):
        raise ProofError("parent exact audit assignment SHA mismatch")
    if parent_audit.get("inputs", {}).get("canonical_manifest", {}).get("sha256") != (
        arguments.canonical_manifest_sha256
    ):
        raise ProofError("parent exact audit canonical manifest SHA mismatch")

    view_digests = {
        "vh": {audit.sequence_sha256(row.vh) for row in rows},
        "vl": {audit.sequence_sha256(row.vl) for row in rows if row.vl},
        "concatenated_antibody": {row.antibody_digest for row in rows},
        "antigen": {row.antigen_digest for row in rows},
    }
    contracts = parent_audit.get("crossing_artifacts")
    if not isinstance(contracts, Mapping) or set(contracts) != set(REQUIRED_VIEWS):
        raise ProofError("parent exact audit does not bind four crossing artifacts")
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
        if not isinstance(contract, Mapping):
            raise ProofError(f"invalid parent {view} artifact contract")
        edges, reference = load_exact_edges(
            edge_paths[view],
            view,
            str(contract.get("sha256")),
            int(contract.get("edge_count")),
            view_digests[view],
        )
        edges_by_view[view] = edges
        edge_references[view] = reference
    claimed_total = parent_audit.get("exhaustive_accounting", {}).get(
        "total_threshold_crossing_edge_count"
    )
    if claimed_total != sum(len(edges) for edges in edges_by_view.values()):
        raise ProofError("parent four-view crossing total mismatch")

    antibody_entities = {row.antibody_digest for row in rows}
    antigen_entities = {row.antigen_digest for row in rows}
    parent_antibody_root, parent_antigen_root = load_parent_cluster_roots(
        assignment, antibody_entities, antigen_entities
    )
    antibody_roots, antigen_roots, exact_pairs, closure_evidence = (
        build_augmented_components(
            rows,
            parent_antibody_root,
            parent_antigen_root,
            edges_by_view,
        )
    )
    antibody_summary, antibody_witness, antibody_degree = summarize_axis(
        antibody_roots, exact_pairs, "antibody"
    )
    antigen_summary, antigen_witness, antigen_degree = summarize_axis(
        antigen_roots, exact_pairs, "antigen"
    )

    protocol = assignment.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ProofError("parent split protocol is missing")
    fold_count = protocol.get("balance", {}).get("fold_count")
    tolerance = protocol.get("balance", {}).get("axis_unique_pair_degree_tolerance")
    if fold_count != 3 or tolerance != 0.4:
        raise ProofError("parent frozen balance contract changed")
    target = len(exact_pairs) / fold_count
    upper = math.floor((1.0 + tolerance) * target)
    lower = math.ceil((1.0 - tolerance) * target)
    witnesses = [
        ("antibody", antibody_witness, antibody_degree),
        ("antigen", antigen_witness, antigen_degree),
    ]
    violating = [item for item in witnesses if item[2] > upper]
    if not violating:
        raise ProofError("known exact closure does not prove three-fold infeasibility")
    witness_axis, witness_members, witness_degree = max(
        violating, key=lambda item: (item[2], item[0])
    )

    immutable_references = {
        "proof_source": source_reference,
        "canonical_manifest": canonical_reference,
        "parent_assignment": assignment_reference,
        "parent_exact_audit": audit_reference,
        **{f"{view}_crossing_edges": edge_references[view] for view in REQUIRED_VIEWS},
    }
    for label, reference in immutable_references.items():
        if audit.sha256_file(Path(reference["path"])) != reference["sha256"]:
            raise ProofError(f"{label} changed during proof construction")

    receipt: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "status": "PASS_THREE_FOLD_BALANCED_S2_INFEASIBILITY_PROOF",
        "valid_infeasibility_proof": True,
        "strict_balanced_three_fold_design_feasible": False,
        "solver_launched": False,
        "assignment_exported": False,
        "authorized_for_training": False,
        "authorized_for_model_selection": False,
        "decision": {
            "stop_failed_three_fold_design": True,
            "identity_or_coverage_threshold_relaxed": False,
            "balance_tolerance_relaxed": False,
            "silent_retry_or_alternative_substitution": False,
            "required_next_step": (
                "freeze a separate train-validation two-axis design, then rerun an "
                "independent exact four-view zero-crossing audit"
            ),
        },
        "inputs": immutable_references,
        "parent_assignment_scientific_payload_sha256": (
            arguments.assignment_scientific_sha256
        ),
        "closure_evidence": closure_evidence,
        "augmented_components": {
            "antibody": antibody_summary,
            "antigen": antigen_summary,
        },
        "frozen_balance_contract": {
            "fold_count": fold_count,
            "original_unique_exact_pair_count": len(exact_pairs),
            "equal_fold_target": target,
            "tolerance": tolerance,
            "integer_degree_lower_bound_inclusive": lower,
            "integer_degree_upper_bound_inclusive": upper,
        },
        "mathematical_proof": {
            "necessary_condition": (
                "Every exact four-view component must be assigned wholly to one fold; "
                "therefore its assigned fold has axis degree at least the component degree."
            ),
            "monotonicity": (
                "The audited crossing edges are a subset of the complete exact relation. "
                "Additional exact edges can only merge components and cannot reduce this degree."
            ),
            "witness_axis": witness_axis,
            "witness_component_entity_count": len(witness_members),
            "witness_component_membership_sha256": audit.membership_sha256(
                witness_members
            ),
            "witness_component_unique_exact_pair_degree": witness_degree,
            "locked_integer_fold_upper_bound": upper,
            "contradictory_inequality": f"{witness_degree} > {upper}",
            "conclusion": (
                "No assignment satisfying the frozen three-fold balance contract can "
                "also keep all known exact four-view 80/80 components indivisible."
            ),
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "data_firewall": {
            "source_pool": "PINDER_train_only",
            "validation_rows_read": False,
            "test_rows_read": False,
            "task_labels_read": False,
            "structures_read": False,
            "predictions_or_scores_read": False,
        },
        "claim_boundary": (
            "This is a split-design infeasibility proof, not a biological result, "
            "performance evaluation, temporal test, or PLM leakage audit."
        ),
    }
    receipt["scientific_payload_sha256"] = audit.canonical_payload_sha256(receipt)
    _atomic_create(arguments.output, audit.canonical_json_bytes(receipt))
    return receipt


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-manifest", type=Path, required=True)
    parser.add_argument("--canonical-manifest-sha256", required=True)
    parser.add_argument("--assignment", type=Path, required=True)
    parser.add_argument("--assignment-sha256", required=True)
    parser.add_argument("--assignment-scientific-sha256", required=True)
    parser.add_argument("--audit-result", type=Path, required=True)
    parser.add_argument("--audit-result-sha256", required=True)
    parser.add_argument("--vh-edges", type=Path, required=True)
    parser.add_argument("--vl-edges", type=Path, required=True)
    parser.add_argument("--concatenated-antibody-edges", type=Path, required=True)
    parser.add_argument("--antigen-edges", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    for name in (
        "canonical_manifest_sha256",
        "assignment_sha256",
        "assignment_scientific_sha256",
        "audit_result_sha256",
        "source_sha256",
    ):
        value = getattr(arguments, name)
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            parser.error(f"{name} is not a lowercase SHA-256")
    return arguments


if __name__ == "__main__":
    payload = run(parse_arguments())
    print(
        json.dumps(
            {
                "status": payload["status"],
                "contradictory_inequality": payload["mathematical_proof"][
                    "contradictory_inequality"
                ],
                "scientific_payload_sha256": payload[
                    "scientific_payload_sha256"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
