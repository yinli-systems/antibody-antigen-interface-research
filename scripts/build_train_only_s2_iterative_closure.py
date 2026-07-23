#!/usr/bin/env python3
"""Publish a necessary-condition infeasibility receipt for exact S2 closure.

This program never launches a solver and never exports a fold assignment. It
authenticates a failed provisional assignment and its exact four-view audit,
reconstructs authority-train sequences behind the firewall, lifts VH and VL
digest edges to concatenated-antibody entities, and computes the transitive
same-fold components. If any component's integer unique-pair degree exceeds
the locked per-fold upper bound, the requested balanced partition is
mathematically infeasible and a create-only receipt is published.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import audit_train_only_s2_exact_four_view_nw as auditor


FORMAT_VERSION = "sabdab2_train_only_s2_exact_closure_infeasibility_receipt_v2"
CONFIG_FORMAT_VERSION = "sabdab2_train_only_s2_iterative_closure_config_v2"
PARENT_AUDIT_FORMAT_VERSION = (
    "sabdab2_train_only_s2_exact_four_view_nw_audit_receipt_v1"
)
REQUIRED_VIEWS = ("vh", "vl", "concatenated_antibody", "antigen")


class UnionFind:
    """Deterministic lexical-root disjoint-set structure."""

    def __init__(self, values: Iterable[str]) -> None:
        identifiers = sorted(set(values))
        if not identifiers:
            raise ValueError("UnionFind requires a non-empty inventory")
        self.parent = {value: value for value in identifiers}

    def find(self, value: str) -> str:
        """Return the lexical component root with path compression."""
        if value not in self.parent:
            raise KeyError(f"Unknown union-find identifier: {value}")
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: str, right: str) -> None:
        """Join two identifiers under the lexical minimum root."""
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        low, high = sorted((left_root, right_root))
        self.parent[high] = low


def load_config(path: Path) -> dict[str, Any]:
    """Load and validate the immutable exact-closure configuration."""
    config = auditor.strict_json_object(path, "exact-closure config")
    auditor.require_exact_keys(
        config,
        {
            "axis_balance_tolerance",
            "constraint_sources",
            "deterministic_policy",
            "equality_constraint_semantics",
            "fold_axis_balance_unit",
            "fold_count",
            "format_version",
            "identity_threshold",
            "infeasibility_precheck",
            "integer_axis_degree_lower_bound",
            "integer_axis_degree_upper_bound",
            "iteration_index",
            "lexicographic_objectives",
            "nonzero_audit_iteration_policy",
            "reciprocal_coverage_threshold",
            "solver_policy",
            "threshold_or_balance_relaxation_after_results",
        },
        "exact-closure config",
    )
    if config["format_version"] != CONFIG_FORMAT_VERSION:
        raise ValueError("Unsupported exact-closure config format_version")
    if auditor.require_integer(config["fold_count"], "fold_count") != 3:
        raise ValueError("Exact closure requires exactly three folds")
    if auditor.require_integer(config["iteration_index"], "iteration_index") != 1:
        raise ValueError("This receipt builder is pinned to closure iteration one")
    if config["identity_threshold"] != 0.8 or config[
        "reciprocal_coverage_threshold"
    ] != 0.8:
        raise ValueError("Exact-closure thresholds must remain locked at 0.8")
    if config["axis_balance_tolerance"] != 0.05:
        raise ValueError("Axis balance tolerance must remain locked at 0.05")
    if auditor.require_integer(
        config["integer_axis_degree_lower_bound"], "integer lower bound"
    ) != 580 or auditor.require_integer(
        config["integer_axis_degree_upper_bound"], "integer upper bound"
    ) != 640:
        raise ValueError("Locked integer axis-degree bounds must be 580..640")
    if config["threshold_or_balance_relaxation_after_results"] is not False:
        raise ValueError("Post-result threshold or balance relaxation is forbidden")
    expected_objectives = [
        "maximize_retained_unique_exact_antibody_antigen_sequence_pairs",
        "conditional_on_prior_optimum_maximize_retained_authority_train_source_rows",
        "conditional_on_prior_optima_maximize_minimum_fold_retained_unique_pair_count",
    ]
    if config["lexicographic_objectives"] != expected_objectives:
        raise ValueError("Lexicographic objectives differ from the recorded contract")
    return config


def validate_parent_audit(
    payload: Mapping[str, Any],
    expected_scientific_sha256: str,
    audit_reference: Mapping[str, Any],
    assignment_reference: Mapping[str, Any],
    assignment_scientific_sha256: str,
    authority_reference: Mapping[str, Any],
    manifest_reference: Mapping[str, Any],
) -> None:
    """Authenticate the failed exact audit and its critical firewall bindings."""
    expected_scientific = auditor.validate_sha256(
        expected_scientific_sha256, "parent audit scientific payload SHA-256"
    )
    claimed = auditor.validate_sha256(
        payload.get("scientific_payload_sha256"),
        "parent audit embedded scientific payload SHA-256",
    )
    scientific = dict(payload)
    scientific.pop("scientific_payload_sha256", None)
    observed = auditor.canonical_payload_sha256(scientific)
    if claimed != expected_scientific or observed != expected_scientific:
        raise ValueError("Parent audit scientific payload digest mismatch")
    if payload.get("format_version") != PARENT_AUDIT_FORMAT_VERSION:
        raise ValueError("Unexpected parent exact-audit format_version")
    if payload.get("valid_audit_execution") is not True:
        raise ValueError("Parent exact audit was not a valid completed execution")
    if payload.get("approved_for_training") is not False or payload.get(
        "approved_for_model_selection"
    ) is not False:
        raise ValueError("Parent audit must be a non-approved closure iteration")
    if payload.get("approval_status") != "not_approved_cross_fold_edges_detected":
        raise ValueError("Parent audit did not record exact crossing edges")
    assignment = payload.get("assignment")
    if not isinstance(assignment, Mapping):
        raise ValueError("Parent audit assignment reference is missing")
    if assignment.get("sha256") != assignment_reference["sha256"] or assignment.get(
        "scientific_payload_sha256"
    ) != assignment_scientific_sha256:
        raise ValueError("Parent audit binds a different provisional assignment")
    inputs = payload.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("Parent audit input references are missing")
    if inputs.get("authority_split", {}).get("sha256") != authority_reference["sha256"]:
        raise ValueError("Parent audit binds a different authority split")
    if inputs.get("canonical_sequence_manifest", {}).get("sha256") != manifest_reference[
        "sha256"
    ]:
        raise ValueError("Parent audit binds a different sequence manifest")
    firewall = payload.get("data_firewall")
    if not isinstance(firewall, Mapping):
        raise ValueError("Parent audit data firewall is missing")
    required_zero_fields = (
        "sealed_rows_materialized",
        "validation_sequence_fields_parsed",
        "test_sequence_fields_parsed",
        "validation_or_test_labels_read",
        "validation_or_test_predictions_read",
        "validation_or_test_scores_read",
    )
    if any(firewall.get(field) != 0 for field in required_zero_fields):
        raise PermissionError("Parent audit firewall does not prove sealed-data exclusion")
    if audit_reference["sha256"] != auditor.sha256_file(
        Path(str(audit_reference["path"]))
    ):
        raise RuntimeError("Parent audit bytes changed during parsing")


def _strict_json_line(line: str, label: str) -> dict[str, Any]:
    """Parse one JSONL row with duplicate and non-finite rejection."""
    payload = json.loads(
        line,
        object_pairs_hook=auditor._reject_duplicate_pairs,
        parse_constant=auditor._reject_nonfinite_constant,
    )
    auditor._validate_finite_json(payload, label)
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def load_edge_artifact(
    path: Path,
    view: str,
    expected_sha256: str,
    expected_count: int,
) -> tuple[set[tuple[str, str]], dict[str, Any]]:
    """Authenticate and load one complete digest-only exact crossing relation."""
    reference = auditor.verify_file(path, expected_sha256, f"{view} exact edge artifact")
    expected_keys = {
        "format_version",
        "sequence_view",
        "left_fold_index",
        "right_fold_index",
        "left_sequence_sha256",
        "right_sequence_sha256",
        "left_length",
        "right_length",
        "alignment",
        "locked_threshold_rule_met",
        "raw_sequences_or_instance_ids_included",
    }
    expected_alignment_keys = {
        "score",
        "identical_residue_pair_count",
        "aligned_residue_pair_count",
        "alignment_column_count",
        "aligned_region_identity",
        "left_reciprocal_coverage",
        "right_reciprocal_coverage",
        "minimum_reciprocal_coverage",
    }
    edges: set[tuple[str, str]] = set()
    row_count = 0
    with Path(reference["path"]).open("r", encoding="ascii") as handle:
        for line_number, line in enumerate(handle, start=1):
            row_count += 1
            row = _strict_json_line(line, f"{view} edge line {line_number}")
            auditor.require_exact_keys(row, expected_keys, f"{view} edge")
            alignment = row["alignment"]
            if not isinstance(alignment, Mapping):
                raise ValueError(f"{view} edge alignment must be a JSON object")
            auditor.require_exact_keys(
                alignment, expected_alignment_keys, f"{view} edge alignment"
            )
            if row["format_version"] != auditor.EDGE_FORMAT_VERSION:
                raise ValueError(f"{view} edge format_version differs")
            if row["sequence_view"] != view:
                raise ValueError(f"{view} edge declares a different sequence view")
            if (
                row["locked_threshold_rule_met"] is not True
                or row["raw_sequences_or_instance_ids_included"] is not False
            ):
                raise ValueError(f"{view} edge violates the digest-only threshold contract")
            left = auditor.validate_sha256(
                row["left_sequence_sha256"], f"{view} left digest"
            )
            right = auditor.validate_sha256(
                row["right_sequence_sha256"], f"{view} right digest"
            )
            left_fold = auditor.require_integer(row["left_fold_index"], "left fold")
            right_fold = auditor.require_integer(row["right_fold_index"], "right fold")
            if (left_fold, right_fold) not in auditor.FOLD_PAIRS:
                raise ValueError(f"{view} edge has an invalid fold pair")
            left_length = auditor.require_integer(row["left_length"], "left length")
            right_length = auditor.require_integer(row["right_length"], "right length")
            paired = auditor.require_integer(
                alignment["aligned_residue_pair_count"], "aligned pair count"
            )
            identical = auditor.require_integer(
                alignment["identical_residue_pair_count"], "identical pair count"
            )
            if not (
                5 * identical >= 4 * paired
                and 5 * paired >= 4 * left_length
                and 5 * paired >= 4 * right_length
            ):
                raise ValueError(f"{view} edge fails the locked exact threshold")
            edge = tuple(sorted((left, right)))
            if edge in edges:
                raise ValueError(f"{view} edge artifact contains a duplicate relation")
            edges.add(edge)
    if row_count != expected_count:
        raise ValueError(
            f"{view} edge count mismatch: expected={expected_count}, observed={row_count}"
        )
    return edges, {
        **reference,
        "edge_count": row_count,
        "sequence_view": view,
        "all_rows_digest_only_and_threshold_verified": True,
    }


def _union_group(union_find: UnionFind, values: Iterable[str]) -> None:
    """Union one non-empty entity group using its lexical first member."""
    ordered = sorted(set(values))
    if not ordered:
        raise ValueError("Cannot union an empty entity group")
    for value in ordered[1:]:
        union_find.union(ordered[0], value)


def build_exact_closure_components(
    records: Mapping[str, auditor.SequenceRecord],
    antibody_assignment: Mapping[str, Mapping[str, Any]],
    antigen_assignment: Mapping[str, Mapping[str, Any]],
    edges_by_view: Mapping[str, set[tuple[str, str]]],
) -> tuple[
    dict[str, str],
    dict[str, str],
    set[tuple[str, str]],
    dict[str, Any],
]:
    """Lift all four exact views and return augmented entity components."""
    source_pairs: list[tuple[str, str]] = []
    concat_by_vh: defaultdict[str, set[str]] = defaultdict(set)
    concat_by_vl: defaultdict[str, set[str]] = defaultdict(set)
    for record in records.values():
        antibody = auditor.sequence_sha256(record.concatenated_antibody)
        antigen = auditor.sequence_sha256(record.antigen)
        vh = auditor.sequence_sha256(record.vh)
        source_pairs.append((antibody, antigen))
        concat_by_vh[vh].add(antibody)
        if record.vl:
            concat_by_vl[auditor.sequence_sha256(record.vl)].add(antibody)
    antibody_entities = {antibody for antibody, _ in source_pairs}
    antigen_entities = {antigen for _, antigen in source_pairs}
    if set(antibody_assignment) != antibody_entities:
        raise ValueError("Parent antibody assignment inventory differs from train sequences")
    if set(antigen_assignment) != antigen_entities:
        raise ValueError("Parent antigen assignment inventory differs from train sequences")

    antibody_union = UnionFind(antibody_entities)
    antigen_union = UnionFind(antigen_entities)
    base_antibody_groups: defaultdict[str, set[str]] = defaultdict(set)
    base_antigen_groups: defaultdict[str, set[str]] = defaultdict(set)
    for digest, row in antibody_assignment.items():
        base_antibody_groups[str(row["cluster_id"])].add(digest)
    for digest, row in antigen_assignment.items():
        base_antigen_groups[str(row["cluster_id"])].add(digest)
    for members in base_antibody_groups.values():
        _union_group(antibody_union, members)
    for members in base_antigen_groups.values():
        _union_group(antigen_union, members)

    for mapping in (concat_by_vh, concat_by_vl):
        for members in mapping.values():
            _union_group(antibody_union, members)

    for view, mapping in (("vh", concat_by_vh), ("vl", concat_by_vl)):
        for left, right in sorted(edges_by_view[view]):
            left_entities = mapping.get(left)
            right_entities = mapping.get(right)
            if not left_entities or not right_entities:
                raise ValueError(f"{view} exact edge cannot be lifted to antibody entities")
            _union_group(antibody_union, left_entities)
            _union_group(antibody_union, right_entities)
            antibody_union.union(min(left_entities), min(right_entities))
    for left, right in sorted(edges_by_view["concatenated_antibody"]):
        if left not in antibody_entities or right not in antibody_entities:
            raise ValueError("Concatenated-antibody edge endpoint is absent from train")
        antibody_union.union(left, right)
    for left, right in sorted(edges_by_view["antigen"]):
        if left not in antigen_entities or right not in antigen_entities:
            raise ValueError("Antigen edge endpoint is absent from train")
        antigen_union.union(left, right)

    antibody_root = {
        digest: antibody_union.find(digest) for digest in sorted(antibody_entities)
    }
    antigen_root = {
        digest: antigen_union.find(digest) for digest in sorted(antigen_entities)
    }
    exact_pairs = set(source_pairs)
    evidence = {
        "base_parent_component_counts": {
            "antibody": len(base_antibody_groups),
            "antigen": len(base_antigen_groups),
        },
        "chain_digest_equivalence_group_counts": {
            "vh": len(concat_by_vh),
            "nonempty_vl": len(concat_by_vl),
        },
        "ingested_exact_edge_counts": {
            view: len(edges_by_view[view]) for view in REQUIRED_VIEWS
        },
        "all_constraint_sources_applied_transitively": True,
    }
    return antibody_root, antigen_root, exact_pairs, evidence


def component_summary(
    root_by_entity: Mapping[str, str],
    exact_pairs: set[tuple[str, str]],
    axis: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Summarize component sizes and original exact-pair degrees."""
    if axis not in {"antibody", "antigen"}:
        raise ValueError(f"Unknown component axis: {axis}")
    members: defaultdict[str, set[str]] = defaultdict(set)
    for entity, root in root_by_entity.items():
        members[root].add(entity)
    degree: Counter[str] = Counter()
    for antibody, antigen in exact_pairs:
        entity = antibody if axis == "antibody" else antigen
        degree[root_by_entity[entity]] += 1
    ranked = sorted(
        members,
        key=lambda root: (-degree[root], -len(members[root]), root),
    )
    largest = ranked[0]
    inventory_rows = [
        f"{root}\t{entity}"
        for entity, root in sorted(root_by_entity.items())
    ]
    degree_rows = [
        f"{root}\t{degree[root]}\t{len(members[root])}" for root in sorted(members)
    ]
    public = {
        "axis": axis,
        "entity_count": len(root_by_entity),
        "component_count": len(members),
        "singleton_component_count": sum(len(values) == 1 for values in members.values()),
        "component_entity_inventory_sha256": auditor.sha256_bytes(
            ("\n".join(inventory_rows) + "\n").encode("ascii")
        ),
        "component_degree_inventory_sha256": auditor.sha256_bytes(
            ("\n".join(degree_rows) + "\n").encode("ascii")
        ),
        "sum_component_unique_pair_degrees": sum(degree.values()),
        "largest_pair_degree_component": {
            "entity_count": len(members[largest]),
            "entity_fraction": len(members[largest]) / len(root_by_entity),
            "unique_pair_degree": degree[largest],
            "unique_pair_degree_fraction": degree[largest] / len(exact_pairs),
            "entity_membership_sha256": auditor.membership_sha256(members[largest]),
        },
        "top_component_pair_degrees": [
            {
                "unique_pair_degree": degree[root],
                "entity_count": len(members[root]),
            }
            for root in ranked[:10]
        ],
        "raw_sequences_or_instance_ids_included": False,
    }
    witness = {
        "root": largest,
        "members": members[largest],
        "degree": degree[largest],
    }
    return public, witness


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the create-only infeasibility-receipt command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority-split", type=Path, required=True)
    parser.add_argument("--authority-sha256", required=True)
    parser.add_argument("--canonical-manifest", type=Path, required=True)
    parser.add_argument("--canonical-manifest-sha256", required=True)
    parser.add_argument("--parent-assignment", type=Path, required=True)
    parser.add_argument("--parent-assignment-sha256", required=True)
    parser.add_argument("--parent-assignment-scientific-sha256", required=True)
    parser.add_argument("--parent-audit-receipt", type=Path, required=True)
    parser.add_argument("--parent-audit-receipt-sha256", required=True)
    parser.add_argument("--parent-audit-scientific-sha256", required=True)
    parser.add_argument("--vh-edges", type=Path, required=True)
    parser.add_argument("--vl-edges", type=Path, required=True)
    parser.add_argument("--concatenated-antibody-edges", type=Path, required=True)
    parser.add_argument("--antigen-edges", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--expected-auditor-source-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    for name in (
        "authority_sha256",
        "canonical_manifest_sha256",
        "parent_assignment_sha256",
        "parent_assignment_scientific_sha256",
        "parent_audit_receipt_sha256",
        "parent_audit_scientific_sha256",
        "config_sha256",
        "expected_source_sha256",
        "expected_auditor_source_sha256",
    ):
        try:
            setattr(args, name, auditor.validate_sha256(getattr(args, name), name))
        except ValueError as error:
            parser.error(str(error))
    return args


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Prove the locked balance infeasible and publish one immutable receipt."""
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    references = {
        "authority_split": auditor.verify_file(
            args.authority_split, args.authority_sha256, "authority split"
        ),
        "canonical_sequence_manifest": auditor.verify_file(
            args.canonical_manifest,
            args.canonical_manifest_sha256,
            "canonical sequence manifest",
        ),
        "parent_provisional_assignment": auditor.verify_file(
            args.parent_assignment,
            args.parent_assignment_sha256,
            "parent provisional assignment",
        ),
        "parent_exact_audit_receipt": auditor.verify_file(
            args.parent_audit_receipt,
            args.parent_audit_receipt_sha256,
            "parent exact audit receipt",
        ),
        "closure_config": auditor.verify_file(
            args.config, args.config_sha256, "exact-closure config"
        ),
    }
    sources = {
        "exact_closure_infeasibility_builder": auditor.verify_file(
            Path(__file__), args.expected_source_sha256, "closure builder source"
        ),
        "independent_exact_four_view_auditor": auditor.verify_file(
            Path(auditor.__file__),
            args.expected_auditor_source_sha256,
            "independent auditor source",
        ),
    }
    config = load_config(Path(references["closure_config"]["path"]))
    train_ids, sealed_ids, authority_summary = auditor.load_authority(
        Path(references["authority_split"]["path"])
    )
    assignment, fold_by_source, antibody_assignment, antigen_assignment = (
        auditor.load_and_validate_assignment(
            Path(references["parent_provisional_assignment"]["path"]),
            args.parent_assignment_scientific_sha256,
            references["parent_provisional_assignment"],
            references["authority_split"],
            references["canonical_sequence_manifest"],
            train_ids,
            authority_summary,
        )
    )
    records, firewall = auditor.load_authorized_sequences(
        Path(references["canonical_sequence_manifest"]["path"]),
        train_ids,
        sealed_ids,
    )
    auditor.validate_assignment_against_sequences(
        assignment,
        records,
        fold_by_source,
        antibody_assignment,
        antigen_assignment,
    )

    parent_audit = auditor.strict_json_object(
        Path(references["parent_exact_audit_receipt"]["path"]),
        "parent exact audit receipt",
    )
    validate_parent_audit(
        parent_audit,
        args.parent_audit_scientific_sha256,
        references["parent_exact_audit_receipt"],
        references["parent_provisional_assignment"],
        args.parent_assignment_scientific_sha256,
        references["authority_split"],
        references["canonical_sequence_manifest"],
    )
    artifact_contracts = parent_audit.get("crossing_edge_artifacts")
    if not isinstance(artifact_contracts, Mapping) or set(artifact_contracts) != set(
        REQUIRED_VIEWS
    ):
        raise ValueError("Parent audit does not bind exactly four edge artifacts")
    edge_paths = {
        "vh": args.vh_edges,
        "vl": args.vl_edges,
        "concatenated_antibody": args.concatenated_antibody_edges,
        "antigen": args.antigen_edges,
    }
    edges_by_view: dict[str, set[tuple[str, str]]] = {}
    edge_references: dict[str, dict[str, Any]] = {}
    for view in REQUIRED_VIEWS:
        contract = artifact_contracts[view]
        if not isinstance(contract, Mapping):
            raise ValueError(f"Parent {view} edge contract is not an object")
        expected_count = auditor.require_integer(
            contract.get("edge_count"), f"parent {view} edge count"
        )
        expected_sha256 = auditor.validate_sha256(
            contract.get("sha256"), f"parent {view} edge SHA-256"
        )
        edges, edge_reference = load_edge_artifact(
            edge_paths[view], view, expected_sha256, expected_count
        )
        edges_by_view[view] = edges
        edge_references[view] = edge_reference
    expected_total_crossings = parent_audit.get("exhaustive_accounting", {}).get(
        "total_threshold_crossing_edge_count"
    )
    if expected_total_crossings != sum(len(edges) for edges in edges_by_view.values()):
        raise ValueError("Four-view edge total differs from the parent audit")

    antibody_root, antigen_root, exact_pairs, closure_evidence = (
        build_exact_closure_components(
            records,
            antibody_assignment,
            antigen_assignment,
            edges_by_view,
        )
    )
    antibody_summary, antibody_witness = component_summary(
        antibody_root, exact_pairs, "antibody"
    )
    antigen_summary, antigen_witness = component_summary(
        antigen_root, exact_pairs, "antigen"
    )
    total_unique_pairs = len(exact_pairs)
    if total_unique_pairs % int(config["fold_count"]) != 0:
        raise RuntimeError("Locked target unexpectedly has a fractional rational value")
    integer_target = total_unique_pairs // int(config["fold_count"])
    continuous_lower = (1.0 - float(config["axis_balance_tolerance"])) * integer_target
    continuous_upper = (1.0 + float(config["axis_balance_tolerance"])) * integer_target
    integer_lower = math.ceil(continuous_lower)
    integer_upper = math.floor(continuous_upper)
    if integer_lower != config["integer_axis_degree_lower_bound"] or integer_upper != config[
        "integer_axis_degree_upper_bound"
    ]:
        raise RuntimeError("Configured integer balance bounds are algebraically inconsistent")
    witnesses = [
        ("antibody", antibody_witness),
        ("antigen", antigen_witness),
    ]
    violations = [
        (axis, witness)
        for axis, witness in witnesses
        if int(witness["degree"]) > integer_upper
    ]
    if not violations:
        raise RuntimeError(
            "The necessary-condition precheck did not prove infeasibility; "
            "this program is forbidden from launching a solver or exporting an assignment"
        )
    proof_axis, proof_witness = max(
        violations, key=lambda item: (int(item[1]["degree"]), item[0])
    )

    all_references = {**references, **sources, **edge_references}
    auditor._verify_references_unchanged(all_references)
    receipt: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "stage": "exact_four_view_transitive_component_balance_infeasibility_proof",
        "intended_use": "authority_train_only_split_design_infeasibility_evidence",
        "valid_infeasibility_proof": True,
        "strict_three_fold_s2_balance_feasible": False,
        "solver_launched": False,
        "assignment_exported": False,
        "authorized_for_training": False,
        "authorized_for_model_selection": False,
        "decision": {
            "status": "stop_strict_transitive_three_fold_s2_workflow",
            "threshold_relaxed": False,
            "balance_tolerance_relaxed": False,
            "silent_alternative_split_substituted": False,
            "reason": "necessary_component_degree_exceeds_locked_integer_fold_upper_bound",
        },
        "inputs": {
            **references,
            "parent_assignment_scientific_payload_sha256": (
                args.parent_assignment_scientific_sha256
            ),
            "parent_audit_scientific_payload_sha256": (
                args.parent_audit_scientific_sha256
            ),
            "exact_crossing_edge_artifacts": edge_references,
        },
        "source_files": sources,
        "config": config,
        "config_payload_sha256": auditor.canonical_payload_sha256(config),
        "authority": authority_summary,
        "data_firewall": firewall,
        "closure_evidence": closure_evidence,
        "augmented_components": {
            "antibody": antibody_summary,
            "antigen": antigen_summary,
        },
        "locked_balance_contract": {
            "fold_count": 3,
            "total_original_unique_exact_pair_degree_per_axis": total_unique_pairs,
            "rational_target_per_axis_per_fold": integer_target,
            "continuous_tolerance_interval": {
                "lower": continuous_lower,
                "upper": continuous_upper,
            },
            "integer_unique_pair_degree_bounds_per_axis_per_fold": {
                "lower_inclusive": integer_lower,
                "upper_inclusive": integer_upper,
            },
        },
        "mathematical_infeasibility_proof": {
            "necessary_condition": (
                "Every transitive same-fold component is assigned wholly to exactly "
                "one fold, so the assigned fold's axis degree is at least that "
                "component's original unique-pair degree. Therefore every component "
                "degree must be at most the locked integer fold upper bound."
            ),
            "witness_axis": proof_axis,
            "witness_component_entity_count": len(proof_witness["members"]),
            "witness_component_entity_membership_sha256": auditor.membership_sha256(
                proof_witness["members"]
            ),
            "witness_component_unique_pair_degree": int(proof_witness["degree"]),
            "locked_integer_fold_upper_bound": integer_upper,
            "contradictory_inequality": f"{int(proof_witness['degree'])} > {integer_upper}",
            "upper_bound_excess_unique_pair_degree": (
                int(proof_witness["degree"]) - integer_upper
            ),
            "upper_bound_ratio": int(proof_witness["degree"]) / integer_upper,
            "conclusion": (
                "No one-hot three-fold assignment can satisfy the locked per-axis "
                "integer degree bounds after exact four-view transitive closure."
            ),
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "solver": "not_launched_by_necessary_condition",
        },
        "publication": {
            "receipt_path": str(output),
            "create_only_no_overwrite": True,
            "inputs_and_sources_rehashed_immediately_before_publication": True,
            "raw_sequences_or_instance_ids_included": False,
        },
        "limitations": [
            "This receipt proves infeasibility only for the recorded transitive same-fold constraints, three-fold count, and integer 580..640 degree bounds.",
            "It does not authorize a relaxed threshold, relaxed balance, alternate split, training run, model selection, or model evaluation.",
            "Authority validation and frozen-test sequence fields, labels, predictions, checkpoints, and scores were not read.",
        ],
    }
    receipt["scientific_payload_sha256"] = auditor.canonical_payload_sha256(receipt)
    staged = auditor.stage_bytes(output, auditor.canonical_json_bytes(receipt))
    try:
        auditor.publish_staged_no_clobber(staged)
    finally:
        staged.temporary_path.unlink(missing_ok=True)
    print(
        json.dumps(
            {
                "receipt_path": str(output),
                "receipt_sha256": auditor.sha256_file(output),
                "scientific_payload_sha256": receipt["scientific_payload_sha256"],
                "solver_launched": False,
                "assignment_exported": False,
                "contradictory_inequality": receipt[
                    "mathematical_infeasibility_proof"
                ]["contradictory_inequality"],
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ),
        flush=True,
    )
    return receipt


def main(argv: Sequence[str] | None = None) -> None:
    """Run the exact-closure infeasibility precheck."""
    run(parse_args(argv))


if __name__ == "__main__":
    main()
