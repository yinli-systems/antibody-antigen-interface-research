#!/usr/bin/env python3
"""Build a development split isolated by MMseqs2 and exact global alignment.

The procedure starts from an existing development train-validation assignment.
It audits every cross-split unique antibody and antigen sequence pair with the
declared Needleman-Wunsch backend, adds every crossing edge to the component
graph, reselects whole components, and repeats until an independent exhaustive
pass finds zero threshold-qualified crossings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from build_identity_isolated_development_split import (
    SequenceRecord,
    SimilarityHit,
    UnionFind,
    assignments_from_components,
    atomic_write_json,
    audit_assignments,
    build_components,
    choose_validation_components,
    group_by_digest,
    load_development_records,
    load_similarity_hits,
    sequence_digest,
    sha256_file,
)
from needleman_wunsch_development_audit import (
    AlignmentMetrics,
    AuditPolicy,
    PairResult,
    _pair_payload,
    _unique_chain_sequences,
    _unique_sequences,
    atomic_write_jsonl,
    audit_role,
    establish_development_barrier,
    load_development_chain_records,
    validate_audited_split,
    write_primary_edge_artifact,
)


FORMAT_VERSION = "sabdab2_iterative_nw_strict_development_split_v1"


def _canonical_sha256(payload: object) -> str:
    """Hash a JSON-compatible object deterministically."""
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _hit_key(hit: SimilarityHit) -> tuple[str, str]:
    """Return an orientation-independent digest-pair key."""
    return tuple(sorted((hit.query, hit.target)))


def _merge_hits(hits: Iterable[SimilarityHit]) -> dict[tuple[str, str], SimilarityHit]:
    """Retain one deterministic strongest hit for each unordered digest pair."""
    result: dict[tuple[str, str], SimilarityHit] = {}
    for hit in hits:
        key = _hit_key(hit)
        candidate = SimilarityHit(
            query=key[0],
            target=key[1],
            identity=float(hit.identity),
            query_coverage=float(hit.query_coverage),
            target_coverage=float(hit.target_coverage),
        )
        existing = result.get(key)
        if existing is None or (
            candidate.identity,
            min(candidate.query_coverage, candidate.target_coverage),
            candidate.query_coverage,
            candidate.target_coverage,
        ) > (
            existing.identity,
            min(existing.query_coverage, existing.target_coverage),
            existing.query_coverage,
            existing.target_coverage,
        ):
            result[key] = candidate
    return result


def pair_result_to_hit(edge: PairResult) -> SimilarityHit:
    """Convert one exact cross-split edge into component-graph evidence."""
    return SimilarityHit(
        query=edge.train_digest,
        target=edge.validation_digest,
        identity=edge.metrics.aligned_region_identity,
        query_coverage=edge.metrics.left_reciprocal_coverage,
        target_coverage=edge.metrics.right_reciprocal_coverage,
    )


def reachable_component_sizes(
    components: Sequence[Sequence[str]], minimum_train_count: int
) -> dict[str, object]:
    """Summarize every validation size reachable by whole-component selection."""
    total = sum(len(component) for component in components)
    reachable = {0}
    for component in components:
        size = len(component)
        reachable.update({value + size for value in tuple(reachable)})
    feasible = sorted(
        value for value in reachable if 1 <= value and total - value >= minimum_train_count
    )
    if not feasible:
        raise ValueError("No non-empty component holdout satisfies the train minimum")
    return {
        "reachable_validation_size_count": len(feasible),
        "minimum_feasible_validation_instance_count": min(feasible),
        "maximum_feasible_validation_instance_count": max(feasible),
        "feasible_validation_instance_counts_sha256": hashlib.sha256(
            ("\n".join(str(value) for value in feasible) + "\n").encode("ascii")
        ).hexdigest(),
        "feasible_values": feasible,
    }


def _split_lists(assignments: Mapping[str, str]) -> dict[str, list[str]]:
    """Convert assignments into sorted train and validation lists."""
    return {
        name: sorted(
            identifier for identifier, split_name in assignments.items() if split_name == name
        )
        for name in ("train", "validation")
    }


def _chain_digest_groups(
    chain_records, chain: str
) -> dict[str, list[str]]:
    """Group instances by one non-empty exact antibody-chain sequence."""
    groups: dict[str, list[str]] = defaultdict(list)
    for identifier, record in chain_records.items():
        sequence = record.heavy if chain == "vh" else record.light
        if sequence:
            groups[sequence_digest(sequence)].append(identifier)
    return {digest: sorted(values) for digest, values in sorted(groups.items())}


def build_extended_components(
    records,
    antibody_hits: Sequence[SimilarityHit],
    antigen_hits: Sequence[SimilarityHit],
    chain_records=None,
    vh_hits: Sequence[SimilarityHit] = (),
    vl_hits: Sequence[SimilarityHit] = (),
) -> tuple[list[tuple[str, ...]], dict[str, object]]:
    """Build components from concat, antigen, and optional chain evidence."""
    base_components, _ = build_components(records, antibody_hits, antigen_hits)
    union_find = UnionFind(records)
    for component in base_components:
        for member in component[1:]:
            union_find.union(component[0], member)
    if chain_records is not None:
        for chain, hits in (("vh", vh_hits), ("vl", vl_hits)):
            groups = _chain_digest_groups(chain_records, chain)
            for members in groups.values():
                for member in members[1:]:
                    union_find.union(members[0], member)
            for hit in hits:
                if hit.query not in groups or hit.target not in groups:
                    raise ValueError(f"{chain} hit references an unknown development digest")
                union_find.union(groups[hit.query][0], groups[hit.target][0])
    members: dict[str, list[str]] = defaultdict(list)
    for identifier in sorted(records):
        members[union_find.find(identifier)].append(identifier)
    components = sorted(
        (tuple(values) for values in members.values()),
        key=lambda values: (-len(values), values),
    )
    sizes = [len(component) for component in components]
    summary = {
        "component_count": len(components),
        "largest_component_size": max(sizes),
        "largest_component_fraction": max(sizes) / len(records),
        "singleton_component_count": sum(size == 1 for size in sizes),
        "top_component_sizes": sorted(sizes, reverse=True)[:20],
        "includes_vh_and_vl_constraints": chain_records is not None,
    }
    return components, summary


def _audit_exact_boundary(
    records,
    chain_records,
    include_chain_constraints: bool,
    assignments: Mapping[str, str],
    policy: AuditPolicy,
    workers: int,
    chunk_size: int,
    edge_output_dir: Path,
    iteration: int,
) -> tuple[dict[str, object], dict[str, list[PairResult]]]:
    """Audit the current boundary and persist every newly observed crossing edge."""
    split_lists = _split_lists(assignments)
    role_results: dict[str, object] = {}
    edges: dict[str, list[PairResult]] = {}
    roles = ["antibody", "antigen"]
    if include_chain_constraints:
        roles.extend(["vh", "vl"])
    for role in roles:
        if role in {"antibody", "antigen"}:
            train = _unique_sequences(records, split_lists["train"], role)
            validation = _unique_sequences(records, split_lists["validation"], role)
        else:
            train = _unique_chain_sequences(chain_records, split_lists["train"], role)
            validation = _unique_chain_sequences(
                chain_records, split_lists["validation"], role
            )
        result = audit_role(train, validation, policy, workers, chunk_size)
        raw_edges = list(result["_primary_crossing_edges"])
        write_primary_edge_artifact(
            result,
            edge_output_dir,
            f"iteration_{iteration:02d}",
            "concat_antibody" if role == "antibody" else role,
        )
        role_results[role] = result
        edges[role] = raw_edges
    return role_results, edges


def run(args: argparse.Namespace) -> dict[str, object]:
    """Iteratively refine components until exact cross-split leakage is zero."""
    development_ids, sealed_ids, sealed_counts = establish_development_barrier(
        args.development_source_split, args.pool_splits, args.sealed_splits
    )
    base_train, base_validation = validate_audited_split(
        args.base_split,
        development_ids,
        sealed_ids,
        args.train_split,
        args.validation_split,
    )
    records = load_development_records(
        args.manifest,
        development_ids,
        args.id_column,
        args.heavy_column,
        args.light_column,
        args.antigen_column,
    )
    chain_records = (
        load_development_chain_records(
            args.manifest,
            development_ids,
            args.id_column,
            args.heavy_column,
            args.light_column,
        )
        if args.include_chain_constraints
        else None
    )
    identity_threshold = float(args.identity_threshold)
    coverage_threshold = float(args.coverage_threshold)
    antibody_groups = group_by_digest(records, "antibody")
    antigen_groups = group_by_digest(records, "antigen")
    antibody_mmseqs, antibody_summary = load_similarity_hits(
        args.antibody_hits,
        set(antibody_groups),
        identity_threshold,
        coverage_threshold,
    )
    antigen_mmseqs, antigen_summary = load_similarity_hits(
        args.antigen_hits,
        set(antigen_groups),
        identity_threshold,
        coverage_threshold,
    )
    constrained_roles = ["antibody", "antigen"]
    if args.include_chain_constraints:
        constrained_roles.extend(["vh", "vl"])
    hit_maps = {
        "antibody": _merge_hits(antibody_mmseqs),
        "antigen": _merge_hits(antigen_mmseqs),
    }
    if args.include_chain_constraints:
        hit_maps.update({"vh": {}, "vl": {}})
    mmseqs_key_sets = {role: set(values) for role, values in hit_maps.items()}
    discovered_edges: dict[str, dict[tuple[str, str], PairResult]] = {
        role: {} for role in constrained_roles
    }
    assignments = {
        **{identifier: "train" for identifier in base_train},
        **{identifier: "validation" for identifier in base_validation},
    }
    target_validation_count = round(len(records) * float(args.validation_fraction))
    policy = AuditPolicy(
        identity_threshold,
        coverage_threshold,
        float(args.calm_scaled_threshold),
        float(args.gap_open),
        float(args.gap_extend),
        int(args.top_pair_count),
    )
    edge_output_dir = args.edge_output_dir.expanduser().resolve()
    iterations: list[dict[str, object]] = []
    final_components: list[tuple[str, ...]] | None = None
    final_component_summary: dict[str, object] | None = None
    final_reachability: dict[str, object] | None = None
    final_selection: dict[str, object] | None = None
    for iteration in range(args.maximum_iterations + 1):
        exact_results, boundary_edges = _audit_exact_boundary(
            records,
            chain_records,
            args.include_chain_constraints,
            assignments,
            policy,
            args.workers,
            args.chunk_size,
            edge_output_dir,
            iteration,
        )
        violation_counts = {
            role: int(
                exact_results[role][
                    "identity_and_reciprocal_coverage_violation_count"
                ]
            )
            for role in constrained_roles
        }
        new_counts: dict[str, int] = {}
        for role in constrained_roles:
            new_count = 0
            for edge in boundary_edges[role]:
                hit = pair_result_to_hit(edge)
                key = _hit_key(hit)
                if key not in hit_maps[role]:
                    new_count += 1
                hit_maps[role][key] = hit
                discovered_edges[role][key] = edge
            new_counts[role] = new_count
        split_counts = {
            name: sum(value == name for value in assignments.values())
            for name in ("train", "validation")
        }
        iteration_payload: dict[str, object] = {
            "iteration": iteration,
            "split_instance_counts": split_counts,
            "exact_boundary_audit": exact_results,
            "new_non_mmseqs_or_prior_nw_edge_counts": new_counts,
        }
        if not any(violation_counts.values()):
            iteration_payload["converged"] = True
            iterations.append(iteration_payload)
            final_components, final_component_summary = build_extended_components(
                records,
                list(hit_maps["antibody"].values()),
                list(hit_maps["antigen"].values()),
                chain_records,
                list(hit_maps.get("vh", {}).values()),
                list(hit_maps.get("vl", {}).values()),
            )
            reachability = reachable_component_sizes(
                final_components, args.minimum_train_count
            )
            final_reachability = {
                key: value for key, value in reachability.items() if key != "feasible_values"
            }
            feasible_values = reachability["feasible_values"]
            final_selection = {
                "target_validation_instance_count": target_validation_count,
                "selected_validation_instance_count": split_counts["validation"],
                "exact_target_feasible_under_final_graph": (
                    target_validation_count in feasible_values
                ),
                "absolute_target_error": abs(
                    split_counts["validation"] - target_validation_count
                ),
            }
            break
        if iteration >= args.maximum_iterations:
            raise RuntimeError("Exact split refinement did not converge within the limit")
        if not any(new_counts.values()):
            raise AssertionError(
                "A crossing edge was already present in the component graph"
            )
        components, component_summary = build_extended_components(
            records,
            list(hit_maps["antibody"].values()),
            list(hit_maps["antigen"].values()),
            chain_records,
            list(hit_maps.get("vh", {}).values()),
            list(hit_maps.get("vl", {}).values()),
        )
        reachability = reachable_component_sizes(components, args.minimum_train_count)
        selected, selection = choose_validation_components(
            components,
            target_count=target_validation_count,
            seed=args.seed,
            minimum_validation_count=args.minimum_validation_count,
            minimum_train_count=args.minimum_train_count,
        )
        assignments = assignments_from_components(components, selected)
        iteration_payload.update({
            "converged": False,
            "updated_component_summary": component_summary,
            "updated_selection_summary": selection,
            "updated_reachability_summary": {
                key: value for key, value in reachability.items() if key != "feasible_values"
            },
            "exact_target_feasible_after_update": (
                target_validation_count in reachability["feasible_values"]
            ),
        })
        iterations.append(iteration_payload)
    if (
        final_components is None
        or final_component_summary is None
        or final_reachability is None
        or final_selection is None
    ):
        raise AssertionError("Converged final state was not recorded")
    graph_audit = audit_assignments(
        records,
        assignments,
        list(hit_maps["antibody"].values()),
        list(hit_maps["antigen"].values()),
    )
    if not graph_audit["valid"]:
        raise AssertionError("Final assignment fails the accumulated evidence graph")
    discovered_artifacts: dict[str, object] = {}
    for role in constrained_roles:
        path = edge_output_dir / f"all_discovered.{role}.identity_coverage_edges.jsonl"
        ordered = sorted(
            discovered_edges[role].values(),
            key=lambda edge: (edge.train_digest, edge.validation_digest),
        )
        atomic_write_jsonl(path, (_pair_payload(edge) for edge in ordered))
        discovered_artifacts[role] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "edge_count": len(ordered),
            "edge_count_absent_from_initial_mmseqs_graph": sum(
                key not in mmseqs_key_sets[role] for key in discovered_edges[role]
            ),
        }
    source_path = Path(__file__).resolve()
    payload: dict[str, object] = {
        "format_version": FORMAT_VERSION,
        "stage": "build_iterative_nw_strict_development_split",
        "constraint_mode": (
            "concat_antibody_antigen_plus_vh_vl"
            if args.include_chain_constraints
            else "concat_antibody_and_antigen"
        ),
        "intended_use": "development_only",
        "valid": True,
        "frozen_test_labels_or_predictions_read": False,
        "frozen_test_sequences_aligned": False,
        "sealed_identifiers_read_for_exclusion_only": True,
        "sealed_source_splits": list(args.sealed_splits),
        "sealed_source_split_counts": sealed_counts,
        "splits": _split_lists(assignments),
        "final_graph_audit": graph_audit,
        "final_component_summary": final_component_summary,
        "final_reachability_summary": final_reachability,
        "final_selection_summary": final_selection,
        "iterations": iterations,
        "discovered_exact_nw_edge_artifacts": discovered_artifacts,
        "evidence": {
            "identity_threshold": identity_threshold,
            "reciprocal_coverage_threshold": coverage_threshold,
            "mmseqs_antibody": antibody_summary,
            "mmseqs_antigen": antigen_summary,
            "exact_alignment_policy": {
                "backend": "Biopython PairwiseAligner",
                "mode": "global",
                "substitution_matrix": "BLOSUM62",
                "gap_open_penalty": policy.gap_open,
                "gap_extend_penalty": policy.gap_extend,
                "identity_denominator": "paired residue-residue columns",
                "coverage_denominators": "each corresponding full sequence length",
                "chain_constraints_included": args.include_chain_constraints,
            },
        },
        "provenance": {
            "manifest": {"path": str(args.manifest), "sha256": sha256_file(args.manifest)},
            "development_source_split": {
                "path": str(args.development_source_split),
                "sha256": sha256_file(args.development_source_split),
            },
            "base_split": {
                "path": str(args.base_split),
                "sha256": sha256_file(args.base_split),
            },
            "source": {"path": str(source_path), "sha256": sha256_file(source_path)},
        },
        "limitations": [
            "This is a development split, not an external, temporal, or prospective test set.",
            (
                "VH and VL exact-alignment constraints are included in the component graph."
                if args.include_chain_constraints
                else "Primary graph isolation uses concatenated VH+VL antibody identity; separate-chain sensitivity remains supplementary."
            ),
            "Global sequence identity does not establish epitope, structure, or functional independence.",
        ],
    }
    payload["scientific_payload_sha256"] = _canonical_sha256(payload)
    atomic_write_json(args.output, payload)
    print(json.dumps({
        "output": str(args.output.expanduser().resolve()),
        "iterations": len(iterations),
        "final_selection_summary": final_selection,
        "final_split_instance_counts": graph_audit["split_instance_counts"],
        "scientific_payload_sha256": payload["scientific_payload_sha256"],
    }, indent=2, sort_keys=True))
    return payload


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--development-source-split", type=Path, required=True)
    parser.add_argument("--base-split", type=Path, required=True)
    parser.add_argument("--antibody-hits", type=Path, required=True)
    parser.add_argument("--antigen-hits", type=Path, required=True)
    parser.add_argument("--pool-splits", nargs="+", default=["train", "validation"])
    parser.add_argument("--sealed-splits", nargs="+", default=["test"])
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--validation-split", default="validation")
    parser.add_argument("--id-column", default="INSTANCE")
    parser.add_argument("--heavy-column", default="vh_seq")
    parser.add_argument("--light-column", default="vl_seq")
    parser.add_argument("--antigen-column", default="ag_seq")
    parser.add_argument("--identity-threshold", type=float, default=0.80)
    parser.add_argument("--coverage-threshold", type=float, default=0.80)
    parser.add_argument("--calm-scaled-threshold", type=float, default=0.80)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--minimum-validation-count", type=int, default=1)
    parser.add_argument("--minimum-train-count", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gap-open", type=float, default=10.0)
    parser.add_argument("--gap-extend", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--top-pair-count", type=int, default=20)
    parser.add_argument("--maximum-iterations", type=int, default=10)
    parser.add_argument(
        "--include-chain-constraints",
        action="store_true",
        help="Include exhaustive VH-to-VH and VL-to-VL edges in split components.",
    )
    parser.add_argument("--edge-output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    """Build the strict development split."""
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
