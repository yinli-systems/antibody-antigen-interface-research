#!/usr/bin/env python3
"""Collapse STRING pairs using the strict 30%-identity development hit asset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import collapse_string_v12_high_confidence_scale_v2 as base


MINIMUM_IDENTITY = 0.30
MINIMUM_COVERAGE = 0.80


def read_strict_excluded_clusters(
    hits: Path, membership: dict[str, str]
) -> set[str]:
    excluded: set[str] = set()
    for line_number, raw_line in enumerate(base.text_lines(hits), start=1):
        row = raw_line.rstrip("\n").split("\t")
        if len(row) != 8:
            raise ValueError(f"Malformed strict development hit row {line_number}")
        if (
            float(row[2]) + 1e-12 < MINIMUM_IDENTITY
            or float(row[4]) + 1e-12 < MINIMUM_COVERAGE
            or float(row[5]) + 1e-12 < MINIMUM_COVERAGE
        ):
            raise ValueError("Strict development hit violates the frozen homology gate")
        representative = membership.get(row[1])
        if representative is None:
            raise ValueError(f"Strict development-hit target lacks membership: {row[1]}")
        excluded.add(representative)
    if not excluded:
        raise ValueError("Strict development hit asset excludes no endpoint clusters")
    return excluded


def decorate(arguments: argparse.Namespace) -> None:
    membership, cluster_count = base.read_membership(arguments.clusters.resolve())
    excluded = read_strict_excluded_clusters(
        arguments.development_hits.resolve(), membership
    )
    counts = base.decorate_edges(
        arguments.edges.resolve(), membership, excluded, sys.stdout
    )
    stats = {
        **counts,
        "cluster_count": cluster_count,
        "development_excluded_clusters": len(excluded),
        "membership_count": len(membership),
    }
    arguments.stats.resolve().write_text(
        "".join(f"{key}={value}\n" for key, value in sorted(stats.items())),
        encoding="utf-8",
    )


def _authenticate_inputs(arguments: argparse.Namespace) -> tuple[dict, dict, dict]:
    edges = arguments.edges.resolve()
    edge_receipt_path = arguments.edge_receipt.resolve()
    edge_receipt = json.loads(edge_receipt_path.read_text(encoding="utf-8"))
    if (
        edge_receipt.get("format_version")
        != "abpairfm_string_high_confidence_filter_v2"
        or edge_receipt.get("status")
        != "FILTERED_HIGH_CONFIDENCE_NOT_CLUSTERED_NOT_DECONTAMINATED"
        or edge_receipt.get("test_access") is not False
        or edge_receipt.get("output", {}).get("sha256") != base.sha256_file(edges)
    ):
        raise ValueError("High-confidence STRING edge evidence differs")

    clusters = arguments.clusters.resolve()
    cluster_receipt_path = arguments.cluster_receipt.resolve()
    cluster_receipt = json.loads(cluster_receipt_path.read_text(encoding="utf-8"))
    published = {
        asset["name"]: asset
        for asset in cluster_receipt.get("published_assets", [])
    }
    if (
        cluster_receipt.get("format_version")
        != "abpairfm_string_high_confidence_cluster_assets_v1"
        or cluster_receipt.get("status")
        != "CLUSTERED_NOT_COLLAPSED_NOT_DECONTAMINATED"
        or cluster_receipt.get("test_access") is not False
        or published.get(clusters.name, {}).get("sha256")
        != base.sha256_file(clusters)
    ):
        raise ValueError("High-confidence STRING cluster evidence differs")

    hits = arguments.development_hits.resolve()
    strict_receipt_path = arguments.strict_hit_receipt.resolve()
    strict_receipt = json.loads(strict_receipt_path.read_text(encoding="utf-8"))
    if (
        strict_receipt.get("format_version")
        != "abpairfm_string_strict_development_hits_v1"
        or strict_receipt.get("status")
        != "STRICT_DEVELOPMENT_HITS_READY_NOT_COLLAPSED"
        or strict_receipt.get("test_access") is not False
        or strict_receipt.get("authorization", {}).get(
            "pair_collapse_with_strict_hits"
        )
        is not True
        or strict_receipt.get("authorization", {}).get("production_pretraining")
        is not False
        or strict_receipt.get("mmseqs", {}).get("minimum_sequence_identity")
        != MINIMUM_IDENTITY
        or strict_receipt.get("mmseqs", {}).get(
            "minimum_bidirectional_coverage"
        )
        != MINIMUM_COVERAGE
        or strict_receipt.get("output", {}).get("sha256")
        != base.sha256_file(hits)
    ):
        raise ValueError("Strict development-hit evidence differs")

    protocol_path = arguments.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    controls = protocol.get("strict_leakage_controls", {}).get(
        "new_corpus_to_development", {}
    )
    if (
        protocol.get("format_version") != "abpairfm_3b_quality_frontier_v1"
        or protocol.get("authorization", {}).get("production_pretraining") is not False
        or protocol.get("authorization", {}).get("test_access") is not False
        or controls.get("maximum_sequence_identity") != MINIMUM_IDENTITY
        or controls.get("minimum_bidirectional_coverage") != MINIMUM_COVERAGE
        or controls.get("exclude_any_pair_touching_a_development_endpoint_cluster")
        is not True
    ):
        raise ValueError("Quality-frontier protocol does not authorize strict collapse")
    return edge_receipt, cluster_receipt, strict_receipt


def finalize(arguments: argparse.Namespace) -> None:
    edge_receipt, cluster_receipt, strict_receipt = _authenticate_inputs(arguments)
    raw_stats = base.parse_stats(arguments.stats.resolve())
    auxiliary: dict[str, int] = {}
    for raw_line in arguments.decorate_stats.resolve().read_text(
        encoding="utf-8"
    ).splitlines():
        key, value = raw_line.split("=", 1)
        auxiliary[key] = int(value)
    expected_auxiliary = {
        "cluster_count",
        "development_excluded_clusters",
        "development_excluded_edges",
        "eligible_nonself_edges",
        "membership_count",
        "missing_endpoint_memberships",
        "self_cluster_edges_removed",
        "source_edges",
    }
    if set(auxiliary) != expected_auxiliary:
        raise ValueError("Strict decorator statistics do not match the frozen schema")
    for key in base.REQUIRED_COUNTS - {"output_cluster_pairs"}:
        if raw_stats[key] != auxiliary[key]:
            raise ValueError(f"Strict decorator and collapse statistics differ for {key}")
    if raw_stats["source_edges"] != int(
        edge_receipt["counts"]["high_confidence_rows"]
    ):
        raise ValueError("Strict collapsed source-edge count differs")
    if auxiliary["membership_count"] != int(
        cluster_receipt["cluster"]["membership_count"]
    ):
        raise ValueError("Strict collapsed membership count differs")
    if auxiliary["development_excluded_clusters"] > int(
        strict_receipt["counts"]["unique_string_endpoints_with_hits"]
    ):
        raise ValueError("Excluded cluster count exceeds strict-hit endpoints")

    protocol_path = arguments.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    minimum = int(
        protocol["compute_optimality"]["strict_scale_pool_minimum_cluster_pairs"]
    )
    output = arguments.output.resolve()
    output_count = raw_stats["output_cluster_pairs"]
    scale_gate = output_count >= minimum
    receipt = {
        "authorization": {
            "production_pretraining": False,
            "strict_scale_pool_sampling": scale_gate,
        },
        "claim_boundary": "One strongest STRING physical-network edge per unordered 50%-identity endpoint-cluster pair after strict removal of every endpoint cluster with at least 30% identity and 80% bidirectional coverage to the frozen AsEP/SAbDab2 development panel. This is a train-pool asset only; production training, development evaluation, test access, and SOTA claims remain closed.",
        "cluster": {
            "coverage": 0.8,
            "coverage_mode": 0,
            "membership_count": auxiliary["membership_count"],
            "minimum_sequence_identity": 0.5,
        },
        "counts": {
            **raw_stats,
            "development_excluded_clusters": auxiliary[
                "development_excluded_clusters"
            ],
            "duplicate_cluster_pair_edges": raw_stats["eligible_nonself_edges"]
            - output_count,
        },
        "format_version": "abpairfm_string_high_confidence_strict_scale_pool_v1",
        "inputs": {
            "cluster_receipt_sha256": base.sha256_file(
                arguments.cluster_receipt.resolve()
            ),
            "edge_receipt_sha256": base.sha256_file(
                arguments.edge_receipt.resolve()
            ),
            "quality_protocol_sha256": base.sha256_file(protocol_path),
            "strict_hit_receipt_sha256": base.sha256_file(
                arguments.strict_hit_receipt.resolve()
            ),
        },
        "output": {
            "bytes": output.stat().st_size,
            "name": arguments.output_name or output.name,
            "sha256": base.sha256_file(output),
        },
        "scale_pool_gate": {
            "minimum_cluster_pairs": minimum,
            "passed": scale_gate,
        },
        "strict_development_decontamination": {
            "exclude_any_pair_touching_a_hit_cluster": True,
            "minimum_bidirectional_coverage": MINIMUM_COVERAGE,
            "minimum_sequence_identity": MINIMUM_IDENTITY,
        },
        "string_task_authority": protocol["string_evidence_tiers"][
            "task_authority"
        ],
        "status": (
            "STRICT_DECONTAMINATED_SCALE_POOL_NOT_AUTHORIZED_FOR_TRAINING"
            if scale_gate
            else "STRICT_DECONTAMINATED_SCALE_POOL_BELOW_MINIMUM"
        ),
        "test_access": False,
    }
    base.publish_create_only(arguments.receipt.resolve(), receipt)
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    decorate_parser = subparsers.add_parser("decorate")
    decorate_parser.add_argument("--edges", type=Path, required=True)
    decorate_parser.add_argument("--clusters", type=Path, required=True)
    decorate_parser.add_argument("--development-hits", type=Path, required=True)
    decorate_parser.add_argument("--stats", type=Path, required=True)
    decorate_parser.set_defaults(function=decorate)

    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--edges", type=Path, required=True)
    finalize_parser.add_argument("--edge-receipt", type=Path, required=True)
    finalize_parser.add_argument("--clusters", type=Path, required=True)
    finalize_parser.add_argument("--cluster-receipt", type=Path, required=True)
    finalize_parser.add_argument("--development-hits", type=Path, required=True)
    finalize_parser.add_argument("--strict-hit-receipt", type=Path, required=True)
    finalize_parser.add_argument("--protocol", type=Path, required=True)
    finalize_parser.add_argument("--decorate-stats", type=Path, required=True)
    finalize_parser.add_argument("--stats", type=Path, required=True)
    finalize_parser.add_argument("--output", type=Path, required=True)
    finalize_parser.add_argument("--output-name")
    finalize_parser.add_argument("--receipt", type=Path, required=True)
    finalize_parser.set_defaults(function=finalize)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    arguments.function(arguments)


if __name__ == "__main__":
    main()
