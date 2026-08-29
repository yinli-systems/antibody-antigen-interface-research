#!/usr/bin/env python3
"""Stream and finalize the decontaminated STRING high-confidence scale pool."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Iterator, TextIO


EXPECTED_HEADER = (
    "protein1",
    "protein2",
    "homology",
    "experiments",
    "experiments_transferred",
    "database",
    "database_transferred",
    "textmining",
    "textmining_transferred",
    "combined_score",
)
REQUIRED_COUNTS = {
    "development_excluded_edges",
    "eligible_nonself_edges",
    "missing_endpoint_memberships",
    "output_cluster_pairs",
    "self_cluster_edges_removed",
    "source_edges",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def zstd_lines(path: Path) -> Iterator[str]:
    process = subprocess.Popen(
        ["zstd", "-q", "-dc", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    assert process.stdout is not None
    try:
        yield from process.stdout
    finally:
        process.stdout.close()
        error = process.stderr.read() if process.stderr is not None else ""
        return_code = process.wait()
        if return_code:
            raise RuntimeError(f"zstd failed with {return_code}: {error[-1000:]}")


def text_lines(path: Path) -> Iterator[str]:
    if path.suffix == ".zst":
        yield from zstd_lines(path)
        return
    with path.open("rt", encoding="utf-8", newline="") as handle:
        yield from handle


def read_membership(path: Path) -> tuple[dict[str, str], int]:
    membership: dict[str, str] = {}
    representatives: set[str] = set()
    for line_number, raw_line in enumerate(text_lines(path), start=1):
        row = raw_line.rstrip("\n").split("\t")
        if len(row) != 2 or not row[0] or not row[1]:
            raise ValueError(f"Malformed MMseqs cluster row {line_number}")
        representative = sys.intern(row[0])
        member = row[1]
        previous = membership.setdefault(member, representative)
        if previous != representative:
            raise ValueError(f"Member assigned to multiple clusters: {member}")
        representatives.add(representative)
    if not membership:
        raise ValueError("MMseqs cluster mapping is empty")
    return membership, len(representatives)


def read_excluded_clusters(hits: Path, membership: dict[str, str]) -> set[str]:
    excluded: set[str] = set()
    for line_number, raw_line in enumerate(text_lines(hits), start=1):
        row = raw_line.rstrip("\n").split("\t")
        if len(row) != 8:
            raise ValueError(f"Malformed development hit row {line_number}")
        if (
            float(row[2]) < 0.5
            or float(row[4]) < 0.8
            or float(row[5]) < 0.8
        ):
            raise ValueError("Development hit violates the frozen homology gate")
        representative = membership.get(row[1])
        if representative is None:
            raise ValueError(f"Development-hit target lacks membership: {row[1]}")
        excluded.add(representative)
    return excluded


def decorate_edges(
    edges: Path,
    membership: dict[str, str],
    excluded: set[str],
    output: TextIO,
) -> dict[str, int]:
    counts = {
        "development_excluded_edges": 0,
        "eligible_nonself_edges": 0,
        "missing_endpoint_memberships": 0,
        "self_cluster_edges_removed": 0,
        "source_edges": 0,
    }
    missing_examples: list[str] = []
    rows = iter(text_lines(edges))
    try:
        header = tuple(next(rows).rstrip("\n").split())
    except StopIteration as error:
        raise ValueError("High-confidence edge file is empty") from error
    if header != EXPECTED_HEADER:
        raise ValueError(f"Unexpected high-confidence header: {header}")
    for line_number, raw_line in enumerate(rows, start=2):
        row = raw_line.rstrip("\n").split()
        if len(row) != len(EXPECTED_HEADER):
            raise ValueError(f"Malformed high-confidence edge row {line_number}")
        counts["source_edges"] += 1
        cluster1 = membership.get(row[0])
        cluster2 = membership.get(row[1])
        if cluster1 is None or cluster2 is None:
            counts["missing_endpoint_memberships"] += 1
            if len(missing_examples) < 5:
                missing_examples.append(row[0] if cluster1 is None else row[1])
            continue
        if cluster1 in excluded or cluster2 in excluded:
            counts["development_excluded_edges"] += 1
            continue
        if cluster1 == cluster2:
            counts["self_cluster_edges_removed"] += 1
            continue
        pair = sorted((cluster1, cluster2))
        combined = int(row[9])
        direct = int(row[3]) + int(row[5])
        output.write("\t".join((*pair, str(combined), str(direct), *row)) + "\n")
        counts["eligible_nonself_edges"] += 1
    if counts["missing_endpoint_memberships"]:
        raise ValueError(
            "High-confidence endpoints missing cluster memberships: "
            f"count={counts['missing_endpoint_memberships']} examples={missing_examples}"
        )
    return counts


def parse_stats(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        fields = raw_line.split("=", 1)
        if len(fields) != 2 or not fields[0] or not fields[1].isdigit():
            raise ValueError(f"Malformed scale-collapse statistic: {raw_line!r}")
        if fields[0] in counts:
            raise ValueError(f"Duplicate scale-collapse statistic: {fields[0]}")
        counts[fields[0]] = int(fields[1])
    if set(counts) != REQUIRED_COUNTS:
        raise ValueError("Scale-collapse statistics do not match the frozen schema")
    accounted = (
        counts["development_excluded_edges"]
        + counts["eligible_nonself_edges"]
        + counts["missing_endpoint_memberships"]
        + counts["self_cluster_edges_removed"]
    )
    if accounted != counts["source_edges"]:
        raise ValueError("Scale-collapse edge accounting is not exhaustive")
    if counts["missing_endpoint_memberships"] != 0:
        raise ValueError("Scale-collapse has missing endpoint memberships")
    if not 0 < counts["output_cluster_pairs"] <= counts["eligible_nonself_edges"]:
        raise ValueError("Scale-collapse output count is invalid")
    return counts


def publish_create_only(path: Path, value: object) -> None:
    payload = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_path, path)
        path.chmod(0o444)
    except FileExistsError as error:
        raise FileExistsError(f"Refusing to overwrite scale receipt: {path}") from error
    finally:
        temporary_path.unlink(missing_ok=True)


def decorate(arguments: argparse.Namespace) -> None:
    membership, cluster_count = read_membership(arguments.clusters.resolve())
    excluded = read_excluded_clusters(arguments.development_hits.resolve(), membership)
    counts = decorate_edges(
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


def finalize(arguments: argparse.Namespace) -> None:
    edges = arguments.edges.resolve()
    edge_receipt_path = arguments.edge_receipt.resolve()
    edge_receipt = json.loads(edge_receipt_path.read_text(encoding="utf-8"))
    if (
        edge_receipt.get("format_version")
        != "abpairfm_string_high_confidence_filter_v2"
        or edge_receipt.get("status")
        != "FILTERED_HIGH_CONFIDENCE_NOT_CLUSTERED_NOT_DECONTAMINATED"
        or edge_receipt.get("test_access") is not False
        or edge_receipt["output"]["sha256"] != sha256_file(edges)
    ):
        raise ValueError("High-confidence STRING edge evidence differs")

    cluster_receipt_path = arguments.cluster_receipt.resolve()
    cluster_receipt = json.loads(cluster_receipt_path.read_text(encoding="utf-8"))
    published = {
        asset["name"]: asset for asset in cluster_receipt.get("published_assets", [])
    }
    clusters = arguments.clusters.resolve()
    development_hits = arguments.development_hits.resolve()
    if (
        cluster_receipt.get("format_version")
        != "abpairfm_string_high_confidence_cluster_assets_v1"
        or cluster_receipt.get("status")
        != "CLUSTERED_NOT_COLLAPSED_NOT_DECONTAMINATED"
        or cluster_receipt.get("test_access") is not False
        or published.get("clusters.tsv.zst", {}).get("sha256")
        != sha256_file(clusters)
        or published.get("development_hits.tsv", {}).get("sha256")
        != sha256_file(development_hits)
    ):
        raise ValueError("High-confidence STRING cluster evidence differs")

    protocol_path = arguments.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    minimum = int(
        protocol["stage_a_scale_pool"]["eligible_pool_targets"][
            "minimum_endpoint_cluster_pairs"
        ]
    )
    if (
        protocol.get("format_version") != "abpairfm_3b_scale_expansion_v2"
        or protocol.get("authorization", {}).get("scale_pool_materialization")
        is not True
        or protocol.get("authorization", {}).get("production_pretraining")
        is not False
    ):
        raise ValueError("Scale protocol does not authorize pair collapse")

    raw_stats = parse_stats(arguments.stats.resolve())
    auxiliary: dict[str, int] = {}
    for raw_line in arguments.decorate_stats.resolve().read_text(
        encoding="utf-8"
    ).splitlines():
        key, value = raw_line.split("=", 1)
        auxiliary[key] = int(value)
    if set(auxiliary) != {
        "cluster_count",
        "development_excluded_clusters",
        "development_excluded_edges",
        "eligible_nonself_edges",
        "membership_count",
        "missing_endpoint_memberships",
        "self_cluster_edges_removed",
        "source_edges",
    }:
        raise ValueError("Decorator statistics do not match the frozen schema")
    for key in REQUIRED_COUNTS - {"output_cluster_pairs"}:
        if raw_stats[key] != auxiliary[key]:
            raise ValueError(f"Decorator and collapse statistics differ for {key}")
    if raw_stats["source_edges"] != int(
        edge_receipt["counts"]["high_confidence_rows"]
    ):
        raise ValueError("Collapsed source-edge count differs from filter receipt")
    if auxiliary["membership_count"] != int(
        cluster_receipt["cluster"]["membership_count"]
    ):
        raise ValueError("Collapsed membership count differs from cluster receipt")

    output = arguments.output.resolve()
    output_count = raw_stats["output_cluster_pairs"]
    scale_gate = output_count >= minimum
    receipt = {
        "authorization": {
            "production_pretraining": False,
            "scale_pool_sampling": scale_gate,
        },
        "claim_boundary": "One strongest calibrated STRING physical-network edge per unordered 50%-identity endpoint-cluster pair after removal of every endpoint cluster homologous to the frozen AsEP/SAbDab2 development panel. Production training and SOTA claims remain unauthorized.",
        "cluster": {
            "cluster_count": auxiliary["cluster_count"],
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
            "duplicate_cluster_pair_edges": raw_stats[
                "eligible_nonself_edges"
            ]
            - output_count,
        },
        "format_version": "abpairfm_string_high_confidence_scale_pool_v1",
        "inputs": {
            "cluster_receipt_sha256": sha256_file(cluster_receipt_path),
            "edge_receipt_sha256": sha256_file(edge_receipt_path),
            "protocol_sha256": sha256_file(protocol_path),
        },
        "output": {
            "bytes": output.stat().st_size,
            "name": arguments.output_name or output.name,
            "sha256": sha256_file(output),
        },
        "scale_pool_gate": {
            "minimum_cluster_pairs": minimum,
            "passed": scale_gate,
        },
        "status": (
            "DECONTAMINATED_SCALE_POOL_NOT_AUTHORIZED_FOR_TRAINING"
            if scale_gate
            else "DECONTAMINATED_SCALE_POOL_BELOW_MINIMUM"
        ),
        "test_access": False,
    }
    publish_create_only(arguments.receipt.resolve(), receipt)
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
    finalize_parser.add_argument("--development-hits", type=Path, required=True)
    finalize_parser.add_argument("--cluster-receipt", type=Path, required=True)
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
