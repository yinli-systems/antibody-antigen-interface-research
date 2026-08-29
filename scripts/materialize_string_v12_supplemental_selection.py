#!/usr/bin/env python3
"""Materialize the preregistered STRING supplemental cluster-pair selection."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Iterable


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def zstd_rows(path: Path, delimiter: str) -> Iterable[list[str]]:
    process = subprocess.Popen(
        ["zstd", "-q", "-dc", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    assert process.stdout is not None
    reader = csv.reader(process.stdout, delimiter=delimiter)
    try:
        yield from reader
    finally:
        process.stdout.close()
        error = process.stderr.read() if process.stderr is not None else ""
        return_code = process.wait()
        if return_code:
            raise RuntimeError(f"zstd failed with {return_code}: {error[-1000:]}")


def read_membership(path: Path) -> dict[str, str]:
    membership: dict[str, str] = {}
    rows = zstd_rows(path, "\t") if path.suffix == ".zst" else csv.reader(
        path.open("rt", encoding="utf-8", newline=""), delimiter="\t"
    )
    for line_number, row in enumerate(rows, start=1):
        if len(row) != 2:
            raise ValueError(f"Malformed STRING cluster mapping row {line_number}")
        representative, member = row
        previous = membership.setdefault(member, representative)
        if previous != representative:
            raise ValueError(f"STRING member has multiple clusters: {member}")
    if not membership:
        raise ValueError("STRING cluster mapping is empty")
    return membership


def read_tier1_pairs(path: Path) -> tuple[set[tuple[str, str]], tuple[str, ...]]:
    rows = iter(zstd_rows(path, "\t"))
    header = tuple(next(rows))
    if header[:2] != ("cluster1", "cluster2"):
        raise ValueError("Unexpected STRING tier-1 clustered header")
    pairs: set[tuple[str, str]] = set()
    for row in rows:
        if len(row) != len(header):
            raise ValueError("Malformed STRING tier-1 cluster pair")
        pair = tuple(sorted((row[0], row[1])))
        if pair[0] == pair[1] or pair in pairs:
            raise ValueError("STRING tier-1 pairs are duplicated or self-clustered")
        pairs.add(pair)
    return pairs, header


def candidate_eligible(row: list[str], candidate: dict[str, int]) -> bool:
    combined = int(row[9])
    direct = max(int(row[3]), int(row[5]))
    return (
        combined >= int(candidate["combined_minimum_inclusive"])
        and combined < int(candidate["combined_maximum_exclusive"])
        and direct >= int(candidate["direct_channel_maximum_minimum_inclusive"])
    )


def select_supplemental_rows(
    rows: Iterable[list[str]],
    membership: dict[str, str],
    tier1_pairs: set[tuple[str, str]],
    excluded_clusters: set[str],
    candidate: dict[str, int],
) -> tuple[
    dict[tuple[str, str], tuple[tuple[object, ...], tuple[str, ...]]],
    dict[str, int],
]:
    selected: dict[
        tuple[str, str], tuple[tuple[object, ...], tuple[str, ...]]
    ] = {}
    counts = {
        "candidate_edges": 0,
        "development_excluded_edges": 0,
        "duplicate_candidate_cluster_pair_edges": 0,
        "eligible_candidate_edges": 0,
        "missing_endpoint_edges": 0,
        "self_cluster_edges": 0,
        "tier1_overlap_edges": 0,
    }
    for row in rows:
        if len(row) != len(EXPECTED_HEADER):
            raise ValueError("Malformed STRING supplemental edge")
        counts["candidate_edges"] += 1
        cluster1 = membership.get(row[0])
        cluster2 = membership.get(row[1])
        if cluster1 is None or cluster2 is None:
            counts["missing_endpoint_edges"] += 1
            continue
        if cluster1 in excluded_clusters or cluster2 in excluded_clusters:
            counts["development_excluded_edges"] += 1
            continue
        if cluster1 == cluster2:
            counts["self_cluster_edges"] += 1
            continue
        pair = tuple(sorted((cluster1, cluster2)))
        if pair in tier1_pairs:
            counts["tier1_overlap_edges"] += 1
            continue
        if not candidate_eligible(row, candidate):
            continue
        counts["eligible_candidate_edges"] += 1
        combined = int(row[9])
        direct_sum = int(row[3]) + int(row[5])
        rank: tuple[object, ...] = (-combined, -direct_sum, row[0], row[1], *row[2:])
        previous = selected.get(pair)
        if previous is not None:
            counts["duplicate_candidate_cluster_pair_edges"] += 1
        if previous is None or rank < previous[0]:
            selected[pair] = (rank, tuple(row))
    counts["selected_supplemental_cluster_pairs"] = len(selected)
    return selected, counts


def write_zstd(
    output: Path,
    selected: dict[tuple[str, str], tuple[tuple[object, ...], tuple[str, ...]]],
    threads: int,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        with temporary_path.open("wb") as output_handle:
            process = subprocess.Popen(
                ["zstd", "-q", f"-T{threads}", "-19", "-c"],
                stdin=subprocess.PIPE,
                stdout=output_handle,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
            )
            assert process.stdin is not None
            writer = csv.writer(process.stdin, delimiter="\t", lineterminator="\n")
            writer.writerow(("cluster1", "cluster2", *EXPECTED_HEADER))
            for pair in sorted(selected):
                writer.writerow((*pair, *selected[pair][1]))
            process.stdin.close()
            error = process.stderr.read() if process.stderr is not None else ""
            return_code = process.wait()
            if return_code:
                raise RuntimeError(f"zstd failed with {return_code}: {error[-1000:]}")
            output_handle.flush()
            os.fsync(output_handle.fileno())
        os.link(temporary_path, output)
        output.chmod(0o444)
    except FileExistsError as error:
        raise FileExistsError(f"Refusing to overwrite supplemental output: {output}") from error
    finally:
        temporary_path.unlink(missing_ok=True)


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
        raise FileExistsError(f"Refusing to overwrite supplemental receipt: {path}") from error
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", type=Path, required=True)
    parser.add_argument("--universe-receipt", type=Path, required=True)
    parser.add_argument("--selection-audit", type=Path, required=True)
    parser.add_argument("--clusters", type=Path, required=True)
    parser.add_argument("--cluster-receipt", type=Path, required=True)
    parser.add_argument("--tier1", type=Path, required=True)
    parser.add_argument("--tier1-receipt", type=Path, required=True)
    parser.add_argument("--excluded-clusters", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=6)
    arguments = parser.parse_args()
    if arguments.threads < 1:
        raise ValueError("threads must be positive")

    universe = arguments.universe.resolve()
    universe_receipt_path = arguments.universe_receipt.resolve()
    universe_receipt = json.loads(universe_receipt_path.read_text(encoding="utf-8"))
    if (
        universe_receipt.get("status") != "FILTERED_UNIVERSE_NOT_SELECTED_NOT_CLUSTERED"
        or universe_receipt.get("test_access") is not False
        or sha256_file(universe) != universe_receipt["output"]["sha256"]
    ):
        raise ValueError("STRING supplemental universe evidence differs")

    audit_path = arguments.selection_audit.resolve()
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    candidate = audit.get("selected_candidate")
    if (
        audit.get("format_version") != "abpairfm_string_supplemental_cluster_pair_audit_v2"
        or audit.get("status") != "AUDITED_SELECTION_PENDING_MATERIALIZATION"
        or audit.get("test_access") is not False
        or not isinstance(candidate, dict)
    ):
        raise ValueError("STRING supplemental selection audit is not eligible")

    cluster_receipt_path = arguments.cluster_receipt.resolve()
    cluster_receipt = json.loads(cluster_receipt_path.read_text(encoding="utf-8"))
    clusters = arguments.clusters.resolve()
    published_clusters = {
        value.get("name"): value
        for value in cluster_receipt.get("published_assets", [])
    }.get("clusters.tsv.zst")
    if (
        cluster_receipt.get("format_version") != "abpairfm_string_cluster_pair_collapse_v1"
        or not isinstance(published_clusters, dict)
        or published_clusters.get("sha256") != sha256_file(clusters)
    ):
        raise ValueError("STRING expanded cluster evidence differs")

    tier1 = arguments.tier1.resolve()
    tier1_receipt_path = arguments.tier1_receipt.resolve()
    tier1_receipt = json.loads(tier1_receipt_path.read_text(encoding="utf-8"))
    excluded_path = arguments.excluded_clusters.resolve()
    if (
        tier1_receipt.get("status") != "DECONTAMINATED_NOT_MERGED_NOT_AUTHORIZED"
        or tier1_receipt.get("test_access") is not False
        or tier1_receipt["output"]["sha256"] != sha256_file(tier1)
        or tier1_receipt["published_evidence"]["excluded_clusters_sha256"]
        != sha256_file(excluded_path)
    ):
        raise ValueError("STRING tier-1 evidence differs")
    expected_audit_inputs = {
        "cluster_receipt_sha256": sha256_file(cluster_receipt_path),
        "tier1_receipt_sha256": sha256_file(tier1_receipt_path),
        "universe_receipt_sha256": sha256_file(universe_receipt_path),
    }
    if any(audit["inputs"].get(key) != value for key, value in expected_audit_inputs.items()):
        raise ValueError("STRING selection audit input binding differs")

    membership = read_membership(clusters)
    tier1_pairs, _ = read_tier1_pairs(tier1)
    excluded = {
        line.strip()
        for line in excluded_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if not excluded:
        raise ValueError("STRING development-excluded cluster inventory is empty")
    universe_rows = iter(zstd_rows(universe, " "))
    if tuple(next(universe_rows)) != EXPECTED_HEADER:
        raise ValueError("Unexpected STRING supplemental header")
    selected, counts = select_supplemental_rows(
        universe_rows,
        membership,
        tier1_pairs,
        excluded,
        candidate,
    )
    expected_selected = int(
        candidate["decontaminated_tier1_novel_current_endpoint_cluster_pairs"]
    )
    if counts["missing_endpoint_edges"] != 0:
        raise ValueError("Expanded cluster membership still omits supplemental endpoints")
    if counts["candidate_edges"] != int(universe_receipt["counts"]["candidate_rows"]):
        raise ValueError("STRING supplemental source count differs")
    if len(selected) != expected_selected:
        raise ValueError("Materialized supplemental pair count differs from frozen audit")
    if len(tier1_pairs) != int(tier1_receipt["counts"]["output_cluster_pairs"]):
        raise ValueError("STRING tier-1 pair count differs")
    combined_count = len(tier1_pairs) + len(selected)
    if not 2_500_000 <= combined_count <= 25_000_000:
        raise ValueError("Materialized STRING pair count misses the frozen Stage A range")

    output = arguments.output.resolve()
    write_zstd(output, selected, arguments.threads)
    receipt = {
        "authorization": {
            "stage_a_training": False,
            "string_pair_count_gate": True,
        },
        "claim_boundary": "Preregistered development-clean supplemental STRING cluster-pair shard plus the authenticated tier-1 count. Cross-source final inventory, real-data training, biological improvement, and SOTA remain unauthorized.",
        "counts": {
            **counts,
            "combined_string_cluster_pairs": combined_count,
            "tier1_cluster_pairs": len(tier1_pairs),
        },
        "format_version": "abpairfm_string_supplemental_materialization_v1",
        "inputs": {
            "cluster_receipt_sha256": sha256_file(cluster_receipt_path),
            "selection_audit_sha256": sha256_file(audit_path),
            "tier1_receipt_sha256": sha256_file(tier1_receipt_path),
            "universe_receipt_sha256": sha256_file(universe_receipt_path),
        },
        "output": {
            "bytes": output.stat().st_size,
            "name": output.name,
            "sha256": sha256_file(output),
        },
        "selected_candidate": candidate,
        "status": "MATERIALIZED_STRING_PAIR_GATE_PASS_NOT_STAGE_A_AUTHORIZED",
        "test_access": False,
    }
    publish_create_only(arguments.receipt.resolve(), receipt)
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
