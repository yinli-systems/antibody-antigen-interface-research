#!/usr/bin/env python3
"""Count decontaminated, tier-1-novel STRING supplemental cluster pairs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def zstd_lines(path: Path):
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


def read_membership(path: Path) -> dict[str, str]:
    membership: dict[str, str] = {}
    with path.open("rt", encoding="utf-8", newline="") as handle:
        for line_number, row in enumerate(csv.reader(handle, delimiter="\t"), start=1):
            if len(row) != 2:
                raise ValueError(f"Malformed STRING cluster mapping row {line_number}")
            representative, member = row
            previous = membership.setdefault(member, representative)
            if previous != representative:
                raise ValueError(f"STRING member has multiple clusters: {member}")
    if not membership:
        raise ValueError("STRING cluster mapping is empty")
    return membership


def read_tier1_pairs(path: Path) -> set[tuple[str, str]]:
    rows = iter(zstd_lines(path))
    header = next(rows).rstrip("\n").split("\t")
    if header[:2] != ["cluster1", "cluster2"]:
        raise ValueError("Unexpected STRING tier-1 clustered header")
    pairs: set[tuple[str, str]] = set()
    for line in rows:
        row = line.rstrip("\n").split("\t")
        if len(row) != len(header):
            raise ValueError("Malformed STRING tier-1 cluster pair")
        pair = tuple(sorted((row[0], row[1])))
        if pair in pairs or pair[0] == pair[1]:
            raise ValueError("STRING tier-1 pairs are duplicated or self-clustered")
        pairs.add(pair)
    return pairs


def read_excluded_clusters(path: Path) -> set[str]:
    values = {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
    if not values:
        raise ValueError("STRING development-excluded cluster inventory is empty")
    return values


def candidate_mask(combined: int, direct: int, candidates: list[dict[str, int]]) -> int:
    mask = 0
    for index, candidate in enumerate(candidates):
        if (
            combined >= int(candidate["combined_minimum_inclusive"])
            and combined < int(candidate["combined_maximum_exclusive"])
            and direct >= int(candidate["direct_channel_maximum_minimum_inclusive"])
        ):
            mask |= 1 << index
    return mask


def audit(
    universe: Path,
    membership: dict[str, str],
    tier1_pairs: set[tuple[str, str]],
    excluded_clusters: set[str],
    candidates: list[dict[str, int]],
) -> tuple[dict[str, int], list[int]]:
    counts = {
        "candidate_edges": 0,
        "current_endpoint_edges": 0,
        "development_excluded_edges": 0,
        "missing_endpoint_edges": 0,
        "self_cluster_edges": 0,
        "tier1_overlap_edges": 0,
    }
    masks: dict[tuple[str, str], int] = {}
    rows = iter(zstd_lines(universe))
    header = next(rows).rstrip("\n").split()
    expected = [
        "protein1", "protein2", "homology", "experiments", "experiments_transferred",
        "database", "database_transferred", "textmining", "textmining_transferred", "combined_score",
    ]
    if header != expected:
        raise ValueError("Unexpected STRING supplemental header")
    for line in rows:
        row = line.rstrip("\n").split()
        if len(row) != len(expected):
            raise ValueError("Malformed STRING supplemental edge")
        counts["candidate_edges"] += 1
        cluster1 = membership.get(row[0])
        cluster2 = membership.get(row[1])
        if cluster1 is None or cluster2 is None:
            counts["missing_endpoint_edges"] += 1
            continue
        counts["current_endpoint_edges"] += 1
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
        direct = max(int(row[3]), int(row[5]))
        mask = candidate_mask(int(row[9]), direct, candidates)
        if mask:
            masks[pair] = masks.get(pair, 0) | mask
    counts["novel_cluster_pairs_in_broad_universe"] = len(masks)
    per_candidate = [sum(bool(mask & (1 << index)) for mask in masks.values()) for index in range(len(candidates))]
    return counts, per_candidate


def publish_create_only(path: Path, value: object) -> None:
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
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
        raise FileExistsError(f"Refusing to overwrite STRING supplemental audit: {path}") from error
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", type=Path, required=True)
    parser.add_argument("--universe-receipt", type=Path, required=True)
    parser.add_argument("--threshold-grid-receipt", type=Path, required=True)
    parser.add_argument("--clusters", type=Path, required=True)
    parser.add_argument("--tier1", type=Path, required=True)
    parser.add_argument("--tier1-receipt", type=Path, required=True)
    parser.add_argument("--excluded-clusters", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    universe = arguments.universe.resolve()
    universe_receipt_path = arguments.universe_receipt.resolve()
    universe_receipt = json.loads(universe_receipt_path.read_text(encoding="utf-8"))
    if universe_receipt.get("status") != "FILTERED_UNIVERSE_NOT_SELECTED_NOT_CLUSTERED" or universe_receipt.get("test_access") is not False or sha256_file(universe) != universe_receipt["output"]["sha256"]:
        raise ValueError("STRING supplemental universe evidence differs")
    grid_path = arguments.threshold_grid_receipt.resolve()
    grid = json.loads(grid_path.read_text(encoding="utf-8"))
    candidates = grid.get("ranked_candidates")
    if grid.get("status") != "AUDITED_NOT_SELECTED_NOT_AUTHORIZED" or not isinstance(candidates, list) or len(candidates) != 10:
        raise ValueError("STRING threshold-grid evidence differs")
    if universe_receipt["input_receipts"]["threshold_grid_sha256"] != sha256_file(grid_path):
        raise ValueError("STRING supplemental universe is not bound to the threshold grid")
    tier1 = arguments.tier1.resolve()
    tier1_receipt_path = arguments.tier1_receipt.resolve()
    tier1_receipt = json.loads(tier1_receipt_path.read_text(encoding="utf-8"))
    if tier1_receipt.get("status") != "DECONTAMINATED_NOT_MERGED_NOT_AUTHORIZED" or tier1_receipt.get("test_access") is not False or sha256_file(tier1) != tier1_receipt["output"]["sha256"]:
        raise ValueError("STRING tier-1 evidence differs")
    membership = read_membership(arguments.clusters.resolve())
    tier1_pairs = read_tier1_pairs(tier1)
    if len(tier1_pairs) != int(tier1_receipt["counts"]["output_cluster_pairs"]):
        raise ValueError("STRING tier-1 pair count differs")
    excluded = read_excluded_clusters(arguments.excluded_clusters.resolve())
    counts, candidate_counts = audit(universe, membership, tier1_pairs, excluded, candidates)
    if counts["candidate_edges"] != universe_receipt["counts"]["candidate_rows"]:
        raise ValueError("STRING supplemental edge count differs")
    minimum_string_pairs = 2_500_000
    required = max(0, minimum_string_pairs - len(tier1_pairs))
    ranked = []
    selected = None
    for candidate, pair_count in zip(candidates, candidate_counts, strict=True):
        record = {**candidate, "decontaminated_tier1_novel_current_endpoint_cluster_pairs": pair_count}
        ranked.append(record)
        if selected is None and pair_count >= required:
            selected = record
    receipt = {
        "authorization": {
            "expanded_endpoint_reclustering_required": selected is None,
            "selected_candidate_materialization": selected is not None,
            "stage_a_training": False,
        },
        "claim_boundary": "Exact lower-bound audit using the existing 50%-identity STRING endpoint clusters: only supplemental edges whose endpoints were already authenticated in tier 1 are counted, after development exclusion and tier-1 pair deduplication. No supplemental training artifact is authorized yet.",
        "counts": {
            **counts,
            "current_tier1_cluster_pairs": len(tier1_pairs),
            "required_novel_cluster_pairs_for_string_only_gate": required,
        },
        "format_version": "abpairfm_string_supplemental_cluster_pair_audit_v1",
        "inputs": {
            "threshold_grid_sha256": sha256_file(grid_path),
            "tier1_receipt_sha256": sha256_file(tier1_receipt_path),
            "universe_receipt_sha256": sha256_file(universe_receipt_path),
        },
        "ranked_candidates": ranked,
        "selected_candidate": selected,
        "selection_rule": "First preregistered candidate yielding at least 2,500,000 decontaminated STRING cluster pairs from tier 1 plus a tier-1-novel supplemental lower bound; structural sources are not needed to mask cross-source overlap.",
        "status": "AUDITED_SELECTION_PENDING_MATERIALIZATION" if selected is not None else "AUDITED_EXPANDED_CLUSTERING_REQUIRED",
        "test_access": False,
    }
    publish_create_only(arguments.output.resolve(), receipt)
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
