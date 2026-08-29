#!/usr/bin/env python3
"""Keep one strongest direct STRING edge per unordered 50%-identity cluster pair."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile


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


def publish_create_only(path: Path, payload: bytes, mode: int = 0o444) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_path, path)
        path.chmod(mode)
    except FileExistsError as error:
        raise FileExistsError(f"Refusing to overwrite artifact: {path}") from error
    finally:
        temporary_path.unlink(missing_ok=True)


def read_clusters(path: Path) -> tuple[dict[str, str], int]:
    membership: dict[str, str] = {}
    representatives: set[str] = set()
    with path.open("rt", encoding="utf-8", newline="") as handle:
        for line_number, row in enumerate(csv.reader(handle, delimiter="\t"), start=1):
            if len(row) != 2 or not row[0] or not row[1]:
                raise ValueError(f"Malformed MMseqs cluster row {line_number}")
            representative, member = row
            previous = membership.setdefault(member, representative)
            if previous != representative:
                raise ValueError(f"Member assigned to multiple clusters: {member}")
            representatives.add(representative)
    if not membership:
        raise ValueError("MMseqs cluster mapping is empty")
    return membership, len(representatives)


def direct_rows(path: Path):
    process = subprocess.Popen(
        ["zstd", "-q", "-dc", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    assert process.stdout is not None
    reader = csv.reader(process.stdout, delimiter=" ")
    header = tuple(next(reader))
    if header != EXPECTED_HEADER:
        process.kill()
        raise ValueError(f"Unexpected direct-edge header: {header}")
    try:
        for row in reader:
            if len(row) != len(EXPECTED_HEADER):
                raise ValueError("Malformed direct STRING edge row")
            yield row
    finally:
        process.stdout.close()
        stderr = process.stderr.read() if process.stderr is not None else ""
        return_code = process.wait()
        if return_code:
            raise RuntimeError(f"zstd failed with {return_code}: {stderr[-1000:]}")


def collapse(
    input_path: Path, membership: dict[str, str]
) -> tuple[dict[tuple[str, str], tuple[tuple[object, ...], tuple[str, ...]]], dict[str, int]]:
    selected: dict[
        tuple[str, str], tuple[tuple[object, ...], tuple[str, ...]]
    ] = {}
    counts = {
        "direct_edges": 0,
        "duplicate_cluster_pair_edges": 0,
        "missing_endpoint_memberships": 0,
        "self_cluster_edges_removed": 0,
    }
    missing_examples: list[str] = []
    for row in direct_rows(input_path):
        counts["direct_edges"] += 1
        protein1, protein2 = row[:2]
        cluster1 = membership.get(protein1)
        cluster2 = membership.get(protein2)
        if cluster1 is None or cluster2 is None:
            counts["missing_endpoint_memberships"] += 1
            if len(missing_examples) < 5:
                missing_examples.append(protein1 if cluster1 is None else protein2)
            continue
        if cluster1 == cluster2:
            counts["self_cluster_edges_removed"] += 1
            continue
        pair = tuple(sorted((cluster1, cluster2)))
        combined = int(row[9])
        direct = int(row[3]) + int(row[5])
        rank: tuple[object, ...] = (-combined, -direct, protein1, protein2, *row[2:])
        candidate = (rank, tuple(row))
        previous = selected.get(pair)
        if previous is None or rank < previous[0]:
            selected[pair] = candidate
        if previous is not None:
            counts["duplicate_cluster_pair_edges"] += 1
    if counts["missing_endpoint_memberships"]:
        raise ValueError(
            "Direct endpoints missing from MMseqs mapping: "
            f"count={counts['missing_endpoint_memberships']} examples={missing_examples}"
        )
    counts["unique_nonself_cluster_pairs"] = len(selected)
    return selected, counts


def write_zstd(
    output_path: Path,
    selected: dict[tuple[str, str], tuple[tuple[object, ...], tuple[str, ...]]],
    threads: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{output_path.name}.", dir=output_path.parent)
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
            stderr = process.stderr.read() if process.stderr is not None else ""
            return_code = process.wait()
            if return_code:
                raise RuntimeError(f"zstd failed with {return_code}: {stderr[-1000:]}")
            output_handle.flush()
            os.fsync(output_handle.fileno())
        os.link(temporary_path, output_path)
        output_path.chmod(0o444)
    except FileExistsError as error:
        raise FileExistsError(f"Refusing to overwrite collapsed edges: {output_path}") from error
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--direct-edges", type=Path, required=True)
    parser.add_argument("--direct-receipt", type=Path, required=True)
    parser.add_argument("--endpoint-receipt", type=Path, required=True)
    parser.add_argument("--clusters-tsv", type=Path, required=True)
    parser.add_argument("--mmseqs", type=Path, required=True)
    parser.add_argument("--asset", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=6)
    arguments = parser.parse_args()
    if arguments.threads < 1:
        raise ValueError("threads must be positive")

    direct_path = arguments.direct_edges.resolve()
    direct_receipt_path = arguments.direct_receipt.resolve()
    direct_receipt = json.loads(direct_receipt_path.read_text(encoding="utf-8"))
    if direct_receipt.get("status") != "FILTERED_NOT_CLUSTERED_NOT_DECONTAMINATED":
        raise ValueError("Direct STRING receipt is not eligible")
    if direct_receipt.get("test_access") is not False:
        raise ValueError("Direct STRING receipt crossed a test boundary")
    if sha256_file(direct_path) != direct_receipt["output"]["sha256"]:
        raise ValueError("Direct STRING edge SHA-256 differs")

    endpoint_receipt_path = arguments.endpoint_receipt.resolve()
    endpoint_receipt = json.loads(endpoint_receipt_path.read_text(encoding="utf-8"))
    if endpoint_receipt.get("status") != "ENDPOINT_SEQUENCES_EXTRACTED_NOT_CLUSTERED":
        raise ValueError("Endpoint receipt is not eligible")
    if endpoint_receipt.get("test_access") is not False:
        raise ValueError("Endpoint receipt crossed a test boundary")

    clusters_path = arguments.clusters_tsv.resolve()
    membership, cluster_count = read_clusters(clusters_path)
    expected_endpoints = int(endpoint_receipt["counts"]["endpoint_ids"])
    if len(membership) != expected_endpoints:
        raise ValueError(
            f"MMseqs membership count differs: {len(membership)} != {expected_endpoints}"
        )
    selected, counts = collapse(direct_path, membership)
    output = arguments.output.resolve()
    write_zstd(output, selected, arguments.threads)
    mmseqs_version = subprocess.check_output(
        [str(arguments.mmseqs.resolve()), "version"], text=True
    ).strip()
    minimum, maximum = 2_500_000, 25_000_000
    output_count = counts["unique_nonself_cluster_pairs"]
    receipt = {
        "authorization": {
            "development_decontamination": False,
            "stage_a_training": False,
        },
        "claim_boundary": "One strongest direct-evidence edge per unordered 50%-identity STRING endpoint-cluster pair; development sequences have not yet been excluded.",
        "cluster": {
            "cluster_count": cluster_count,
            "coverage": 0.8,
            "coverage_mode": 0,
            "membership_count": len(membership),
            "minimum_sequence_identity": 0.5,
            "mmseqs_version": mmseqs_version,
        },
        "counts": counts,
        "execution": {
            "allocated_gpu_count": int(os.environ.get("ABPAIRFM_GPU_COUNT", "0")),
            "allocated_gpu_name": os.environ.get("ABPAIRFM_GPU_NAME", "UNKNOWN"),
            "gpu_compute_used": False,
            "job_id": int(os.environ.get("SLURM_JOB_ID", "0")),
            "partition": os.environ.get("SLURM_JOB_PARTITION", "UNKNOWN"),
            "reason_for_gpu_allocation": "The gpu_4090 partition rejects jobs that request zero GPUs.",
        },
        "format_version": "abpairfm_string_cluster_pair_collapse_v1",
        "input_receipts": {
            "direct_filter_sha256": sha256_file(direct_receipt_path),
            "endpoint_fasta_sha256": sha256_file(endpoint_receipt_path),
        },
        "published_assets": [
            {"bytes": path.resolve().stat().st_size, "name": path.name, "sha256": sha256_file(path.resolve())}
            for path in arguments.asset
        ],
        "stage_a_example_range": {"maximum": maximum, "minimum": minimum},
        "stage_a_range_gate": minimum <= output_count <= maximum,
        "output": {
            "bytes": output.stat().st_size,
            "name": output.name,
            "sha256": sha256_file(output),
        },
        "status": "CLUSTERED_NOT_DECONTAMINATED",
        "test_access": False,
    }
    publish_create_only(
        arguments.receipt.resolve(),
        (json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n").encode(),
    )
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
