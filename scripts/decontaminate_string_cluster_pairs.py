#!/usr/bin/env python3
"""Remove STRING endpoint clusters homologous to frozen validation sequences."""

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


def read_membership(path: Path) -> dict[str, str]:
    membership: dict[str, str] = {}
    with path.open("rt", encoding="utf-8", newline="") as handle:
        for line_number, row in enumerate(csv.reader(handle, delimiter="\t"), start=1):
            if len(row) != 2:
                raise ValueError(f"Malformed cluster mapping row {line_number}")
            representative, member = row
            previous = membership.setdefault(member, representative)
            if previous != representative:
                raise ValueError(f"Cluster membership differs for {member}")
    if not membership:
        raise ValueError("Cluster membership is empty")
    return membership


def excluded_clusters(
    hits_path: Path, membership: dict[str, str]
) -> tuple[set[str], dict[str, int]]:
    excluded: set[str] = set()
    queries: set[str] = set()
    targets: set[str] = set()
    hit_rows = 0
    with hits_path.open("rt", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for line_number, row in enumerate(reader, start=1):
            if len(row) != 8:
                raise ValueError(f"Malformed MMseqs hit row {line_number}")
            query, target = row[:2]
            identity = float(row[2])
            query_coverage = float(row[4])
            target_coverage = float(row[5])
            if identity < 0.5 or min(query_coverage, target_coverage) < 0.8:
                raise ValueError("MMseqs output violates the frozen identity/coverage gate")
            representative = membership.get(target)
            if representative is None:
                raise ValueError(f"MMseqs target lacks cluster membership: {target}")
            excluded.add(representative)
            queries.add(query)
            targets.add(target)
            hit_rows += 1
    return excluded, {
        "development_queries_with_hits": len(queries),
        "endpoint_sequences_with_hits": len(targets),
        "excluded_endpoint_clusters": len(excluded),
        "homology_hit_rows": hit_rows,
    }


def decompressed_rows(path: Path):
    process = subprocess.Popen(
        ["zstd", "-q", "-dc", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    assert process.stdout is not None
    reader = csv.reader(process.stdout, delimiter="\t")
    try:
        for row in reader:
            yield row
    finally:
        process.stdout.close()
        stderr = process.stderr.read() if process.stderr is not None else ""
        return_code = process.wait()
        if return_code:
            raise RuntimeError(f"zstd failed with {return_code}: {stderr[-1000:]}")


def write_filtered(
    input_path: Path, output_path: Path, excluded: set[str], threads: int
) -> tuple[int, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{output_path.name}.", dir=output_path.parent)
    os.close(descriptor)
    temporary_path = Path(temporary)
    input_count = 0
    output_count = 0
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
            rows = decompressed_rows(input_path)
            header = next(rows)
            if len(header) < 4 or header[:2] != ["cluster1", "cluster2"]:
                raise ValueError("Unexpected clustered STRING header")
            writer.writerow(header)
            for row in rows:
                if len(row) != len(header):
                    raise ValueError("Malformed clustered STRING edge row")
                input_count += 1
                if row[0] in excluded or row[1] in excluded:
                    continue
                writer.writerow(row)
                output_count += 1
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
        raise FileExistsError(f"Refusing to overwrite decontaminated STRING edges: {output_path}") from error
    finally:
        temporary_path.unlink(missing_ok=True)
    return input_count, output_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clustered-edges", type=Path, required=True)
    parser.add_argument("--cluster-receipt", type=Path, required=True)
    parser.add_argument("--clusters-tsv", type=Path, required=True)
    parser.add_argument("--development-fasta", type=Path, required=True)
    parser.add_argument("--development-receipt", type=Path, required=True)
    parser.add_argument("--hits", type=Path, required=True)
    parser.add_argument("--hits-output", type=Path, required=True)
    parser.add_argument("--excluded-clusters-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--mmseqs-version", required=True)
    parser.add_argument("--threads", type=int, default=6)
    arguments = parser.parse_args()

    cluster_receipt_path = arguments.cluster_receipt.resolve()
    cluster_receipt = json.loads(cluster_receipt_path.read_text(encoding="utf-8"))
    if cluster_receipt.get("status") != "CLUSTERED_NOT_DECONTAMINATED":
        raise ValueError("STRING cluster receipt is not eligible")
    if cluster_receipt.get("test_access") is not False:
        raise ValueError("STRING cluster receipt crossed a test boundary")
    clustered_edges = arguments.clustered_edges.resolve()
    if sha256_file(clustered_edges) != cluster_receipt["output"]["sha256"]:
        raise ValueError("Clustered STRING edge SHA-256 differs")

    development_receipt_path = arguments.development_receipt.resolve()
    development_receipt = json.loads(development_receipt_path.read_text(encoding="utf-8"))
    if development_receipt.get("status") != "FROZEN_VALIDATION_SEQUENCE_PANEL":
        raise ValueError("Development panel receipt is not eligible")
    if development_receipt.get("test_access") is not False:
        raise ValueError("Development panel crossed a test boundary")
    development_fasta = arguments.development_fasta.resolve()
    if sha256_file(development_fasta) != development_receipt["output_fasta"]["sha256"]:
        raise ValueError("Development FASTA SHA-256 differs")

    clusters_path = arguments.clusters_tsv.resolve()
    membership = read_membership(clusters_path)
    hits_path = arguments.hits.resolve()
    excluded, hit_counts = excluded_clusters(hits_path, membership)
    excluded_payload = "".join(f"{value}\n" for value in sorted(excluded)).encode("utf-8")
    publish_create_only(arguments.excluded_clusters_output.resolve(), excluded_payload)
    hits_output = arguments.hits_output.resolve()
    try:
        os.link(hits_path, hits_output)
        hits_output.chmod(0o444)
    except FileExistsError as error:
        raise FileExistsError(f"Refusing to overwrite MMseqs hit evidence: {hits_output}") from error

    output = arguments.output.resolve()
    input_count, output_count = write_filtered(
        clustered_edges, output, excluded, arguments.threads
    )
    if input_count != int(cluster_receipt["counts"]["unique_nonself_cluster_pairs"]):
        raise ValueError("Clustered STRING edge count differs from receipt")
    minimum, maximum = 2_500_000, 25_000_000
    receipt = {
        "authorization": {"stage_a_training": False},
        "claim_boundary": "STRING cluster pairs after removal of every endpoint cluster with a >=50% identity and >=80% bidirectional-coverage hit to the frozen AsEP/SAbDab2 validation panel. PINDER merge and final Stage A authorization remain pending.",
        "counts": {
            **hit_counts,
            "input_cluster_pairs": input_count,
            "output_cluster_pairs": output_count,
            "removed_cluster_pairs": input_count - output_count,
        },
        "execution": {
            "allocated_gpu_count": int(os.environ.get("ABPAIRFM_GPU_COUNT", "0")),
            "allocated_gpu_name": os.environ.get("ABPAIRFM_GPU_NAME", "UNKNOWN"),
            "gpu_compute_used": False,
            "job_id": int(os.environ.get("SLURM_JOB_ID", "0")),
            "partition": os.environ.get("SLURM_JOB_PARTITION", "UNKNOWN"),
        },
        "format_version": "abpairfm_string_development_decontamination_v1",
        "homology_gate": {
            "coverage": 0.8,
            "coverage_mode": 0,
            "minimum_sequence_identity": 0.5,
            "mmseqs_version": arguments.mmseqs_version,
        },
        "input_receipts": {
            "cluster_sha256": sha256_file(cluster_receipt_path),
            "development_panel_sha256": sha256_file(development_receipt_path),
        },
        "output": {
            "bytes": output.stat().st_size,
            "sha256": sha256_file(output),
        },
        "published_evidence": {
            "excluded_clusters_sha256": sha256_file(arguments.excluded_clusters_output.resolve()),
            "hits_sha256": sha256_file(hits_output),
        },
        "stage_a_example_range": {"maximum": maximum, "minimum": minimum},
        "stage_a_range_gate": minimum <= output_count <= maximum,
        "status": "DECONTAMINATED_NOT_MERGED_NOT_AUTHORIZED",
        "test_access": False,
    }
    publish_create_only(
        arguments.receipt.resolve(),
        (json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n").encode(),
    )
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
