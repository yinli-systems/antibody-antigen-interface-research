#!/usr/bin/env python3
"""Build a deterministic train-only STRING slice for the ABPairFM 3B real gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import os
from pathlib import Path
import statistics
import subprocess
import tempfile
from typing import Iterable


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def zstd_text_lines(path: Path) -> Iterable[str]:
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


def pair_hash(seed: int, cluster1: str, cluster2: str) -> int:
    payload = f"{seed}\0{cluster1}\0{cluster2}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest(), "big")


def select_candidate_rows(
    rows: Iterable[list[str]], *, seed: int, candidate_count: int
) -> tuple[list[tuple[int, tuple[str, ...]]], int]:
    heap: list[tuple[int, tuple[str, ...]]] = []
    source_rows = 0
    for row in rows:
        if len(row) != 12:
            raise ValueError("Malformed development-clean STRING cluster pair")
        cluster1, cluster2 = row[:2]
        if not cluster1 or not cluster2 or cluster1 >= cluster2:
            raise ValueError("STRING cluster pair is not canonical")
        source_rows += 1
        digest = pair_hash(seed, cluster1, cluster2)
        item = (-digest, tuple(row))
        if len(heap) < candidate_count:
            heapq.heappush(heap, item)
        elif digest < -heap[0][0]:
            heapq.heapreplace(heap, item)
    if len(heap) != candidate_count:
        raise ValueError("STRING source has fewer rows than the candidate pool")
    selected = sorted((-negative, row) for negative, row in heap)
    return selected, source_rows


def read_selected_sequences(path: Path, targets: set[str]) -> dict[str, str]:
    sequences: dict[str, str] = {}
    current: str | None = None
    chunks: list[str] = []

    def emit() -> None:
        if current is None or current not in targets:
            return
        value = "".join(chunks).upper()
        if not value or any(character.isspace() for character in value):
            raise ValueError(f"Malformed STRING sequence for {current}")
        if current in sequences:
            raise ValueError(f"Duplicate STRING sequence for {current}")
        sequences[current] = value

    for line in zstd_text_lines(path):
        if line.startswith(">"):
            emit()
            current = line[1:].split(maxsplit=1)[0]
            chunks = []
        else:
            chunks.append(line.strip())
    emit()
    missing = targets - sequences.keys()
    if missing:
        raise ValueError(f"Selected STRING endpoints lack sequences: {sorted(missing)[:5]}")
    return sequences


def write_manifest(path: Path, records: list[dict[str, object]], threads: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        with temporary_path.open("wb") as output_handle:
            process = subprocess.Popen(
                ["zstd", "-q", f"-T{threads}", "-9", "-c"],
                stdin=subprocess.PIPE,
                stdout=output_handle,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
            )
            assert process.stdin is not None
            for record in records:
                process.stdin.write(
                    json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
                )
            process.stdin.close()
            error = process.stderr.read() if process.stderr is not None else ""
            return_code = process.wait()
            if return_code:
                raise RuntimeError(f"zstd failed with {return_code}: {error[-1000:]}")
            output_handle.flush()
            os.fsync(output_handle.fileno())
        os.link(temporary_path, path)
        path.chmod(0o444)
    except FileExistsError as error:
        raise FileExistsError(f"Refusing to overwrite real-gate manifest: {path}") from error
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
        raise FileExistsError(f"Refusing to overwrite real-gate receipt: {path}") from error
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--pair-receipt", type=Path, required=True)
    parser.add_argument("--endpoint-fasta", type=Path, required=True)
    parser.add_argument("--cluster-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--record-count", type=int, default=12800)
    parser.add_argument("--candidate-multiplier", type=int, default=2)
    parser.add_argument("--minimum-residues-per-chain", type=int, default=20)
    parser.add_argument("--threads", type=int, default=6)
    arguments = parser.parse_args()
    if arguments.seed < 0 or arguments.record_count < 1:
        raise ValueError("Invalid real-gate seed or record count")
    if arguments.candidate_multiplier < 2 or arguments.minimum_residues_per_chain < 1:
        raise ValueError("Invalid candidate multiplier or sequence-length gate")
    if arguments.threads < 1:
        raise ValueError("threads must be positive")

    pairs = arguments.pairs.resolve()
    pair_receipt_path = arguments.pair_receipt.resolve()
    pair_receipt = json.loads(pair_receipt_path.read_text(encoding="utf-8"))
    if (
        pair_receipt.get("status") != "DECONTAMINATED_NOT_MERGED_NOT_AUTHORIZED"
        or pair_receipt.get("test_access") is not False
        or pair_receipt["output"]["sha256"] != sha256_file(pairs)
    ):
        raise ValueError("STRING development-clean pair evidence differs")
    cluster_receipt_path = arguments.cluster_receipt.resolve()
    cluster_receipt = json.loads(cluster_receipt_path.read_text(encoding="utf-8"))
    endpoint_fasta = arguments.endpoint_fasta.resolve()
    published_fasta = {
        asset.get("name"): asset for asset in cluster_receipt.get("published_assets", [])
    }.get("endpoint_sequences.fasta.zst")
    if (
        cluster_receipt.get("format_version") != "abpairfm_string_cluster_pair_collapse_v1"
        or not isinstance(published_fasta, dict)
        or published_fasta.get("sha256") != sha256_file(endpoint_fasta)
        or pair_receipt.get("input_receipts", {}).get("cluster_sha256")
        != sha256_file(cluster_receipt_path)
    ):
        raise ValueError("STRING expanded cluster sequence evidence differs")

    pair_rows = iter(csv.reader(zstd_text_lines(pairs), delimiter="\t"))
    header = next(pair_rows)
    if header[:2] != ["cluster1", "cluster2"] or len(header) != 12:
        raise ValueError("Unexpected development-clean STRING pair header")
    candidate_count = arguments.record_count * arguments.candidate_multiplier
    candidates, source_rows = select_candidate_rows(
        pair_rows,
        seed=arguments.seed,
        candidate_count=candidate_count,
    )
    if source_rows != int(pair_receipt["counts"]["output_cluster_pairs"]):
        raise ValueError("STRING pair count differs from its receipt")
    target_ids = {value for _, row in candidates for value in row[2:4]}
    sequences = read_selected_sequences(endpoint_fasta, target_ids)
    records: list[dict[str, object]] = []
    rejected_short = 0
    for digest, row in candidates:
        protein1, protein2 = row[2:4]
        sequence1 = sequences[protein1]
        sequence2 = sequences[protein2]
        if min(len(sequence1), len(sequence2)) < arguments.minimum_residues_per_chain:
            rejected_short += 1
            continue
        records.append(
            {
                "cluster1": row[0],
                "cluster2": row[1],
                "combined_score": int(row[11]),
                "pair_hash_sha256": f"{digest:064x}",
                "protein1": protein1,
                "protein2": protein2,
                "selection_index": len(records),
                "sequence1": sequence1,
                "sequence2": sequence2,
            }
        )
        if len(records) == arguments.record_count:
            break
    if len(records) != arguments.record_count:
        raise ValueError("Eligible deterministic real-gate slice is too small")
    if len({record["pair_hash_sha256"] for record in records}) != len(records):
        raise ValueError("Real-gate slice contains duplicate pair hashes")
    output = arguments.output.resolve()
    write_manifest(output, records, arguments.threads)
    pair_lengths = [
        len(str(record["sequence1"])) + len(str(record["sequence2"]))
        for record in records
    ]
    receipt = {
        "authorization": {
            "production_pretraining": False,
            "real_data_gpu_gate": True,
        },
        "claim_boundary": "Deterministic development-clean STRING train slice for systems and numerical qualification only. It contains no validation or test examples and does not authorize production training, biological improvement, or SOTA claims.",
        "counts": {
            "candidate_pool": candidate_count,
            "manifest_records": len(records),
            "rejected_short_candidates_before_fill": rejected_short,
            "source_cluster_pairs": source_rows,
            "unique_manifest_endpoints": len(
                {str(record[key]) for record in records for key in ("protein1", "protein2")}
            ),
        },
        "format_version": "abpairfm_3b_real_gate_slice_v1",
        "inputs": {
            "cluster_receipt_sha256": sha256_file(cluster_receipt_path),
            "pair_receipt_sha256": sha256_file(pair_receipt_path),
        },
        "output": {
            "bytes": output.stat().st_size,
            "name": output.name,
            "sha256": sha256_file(output),
        },
        "pair_residue_lengths": {
            "maximum": max(pair_lengths),
            "median": statistics.median(pair_lengths),
            "minimum": min(pair_lengths),
        },
        "selection": {
            "candidate_multiplier": arguments.candidate_multiplier,
            "minimum_residues_per_chain": arguments.minimum_residues_per_chain,
            "rule": "smallest_sha256_of_seed_and_unordered_endpoint_cluster_pair",
            "seed": arguments.seed,
        },
        "status": "REAL_GATE_SLICE_READY_NOT_TRAINED",
        "test_access": False,
    }
    publish_create_only(arguments.receipt.resolve(), receipt)
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
