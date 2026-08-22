#!/usr/bin/env python3
"""Materialize one immutable Stage-A0 STRING segment from the frozen campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

from build_abpairfm_3b_real_gate_slice import (  # noqa: E402
    publish_create_only,
    read_selected_sequences,
    select_candidate_rows,
    sha256_file,
    write_manifest,
    zstd_text_lines,
)


FORMAT_VERSION = "abpairfm_3b_stage_a0_segment_manifest_v1"
READY_STATUS = "PRODUCTION_STAGE_A0_SEGMENT_READY"


def _sha256_json(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_protocol(path: Path, expected_sha256: str) -> dict[str, object]:
    if sha256_file(path) != expected_sha256:
        raise ValueError("Stage-A0 campaign protocol SHA-256 differs")
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if (
        protocol.get("format_version") != "abpairfm_3b_stage_a0_campaign_v1"
        or protocol.get("authorization", {}).get("production_stage_a0_pair_mlm") is not True
        or protocol.get("authorization", {}).get("production_stage_a_full_objective") is not False
        or protocol.get("authorization", {}).get("test_access") is not False
        or protocol.get("prohibited", {}).get("pinder_bytes") is not True
    ):
        raise ValueError("Stage-A0 campaign authorization differs")
    return protocol


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--pair-receipt", type=Path, required=True)
    parser.add_argument("--endpoint-fasta", type=Path, required=True)
    parser.add_argument("--cluster-receipt", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--segment-index", type=int, default=0)
    parser.add_argument("--record-count", type=int, default=22400)
    parser.add_argument("--candidate-headroom", type=int, default=22400)
    parser.add_argument("--minimum-residues-per-chain", type=int, default=20)
    parser.add_argument("--threads", type=int, default=6)
    arguments = parser.parse_args()
    if arguments.segment_index < 0 or arguments.record_count < 1:
        raise ValueError("Invalid Stage-A0 segment index or record count")
    if arguments.candidate_headroom < arguments.record_count:
        raise ValueError("Candidate headroom must cover at least one segment")
    if len(arguments.expected_protocol_sha256) != 64:
        raise ValueError("Expected protocol SHA-256 is malformed")

    protocol_path = arguments.protocol.resolve()
    protocol = _validate_protocol(protocol_path, arguments.expected_protocol_sha256)
    source = protocol["corpus"]["source"]
    pairs = arguments.pairs.resolve()
    pair_receipt_path = arguments.pair_receipt.resolve()
    pair_receipt = json.loads(pair_receipt_path.read_text(encoding="utf-8"))
    if (
        pair_receipt.get("format_version")
        != "abpairfm_string_high_confidence_strict_scale_pool_v1"
        or pair_receipt.get("status")
        != "STRICT_DECONTAMINATED_SCALE_POOL_NOT_AUTHORIZED_FOR_TRAINING"
        or pair_receipt.get("test_access") is not False
        or pair_receipt.get("output", {}).get("sha256") != sha256_file(pairs)
        or pair_receipt.get("output", {}).get("sha256") != source["pair_pool_sha256"]
        or pair_receipt.get("counts", {}).get("output_cluster_pairs")
        != source["pair_count"]
        or pair_receipt.get("strict_development_decontamination")
        != source["strict_development_gate"]
    ):
        raise ValueError("Strict STRING scale-pool evidence differs")

    endpoint_fasta = arguments.endpoint_fasta.resolve()
    cluster_receipt_path = arguments.cluster_receipt.resolve()
    cluster_receipt = json.loads(cluster_receipt_path.read_text(encoding="utf-8"))
    published = {
        value.get("name"): value for value in cluster_receipt.get("published_assets", [])
    }
    if (
        cluster_receipt.get("format_version")
        != "abpairfm_string_high_confidence_cluster_assets_v1"
        or cluster_receipt.get("test_access") is not False
        or published.get("endpoint_sequences.fasta.zst", {}).get("sha256")
        != source["endpoint_fasta_sha256"]
        or sha256_file(endpoint_fasta) != source["endpoint_fasta_sha256"]
    ):
        raise ValueError("Strict STRING endpoint-sequence evidence differs")

    selection_start = arguments.segment_index * arguments.record_count
    selection_stop = selection_start + arguments.record_count
    candidate_count = selection_stop + arguments.candidate_headroom
    pair_rows = iter(csv.reader(zstd_text_lines(pairs), delimiter="\t"))
    header = next(pair_rows)
    if header[:4] != ["cluster1", "cluster2", "protein1", "protein2"] or len(header) != 12:
        raise ValueError("Unexpected strict STRING scale-pool header")
    candidates, source_rows = select_candidate_rows(
        pair_rows,
        seed=int(protocol["corpus"]["selection"]["seed"]),
        candidate_count=candidate_count,
    )
    if source_rows != source["pair_count"]:
        raise ValueError("Strict STRING source count differs during materialization")
    targets = {endpoint for _, row in candidates for endpoint in row[2:4]}
    sequences = read_selected_sequences(endpoint_fasta, targets)

    eligible: list[tuple[int, tuple[str, ...], str, str]] = []
    rejected_short = 0
    for selection_digest, row in candidates:
        sequence1 = sequences[row[2]]
        sequence2 = sequences[row[3]]
        if min(len(sequence1), len(sequence2)) < arguments.minimum_residues_per_chain:
            rejected_short += 1
            continue
        eligible.append((selection_digest, row, sequence1, sequence2))
    selected = eligible[selection_start:selection_stop]
    if len(selected) != arguments.record_count:
        raise ValueError("Eligible deterministic Stage-A0 segment is too small")

    records: list[dict[str, object]] = []
    direct_count = 0
    for global_index, (selection_digest, row, sequence1, sequence2) in enumerate(
        selected, start=selection_start
    ):
        direct = int(row[5]) > 0 or int(row[7]) > 0
        direct_count += int(direct)
        biological_hash = hashlib.sha256(
            f"{row[0]}\0{row[1]}".encode("utf-8")
        ).hexdigest()
        records.append(
            {
                "cluster1": row[0],
                "cluster2": row[1],
                "combined_score": int(row[11]),
                "evidence_role": "direct" if direct else "transferred_only",
                "pair_hash_sha256": biological_hash,
                "protein1": row[2],
                "protein2": row[3],
                "segment_record_index": len(records),
                "selection_hash_sha256": f"{selection_digest:064x}",
                "selection_index_global": global_index,
                "sequence1": sequence1,
                "sequence2": sequence2,
            }
        )
    if len({record["pair_hash_sha256"] for record in records}) != len(records):
        raise ValueError("Stage-A0 segment contains duplicate biological pair hashes")

    output = arguments.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    write_manifest(output, records, arguments.threads)
    lengths = [len(str(record["sequence1"])) + len(str(record["sequence2"])) for record in records]
    receipt = {
        "authorization": {
            "production_stage_a0_pair_mlm": True,
            "production_stage_a_full_objective": False,
        },
        "claim_boundary": protocol["claim_boundary"],
        "counts": {
            "direct_evidence_records": direct_count,
            "manifest_records": len(records),
            "rejected_short_candidates_before_slice": rejected_short,
            "source_cluster_pairs": source_rows,
            "transferred_only_records": len(records) - direct_count,
            "unique_biological_pairs": len(records),
            "unique_endpoints": len(
                {str(record[key]) for record in records for key in ("protein1", "protein2")}
            ),
        },
        "format_version": FORMAT_VERSION,
        "inputs": {
            "cluster_receipt_sha256": sha256_file(cluster_receipt_path),
            "pair_receipt_sha256": sha256_file(pair_receipt_path),
            "protocol_sha256": arguments.expected_protocol_sha256,
        },
        "output": {
            "bytes": output.stat().st_size,
            "name": output.name,
            "sha256": sha256_file(output),
        },
        "pair_residue_lengths": {
            "maximum": max(lengths),
            "median": statistics.median(lengths),
            "minimum": min(lengths),
        },
        "segment": {
            "candidate_count": candidate_count,
            "index": arguments.segment_index,
            "selection_start": selection_start,
            "selection_stop_exclusive": selection_stop,
        },
        "selection_contract_sha256": _sha256_json(protocol["corpus"]["selection"]),
        "status": READY_STATUS,
        "test_access": False,
    }
    publish_create_only(arguments.receipt.resolve(), receipt)
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
