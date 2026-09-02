#!/usr/bin/env python3
"""Run the frozen label-free SABLE double-cold eligibility screen."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
from typing import Any

from Bio.Align import PairwiseAligner, substitution_matrices


PLAN_SHA256 = "6187210253ca07b2d6c5e11e017552485bbb2699bc3772f67d0ee6ad9657d9a4"
METADATA_SHA256 = "d6b16885c7ab44bbe2a38b3fc02ab085adff73479c1a119e3786831f464b20b7"
DEVELOPMENT_SHA256 = "27e0beb755c3d8e4b9910d4ad2ca979161668a1e1fe7b4d09bf6a60251e07468"
DEVELOPMENT_RECEIPT_SHA256 = (
    "36a9249d5d1cec4065292e79d1efae8623987a4643675557f32252bcde7752ee"
)
AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
ALLOWED_ANTIGEN_TYPES = frozenset(("protein", "peptide"))

_development_antibodies: tuple[str, ...] = ()
_development_antigens: tuple[str, ...] = ()
_global_aligner: PairwiseAligner | None = None
_local_aligner: PairwiseAligner | None = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def normalize(sequence: str) -> str:
    return "".join(sequence.split()).upper()


def valid_amino_acid_sequence(sequence: str) -> bool:
    return bool(sequence) and set(sequence) <= AMINO_ACIDS


def make_aligner(mode: str) -> PairwiseAligner:
    aligner = PairwiseAligner(mode=mode)
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -11.0
    aligner.extend_gap_score = -1.0
    return aligner


def initialize_worker(
    development_antibodies: tuple[str, ...], development_antigens: tuple[str, ...]
) -> None:
    global _development_antibodies, _development_antigens
    global _global_aligner, _local_aligner
    _development_antibodies = development_antibodies
    _development_antigens = development_antigens
    _global_aligner = make_aligner("global")
    _local_aligner = make_aligner("local")


def alignment_statistics(aligner: PairwiseAligner, query: str, target: str) -> dict[str, float]:
    alignment = aligner.align(query, target)[0]
    counts = alignment.counts()
    denominator = counts.identities + counts.mismatches + counts.gaps
    if denominator <= 0:
        raise ValueError("alignment has no scored columns")
    aligned_query = sum(end - start for start, end in alignment.aligned[0])
    aligned_target = sum(end - start for start, end in alignment.aligned[1])
    aligned_residues = min(aligned_query, aligned_target)
    return {
        "identity": float(counts.identities / denominator),
        "shorter_coverage": float(aligned_residues / min(len(query), len(target))),
        "aligned_residues": int(aligned_residues),
    }


def screen_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    if _global_aligner is None or _local_aligner is None:
        raise RuntimeError("worker aligners are not initialized")
    antibody = candidate["antibody_sequence"]
    maximum_antibody_identity = 0.0
    closest_antibody_sha256 = None
    for development in _development_antibodies:
        if min(len(antibody), len(development)) / max(len(antibody), len(development)) <= 0.9:
            continue
        statistics = alignment_statistics(_global_aligner, antibody, development)
        if statistics["identity"] > maximum_antibody_identity:
            maximum_antibody_identity = statistics["identity"]
            closest_antibody_sha256 = hashlib.sha256(development.encode()).hexdigest()
        if statistics["identity"] > 0.9:
            return {
                **candidate,
                "eligible": False,
                "exclusion_reason": "antibody_global_identity_above_0.9",
                "maximum_antibody_identity": maximum_antibody_identity,
                "closest_development_antibody_sha256": closest_antibody_sha256,
            }

    maximum_antigen_identity = 0.0
    maximum_antigen_shorter_coverage = 0.0
    closest_antigen_sha256 = None
    for candidate_chain in candidate["antigen_chains"]:
        for development in _development_antigens:
            statistics = alignment_statistics(_local_aligner, candidate_chain, development)
            if statistics["identity"] > maximum_antigen_identity:
                maximum_antigen_identity = statistics["identity"]
                maximum_antigen_shorter_coverage = statistics["shorter_coverage"]
                closest_antigen_sha256 = hashlib.sha256(development.encode()).hexdigest()
            standard_hit = (
                statistics["identity"] >= 0.3
                and statistics["shorter_coverage"] >= 0.7
                and statistics["aligned_residues"] >= 20
            )
            high_identity_hit = (
                statistics["identity"] >= 0.7
                and statistics["shorter_coverage"] >= 0.3
                and statistics["aligned_residues"] >= 12
            )
            if standard_hit or high_identity_hit:
                return {
                    **candidate,
                    "eligible": False,
                    "exclusion_reason": "antigen_local_similarity_gate",
                    "maximum_antibody_identity": maximum_antibody_identity,
                    "closest_development_antibody_sha256": closest_antibody_sha256,
                    "maximum_antigen_identity": statistics["identity"],
                    "maximum_antigen_shorter_coverage": statistics[
                        "shorter_coverage"
                    ],
                    "closest_development_antigen_sha256": hashlib.sha256(
                        development.encode()
                    ).hexdigest(),
                }
    return {
        **candidate,
        "eligible": True,
        "exclusion_reason": None,
        "maximum_antibody_identity": maximum_antibody_identity,
        "closest_development_antibody_sha256": closest_antibody_sha256,
        "maximum_antigen_identity": maximum_antigen_identity,
        "maximum_antigen_shorter_coverage": maximum_antigen_shorter_coverage,
        "closest_development_antigen_sha256": closest_antigen_sha256,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--development-sequences", type=Path, required=True)
    parser.add_argument("--development-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    for output in (args.output, args.receipt_output):
        if output.exists() or output.is_symlink():
            raise FileExistsError(output)
    expected_sha = {
        args.plan: PLAN_SHA256,
        args.metadata: METADATA_SHA256,
        args.development_sequences: DEVELOPMENT_SHA256,
        args.development_receipt: DEVELOPMENT_RECEIPT_SHA256,
    }
    for path, digest in expected_sha.items():
        if sha256_file(path) != digest:
            raise ValueError(f"authority SHA differs: {path}")
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    development_receipt = json.loads(
        args.development_receipt.read_text(encoding="utf-8")
    )
    if (
        plan.get("status") != "FROZEN_BEFORE_LABEL_FREE_SCREEN"
        or plan.get("task_labels_read") is not False
        or plan.get("sable_bound_structures_downloaded_before_gate") is not False
        or development_receipt.get("task_labels_read") is not False
        or development_receipt.get("test_loaded") is not False
        or development_receipt.get("test_rows_emitted") is not False
    ):
        raise ValueError("label-free authority differs")

    development_rows: list[dict[str, Any]] = []
    with args.development_sequences.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["role"] not in ("train", "validation"):
                raise ValueError("development sequence role differs")
            development_rows.append(row)
    if len(development_rows) != 3114:
        raise ValueError("development sequence count differs")
    development_antibodies = tuple(
        sorted({normalize(row["ab_seq"]) for row in development_rows})
    )
    development_antigens = tuple(
        sorted({normalize(row["ag_seq"]) for row in development_rows})
    )
    development_antibody_set = set(development_antibodies)
    development_antigen_set = set(development_antigens)

    candidate_rows: list[dict[str, Any]] = []
    source_test_count = 0
    source_test_has_antigen_count = 0
    with args.metadata.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["split"] != "test":
                continue
            source_test_count += 1
            if row["has_antigen"].lower() != "true":
                continue
            source_test_has_antigen_count += 1
            antigen_types = tuple(
                part.strip().lower() for part in row["antigen_type"].split("|")
            )
            antigen_chains = tuple(
                normalize(part) for part in row["antigen_sequence"].split("|")
            )
            heavy = normalize(row["H_sequence"])
            light = normalize(row["L_sequence"])
            antibody = heavy + light
            if (
                row["synthetic"].lower() != "false"
                or not antigen_types
                or any(kind not in ALLOWED_ANTIGEN_TYPES for kind in antigen_types)
                or not valid_amino_acid_sequence(heavy)
                or (light and not valid_amino_acid_sequence(light))
                or not antigen_chains
                or any(not valid_amino_acid_sequence(chain) for chain in antigen_chains)
            ):
                continue
            exact_antibody_overlap = antibody in development_antibody_set
            exact_antigen_overlap = any(
                chain in development_antigen_set for chain in antigen_chains
            )
            candidate_rows.append(
                {
                    "entry_id": row["entry_id"],
                    "pdb_id": row["pdb_id"].lower(),
                    "release_date": row["release_date"],
                    "antibody_type": row["antibody_type"],
                    "heavy_sequence": heavy,
                    "light_sequence": light,
                    "antibody_sequence": antibody,
                    "antigen_chains": antigen_chains,
                    "antibody_sequence_sha256": hashlib.sha256(
                        antibody.encode()
                    ).hexdigest(),
                    "antigen_chain_sha256": [
                        hashlib.sha256(chain.encode()).hexdigest()
                        for chain in antigen_chains
                    ],
                    "exact_antibody_overlap": exact_antibody_overlap,
                    "exact_antigen_overlap": exact_antigen_overlap,
                }
            )
    exact_filtered: list[dict[str, Any]] = []
    preexcluded: list[dict[str, Any]] = []
    for candidate in candidate_rows:
        if candidate["exact_antibody_overlap"] or candidate["exact_antigen_overlap"]:
            preexcluded.append(
                {
                    **candidate,
                    "eligible": False,
                    "exclusion_reason": "exact_development_sequence_overlap",
                }
            )
        else:
            exact_filtered.append(candidate)

    context = mp.get_context("spawn")
    with context.Pool(
        processes=args.workers,
        initializer=initialize_worker,
        initargs=(development_antibodies, development_antigens),
    ) as pool:
        aligned_rows = list(pool.imap(screen_candidate, exact_filtered, chunksize=1))
    rows = preexcluded + aligned_rows
    rows.sort(key=lambda row: row["entry_id"])
    eligible = [row for row in rows if row["eligible"]]
    eligible_pdb_ids = {row["pdb_id"] for row in eligible}
    minimum = plan["minimum_gate"]
    passed = (
        len(eligible) >= int(minimum["eligible_entries"])
        and len(eligible_pdb_ids) >= int(minimum["unique_pdb_ids"])
    )
    payload = {
        "format_version": "sable_v0_2_0_framecontact_label_free_screen_v1",
        "status": "PASS_LABEL_FREE_GATE" if passed else "FAIL_CLOSED_LABEL_FREE_GATE",
        "authority": {
            "plan_sha256": PLAN_SHA256,
            "metadata_sha256": METADATA_SHA256,
            "development_sequence_view_sha256": DEVELOPMENT_SHA256,
            "development_sequence_view_receipt_sha256": DEVELOPMENT_RECEIPT_SHA256,
        },
        "counts": {
            "source_test_entries": source_test_count,
            "source_test_entries_with_antigen": source_test_has_antigen_count,
            "amino_acid_candidate_entries": len(candidate_rows),
            "eligible_entries": len(eligible),
            "eligible_unique_pdb_ids": len(eligible_pdb_ids),
            "unique_development_antibody_sequences": len(development_antibodies),
            "unique_development_antigen_sequences": len(development_antigens),
        },
        "minimum_gate": minimum,
        "records": rows,
        "task_labels_read": False,
        "sable_bound_structures_downloaded": False,
        "old_sabdab2_test_loaded": False,
        "temporal_holdout_loaded": False,
        "selection_or_tuning": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_bytes(payload))
    receipt = {
        "format_version": "sable_v0_2_0_framecontact_label_free_screen_receipt_v1",
        "status": payload["status"],
        "result_sha256": sha256_file(args.output),
        "eligible_entries": len(eligible),
        "eligible_unique_pdb_ids": len(eligible_pdb_ids),
        "task_labels_read": False,
        "sable_bound_structures_downloaded": False,
    }
    args.receipt_output.write_bytes(canonical_bytes(receipt))
    print(
        json.dumps(
            {
                **receipt,
                "receipt_sha256": sha256_file(args.receipt_output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
