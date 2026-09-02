#!/usr/bin/env python3
"""Build nested, cluster-balanced PINDER-Train contact curricula.

The input manifest is already restricted to PINDER-Train and decontaminated
against the frozen development panel.  This script only joins authenticated
holo endpoint sequences, computes multiplicity weights, and emits deterministic
nested prefixes.  It never reads PINDER test rows or test structures.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


REQUIRED_COLUMNS = {
    "split",
    "id",
    "pdb_id",
    "cluster_id",
    "holo_R_pdb",
    "holo_L_pdb",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_digest(seed: int, namespace: str, value: str) -> str:
    payload = f"{seed}\0{namespace}\0{value}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_sequence_manifest(path: Path) -> dict[str, str]:
    sequences: dict[str, str] = {}
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or not {"mapping_id", "sequence"}.issubset(
            reader.fieldnames
        ):
            raise ValueError("Sequence manifest schema differs")
        for row in reader:
            mapping_id = row["mapping_id"]
            sequence = row["sequence"]
            if not mapping_id or not sequence:
                raise ValueError("Sequence manifest contains an empty key or sequence")
            previous = sequences.setdefault(mapping_id, sequence)
            if previous != sequence:
                raise ValueError(f"Conflicting sequence for {mapping_id}")
    return sequences


def read_train_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or not REQUIRED_COLUMNS.issubset(reader.fieldnames):
            raise ValueError("PINDER train manifest schema differs")
        rows = list(reader)
        fieldnames = list(reader.fieldnames)
    if not rows or any(row["split"] != "train" for row in rows):
        raise ValueError("Curriculum input must contain non-empty train-only rows")
    identifiers = [row["id"] for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("PINDER system identifiers must be unique")
    return fieldnames, rows


def cluster_balanced_order(
    rows: Iterable[dict[str, str]], *, seed: int
) -> list[dict[str, str]]:
    """Return one system per cluster before taking a second from any cluster."""

    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["cluster_id"]].append(row)
    ordered_clusters = sorted(
        groups,
        key=lambda value: (stable_digest(seed, "cluster", value), value),
    )
    for cluster_id, members in groups.items():
        members.sort(
            key=lambda row: (
                stable_digest(seed, f"system:{cluster_id}", row["id"]),
                row["id"],
            )
        )
    result: list[dict[str, str]] = []
    maximum_depth = max(len(members) for members in groups.values())
    for depth in range(maximum_depth):
        for cluster_id in ordered_clusters:
            members = groups[cluster_id]
            if depth < len(members):
                result.append(members[depth])
    return result


def enrich_and_weight(
    rows: list[dict[str, str]], sequences: dict[str, str]
) -> list[dict[str, str]]:
    exact_pairs: list[str] = []
    for row in rows:
        mapping_r = row["holo_R_pdb"].removesuffix(".pdb")
        mapping_l = row["holo_L_pdb"].removesuffix(".pdb")
        try:
            sequence_r = sequences[mapping_r]
            sequence_l = sequences[mapping_l]
        except KeyError as error:
            raise ValueError(f"Missing authenticated holo sequence: {error.args[0]}") from error
        sha_r = hashlib.sha256(sequence_r.encode("ascii")).hexdigest()
        sha_l = hashlib.sha256(sequence_l.encode("ascii")).hexdigest()
        exact_pair = hashlib.sha256("\0".join(sorted((sha_r, sha_l))).encode()).hexdigest()
        row.update(
            {
                "sequence_R": sequence_r,
                "sequence_L": sequence_l,
                "sequence_sha256_R": sha_r,
                "sequence_sha256_L": sha_l,
                "exact_sequence_pair_sha256": exact_pair,
            }
        )
        exact_pairs.append(exact_pair)

    pair_counts = Counter(exact_pairs)
    pdb_counts = Counter(row["pdb_id"] for row in rows)
    cluster_counts = Counter(row["cluster_id"] for row in rows)
    raw_weights = [
        (
            pair_counts[row["exact_sequence_pair_sha256"]]
            * pdb_counts[row["pdb_id"]]
            * cluster_counts[row["cluster_id"]]
        )
        ** (-1.0 / 3.0)
        for row in rows
    ]
    mean_weight = sum(raw_weights) / len(raw_weights)
    if not math.isfinite(mean_weight) or mean_weight <= 0.0:
        raise ValueError("Multiplicity weights are invalid")
    for row, raw_weight in zip(rows, raw_weights):
        row.update(
            {
                "exact_sequence_pair_multiplicity": str(
                    pair_counts[row["exact_sequence_pair_sha256"]]
                ),
                "pdb_multiplicity": str(pdb_counts[row["pdb_id"]]),
                "interface_cluster_multiplicity": str(
                    cluster_counts[row["cluster_id"]]
                ),
                "system_weight": f"{raw_weight / mean_weight:.12g}",
            }
        )
    return rows


def write_deterministic_tsv_gz(
    path: Path, fieldnames: list[str], rows: list[dict[str, str]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                writer = csv.DictWriter(
                    text,
                    fieldnames=fieldnames,
                    delimiter="\t",
                    lineterminator="\n",
                    extrasaction="raise",
                )
                writer.writeheader()
                writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--train-manifest-sha256", required=True)
    parser.add_argument("--sequence-manifest", type=Path, required=True)
    parser.add_argument("--sequence-manifest-sha256", required=True)
    parser.add_argument("--license-receipt", type=Path, required=True)
    parser.add_argument("--license-receipt-sha256", required=True)
    parser.add_argument("--decontamination-receipt", type=Path, required=True)
    parser.add_argument("--decontamination-receipt-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument(
        "--stages",
        default="B-canary:512,B32K:32768,B500K:500000,B-full:865471",
    )
    args = parser.parse_args()

    authority_paths = {
        "train_manifest": (args.train_manifest, args.train_manifest_sha256),
        "sequence_manifest": (args.sequence_manifest, args.sequence_manifest_sha256),
        "license_receipt": (args.license_receipt, args.license_receipt_sha256),
        "decontamination_receipt": (
            args.decontamination_receipt,
            args.decontamination_receipt_sha256,
        ),
    }
    observed_authority: dict[str, str] = {}
    for name, (path, expected) in authority_paths.items():
        observed = sha256_file(path)
        if observed != expected:
            raise SystemExit(f"{name} SHA-256 differs: {observed}")
        observed_authority[name] = observed

    stages: list[tuple[str, int]] = []
    for item in args.stages.split(","):
        name, count = item.split(":", 1)
        stages.append((name, int(count)))
    counts = [count for _, count in stages]
    if not stages or counts != sorted(counts) or len(counts) != len(set(counts)):
        raise SystemExit("Stages must have unique increasing sizes")

    original_fields, rows = read_train_rows(args.train_manifest)
    if counts[-1] > len(rows):
        raise SystemExit("Largest stage exceeds the authenticated train inventory")
    sequences = read_sequence_manifest(args.sequence_manifest)
    ordered = cluster_balanced_order(rows, seed=args.seed)
    enriched = enrich_and_weight(ordered, sequences)
    extra_fields = [
        "sequence_R",
        "sequence_L",
        "sequence_sha256_R",
        "sequence_sha256_L",
        "exact_sequence_pair_sha256",
        "exact_sequence_pair_multiplicity",
        "pdb_multiplicity",
        "interface_cluster_multiplicity",
        "system_weight",
    ]
    fieldnames = original_fields + extra_fields

    stage_receipts: dict[str, object] = {}
    previous_ids: set[str] = set()
    for name, count in stages:
        selected = enriched[:count]
        selected_ids = {row["id"] for row in selected}
        if not previous_ids.issubset(selected_ids):
            raise RuntimeError("Curriculum prefixes are not nested")
        previous_ids = selected_ids
        output = args.output_root / f"{name}.tsv.gz"
        write_deterministic_tsv_gz(output, fieldnames, selected)
        weights = [float(row["system_weight"]) for row in selected]
        stage_receipts[name] = {
            "systems": count,
            "clusters": len({row["cluster_id"] for row in selected}),
            "pdbs": len({row["pdb_id"] for row in selected}),
            "exact_sequence_pairs": len(
                {row["exact_sequence_pair_sha256"] for row in selected}
            ),
            "system_weight_mean": sum(weights) / len(weights),
            "bytes": output.stat().st_size,
            "sha256": sha256_file(output),
            "path": output.name,
        }

    receipt = {
        "format_version": "abpairfm_pinder_contact_curriculum_v1",
        "status": "PASS_TRAIN_ONLY_NESTED_CLUSTER_BALANCED_CURRICULUM",
        "seed": args.seed,
        "selection": "deterministic_cluster_round_robin_then_seeded_system_order",
        "multiplicity_weight": "(exact_sequence_pair*pdb*interface_cluster)^(-1/3), normalized to mean one over full authenticated inventory",
        "authority_sha256": observed_authority,
        "stages": stage_receipts,
        "checks": {
            "all_rows_train": True,
            "all_holo_sequences_authenticated": True,
            "nested_prefixes": True,
            "system_ids_unique": True,
            "test_rows_read": False,
            "test_structures_read": False,
        },
        "claim_boundary": "Frozen PINDER-Train curricula for internal pair-contact pretraining. PINDER test rows and test structures were not read. This receipt does not authorize antibody-specific adaptation or external-test access.",
    }
    receipt_path = args.output_root / "PINDER_CONTACT_CURRICULUM_RECEIPT_V1.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({**receipt, "receipt_sha256": sha256_file(receipt_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
