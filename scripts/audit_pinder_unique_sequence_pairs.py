#!/usr/bin/env python3
"""Count exact biological sequence pairs behind decontaminated PINDER systems."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mapping_id(value: str) -> str:
    path = PurePosixPath(value)
    if path.name != value or path.suffix != ".pdb":
        raise RuntimeError(f"Invalid PINDER endpoint filename: {value!r}")
    return path.stem


def sequence_digests(path: Path) -> dict[str, bytes]:
    result: dict[str, bytes] = {}
    with gzip.open(path, "rt", encoding="ascii", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ())[:2] != ("mapping_id", "sequence"):
            raise RuntimeError("PINDER sequence manifest schema differs")
        for row in reader:
            identifier = row["mapping_id"]
            sequence = row["sequence"]
            if not identifier or not sequence or identifier in result:
                raise RuntimeError("PINDER sequence manifest contains an invalid row")
            result[identifier] = hashlib.sha256(sequence.encode("ascii")).digest()
    if not result:
        raise RuntimeError("PINDER sequence manifest is empty")
    return result


def pair_inventory(
    path: Path, sequences: dict[str, bytes]
) -> dict[str, int]:
    exact_pairs: set[tuple[bytes, bytes]] = set()
    uniprot_pairs: set[tuple[str, str]] = set()
    clusters: set[str] = set()
    endpoints: set[bytes] = set()
    systems = 0
    exact_self_pairs = 0
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = set(reader.fieldnames or ())
        required = {
            "split",
            "id",
            "cluster_id",
            "holo_R_pdb",
            "holo_L_pdb",
            "uniprot_R",
            "uniprot_L",
        }
        if not required.issubset(fields):
            raise RuntimeError("PINDER decontaminated manifest schema differs")
        for row in reader:
            if row["split"] != "train":
                raise RuntimeError("Non-train row reached PINDER pair audit")
            receptor = sequences.get(mapping_id(row["holo_R_pdb"]))
            ligand = sequences.get(mapping_id(row["holo_L_pdb"]))
            if receptor is None or ligand is None:
                raise RuntimeError("PINDER system endpoint lacks a sequence digest")
            pair = tuple(sorted((receptor, ligand)))
            exact_pairs.add(pair)
            endpoints.update(pair)
            exact_self_pairs += int(receptor == ligand)
            uniprot_pairs.add(tuple(sorted((row["uniprot_R"], row["uniprot_L"]))))
            clusters.add(row["cluster_id"])
            systems += 1
    if systems == 0:
        raise RuntimeError("PINDER decontaminated manifest is empty")
    return {
        "exact_self_pair_systems": exact_self_pairs,
        "post_decontamination_systems": systems,
        "repeated_structural_views": systems - len(exact_pairs),
        "unique_exact_endpoint_sequences": len(endpoints),
        "unique_exact_unordered_sequence_pairs": len(exact_pairs),
        "unique_interface_clusters": len(clusters),
        "unique_unordered_uniprot_pairs": len(uniprot_pairs),
    }


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("ascii")


def publish_create_only(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
        raise FileExistsError(f"Refusing to overwrite PINDER pair audit: {path}") from error
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--systems", type=Path, required=True)
    parser.add_argument("--systems-receipt", type=Path, required=True)
    parser.add_argument("--sequences", type=Path, required=True)
    parser.add_argument("--sequences-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    systems = arguments.systems.resolve()
    systems_receipt_path = arguments.systems_receipt.resolve()
    systems_receipt = json.loads(systems_receipt_path.read_text(encoding="ascii"))
    sequences = arguments.sequences.resolve()
    sequences_receipt_path = arguments.sequences_receipt.resolve()
    sequences_receipt = json.loads(sequences_receipt_path.read_text(encoding="ascii"))
    if (
        systems_receipt.get("format_version")
        != "abpairfm_pinder_development_decontamination_v1"
        or systems_receipt.get("test_access") is not False
        or sha256_file(systems) != systems_receipt["output"]["sha256"]
    ):
        raise RuntimeError("PINDER decontamination evidence differs")
    if (
        sequences_receipt.get("format_version")
        != "abpairfm_pinder_holo_sequences_v3"
        or sequences_receipt.get("test_access") is not False
        or sha256_file(sequences)
        != sequences_receipt["outputs"]["manifest_sha256"]
    ):
        raise RuntimeError("PINDER sequence evidence differs")

    counts = pair_inventory(systems, sequence_digests(sequences))
    if counts["post_decontamination_systems"] != systems_receipt["counts"][
        "output_systems"
    ]:
        raise RuntimeError("PINDER system count differs from decontamination receipt")
    payload: dict[str, object] = {
        "authorization": {"stage_a_training": False},
        "claim_boundary": (
            "Exact sequence-pair accounting for decontaminated PINDER train systems. "
            "Repeated structural views are reported separately and never counted as "
            "new sequence pairs. Combined Stage A admission remains pending."
        ),
        "counts": counts,
        "format_version": "abpairfm_pinder_unique_sequence_pair_audit_v1",
        "inputs": {
            "sequence_receipt_sha256": sha256_file(sequences_receipt_path),
            "systems_receipt_sha256": sha256_file(systems_receipt_path),
        },
        "status": "PAIR_COUNT_AUDITED_STAGE_A_MERGE_PENDING",
        "test_access": False,
    }
    payload["receipt_sha256"] = hashlib.sha256(canonical_bytes(payload)).hexdigest()
    publish_create_only(arguments.output.resolve(), canonical_bytes(payload))
    print(canonical_bytes(payload).decode("ascii"), end="")


if __name__ == "__main__":
    main()
