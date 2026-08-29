#!/usr/bin/env python3
"""Remove strict PINDER train systems homologous to frozen development sequences."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mapping_id(pdb_name: str) -> str:
    path = PurePosixPath(pdb_name)
    if path.name != pdb_name or path.suffix != ".pdb":
        raise ValueError(f"Invalid PINDER holo endpoint: {pdb_name!r}")
    return path.stem


def sequence_ids(path: Path) -> set[str]:
    identifiers: set[str] = set()
    with gzip.open(path, "rt", encoding="ascii", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if "mapping_id" not in (reader.fieldnames or ()):
            raise RuntimeError("PINDER sequence manifest lacks mapping_id")
        for row in reader:
            identifier = row["mapping_id"]
            if not identifier or identifier in identifiers:
                raise RuntimeError("PINDER sequence IDs are empty or duplicated")
            identifiers.add(identifier)
    if not identifiers:
        raise RuntimeError("PINDER sequence manifest is empty")
    return identifiers


def excluded_endpoints(
    path: Path, known_sequences: set[str]
) -> tuple[set[str], dict[str, int]]:
    excluded: set[str] = set()
    queries: set[str] = set()
    hit_rows = 0
    with path.open("rt", encoding="utf-8", newline="") as handle:
        for line_number, row in enumerate(csv.reader(handle, delimiter="\t"), start=1):
            if len(row) != 8:
                raise RuntimeError(f"Malformed MMseqs row {line_number}")
            query, target = row[:2]
            identity = float(row[2])
            query_coverage = float(row[4])
            target_coverage = float(row[5])
            if identity < 0.5 or min(query_coverage, target_coverage) < 0.8:
                raise RuntimeError("MMseqs row violates the frozen homology gate")
            if target not in known_sequences:
                raise RuntimeError(f"MMseqs target is not a PINDER endpoint: {target}")
            queries.add(query)
            excluded.add(target)
            hit_rows += 1
    return excluded, {
        "development_queries_with_hits": len(queries),
        "excluded_endpoint_sequences": len(excluded),
        "homology_hit_rows": hit_rows,
    }


def _deterministic_gzip_text(path: Path):
    raw = path.open("xb")
    compressed = gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="")
    text = io.TextIOWrapper(compressed, encoding="utf-8", newline="")
    return raw, compressed, text


def filter_manifest(
    source: Path,
    output: Path,
    known_sequences: set[str],
    excluded: set[str],
) -> dict[str, int]:
    input_count = 0
    output_count = 0
    input_clusters: set[str] = set()
    output_clusters: set[str] = set()
    raw, compressed, text = _deterministic_gzip_text(output)
    try:
        with gzip.open(source, "rt", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            fields = tuple(reader.fieldnames or ())
            required = {"split", "cluster_id", "holo_R_pdb", "holo_L_pdb"}
            if not required.issubset(fields):
                raise RuntimeError("PINDER manifest schema differs")
            writer = csv.DictWriter(
                text,
                fieldnames=fields,
                delimiter="\t",
                lineterminator="\n",
            )
            writer.writeheader()
            for row in reader:
                input_count += 1
                if row["split"] != "train":
                    raise RuntimeError("Non-train row reached PINDER decontamination")
                receptor = mapping_id(row["holo_R_pdb"])
                ligand = mapping_id(row["holo_L_pdb"])
                if receptor not in known_sequences or ligand not in known_sequences:
                    raise RuntimeError("PINDER system endpoint lacks an assembled sequence")
                input_clusters.add(row["cluster_id"])
                if receptor in excluded or ligand in excluded:
                    continue
                writer.writerow(row)
                output_count += 1
                output_clusters.add(row["cluster_id"])
        text.flush()
        text.detach()
        compressed.close()
        raw.flush()
        os.fsync(raw.fileno())
    finally:
        raw.close()
    return {
        "input_clusters": len(input_clusters),
        "input_systems": input_count,
        "output_clusters": len(output_clusters),
        "output_systems": output_count,
        "removed_systems": input_count - output_count,
    }


def publish_create_only(path: Path, payload: bytes) -> None:
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
        raise FileExistsError(f"Refusing to overwrite artifact: {path}") from error
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--train-manifest-receipt", type=Path, required=True)
    parser.add_argument("--sequence-manifest", type=Path, required=True)
    parser.add_argument("--sequence-receipt", type=Path, required=True)
    parser.add_argument("--development-fasta", type=Path, required=True)
    parser.add_argument("--development-receipt", type=Path, required=True)
    parser.add_argument("--hits", type=Path, required=True)
    parser.add_argument("--hits-output", type=Path, required=True)
    parser.add_argument("--excluded-endpoints-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--mmseqs-version", required=True)
    arguments = parser.parse_args()

    train_manifest = arguments.train_manifest.resolve()
    train_receipt_path = arguments.train_manifest_receipt.resolve()
    train_receipt = json.loads(train_receipt_path.read_text(encoding="ascii"))
    if (
        train_receipt.get("format_version")
        != "abpairfm_pinder_train_system_manifest_v1"
        or train_receipt.get("test_access", {}).get(
            "pinder_test_structure_bytes_accessed"
        )
        is not False
        or sha256_file(train_manifest) != train_receipt["output"]["sha256"]
    ):
        raise RuntimeError("PINDER train manifest evidence differs")

    sequence_manifest = arguments.sequence_manifest.resolve()
    sequence_receipt_path = arguments.sequence_receipt.resolve()
    sequence_receipt = json.loads(sequence_receipt_path.read_text(encoding="ascii"))
    if (
        sequence_receipt.get("format_version")
        != "abpairfm_pinder_holo_sequences_v3"
        or sequence_receipt.get("status")
        != "SEQUENCES_ASSEMBLED_NOT_DECONTAMINATED"
        or sequence_receipt.get("test_access") is not False
        or sha256_file(sequence_manifest)
        != sequence_receipt["outputs"]["manifest_sha256"]
    ):
        raise RuntimeError("PINDER sequence evidence differs")
    known_sequences = sequence_ids(sequence_manifest)

    development_fasta = arguments.development_fasta.resolve()
    development_receipt_path = arguments.development_receipt.resolve()
    development_receipt = json.loads(
        development_receipt_path.read_text(encoding="utf-8")
    )
    if (
        development_receipt.get("status") != "FROZEN_VALIDATION_SEQUENCE_PANEL"
        or development_receipt.get("test_access") is not False
        or sha256_file(development_fasta)
        != development_receipt["output_fasta"]["sha256"]
    ):
        raise RuntimeError("Frozen development panel evidence differs")

    hits = arguments.hits.resolve()
    excluded, hit_counts = excluded_endpoints(hits, known_sequences)
    excluded_output = arguments.excluded_endpoints_output.resolve()
    publish_create_only(
        excluded_output,
        "".join(f"{item}\n" for item in sorted(excluded)).encode("utf-8"),
    )
    hits_output = arguments.hits_output.resolve()
    if hits_output.exists() or hits_output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite hits: {hits_output}")
    shutil.copyfile(hits, hits_output)
    hits_output.chmod(0o444)

    output = arguments.output.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite PINDER output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    counts = filter_manifest(train_manifest, output, known_sequences, excluded)
    if counts["input_systems"] != train_receipt["inventory"][
        "strict_quality_system_count"
    ]:
        raise RuntimeError("PINDER input system count differs from receipt")

    payload = {
        "authorization": {
            "license_review_complete": False,
            "stage_a_training": False,
        },
        "claim_boundary": (
            "Strict PINDER train systems after removal of every system with a "
            "holo endpoint having a >=50% identity and >=80% bidirectional-coverage "
            "hit to the frozen AsEP/SAbDab2 development panel. License review and "
            "combined Stage A authorization remain pending."
        ),
        "counts": {**counts, **hit_counts},
        "format_version": "abpairfm_pinder_development_decontamination_v1",
        "homology_gate": {
            "alignment_mode": 3,
            "coverage": 0.8,
            "coverage_mode": 0,
            "minimum_sequence_identity": 0.5,
            "mmseqs_version": arguments.mmseqs_version,
        },
        "inputs": {
            "development_receipt_sha256": sha256_file(development_receipt_path),
            "sequence_receipt_sha256": sha256_file(sequence_receipt_path),
            "train_manifest_receipt_sha256": sha256_file(train_receipt_path),
        },
        "output": {
            "bytes": output.stat().st_size,
            "sha256": sha256_file(output),
        },
        "published_evidence": {
            "excluded_endpoints_sha256": sha256_file(excluded_output),
            "hits_sha256": sha256_file(hits_output),
        },
        "status": "DECONTAMINATED_LICENSE_AND_MERGE_PENDING",
        "test_access": False,
    }
    publish_create_only(
        arguments.receipt.resolve(),
        (json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True)
         + "\n").encode("ascii"),
    )
    print(json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True))


if __name__ == "__main__":
    main()
