#!/usr/bin/env python3
"""Remove PPIRef50K v2 interfaces homologous to the frozen development panel."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fasta_records(path: Path):
    header: str | None = None
    chunks: list[str] = []
    with path.open("rt", encoding="ascii") as handle:
        for line in handle:
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks)
                header = line[1:].strip().split(maxsplit=1)[0]
                chunks = []
            else:
                chunks.append(line.strip())
        if header is not None:
            yield header, "".join(chunks)


def endpoint_ids(path: Path) -> set[str]:
    identifiers: set[str] = set()
    for header, sequence in fasta_records(path):
        if not sequence or not sequence.isalpha() or header in identifiers:
            raise ValueError("Malformed normalized PPIRef FASTA")
        if header.rsplit("|", 1)[-1] not in {"1", "2"}:
            raise ValueError(f"Malformed PPIRef endpoint id: {header}")
        identifiers.add(header)
    if not identifiers:
        raise ValueError("Normalized PPIRef FASTA is empty")
    return identifiers


def excluded_examples(
    path: Path, known_endpoints: set[str]
) -> tuple[set[str], dict[str, int]]:
    examples: set[str] = set()
    endpoints: set[str] = set()
    queries: set[str] = set()
    rows = 0
    with path.open("rt", encoding="utf-8", newline="") as handle:
        for line_number, row in enumerate(csv.reader(handle, delimiter="\t"), start=1):
            if len(row) != 8:
                raise ValueError(f"Malformed MMseqs hit row {line_number}")
            query, target = row[:2]
            identity = float(row[2])
            query_coverage = float(row[4])
            target_coverage = float(row[5])
            if identity < 0.5 or min(query_coverage, target_coverage) < 0.8:
                raise ValueError("MMseqs hit violates the frozen homology gate")
            if target not in known_endpoints:
                raise ValueError(f"MMseqs target is not a PPIRef endpoint: {target}")
            example, partner = target.rsplit("|", 1)
            if partner not in {"1", "2"}:
                raise ValueError(f"Malformed PPIRef MMseqs target: {target}")
            examples.add(example)
            endpoints.add(target)
            queries.add(query)
            rows += 1
    return examples, {
        "development_queries_with_hits": len(queries),
        "excluded_endpoint_sequences": len(endpoints),
        "excluded_interfaces": len(examples),
        "homology_hit_rows": rows,
    }


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
        raise FileExistsError(f"Refusing to overwrite PPIRef evidence: {path}") from error
    finally:
        temporary_path.unlink(missing_ok=True)


def filter_fasta(source: Path, output: Path, excluded: set[str]) -> dict[str, int]:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    temporary_path = Path(temporary)
    input_records = 0
    output_records = 0
    admitted_examples: set[str] = set()
    endpoint_sequences: set[str] = set()
    try:
        with os.fdopen(descriptor, "wt", encoding="ascii", newline="") as handle:
            for header, sequence in fasta_records(source):
                input_records += 1
                example = header.rsplit("|", 1)[0]
                if example in excluded:
                    continue
                handle.write(f">{header}\n{sequence}\n")
                output_records += 1
                admitted_examples.add(example)
                endpoint_sequences.add(sequence)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_path, output)
        output.chmod(0o444)
    except FileExistsError as error:
        raise FileExistsError(f"Refusing to overwrite PPIRef FASTA: {output}") from error
    finally:
        temporary_path.unlink(missing_ok=True)
    if input_records % 2 or output_records != 2 * len(admitted_examples):
        raise ValueError("PPIRef FASTA partner cardinality differs")
    return {
        "input_endpoint_records": input_records,
        "output_endpoint_records": output_records,
        "unique_output_endpoint_sequences": len(endpoint_sequences),
    }


def filter_manifest(
    source: Path, output: Path, excluded: set[str], threads: int
) -> dict[str, int]:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    os.close(descriptor)
    temporary_path = Path(temporary)
    input_interfaces = 0
    output_interfaces = 0
    contacts = 0
    exact_pairs: set[tuple[str, str]] = set()
    decompressor = subprocess.Popen(
        ["zstd", "-q", "-dc", str(source)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="ascii",
    )
    assert decompressor.stdout is not None
    try:
        with temporary_path.open("wb") as output_handle:
            compressor = subprocess.Popen(
                ["zstd", "-q", f"-T{threads}", "-19", "-c"],
                stdin=subprocess.PIPE,
                stdout=output_handle,
                stderr=subprocess.PIPE,
                text=True,
                encoding="ascii",
            )
            assert compressor.stdin is not None
            for line in decompressor.stdout:
                record = json.loads(line)
                input_interfaces += 1
                example = record.get("example_id")
                if not isinstance(example, str):
                    raise ValueError("PPIRef interface manifest lacks example_id")
                if example in excluded:
                    continue
                sequence1 = record.get("chain1_full_sequence")
                sequence2 = record.get("chain2_full_sequence")
                pairs = record.get("contact_pairs")
                if not isinstance(sequence1, str) or not isinstance(sequence2, str) or not isinstance(pairs, list):
                    raise ValueError("PPIRef interface manifest schema differs")
                compressor.stdin.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
                output_interfaces += 1
                contacts += len(pairs)
                exact_pairs.add(tuple(sorted((sequence1, sequence2))))
            compressor.stdin.close()
            compressor_error = compressor.stderr.read() if compressor.stderr is not None else ""
            compressor_code = compressor.wait()
            if compressor_code:
                raise RuntimeError(f"zstd compression failed: {compressor_error[-1000:]}")
            output_handle.flush()
            os.fsync(output_handle.fileno())
        decompressor.stdout.close()
        decompressor_error = decompressor.stderr.read() if decompressor.stderr is not None else ""
        decompressor_code = decompressor.wait()
        if decompressor_code:
            raise RuntimeError(f"zstd decompression failed: {decompressor_error[-1000:]}")
        os.link(temporary_path, output)
        output.chmod(0o444)
    except FileExistsError as error:
        raise FileExistsError(f"Refusing to overwrite PPIRef manifest: {output}") from error
    finally:
        if decompressor.poll() is None:
            decompressor.kill()
            decompressor.wait()
        temporary_path.unlink(missing_ok=True)
    return {
        "input_interfaces": input_interfaces,
        "output_contact_pairs": contacts,
        "output_interfaces": output_interfaces,
        "removed_interfaces": input_interfaces - output_interfaces,
        "unique_output_exact_full_sequence_pairs": len(exact_pairs),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--normalization-receipt", type=Path, required=True)
    parser.add_argument("--development-fasta", type=Path, required=True)
    parser.add_argument("--development-receipt", type=Path, required=True)
    parser.add_argument("--hits", type=Path, required=True)
    parser.add_argument("--hits-output", type=Path, required=True)
    parser.add_argument("--excluded-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--fasta-output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--mmseqs-version", required=True)
    parser.add_argument("--threads", type=int, default=6)
    arguments = parser.parse_args()
    if arguments.threads < 1:
        raise ValueError("threads must be positive")

    normalization_receipt_path = arguments.normalization_receipt.resolve()
    normalization = json.loads(normalization_receipt_path.read_text(encoding="utf-8"))
    if (
        normalization.get("format_version") != "abpairfm_ppiref50k_interfaces_v2"
        or normalization.get("status") != "NORMALIZED_NOT_DECONTAMINATED_NOT_AUTHORIZED"
        or normalization.get("counts", {}).get("archive_unresolved_members") != 0
        or normalization.get("test_access") is not False
    ):
        raise ValueError("PPIRef normalization receipt is not eligible")
    manifest = arguments.manifest.resolve()
    fasta = arguments.fasta.resolve()
    if sha256_file(manifest) != normalization["outputs"]["interface_manifest"]["sha256"]:
        raise ValueError("PPIRef manifest SHA-256 differs")
    if sha256_file(fasta) != normalization["outputs"]["full_sequence_fasta"]["sha256"]:
        raise ValueError("PPIRef FASTA SHA-256 differs")

    development_receipt_path = arguments.development_receipt.resolve()
    development = json.loads(development_receipt_path.read_text(encoding="utf-8"))
    development_fasta = arguments.development_fasta.resolve()
    if development.get("status") != "FROZEN_VALIDATION_SEQUENCE_PANEL" or development.get("test_access") is not False or sha256_file(development_fasta) != development["output_fasta"]["sha256"]:
        raise ValueError("Frozen development panel evidence differs")

    known_endpoints = endpoint_ids(fasta)
    excluded, hit_counts = excluded_examples(arguments.hits.resolve(), known_endpoints)
    excluded_output = arguments.excluded_output.resolve()
    publish_create_only(excluded_output, "".join(f"{value}\n" for value in sorted(excluded)).encode())
    hits_output = arguments.hits_output.resolve()
    if hits_output.exists():
        raise FileExistsError(f"Refusing to overwrite PPIRef hits: {hits_output}")
    shutil.copyfile(arguments.hits.resolve(), hits_output)
    hits_output.chmod(0o444)

    manifest_output = arguments.manifest_output.resolve()
    fasta_output = arguments.fasta_output.resolve()
    manifest_counts = filter_manifest(manifest, manifest_output, excluded, arguments.threads)
    fasta_counts = filter_fasta(fasta, fasta_output, excluded)
    if manifest_counts["input_interfaces"] != normalization["counts"]["admitted_interfaces"]:
        raise ValueError("PPIRef normalization interface count differs")
    if manifest_counts["removed_interfaces"] != hit_counts["excluded_interfaces"]:
        raise ValueError("PPIRef excluded interface count differs")
    if manifest_counts["output_interfaces"] * 2 != fasta_counts["output_endpoint_records"]:
        raise ValueError("PPIRef filtered manifest and FASTA counts differ")

    receipt = {
        "authorization": {"cross_source_deduplication": True, "stage_a_training": False},
        "claim_boundary": "PPIRef50K interfaces after removal of any full-chain endpoint with a >=50% identity and >=80% bidirectional-coverage hit to the frozen AsEP/SAbDab2 development panel; PINDER-priority cross-source deduplication remains pending.",
        "counts": {**manifest_counts, **fasta_counts, **hit_counts},
        "format_version": "abpairfm_ppiref50k_development_decontamination_v2",
        "homology_gate": {
            "alignment_mode": 3,
            "coverage": 0.8,
            "coverage_mode": 0,
            "minimum_sequence_identity": 0.5,
            "mmseqs_version": arguments.mmseqs_version,
        },
        "inputs": {
            "development_receipt_sha256": sha256_file(development_receipt_path),
            "normalization_receipt_sha256": sha256_file(normalization_receipt_path),
        },
        "outputs": {
            "fasta": {"bytes": fasta_output.stat().st_size, "sha256": sha256_file(fasta_output)},
            "manifest": {"bytes": manifest_output.stat().st_size, "sha256": sha256_file(manifest_output)},
        },
        "published_evidence": {
            "excluded_interfaces_sha256": sha256_file(excluded_output),
            "hits_sha256": sha256_file(hits_output),
        },
        "status": "DECONTAMINATED_CROSS_SOURCE_DEDUP_PENDING",
        "test_access": False,
    }
    publish_create_only(
        arguments.receipt.resolve(),
        (json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n").encode(),
    )
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
