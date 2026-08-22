#!/usr/bin/env python3
"""Apply frozen PINDER priority to development-clean PPIRef50K v2."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import tempfile

from decontaminate_ppiref50k_interfaces_v2 import fasta_records, filter_fasta, sha256_file


def mapping_id(value: str) -> str:
    path = PurePosixPath(value)
    if path.name != value or path.suffix != ".pdb":
        raise ValueError(f"Invalid PINDER endpoint filename: {value!r}")
    return path.stem


def sequence_digests(path: Path) -> dict[str, bytes]:
    result: dict[str, bytes] = {}
    with gzip.open(path, "rt", encoding="ascii", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ())[:2] != ("mapping_id", "sequence"):
            raise ValueError("PINDER sequence manifest schema differs")
        for row in reader:
            identifier = row["mapping_id"]
            sequence = row["sequence"]
            if not identifier or not sequence or identifier in result:
                raise ValueError("PINDER sequence manifest contains an invalid row")
            result[identifier] = hashlib.sha256(sequence.encode("ascii")).digest()
    if not result:
        raise ValueError("PINDER sequence manifest is empty")
    return result


def pinder_priority_inventory(
    path: Path, sequences: dict[str, bytes]
) -> tuple[set[tuple[bytes, bytes]], set[str], int]:
    pairs: set[tuple[bytes, bytes]] = set()
    pdb_ids: set[str] = set()
    systems = 0
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"split", "pdb_id", "holo_R_pdb", "holo_L_pdb"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError("PINDER development-clean manifest schema differs")
        for row in reader:
            if row["split"] != "train":
                raise ValueError("Non-train PINDER row reached cross-source deduplication")
            receptor = sequences.get(mapping_id(row["holo_R_pdb"]))
            ligand = sequences.get(mapping_id(row["holo_L_pdb"]))
            if receptor is None or ligand is None:
                raise ValueError("PINDER endpoint lacks a sequence digest")
            pairs.add(tuple(sorted((receptor, ligand))))
            pdb_ids.add(row["pdb_id"].lower())
            systems += 1
    if systems == 0:
        raise ValueError("PINDER priority inventory is empty")
    return pairs, pdb_ids, systems


def deduplication_reasons(
    record: dict[str, object],
    pinder_pairs: set[tuple[bytes, bytes]],
    pinder_pdb_ids: set[str],
) -> set[str]:
    pdb_id = record.get("pdb_id")
    sequence1 = record.get("chain1_full_sequence")
    sequence2 = record.get("chain2_full_sequence")
    if not isinstance(pdb_id, str) or not isinstance(sequence1, str) or not isinstance(sequence2, str):
        raise ValueError("PPIRef record lacks cross-source deduplication fields")
    pair = tuple(
        sorted(
            (
                hashlib.sha256(sequence1.encode("ascii")).digest(),
                hashlib.sha256(sequence2.encode("ascii")).digest(),
            )
        )
    )
    reasons: set[str] = set()
    if pdb_id.lower() in pinder_pdb_ids:
        reasons.add("same_pdb_entry")
    if pair in pinder_pairs:
        reasons.add("exact_sequence_pair")
    return reasons


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
        raise FileExistsError(f"Refusing to overwrite PPIRef cross-source evidence: {path}") from error
    finally:
        temporary_path.unlink(missing_ok=True)


def filter_manifest(
    source: Path,
    output: Path,
    exclusions_output: Path,
    pinder_pairs: set[tuple[bytes, bytes]],
    pinder_pdb_ids: set[str],
    threads: int,
) -> tuple[set[str], dict[str, int]]:
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    os.close(descriptor)
    temporary_path = Path(temporary)
    exclusions: list[tuple[str, str]] = []
    counts = {
        "input_interfaces": 0,
        "output_contact_pairs": 0,
        "output_interfaces": 0,
        "removed_exact_sequence_pair": 0,
        "removed_same_pdb_entry": 0,
        "removed_with_both_reasons": 0,
    }
    output_pairs: set[tuple[bytes, bytes]] = set()
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
                counts["input_interfaces"] += 1
                example_id = record.get("example_id")
                if not isinstance(example_id, str):
                    raise ValueError("PPIRef record lacks example_id")
                reasons = deduplication_reasons(record, pinder_pairs, pinder_pdb_ids)
                if reasons:
                    counts["removed_exact_sequence_pair"] += int("exact_sequence_pair" in reasons)
                    counts["removed_same_pdb_entry"] += int("same_pdb_entry" in reasons)
                    counts["removed_with_both_reasons"] += int(len(reasons) == 2)
                    exclusions.append((example_id, ",".join(sorted(reasons))))
                    continue
                sequence1 = record["chain1_full_sequence"]
                sequence2 = record["chain2_full_sequence"]
                contacts = record.get("contact_pairs")
                if not isinstance(sequence1, str) or not isinstance(sequence2, str) or not isinstance(contacts, list):
                    raise ValueError("PPIRef record schema differs")
                compressor.stdin.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
                counts["output_interfaces"] += 1
                counts["output_contact_pairs"] += len(contacts)
                output_pairs.add(
                    tuple(
                        sorted(
                            (
                                hashlib.sha256(sequence1.encode("ascii")).digest(),
                                hashlib.sha256(sequence2.encode("ascii")).digest(),
                            )
                        )
                    )
                )
            compressor.stdin.close()
            compressor_error = compressor.stderr.read() if compressor.stderr is not None else ""
            if compressor.wait():
                raise RuntimeError(f"zstd compression failed: {compressor_error[-1000:]}")
            output_handle.flush()
            os.fsync(output_handle.fileno())
        decompressor.stdout.close()
        decompressor_error = decompressor.stderr.read() if decompressor.stderr is not None else ""
        if decompressor.wait():
            raise RuntimeError(f"zstd decompression failed: {decompressor_error[-1000:]}")
        os.link(temporary_path, output)
        output.chmod(0o444)
    finally:
        if decompressor.poll() is None:
            decompressor.kill()
            decompressor.wait()
        temporary_path.unlink(missing_ok=True)
    excluded = {example_id for example_id, _ in exclusions}
    counts["removed_interfaces_union"] = len(excluded)
    counts["unique_output_exact_sequence_pairs"] = len(output_pairs)
    if counts["input_interfaces"] != counts["output_interfaces"] + len(excluded):
        raise ValueError("PPIRef cross-source partition differs")
    publish_create_only(
        exclusions_output,
        ("example_id\treasons\n" + "".join(f"{example_id}\t{reasons}\n" for example_id, reasons in sorted(exclusions))).encode("ascii"),
    )
    return excluded, counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pinder-systems", type=Path, required=True)
    parser.add_argument("--pinder-systems-receipt", type=Path, required=True)
    parser.add_argument("--pinder-sequences", type=Path, required=True)
    parser.add_argument("--pinder-sequences-receipt", type=Path, required=True)
    parser.add_argument("--ppiref-manifest", type=Path, required=True)
    parser.add_argument("--ppiref-fasta", type=Path, required=True)
    parser.add_argument("--ppiref-receipt", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--fasta-output", type=Path, required=True)
    parser.add_argument("--exclusions-output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=6)
    arguments = parser.parse_args()
    if arguments.threads < 1:
        raise ValueError("threads must be positive")

    pinder_systems = arguments.pinder_systems.resolve()
    pinder_systems_receipt_path = arguments.pinder_systems_receipt.resolve()
    pinder_systems_receipt = json.loads(pinder_systems_receipt_path.read_text(encoding="ascii"))
    pinder_sequences = arguments.pinder_sequences.resolve()
    pinder_sequences_receipt_path = arguments.pinder_sequences_receipt.resolve()
    pinder_sequences_receipt = json.loads(pinder_sequences_receipt_path.read_text(encoding="ascii"))
    if (
        pinder_systems_receipt.get("format_version") != "abpairfm_pinder_development_decontamination_v1"
        or pinder_systems_receipt.get("test_access") is not False
        or sha256_file(pinder_systems) != pinder_systems_receipt["output"]["sha256"]
    ):
        raise ValueError("PINDER development-clean evidence differs")
    if (
        pinder_sequences_receipt.get("format_version") != "abpairfm_pinder_holo_sequences_v3"
        or pinder_sequences_receipt.get("test_access") is not False
        or sha256_file(pinder_sequences) != pinder_sequences_receipt["outputs"]["manifest_sha256"]
    ):
        raise ValueError("PINDER sequence evidence differs")
    pinder_pairs, pinder_pdb_ids, pinder_system_count = pinder_priority_inventory(
        pinder_systems, sequence_digests(pinder_sequences)
    )
    if pinder_system_count != pinder_systems_receipt["counts"]["output_systems"]:
        raise ValueError("PINDER priority system count differs")

    ppiref_receipt_path = arguments.ppiref_receipt.resolve()
    ppiref_receipt = json.loads(ppiref_receipt_path.read_text(encoding="utf-8"))
    ppiref_manifest = arguments.ppiref_manifest.resolve()
    ppiref_fasta = arguments.ppiref_fasta.resolve()
    if (
        ppiref_receipt.get("format_version") != "abpairfm_ppiref50k_development_decontamination_v2"
        or ppiref_receipt.get("authorization", {}).get("cross_source_deduplication") is not True
        or ppiref_receipt.get("test_access") is not False
        or sha256_file(ppiref_manifest) != ppiref_receipt["outputs"]["manifest"]["sha256"]
        or sha256_file(ppiref_fasta) != ppiref_receipt["outputs"]["fasta"]["sha256"]
    ):
        raise ValueError("PPIRef development-clean evidence differs")

    manifest_output = arguments.manifest_output.resolve()
    fasta_output = arguments.fasta_output.resolve()
    for path in (manifest_output, fasta_output, arguments.exclusions_output.resolve(), arguments.receipt.resolve()):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Refusing to overwrite PPIRef cross-source output: {path}")
    excluded, counts = filter_manifest(
        ppiref_manifest,
        manifest_output,
        arguments.exclusions_output.resolve(),
        pinder_pairs,
        pinder_pdb_ids,
        arguments.threads,
    )
    fasta_counts = filter_fasta(ppiref_fasta, fasta_output, excluded)
    if counts["input_interfaces"] != ppiref_receipt["counts"]["output_interfaces"]:
        raise ValueError("PPIRef input count differs")
    if counts["output_interfaces"] * 2 != fasta_counts["output_endpoint_records"]:
        raise ValueError("PPIRef cross-source manifest and FASTA counts differ")

    receipt = {
        "authorization": {"string_priority_deduplication": True, "stage_a_training": False},
        "claim_boundary": "Development-clean PPIRef50K after PINDER-priority exact unordered full-sequence-pair removal and conservative same-PDB-entry removal. Same-PDB entry exclusion is stronger than the frozen same-assembly minimum because PPIRef does not preserve assembly identifiers. STRING priority and final Stage A authorization remain pending.",
        "counts": {
            **counts,
            **fasta_counts,
            "pinder_priority_exact_sequence_pairs": len(pinder_pairs),
            "pinder_priority_pdb_ids": len(pinder_pdb_ids),
            "pinder_priority_systems": pinder_system_count,
        },
        "format_version": "abpairfm_ppiref50k_pinder_priority_deduplication_v1",
        "inputs": {
            "pinder_sequences_receipt_sha256": sha256_file(pinder_sequences_receipt_path),
            "pinder_systems_receipt_sha256": sha256_file(pinder_systems_receipt_path),
            "ppiref_receipt_sha256": sha256_file(ppiref_receipt_path),
        },
        "outputs": {
            "exclusions": {"bytes": arguments.exclusions_output.resolve().stat().st_size, "sha256": sha256_file(arguments.exclusions_output.resolve())},
            "fasta": {"bytes": fasta_output.stat().st_size, "sha256": sha256_file(fasta_output)},
            "manifest": {"bytes": manifest_output.stat().st_size, "sha256": sha256_file(manifest_output)},
        },
        "status": "PINDER_PRIORITY_DEDUPLICATED_STRING_MERGE_PENDING",
        "test_access": False,
    }
    publish_create_only(
        arguments.receipt.resolve(),
        (json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii"),
    )
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
