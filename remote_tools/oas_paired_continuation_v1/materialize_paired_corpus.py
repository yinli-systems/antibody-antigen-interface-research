#!/usr/bin/env python3
"""Materialize an authenticated development-clean paired OAS binary shard."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import struct
from typing import Iterable

from audit_paired_corpus_quality import HEADER
from paired_pretraining import encode_pair


class BinarySink:
    """Write and hash a binary stream with bounded syscall overhead."""

    def __init__(self, path: Path, *, flush_bytes: int = 8 << 20) -> None:
        self.path = path
        self.handle = path.open("xb")
        self.buffer = bytearray()
        self.digest = hashlib.sha256()
        self.byte_count = 0
        self.flush_bytes = flush_bytes

    def write(self, value: bytes) -> None:
        self.buffer.extend(value)
        self.digest.update(value)
        self.byte_count += len(value)
        if len(self.buffer) >= self.flush_bytes:
            self.flush()

    def flush(self) -> None:
        if self.buffer:
            self.handle.write(self.buffer)
            self.buffer.clear()

    def close(self) -> dict[str, object]:
        self.flush()
        self.handle.close()
        return {
            "bytes": self.byte_count,
            "name": self.path.name,
            "sha256": self.digest.hexdigest(),
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_exclusions(path: Path, expected_count: int) -> set[bytes]:
    values: set[bytes] = set()
    previous: bytes | None = None
    with gzip.open(path, "rt", encoding="ascii", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                current = bytes.fromhex(line.rstrip("\n"))
            except ValueError as error:
                raise RuntimeError(f"Malformed exclusion at line {line_number}") from error
            if len(current) != 32 or (previous is not None and current <= previous):
                raise RuntimeError("Exclusion artifact is not strictly sorted and unique")
            values.add(current)
            previous = current
    if len(values) != expected_count:
        raise RuntimeError("Exclusion artifact count differs")
    return values


def _validation_split(donor_hash: str) -> bool:
    try:
        raw = bytes.fromhex(donor_hash)
    except ValueError as error:
        raise RuntimeError("Malformed donor hash") from error
    if len(raw) != 32 or raw.hex() != donor_hash:
        raise RuntimeError("Malformed donor hash")
    return int.from_bytes(raw[:8], "big") % 32 == 0


def materialize_rows(
    rows: Iterable[list[str]],
    *,
    exclusions: set[bytes],
    output_root: Path,
    expected_source_rows: int,
) -> dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=False)
    sinks = {
        "input_ids": BinarySink(output_root / "input_ids.u8.bin"),
        "type_ids": BinarySink(output_root / "type_ids.u8.bin"),
        "region_ids": BinarySink(output_root / "region_ids.u8.bin"),
        "offsets": BinarySink(output_root / "offsets.u64.bin"),
        "sample_keys": BinarySink(output_root / "sample_keys.u64.bin"),
        "split": BinarySink(output_root / "split.u8.bin"),
    }
    sinks["offsets"].write(struct.pack("<Q", 0))
    observed_rows = 0
    excluded_rows = 0
    train_rows = 0
    validation_rows = 0
    total_tokens = 0
    validation_by_species: dict[str, int] = {}
    try:
        for observed_rows, row in enumerate(rows, start=1):
            if len(row) != len(HEADER):
                raise RuntimeError("Paired source column count differs")
            values = dict(zip(HEADER, row, strict=True))
            row_hash = bytes.fromhex(values["row_hash"])
            if row_hash in exclusions:
                excluded_rows += 1
                continue
            encoded = encode_pair(
                values["heavy_sequence"],
                values["light_sequence"],
                (
                    values["cdr1_aa_heavy"],
                    values["cdr2_aa_heavy"],
                    values["cdr3_aa_heavy"],
                ),
                (
                    values["cdr1_aa_light"],
                    values["cdr2_aa_light"],
                    values["cdr3_aa_light"],
                ),
                maximum_tokens=320,
            )
            sinks["input_ids"].write(encoded.input_ids.tobytes())
            sinks["type_ids"].write(encoded.type_ids.tobytes())
            sinks["region_ids"].write(encoded.region_ids.tobytes())
            total_tokens += len(encoded.input_ids)
            sinks["offsets"].write(struct.pack("<Q", total_tokens))
            sinks["sample_keys"].write(row_hash[:8])
            is_validation = _validation_split(values["donor_hash"])
            sinks["split"].write(bytes((int(is_validation),)))
            if is_validation:
                validation_rows += 1
                validation_by_species[values["species"]] = (
                    validation_by_species.get(values["species"], 0) + 1
                )
            else:
                train_rows += 1
    finally:
        artifacts = {name: sink.close() for name, sink in sinks.items()}
    if observed_rows != expected_source_rows:
        raise RuntimeError("Observed source count differs")
    if excluded_rows != len(exclusions):
        raise RuntimeError("Some frozen exclusions were absent from paired source")
    kept_rows = train_rows + validation_rows
    if kept_rows != expected_source_rows - excluded_rows:
        raise RuntimeError("Materialized row accounting differs")
    if not train_rows or not validation_rows:
        raise RuntimeError("Frozen donor split produced an empty partition")
    return {
        "artifacts": artifacts,
        "counts": {
            "excluded_rows": excluded_rows,
            "kept_rows": kept_rows,
            "source_rows": observed_rows,
            "train_rows": train_rows,
            "validation_rows": validation_rows,
        },
        "split": {
            "method": "first_64_bits_of_donor_sha256_mod_32_equals_0",
            "validation_by_species": dict(sorted(validation_by_species.items())),
        },
        "total_paired_tokens": total_tokens,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--source-receipt", type=Path, required=True)
    parser.add_argument("--quality-result", type=Path, required=True)
    parser.add_argument("--decontamination-result", type=Path, required=True)
    parser.add_argument("--decontamination-script", type=Path, required=True)
    parser.add_argument("--decontamination-protocol", type=Path, required=True)
    parser.add_argument("--exclusions", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    protocol_bytes = args.protocol.read_bytes()
    protocol = json.loads(protocol_bytes.decode("ascii"))
    source_receipt = json.loads(args.source_receipt.read_text(encoding="utf-8"))
    quality = json.loads(args.quality_result.read_text(encoding="ascii"))
    decontamination = json.loads(args.decontamination_result.read_text(encoding="ascii"))
    if (
        protocol.get("format_version") != "oas_paired_development_clean_materialization_v1"
        or protocol.get("training_authorized") is not False
        or protocol.get("test_access") is not False
        or quality.get("status") != "PASS_PAIRED_CORPUS_QUALITY_AUDIT"
        or _sha256(args.quality_result) != protocol["quality_result_sha256"]
        or decontamination.get("status") != "PASS_PAIRWISE_DECONTAMINATION_AUDIT_V2"
        or decontamination.get("test_access") is not False
        or source_receipt.get("output", {}).get("sha256") != protocol["source_pairs_sha256"]
        or _sha256(args.decontamination_script)
        != protocol["decontamination_script_sha256"]
        or _sha256(args.decontamination_protocol)
        != protocol["decontamination_protocol_sha256"]
        or decontamination.get("frozen_composite_rule", {}).get("single_chain_identity")
        != 0.90
        or decontamination.get("frozen_composite_rule", {}).get(
            "paired_same_instance_identity"
        )
        != 0.80
    ):
        raise RuntimeError("Materialization authority or upstream audit differs")
    exclusion_spec = decontamination.get("exclusion_artifact", {})
    if (
        exclusion_spec.get("sha256") != _sha256(args.exclusions)
        or exclusion_spec.get("name") != args.exclusions.name
    ):
        raise RuntimeError("Exclusion artifact differs")
    if _sha256(args.pairs) != protocol["source_pairs_sha256"]:
        raise RuntimeError("Paired source hash differs")
    exclusions = _load_exclusions(args.exclusions, int(exclusion_spec["count"]))
    with gzip.open(args.pairs, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        if tuple(next(reader)) != HEADER:
            raise RuntimeError("Paired source header differs")
        result = materialize_rows(
            reader,
            exclusions=exclusions,
            output_root=args.output_root,
            expected_source_rows=int(source_receipt["counts"]["unique_exact_pairs"]),
        )
    manifest = {
        "decontamination_result_sha256": _sha256(args.decontamination_result),
        "exclusions_sha256": exclusion_spec["sha256"],
        "format_version": "oas_paired_development_clean_binary_v1",
        "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "quality_result_sha256": _sha256(args.quality_result),
        **result,
        "status": "PASS_PAIRED_DEVELOPMENT_CLEAN_MATERIALIZATION",
        "test_access": False,
        "training_performed": False,
    }
    manifest_path = args.output_root / "MANIFEST.json"
    data = (json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("ascii")
    descriptor = os.open(manifest_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)
    print(json.dumps(manifest, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
