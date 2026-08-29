#!/usr/bin/env python3
"""Recover exact CDR spans from authenticated raw OAS coordinates and row hashes."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any


AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
CDR_COLUMNS = {
    "heavy": ("cdr1_aa_heavy", "cdr2_aa_heavy", "cdr3_aa_heavy"),
    "light": ("cdr1_aa_light", "cdr2_aa_light", "cdr3_aa_light"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def atomic_write(path: Path, value: object) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(canonical_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def amino(value: str) -> str:
    return value.upper().replace("-", "").replace(".", "").strip()


def metadata(handle: Any) -> dict[str, Any]:
    fields = next(csv.reader([handle.readline()]))
    if len(fields) != 1:
        raise ValueError("malformed OAS metadata")
    value = json.loads(fields[0])
    if not isinstance(value, dict):
        raise ValueError("OAS metadata is not an object")
    return value


def integer_coordinate(value: str) -> int:
    number = float(value)
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError("non-integral coordinate")
    return int(number)


def exact_span(
    row: dict[str, str], chain: str, cdr_number: int, expected_sequence: str, expected_cdr: str
) -> tuple[int, int]:
    aligned = row[f"sequence_alignment_aa_{chain}"].upper().strip()
    if amino(aligned) != expected_sequence:
        raise ValueError("normalized sequence mismatch")
    v_start = integer_coordinate(row[f"v_sequence_start_{chain}"])
    cdr_start = integer_coordinate(row[f"cdr{cdr_number}_start_{chain}"])
    cdr_end = integer_coordinate(row[f"cdr{cdr_number}_end_{chain}"])
    start_nt = cdr_start - v_start
    end_nt = cdr_end - v_start + 1
    if start_nt < 0 or end_nt <= start_nt or start_nt % 3 or end_nt % 3:
        raise ValueError("coordinate frame mismatch")
    aligned_start = start_nt // 3
    aligned_end = end_nt // 3
    if aligned_end > len(aligned):
        raise ValueError("coordinate beyond aligned sequence")
    aligned_cdr = amino(aligned[aligned_start:aligned_end])
    if aligned_cdr != expected_cdr:
        raise ValueError("coordinate CDR mismatch")
    start = sum(residue not in "-." for residue in aligned[:aligned_start])
    end = start + sum(residue not in "-." for residue in aligned[aligned_start:aligned_end])
    if expected_sequence[start:end] != expected_cdr:
        raise ValueError("ungapped CDR mismatch")
    return start, end


def row_hash(study: str, run: str, row: dict[str, str]) -> str:
    identity = "|".join(
        (study, run, row.get("sequence_id_heavy", ""), row.get("sequence_id_light", ""))
    )
    return hashlib.sha256(identity.encode()).hexdigest()


def pair_hash(heavy: str, light: str) -> str:
    return hashlib.sha256((heavy + "\0" + light).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--raw-plan", type=Path, required=True)
    parser.add_argument("--output-spans", type=Path, required=True)
    parser.add_argument("--output-result", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    args = parser.parse_args()
    outputs = (args.output_spans, args.output_result, args.output_receipt)
    if any(path.exists() or path.is_symlink() for path in outputs):
        raise FileExistsError("one or more V3 output paths already exist")
    authority = json.loads(args.authority.read_text())
    if authority.get("protocol_status") != "frozen_before_exact_coordinate_recovery":
        raise RuntimeError("authority is not frozen")
    if sha256_file(args.source) != authority["target_source"]["sha256"]:
        raise RuntimeError("target source SHA differs")
    if sha256_file(args.raw_plan) != authority["raw_oas_authority"]["plan_sha256"]:
        raise RuntimeError("raw OAS plan SHA differs")
    if sha256_file(Path(authority["source_split_access_disclosure"]["receipt_uri"])) != authority["source_split_access_disclosure"]["receipt_sha256"]:
        raise RuntimeError("source split disclosure SHA differs")

    target: dict[str, tuple[int, tuple[str, ...]]] = {}
    with gzip.open(args.source, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {
            "heavy_sequence", "light_sequence", "row_hash", "study", "run",
            *[column for columns in CDR_COLUMNS.values() for column in columns],
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise RuntimeError("target source columns differ")
        for index, row in enumerate(reader):
            key = row["row_hash"]
            if key in target:
                raise RuntimeError("duplicate target row hash")
            values = (
                row["heavy_sequence"].strip().upper(),
                row["light_sequence"].strip().upper(),
                *(row[column].strip().upper() for chain in ("heavy", "light") for column in CDR_COLUMNS[chain]),
                row["study"], row["run"],
            )
            target[key] = (index, values)
    if len(target) != authority["target_source"]["expected_rows"]:
        raise RuntimeError("target source row count differs")

    plan = json.loads(args.raw_plan.read_text())
    files = plan["files"]
    if len(files) != authority["raw_oas_authority"]["expected_data_units"]:
        raise RuntimeError("raw plan file count differs")
    total_bytes = sum((args.raw_root / entry["relative_path"]).stat().st_size for entry in files)
    if total_bytes != authority["raw_oas_authority"]["expected_total_bytes"]:
        raise RuntimeError("raw data-unit byte accounting differs")

    args.output_spans.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=args.output_spans.name + ".tmp.", dir=args.output_spans.parent
    )
    os.close(descriptor)
    counts = Counter()
    exclusions = Counter()
    found: set[str] = set()
    pair_splits: dict[str, str] = {}
    fields = [
        "source_row_index", "row_hash", "pair_sha256", "split",
        "heavy_length", "light_length",
        "h1_start", "h1_end", "h2_start", "h2_end", "h3_start", "h3_end",
        "l1_start", "l1_end", "l2_start", "l2_end", "l3_start", "l3_end",
    ]
    try:
        with open(temporary_name, "wb") as raw_output:
            with gzip.GzipFile(filename="", fileobj=raw_output, mode="wb", mtime=0) as compressed:
                with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                    writer = csv.DictWriter(text, fieldnames=fields, delimiter="\t", lineterminator="\n")
                    writer.writeheader()
                    for entry in files:
                        relative = Path(entry["relative_path"])
                        path = args.raw_root / relative
                        study = relative.parts[0]
                        counts["raw_files"] += 1
                        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
                            source_metadata = metadata(handle)
                            run = str(source_metadata.get("Run", ""))
                            reader = csv.DictReader(handle)
                            for row in reader:
                                counts["raw_source_rows"] += 1
                                key = row_hash(study, run, row)
                                item = target.get(key)
                                if item is None:
                                    continue
                                if key in found:
                                    exclusions["duplicate_raw_row_hash"] += 1
                                    continue
                                found.add(key)
                                source_index, values = item
                                heavy, light, h1, h2, h3, l1, l2, l3, expected_study, expected_run = values
                                if study != expected_study or run != expected_run:
                                    exclusions["source_provenance_mismatch"] += 1
                                    continue
                                if len(heavy) + len(light) + 1 > authority["coordinate_rule"]["maximum_model_input_residues_including_separator"]:
                                    exclusions["model_input_too_long"] += 1
                                    continue
                                if not set(heavy + light + h1 + h2 + h3 + l1 + l2 + l3).issubset(AMINO_ACIDS):
                                    exclusions["noncanonical_normalized_content"] += 1
                                    continue
                                try:
                                    spans = [
                                        *(exact_span(row, "heavy", i, heavy, cdr) for i, cdr in enumerate((h1, h2, h3), 1)),
                                        *(exact_span(row, "light", i, light, cdr) for i, cdr in enumerate((l1, l2, l3), 1)),
                                    ]
                                except (KeyError, TypeError, ValueError) as error:
                                    exclusions[f"coordinate_error:{type(error).__name__}:{str(error)}"] += 1
                                    continue
                                if not (spans[0][1] <= spans[1][0] and spans[1][1] <= spans[2][0] and spans[3][1] <= spans[4][0] and spans[4][1] <= spans[5][0]):
                                    exclusions["unordered_or_overlapping_spans"] += 1
                                    continue
                                group = pair_hash(heavy, light)
                                split = "validation" if int(group[:16], 16) % 100 < 2 else "train"
                                if pair_splits.setdefault(group, split) != split:
                                    raise RuntimeError("pair-group split differs")
                                counts[f"eligible_{split}_rows"] += 1
                                value: dict[str, object] = {
                                    "source_row_index": source_index,
                                    "row_hash": key,
                                    "pair_sha256": group,
                                    "split": split,
                                    "heavy_length": len(heavy),
                                    "light_length": len(light),
                                }
                                for name, span in zip(("h1", "h2", "h3", "l1", "l2", "l3"), spans):
                                    value[f"{name}_start"], value[f"{name}_end"] = span
                                writer.writerow(value)
                    text.flush()
                raw_output.flush()
                os.fsync(raw_output.fileno())
        os.replace(temporary_name, args.output_spans)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise

    counts["target_rows"] = len(target)
    counts["target_rows_found_in_raw"] = len(found)
    counts["eligible_rows"] = counts["eligible_train_rows"] + counts["eligible_validation_rows"]
    counts["excluded_target_rows"] = counts["target_rows"] - counts["eligible_rows"]
    counts["unique_pair_groups"] = len(pair_splits)
    exclusions["target_row_hash_not_found"] = counts["target_rows"] - len(found)
    fraction = counts["eligible_rows"] / counts["target_rows"]
    gate = authority["frozen_gate"]
    checks = {
        "raw_file_count": counts["raw_files"] == authority["raw_oas_authority"]["expected_data_units"],
        "raw_source_row_count": counts["raw_source_rows"] == authority["raw_oas_authority"]["expected_source_rows"],
        "target_source_row_count": counts["target_rows"] == authority["target_source"]["expected_rows"],
        "minimum_eligible_rows": counts["eligible_rows"] >= gate["minimum_eligible_rows"],
        "minimum_eligible_fraction": fraction >= gate["minimum_eligible_fraction"],
        "minimum_validation_rows": counts["eligible_validation_rows"] >= gate["minimum_validation_rows"],
        "row_hash_join_unique": exclusions["duplicate_raw_row_hash"] == 0,
        "pair_group_disjoint": True,
        "target_accounting": counts["eligible_rows"] + counts["excluded_target_rows"] == counts["target_rows"],
    }
    passed = all(checks.values())
    result = {
        "format_version": "paired_esmc_cdr_span_gate_result_v3",
        "status": "PASS_EXACT_RAW_OAS_CDR_COORDINATE_GATE" if passed else "FAIL_CLOSED_EXACT_RAW_OAS_CDR_COORDINATE_GATE",
        "authority_sha256": sha256_file(args.authority),
        "source_sha256": authority["target_source"]["sha256"],
        "raw_plan_sha256": authority["raw_oas_authority"]["plan_sha256"],
        "span_artifact": {"path": str(args.output_spans), "sha256": sha256_file(args.output_spans)},
        "counts": dict(sorted(counts.items())),
        "eligible_fraction": fraction,
        "exclusion_reasons": {key: value for key, value in sorted(exclusions.items()) if value},
        "checks": checks,
        "audit": {
            "exact_raw_row_hash_join": True,
            "raw_coordinates_used": True,
            "motif_search_used": False,
            "pair_group_disjoint_by_construction": True,
            "checkpoint_selection": False,
            "abcdr_validation_or_test_rows_used": False,
            "task_labels_read": False,
            "sabdab2_test_access": False,
        },
        "authorization": {
            "local_whole_chain_vs_cdr_canary_authorized": passed,
            "production_training_authorized": False,
            "downstream_test_authorized": False,
        },
    }
    atomic_write(args.output_result, result)
    receipt = {
        "format_version": "paired_esmc_cdr_span_gate_receipt_v3",
        "status": result["status"],
        "result_sha256": sha256_file(args.output_result),
        "span_artifact_sha256": result["span_artifact"]["sha256"],
        "local_whole_chain_vs_cdr_canary_authorized": passed,
        "production_training_authorized": False,
        "abcdr_validation_or_test_rows_used": False,
        "task_labels_read": False,
        "sabdab2_test_access": False,
    }
    atomic_write(args.output_receipt, receipt)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
