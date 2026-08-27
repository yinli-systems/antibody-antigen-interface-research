#!/usr/bin/env python3
"""Recover exact CDR spans from AIRR query coordinates and the query reading frame."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
import io
import json
import os
from pathlib import Path
import tempfile

from build_paired_esmc_cdr_span_gate_v3 import (
    AMINO_ACIDS,
    CDR_COLUMNS,
    amino,
    atomic_write,
    integer_coordinate,
    metadata,
    pair_hash,
    row_hash,
    sha256_file,
)


CODONS = {
    "TTT":"F","TTC":"F","TTA":"L","TTG":"L","TCT":"S","TCC":"S","TCA":"S","TCG":"S",
    "TAT":"Y","TAC":"Y","TAA":"*","TAG":"*","TGT":"C","TGC":"C","TGA":"*","TGG":"W",
    "CTT":"L","CTC":"L","CTA":"L","CTG":"L","CCT":"P","CCC":"P","CCA":"P","CCG":"P",
    "CAT":"H","CAC":"H","CAA":"Q","CAG":"Q","CGT":"R","CGC":"R","CGA":"R","CGG":"R",
    "ATT":"I","ATC":"I","ATA":"I","ATG":"M","ACT":"T","ACC":"T","ACA":"T","ACG":"T",
    "AAT":"N","AAC":"N","AAA":"K","AAG":"K","AGT":"S","AGC":"S","AGA":"R","AGG":"R",
    "GTT":"V","GTC":"V","GTA":"V","GTG":"V","GCT":"A","GCC":"A","GCA":"A","GCG":"A",
    "GAT":"D","GAC":"D","GAA":"E","GAG":"E","GGT":"G","GGC":"G","GGA":"G","GGG":"G",
}


def translate_query(sequence: str, frame: int) -> str:
    sequence = sequence.upper().replace("U", "T").strip()
    return "".join(
        CODONS.get(sequence[index:index + 3], "X")
        for index in range(frame, len(sequence) - 2, 3)
    )


def exact_chain_spans(
    row: dict[str, str], chain: str, expected_sequence: str, expected_cdrs: tuple[str, str, str]
) -> tuple[list[tuple[int, int]], int]:
    aligned = row[f"sequence_alignment_aa_{chain}"].upper().strip()
    if amino(aligned) != expected_sequence:
        raise ValueError("normalized sequence mismatch")
    query = row[f"sequence_{chain}"].upper().strip()
    starts = [integer_coordinate(row[f"cdr{i}_start_{chain}"]) for i in (1, 2, 3)]
    ends = [integer_coordinate(row[f"cdr{i}_end_{chain}"]) for i in (1, 2, 3)]
    phases = {(start - 1) % 3 for start in starts} | {end % 3 for end in ends}
    if len(phases) != 1:
        raise ValueError("inconsistent query reading frame")
    frame = next(iter(phases))
    protein = translate_query(query, frame)
    occurrences = [
        index for index in range(len(protein) - len(expected_sequence) + 1)
        if protein.startswith(expected_sequence, index)
    ]
    if len(occurrences) != 1:
        raise ValueError(f"normalized chain query-translation occurrences={len(occurrences)}")
    offset = occurrences[0]
    spans: list[tuple[int, int]] = []
    for start, end, cdr in zip(starts, ends, expected_cdrs):
        start_offset = start - 1 - frame
        end_offset = end - frame
        if start_offset < 0 or end_offset <= start_offset or start_offset % 3 or end_offset % 3:
            raise ValueError("query coordinate frame mismatch")
        chain_start = start_offset // 3 - offset
        chain_end = end_offset // 3 - offset
        if chain_start < 0 or chain_end > len(expected_sequence) or chain_end <= chain_start:
            raise ValueError("query coordinate beyond normalized chain")
        if expected_sequence[chain_start:chain_end] != cdr:
            raise ValueError("query-coordinate CDR mismatch")
        spans.append((chain_start, chain_end))
    if not (spans[0][1] <= spans[1][0] and spans[1][1] <= spans[2][0]):
        raise ValueError("unordered or overlapping CDR spans")
    return spans, frame


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
        raise FileExistsError("one or more V5 outputs already exist")
    authority = json.loads(args.authority.read_text())
    if authority.get("protocol_status") != "frozen_before_airr_query_coordinate_recovery":
        raise RuntimeError("authority is not frozen")
    if sha256_file(args.source) != authority["target_source"]["sha256"]:
        raise RuntimeError("target source SHA differs")
    if sha256_file(args.raw_plan) != authority["raw_oas_authority"]["plan_sha256"]:
        raise RuntimeError("raw OAS plan SHA differs")

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
            target[key] = (
                index,
                (
                    row["heavy_sequence"].strip().upper(), row["light_sequence"].strip().upper(),
                    *(row[column].strip().upper() for chain in ("heavy", "light") for column in CDR_COLUMNS[chain]),
                    row["study"], row["run"],
                ),
            )
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
        "source_row_index", "row_hash", "pair_sha256", "split", "heavy_reading_frame", "light_reading_frame",
        "heavy_length", "light_length", "h1_start", "h1_end", "h2_start", "h2_end", "h3_start", "h3_end",
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
                            for row in csv.DictReader(handle):
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
                                    heavy_spans, heavy_frame = exact_chain_spans(row, "heavy", heavy, (h1, h2, h3))
                                    light_spans, light_frame = exact_chain_spans(row, "light", light, (l1, l2, l3))
                                except (KeyError, TypeError, ValueError) as error:
                                    exclusions[f"coordinate_error:{type(error).__name__}:{str(error)}"] += 1
                                    continue
                                spans = heavy_spans + light_spans
                                group = pair_hash(heavy, light)
                                split = "validation" if int(group[:16], 16) % 100 < 2 else "train"
                                if pair_splits.setdefault(group, split) != split:
                                    raise RuntimeError("pair-group split differs")
                                counts[f"eligible_{split}_rows"] += 1
                                counts[f"heavy_frame_{heavy_frame}"] += 1
                                counts[f"light_frame_{light_frame}"] += 1
                                value: dict[str, object] = {
                                    "source_row_index": source_index, "row_hash": key, "pair_sha256": group,
                                    "split": split, "heavy_reading_frame": heavy_frame, "light_reading_frame": light_frame,
                                    "heavy_length": len(heavy), "light_length": len(light),
                                }
                                for name, span in zip(("h1", "h2", "h3", "l1", "l2", "l3"), spans):
                                    value[f"{name}_start"], value[f"{name}_end"] = span
                                writer.writerow(value)
                    text.flush()
                raw_output.flush(); os.fsync(raw_output.fileno())
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
        "format_version": "paired_esmc_cdr_span_gate_result_v5",
        "status": "PASS_EXACT_AIRR_QUERY_COORDINATE_GATE" if passed else "FAIL_CLOSED_EXACT_AIRR_QUERY_COORDINATE_GATE",
        "authority_sha256": sha256_file(args.authority),
        "source_sha256": authority["target_source"]["sha256"],
        "raw_plan_sha256": authority["raw_oas_authority"]["plan_sha256"],
        "span_artifact": {"path": str(args.output_spans), "sha256": sha256_file(args.output_spans)},
        "counts": dict(sorted(counts.items())), "eligible_fraction": fraction,
        "exclusion_reasons": {key: value for key, value in sorted(exclusions.items()) if value},
        "checks": checks,
        "audit": {
            "exact_raw_row_hash_join": True, "airr_query_coordinates_used": True,
            "query_reading_frame_required_consistent_across_all_six_coordinate_endpoints": True,
            "exact_full_normalized_chain_translation_match_required": True,
            "exact_all_six_cdr_substring_matches_required": True, "nearest_coordinate_rounding_used": False,
            "abcdr_validation_or_test_rows_used": False, "task_labels_read": False, "sabdab2_test_access": False,
        },
        "authorization": {
            "local_whole_chain_vs_cdr_canary_authorized": passed,
            "production_training_authorized": False, "downstream_test_authorized": False,
        },
    }
    atomic_write(args.output_result, result)
    receipt = {
        "format_version": "paired_esmc_cdr_span_gate_receipt_v5", "status": result["status"],
        "result_sha256": sha256_file(args.output_result), "span_artifact_sha256": result["span_artifact"]["sha256"],
        "local_whole_chain_vs_cdr_canary_authorized": passed, "production_training_authorized": False,
        "abcdr_validation_or_test_rows_used": False, "task_labels_read": False, "sabdab2_test_access": False,
    }
    atomic_write(args.output_receipt, receipt)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
