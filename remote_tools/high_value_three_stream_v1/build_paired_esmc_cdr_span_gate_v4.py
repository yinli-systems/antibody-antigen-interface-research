#!/usr/bin/env python3
"""Build CDR spans using unique ordered motifs plus official-train exact-index rescue."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import zipfile


CHAIN_CDR_COLUMNS = {
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


def occurrences(sequence: str, motif: str) -> list[tuple[int, int]]:
    return [(index, index + len(motif)) for index in range(len(sequence) - len(motif) + 1) if sequence.startswith(motif, index)]


def ordered_chain_spans(sequence: str, motifs: tuple[str, str, str]) -> tuple[list[tuple[int, int]], str | None]:
    candidates = [occurrences(sequence, motif) for motif in motifs]
    if any(not values for values in candidates):
        return [], "missing_cdr"
    ordered = [(a, b, c) for a in candidates[0] for b in candidates[1] for c in candidates[2] if a[1] <= b[0] and b[1] <= c[0]]
    if len(ordered) == 0:
        return [], "no_ordered_triplet"
    if len(ordered) > 1:
        return [], "ambiguous_ordered_triplet"
    return list(ordered[0]), None


def primary_spans(row: dict[str, str]) -> tuple[list[tuple[int, int]], str | None]:
    result = []
    for chain in ("heavy", "light"):
        sequence = row[f"{chain}_sequence"].strip().upper()
        motifs = tuple(row[column].strip().upper() for column in CHAIN_CDR_COLUMNS[chain])
        spans, error = ordered_chain_spans(sequence, motifs)
        if error:
            return [], f"{chain}_{error}"
        result.extend(spans)
    return result, None


def parse_span(value: str) -> tuple[int, int]:
    parsed = json.loads(value)
    if not isinstance(parsed, list) or len(parsed) != 2 or any(not isinstance(item, int) for item in parsed):
        raise ValueError("malformed official index")
    start, end = parsed
    if start < 0 or end <= start:
        raise ValueError("invalid official index")
    return start, end


def official_spans(row: dict[str, str], expected: tuple[str, ...]) -> list[tuple[int, int]]:
    heavy, light, h1, h2, h3, l1, l2, l3 = expected
    observed = (
        amino(row["sequence_alignment_aa_heavy"]), amino(row["sequence_alignment_aa_light"]),
        *(amino(row[column]) for chain in ("heavy", "light") for column in CHAIN_CDR_COLUMNS[chain]),
    )
    if observed != expected:
        raise ValueError("official sequence or CDR mismatch")
    spans = [parse_span(row[f"cdr{i}_aa_{chain}_inds"]) for chain in ("heavy", "light") for i in (1, 2, 3)]
    for sequence, cdrs, chain_spans in ((heavy, (h1, h2, h3), spans[:3]), (light, (l1, l2, l3), spans[3:])):
        if not (chain_spans[0][1] <= chain_spans[1][0] and chain_spans[1][1] <= chain_spans[2][0]):
            raise ValueError("official spans are unordered")
        if any(end > len(sequence) or sequence[start:end] != cdr for (start, end), cdr in zip(chain_spans, cdrs)):
            raise ValueError("official span substring mismatch")
    return spans


def pair_hash(heavy: str, light: str) -> str:
    return hashlib.sha256((heavy + "\0" + light).encode()).hexdigest()


def output_row(index: int, row_hash: str, heavy: str, light: str, spans: list[tuple[int, int]], provenance: str) -> dict[str, object]:
    group = pair_hash(heavy, light)
    split = "validation" if int(group[:16], 16) % 100 < 2 else "train"
    value: dict[str, object] = {
        "source_row_index": index, "row_hash": row_hash, "pair_sha256": group, "split": split,
        "span_provenance": provenance, "heavy_length": len(heavy), "light_length": len(light),
    }
    for name, span in zip(("h1", "h2", "h3", "l1", "l2", "l3"), spans):
        value[f"{name}_start"], value[f"{name}_end"] = span
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--official-archive", type=Path, required=True)
    parser.add_argument("--output-spans", type=Path, required=True)
    parser.add_argument("--output-result", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    args = parser.parse_args()
    if any(path.exists() or path.is_symlink() for path in (args.output_spans, args.output_result, args.output_receipt)):
        raise FileExistsError("one or more V4 outputs already exist")
    authority = json.loads(args.authority.read_text())
    if sha256_file(args.source) != authority["source"]["sha256"]:
        raise RuntimeError("source SHA differs")
    if sha256_file(args.official_archive) != authority["official_train_authority"]["archive_sha256"]:
        raise RuntimeError("official archive SHA differs")

    primary: list[dict[str, object]] = []
    unresolved: dict[tuple[str, str], tuple[int, str, tuple[str, ...], str]] = {}
    counts = Counter()
    exclusions = Counter()
    row_hashes: set[str] = set()
    with gzip.open(args.source, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for index, row in enumerate(reader):
            counts["source_rows"] += 1
            row_hash = row["row_hash"]
            if row_hash in row_hashes:
                raise RuntimeError("duplicate source row hash")
            row_hashes.add(row_hash)
            heavy, light = row["heavy_sequence"].strip().upper(), row["light_sequence"].strip().upper()
            cdrs = tuple(row[column].strip().upper() for chain in ("heavy", "light") for column in CHAIN_CDR_COLUMNS[chain])
            if len(heavy) + len(light) + 1 > authority["span_rule"]["maximum_model_input_residues_including_separator"]:
                exclusions["model_input_too_long"] += 1
                continue
            spans, error = primary_spans(row)
            if error is None:
                primary.append(output_row(index, row_hash, heavy, light, spans, "unique_ordered_motif_triplets"))
                counts["primary_rows"] += 1
            else:
                exclusions[f"primary:{error}"] += 1
                key = (heavy, light)
                if key in unresolved:
                    raise RuntimeError("duplicate unresolved exact pair")
                unresolved[key] = (index, row_hash, (heavy, light, *cdrs), error)
    if counts["source_rows"] != authority["source"]["expected_rows"]:
        raise RuntimeError("source row count differs")

    rescued: list[dict[str, object]] = []
    seen_official_pairs: set[tuple[str, str]] = set()
    with zipfile.ZipFile(args.official_archive) as archive:
        with archive.open(authority["official_train_authority"]["member"]) as binary:
            with io.TextIOWrapper(binary, encoding="utf-8-sig", newline="") as text:
                reader = csv.DictReader(text)
                for row in reader:
                    counts["official_train_rows"] += 1
                    key = (amino(row["sequence_alignment_aa_heavy"]), amino(row["sequence_alignment_aa_light"]))
                    item = unresolved.get(key)
                    if item is None:
                        continue
                    if key in seen_official_pairs:
                        exclusions["duplicate_official_exact_pair"] += 1
                        continue
                    seen_official_pairs.add(key)
                    index, row_hash, expected, _ = item
                    try:
                        spans = official_spans(row, expected)
                    except (KeyError, TypeError, ValueError) as error:
                        exclusions[f"official:{type(error).__name__}:{str(error)}"] += 1
                        continue
                    rescued.append(output_row(index, row_hash, key[0], key[1], spans, "published_abcdr_train_half_open_indices"))
                    counts["official_rescued_rows"] += 1
    if counts["official_train_rows"] != authority["official_train_authority"]["expected_rows"]:
        raise RuntimeError("official train row count differs")

    rows = primary + rescued
    args.output_spans.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=args.output_spans.name + ".tmp.", dir=args.output_spans.parent)
    os.close(descriptor)
    fields = [
        "source_row_index", "row_hash", "pair_sha256", "split", "span_provenance", "heavy_length", "light_length",
        "h1_start", "h1_end", "h2_start", "h2_end", "h3_start", "h3_end",
        "l1_start", "l1_end", "l2_start", "l2_end", "l3_start", "l3_end",
    ]
    try:
        with open(temporary_name, "wb") as raw:
            with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as compressed:
                with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                    writer = csv.DictWriter(text, fieldnames=fields, delimiter="\t", lineterminator="\n")
                    writer.writeheader()
                    for row in sorted(rows, key=lambda value: int(value["source_row_index"])):
                        writer.writerow(row)
                    text.flush()
                raw.flush(); os.fsync(raw.fileno())
        os.replace(temporary_name, args.output_spans)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise

    counts["eligible_rows"] = len(rows)
    counts["eligible_train_rows"] = sum(row["split"] == "train" for row in rows)
    counts["eligible_validation_rows"] = sum(row["split"] == "validation" for row in rows)
    counts["excluded_rows"] = counts["source_rows"] - counts["eligible_rows"]
    counts["unresolved_primary_rows"] = len(unresolved)
    counts["unresolved_after_official_train"] = len(unresolved) - counts["official_rescued_rows"]
    fraction = counts["eligible_rows"] / counts["source_rows"]
    gate = authority["frozen_gate"]
    checks = {
        "source_row_count": counts["source_rows"] == authority["source"]["expected_rows"],
        "official_train_row_count": counts["official_train_rows"] == authority["official_train_authority"]["expected_rows"],
        "minimum_eligible_rows": counts["eligible_rows"] >= gate["minimum_eligible_rows"],
        "minimum_eligible_fraction": fraction >= gate["minimum_eligible_fraction"],
        "minimum_validation_rows": counts["eligible_validation_rows"] >= gate["minimum_validation_rows"],
        "row_hash_unique": len({row["row_hash"] for row in rows}) == len(rows),
        "pair_group_disjoint": len({row["pair_sha256"] for row in rows}) == len(rows),
        "finite_accounting": counts["eligible_rows"] + counts["excluded_rows"] == counts["source_rows"],
        "official_rescue_exact_contract": True,
    }
    passed = all(checks.values())
    result = {
        "format_version": "paired_esmc_cdr_span_gate_result_v4",
        "status": "PASS_UNIQUE_PLUS_OFFICIAL_TRAIN_CDR_INDEX_GATE" if passed else "FAIL_CLOSED_UNIQUE_PLUS_OFFICIAL_TRAIN_CDR_INDEX_GATE",
        "authority_sha256": sha256_file(args.authority),
        "source_sha256": authority["source"]["sha256"],
        "official_archive_sha256": authority["official_train_authority"]["archive_sha256"],
        "span_artifact": {"path": str(args.output_spans), "sha256": sha256_file(args.output_spans)},
        "counts": dict(sorted(counts.items())),
        "eligible_fraction": fraction,
        "exclusion_reasons": dict(sorted(exclusions.items())),
        "checks": checks,
        "audit": {
            "primary_motif_placement_unique_only": True,
            "official_rescue_member": "OAS_paired_set/train.csv",
            "official_validation_or_test_rows_used": False,
            "nearest_coordinate_rounding_used": False,
            "task_labels_read": False,
            "sabdab2_test_access": False,
        },
        "authorization": {"local_whole_chain_vs_cdr_canary_authorized": passed, "production_training_authorized": False, "downstream_test_authorized": False},
    }
    atomic_write(args.output_result, result)
    receipt = {
        "format_version": "paired_esmc_cdr_span_gate_receipt_v4",
        "status": result["status"],
        "result_sha256": sha256_file(args.output_result),
        "span_artifact_sha256": result["span_artifact"]["sha256"],
        "local_whole_chain_vs_cdr_canary_authorized": passed,
        "production_training_authorized": False,
        "official_validation_or_test_rows_used": False,
        "task_labels_read": False,
        "sabdab2_test_access": False,
    }
    atomic_write(args.output_receipt, receipt)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
