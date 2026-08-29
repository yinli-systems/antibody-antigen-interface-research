#!/usr/bin/env python3
"""Audit AIRR query-coordinate CDR translation frames on raw paired OAS rows."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
import json
from pathlib import Path


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


def amino(value: str) -> str:
    return value.upper().replace("-", "").replace(".", "").strip()


def integer(value: str) -> int:
    number = float(value)
    if not number.is_integer():
        raise ValueError("non-integral coordinate")
    return int(number)


def translate(sequence: str, frame: int) -> str:
    sequence = sequence.upper().replace("U", "T")
    return "".join(CODONS.get(sequence[index:index + 3], "X") for index in range(frame, len(sequence) - 2, 3))


def metadata(handle: object) -> None:
    fields = next(csv.reader([handle.readline()]))
    if len(fields) != 1 or not isinstance(json.loads(fields[0]), dict):
        raise ValueError("malformed OAS metadata")


def audit_chain(row: dict[str, str], chain: str) -> str:
    query = row[f"sequence_{chain}"].upper().strip()
    expected = amino(row[f"sequence_alignment_aa_{chain}"])
    starts = [integer(row[f"cdr{i}_start_{chain}"]) for i in (1, 2, 3)]
    ends = [integer(row[f"cdr{i}_end_{chain}"]) for i in (1, 2, 3)]
    cdrs = [amino(row[f"cdr{i}_aa_{chain}"]) for i in (1, 2, 3)]
    phases = {(start - 1) % 3 for start in starts} | {end % 3 for end in ends}
    if len(phases) != 1:
        return "inconsistent_phase"
    frame = next(iter(phases))
    protein = translate(query, frame)
    occurrences = [index for index in range(len(protein) - len(expected) + 1) if protein.startswith(expected, index)]
    if len(occurrences) != 1:
        return f"expected_chain_occurrences:{len(occurrences)}:frame{frame}"
    offset = occurrences[0]
    spans = []
    for start, end, cdr in zip(starts, ends, cdrs):
        start_aa = (start - 1 - frame) // 3 - offset
        end_aa = (end - frame) // 3 - offset
        if start_aa < 0 or end_aa <= start_aa or expected[start_aa:end_aa] != cdr:
            return f"cdr_mismatch:frame{frame}"
        spans.append((start_aa, end_aa))
    if not (spans[0][1] <= spans[1][0] and spans[1][1] <= spans[2][0]):
        return f"unordered:frame{frame}"
    return f"pass:frame{frame}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--raw-plan", type=Path, required=True)
    parser.add_argument("--maximum-rows", type=int, default=100_000)
    args = parser.parse_args()
    counts = Counter()
    plan = json.loads(args.raw_plan.read_text())
    for entry in plan["files"]:
        with gzip.open(args.raw_root / entry["relative_path"], "rt", encoding="utf-8", newline="") as handle:
            metadata(handle)
            for row in csv.DictReader(handle):
                counts["rows"] += 1
                for chain in ("heavy", "light"):
                    try:
                        outcome = audit_chain(row, chain)
                    except (KeyError, TypeError, ValueError) as error:
                        outcome = f"error:{type(error).__name__}:{error}"
                    counts[f"{chain}:{outcome}"] += 1
                if counts["rows"] >= args.maximum_rows:
                    print(json.dumps(dict(sorted(counts.items())), sort_keys=True))
                    return
    print(json.dumps(dict(sorted(counts.items())), sort_keys=True))


if __name__ == "__main__":
    main()
