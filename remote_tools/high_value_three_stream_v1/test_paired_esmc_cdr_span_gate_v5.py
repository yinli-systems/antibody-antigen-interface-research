#!/usr/bin/env python3
"""Contract tests for AIRR query-coordinate CDR span conversion."""

from __future__ import annotations

from build_paired_esmc_cdr_span_gate_v5 import CODONS, exact_chain_spans


ENCODE = {amino: codon for codon, amino in CODONS.items() if amino != "*"}


def row_for(sequence: str, frame: int) -> tuple[dict[str, str], tuple[str, str, str]]:
    cdrs = (sequence[2:4], sequence[6:9], sequence[11:13])
    spans = ((2, 4), (6, 9), (11, 13))
    prefix = "A" * frame
    query = prefix + "".join(ENCODE[amino] for amino in sequence)
    row = {"sequence_alignment_aa_heavy": sequence, "sequence_heavy": query}
    for number, ((start, end), cdr) in enumerate(zip(spans, cdrs), 1):
        row[f"cdr{number}_start_heavy"] = str(frame + start * 3 + 1)
        row[f"cdr{number}_end_heavy"] = str(frame + end * 3)
        row[f"cdr{number}_aa_heavy"] = cdr
    return row, cdrs


def main() -> None:
    sequence = "ACDEFGHIKLMNPQRSTVWY"
    expected = [(2, 4), (6, 9), (11, 13)]
    for frame in (0, 1, 2):
        row, cdrs = row_for(sequence, frame)
        spans, observed_frame = exact_chain_spans(row, "heavy", sequence, cdrs)
        assert spans == expected
        assert observed_frame == frame
    row, cdrs = row_for(sequence, 2)
    row["cdr3_end_heavy"] = str(int(row["cdr3_end_heavy"]) + 1)
    try:
        exact_chain_spans(row, "heavy", sequence, cdrs)
    except ValueError as error:
        assert str(error) == "inconsistent query reading frame"
    else:
        raise AssertionError("inconsistent frame was accepted")
    print("PASS")


if __name__ == "__main__":
    main()
