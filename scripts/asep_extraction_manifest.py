#!/usr/bin/env python3
"""Write the manifest the representation extractor reads, for AsEP.

    python3 scripts/asep_extraction_manifest.py \
        --instances interaction_first_asep/instances.parquet \
        --split-manifest asep_epitope_group_split.json \
        --output asep_manifest.csv

The extractor reads a manifest with ``INSTANCE``, ``split``, ``vh_seq``,
``vl_seq`` and ``ag_seq``, and builds its ``ab`` chain by concatenating the two
antibody columns. AsEP does not store heavy and light separately: its antibody
side is one sequence of CDR residues drawn from both chains, and that is the
sequence the model's own store is keyed by. It therefore goes in ``vh_seq``
whole, with ``vl_seq`` empty, so that ``vh_seq + vl_seq`` is exactly the
antibody sequence and its digest is exactly ``ab_sequence_sha256``.

Splitting it would be wrong, not merely inconvenient: the halves would hash to
digests that appear nowhere, and an encoder taking the two separately would see
two fragments that are not chains.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

COLUMNS = ("INSTANCE", "split", "vh_seq", "vl_seq", "ag_seq")


def rows(instances, assignments: dict[str, str]) -> list[dict[str, str]]:
    """One manifest row per prepared instance that the split places."""
    result = []
    for record in instances.to_dict(orient="records"):
        identifier = str(record["instance_id"])
        if identifier not in assignments:
            continue
        antibody = str(record["ab_seq"])
        antigen = str(record["ag_seq"])
        expected = str(record["ab_sequence_sha256"])
        actual = hashlib.sha256(antibody.encode("ascii")).hexdigest()
        if actual != expected:
            raise ValueError(
                f"{identifier}: sha256(ab_seq) is {actual[:12]} but the table records "
                f"{expected[:12]}; the extractor would key its artifacts where nothing "
                "looks for them"
            )
        result.append(
            {
                "INSTANCE": identifier,
                "split": assignments[identifier],
                "vh_seq": antibody,
                "vl_seq": "",
                "ag_seq": antigen,
            }
        )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)

    import pandas

    payload = json.loads(arguments.split_manifest.read_text(encoding="utf-8"))
    assignments = {
        str(identifier): name
        for name, members in payload["splits"].items()
        for identifier in members
    }
    records = rows(pandas.read_parquet(arguments.instances), assignments)
    if not records:
        print("no instances matched the split manifest", file=sys.stderr)
        return 2

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COLUMNS))
        writer.writeheader()
        writer.writerows(records)

    counts: dict[str, int] = {}
    for record in records:
        counts[record["split"]] = counts.get(record["split"], 0) + 1
    print("  ".join(f"{name} {count}" for name, count in sorted(counts.items())))
    print(f"wrote {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
