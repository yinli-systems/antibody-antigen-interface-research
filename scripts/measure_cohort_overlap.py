#!/usr/bin/env python3
"""Measure structural overlap between two cohorts, by PDB entry.

    python3 scripts/measure_cohort_overlap.py \
        --cohort sabdab2=instances.parquet:split.json \
        --cohort asep=asep_instances.parquet:asep_split.json \
        --json overlap.json

Two cohorts assembled from the Protein Data Bank share entries whether or not
they share identifiers, and their identifier schemes usually differ. Sequence
digests cannot answer this either, because cohorts select different residues:
one may store a full variable region and another only the CDRs, so the same
structure hashes differently in each.

The PDB accession is what survives both. It is the coarsest honest unit for the
question that matters here -- whether a model that trained on one cohort has
already seen the structures another cohort will test it on.

Overlap is reported per split pair rather than as one number, because that is
the form the answer is needed in: a train-to-test overlap invalidates a
benchmark claim, while a train-to-train overlap merely wastes data.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# pdb_00001j1p_H_L -> 1j1p ; 1a14_0P -> 1a14
PATTERNS = (
    re.compile(r"^pdb_0*([0-9a-zA-Z]{4})_"),
    re.compile(r"^([0-9a-zA-Z]{4})_"),
)
SPLITS = ("train", "validation", "test")


class OverlapError(RuntimeError):
    """Raised when a cohort cannot be read as instances plus splits."""


def pdb_code(instance_id: str) -> str | None:
    """Extract the PDB accession from an instance identifier, or None."""
    text = str(instance_id)
    for pattern in PATTERNS:
        match = pattern.match(text)
        if match:
            return match.group(1).lower()
    return None


def load_cohort(instances: Path, split_manifest: Path) -> dict[str, set[str]]:
    """Return the PDB accessions per split for one cohort."""
    import pandas

    known = set(pandas.read_parquet(instances)["instance_id"].astype(str))
    payload = json.loads(split_manifest.read_text(encoding="utf-8"))
    splits = payload.get("splits")
    if not isinstance(splits, dict):
        raise OverlapError(f"{split_manifest} has no 'splits' mapping")

    result: dict[str, set[str]] = {}
    unresolved = 0
    for name in SPLITS:
        codes = set()
        for identifier in splits.get(name, []):
            if str(identifier) not in known:
                continue
            code = pdb_code(identifier)
            if code is None:
                unresolved += 1
                continue
            codes.add(code)
        result[name] = codes
    if unresolved:
        raise OverlapError(
            f"{unresolved} identifiers in {split_manifest} carry no recognisable "
            "PDB accession; a silent drop would understate the overlap"
        )
    return result


def compare(left: dict[str, set[str]], right: dict[str, set[str]]) -> dict:
    """Report every split-pair overlap, and flag the ones that invalidate claims."""
    pairs = {}
    for left_name in SPLITS:
        for right_name in SPLITS:
            shared = left[left_name] & right[right_name]
            denominator = len(right[right_name])
            pairs[f"{left_name}_vs_{right_name}"] = {
                "left_split": left_name,
                "right_split": right_name,
                "left_pdb_count": len(left[left_name]),
                "right_pdb_count": denominator,
                "shared_pdb_count": len(shared),
                "share_of_right": (len(shared) / denominator) if denominator else None,
                "contaminating": left_name == "train" and right_name in ("validation", "test"),
            }
    return pairs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cohort", action="append", required=True,
                        help="name=instances.parquet:split.json, given twice")
    parser.add_argument("--json", type=Path, default=None)
    arguments = parser.parse_args(argv)

    if len(arguments.cohort) != 2:
        print("exactly two --cohort arguments are required", file=sys.stderr)
        return 2

    cohorts = {}
    for specification in arguments.cohort:
        name, _, paths = specification.partition("=")
        instances, _, split_manifest = paths.partition(":")
        if not name or not instances or not split_manifest:
            print(f"malformed --cohort {specification!r}", file=sys.stderr)
            return 2
        cohorts[name] = load_cohort(Path(instances), Path(split_manifest))

    (left_name, left), (right_name, right) = cohorts.items()
    report = {
        "format_version": "cohort_overlap_v1",
        "unit": "pdb_accession",
        "left": left_name,
        "right": right_name,
        "left_totals": {name: len(left[name]) for name in SPLITS},
        "right_totals": {name: len(right[name]) for name in SPLITS},
        "pairs": compare(left, right),
        "reverse_pairs": compare(right, left),
        "claim_boundary": (
            "Overlap by PDB accession only. Two entries of the same structure are "
            "counted as shared regardless of chains, resolution or preparation."
        ),
    }

    print(f"{left_name} -> {right_name}   (share of the right-hand split)")
    for key, entry in report["pairs"].items():
        if entry["share_of_right"] is None:
            continue
        flag = "  <-- contaminating" if entry["contaminating"] and entry["shared_pdb_count"] else ""
        print(f"  {key:<26s} {entry['shared_pdb_count']:4d} / {entry['right_pdb_count']:4d}"
              f"  {100 * entry['share_of_right']:5.1f}%{flag}")

    if arguments.json:
        arguments.json.parent.mkdir(parents=True, exist_ok=True)
        arguments.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                                  encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
