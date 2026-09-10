#!/usr/bin/env python3
"""Where every frozen feature in this project came from.

    python3 scripts/feature_provenance.py --receipt name=/path/receipt.json ... \
        --output results/feature_provenance_v1/feature_provenance.tsv

Written after an arm reported as a structural encoder turned out to be a
sequence model read from someone else's cache. The name of a file is not
evidence about the tensor inside it, so this records, per arm, what can be
checked: the model actually run, its revision, the modality it consumed, which
chains it covers, and whether this project produced the features or copied them.

The column that matters most is `extracted_here`. An arm marked `copied` has a
model identity that rests on the upstream project's documentation unless a
tensor-level verification is recorded against it, and the `verification` column
says which.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# Tensor-level identity checks performed by
# scripts/identify_cached_representation.py. Keyed by receipt backend.
VERIFICATIONS = {
    "igfold": (
        "AntiBERTy confirmed by tensor identity: mean nearest-neighbour distance "
        "0.0055 against a row norm of 22.3 (ratio 2.5e-4) over the full antibody "
        "domain, heavy/light split swept; see results/feature_provenance_v1/"
    ),
}

# What the backend consumed. Read from the code, not guessed from the name.
MODALITY = {
    "antiberty": "sequence (antibody VH/VL, encoded per chain)",
    "esm2_t33_650M_UR50D": "sequence",
    "esm_if1_gvp4_t16_142M_UR50": "structure (backbone N, CA, C)",
    "igfold": "sequence (AntiBERTy over the full antibody domain)",
}

TRUE_MODEL = {
    "igfold": "AntiBERTy (NOT IgFold; see verification)",
}

COLUMNS = (
    "arm", "backend_recorded", "model_actually_run", "revision", "modality",
    "dim", "chains", "extracted_here", "source", "units_written",
    "instances_considered", "receipt_path", "verification",
)


def row_from_receipt(name: str, path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    backend = str(payload.get("backend", ""))
    revision = str(payload.get("revision", payload.get("model_revision", "")))
    copied = revision.startswith("asep-") or "interim" in revision
    return {
        "arm": name,
        "backend_recorded": backend,
        "model_actually_run": TRUE_MODEL.get(backend, backend),
        "revision": revision,
        "modality": MODALITY.get(backend, "UNRECORDED -- resolve before publication"),
        "dim": str(payload.get("representation_dim", "")),
        "chains": ",".join(payload.get("chains", []) or []),
        "extracted_here": "no (copied)" if copied else "yes",
        "source": ("upstream cache" if copied else "this project"),
        "units_written": str(payload.get("unique_sequences_written", "")),
        "instances_considered": str(payload.get("instances_considered", "")),
        "receipt_path": str(path),
        "verification": VERIFICATIONS.get(backend, "none -- identity rests on the receipt"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--receipt", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)

    rows, missing = [], []
    for specification in arguments.receipt:
        name, _, path = specification.partition("=")
        if not path:
            print(f"--receipt wants NAME=PATH, got {specification!r}", file=sys.stderr)
            return 2
        target = Path(path)
        if not target.is_file():
            missing.append(specification)
            continue
        rows.append(row_from_receipt(name, target))

    if not rows:
        print("no receipts found", file=sys.stderr)
        for entry in missing:
            print(f"  missing: {entry}", file=sys.stderr)
        return 2

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    width = max(len(row["arm"]) for row in rows)
    for row in rows:
        print(f"{row['arm']:<{width}}  {row['model_actually_run']:<40}  "
              f"{row['modality']:<45}  {row['extracted_here']}")
    for entry in missing:
        print(f"  missing receipt: {entry}", file=sys.stderr)
    print(f"\nwritten to {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
