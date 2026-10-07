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
import hashlib
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

# Which tensor the encoder was asked for. Read from this project's own code for
# what it extracts, and established by the identity check for what it copied.
EXTRACTION_LAYER = {
    "antiberty": "AntiBERTyRunner.embed, final hidden state, BOS/EOS dropped",
    "esm2_t33_650M_UR50D": "final representation layer, BOS/EOS dropped",
    "esm_if1_gvp4_t16_142M_UR50": (
        "encoder_out['encoder_out'][0], the GVP-transformer encoder output; "
        "the two CoordBatchConverter pad rows are dropped"
    ),
    "igfold": (
        "matches AntiBERTyRunner.embed to 2.5e-4, so the cache is the same "
        "layer this project extracts for its own AntiBERTy arm"
    ),
    "proteinmpnn_v_48_020": (
        "node embeddings after the three encoder layers; the autoregressive "
        "decoder is not run, so nothing is conditioned on sequence"
    ),
}

# What one row of the stored array indexes, measured by
# scripts/audit_row_to_residue_mapping.py rather than assumed from the name.
#
# Keyed by ARM, not by backend. The same encoder over two cohorts produces two
# different mappings -- sabdab2 stores full chains and AsEP stores
# interface-selected rows -- and a backend-keyed table gave the AsEP ESM-IF1 arm
# the sabdab2 answer, which is the same shape of error as every other lookup in
# this project that was keyed by the wrong thing.
RESIDUE_MAPPING = {
    "esm_if1_sabdab2": (
        "sabdab2: full VH+VL and full antigen chains; consecutive rows are "
        "consecutive residues at 99.57% (antibody) and 99.86% (antigen) of steps"
    ),
    "esm_if1_asep": (
        "AsEP: interface-selected rows; consecutive rows jump segments at 8.0% "
        "(antibody) and 7.0% (antigen) of steps, 99th-percentile Ca-Ca step "
        "24.9 A -- index space only"
    ),
    "antiberty_sabdab2": "sabdab2: full VH and VL, encoded per chain, concatenated",
    "igfold_arm_asep": (
        "AsEP: interface-selected rows; consecutive rows jump segments at 8.0% "
        "of steps, 99th-percentile Ca-Ca step 24.9 A -- index space only"
    ),
    "proteinmpnn_sabdab2": (
        "sabdab2: the same backbone store and the same row order as ESM-IF1"
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
    "extraction_layer", "residue_mapping", "dim", "chains", "extracted_here",
    "source", "source_digest", "store_digest", "units_written",
    "instances_considered", "receipt_path", "verification",
)


def store_digest(root: Path, limit: int = 40000) -> str:
    """A content digest of an artifact store.

    sha256 over the sorted (relative path, file digest) pairs, so it changes if
    any artifact's bytes or name change and does not depend on filesystem order.
    Returns a marker rather than a wrong answer when the store is absent.
    """
    if not root.is_dir():
        return "store not on this filesystem"
    outer = hashlib.sha256()
    count = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        count += 1
        if count > limit:
            return f"more than {limit} artifacts; digest not computed"
        inner = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                inner.update(block)
        outer.update(str(path.relative_to(root)).encode("utf-8"))
        outer.update(inner.digest())
    return f"{outer.hexdigest()} over {count} artifacts" if count else "store is empty"


def row_from_receipt(name: str, path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    backend = str(payload.get("backend", ""))
    revision = str(payload.get("revision", payload.get("model_revision", "")))
    copied = revision.startswith("asep-") or "interim" in revision
    root = path.parent / path.name.replace("_receipt.json", "")
    return {
        "arm": name,
        "backend_recorded": backend,
        "model_actually_run": TRUE_MODEL.get(backend, backend),
        "revision": revision,
        "modality": MODALITY.get(backend, "UNRECORDED -- resolve before publication"),
        "extraction_layer": EXTRACTION_LAYER.get(
            backend, "UNRECORDED -- resolve before publication"),
        "residue_mapping": RESIDUE_MAPPING.get(
            name, "UNMEASURED for this arm -- run audit_row_to_residue_mapping.py "
                  "on its own cohort; the mapping is a property of the cohort, "
                  "not of the encoder"),
        "dim": str(payload.get("representation_dim", "")),
        "chains": ",".join(payload.get("chains", []) or []),
        "extracted_here": "no (copied)" if copied else "yes",
        "source": ("upstream cache" if copied else "this project"),
        "source_digest": (
            "AsEP interim graphs, no longer on this filesystem; identity rests on "
            "the tensor check in the verification column, not on a hash"
            if copied else "model weights as distributed, see revision"
        ),
        "store_digest": store_digest(root),
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
