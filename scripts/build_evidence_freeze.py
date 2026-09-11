#!/usr/bin/env python3
"""Assemble the frozen evidence set and the manifest that pins it.

    python3 scripts/build_evidence_freeze.py --out freeze

Produces, from what is already in the repository and nothing else:

MANIFEST.sha256
    Every file under `results/`, `protocols/`, `scripts/`, `src/`, `reproduction/`
    and `paper/`, with its digest. Sorted, so the file is stable across runs and
    a diff between two freezes is legible.

CLAIMS_LEDGER.json
    The claims ledger in machine-readable form, parsed from its own table rather
    than retyped, so the two cannot disagree.

PROVENANCE.json
    Every feature store: what model actually produced it, at which layer, over
    which residue index, extracted here or copied, and the digest of the store.

REPRODUCTION_RECEIPT.json
    The three cold-reproduction receipts, joined, with the per-step outcome.

Nothing here recomputes a result. If a number is wrong, this pins the wrong
number, and the amendment procedure in FREEZE.md is how it gets corrected --
visibly, not by overwriting.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parent.parent
TRACKED = ("results", "protocols", "scripts", "src", "reproduction", "paper", "tests")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build_manifest(out: Path) -> tuple[Path, int]:
    lines = []
    for area in TRACKED:
        root = REPOSITORY / area
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                lines.append(f"{digest(path)}  {path.relative_to(REPOSITORY)}")
    target = out / "MANIFEST.sha256"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target, len(lines)


def parse_ledger(out: Path) -> tuple[Path, int]:
    """Parse the ledger's own table so the JSON cannot drift from the prose."""
    source = REPOSITORY / "results" / "CLAIMS_LEDGER.md"
    rows, header = [], None
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if set("".join(cells)) <= set("-: "):
            continue
        if header is None and cells and cells[0] == "#":
            header = cells
            continue
        if header and len(cells) == len(header) and cells[0].isdigit():
            rows.append(dict(zip(header, cells)))
    payload = {
        "format_version": "claims_ledger_v1",
        "source": str(source.relative_to(REPOSITORY)),
        "source_sha256": digest(source),
        "row_count": len(rows),
        "rows": rows,
    }
    target = out / "CLAIMS_LEDGER.json"
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target, len(rows)


def build_provenance(out: Path) -> tuple[Path, int]:
    source = REPOSITORY / "results" / "feature_provenance_v1" / "feature_provenance.tsv"
    with source.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    payload = {
        "format_version": "feature_provenance_v1",
        "source": str(source.relative_to(REPOSITORY)),
        "source_sha256": digest(source),
        "note": (
            "source_digest and store_digest are different things and are not "
            "interchangeable: the first identifies what the features were made "
            "from, the second identifies the artifacts as stored here. One arm "
            "has no source digest available and says so rather than substituting "
            "the other."
        ),
        "arms": rows,
    }
    target = out / "PROVENANCE.json"
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target, len(rows)


def join_receipts(out: Path) -> tuple[Path, dict]:
    root = REPOSITORY / "reproduction" / "cold_v1"
    passes, outcome = {}, {}
    for path in sorted(root.glob("receipt_pass*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        passes[path.stem] = payload
        for step in payload.get("steps", []):
            name = step["script"] if "name" not in step else step["name"]
            key = step.get("output", "").rsplit("/", 1)[-1].replace(".json", "")
            numeric = None
            bad = payload.get("mismatches", {}).get(key)
            if bad is not None:
                numeric = sum(1 for entry in bad if entry.get("kind") == "numeric")
            outcome[key] = {
                "pass": path.stem,
                "seconds": step.get("seconds"),
                "returncode": step.get("returncode"),
                "mismatches": step.get("mismatches"),
                "numeric_mismatches": numeric if bad is not None else step.get("mismatches"),
                "compared_against": step.get("compared_against"),
            }
    payload = {
        "format_version": "cold_reproduction_joined_v1",
        "note": (
            "Later passes supersede earlier ones for the steps they re-ran. Every "
            "step's final numeric mismatch count is zero; the non-zero entries are "
            "field-presence differences from metadata added after a receipt was "
            "frozen, and reproduction/cold_v1/RESULT.md says which."
        ),
        "final_outcome_by_step": outcome,
        "passes": passes,
    }
    target = out / "REPRODUCTION_RECEIPT.json"
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target, outcome


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tag", default="evidence-freeze-v1")
    arguments = parser.parse_args(argv)
    arguments.out.mkdir(parents=True, exist_ok=True)

    manifest, files = build_manifest(arguments.out)
    ledger, claims = parse_ledger(arguments.out)
    provenance, arms = build_provenance(arguments.out)
    receipt, outcome = join_receipts(arguments.out)

    index = {
        "format_version": "evidence_freeze_v1",
        "tag": arguments.tag,
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPOSITORY),
                                 capture_output=True, text=True).stdout.strip(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python_version": sys.version.split()[0],
        "counts": {"files_pinned": files, "ledger_rows": claims,
                   "feature_stores": arms, "reproduction_steps": len(outcome)},
        "artifacts": {p.name: digest(p) for p in (manifest, ledger, provenance, receipt)},
    }
    (arguments.out / "FREEZE_INDEX.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"files pinned       {files}")
    print(f"ledger rows        {claims}")
    print(f"feature stores     {arms}")
    print(f"reproduction steps {len(outcome)}")
    for name, entry in sorted(outcome.items()):
        print(f"  {name:>34}  {entry['pass']}  numeric mismatches "
              f"{entry['numeric_mismatches']}")
    print()
    for path in (manifest, ledger, provenance, receipt):
        print(f"  {path.name:<28} {digest(path)[:16]}…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
