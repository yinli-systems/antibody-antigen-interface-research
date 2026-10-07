#!/usr/bin/env python3
"""Report protocol-governed source files whose current bytes are unregistered.

Frozen protocols register the SHA-256 of the source they govern. Evidence is
append-only, so a superseded protocol keeps its older hash and that is correct.
The defect this check detects is narrower: a governed file for which *no*
registration matches the bytes now in the tree. Such a file has silently escaped
its own governance, and every receipt that cites the protocol inherits the gap.

Usage:

    python3 scripts/verify_protocol_source_bindings.py
    python3 scripts/verify_protocol_source_bindings.py --json
    python3 scripts/verify_protocol_source_bindings.py --write-baseline

``--write-baseline`` records the drift that exists today so continuous
integration can fail on *new* drift. It never edits a protocol: recording an
observation is not the same as reconciling it, and each recorded entry still
needs a numbered supersession before its evidence may be cited as current.

Exit status is ``0`` when no unbaselined drift remains and ``1`` otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from interaction_first.protocol_source_bindings import (  # noqa: E402
    BASELINE_FORMAT_VERSION,
    BASELINE_RELATIVE_PATH,
    collect_bindings,
    find_drift,
    load_baseline,
    resolved_baseline_paths,
    source_roots,
    unbaselined_drift,
)


def build_argument_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit a machine-readable report instead of the text summary",
    )
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help=f"record current drift in {BASELINE_RELATIVE_PATH} and exit successfully",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Verify the bindings and report unregistered source drift."""
    arguments = build_argument_parser().parse_args(argv)

    bindings = collect_bindings(REPOSITORY_ROOT)
    drift = find_drift(REPOSITORY_ROOT, bindings)
    baseline = load_baseline(REPOSITORY_ROOT)
    fresh = unbaselined_drift(drift, baseline)
    resolved = resolved_baseline_paths(drift, baseline)

    if arguments.write_baseline:
        entries = []
        for record in drift:
            entry = record.as_json()
            previous = baseline.get(record.registered_path)
            if previous is not None and previous.get("actual_sha256") == record.actual_sha256:
                for annotation in ("classification", "first_unregistered_in", "note"):
                    if annotation in previous:
                        entry[annotation] = previous[annotation]
            entries.append(entry)
        payload = {
            "format_version": BASELINE_FORMAT_VERSION,
            "claim_boundary": (
                "Observation only. Each entry records a protocol-governed file whose "
                "current bytes no protocol registers. Recording drift does not "
                "reconcile it and does not authorize citing the affected protocol as "
                "current evidence; that requires a numbered supersession."
            ),
            "accepted_drift": entries,
        }
        target = REPOSITORY_ROOT / BASELINE_RELATIVE_PATH
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"recorded {len(drift)} drifted file(s) in {BASELINE_RELATIVE_PATH}")
        return 0

    if arguments.json:
        report = {
            "source_roots": list(source_roots(REPOSITORY_ROOT)),
            "binding_count": len(bindings),
            "governed_file_count": len({binding.registered_path for binding in bindings}),
            "drifted": [record.as_json() for record in drift],
            "unbaselined": [record.as_json() for record in fresh],
            "resolved_baseline_paths": list(resolved),
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        return 1 if fresh else 0

    governed = len({binding.registered_path for binding in bindings})
    print(f"source roots        : {', '.join(source_roots(REPOSITORY_ROOT))}")
    print(f"registrations       : {len(bindings)}")
    print(f"governed files      : {governed}")
    print(f"drifted files       : {len(drift)} ({len(baseline)} recorded in the baseline)")

    if resolved:
        print()
        print("Baseline entries that no longer drift; drop them from the baseline:")
        for registered_path in resolved:
            print(f"  {registered_path}")

    if not fresh:
        print()
        print("No unbaselined source drift.")
        return 0

    print()
    print(f"{len(fresh)} protocol-governed file(s) have unregistered bytes:")
    for record in fresh:
        print()
        print(f"  {record.registered_path}")
        print(f"    current    {record.actual_sha256}")
        for registered in record.registered_sha256:
            print(f"    registered {registered}")
        for protocol in record.protocols:
            print(f"    governed by {protocol}")
    print()
    print(
        "Add a numbered supersession that registers the current bytes, or revert the "
        f"source. Use --write-baseline only to record drift you have decided to carry."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
