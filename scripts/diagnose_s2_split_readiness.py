#!/usr/bin/env python3
"""Report what still blocks the provisional two-axis S2 fold assignment.

The assignment is an optimal, zero-leakage, three-fold dual-axis split of the
authority-train pairs that has stayed unapproved. This reports its internal
consistency, the per-fold training floor its retention can support, and -- when
the split file is supplied -- whether that file still carries the membership the
assignment binds.

    python3 scripts/diagnose_s2_split_readiness.py
    python3 scripts/diagnose_s2_split_readiness.py --required-floor 1098
    python3 scripts/diagnose_s2_split_readiness.py --authority-split /path/to/double_remote_split.json

The split file is deliberately not in Git. Without it the membership section is
skipped and everything else still runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from interaction_first.s2_split_readiness import (  # noqa: E402
    S2ReadinessError,
    authority_membership_report,
    consistency_report,
    retention_frontier,
)

FORMAT_VERSION = "s2_split_readiness_diagnostic_v1"
ASSIGNMENT = (
    REPOSITORY_ROOT / "protocols" / "train_only_s2_feasibility" / "s2_assignment_provisional_v1.json"
)


def build_report(required_floor: int | None, authority_split: Path | None) -> dict:
    assignment = json.loads(ASSIGNMENT.read_text(encoding="utf-8"))
    consistency = consistency_report(assignment)
    frontier = retention_frontier(
        consistency["fold_pair_counts"], required_floor=required_floor
    )

    report = {
        "format_version": FORMAT_VERSION,
        "assignment": ASSIGNMENT.relative_to(REPOSITORY_ROOT).as_posix(),
        "assignment_sha256": hashlib.sha256(ASSIGNMENT.read_bytes()).hexdigest(),
        "approval_status": assignment.get("approval_status"),
        "required_downstream_gate": assignment.get("required_downstream_gate", {}).get("gate_name"),
        "gate_passed": assignment.get("required_downstream_gate", {}).get("current_gate_passed"),
        "consistency": consistency,
        "retention_frontier": frontier.as_json(),
        "claim_boundary": (
            "Diagnostic over one committed provisional assignment. It approves nothing, "
            "supersedes nothing, and does not grant the exact four-view Needleman-Wunsch "
            "audit the assignment still requires."
        ),
    }

    if authority_split is not None:
        document = json.loads(authority_split.read_text(encoding="utf-8"))
        splits = document.get("splits")
        if not isinstance(splits, dict):
            raise S2ReadinessError("authority split file has no splits object")
        membership = authority_membership_report(assignment, splits)
        membership["file"] = str(authority_split)
        membership["file_sha256"] = hashlib.sha256(authority_split.read_bytes()).hexdigest()
        recorded = assignment.get("inputs", {}).get("authority_split", {}).get("sha256")
        membership["recorded_file_sha256"] = recorded
        membership["file_sha256_matches"] = membership["file_sha256"] == recorded
        report["authority_membership"] = membership

    return report


def _render(report: dict) -> str:
    lines: list[str] = []
    add = lines.append
    consistency = report["consistency"]
    frontier = report["retention_frontier"]

    add(f"assignment : {report['assignment']}")
    add(f"status     : {report['approval_status']}")
    add(f"blocked by : {report['required_downstream_gate']} (passed={report['gate_passed']})")
    add("")

    add("Internal consistency")
    for name, passed in consistency["checks"].items():
        add(f"  [{'ok' if passed else 'FAIL'}] {name}")
    add("")

    add("Retention")
    add(f"  retained {consistency['retained_pairs']} of {consistency['total_pairs']} pairs "
        f"({consistency['retained_pair_fraction']:.4f}), discarded {consistency['discarded_pairs']}")
    add(f"  fold sizes                      {frontier['fold_sizes']}")
    add(f"  achieved per-fold training floor {frontier['achieved_training_floor']}")
    add(f"  best possible at this retention  {frontier['best_possible_training_floor']}"
        "   (perfectly balanced folds)")
    if frontier["required_floor"] is not None:
        add(f"  required floor                   {frontier['required_floor']}")
        add(f"  retention needed for it          {frontier['retention_required_for_floor']}")
        if frontier["floor_is_reachable"]:
            add("  -> reachable by rebalancing the folds at this retention")
        else:
            add(f"  -> NOT reachable at any balance: short by "
                f"{frontier['retention_shortfall']} retained pairs")
    add("")

    membership = report.get("authority_membership")
    if membership:
        add("Authority split identity")
        add(f"  file digest matches registration : {membership['file_sha256_matches']}")
        add(f"  train membership matches         : {membership['authorized_membership_matches']}")
        add(f"  sealed pool membership matches   : {membership['sealed_membership_matches']}")
        add(f"  sealed partition matches         : {membership['sealed_partition_matches']}"
            f"   recorded={membership['recorded_sealed_counts']} "
            f"observed={membership['observed_sealed_counts']}")
        add(f"  train n sealed overlap           : {membership['train_sealed_overlap']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--required-floor",
        type=int,
        default=None,
        help="per-fold outer training pair floor to test the retention against",
    )
    parser.add_argument(
        "--authority-split",
        type=Path,
        default=None,
        help="path to double_remote_split.json, which is kept outside Git",
    )
    arguments = parser.parse_args(argv)

    try:
        report = build_report(arguments.required_floor, arguments.authority_split)
    except (S2ReadinessError, OSError, json.JSONDecodeError) as error:
        print(f"diagnostic could not run: {error}", file=sys.stderr)
        return 2

    print(json.dumps(report, indent=2, sort_keys=True) if arguments.json else _render(report))
    return 0 if report["consistency"]["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
