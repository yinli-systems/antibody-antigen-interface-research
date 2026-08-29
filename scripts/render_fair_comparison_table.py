#!/usr/bin/env python3
"""Render a teacher-facing baseline table without mixing evaluation scopes.

The local receipts intentionally cover different roles: EpiFormer/SPPIDER use
the frozen SAbDab2 test, while MIPE is an official five-fold held-out result.
This renderer keeps those scopes visible and never inserts a FrameContact
development number into the frozen-test table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

# ``python scripts/render_fair_comparison_table.py`` puts ``scripts/`` rather
# than the repository root on ``sys.path``.  Add the root so the renderer and
# the audit module behave identically when invoked as a script or imported by
# pytest.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.audit_baseline_metrics import MIPE_JSON, epiformer_rows, mipe_rows, sppider_rows


METRICS = ("auroc", "auprc", "precision", "recall", "f1", "mcc")
CONTACT_RECEIPTS = REPOSITORY_ROOT / "framecontact" / "receipts" / "live-array-audit-20260801T145157Z"


def _fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, (int, float)):
        return f"{value:.4f}"
    return str(value)


def _line(label: str, row: dict[str, Any] | None) -> str:
    values = [label] + [_fmt(row.get(metric) if row else None) for metric in METRICS]
    return "| " + " | ".join(values) + " |"


def _table(title: str, rows: list[tuple[str, dict[str, Any] | None]]) -> list[str]:
    lines = [
        f"### {title}",
        "",
        "| Model | AUROC | AUPRC | Precision | Recall | F1 | MCC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    lines.extend(_line(label, row) for label, row in rows)
    lines.append("")
    return lines


def _contact_snapshot() -> list[str]:
    """Render native pair metrics from the matched development receipts.

    These are intentionally separate from the external baseline table: they
    are antigen-fold validation trajectories, not a selected frozen-test
    checkpoint.
    """
    snapshots = []
    for filename, condition, epoch in (
        ("residue-head-compact.json", "FrameContact unary-only", 21),
        ("distogram-compact.json", "FrameContact distogram", 18),
    ):
        payload = json.loads((CONTACT_RECEIPTS / filename).read_text(encoding="utf-8"))
        condition_payload = payload["conditions"]
        key = "framecontact_unary_only" if "unary" in condition.lower() else "framecontact_distogram"
        sync = condition_payload[key]["latest_synchronized_epoch"]
        if sync is None or int(sync["epoch"]) != epoch:
            raise ValueError(f"contact receipt epoch mismatch for {condition}")
        metrics = sync["metrics"]

        def mean_sd(name: str) -> str:
            item = metrics[name]
            return f"{item['mean']:.4f} ± {item['sample_sd']:.4f}"

        snapshots.append(
            (
                condition,
                epoch,
                mean_sd("contact_pooled_auprc"),
                mean_sd("contact_mean_per_complex_auprc"),
                mean_sd("contact_fixed_raw_precision"),
                mean_sd("contact_fixed_raw_recall"),
                mean_sd("contact_fixed_raw_f1"),
                mean_sd("contact_fixed_raw_mcc"),
            )
        )

    lines = [
        "## FrameContact contact development snapshot",
        "",
        "> Native residue-pair contact output on the SAbDab2 antigen-fold validation split; 5 seeds at matched synchronized epochs. These values are development evidence only, not frozen-test results.",
        "",
        "| Condition | Epoch | Pooled contact AUPRC | Mean per-complex AUPRC | Precision | Recall | F1 | MCC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in snapshots:
        lines.append("| " + " | ".join(map(str, row)) + " |")
    lines.extend(
        [
            "",
            "The fixed threshold is the raw-logit development diagnostic; no test checkpoint or threshold was selected from these rows.",
            "",
        ]
    )
    return lines


def render() -> str:
    mipe_payload = json.loads(MIPE_JSON.read_text(encoding="utf-8"))
    epiformer = epiformer_rows()[0]
    sppider = sppider_rows()
    mipe = mipe_rows(mipe_payload)
    mipe_best = {
        row["target"]: row
        for row in mipe
        if row["role"] == "secondary_validation_best"
    }
    sppider_by_target = {row["target"]: row for row in sppider}

    lines = [
        "# Fair baseline tables (receipt-generated)",
        "",
        "> Generated from the frozen receipts by `scripts/render_fair_comparison_table.py`.",
        "> Do not remove the scope labels: these rows are not one common leaderboard.",
        "",
        "## Scope ledger",
        "",
        "| Model | Target | Scope | Test opened for selection? | Native pair output? |",
        "|---|---|---|---:|---:|",
        f"| EpiFormer | epitope | {epiformer['scope']} | no | no |",
        f"| MIPE (validation-best) | epitope/paratope | {mipe_best['epitope']['scope']} | no | no |",
        f"| SPPIDER-seq | epitope/paratope | {sppider_by_target['epitope']['scope']} | no | no |",
        "| FrameContact | epitope/paratope/contact | antigen-fold development validation | no | contact only in the pair head |",
        "",
    ]
    lines.extend(
        _table(
            "Epitope (receipt-backed baseline rows)",
            [
                ("EpiFormer", epiformer),
                ("MIPE (validation-best)", mipe_best["epitope"]),
                ("SPPIDER-seq", sppider_by_target["epitope"]),
                ("FrameContact (frozen test not opened)", None),
            ],
        )
    )
    lines.extend(_contact_snapshot())
    lines.extend(
        _table(
            "Paratope (receipt-backed baseline rows)",
            [
                ("EpiFormer (no paratope head)", None),
                ("MIPE (validation-best)", mipe_best["paratope"]),
                ("SPPIDER-seq", sppider_by_target["paratope"]),
                ("FrameContact (frozen test not opened)", None),
            ],
        )
    )
    lines.extend(
        _table(
            "Contact (native pair score only)",
            [
                ("EpiFormer (residue projection only)", None),
                ("MIPE (residue marginal only)", None),
                ("SPPIDER-seq (residue scores only)", None),
                ("FrameContact (development-only; not ranked here)", None),
            ],
        )
    )
    lines.extend(
        [
            "## Required interpretation",
            "",
            "- Use the epitope/paratope rows only as receipt-backed context; MIPE is five-fold held-out, while EpiFormer and SPPIDER are frozen SAbDab2 test.",
            "- Keep FrameContact out of this frozen-test table until the validation-only winner, threshold and test-opening receipt exist.",
            "- Never convert a residue score into a contact score. Contact AUPRC/precision/recall/F1/MCC require a native residue-pair output and the same candidate mask.",
            "- The final fair comparison still requires one common AsEP split, five seeds, validation-only selection, calibration, partner shuffle/retrieval and the pre-registered external opening.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rendered = render()
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
