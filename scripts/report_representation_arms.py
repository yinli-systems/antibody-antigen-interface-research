#!/usr/bin/env python3
"""Compare several representation arms that share their seeds.

    python3 scripts/report_representation_arms.py --run-root /path/to/arms \
        --baseline esm2 --arm random_control --arm antiberty_only

Each named arm is compared to the baseline arm, paired within seed. The metrics
come from the evaluation receipts rather than from the prediction parquets: the
two agree to 1.7e-17 on this data, and reading receipts turns an hour of
recomputation into a second.

The comparison that matters most here is not against the narrow arm. An arm with
extra columns has more parameters in its input projections, so a gain over
``esm2`` cannot separate the content of those columns from their width. The
width-matched control (``random_control``) holds width and parameter count fixed
and carries a fixed random projection of the base embedding, so a comparison
against *it* isolates the content. Both comparisons are reported, and where they
disagree the width-matched one is the honest number.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
if str(REPOSITORY_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

from aggregate_baseline_margin import (  # noqa: E402
    bootstrap_mean_interval,
    clopper_pearson_lower_bound,
)

DEFAULT_METRICS = ("auprc", "auroc", "mcc", "f1", "precision_at_true_contact_count")


def arm_receipts(root: Path, arm: str) -> dict[int, dict]:
    """Return the macro metrics per seed for one arm, from its receipts."""
    receipts: dict[int, dict] = {}
    for directory in sorted((root / arm).glob("seed_*")):
        receipt = directory / "validation" / "evaluation.json"
        if not receipt.is_file():
            continue
        document = json.loads(receipt.read_text(encoding="utf-8"))
        receipts[int(directory.name.split("_")[1])] = document["pair"]["macro"]
    return receipts


def compare(baseline: dict[int, dict], arm: dict[int, dict], metric: str) -> dict | None:
    """Paired comparison of one metric between two arms."""
    shared = sorted(
        seed for seed in set(baseline) & set(arm)
        if metric in baseline[seed] and metric in arm[seed]
    )
    if len(shared) < 2:
        return None
    differences = [arm[seed][metric] - baseline[seed][metric] for seed in shared]
    favourable = sum(1 for value in differences if value > 0.0)
    mean = statistics.mean(differences)
    sd = statistics.stdev(differences) if len(differences) > 1 else 0.0
    lower, upper = bootstrap_mean_interval(differences)
    baseline_mean = statistics.mean(baseline[seed][metric] for seed in shared)
    return {
        "metric": metric,
        "seed_count": len(shared),
        "baseline_mean": baseline_mean,
        "arm_mean": statistics.mean(arm[seed][metric] for seed in shared),
        "mean_difference": mean,
        "relative_difference": (mean / baseline_mean) if baseline_mean else None,
        "sd_difference": sd,
        "effect_size": (mean / sd) if sd > 0 else None,
        "bootstrap_ci95": [lower, upper],
        "favourable_seed_count": favourable,
        "sign_lower_bound": clopper_pearson_lower_bound(favourable, len(shared)),
        "decided": bool(lower > 0.0 or upper < 0.0),
        "per_seed": [
            {"seed": seed, "baseline": baseline[seed][metric], "arm": arm[seed][metric]}
            for seed in shared
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--baseline", default="esm2")
    parser.add_argument("--arm", action="append", default=None,
                        help="repeatable; defaults to every other directory present")
    parser.add_argument("--metric", action="append", default=None)
    parser.add_argument("--json", type=Path, default=None)
    arguments = parser.parse_args(argv)

    metrics = tuple(arguments.metric or DEFAULT_METRICS)
    baseline = arm_receipts(arguments.run_root, arguments.baseline)
    if not baseline:
        print(f"no receipts under {arguments.run_root / arguments.baseline}", file=sys.stderr)
        return 2

    names = arguments.arm or sorted(
        path.name for path in arguments.run_root.iterdir()
        if path.is_dir() and path.name != arguments.baseline
    )
    report = {
        "format_version": "representation_arms_v1",
        "baseline_arm": arguments.baseline,
        "baseline_seed_count": len(baseline),
        "comparisons": {},
        "claim_boundary": (
            "Development-only paired comparisons on the validation split. They "
            "compare input representations and authorize no test access and no "
            "model selection."
        ),
    }

    for name in names:
        arm = arm_receipts(arguments.run_root, name)
        if not arm:
            continue
        entries = [entry for metric in metrics if (entry := compare(baseline, arm, metric))]
        if not entries:
            continue
        report["comparisons"][name] = entries
        headline = entries[0]
        print(f"\n{name}  vs  {arguments.baseline}   ({headline['seed_count']} paired seeds)")
        for entry in entries:
            relative = entry["relative_difference"]
            print(
                f"  {entry['metric']:<32s} {entry['baseline_mean']:.6f} -> {entry['arm_mean']:.6f}"
                f"  {entry['mean_difference']:+.6f}"
                + (f" ({100 * relative:+.1f}%)" if relative is not None else "")
                + f"  CI [{entry['bootstrap_ci95'][0]:+.6f}, {entry['bootstrap_ci95'][1]:+.6f}]"
                f"  {entry['favourable_seed_count']}/{entry['seed_count']}"
                f"  bound {entry['sign_lower_bound']:.4f}"
                + ("" if entry["decided"] else "   NO DECISION")
            )

    if arguments.json:
        arguments.json.parent.mkdir(parents=True, exist_ok=True)
        arguments.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                                  encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
