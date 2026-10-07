#!/usr/bin/env python3
"""Does gradient alignment change over training, and does it differ by cohort?

Registered in ``protocols/GRADIENT_ALIGNMENT_DYNAMICS_V1.md``, which fixes the
transition criterion before any trajectory is read.

    python3 scripts/aggregate_gradient_trajectory.py \
        --cohort sabdab2=/path/measure_traj/sabdab2* \
        --cohort asep=/path/measure_traj/asep* \
        --json dynamics.json

The quantity is the epoch-wise mean cosine between the node gradient and the
contact-edge gradient on the shared trunk, which the ``measure`` balancer writes
into the training history. A cohort shows a sign transition only when it starts
decidedly negative, ends decidedly positive, and rises by an interval that
excludes zero. A cohort that rises without crossing is reported as rising, not
rounded up to a transition.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

from aggregate_baseline_margin import (  # noqa: E402
    bootstrap_mean_interval,
    clopper_pearson_lower_bound,
)

COSINE_KEY = "balancer_cosine_node_edge"
# The value gradient_conflict_v1 reports at a converged sabdab2 checkpoint, and
# the number the final epoch of the sabdab2 trajectories is compared against.
CHECKPOINT_COSINE = 0.70


def thirds(count: int) -> tuple[int, int]:
    """Sizes of the leading and trailing windows.

    At least one epoch each, and they never overlap, so a short run still yields
    a comparison rather than reading the same epochs as both early and late.
    """
    window = max(1, count // 3)
    if 2 * window > count:
        window = count // 2
    return max(1, window), max(1, window)


def read_trajectory(history_path: Path) -> list[float] | None:
    try:
        payload = json.loads(history_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    epochs = payload.get("history")
    if not isinstance(epochs, list) or not epochs:
        return None
    series = []
    for entry in epochs:
        value = entry.get("train", {}).get(COSINE_KEY)
        if not isinstance(value, (int, float)):
            return None
        series.append(float(value))
    return series if len(series) >= 2 else None


def summarise(series_by_seed: dict[str, list[float]]) -> dict:
    seeds = sorted(series_by_seed)
    early, late, delta, finals = [], [], [], []
    for seed in seeds:
        series = series_by_seed[seed]
        head, tail = thirds(len(series))
        first = sum(series[:head]) / head
        last = sum(series[-tail:]) / tail
        early.append(first)
        late.append(last)
        delta.append(last - first)
        finals.append(series[-1])

    count = len(seeds)
    negative_early = sum(1 for value in early if value < 0.0)
    positive_late = sum(1 for value in late if value > 0.0)
    rising = sum(1 for value in delta if value > 0.0)
    summary = {
        "seeds": seeds,
        "seed_count": count,
        "epochs": {seed: len(series_by_seed[seed]) for seed in seeds},
        "mean_early": sum(early) / count,
        "mean_late": sum(late) / count,
        "mean_delta": sum(delta) / count,
        "mean_final_epoch": sum(finals) / count,
        "early_negative_seeds": negative_early,
        "late_positive_seeds": positive_late,
        "rising_seeds": rising,
        "per_seed": {seed: {"early": early[i], "late": late[i], "delta": delta[i],
                            "final": finals[i], "series": series_by_seed[seed]}
                     for i, seed in enumerate(seeds)},
    }
    if count >= 2:
        summary["early_ci95"] = list(bootstrap_mean_interval(early))
        summary["late_ci95"] = list(bootstrap_mean_interval(late))
        summary["delta_ci95"] = list(bootstrap_mean_interval(delta))
        summary["early_negative_bound"] = clopper_pearson_lower_bound(negative_early, count)
        summary["late_positive_bound"] = clopper_pearson_lower_bound(positive_late, count)

    starts_negative = summary.get("early_negative_bound", 0.0) > 0.5
    ends_positive = summary.get("late_positive_bound", 0.0) > 0.5
    interval = summary.get("delta_ci95")
    rose = bool(interval and interval[0] > 0.0)
    summary["starts_negative"] = starts_negative
    summary["ends_positive"] = ends_positive
    summary["rose"] = rose
    summary["transition"] = bool(starts_negative and ends_positive and rose)
    summary["rise_without_transition"] = bool(rose and not summary["transition"])
    return summary


def classify(cohorts: dict[str, dict]) -> dict:
    """The registered outcomes, applied mechanically."""
    if "sabdab2" not in cohorts or "asep" not in cohorts:
        return {"verdict": "not classifiable",
                "reason": "both cohorts are required to separate stage from cohort"}
    sabdab2, asep = cohorts["sabdab2"], cohorts["asep"]
    if sabdab2["transition"] and asep["transition"]:
        verdict = "stage"
        reason = ("both cohorts start negative and end positive, so the disagreement "
                  "is the stage of training, not the data")
    elif sabdab2["transition"] and not asep["transition"]:
        verdict = "cohort"
        reason = ("sabdab2 transitions and AsEP does not, so the geometry depends on "
                  "the data distribution")
    elif not sabdab2["transition"] and not asep["transition"]:
        verdict = "neither"
        reason = ("no cohort transitions, so neither stage nor cohort reconciles the "
                  "two measurements and the instruments disagree")
    else:
        verdict = "stage and cohort"
        reason = ("AsEP transitions and sabdab2 does not, which is neither registered "
                  "direction and is reported as it stands")
    return {"verdict": verdict, "reason": reason,
            "sabdab2_final_vs_checkpoint": sabdab2["mean_final_epoch"] - CHECKPOINT_COSINE}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cohort", action="append", required=True, metavar="NAME=GLOB",
                        help="cohort name and a glob of run directories")
    parser.add_argument("--json", type=Path, default=None)
    arguments = parser.parse_args(argv)

    cohorts, missing = {}, []
    for specification in arguments.cohort:
        name, _, pattern = specification.partition("=")
        if not pattern:
            print(f"--cohort wants NAME=GLOB, got {specification!r}", file=sys.stderr)
            return 2
        series_by_seed = {}
        for directory in sorted(glob.glob(pattern)):
            series = read_trajectory(Path(directory) / "history.json")
            if series is None:
                missing.append(directory)
                continue
            series_by_seed[Path(directory).name] = series
        if series_by_seed:
            cohorts[name] = summarise(series_by_seed)
        else:
            missing.append(pattern)

    if not cohorts:
        print("no trajectories carrying " + COSINE_KEY, file=sys.stderr)
        for entry in missing:
            print(f"  missing: {entry}", file=sys.stderr)
        return 2

    for name, summary in cohorts.items():
        print(f"### {name}   {summary['seed_count']} seeds   "
              f"epochs {sorted(set(summary['epochs'].values()))}")
        for key in ("early", "late", "delta"):
            interval = summary.get(f"{key}_ci95")
            rendered = (f"[{interval[0]:+.4f}, {interval[1]:+.4f}]" if interval else "n/a")
            print(f"  mean {key:<6} {summary['mean_' + key]:+.4f}   CI {rendered}")
        print(f"  final epoch  {summary['mean_final_epoch']:+.4f}")
        print(f"  starts negative {summary['starts_negative']}"
              f" ({summary['early_negative_seeds']}/{summary['seed_count']})   "
              f"ends positive {summary['ends_positive']}"
              f" ({summary['late_positive_seeds']}/{summary['seed_count']})   "
              f"rose {summary['rose']}")
        print(f"  TRANSITION: {summary['transition']}"
              + ("   (rise without transition)" if summary["rise_without_transition"] else ""))
        for seed in summary["seeds"]:
            series = summary["per_seed"][seed]["series"]
            print(f"    {seed:<16} " + " ".join(f"{value:+.3f}" for value in series))
        print()

    verdict = classify(cohorts)
    print(f"VERDICT : {verdict['verdict']}")
    print(f"because : {verdict['reason']}")
    if "sabdab2_final_vs_checkpoint" in verdict:
        print(f"sabdab2 final epoch minus the {CHECKPOINT_COSINE:+.2f} measured at a "
              f"converged checkpoint: {verdict['sabdab2_final_vs_checkpoint']:+.4f}")
    if missing:
        print()
        for entry in missing:
            print(f"  no usable trajectory: {entry}")

    if arguments.json:
        arguments.json.parent.mkdir(parents=True, exist_ok=True)
        arguments.json.write_text(json.dumps({
            "format_version": "gradient_alignment_dynamics_v1",
            "quantity": COSINE_KEY,
            "checkpoint_reference": CHECKPOINT_COSINE,
            "cohorts": cohorts,
            "classification": verdict,
            "missing": missing,
            "claim_boundary": (
                "Development cohorts only, no test split. The cosine describes the "
                "optimisation geometry; it does not show that anything downstream "
                "follows from it, which is what the ablation is for."
            ),
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
