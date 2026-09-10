#!/usr/bin/env python3
"""Where, if anywhere, does the node/edge gradient cosine change sign?

Registered in the 2026-09-11 amendment to
``protocols/GRADIENT_ALIGNMENT_DYNAMICS_V1.md``, which fixes the three outcomes
before any trace exists.

    python3 scripts/aggregate_step_trace.py --cohort asep=/path/steptrace/asep_seed* \
        --json steps.json

An epoch mean cannot resolve a sign change inside an epoch. A 40-instance smoke
reported -0.059 over its whole epoch; a full 1383-instance epoch of the same
configuration reported +0.468. Both are consistent with the cosine being
negative for the first few dozen steps, and neither can show it. This reads the
individual steps.

The crossing is defined so that a single noisy step cannot declare one: the
cosine must be negative on average over a leading window and positive on average
over a trailing window, and the crossing step is the first index after which it
never again returns to a negative running mean.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

COSINE_KEY = "cosine_node_edge"
# A step is not a measurement; a run of steps is. Both windows are in steps.
SMOOTHING = 10
# How many leading steps count as "the beginning" when asking whether training
# starts in conflict. Chosen to bracket the 40-step smoke that raised the
# question, and fixed before any trace was read.
OPENING = 40
# A shorter probe. If the negative phase is briefer than OPENING, the 40-step
# mean can already be positive and the run would be recorded as never having
# started in conflict -- the exact transient this is looking for, missed by the
# window meant to find it. A run counts as opening negative if either window
# says so, which can only make the instrument more sensitive, never less.
SHORT_OPENING = 10
# A crossing this late in the epoch is not a transient of early optimisation,
# whatever else it is. Without this, a run that returns to conflict for a long
# stretch and recovers near the end would be classified alongside one that
# settles in the first few dozen steps.
EARLY_FRACTION = 0.25


def read_trace(path: Path) -> list[float] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    steps = payload.get("steps")
    if not isinstance(steps, list) or not steps:
        return None
    series = [entry.get(COSINE_KEY) for entry in steps]
    if any(not isinstance(value, (int, float)) for value in series):
        return None
    return [float(value) for value in series]


def smooth(series: list[float], window: int = SMOOTHING) -> list[float]:
    """Trailing mean, so index i is the average of the steps up to i."""
    out, total = [], 0.0
    for index, value in enumerate(series):
        total += value
        if index >= window:
            total -= series[index - window]
        out.append(total / min(index + 1, window))
    return out


def analyse(series: list[float]) -> dict:
    smoothed = smooth(series)
    opening = series[:OPENING]
    opening_mean = sum(opening) / len(opening)
    short = series[:SHORT_OPENING]
    short_mean = sum(short) / len(short)
    negative_fraction = sum(1 for value in series if value < 0.0) / len(series)

    # The first index after which the smoothed cosine is never negative again.
    crossing = None
    for index in range(len(smoothed)):
        if smoothed[index] >= 0.0 and all(value >= 0.0 for value in smoothed[index:]):
            crossing = index
            break
    # A crossing only counts if there was something to cross from.
    starts_negative = opening_mean < 0.0 or short_mean < 0.0
    return {
        "steps": len(series),
        "opening_mean": opening_mean,
        "opening_window": OPENING,
        "short_opening_mean": short_mean,
        "short_opening_window": SHORT_OPENING,
        "final_mean": sum(series[-OPENING:]) / len(series[-OPENING:]),
        "negative_step_fraction": negative_fraction,
        "starts_negative": starts_negative,
        "crossing_step": crossing if starts_negative else None,
        "crossed": bool(starts_negative and crossing is not None),
        "first_steps": [round(value, 4) for value in series[:20]],
    }


def classify(per_seed: dict[str, dict]) -> dict:
    """The registered outcomes, applied to one cohort."""
    seeds = sorted(per_seed)
    starting = sum(1 for seed in seeds if per_seed[seed]["starts_negative"])
    crossed = sum(1 for seed in seeds if per_seed[seed]["crossed"])
    fractions = [per_seed[seed]["negative_step_fraction"] for seed in seeds]
    mean_negative = sum(fractions) / len(fractions)

    crossing_steps = [per_seed[seed]["crossing_step"] for seed in seeds
                      if per_seed[seed]["crossed"]]
    early = [seed for seed in seeds if per_seed[seed]["crossed"]
             and per_seed[seed]["crossing_step"]
             <= EARLY_FRACTION * per_seed[seed]["steps"]]

    if crossed == len(seeds) and len(early) == len(seeds) and len(seeds) >= 2:
        verdict = "transient"
        reason = ("every seed opens with a negative mean cosine and crosses to a "
                  f"durably positive one within the first {EARLY_FRACTION:.0%} of "
                  "the epoch")
    elif crossed == len(seeds) and len(seeds) >= 2:
        verdict = "late crossing"
        reason = ("every seed crosses, but not early enough to be a transient of "
                  f"initialisation; crossings at steps {sorted(crossing_steps)} of "
                  f"{[per_seed[s]['steps'] for s in seeds]}")
    elif starting == 0:
        verdict = "positive throughout"
        reason = ("no seed opens negative, so the -0.059 came from the diagnostic "
                  "configuration and never described training")
    elif crossed == 0 and mean_negative > 0.2:
        verdict = "oscillating"
        reason = (f"seeds open negative but no durable crossing exists; "
                  f"{mean_negative:.0%} of steps are negative")
    else:
        verdict = "mixed"
        reason = (f"{starting}/{len(seeds)} seeds open negative and "
                  f"{crossed}/{len(seeds)} cross durably")
    return {"verdict": verdict, "reason": reason,
            "seeds_opening_negative": starting, "seeds_crossing": crossed,
            "seeds_crossing_early": len(early), "crossing_steps": sorted(crossing_steps),
            "early_fraction": EARLY_FRACTION,
            "seed_count": len(seeds), "mean_negative_step_fraction": mean_negative}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cohort", action="append", required=True, metavar="NAME=GLOB")
    parser.add_argument("--json", type=Path, default=None)
    arguments = parser.parse_args(argv)

    cohorts, missing = {}, []
    for specification in arguments.cohort:
        name, _, pattern = specification.partition("=")
        if not pattern:
            print(f"--cohort wants NAME=GLOB, got {specification!r}", file=sys.stderr)
            return 2
        per_seed = {}
        for directory in sorted(glob.glob(pattern)):
            series = read_trace(Path(directory) / "balancer_trace.json")
            if series is None:
                missing.append(directory)
                continue
            per_seed[Path(directory).name] = analyse(series)
        if per_seed:
            cohorts[name] = {"per_seed": per_seed, "classification": classify(per_seed)}
        else:
            missing.append(pattern)

    if not cohorts:
        print(f"no traces carrying {COSINE_KEY}", file=sys.stderr)
        for entry in missing:
            print(f"  missing: {entry}", file=sys.stderr)
        return 2

    for name, cohort in cohorts.items():
        verdict = cohort["classification"]
        print(f"### {name}   {verdict['seed_count']} seeds")
        header = (f"{'seed':>16}  {'steps':>6}  {'open10':>8}  {'open40':>8}  "
                  f"{'final40':>8}  {'neg%':>6}  {'cross@':>7}")
        print(header)
        print("-" * len(header))
        for seed, entry in sorted(cohort["per_seed"].items()):
            crossing = ("-" if entry["crossing_step"] is None
                        else str(entry["crossing_step"]))
            print(f"{seed:>16}  {entry['steps']:>6}  "
                  f"{entry['short_opening_mean']:>+8.4f}  "
                  f"{entry['opening_mean']:>+8.4f}  {entry['final_mean']:>+8.4f}  "
                  f"{100 * entry['negative_step_fraction']:>5.1f}%  {crossing:>7}")
        print(f"  VERDICT: {verdict['verdict']}")
        print(f"  because: {verdict['reason']}")
        first = next(iter(sorted(cohort["per_seed"])))
        print(f"  first 20 steps of {first}: {cohort['per_seed'][first]['first_steps']}")
        print()

    if arguments.json:
        arguments.json.parent.mkdir(parents=True, exist_ok=True)
        arguments.json.write_text(json.dumps({
            "format_version": "gradient_step_trace_v1",
            "quantity": COSINE_KEY,
            "smoothing_window": SMOOTHING,
            "opening_window": OPENING,
            "short_opening_window": SHORT_OPENING,
            "cohorts": cohorts,
            "missing": missing,
            "claim_boundary": (
                "Training-time gradient geometry on development cohorts. It locates a "
                "sign change in optimisation steps; it does not show that anything "
                "downstream depends on it."
            ),
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
