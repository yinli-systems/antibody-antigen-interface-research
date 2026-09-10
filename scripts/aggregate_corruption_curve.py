#!/usr/bin/env python3
"""Read a correspondence-corruption ladder into a shape, per protocol.

Registered in ``protocols/CORRESPONDENCE_CORRUPTION_CURVE_V1.md`` and its
2026-09-11 amendment, which fix the endpoint, the pairing and the rule that
turns the curve into one of three named outcomes.

    python3 scripts/aggregate_corruption_curve.py \
        --aligned-root /path/to/esm_if1_only \
        --corrupt-root /path/to/corrupt/esm_if1

Each ladder point is paired to ``k = 0`` on the *same seed and checkpoint*: the
perturbation is applied at evaluation to an already-trained model, so the
difference is within-model and seed variance cancels rather than being averaged
over. That is why six seeds can say something here that six seeds could not say
about a training-time comparison.

The classification is mechanical. It reports ``intermediate`` when the curve
falls between the registered boundaries instead of moving one.
"""

from __future__ import annotations

import argparse
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

# The far end of the ladder is stored as k999 because it is a global permutation,
# not a displacement of 999 positions.
GLOBAL_KEY = 999
LADDER = (1, 2, 4, 8, 16, GLOBAL_KEY)
ENDPOINT = "auprc"
COMPANIONS = ("auroc", "mcc", "f1", "precision_at_true_contact_count")
CLIFF_FRACTION = 0.8
SLOPE_FRACTION = 0.5


def read_metric(evaluation: Path, metric: str) -> float | None:
    """Pull one macro pair metric out of an evaluation receipt."""
    try:
        payload = json.loads(evaluation.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    for holder in (payload.get("macro"), payload.get("pair", {}).get("macro"), payload):
        if isinstance(holder, dict) and isinstance(holder.get(metric), (int, float)):
            return float(holder[metric])
    return None


def collect(aligned_root: Path, corrupt_root: Path, metric: str,
            aligned_subdir: str = "validation") -> dict:
    """Per-k, per-seed paired differences against the unperturbed model.

    ``aligned_subdir`` is a path concern only. Training runs write their
    evaluation under ``seed_N/validation/``; a ladder that evaluates ``k = 0``
    as its own arm writes ``seed_N/`` directly. Nothing about the comparison
    changes with it.
    """
    baseline: dict[int, float] = {}
    for directory in sorted(aligned_root.glob("seed_*")):
        receipt = directory / aligned_subdir / "evaluation.json" if aligned_subdir \
            else directory / "evaluation.json"
        value = read_metric(receipt, metric)
        if value is not None:
            baseline[int(directory.name.split("_")[1])] = value

    points: dict[int, list[dict]] = {}
    for k in LADDER:
        rows = []
        for directory in sorted((corrupt_root / f"k{k}").glob("seed_*")):
            seed = int(directory.name.split("_")[1])
            value = read_metric(directory / "evaluation.json", metric)
            if value is None or seed not in baseline:
                continue
            rows.append({"seed": seed, "aligned": baseline[seed], "corrupted": value,
                         "difference": value - baseline[seed]})
        if rows:
            points[k] = sorted(rows, key=lambda row: row["seed"])
    return {"baseline_seeds": sorted(baseline), "points": points}


def summarise(rows: list[dict]) -> dict:
    differences = [row["difference"] for row in rows]
    count = len(differences)
    mean = sum(differences) / count
    negative = sum(1 for value in differences if value < 0.0)
    summary = {
        "seed_count": count,
        "seeds": [row["seed"] for row in rows],
        "mean_difference": mean,
        "negative_seed_count": negative,
        "damaged": False,
        "sign_lower_bound": None,
        "bootstrap_ci95": None,
    }
    if count >= 2:
        summary["bootstrap_ci95"] = list(bootstrap_mean_interval(differences))
        bound = clopper_pearson_lower_bound(negative, count)
        summary["sign_lower_bound"] = bound
        summary["damaged"] = bound > 0.5
    return summary


def adjacent_interval(points: dict[int, list[dict]], earlier: int, later: int):
    """Bootstrap interval of the step between two ladder points, paired by seed.

    The protocol calls a monotonicity violation noise when *the violation's*
    interval covers zero. The violation is the step from one point to the next,
    so this is the interval of that step -- not the interval of the later point
    against zero, which is a different question and one that is almost always
    answered "damaged" on a ladder where every point is damaged.
    """
    if earlier not in points or later not in points:
        return None
    before = {row["seed"]: row["difference"] for row in points[earlier]}
    after = {row["seed"]: row["difference"] for row in points[later]}
    shared = sorted(set(before) & set(after))
    if len(shared) < 2:
        return None
    return list(bootstrap_mean_interval([after[seed] - before[seed] for seed in shared]))


def classify(summaries: dict[int, dict], points: dict[int, list[dict]] | None = None) -> dict:
    """Apply the registered rule. Returns the verdict and why it fired."""
    if GLOBAL_KEY not in summaries or 1 not in summaries:
        return {"verdict": "not classifiable",
                "reason": "the ladder is missing k=1 or the global permutation"}

    far = abs(summaries[GLOBAL_KEY]["mean_difference"])
    near = abs(summaries[1]["mean_difference"])
    if far <= 0.0:
        return {"verdict": "not classifiable",
                "reason": "the global permutation did no damage, so there is no scale to compare to"}

    local = [k for k in (1, 2, 4, 8, 16) if k in summaries]
    violations = []
    for earlier, later in zip(local, local[1:]):
        if summaries[later]["mean_difference"] > summaries[earlier]["mean_difference"]:
            interval = adjacent_interval(points or {}, earlier, later)
            violations.append({
                "from_k": earlier, "to_k": later,
                "step_ci95": interval,
                "covers_zero": bool(interval and interval[0] <= 0.0 <= interval[1]),
            })
    # A violation that is indistinguishable from no step at all is noted and
    # does not break the classification, which is what the protocol registers.
    monotone = all(violation["covers_zero"] for violation in violations)

    if not summaries[1]["damaged"]:
        verdict, reason = "coarse-only", "k=1 is not established as damaging"
    elif near >= CLIFF_FRACTION * far:
        verdict, reason = "cliff", f"k=1 costs {near / far:.0%} of the global permutation"
    elif near < SLOPE_FRACTION * far and monotone:
        verdict, reason = "slope", f"k=1 costs {near / far:.0%} of it and the ladder is non-increasing"
    elif near < SLOPE_FRACTION * far:
        verdict, reason = "intermediate", "below the slope boundary but not non-increasing"
    else:
        verdict, reason = "intermediate", (
            f"k=1 costs {near / far:.0%}, between the {SLOPE_FRACTION:.0%} and "
            f"{CLIFF_FRACTION:.0%} boundaries"
        )
    return {"verdict": verdict, "reason": reason, "k1_fraction_of_global": near / far,
            "monotone": monotone, "monotonicity_violations": violations}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--aligned-root", type=Path, required=True,
                        help="directory of seed_*/validation for the unperturbed model")
    parser.add_argument("--corrupt-root", type=Path, required=True,
                        help="directory of k*/seed_* evaluations")
    parser.add_argument("--label", default="")
    parser.add_argument("--aligned-subdir", default="validation",
                        help="subdirectory of each aligned seed holding evaluation.json; "
                             "empty when the baseline is a ladder arm of its own")
    parser.add_argument("--json", type=Path, default=None)
    arguments = parser.parse_args(argv)

    collected = collect(arguments.aligned_root, arguments.corrupt_root, ENDPOINT,
                        arguments.aligned_subdir)
    points = collected["points"]
    if not points:
        print(f"no ladder points under {arguments.corrupt_root}", file=sys.stderr)
        return 2

    summaries = {k: summarise(rows) for k, rows in points.items()}
    verdict = classify(summaries, points)
    companions = {
        name: {k: (lambda rows: sum(r["difference"] for r in rows) / len(rows))(rows)
               for k, rows in collect(arguments.aligned_root, arguments.corrupt_root,
                                      name, arguments.aligned_subdir)["points"].items()}
        for name in COMPANIONS
    }

    label = arguments.label or arguments.corrupt_root.name
    print(f"ladder      : {label}, macro pair {ENDPOINT}, paired to k=0 per seed")
    print(f"missing     : {sorted(set(LADDER) - set(points))}" if set(LADDER) - set(points)
          else "missing     : none, the ladder is complete")
    print()
    header = f"{'k':>7}  {'seeds':>5}  {'mean d':>10}  {'CI95':>22}  {'neg':>5}  {'bound':>6}  damaged"
    print(header)
    print("-" * len(header))
    for k in LADDER:
        if k not in summaries:
            continue
        s = summaries[k]
        name = "global" if k == GLOBAL_KEY else str(k)
        interval = (f"[{s['bootstrap_ci95'][0]:+.5f}, {s['bootstrap_ci95'][1]:+.5f}]"
                    if s["bootstrap_ci95"] else "n/a")
        bound = f"{s['sign_lower_bound']:.4f}" if s["sign_lower_bound"] is not None else "n/a"
        print(f"{name:>7}  {s['seed_count']:>5}  {s['mean_difference']:>+10.5f}  {interval:>22}  "
              f"{s['negative_seed_count']:>5}  {bound:>6}  {s['damaged']}")
    print()
    print(f"VERDICT     : {verdict['verdict']}")
    print(f"because     : {verdict['reason']}")
    for violation in verdict.get("monotonicity_violations", []):
        note = "within noise" if violation["covers_zero"] else "outside its interval"
        print(f"  non-monotone k={violation['from_k']} -> k={violation['to_k']} ({note})")
    print()
    print("companion metrics (mean paired difference):")
    for name, values in companions.items():
        if not values:
            continue
        rendered = "  ".join(
            f"k={'global' if k == GLOBAL_KEY else k}:{values[k]:+.4f}"
            for k in LADDER if k in values
        )
        print(f"  {name:>32}  {rendered}")

    if arguments.json:
        arguments.json.parent.mkdir(parents=True, exist_ok=True)
        arguments.json.write_text(json.dumps({
            "format_version": "correspondence_corruption_curve_v1",
            "label": label,
            "endpoint": ENDPOINT,
            "aligned_seeds": collected["baseline_seeds"],
            "per_k": {str(k): summaries[k] for k in summaries},
            "per_seed": {str(k): points[k] for k in points},
            "companion_metrics": {name: {str(k): v for k, v in values.items()}
                                  for name, values in companions.items()},
            "classification": verdict,
            "claim_boundary": (
                "Validation splits only, evaluation-time perturbation of already-trained "
                "models. No test split is read. This measures how much correspondence "
                "fidelity is worth, not how well the model performs."
            ),
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
