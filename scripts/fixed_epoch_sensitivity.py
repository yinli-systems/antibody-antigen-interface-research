#!/usr/bin/env python3
"""The same contrast at a common epoch, since every arm ran all twelve.

The arms select different epochs and follow different learning-rate schedules,
because the quantity they monitor -- named `total` in all four -- is composed of
whichever loss terms each arm enables. Comparing them at their own selected
epochs therefore measures the loss term together with the schedule and selection
its removal induces.

Every arm ran the full budget, so the same contrast can be taken at a fixed
epoch index. That holds the selection point constant. It does NOT hold the
learning-rate trajectory constant, because the schedule already diverged on the
way there, so this is a sensitivity check and not a clean separation.
"""
import json, sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, "/ssd/scxi253/repr_ablation/source/src")
sys.path.insert(0, "/ssd/scxi253/repr_ablation/source/scripts")
from aggregate_baseline_margin import bootstrap_mean_interval, clopper_pearson_lower_bound

ROOT = Path("/ssd/scxi253/asep_bench/localise")
KEY = "node_epitope_mcc"

def series(arm):
    out = {}
    for d in sorted(ROOT.glob(f"{arm}/seed_*"), key=lambda p: int(p.name.split("_")[1])):
        h = json.load(open(d / "train" / "history.json"))["history"]
        vals = [e["validation"].get(KEY) for e in h]
        if all(isinstance(v, (int, float)) for v in vals):
            out[int(d.name.split("_")[1])] = vals
    return out

arms = {a: series(a) for a in ("node_only", "node_edge", "node_rank", "node_edge_rank")}
if not arms["node_edge"]:
    print(f"the histories do not carry {KEY}; this run predates per-epoch node metrics")
    raise SystemExit(1)

print(f"per-epoch {KEY}, mean over 10 seeds")
print(f"{'epoch':>6}  " + "  ".join(f"{a:>15}" for a in arms))
for ep in range(12):
    row = []
    for a in arms:
        vals = [v[ep] for v in arms[a].values() if len(v) > ep]
        row.append(f"{mean(vals):>15.4f}" if vals else f"{'-':>15}")
    print(f"{ep+1:>6}  " + "  ".join(row))

print()
print("node_edge minus node_edge_rank at each fixed epoch, paired by seed")
print(f"{'epoch':>6}  {'mean':>10}  {'CI95':>24}  {'up':>6}  verdict")
base = arms["node_edge_rank"]
for ep in range(12):
    seeds = sorted(set(arms["node_edge"]) & set(base))
    diff = [arms["node_edge"][s][ep] - base[s][ep] for s in seeds
            if len(arms["node_edge"][s]) > ep and len(base[s]) > ep]
    if len(diff) < 2:
        continue
    m = mean(diff)
    lo, hi = bootstrap_mean_interval(diff)
    up = sum(1 for d in diff if d > 0)
    bound = clopper_pearson_lower_bound(up, len(diff))
    dn = clopper_pearson_lower_bound(len(diff) - up, len(diff))
    v = "better" if bound > 0.5 else ("worse" if dn > 0.5 else "undecided")
    print(f"{ep+1:>6}  {m:>+10.5f}  [{lo:>+9.5f},{hi:>+9.5f}]  {up:>2}/{len(diff)}  {v}")
