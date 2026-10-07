#!/usr/bin/env python3
"""The registered read of the second structural encoder."""
import json, sys, statistics
from pathlib import Path
sys.path.insert(0, "/ssd/scxi253/repr_ablation/source/src")
sys.path.insert(0, "/ssd/scxi253/repr_ablation/source/scripts")
from aggregate_baseline_margin import bootstrap_mean_interval, clopper_pearson_lower_bound

ROOT = Path("/ssd/scxi253/repr_ablation/out")
ARMS = ["esm2", "proteinmpnn_aligned", "proteinmpnn_shuffled", "proteinmpnn_random"]

def read(arm):
    out = {}
    for d in sorted(ROOT.glob(f"{arm}/seed_*"), key=lambda p: int(p.name.split("_")[1])):
        f = d / "validation" / "evaluation.json"
        if f.is_file():
            out[int(d.name.split("_")[1])] = json.load(open(f))["pair"]["macro"]["auprc"]
    return out

arms = {a: read(a) for a in ARMS}
print("macro pair AUPRC, sabdab2 validation")
for a in ARMS:
    v = arms[a]
    print(f"  {a:>22}  {statistics.mean(v.values()):.6f}   (n={len(v)})")

def compare(a, b):
    seeds = sorted(set(arms[a]) & set(arms[b]))
    diff = [arms[a][s] - arms[b][s] for s in seeds]
    m = statistics.mean(diff)
    lo, hi = bootstrap_mean_interval(diff)
    up = sum(1 for d in diff if d > 0)
    up_bound = clopper_pearson_lower_bound(up, len(diff))
    dn_bound = clopper_pearson_lower_bound(len(diff) - up, len(diff))
    rom = m / statistics.mean(arms[b][s] for s in seeds)
    ci_rule = "yes" if (lo > 0 or hi < 0) else "no"
    sign_rule = ("better" if up_bound > 0.5 else
                 "worse" if dn_bound > 0.5 else "neither")
    print(f"  {a.replace('proteinmpnn_',''):>10} - {b.replace('proteinmpnn_',''):<10} "
          f"{m:>+10.6f}  [{lo:>+9.6f},{hi:>+9.6f}]  {up:>2}/{len(diff)}  "
          f"rom {100*rom:>+6.1f}%   ci-excludes-0 {ci_rule:>3}   sign-rule {sign_rule}")

print()
print("paired, 10 seeds; two registered rules reported side by side")
print(f"  {'contrast':>23}  {'mean diff':>10}  {'bootstrap CI95':>21}  {'up':>5}  "
      f"{'rel':>10}  rules")
for a, b in (("proteinmpnn_aligned", "proteinmpnn_shuffled"),
             ("proteinmpnn_aligned", "proteinmpnn_random"),
             ("proteinmpnn_aligned", "esm2"),
             ("proteinmpnn_random", "esm2"),
             ("proteinmpnn_shuffled", "esm2"),
             ("proteinmpnn_shuffled", "proteinmpnn_random")):
    compare(a, b)

print()
order = sorted(ARMS, key=lambda a: statistics.mean(arms[a].values()))
print("observed ordering by mean:", "  <  ".join(
    a.replace("proteinmpnn_", "") for a in order))
print("registered prediction    : shuffled  <  random  ~  esm2  <  aligned")
