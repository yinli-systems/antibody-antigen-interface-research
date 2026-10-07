#!/usr/bin/env python3
"""What actually differed between the four localisation arms.

`selection_metric` names the validation quantity that drives three things at
once: ReduceLROnPlateau, the best-checkpoint test, and the early-stopping
counter. All four arms set it to `total` -- but `total` is the weighted sum of
the terms each arm enables, so the four arms monitor four different quantities
under one name.

That does not invalidate the ablation. It decides what the ablation is an
ablation of: a loss term alone, or a loss term together with the schedule and
stopping behaviour its removal induces. This reads the run histories to find out.
"""
import json
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path("/ssd/scxi253/asep_bench/localise")
ARMS = ["node_only", "node_edge", "node_rank", "node_edge_rank"]

print(f"{'arm':>16} {'seeds':>5} {'epochs run':>18} {'selected epoch':>18} "
      f"{'LR drops':>10} {'final LR':>11}")
print("-" * 84)
summary = {}
for arm in ARMS:
    ran, selected, drops, finals, weights, metrics, negs = [], [], [], [], set(), set(), set()
    for d in sorted(ROOT.glob(f"{arm}/seed_*"), key=lambda p: int(p.name.split("_")[1])):
        f = d / "train" / "history.json"
        if not f.is_file():
            continue
        payload = json.load(open(f))
        h, c = payload["history"], payload["config"]
        ran.append(len(h))
        best = min(range(len(h)), key=lambda i: h[i]["selection_value"])
        selected.append(best + 1)
        lrs = [e["learning_rate"] for e in h]
        drops.append(sum(1 for a, b in zip(lrs, lrs[1:]) if b < a))
        finals.append(lrs[-1])
        w = c["loss_weights"]
        weights.add((w["edge"], w["partner_rank"]))
        metrics.add(c.get("selection_metric"))
        negs.add((c.get("negative_count"), c.get("hard_pool_size")))
    summary[arm] = dict(ran=ran, selected=selected, drops=drops, finals=finals,
                        weights=weights, metrics=metrics, negs=negs)
    print(f"{arm:>16} {len(ran):>5} "
          f"{mean(ran):>8.1f} +-{pstdev(ran):<6.2f} "
          f"{mean(selected):>9.1f} +-{pstdev(selected):<6.2f} "
          f"{mean(drops):>10.1f} {mean(finals):>11.2e}")

print()
print("declared configuration, which must be identical apart from the two weights")
for arm in ARMS:
    s = summary[arm]
    print(f"  {arm:>16}  weights(edge,rank)={sorted(s['weights'])}  "
          f"selection_metric={sorted(s['metrics'])}  negatives={sorted(s['negs'])}")

print()
print("per-seed selected epoch, to see whether the arms stop in the same place")
for arm in ARMS:
    print(f"  {arm:>16}  {summary[arm]['selected']}")
print()
print("per-seed epochs actually run")
for arm in ARMS:
    print(f"  {arm:>16}  {summary[arm]['ran']}")
