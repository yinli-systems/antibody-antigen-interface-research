#!/usr/bin/env python3
"""Was the ESM-IF1 random control actually width- and coverage-matched?

`aligned > random` is the comparison that carries the capacity argument: it is
what excludes extra input width and extra parameters as the explanation for the
aligned arm's gain. That argument only holds if the control really presented the
model with the same widths and the same parameter count.

The older `build_random_control` hardcoded one arm's shape -- an antibody-only
source plus a both-sides source, both 512 wide -- and the aligned ESM-IF1 arm has
a single both-sides source. Whether those coincide is a question about the runs,
not about the README, so this reads the run configs.

If they match, the capacity control survives. If they do not, the
aligned-vs-random comparison loses its capacity interpretation, while
aligned-vs-shuffled is untouched: both those arms carry identical widths by
construction, differing only in the row order of the same tensors.
"""
import json
from pathlib import Path

ROOT = Path("/ssd/scxi253/repr_ablation/out")
ARMS = ["esm2", "esm_if1_only", "esm_if1_shuffled", "random_control",
        "antiberty_only", "esm2_antiberty_esmif1"]

FIELDS = ["antibody_input_dim", "antigen_input_dim", "input_dim", "d_model"]

print(f"{'arm':>24} {'ab dim':>8} {'ag dim':>8} {'base':>7} {'d_model':>8} "
      f"{'params':>12}  sources")
print("-" * 108)
rows = {}
for arm in ARMS:
    seen = {}
    for d in sorted(ROOT.glob(f"{arm}/seed_*")):
        f = d / "train" / "history.json"
        if not f.is_file():
            continue
        c = json.load(open(f))["config"]
        reps = c.get("representations") or {}
        srcs = reps.get("sources")
        names = tuple(
            (s.get("name"), s.get("dim") or s.get("width"), s.get("covers") or s.get("keys"))
            if isinstance(s, dict) else (s,)
            for s in (srcs or [])
        ) if isinstance(srcs, list) else (str(srcs),)
        key = tuple(c.get(k) for k in FIELDS) + (c.get("parameter_count"),) + (names,)
        seen[key] = seen.get(key, 0) + 1
    if not seen:
        print(f"{arm:>24}  no runs on disk")
        continue
    if len(seen) > 1:
        print(f"{arm:>24}  ** seeds disagree on configuration: {len(seen)} variants **")
    for key, n in seen.items():
        ab, ag, base, dm, params, names = key
        rows[arm] = key
        rendered = "; ".join(
            f"{t[0]}:{t[1]}" if len(t) > 1 else str(t[0]) for t in names
        ) if names and names[0] else "none"
        print(f"{arm:>24} {str(ab):>8} {str(ag):>8} {str(base):>7} {str(dm):>8} "
              f"{str(params):>12}  {rendered[:48]}  (n={n})")

print()
print("the comparison that carries the capacity argument")
for a, b in (("esm_if1_only", "random_control"), ("esm_if1_only", "esm_if1_shuffled")):
    if a in rows and b in rows:
        same_dims = rows[a][:4] == rows[b][:4]
        same_params = rows[a][4] == rows[b][4]
        print(f"  {a} vs {b}: dims match {same_dims}, parameter count match {same_params}")
        if not same_dims:
            print(f"      {a}: {rows[a][:4]}")
            print(f"      {b}: {rows[b][:4]}")
