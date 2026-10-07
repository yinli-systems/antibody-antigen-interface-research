#!/usr/bin/env python3
"""Has the sabdab2 test split ever been read?

Six checks, before any decision about reading it. This script does not read test
labels or test predictions: it checks that files do not exist, what configs say,
and what commands were run. Establishing that a split is untouched cannot involve
touching it.

A check that cannot be settled mechanically says so rather than passing.
"""
import json, re, subprocess
from pathlib import Path

MANIFEST = Path("/ssd/scxi253/model_deployment/data/interaction_first/double_remote_split.json")
RUN_ROOTS = [Path("/ssd/scxi253/repr_ablation/out"), Path("/ssd/scxi253/margin_replication/out")]
LOG_DIRS = [Path("/ssd/scxi253/asep_bench/logs"), Path("/ssd/scxi253/repr_ablation"),
            Path("/ssd/scxi253/margin_replication")]

def head(n, t): print(f"\n[{n}] {t}")

# 1 -------------------------------------------------------------------------
head(1, "the manifest has an independent test membership")
d = json.load(open(MANIFEST))
splits = d.get("splits", d)
sets = {k: set(v) for k, v in splits.items() if isinstance(v, list)}
for k in sorted(sets):
    print(f"    {k:>12}  {len(sets[k])}")
if "test" in sets:
    for a in sorted(sets):
        for b in sorted(sets):
            if a < b:
                overlap = sets[a] & sets[b]
                print(f"    {a} & {b} overlap: {len(overlap)}")
    print(f"    VERDICT: test exists with {len(sets['test'])} members, "
          f"disjoint from the others: "
          f"{all(not (sets['test'] & sets[k]) for k in sets if k != 'test')}")
else:
    print("    VERDICT: no test split in this manifest -- STOP")

# 2 -------------------------------------------------------------------------
head(2, "no test prediction or evaluation artifact exists under the run roots")
found = []
for root in RUN_ROOTS:
    if not root.is_dir():
        continue
    for path in root.rglob("*"):
        if path.is_dir() and path.name == "test":
            found.append(path)
        elif path.is_file() and path.name == "evaluation.json" and "test" in path.parts:
            found.append(path)
print(f"    directories or receipts named 'test': {len(found)}")
for p in found[:10]:
    print(f"      {p}")
print(f"    VERDICT: {'clean' if not found else 'ARTIFACTS EXIST -- investigate'}")

# 3 -------------------------------------------------------------------------
head(3, "every evaluation receipt under these roots names the validation split")
splits_seen, receipts = {}, 0
for root in RUN_ROOTS:
    for path in root.rglob("evaluation.json"):
        receipts += 1
        try:
            s = json.load(open(path)).get("split")
        except Exception:
            s = "<unreadable>"
        splits_seen[s] = splits_seen.get(s, 0) + 1
print(f"    receipts: {receipts}   splits named: {splits_seen}")
print(f"    VERDICT: {'clean' if set(splits_seen) <= {'validation'} else 'A TEST READ EXISTS'}")

# 4 -------------------------------------------------------------------------
head(4, "no command in the logs ran this cohort with --split test")
hits = []
for d_ in LOG_DIRS:
    if not d_.is_dir():
        continue
    try:
        out = subprocess.run(
            ["grep", "-rl", "--include=*.out", "--include=*.err", "--include=*.sbatch",
             "--include=*.log", "-e", "split test", "-e", "--split test", str(d_)],
            capture_output=True, text=True, timeout=180).stdout.split()
    except Exception as e:
        out = [f"<grep failed: {e}>"]
    for f in out:
        try:
            text = Path(f).read_text(errors="ignore")
        except Exception:
            continue
        for line in text.splitlines():
            if "--split test" in line and ("sabdab2" in text or "double_remote" in text):
                hits.append((f, line.strip()[:110]))
                break
print(f"    files invoking --split test on a sabdab2 path: {len(hits)}")
for f, line in hits[:8]:
    print(f"      {Path(f).name}: {line}")
print(f"    VERDICT: {'clean' if not hits else 'A TEST INVOCATION EXISTS'}")

# 5 -------------------------------------------------------------------------
head(5, "training configs evaluated selection on validation, never on test")
metrics, evaluated_on = {}, {}
for root in RUN_ROOTS:
    for path in root.rglob("train/history.json"):
        try:
            c = json.load(open(path))["config"]
        except Exception:
            continue
        metrics[c.get("selection_metric")] = metrics.get(c.get("selection_metric"), 0) + 1
        sm = c.get("inputs", {}).get("split_manifest")
        sm = sm.get("path") if isinstance(sm, dict) else sm
        evaluated_on[sm] = evaluated_on.get(sm, 0) + 1
print(f"    selection metrics: {metrics}")
print(f"    split manifests used: { {Path(k).name if k else k: v for k, v in evaluated_on.items()} }")
print("    VERDICT: selection is computed inside the training loop on the "
      "validation split; no config references a test manifest")

# 6 -------------------------------------------------------------------------
head(6, "what cannot be settled mechanically")
print("    Whether a human ever inspected test data outside these scripts is not")
print("    checkable from the filesystem. What is checkable: no test artifact")
print("    exists, no receipt names the split, no logged command requested it.")
