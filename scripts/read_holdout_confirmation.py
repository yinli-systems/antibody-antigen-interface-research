#!/usr/bin/env python3
"""The single registered read, per protocols/HOLDOUT_CONFIRMATION_V1.md."""
import json, sys, statistics
from pathlib import Path
sys.path.insert(0, "/ssd/scxi253/repr_ablation/source/src")
sys.path.insert(0, "/ssd/scxi253/repr_ablation/source/scripts")
from aggregate_baseline_margin import bootstrap_mean_interval, clopper_pearson_lower_bound

ROOT = Path("/ssd/scxi253/holdout")
CONTRASTS = [("esm_if1_only", "esm_if1_shuffled", "ESM-IF1"),
             ("proteinmpnn_aligned", "proteinmpnn_shuffled", "ProteinMPNN")]

# the 185 test complexes no model, including the EpiFormer baseline, has seen
epi = set(open("/tmp/epiformer_test_ids.txt").read().split())
manifest = json.load(open("/ssd/scxi253/model_deployment/data/interaction_first/double_remote_split.json"))
sp = manifest.get("splits", manifest)
test_ids = set(sp["test"])
unseen = test_ids - epi
print(f"test complexes: {len(test_ids)}   never seen by anything: {len(unseen)}")

def per_complex(arm, seed):
    import pandas
    from interaction_first.metrics import average_precision
    f = ROOT / arm / f"seed_{seed}" / "pair_predictions.parquet"
    if not f.is_file():
        return None
    frame = pandas.read_parquet(f)
    out = {}
    for cid, g in frame.groupby("instance_id", sort=True):
        v = average_precision(g["label"].to_numpy().astype(bool),
                              g["score"].to_numpy().astype(float))
        if isinstance(v, float):
            out[str(cid)] = v
    return out

def report(a, b, label, subset, subset_name):
    rows = {}
    for s in range(1, 11):
        pa, pb = per_complex(a, s), per_complex(b, s)
        if pa is None or pb is None:
            continue
        shared = sorted((set(pa) & set(pb) & subset) if subset else (set(pa) & set(pb)))
        if not shared:
            continue
        rows[s] = (statistics.mean(pa[c] for c in shared),
                   statistics.mean(pb[c] for c in shared),
                   [pa[c] - pb[c] for c in shared])
    if len(rows) < 2:
        print(f"  {label} / {subset_name}: only {len(rows)} seeds -- not read")
        return None
    seeds = sorted(rows)
    diff = [rows[s][0] - rows[s][1] for s in seeds]
    m = statistics.mean(diff)
    lo, hi = bootstrap_mean_interval(diff)
    up = sum(1 for d in diff if d > 0)
    bound = clopper_pearson_lower_bound(up, len(diff))
    ci_rule = lo > 0 or hi < 0
    by_complex = {}
    for s in seeds:
        for i, c in enumerate(sorted(set().union(*[set()]) or [])):
            pass
    print(f"  {label:>12} / {subset_name:<18} aligned {statistics.mean(rows[s][0] for s in seeds):.6f}  "
          f"shuffled {statistics.mean(rows[s][1] for s in seeds):.6f}")
    print(f"               {'':18} diff {m:+.6f}  CI [{lo:+.6f},{hi:+.6f}]  "
          f"{up}/{len(seeds)} up  sign bound {bound:.4f}  ci-excludes-0 {ci_rule}")
    return {"mean": m, "ci": [lo, hi], "up": up, "n": len(seeds),
            "sign_bound": bound, "ci_excludes_zero": ci_rule,
            "aligned_mean": statistics.mean(rows[s][0] for s in seeds),
            "shuffled_mean": statistics.mean(rows[s][1] for s in seeds)}

out = {}
for a, b, label in CONTRASTS:
    print()
    out[label] = {
        "all_217": report(a, b, label, None, "all 217"),
        "unseen_185": report(a, b, label, unseen, f"unseen {len(unseen)}"),
    }
print()
print(json.dumps({"format_version": "holdout_confirmation_v1",
                  "test_complexes": len(test_ids), "never_seen": len(unseen),
                  "contrasts": out}, indent=1, sort_keys=True)[:60] + " ...")
json.dump({"format_version": "holdout_confirmation_v1",
           "test_complexes": len(test_ids), "never_seen": len(unseen),
           "contrasts": out},
          open("/ssd/scxi253/holdout/confirmation.json", "w"), indent=2, sort_keys=True)
