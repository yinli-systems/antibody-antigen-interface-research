#!/usr/bin/env python3
"""Phase B: Candidate Space Reduction Analysis (Optimized).

Pre-groups data by instance_id to avoid repeated full-table scans.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd


def _auprc(labels, scores):
    pos_total = int(labels.sum())
    if pos_total == 0: return None
    order = np.argsort(-scores, kind="stable")
    l, s = labels[order], scores[order]
    tp = fp = 0; result = 0.0; start = 0
    while start < len(l):
        end = start + 1
        while end < len(l) and s[end] == s[start]: end += 1
        gp = int(l[start:end].sum()); gn = end - start - gp
        tp += gp; fp += gn
        if gp: result += (gp / pos_total) * (tp / (tp + fp))
        start = end
    return float(result)

def _auroc(labels, scores):
    pos_total = int(labels.sum()); neg_total = len(labels) - pos_total
    if pos_total == 0 or neg_total == 0: return None
    order = np.argsort(-scores, kind="stable")
    l, s = labels[order], scores[order]
    tp = fp = 0; prev_tpr = prev_fpr = area = 0.0; start = 0
    while start < len(l):
        end = start + 1
        while end < len(l) and s[end] == s[start]: end += 1
        gp = int(l[start:end].sum()); gn = end - start - gp
        tp += gp; fp += gn
        tpr, fpr = tp/pos_total, fp/neg_total
        area += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0
        prev_tpr, prev_fpr = tpr, fpr
        start = end
    return float(area)

def compute_metrics(labels, scores):
    n = len(labels); pos_total = int(labels.sum())
    prevalence = pos_total / n if n > 0 else 0.0
    preds = scores >= 0.5; truth = labels.astype(bool)
    tp = int((preds & truth).sum()); fp = int((preds & ~truth).sum())
    fn = int((~preds & truth).sum())
    prec = tp/(tp+fp) if tp+fp else 0.0; rec = tp/(tp+fn) if tp+fn else 0.0
    f1 = 2*prec*rec/(prec+rec) if prec+rec else 0.0
    order = np.argsort(-scores, kind="stable"); sl = labels[order]
    pk_rk = {}
    for k in [10, 50, 100]:
        ek = min(k, n); hits = int(sl[:ek].sum())
        pk_rk["p@%d"%k] = hits/ek if ek else 0.0
        pk_rk["r@%d"%k] = hits/pos_total if pos_total else 0.0
    ef = {}
    for frac in [0.01, 0.05, 0.10]:
        k = max(1, int(round(n*frac))); hits = int(sl[:k].sum())
        ef["ef@%d%%"%(frac*100)] = (hits/k)/prevalence if prevalence > 0 else 0.0
    pos_s, neg_s = scores[truth], scores[~truth]
    if len(pos_s)>1 and len(neg_s)>1:
        pm, nm = float(pos_s.mean()), float(neg_s.mean())
        ps, ns = float(pos_s.std(ddof=1)), float(neg_s.std(ddof=1))
        n1, n2 = len(pos_s), len(neg_s)
        pv = ((n1-1)*ps**2+(n2-1)*ns**2)/(n1+n2-2)
        cd = (pm-nm)/np.sqrt(pv) if pv>0 else 0
    else: pm=nm=ps=ns=cd=None
    # FP at fixed recall
    fp_at = {}
    for tr in [0.10, 0.25, 0.50]:
        tgt = int(np.ceil(pos_total*tr))
        if tgt == 0: continue
        cum = np.cumsum(sl); idx = np.where(cum>=tgt)[0]
        if len(idx)==0: continue
        i = idx[0]; fp_at["fp@r%d%%"%(tr*100)] = int(i+1-int(cum[i]))
    r = {"n":n,"pos":pos_total,"prev":prevalence,"auprc":_auprc(labels,scores),
         "auroc":_auroc(labels,scores),"prec05":prec,"rec05":rec,"f1":f1,"tp":tp,"fp":fp}
    r.update(pk_rk); r.update(ef)
    if cd is not None: r["cohens_d"]=cd; r["gap"]=pm-nm; r["pos_mean"]=pm; r["neg_mean"]=nm
    r.update(fp_at)
    return r

def main():
    pairs_path = Path(sys.argv[1]); nodes_path = Path(sys.argv[2]); out_path = Path(sys.argv[3])
    print("Loading...", flush=True)
    pairs = pd.read_parquet(pairs_path); nodes = pd.read_parquet(nodes_path)
    
    # Pre-group by instance for O(1) lookup
    print("Grouping by instance...", flush=True)
    pairs_by_inst = {k: v for k, v in pairs.groupby("instance_id")}
    nodes_by_inst = {k: v for k, v in nodes.groupby("instance_id")}
    instances = sorted(pairs_by_inst.keys())
    print("  %d instances" % len(instances), flush=True)

    strategies = [
        ("full_grid", None, None),
        ("epi_top30%", 0.30, None),
        ("epi_top20%", 0.20, None),
        ("epi_top10%", 0.10, None),
        ("dual_30e_50p", 0.30, 0.50),
        ("dual_20e_50p", 0.20, 0.50),
        ("dual_10e_50p", 0.10, 0.50),
        ("dual_10e_30p", 0.10, 0.30),
    ]

    all_results = {}
    for sname, epi_frac, para_frac in strategies:
        print("\n=== %s ===" % sname, flush=True)
        all_l, all_s = [], []
        tot_orig = tot_kept = cont_orig = cont_kept = 0

        for inst_id in instances:
            ip = pairs_by_inst[inst_id]
            inodes = nodes_by_inst[inst_id]
            tot_orig += len(ip); cont_orig += int(ip["label"].sum())

            if epi_frac is None and para_frac is None:
                fp = ip
            else:
                mask = np.ones(len(ip), dtype=bool)
                if epi_frac is not None:
                    epi = inodes[inodes["role"]=="epitope"].sort_values("score",ascending=False)
                    k = max(1, int(np.ceil(len(epi)*epi_frac)))
                    top_ag = set(epi.head(k)["res_idx"].values)
                    mask &= ip["ag_res_idx"].isin(top_ag).values
                if para_frac is not None:
                    par = inodes[inodes["role"]=="paratope"].sort_values("score",ascending=False)
                    k = max(1, int(np.ceil(len(par)*para_frac)))
                    top_ab = set(par.head(k)["res_idx"].values)
                    mask &= ip["ab_res_idx"].isin(top_ab).values
                fp = ip[mask]

            tot_kept += len(fp); cont_kept += int(fp["label"].sum())
            if len(fp) > 0:
                all_l.append(fp["label"].values.astype(np.uint8))
                all_s.append(fp["score"].values.astype(np.float64))

        gl = np.concatenate(all_l); gs = np.concatenate(all_s)
        micro = compute_metrics(gl, gs)
        grid_pct = tot_kept/tot_orig if tot_orig else 0
        cont_ret = cont_kept/cont_orig if cont_orig else 0
        
        res = {"strategy":sname,"epi_frac":epi_frac,"para_frac":para_frac,
               "grid_pct":grid_pct,"cont_ret":cont_ret,
               "orig_pairs":tot_orig,"kept_pairs":tot_kept,
               "orig_contacts":cont_orig,"kept_contacts":cont_kept,
               "micro":micro}
        all_results[sname] = res
        
        print("  Grid: %d -> %d (%.1f%%)" % (tot_orig, tot_kept, grid_pct*100), flush=True)
        print("  Contacts: %d/%d retained (%.1f%%)" % (cont_kept, cont_orig, cont_ret*100), flush=True)
        print("  Prevalence: %.4f%% -> %.4f%%" % (cont_orig/tot_orig*100, micro["prev"]*100), flush=True)
        print("  AUPRC: %.4f  AUROC: %.4f" % (micro.get("auprc",0) or 0, micro.get("auroc",0) or 0), flush=True)
        print("  P@10: %.3f  P@50: %.3f  P@100: %.3f" % (micro.get("p@10",0), micro.get("p@50",0), micro.get("p@100",0)), flush=True)
        print("  EF@1%%: %.1fx  EF@5%%: %.1fx" % (micro.get("ef@1%",0), micro.get("ef@5%",0)), flush=True)
        if "cohens_d" in micro:
            print("  Cohen's d: %.2f  Gap: %.3f" % (micro["cohens_d"], micro["gap"]), flush=True)
        if "fp@r50%" in micro:
            print("  FP@R50%%: %d" % micro["fp@r50%"], flush=True)
        if "fp@r25%" in micro:
            print("  FP@R25%%: %d" % micro["fp@r25%"], flush=True)

    # Summary table
    print("\n" + "="*110, flush=True)
    print("SUMMARY", flush=True)
    print("="*110, flush=True)
    print("%-18s %6s %8s %10s %7s %7s %6s %6s %6s %7s %8s %8s" % 
          ("Strategy","Grid%","ContRet%","Prevalence","AUPRC","AUROC","P@50","EF@1%","EF@5%","Cohen_d","FP@R25%","FP@R50%"), flush=True)
    print("-"*110, flush=True)
    for name, res in all_results.items():
        m = res["micro"]
        print("%-18s %5.1f%% %7.1f%% %9.4f%% %7.4f %7.4f %6.3f %5.1fx %5.1fx %7.2f %8s %8s" % (
            name, res["grid_pct"]*100, res["cont_ret"]*100, m["prev"]*100,
            m.get("auprc",0) or 0, m.get("auroc",0) or 0,
            m.get("p@50",0), m.get("ef@1%",0), m.get("ef@5%",0),
            m.get("cohens_d",0) or 0,
            str(m.get("fp@r25%","N/A")), str(m.get("fp@r50%","N/A"))
        ), flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print("\nSaved to", out_path, flush=True)

if __name__ == "__main__":
    main()
