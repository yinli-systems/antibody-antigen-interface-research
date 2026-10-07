#!/usr/bin/env python3
"""Label-free gate diagnostics only; never reads contact arrays or AP outcomes."""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from qualification_core import digest,validate_rescue_checkpoint,new_json

def imp(path):
    s=importlib.util.spec_from_file_location("gate_diag_original",path);m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m);return m

def sls(parts):
    out=[];s=0
    for x in parts: out.append(slice(s,s+len(x)));s+=len(x)
    return out

def metrics(matcher,base,extra,parts):
    import torch
    names=("optimal_gain","moved_fraction","mean_abs_displacement","row_top1_nonidentity","row_max_minus_diagonal")
    v={k:[] for k in names};w=[]
    with torch.inference_mode():
        for sl in sls(parts):
            L=matcher.logits(base[sl],extra[sl]).detach().cpu().numpy().astype(np.float64);n=len(L)
            rows,cols=linear_sum_assignment(-L);diag=np.diag(L);arg=L.argmax(1)
            v["optimal_gain"].append(float((L[rows,cols].sum()-diag.sum())/n))
            v["moved_fraction"].append(float(np.mean(cols!=np.arange(n))))
            v["mean_abs_displacement"].append(float(np.mean(np.abs(cols-np.arange(n)))/max(n-1,1)))
            v["row_top1_nonidentity"].append(float(np.mean(arg!=np.arange(n))))
            v["row_max_minus_diagonal"].append(float(np.mean(L.max(1)-diag)));w.append(n)
    return {k:float(np.average(x,weights=w)) for k,x in v.items()}

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--plan",type=Path,required=True);ap.add_argument("--plan-sha256",required=True);ap.add_argument("--index",type=int,required=True);ap.add_argument("--output",type=Path,required=True);ap.add_argument("--calibration-size",type=int,default=256);a=ap.parse_args()
    if digest(a.plan)!=a.plan_sha256: raise ValueError("plan hash mismatch")
    p=json.load(open(a.plan));assert not p["test_access"] and not p["ood_access"] and not p["contact_selection"]
    o=imp(p["original_source"]);old=o.verify_plan(Path(p["training_plan"]),p["training_plan_sha256"]);rseed,dseed=p["seed_pairs"][a.index]
    import torch,pandas as pd
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded=torch.load(p["rescue_checkpoints"][a.index]["path"],map_location="cpu",weights_only=False)
    validate_rescue_checkpoint(loaded,json.load(open(p["rescue_checkpoints"][a.index]["history"])),p["training_plan_sha256"],rseed)
    matcher=o.Matcher().to(device);matcher.load_state_dict(loaded["state"]);matcher.eval()
    split=json.load(open(o.SPLIT));frame=pd.read_parquet(o.PREP,columns=["instance_id","ab_sequence_sha256","ag_sequence_sha256"]).set_index("instance_id",verify_integrity=True)
    seqs,inst=o.sequence_metadata();source=o.build_sources(o.MANIFEST,esm_if1_root=o.ESMIF1)[0];base_store=o.EmbeddingStore(o.BASE,.5)
    keys=o.unique_keys("train",split,frame,inst,source);keys=sorted(keys,key=lambda k:hashlib.sha256(f"{rseed}|{k}".encode()).hexdigest())[:a.calibration_size]
    def full(k): return o.load_full_features(k,base_store,source,device)
    names=("optimal_gain","moved_fraction","mean_abs_displacement","row_top1_nonidentity","row_max_minus_diagonal")
    clean={n:[] for n in names};cor={c:{n:[] for n in names} for c in p["conditions"]};oracle={c:[] for c in p["conditions"]}
    for key in keys:
        b,e=full(key);parts=seqs[key];m=metrics(matcher,b,e,parts)
        for n in names: clean[n].append(m[n])
        for c in p["conditions"]:
            mode,k=c.split(":");order=o.synthetic_order(parts,mode,int(k),old["corruption_seed"],key);idx=torch.as_tensor(order,device=device,dtype=torch.long);m=metrics(matcher,b,e.index_select(0,idx),parts)
            for n in names: cor[c][n].append(m[n])
            oracle[c].append(float(np.mean(order!=np.arange(len(order)))))
    th={n:float(np.quantile(clean[n],.995,method="higher")) for n in names}
    representation=o.build_sources(o.MANIFEST,esm_if1_root=o.ESMIF1);store=o.CompositeEmbeddingStore(o.EmbeddingStore(o.BASE,.5),representation)
    def joined(k):
        chunks,starts,length=store.load(k,device);return o.merge_chunk_embeddings(chunks,starts,length,device)
    val={n:[] for n in names};vcor={c:{n:[] for n in names} for c in p["conditions"]};voracle={c:[] for c in p["conditions"]}
    for iid in p["ids"]:
        cs=[];cc={c:[] for c in p["conditions"]};co={c:[] for c in p["conditions"]}
        for col in ("ag_sequence_sha256","ab_sequence_sha256"):
            key=str(frame.loc[iid,col]);j=joined(key);parts=seqs[key];presence=j[:,-1]
            if not bool(presence[0]): continue
            b=j[:,:1280];e=j[:,1280:-1];cs.append(metrics(matcher,b,e,parts))
            for c in p["conditions"]:
                mode,k=c.split(":");order=o.synthetic_order(parts,mode,int(k),old["corruption_seed"],key);idx=torch.as_tensor(order,device=device,dtype=torch.long);cc[c].append(metrics(matcher,b,e.index_select(0,idx),parts));co[c].append(float(np.mean(order!=np.arange(len(order)))))
        for n in names: val[n].append(max([x[n] for x in cs],default=float("-inf")))
        for c in p["conditions"]:
            for n in names: vcor[c][n].append(max([x[n] for x in cc[c]],default=float("-inf")))
            voracle[c].append(max(co[c],default=0.0))
    out={"status":"COMPLETE_LABEL_FREE_GATE_HEURISTIC_DIAGNOSTIC","rescue_seed":rseed,"downstream_seed":dseed,"calibration_size":len(keys),"validation_complex_count":len(p["ids"]),"threshold_rule":"99.5th percentile clean train score; numpy method=higher","metrics":{},"oracle_synthetic_construction":{},"contact_labels_read":False,"contact_metrics_read":False,"test_labels_read":False,"ood_labels_read":False}
    for n in names:
        out["metrics"][n]={"threshold":th[n],"calibration_clean_trigger":float(np.mean(np.asarray(clean[n])>th[n])),"validation_clean_trigger":float(np.mean(np.asarray(val[n])>th[n])),"calibration_corruption_trigger":{c:float(np.mean(np.asarray(cor[c][n])>th[n])) for c in p["conditions"]},"validation_corruption_trigger":{c:float(np.mean(np.asarray(vcor[c][n])>th[n])) for c in p["conditions"]},"calibration_clean_quantiles":list(map(float,np.quantile(clean[n],[.5,.9,.95,.99,.995,1]))),"validation_clean_quantiles":list(map(float,np.quantile(val[n],[.5,.9,.95,.99,.995,1])))}
    for c in p["conditions"]: out["oracle_synthetic_construction"][c]={"calibration_true_moved_fraction_mean":float(np.mean(oracle[c])),"validation_true_moved_fraction_mean":float(np.mean(voracle[c])),"note":"construction-only severity diagnostic; forbidden as deployable gate"}
    new_json(a.output,out);print(json.dumps({"seed":rseed,"metrics":out["metrics"],"oracle":out["oracle_synthetic_construction"]},indent=2))
if __name__=="__main__": main()
