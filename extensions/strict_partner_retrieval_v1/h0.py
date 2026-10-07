#!/usr/bin/env python3
"""No-training homology-transfer controls for strict partner retrieval."""
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path
from typing import Any
import numpy as np
# Keep direct-script and package imports on the same namespaced module.
import sys as _sys
_package_parent = str(Path(__file__).resolve().parent.parent)
if _package_parent not in _sys.path:
    _sys.path.insert(0, _package_parent)
from strict_partner_retrieval_v1.metrics import bidirectional_report, score_matrix_sha256

VALID=set("ACDEFGHIKLMNPQRSTVWY")

def load_json(p:Path)->dict[str,Any]:
    x=json.loads(p.read_text())
    if not isinstance(x,dict):raise ValueError("object required")
    return x

def kmers(s:str,k:int=3)->set[str]:
    if any(c not in VALID for c in s):raise ValueError("noncanonical sequence")
    return {s[i:i+k] for i in range(max(0,len(s)-k+1))}

def jac(a:str,b:str,k:int=3)->float:
    A=kmers(a,k);B=kmers(b,k)
    if not A and not B:return 1.0 if a==b else 0.0
    return len(A&B)/len(A|B) if A|B else 0.0

def antibody_kmer(x:dict[str,str],y:dict[str,str])->float:
    vals=[jac(x["vh"],y["vh"])]
    if x.get("vl") and y.get("vl"):vals.append(jac(x["vl"],y["vl"]))
    vals.append(jac(x["vh"]+x.get("vl",""),y["vh"]+y.get("vl","")))
    return max(vals)

def _align_counts(left:str,right:str,aligner)->tuple[int,int]:
    a=next(iter(aligner.align(left,right)));lb,rb=a.aligned;ident=paired=0
    for L,R in zip(lb,rb):
        ls,le=map(int,L);rs,re=map(int,R);w=le-ls
        if w<=0 or re-rs!=w:raise RuntimeError("alignment blocks")
        paired+=w;ident+=sum(x==y for x,y in zip(left[ls:le],right[rs:re]))
    if paired<=0:raise RuntimeError("empty alignment")
    return ident,paired

def build_aligner():
    import Bio
    from Bio.Align import PairwiseAligner, substitution_matrices
    if Bio.__version__!="1.83":raise RuntimeError(f"Biopython 1.83 required, got {Bio.__version__}")
    mat=substitution_matrices.load("BLOSUM62")
    payload={"alphabet":str(mat.alphabet),"scores":[[float(mat[a,b]) for b in str(mat.alphabet)] for a in str(mat.alphabet)]}
    sha=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":"),ensure_ascii=True).encode("ascii")).hexdigest()
    if sha!="2c62ca4bb7a4c3ec5fe5faee64276841028a7e6638cf846c801dd9c5e64527e2":raise RuntimeError("BLOSUM62 payload mismatch")
    al=PairwiseAligner();al.mode="global";al.substitution_matrix=mat;al.open_gap_score=-10.;al.extend_gap_score=-1.
    return al

def nw_similarity(a:str,b:str,aligner)->float:
    vals=[]
    for x,y in ((a,b),(b,a)):
        ident,paired=_align_counts(x,y,aligner)
        vals.append((ident/paired)*min(paired/len(x),paired/len(y)))
    return max(vals)

def antibody_nw(x:dict[str,str],y:dict[str,str],aligner)->float:
    vals=[nw_similarity(x["vh"],y["vh"],aligner)]
    if x.get("vl") and y.get("vl"):vals.append(nw_similarity(x["vl"],y["vl"],aligner))
    vals.append(nw_similarity(x["vh"]+x.get("vl",""),y["vh"]+y.get("vl",""),aligner))
    return max(vals)

def validate_cache(d:dict[str,Any]):
    if d.get("format_version")!="strict_partner_retrieval_h0_cache_v1":raise ValueError("cache format")
    if d.get("role")!="h0_evaluation_only":raise ValueError("cache role")
    if d.get("h0_execution_authorized") is not True:raise PermissionError("H0 execution authority absent")
    if d.get("test_access") is not False or d.get("ood_access") is not False:raise PermissionError("sealed access")
    for key in ("training_antibodies","training_antigens","validation_antibodies","validation_antigens","training_pairs","validation_pairs"):
        if key not in d:raise ValueError(f"missing {key}")
    for side in ("training_antibodies","validation_antibodies"):
        for i,x in d[side].items():
            if not x.get("vh") or any(c not in VALID for c in x["vh"]+x.get("vl","")):raise ValueError(f"bad antibody {i}")
    for side in ("training_antigens","validation_antigens"):
        for i,s in d[side].items():
            if not s or any(c not in VALID for c in s):raise ValueError(f"bad antigen {i}")

def score(cache:dict[str,Any],method:str)->tuple[np.ndarray,dict[str,Any]]:
    ta=cache["training_antibodies"];tg=cache["training_antigens"];va=cache["validation_antibodies"];vg=cache["validation_antigens"]
    train_pairs=[(x["antibody_id"],x["antigen_id"]) for x in cache["training_pairs"]]
    aa=sorted(va);gg=sorted(vg);tai=sorted(ta);tgi=sorted(tg)
    if not train_pairs:raise ValueError("no training pairs")
    al=build_aligner() if method=="nw" else None
    SA=np.empty((len(aa),len(tai)),np.float32);SG=np.empty((len(gg),len(tgi)),np.float32)
    for i,a in enumerate(aa):
        for j,b in enumerate(tai):
            SA[i,j]=antibody_nw(va[a],ta[b],al) if method=="nw" else antibody_kmer(va[a],ta[b])
    for i,g in enumerate(gg):
        for j,h in enumerate(tgi):
            SG[i,j]=nw_similarity(vg[g],tg[h],al) if method=="nw" else jac(vg[g],tg[h])
    ai={x:i for i,x in enumerate(tai)};gi={x:i for i,x in enumerate(tgi)}
    incidence=[(ai[a],gi[g]) for a,g in train_pairs]
    scores=np.full((len(aa),len(gg)),-np.inf,np.float32)
    # max observed-pair transfer.  Batch over train pairs to bound memory.
    for start in range(0,len(incidence),128):
        chunk=incidence[start:start+128]
        A=np.stack([SA[:,i] for i,_ in chunk],axis=1) # valA x P
        G=np.stack([SG[:,j] for _,j in chunk],axis=1) # valG x P
        # score(i,j)=max_p sqrt(A_ip*G_jp)
        cand=np.sqrt(np.maximum(A[:,:,None],0)*np.maximum(G.T[None,:,:],0))
        scores=np.maximum(scores,cand.max(axis=1).astype(np.float32))
    return scores,{"antibody_ids":aa,"antigen_ids":gg,
                   "antibody_similarity_sha256":hashlib.sha256(SA.astype("<f4").tobytes()).hexdigest(),
                   "antigen_similarity_sha256":hashlib.sha256(SG.astype("<f4").tobytes()).hexdigest()}

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--cache",type=Path,required=True);ap.add_argument("--method",choices=("kmer","nw"),required=True);ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args();
    if a.output.exists():raise FileExistsError(a.output)
    d=load_json(a.cache);validate_cache(d);scores,meta=score(d,a.method)
    pp=[(x["antibody_id"],x["antigen_id"]) for x in d["validation_pairs"]]
    report=bidirectional_report(scores,meta["antibody_ids"],meta["antigen_ids"],pp)
    a.output.mkdir(parents=True)
    np.savez_compressed(a.output/"scores.npz",scores=scores,antibody_ids=np.asarray(meta["antibody_ids"]),antigen_ids=np.asarray(meta["antigen_ids"]))
    result={"format_version":"strict_partner_retrieval_h0_result_v1","status":"COMPLETE_H0_"+a.method.upper(),
            "method":a.method,"cache_sha256":hashlib.sha256(a.cache.read_bytes()).hexdigest(),
            "score_matrix_sha256":score_matrix_sha256(scores,meta["antibody_ids"],meta["antigen_ids"]),
            "similarity_receipts":meta,"metrics":report,"nonpositive_semantics":"unlabeled_ranking_candidates",
            "training_performed":False,"contact_labels_read":False,"test_access":False,"ood_access":False}
    (a.output/"result.json").write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps({"status":result["status"],"mrr":report["bidirectional"]["mrr"],"r10":report["bidirectional"]["recall_at_10"]},indent=2))
if __name__=="__main__":main()
