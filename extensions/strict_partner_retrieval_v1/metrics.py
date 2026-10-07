#!/usr/bin/env python3
"""Protocol-locked partner-retrieval metrics and paired uncertainty."""
from __future__ import annotations
import hashlib, math
from typing import Any, Mapping, Sequence
import numpy as np

METRICS=("mrr","recall_at_1","recall_at_5","recall_at_10")

def exact_random_hit_probability(N:int,m:int,K:int)->float:
    if N<1 or m<1 or m>N: raise ValueError("invalid N/m")
    K=min(int(K),N)
    if K>=N-m+1:return 1.0
    return 1.0-math.comb(N-m,K)/math.comb(N,K)

def _stable_order(scores:np.ndarray,candidate_ids:Sequence[str])->np.ndarray:
    s=np.asarray(scores,dtype=np.float64)
    if s.ndim!=1 or len(s)!=len(candidate_ids) or not np.isfinite(s).all():raise ValueError("bad score vector")
    if len(set(candidate_ids))!=len(candidate_ids):raise ValueError("duplicate candidate IDs")
    # np.lexsort uses last key as primary: descending score, then canonical digest.
    return np.lexsort((np.asarray(candidate_ids,dtype="U"),-s))

def query_vector(scores:np.ndarray,candidate_ids:Sequence[str],positive_ids:Sequence[str])->dict[str,float]:
    pos=set(map(str,positive_ids))
    if not pos:raise ValueError("query has zero positives")
    index={str(x):i for i,x in enumerate(candidate_ids)}
    if not pos.issubset(index):raise ValueError("positive outside candidate library")
    order=_stable_order(scores,candidate_ids)
    ranks=[rank for rank,j in enumerate(order,1) if str(candidate_ids[int(j)]) in pos]
    best=min(ranks);N=len(candidate_ids);m=len(pos)
    out={"mrr":1.0/best,"recall_at_1":float(best<=1),"recall_at_5":float(best<=min(5,N)),
         "recall_at_10":float(best<=min(10,N)),"best_positive_rank":float(best),
         "candidate_count":float(N),"positive_count":float(m)}
    for k in (1,5,10):
        out[f"chance_recall_at_{k}"]=exact_random_hit_probability(N,m,k)
    return out

def direction_report(score_matrix:np.ndarray,query_ids:Sequence[str],candidate_ids:Sequence[str],
                     positives:Mapping[str,Sequence[str]])->dict[str,Any]:
    x=np.asarray(score_matrix,dtype=np.float64)
    if x.shape!=(len(query_ids),len(candidate_ids)) or not np.isfinite(x).all():raise ValueError("matrix shape/nonfinite")
    if set(map(str,query_ids))!=set(map(str,positives)):raise ValueError("positive query inventory mismatch")
    rows=[]
    for i,q in enumerate(query_ids):
        r=query_vector(x[i],candidate_ids,positives[str(q)]);r["query_id"]=str(q);rows.append(r)
    point={m:float(np.mean([r[m] for r in rows])) for m in METRICS}
    chance={f"recall_at_{k}":float(np.mean([r[f"chance_recall_at_{k}"] for r in rows])) for k in (1,5,10)}
    enrichment={k:point[k]/chance[k] if chance[k]>0 else None for k in chance}
    return {"query_count":len(rows),"candidate_count":len(candidate_ids),"point":point,
            "mean_exact_random_chance":chance,"chance_enrichment":enrichment,"per_query":rows}

def bidirectional_report(score_matrix:np.ndarray,antibody_ids:Sequence[str],antigen_ids:Sequence[str],
                         positive_pairs:Sequence[tuple[str,str]])->dict[str,Any]:
    pairs={(str(a),str(g)) for a,g in positive_pairs}
    if not pairs:raise ValueError("no positives")
    ab_pos={str(a):[] for a in antibody_ids};ag_pos={str(g):[] for g in antigen_ids}
    for a,g in pairs:
        if a not in ab_pos or g not in ag_pos:raise ValueError("positive pair outside library")
        ab_pos[a].append(g);ag_pos[g].append(a)
    if any(not x for x in ab_pos.values()) or any(not x for x in ag_pos.values()):raise ValueError("zero-positive query")
    ab=direction_report(score_matrix,antibody_ids,antigen_ids,ab_pos)
    ag=direction_report(np.asarray(score_matrix).T,antigen_ids,antibody_ids,ag_pos)
    bi={m:0.5*(ab["point"][m]+ag["point"][m]) for m in METRICS}
    return {"antibody_to_antigen":ab,"antigen_to_antibody":ag,"bidirectional":bi,
            "positive_pair_count":len(pairs)}

def paired_query_bootstrap(primary:dict[str,Any],control:dict[str,Any],seed:int=20260930,
                           draws:int=10000)->dict[str,Any]:
    rng=np.random.default_rng(seed);out={}
    for direction in ("antibody_to_antigen","antigen_to_antibody"):
        pa=primary[direction]["per_query"];co=control[direction]["per_query"]
        if [r["query_id"] for r in pa]!=[r["query_id"] for r in co]:raise ValueError("query order differs")
        n=len(pa);idx=rng.integers(0,n,size=(draws,n))
        out[direction]={}
        for m in METRICS:
            d=np.asarray([a[m]-b[m] for a,b in zip(pa,co)],dtype=np.float64)
            z=d[idx].mean(1)
            out[direction][m]={"effect":float(d.mean()),"ci95":list(map(float,np.quantile(z,[.025,.975])))}
    out["bidirectional"]={m:{"effect":0.5*(out["antibody_to_antigen"][m]["effect"]+out["antigen_to_antibody"][m]["effect"])}
                          for m in METRICS}
    return out

def grouped_bootstrap(primary:dict[str,Any],control:dict[str,Any],
                      query_to_component:Mapping[str,str],seed:int=20260930,
                      draws:int=10000)->dict[str,Any]:
    rng=np.random.default_rng(seed);result={}
    for direction in ("antibody_to_antigen","antigen_to_antibody"):
        p={x["query_id"]:x for x in primary[direction]["per_query"]};c={x["query_id"]:x for x in control[direction]["per_query"]}
        if set(p)!=set(c) or not set(p).issubset(query_to_component):raise ValueError("component inventory mismatch")
        groups=sorted({query_to_component[q] for q in p})
        result[direction]={}
        for m in METRICS:
            vals=np.asarray([np.mean([p[q][m]-c[q][m] for q in p if query_to_component[q]==g]) for g in groups])
            z=vals[rng.integers(0,len(vals),size=(draws,len(vals)))].mean(1)
            result[direction][m]={"groups":len(groups),"equal_group_effect":float(vals.mean()),
                                  "ci95":list(map(float,np.quantile(z,[.025,.975]))),
                                  "positive_group_fraction":float((vals>0).mean())}
    return result

def crossed_axis_bootstrap(primary_scores:np.ndarray,control_scores:np.ndarray,
                           antibody_ids:Sequence[str],antigen_ids:Sequence[str],
                           positive_pairs:Sequence[tuple[str,str]],seed:int=20260930,
                           draws:int=5000)->dict[str,Any]:
    """Pigeonhole-style row/column resampling; recomputes retrieval on sampled libraries.

    Duplicated sampled entities receive occurrence-suffixed IDs so each draw is a
    finite multiset library. Positives are inherited from the original pair incidence.
    This is an exploratory conditional interval, not an iid population guarantee.
    """
    p=np.asarray(primary_scores,float);c=np.asarray(control_scores,float)
    if p.shape!=c.shape or p.shape!=(len(antibody_ids),len(antigen_ids)):raise ValueError("shape")
    pairset={(str(a),str(g)) for a,g in positive_pairs};rng=np.random.default_rng(seed)
    vals={m:[] for m in METRICS}
    for _ in range(draws):
        ai=rng.integers(0,len(antibody_ids),size=len(antibody_ids));gi=rng.integers(0,len(antigen_ids),size=len(antigen_ids))
        aa=[f"{antibody_ids[j]}#r{k}" for k,j in enumerate(ai)];gg=[f"{antigen_ids[j]}#c{k}" for k,j in enumerate(gi)]
        pp=[]
        for ar,j in enumerate(ai):
            for gr,k in enumerate(gi):
                if (str(antibody_ids[j]),str(antigen_ids[k])) in pairset:pp.append((aa[ar],gg[gr]))
        # A resample may produce zero-positive rows/columns; condition on retained positive queries.
        pos_a={a for a,_ in pp};pos_g={g for _,g in pp}
        if not pos_a or not pos_g:continue
        keep_a=[i for i,a in enumerate(aa) if a in pos_a];keep_g=[i for i,g in enumerate(gg) if g in pos_g]
        aa2=[aa[i] for i in keep_a];gg2=[gg[i] for i in keep_g];pp2=[x for x in pp if x[0] in set(aa2) and x[1] in set(gg2)]
        rp=bidirectional_report(p[np.ix_(ai[keep_a],gi[keep_g])],aa2,gg2,pp2)
        rc=bidirectional_report(c[np.ix_(ai[keep_a],gi[keep_g])],aa2,gg2,pp2)
        for m in METRICS:vals[m].append(rp["bidirectional"][m]-rc["bidirectional"][m])
    return {m:{"draws_used":len(v),"ci95":list(map(float,np.quantile(v,[.025,.975]))) if v else None}
            for m,v in vals.items()}

def score_matrix_sha256(scores:np.ndarray,rows:Sequence[str],cols:Sequence[str])->str:
    x=np.asarray(scores,dtype="<f4",order="C")
    if x.shape!=(len(rows),len(cols)) or not np.isfinite(x).all():raise ValueError("matrix")
    h=hashlib.sha256()
    for r in rows:h.update(str(r).encode()+b"\n")
    h.update(b"--COLS--\n")
    for c in cols:h.update(str(c).encode()+b"\n")
    h.update(x.tobytes())
    return h.hexdigest()
