#!/usr/bin/env python3
"""Aggregate frozen full-pair versus marginal/unary diagnostics across seeds/clusters."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np

SCORES=("full_contact","unary_only","residue_head_outer_product")
METRICS=("average_precision","recall_at_P","reciprocal_rank_first_positive","hits_at_1")

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 files=sorted(a.root.glob('seed-*/ABF30B_BEYOND_MARGINALS_COMPONENT_V1.json'))
 if len(files)!=3: raise ValueError(f'expected 3 seeds, got {len(files)}')
 docs=[json.load(open(p)) for p in files];seeds=[int(d['seed']) for d in docs]
 ids=sorted(set.intersection(*(set(d['per_instance']) for d in docs)))
 if len(ids)!=549: raise ValueError(len(ids))
 clusters={i:str(docs[0]['per_instance'][i]['antigen_cluster_id']) for i in ids};cluster_names=sorted(set(clusters.values()))
 rng=np.random.default_rng(20260930)
 def crossed(score_a,score_b,metric):
  mat=np.empty((len(docs),len(cluster_names)),dtype=float);raw=[];seed_effects={}
  for si,d in enumerate(docs):
   sr=[]
   for ci,c in enumerate(cluster_names):
    vals=[]
    for iid in ids:
     if clusters[iid]!=c: continue
     m=d['per_instance'][iid]['metrics_by_score']; vals.append(float(m[score_a][metric])-float(m[score_b][metric]))
    if not vals: raise ValueError(c)
    mat[si,ci]=np.mean(vals);sr.extend(vals);raw.extend(vals)
   seed_effects[str(d['seed'])]=float(np.mean(sr))
  B=10000;draw=np.empty(B);component_only=np.empty(B)
  for b in range(B):
   ss=rng.integers(len(docs),size=len(docs));cc=rng.integers(len(cluster_names),size=len(cluster_names));draw[b]=mat[ss][:,cc].mean();component_only[b]=mat[:,cc].mean()
  return {'equal_complex':float(np.mean(raw)),'equal_antigen_cluster_equal_seed':float(mat.mean()),'crossed_seed_cluster_ci95':list(map(float,np.quantile(draw,[.025,.975]))),'cluster_only_ci95_conditional_on_fixed_seeds':list(map(float,np.quantile(component_only,[.025,.975]))),'positive_seed_cluster_cell_fraction':float((mat>0).mean()),'seed_effects':seed_effects,'cluster_count':len(cluster_names),'complex_count':len(ids),'bootstrap_repeats':B}
 out={'format_version':'epic_pair_decoder_existing_evidence_v1','status':'COMPLETE_EXISTING_THREE_SEED_DIAGNOSTIC','source_files':{str(p):sha(p) for p in files},'seeds':seeds,'complex_count':len(ids),'antigen_cluster_count':len(cluster_names),'authority':docs[0]['authority'],'comparisons':{},'claim_boundary':'Post-submission development/historical diagnostic from the successor FrameContact checkpoint family; no retraining, no test selection, and not an exact ablation of the anonymous submitted EPIC checkpoint.'}
 for b in ('unary_only','residue_head_outer_product'):
  out['comparisons'][f'full_contact_minus_{b}']={m:crossed('full_contact',b,m) for m in METRICS}
 out['seed_score_summaries']={str(d['seed']):d['score_summary'] for d in docs}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'output':str(a.output),'sha256':sha(a.output),'ap_full_minus_unary':out['comparisons']['full_contact_minus_unary_only']['average_precision'],'ap_full_minus_outer':out['comparisons']['full_contact_minus_residue_head_outer_product']['average_precision']},indent=2))
if __name__=='__main__': main()
