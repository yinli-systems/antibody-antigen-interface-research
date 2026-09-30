#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata,spearmanr
from sklearn.metrics import average_precision_score

METRICS=('average_precision','auroc','epitope_average_precision','paratope_average_precision')
def sha(p:Path): return hashlib.sha256(p.read_bytes()).hexdigest()
def grid(root:Path):
 h=hashlib.sha256(); pairs=pos=0
 for p in sorted((root/'pair_predictions').glob('*.npz'),key=lambda x:x.name):
  z=np.load(p,allow_pickle=False); iid=str(z['instance_id'].item()); ag=z['antigen_residue_index'].astype('<i8'); ab=z['antibody_residue_index'].astype('<i8'); y=z['label'].astype('u1'); order=np.lexsort((ab,ag)); h.update(iid.encode()+b'\0'); h.update(ag[order].tobytes());h.update(ab[order].tobytes());h.update(y[order].tobytes()); pairs+=len(y);pos+=int(y.sum())
 return {'file_count':len(list((root/'pair_predictions').glob('*.npz'))),'digest':h.hexdigest(),'pair_count':pairs,'positive_count':pos}
def main():
 ap=argparse.ArgumentParser();
 for n in ('sequence','uncentered','centered'): ap.add_argument('--'+n,type=Path,required=True)
 ap.add_argument('--structure-manifest',type=Path,required=True); ap.add_argument('--failure-2386',type=Path,required=True);ap.add_argument('--failure-2395',type=Path,required=True);ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
 roots={k:getattr(a,k).resolve() for k in ('sequence','uncentered','centered')}; ev={k:json.load(open(v/'evaluation.json')) for k,v in roots.items()}; rows={k:{r['instance_id']:r for r in d['per_complex']} for k,d in ev.items()}; ids=sorted(rows['sequence']); assert len(ids)==217 and all(set(rows[k])==set(ids) for k in rows)
 grids={k:grid(v) for k,v in roots.items()}; assert len({x['digest'] for x in grids.values()})==1 and all(x['pair_count']==9638880 and x['positive_count']==9036 and x['file_count']==217 for x in grids.values())
 assert ev['centered']['checkpoint_sha256']==ev['uncentered']['checkpoint_sha256']; assert ev['centered']['structure_manifest_sha256']==ev['uncentered']['structure_manifest_sha256']==sha(a.structure_manifest); assert len({ev[k]['canonical_manifest_sha256'] for k in ev})==1 and len({ev[k]['split_manifest_sha256'] for k in ev})==1
 comps={i:rows['sequence'][i]['component_id'] for i in ids}; assert len(set(comps.values()))==68; rng=np.random.default_rng(20260930)
 def contrast(x,y,m):
  raw=np.array([rows[x][i][m]-rows[y][i][m] for i in ids]); vals=[]
  for c in sorted(set(comps.values())):
   ix=[j for j,i in enumerate(ids) if comps[i]==c]; vals.append(float(raw[ix].mean()))
  vals=np.array(vals); draws=vals[rng.integers(len(vals),size=(20000,len(vals)))].mean(1)
  return {'equal_complex':float(raw.mean()),'equal_component':float(vals.mean()),'component_bootstrap_ci95':list(map(float,np.quantile(draws,[.025,.975]))),'positive_complex_fraction':float((raw>0).mean()),'positive_component_fraction':float((vals>0).mean()),'complex_q10_q50_q90':list(map(float,np.quantile(raw,[.1,.5,.9]))),'complex_min_max':[float(raw.min()),float(raw.max())]}
 contrasts={m:{f'{x}_minus_{y}':contrast(x,y,m) for x,y in [('centered','uncentered'),('centered','sequence'),('uncentered','sequence')]} for m in METRICS}
 # pooled calibration diagnostic; post-hoc, never primary
 pooled={}
 for k in ('uncentered','centered'):
  labs=[]; raw=[]; pct=[]; zs=[]; means=[]; stds=[]; pcs=[]; prev=[]
  for p in sorted((roots[k]/'pair_predictions').glob('*.npz')):
   z=np.load(p,allow_pickle=False); y=z['label'].astype(np.uint8); s=z['score'].astype(np.float64); n=len(s); sd=s.std(); labs.append(y);raw.append(s);pct.append(rankdata(s,method='average')/n);zs.append((s-s.mean())/(sd if sd else 1));means.append(s.mean());stds.append(sd);pcs.append(n);prev.append(y.mean())
  y=np.concatenate(labs); pooled[k]={'raw_ap':float(average_precision_score(y,np.concatenate(raw))),'within_complex_percentile_ap':float(average_precision_score(y,np.concatenate(pct))),'within_complex_z_ap':float(average_precision_score(y,np.concatenate(zs))),'score_mean_vs_pair_count_spearman':float(spearmanr(means,pcs).statistic),'score_mean_vs_prevalence_spearman':float(spearmanr(means,prev).statistic),'score_std_vs_pair_count_spearman':float(spearmanr(stds,pcs).statistic),'role':'post_hoc_calibration_diagnostic_not_registered_endpoint'}
 # CenterMesh semantic equivalence: same formula/dtype, not bitwise identical reduction kernel.
 import torch
 rg=np.random.default_rng(20260930); mx=post_np=post_t=0.
 for n in (1,2,17,78,121,573,1000):
  for _ in range(50):
   x=(rg.normal(size=(n,4,3))*100).astype(np.float32); aa=(x-x.mean(axis=(0,1),keepdims=True)).astype(np.float32); tt=torch.from_numpy(x.copy()); bb=(tt-tt.mean(dim=(0,1),keepdim=True)).numpy(); mx=max(mx,float(np.max(np.abs(aa-bb))));post_np=max(post_np,float(np.max(np.abs(aa.mean(axis=(0,1))))));post_t=max(post_t,float(np.max(np.abs(bb.mean(axis=(0,1))))))
 center={'formula':'points - points.mean((residue, atom), keepdim=True), independently per chain','numpy_vs_torch_max_abs_on_synthetic_float32':mx,'post_center_numpy_mean_max_abs':post_np,'post_center_torch_mean_max_abs':post_t,'bitwise_equivalence_claimed':False,'semantic_equivalence_claimed':True}
 # exploratory structure ranking-score stratification, only cases with direct chain receipts.
 man=json.load(open(a.structure_manifest)); score={r['sequence_sha256']:float(r['chosen']['score']) for r in man['records']}; qr=[]
 for i in ids:
  q=[]
  for v in rows['centered'][i].get('structure_receipts',{}).values():
   if isinstance(v,dict) and v.get('sequence_sha256') in score:q.append(score[v['sequence_sha256']])
  if q:qr.append((i,min(q),rows['centered'][i]['average_precision']-rows['sequence'][i]['average_precision'],rows['centered'][i]['average_precision']-rows['uncentered'][i]['average_precision']))
 q=np.array([x[1] for x in qr]); cuts=np.quantile(q,[0,.25,.5,.75,1]); strata=[]
 for j in range(4):
  ix=(q>=cuts[j]) & ((q<=cuts[j+1]) if j==3 else (q<cuts[j+1])); rr=[qr[z] for z in np.where(ix)[0]]; strata.append({'quartile':j+1,'n':len(rr),'min_score_range':[float(cuts[j]),float(cuts[j+1])],'centered_minus_sequence_mean_ap':float(np.mean([x[2] for x in rr])),'centered_minus_uncentered_mean_ap':float(np.mean([x[3] for x in rr]))})
 out={'status':'COMPLETE_REPARANK_THREE_ARM_AUDIT','evidence_class':'exposed_pretrained_development_diagnostic','instance_count':217,'component_count':68,'grids':grids,'evaluation_sha256':{k:sha(v/'evaluation.json') for k,v in roots.items()},'checkpoint_sha256':{k:ev[k]['checkpoint_sha256'] for k in ev},'structure_manifest_sha256':ev['centered']['structure_manifest_sha256'],'official_source_commit':ev['centered']['official_source_commit'],'centered_resume':{'fresh_model_forward_count':ev['centered'].get('fresh_model_forward_count'),'reused_pair_prediction_count':ev['centered'].get('reused_pair_prediction_count')},'headline':{k:{'macro_pair_ap':ev[k]['primary_macro_mean_per_complex_AP'],'pooled':ev[k]['pooled_pair_metrics'],'epitope_macro_ap':ev[k]['epitope_macro_AP'],'paratope_macro_ap':ev[k]['paratope_macro_AP']} for k in ev},'paired_contrasts':contrasts,'pooled_calibration_diagnostic':pooled,'center_mesh_audit':center,'structure_ranking_score_stratification':{'n_with_direct_receipts':len(qr),'warning':'AF3 ranking score is a model confidence/ranking quantity, not experimental structure accuracy; exploratory only.','quartiles':strata},'failure_provenance':{'1642386_stderr_sha256':sha(a.failure_2386),'1642395_stderr_sha256':sha(a.failure_2395)},'test_labels_read':False,'ood_labels_read':False,'checkpoint_training_overlap_audited':False,'claims':{'structure_useless':False,'sota':False,'clean_architecture_comparison':False},'limitations':['Official checkpoint training overlap with this 217-complex cohort is unknown.','Centered implementation matches the official CenterMesh formula and float32 semantics but NumPy and PyTorch reduction kernels are not bitwise identical; synthetic max absolute difference is recorded.','Structure confidence stratification is post-hoc and does not establish causal dependence on structure accuracy.','Within-complex percentile/z pooled AP values are post-hoc calibration diagnostics, not replacement endpoints.']}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps({'output':str(a.output),'sha256':sha(a.output),'centered_minus_uncentered_ap':contrasts['average_precision']['centered_minus_uncentered'],'centered_minus_sequence_ap':contrasts['average_precision']['centered_minus_sequence'],'pooled_diag':pooled},indent=2))
if __name__=='__main__':main()
