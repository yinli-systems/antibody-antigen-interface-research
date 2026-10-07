#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from qualification_core import digest,validate_rescue_checkpoint,new_json

def import_original(path):
 spec=importlib.util.spec_from_file_location('original_rescue_gate',path);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);return m

def gate_score(matcher,base,extra,parts):
 import torch
 total=0.;nall=0
 with torch.inference_mode():
  for sl in matcher_chain_slices(parts):
   L=matcher.logits(base[sl],extra[sl]).detach().cpu().numpy().astype(np.float64); n=len(L); rows,cols=linear_sum_assignment(-L); total+=float(L[rows,cols].sum()-np.trace(L));nall+=n
 return total/max(nall,1)
def matcher_chain_slices(parts):
 out=[];s=0
 for x in parts:out.append(slice(s,s+len(x)));s+=len(x)
 return out

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--plan-sha256',required=True);ap.add_argument('--aggregate',type=Path,required=True);ap.add_argument('--index',type=int,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--calibration-size',type=int,default=256);a=ap.parse_args()
 if digest(a.plan)!=a.plan_sha256:raise ValueError('plan hash mismatch')
 p=json.load(open(a.plan)); raw=json.load(open(a.aggregate)); o=import_original(p['original_source']); old=o.verify_plan(Path(p['training_plan']),p['training_plan_sha256']); rseed,dseed=p['seed_pairs'][a.index]
 for path,h in {**p['source_sha256'],**p['input_sha256']}.items():
  if digest(path)!=h:raise ValueError('frozen bytes changed '+path)
 if p['test_access'] or p['ood_access'] or p['contact_selection']:raise ValueError('invalid authority')
 import torch,pandas as pd
 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); loaded=torch.load(p['rescue_checkpoints'][a.index]['path'],map_location='cpu',weights_only=False); validate_rescue_checkpoint(loaded,json.load(open(p['rescue_checkpoints'][a.index]['history'])),p['training_plan_sha256'],rseed); matcher=o.Matcher().to(device);matcher.load_state_dict(loaded['state']);matcher.eval()
 split=json.load(open(o.SPLIT)); frame=pd.read_parquet(o.PREP,columns=['instance_id','ab_sequence_sha256','ag_sequence_sha256']).set_index('instance_id',verify_integrity=True); seqs,inst=o.sequence_metadata(); source=o.build_sources(o.MANIFEST,esm_if1_root=o.ESMIF1)[0]; base_store=o.EmbeddingStore(o.BASE,.5)
 keys=o.unique_keys('train',split,frame,inst,source); keys=sorted(keys,key=lambda k:hashlib.sha256(f'{rseed}|{k}'.encode()).hexdigest())[:a.calibration_size]
 def full_pair(key):return o.load_full_features(key,base_store,source,device)
 clean=[];cor={c:[] for c in p['conditions']}
 for key in keys:
  base,extra=full_pair(key);parts=seqs[key];clean.append(gate_score(matcher,base,extra,parts))
  for c in p['conditions']:
   mode,k=c.split(':');order=o.synthetic_order(parts,mode,int(k),old['corruption_seed'],key);idx=torch.as_tensor(order,device=device,dtype=torch.long);cor[c].append(gate_score(matcher,base,extra.index_select(0,idx),parts))
 threshold=float(np.quantile(np.asarray(clean),.995,method='higher')); calibration={'key_count':len(keys),'threshold':threshold,'clean_trigger_rate':float(np.mean(np.asarray(clean)>threshold)),'corruption_trigger_rate':{c:float(np.mean(np.asarray(v)>threshold)) for c,v in cor.items()},'selection_uses_contact_labels':False}
 representation=o.build_sources(o.MANIFEST,esm_if1_root=o.ESMIF1);store=o.CompositeEmbeddingStore(o.EmbeddingStore(o.BASE,.5),representation)
 def joined(key):
  chunks,starts,length=store.load(key,device);return o.merge_chunk_embeddings(chunks,starts,length,device)
 scores={}
 for iid in p['ids']:
  sides=[]
  for col in ('ag_sequence_sha256','ab_sequence_sha256'):
   key=str(frame.loc[iid,col]);j=joined(key);parts=seqs[key];presence=j[:,-1]
   if not bool(presence[0]): sides.append((None,{c:None for c in p['conditions']}));continue
   if not bool((presence==presence[0]).all()):raise ValueError('partial missingness')
   base=j[:,:1280];extra=j[:,1280:-1];cs=gate_score(matcher,base,extra,parts);cc={}
   for c in p['conditions']:
    mode,k=c.split(':');order=o.synthetic_order(parts,mode,int(k),old['corruption_seed'],key);idx=torch.as_tensor(order,device=device,dtype=torch.long);cc[c]=gate_score(matcher,base,extra.index_select(0,idx),parts)
   sides.append((cs,cc))
  clean_vals=[x[0] for x in sides if x[0] is not None]; scores[iid]={'clean':max(clean_vals) if clean_vals else None,'corrupted':{c:max([x[1][c] for x in sides if x[1][c] is not None],default=None) for c in p['conditions']}}
 rec=[x for x in raw['records'] if x['rescue_seed']==rseed and x['downstream_seed']==dseed]; out=[]
 for x in rec:
  sc=scores[x['instance_id']]; ac=sc['corrupted'][x['condition']] is not None and sc['corrupted'][x['condition']]>threshold; acl=sc['clean'] is not None and sc['clean']>threshold
  out.append({**x,'corrupted_gate_score':sc['corrupted'][x['condition']],'clean_gate_score':sc['clean'],'apply_rescue_corrupted':ac,'apply_rescue_clean':acl,'gated_corrupted_ap':x['rescued_ap'] if ac else x['corrupted_ap'],'gated_clean_ap':x['clean_rescued_ap'] if acl else x['aligned_ap']})
 new_json(a.output,{'status':'COMPLETE_LABEL_FREE_GATED_RESCUE_SEED','plan_sha256':a.plan_sha256,'rescue_seed':rseed,'downstream_seed':dseed,'calibration':calibration,'records':out,'test_labels_read':False,'ood_labels_read':False,'contact_labels_used_for_gate':False,'downstream_model_forward_performed':False})
 print(json.dumps({'seed':rseed,'threshold':threshold,'clean_calibration_trigger':calibration['clean_trigger_rate'],'corruption_calibration_trigger':calibration['corruption_trigger_rate'],'development_clean_trigger':float(np.mean([r['apply_rescue_clean'] for r in out])),'development_corruption_trigger':{c:float(np.mean([r['apply_rescue_corrupted'] for r in out if r['condition']==c])) for c in p['conditions']}},indent=2))
if __name__=='__main__':main()
