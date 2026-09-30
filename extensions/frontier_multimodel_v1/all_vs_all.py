#!/usr/bin/env python3
"""Full validation all-vs-all partner retrieval without contact-label access.

Every validation antibody is paired with every validation antigen in both
anchor directions. Known complexes are positives; all other combinations are
unlabelled distractors, never asserted non-binders. Full residue sequences are
used, and candidate scores are frozen checkpoint outputs.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, tempfile, time
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[2]; sys.path[:0]=[str(REPO/'src'),str(REPO/'scripts')]

def sha256_file(path:Path)->str:
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''): h.update(b)
 return h.hexdigest()

def atomic_json(path:Path,payload:dict):
 path.parent.mkdir(parents=True,exist_ok=True); data=json.dumps(payload,indent=2,sort_keys=True,allow_nan=False)+'\n'
 fd,tmp=tempfile.mkstemp(dir=path.parent,prefix='.tmp-',suffix='.json'); os.close(fd)
 try: Path(tmp).write_text(data); os.replace(tmp,path)
 except BaseException: Path(tmp).unlink(missing_ok=True); raise

def pad_batch(tensors,device):
 import torch
 width={int(x.shape[-1]) for x in tensors}
 if len(width)!=1: raise ValueError('feature widths differ')
 length=max(int(x.shape[0]) for x in tensors); batch=len(tensors); dim=next(iter(width))
 out=torch.zeros(batch,length,dim,dtype=tensors[0].dtype,device=device); mask=torch.zeros(batch,length,dtype=torch.bool,device=device)
 for i,x in enumerate(tensors): out[i,:len(x)]=x; mask[i,:len(x)]=True
 return out,mask

def expected_first_positive_reciprocal_rank(scores,positives):
 import numpy as np
 s=np.asarray(scores,float); p=np.asarray(positives,bool)
 if not p.any(): return None
 # Average-rank convention for exact ties; deterministic and ID-order invariant.
 best=float(s[p].max()); before=int((s>best).sum()); tied=s==best; m=int(tied.sum()); k=int(p[tied].sum())
 import math
 weights=[math.comb(m-j,k-1)/math.comb(m,k) for j in range(1,m-k+2)]
 return float(sum(w/(before+j) for j,w in enumerate(weights,1)))

def main():
 a=parser().parse_args()
 import pandas as pd, torch
 from interaction_first.runtime import EmbeddingStore,safe_torch_load,set_reproducible_seed
 from interaction_first.model import PartnerSpecificInteractionModel,merge_chunk_embeddings
 from interaction_first.composite_embeddings import CompositeEmbeddingStore,build_sources,build_generic_sources
 from sabdab2_interaction_first import load_split_assignments
 if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
 if not 0<=a.shard_index<a.shard_count: raise ValueError('bad shard')
 output=a.output.resolve(); output.mkdir(parents=True,exist_ok=False)
 split_path=a.split_manifest.resolve(); assignments=load_split_assignments(split_path)
 frame=pd.read_parquet(a.prepare_dir/'instances.parquet').set_index('instance_id',verify_integrity=True)
 if set(frame.index.astype(str))!=set(assignments): raise ValueError('prepared/split authority mismatch')
 validation=sorted(i for i,s in assignments.items() if s=='validation')
 anchors=[i for i in validation if int(hashlib.sha256(i.encode()).hexdigest()[:16],16)%a.shard_count==a.shard_index]
 if a.max_anchors is not None: anchors=anchors[:a.max_anchors]
 if not anchors: raise ValueError('empty anchor shard')
 checkpoint=safe_torch_load(a.checkpoint,map_location='cpu'); config=checkpoint['config']
 if config['split_manifest_sha256']!=sha256_file(split_path): raise ValueError('checkpoint/split mismatch')
 base=EmbeddingStore(a.embedding_dir.resolve(),a.embedding_cache_gib)
 sources=[]
 if a.esm_if1_root:
  sources.extend(build_sources(a.canonical_manifest.resolve(),esm_if1_root=a.esm_if1_root.resolve(),esm_if1_dim=a.esm_if1_dim))
 sources.extend(build_generic_sources(a.canonical_manifest.resolve(),a.generic_representation_source))
 store=CompositeEmbeddingStore(base,sources) if sources else base
 observed=store.describe() if hasattr(store,'describe') else None
 if config.get('representations')!=observed: raise ValueError('feature source/provenance differs from checkpoint')
 device=torch.device('cuda'); set_reproducible_seed(a.seed,deterministic=True)
 kwargs={k:config[k] for k in ['d_model','pair_dim','num_heads','num_layers','dropout','marginal_pooling','marginal_top_k','marginal_temperature','score_top_k','score_pooling','score_temperature','score_top_fraction','center_partner_score','cross_attention_gate_initial_logit']}
 kwargs['antigen_input_dim']=int(config.get('antigen_input_dim',config['input_dim'])); kwargs['antibody_input_dim']=int(config.get('antibody_input_dim',config['input_dim'])); kwargs['pair_decoder']=config.get('pair_decoder','bilinear')
 model=PartnerSpecificInteractionModel(**kwargs).to(device); model.load_state_dict(checkpoint['model_state_dict'],strict=True); model.eval()
 original={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
 feature_cache={}
 def feature(key):
  if key not in feature_cache:
   chunks,starts,length=store.load(key,device); feature_cache[key]=merge_chunk_embeddings(chunks,starts,length,device)
  return feature_cache[key]
 # Preload every unique validation chain once and verify lengths without contacts.
 for iid in validation:
  row=frame.loc[iid]
  if len(feature(str(row.ab_sequence_sha256)))!=int(row.ab_length) or len(feature(str(row.ag_sequence_sha256)))!=int(row.ag_length): raise ValueError('full sequence feature length mismatch')
 known={(str(r.ab_sequence_sha256),str(r.ag_sequence_sha256)) for _,r in frame.iterrows()}
 records=[]; anchor_summaries=[]
 @torch.inference_mode()
 def score_many(fixed,candidates,swap):
  values=[]
  for start in range(0,len(candidates),a.batch_size):
   group=candidates[start:start+a.batch_size]
   moving,moving_mask=pad_batch([feature(x) for x in group],device)
   fixed_batch=fixed.unsqueeze(0).expand(len(group),-1,-1); fixed_mask=torch.ones(fixed_batch.shape[:2],dtype=torch.bool,device=device)
   if swap=='antigen': out=model(moving,fixed_batch,moving_mask,fixed_mask)
   else: out=model(fixed_batch,moving,fixed_mask,moving_mask)
   values.extend(float(x) for x in out['complex_score'].detach().cpu())
  return values
 for position,iid in enumerate(anchors,1):
  row=frame.loc[iid]; abkey=str(row.ab_sequence_sha256); agkey=str(row.ag_sequence_sha256)
  for swap in ['antigen','antibody']:
   if swap=='antigen': fixed=feature(abkey); candidate_keys=[str(frame.loc[j].ag_sequence_sha256) for j in validation]
   else: fixed=feature(agkey); candidate_keys=[str(frame.loc[j].ab_sequence_sha256) for j in validation]
   scores=score_many(fixed,candidate_keys,swap); positives=[]
   for candidate_id,key,score in zip(validation,candidate_keys,scores):
    pair=(abkey,key) if swap=='antigen' else (key,agkey); positive=pair in known; positives.append(positive)
    records.append({'anchor_instance_id':iid,'candidate_instance_id':candidate_id,'swap':swap,'score':score,'known_complex':positive,
      'anchor_ab_sequence_sha256':abkey,'anchor_ag_sequence_sha256':agkey,'candidate_sequence_sha256':key})
   mrr=expected_first_positive_reciprocal_rank(scores,positives)
   order=sorted(range(len(scores)),key=lambda n:(-scores[n],validation[n])); ranks=[rank+1 for rank,n in enumerate(order) if positives[n]]
   anchor_summaries.append({'anchor_instance_id':iid,'swap':swap,'known_positive_count':sum(positives),'candidate_count':len(scores),
    'mrr_tie_expected':mrr,'recall_at_1':float(any(r<=1 for r in ranks)),'recall_at_5':float(any(r<=5 for r in ranks)),
    'best_known_rank_id_tiebroken_for_display_only':min(ranks) if ranks else None})
  if position%5==0 or position==len(anchors): print(f'all-vs-all seed={a.seed} shard={a.shard_index} {position}/{len(anchors)}',flush=True)
 pd.DataFrame(records).to_parquet(output/'scores.parquet',index=False)
 for k,v in model.state_dict().items():
  if not torch.equal(v.cpu(),original[k]): raise RuntimeError('model mutated')
 receipt={'status':'PASS_FULL_VALIDATION_ALL_VS_ALL','format_version':'partner_all_vs_all_v1','seed':a.seed,'shard_index':a.shard_index,'shard_count':a.shard_count,
  'anchor_count':len(anchors),'candidate_count_per_direction':len(validation),'score_row_count':len(records),'directions':['antigen','antibody'],
  'anchors':anchors,'summaries':anchor_summaries,'checkpoint':str(a.checkpoint.resolve()),'checkpoint_sha256':sha256_file(a.checkpoint),
  'split_manifest_sha256':sha256_file(split_path),'candidate_semantics':'known complexes are positives; every other combination is an unlabelled distractor, not a confirmed non-binder',
  'residue_policy':'full sequence features; no contact artifact or contact labels loaded','test_contact_labels_loaded':False,'validation_contact_labels_loaded':False,
  'known_pair_metadata_scope':'all prepared complex identities used only to identify/exclude known positives','representations':observed,
  'artifacts':{'scores':'scores.parquet','scores_sha256':sha256_file(output/'scores.parquet')},'peak_cuda_bytes':torch.cuda.max_memory_allocated()}
 atomic_json(output/'RECEIPT.json',receipt); print(json.dumps({k:receipt[k] for k in ['status','anchor_count','score_row_count','peak_cuda_bytes']}))

def parser():
 p=argparse.ArgumentParser(); p.add_argument('--prepare-dir',type=Path,required=True); p.add_argument('--split-manifest',type=Path,required=True); p.add_argument('--canonical-manifest',type=Path,required=True); p.add_argument('--embedding-dir',type=Path,required=True); p.add_argument('--checkpoint',type=Path,required=True); p.add_argument('--seed',type=int,required=True); p.add_argument('--esm-if1-root',type=Path); p.add_argument('--esm-if1-dim',type=int,default=512); p.add_argument('--generic-representation-source',type=Path,action='append',default=[]); p.add_argument('--shard-index',type=int,required=True); p.add_argument('--max-anchors',type=int); p.add_argument('--shard-count',type=int,required=True); p.add_argument('--batch-size',type=int,default=8); p.add_argument('--embedding-cache-gib',type=float,default=4.0); p.add_argument('--output',type=Path,required=True); return p
if __name__=='__main__': main()
