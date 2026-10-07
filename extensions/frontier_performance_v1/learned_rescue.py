#!/usr/bin/env python3
"""Train and evaluate a label-free residue-correspondence rescue matcher."""
from __future__ import annotations
import argparse, csv, hashlib, inspect, json, math, os, random, sys, time
from pathlib import Path
from typing import Any

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'extensions/correspondence_v2')]
from interaction_first.runtime import EmbeddingStore,safe_torch_load,set_reproducible_seed
from interaction_first.composite_embeddings import CompositeEmbeddingStore,build_sources
from interaction_first.model import PartnerSpecificInteractionModel,merge_chunk_embeddings
from interaction_first.contacts import load_contact_map
from core import chain_permutation,crossed_cluster_bootstrap

D=Path('/ssd/scxi253/model_deployment/data')
R=Path('/ssd/scxi253/repr_extract/out')
RUNS=Path('/ssd/scxi253/repr_ablation/out')
BASE=D/'sppider_seq_sabdab2_embeddings'
PREP=D/'interaction_first_sabdab2/instances.parquet'
SPLIT=D/'interaction_first/double_remote_split.json'
MANIFEST=D/'interaction_first/canonical_manifest.csv'
CONTACTS=D/'interaction_first/contacts'
ESMIF1=R/'esm_if1'
RESCUE_SEEDS=(101,202,303)
DOWNSTREAM_SEEDS=(1,2,3)
CONDITIONS=(('same_aa_shuffle',0),('chain_shift',8),('global_shuffle',0))


def sha(path: Path)->str:
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()


def exclusive(path:Path,payload:dict):
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('x') as f:json.dump(payload,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')


def sequence_metadata(path=MANIFEST):
 seqs={}; inst={}
 with path.open(newline='') as f:
  for row in csv.DictReader(f):
   h,l,a=[(row.get(k) or '').strip() for k in ('vh_seq','vl_seq','ag_seq')]
   if not h or not a:continue
   for parts in (tuple(x for x in (h,l) if x),(a,)):
    key=hashlib.sha256(''.join(parts).encode('ascii')).hexdigest()
    if key in seqs and seqs[key]!=parts:raise ValueError('ambiguous chain boundary')
    seqs[key]=parts
   inst[str(row['INSTANCE'])]=(hashlib.sha256((h+l).encode()).hexdigest(),hashlib.sha256(a.encode()).hexdigest())
 return seqs,inst


def unique_keys(role:str, split_data:dict, frame, instmap:dict, source)->list[str]:
 ids=split_data['splits'][role]; keys=[];seen=set()
 for iid in ids:
  if iid not in frame.index or iid not in instmap:continue
  for key in instmap[iid]:
   if key in seen or key not in source.parts_by_key:continue
   # require every ESMIF1 part to exist; rescue does not learn from zero-filled missingness
   from interaction_first.pretrained_representations import representation_path
   if any(not representation_path(source.root,p).exists() for p in source.parts_by_key[key]):continue
   seen.add(key);keys.append(key)
 return sorted(keys)


class Matcher:
 def __init__(self,base_dim=1280,extra_dim=512,hidden=128):
  import torch
  self.model=torch.nn.ModuleDict({'q':torch.nn.Sequential(torch.nn.Linear(base_dim,256),torch.nn.GELU(),torch.nn.Linear(256,hidden)),
                                  'k':torch.nn.Sequential(torch.nn.Linear(extra_dim,256),torch.nn.GELU(),torch.nn.Linear(256,hidden))})
  self.temperature=torch.nn.Parameter(torch.tensor(math.log(1/0.07)))
  self.model['temperature_holder']=torch.nn.Identity()
 def parameters(self):return list(self.model.parameters())+[self.temperature]
 def state_dict(self):return {'model':self.model.state_dict(),'temperature':self.temperature.detach().cpu()}
 def load_state_dict(self,d):self.model.load_state_dict(d['model']);self.temperature.data.copy_(d['temperature'].to(self.temperature.device))
 def to(self,device):self.model.to(device);self.temperature.data=self.temperature.data.to(device);return self
 def train(self):self.model.train()
 def eval(self):self.model.eval()
 def logits(self,base,extra):
  import torch
  q=torch.nn.functional.normalize(self.model['q'](base),dim=-1);k=torch.nn.functional.normalize(self.model['k'](extra),dim=-1)
  scale=self.temperature.exp().clamp(max=100.0)
  return q@k.T*scale


def load_full_features(key, base_store, source, device):
 import torch
 chunks,starts,length=base_store.load(key,device);base=merge_chunk_embeddings(chunks,starts,length,device)
 extra=source.features(key,length,device,base)
 if extra is None:raise ValueError('missing ESMIF1 feature '+key)
 if extra.shape!=(length,512):raise ValueError(f'ESMIF1 shape mismatch {extra.shape}')
 return base,extra


def chain_slices(parts):
 out=[];start=0
 for seq in parts:out.append(slice(start,start+len(seq)));start+=len(seq)
 return out


def synthetic_order(parts,mode,k,seed,key):
 return chain_permutation(parts,mode,k,seed,key)[0]


def loss_for_key(matcher,base,extra,parts,mode,k,seed,key):
 import torch
 order=synthetic_order(parts,mode,k,seed,key)
 total=base.new_tensor(0.);tokens=0
 for sl in chain_slices(parts):
  idx=torch.as_tensor(order[sl]-sl.start,device=base.device,dtype=torch.long)
  corrupted=extra[sl].index_select(0,idx)
  logits=matcher.logits(base[sl],corrupted)
  inv=torch.argsort(idx)
  row=torch.nn.functional.cross_entropy(logits,inv)
  col=torch.nn.functional.cross_entropy(logits.T,idx)
  total+=(row+col)*0.5*len(idx);tokens+=len(idx)
 return total/max(tokens,1)


def matching_accuracy(matcher,base,extra,parts,mode,k,seed,key):
 import torch
 hit=tot=0
 with torch.inference_mode():
  order=synthetic_order(parts,mode,k,seed,key)
  for sl in chain_slices(parts):
   idx=torch.as_tensor(order[sl]-sl.start,device=base.device,dtype=torch.long)
   corrupted=extra[sl].index_select(0,idx)
   logits=matcher.logits(base[sl],corrupted)
   pred=logits.argmax(dim=1); truth=torch.argsort(idx)
   hit+=int((pred==truth).sum());tot+=len(idx)
 return hit/tot


def hungarian_rescue(matcher,base,corrupted,parts):
 import torch
 from scipy.optimize import linear_sum_assignment
 rescued=corrupted.clone()
 with torch.inference_mode():
  for sl in chain_slices(parts):
   logits=matcher.logits(base[sl],corrupted[sl]).detach().cpu().numpy()
   rows,cols=linear_sum_assignment(-logits)
   if not (rows==list(range(len(rows)))).all():raise AssertionError('row assignment changed')
   idx=torch.as_tensor(cols,device=base.device,dtype=torch.long)
   rescued[sl]=corrupted[sl].index_select(0,idx)
 return rescued


def freeze(args):
 import pandas as pd
 split=json.loads(SPLIT.read_text());frame=pd.read_parquet(PREP,columns=['instance_id','ab_sequence_sha256','ag_sequence_sha256']).set_index('instance_id')
 seqs,inst=sequence_metadata();source=build_sources(MANIFEST,esm_if1_root=ESMIF1)[0]
 train=unique_keys('train',split,frame,inst,source);val=unique_keys('validation',split,frame,inst,source)
 if set(train)&set(val):raise ValueError('train/validation sequence-key overlap')
 checkpoints=[]
 for seed in DOWNSTREAM_SEEDS:
  p=RUNS/f'esm_if1_only/seed_{seed}/train/best.pt';checkpoints.append({'seed':seed,'path':str(p),'sha256':sha(p)})
 pins={str(x):sha(x) for x in (SPLIT,PREP,MANIFEST)}
 for rel in ['extensions/frontier_performance_v1/learned_rescue.py','extensions/frontier_performance_v1/LEARNED_RESCUE_PROTOCOL.md','extensions/correspondence_v2/core.py','src/interaction_first/runtime.py','src/interaction_first/composite_embeddings.py','src/interaction_first/model.py','src/interaction_first/contacts.py']:
  pins[str(ROOT/rel)]=sha(ROOT/rel)
 plan={'schema':'learned_correspondence_rescue_v1','created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
       'source_commit':args.source_commit,'train_keys':train,'validation_keys':val,'rescue_seeds':list(RESCUE_SEEDS),
       'downstream_seeds':list(DOWNSTREAM_SEEDS),'conditions':[list(x) for x in CONDITIONS],
       'corruption_seed':20260930,'epochs':12,'lr':3e-4,'weight_decay':1e-4,'hidden_dim':128,
       'primary_selection_metric':'mean synthetic residue matching accuracy over all three registered corruption families on validation sequence keys',
       'contact_metric_access_during_training':False,'test_access':False,'ood_access':False,
       'input_sha256':pins,'downstream_checkpoints':checkpoints,
       'train_key_count':len(train),'validation_key_count':len(val)}
 exclusive(args.output,plan);print(json.dumps({'plan':str(args.output),'sha256':sha(args.output),'train_keys':len(train),'validation_keys':len(val)},indent=2))


def verify_plan(path,expected):
 if sha(path)!=expected:raise ValueError('plan hash mismatch')
 p=json.loads(Path(path).read_text())
 if p['test_access'] or p['ood_access'] or p['contact_metric_access_during_training']:raise ValueError('closed-data authority violation')
 for f,h in p['input_sha256'].items():
  if sha(Path(f))!=h:raise ValueError('source/input drift '+f)
 for c in p['downstream_checkpoints']:
  if sha(Path(c['path']))!=c['sha256']:raise ValueError('checkpoint drift')
 return p


def train(args):
 import torch
 plan=verify_plan(args.plan,args.plan_sha256);seed=plan['rescue_seeds'][args.index];set_reproducible_seed(seed,True)
 out=args.output/f'seed_{seed}';out.mkdir(parents=True,exist_ok=False)
 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 seqs,_=sequence_metadata();source=build_sources(MANIFEST,esm_if1_root=ESMIF1)[0];base_store=EmbeddingStore(BASE,8.0)
 matcher=Matcher(hidden=plan['hidden_dim']).to(device);opt=torch.optim.AdamW(matcher.parameters(),lr=plan['lr'],weight_decay=plan['weight_decay'])
 feature_cache={}
 def cached(key):
  if key not in feature_cache: feature_cache[key]=load_full_features(key,base_store,source,device)
  return feature_cache[key]
 best=(-1,None);history=[];modes=CONDITIONS
 rng=random.Random(seed)
 for epoch in range(1,plan['epochs']+1):
  matcher.train();keys=list(plan['train_keys']);rng.shuffle(keys);losses=[]
  for n,key in enumerate(keys):
   base,extra=cached(key);mode,k=modes[(n+epoch+args.index)%len(modes)]
   loss=loss_for_key(matcher,base,extra,seqs[key],mode,k,plan['corruption_seed']+epoch,key)
   opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(matcher.parameters(),1.0);opt.step();losses.append(float(loss.detach()))
  matcher.eval();acc=[]
  for key in plan['validation_keys']:
   base,extra=cached(key)
   per=[]
   for mode,k in modes:per.append(matching_accuracy(matcher,base,extra,seqs[key],mode,k,plan['corruption_seed'],key))
   acc.append(sum(per)/len(per))
  val=sum(acc)/len(acc);row={'epoch':epoch,'train_loss':sum(losses)/len(losses),'validation_matching_accuracy':val};history.append(row);print(json.dumps(row),flush=True)
  if val>best[0]:
   best=(val,epoch);torch.save({'state':matcher.state_dict(),'seed':seed,'epoch':epoch,'validation_matching_accuracy':val,'plan_sha256':args.plan_sha256},out/'best.pt')
 exclusive(out/'history.json',{'history':history,'selected_epoch':best[1],'best_validation_matching_accuracy':best[0],'selection_uses_contact_metrics':False,'test_access':False})
 print(json.dumps({'status':'COMPLETE_RESCUE_TRAINING','seed':seed,'best':best,'device':str(device)}))


def evaluate(args):
 import torch,pandas as pd,numpy as np
 from sklearn.metrics import average_precision_score
 plan=verify_plan(args.plan,args.plan_sha256);split=json.loads(SPLIT.read_text());ids=list(map(str,split['splits']['validation']))
 frame=pd.read_parquet(PREP,columns=['instance_id','ab_sequence_sha256','ag_sequence_sha256']).set_index('instance_id');seqs,_=sequence_metadata();source=build_sources(MANIFEST,esm_if1_root=ESMIF1)[0]
 records=[]
 for idx,(rseed,dseed) in enumerate(zip(plan['rescue_seeds'],plan['downstream_seeds'])):
  rckpt=Path(args.rescue_root)/f'seed_{rseed}/best.pt';rc=torch.load(rckpt,map_location='cpu',weights_only=False)
  matcher=Matcher(hidden=plan['hidden_dim']);matcher.load_state_dict(rc['state']);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');matcher.to(device);matcher.eval()
  downstream=next(x for x in plan['downstream_checkpoints'] if x['seed']==dseed);ck=safe_torch_load(Path(downstream['path']),map_location=device);config=ck['config'];kwargs={name:config[name] for name in inspect.signature(PartnerSpecificInteractionModel).parameters if name in config};kwargs['antigen_input_dim']=int(config.get('antigen_input_dim',config['input_dim']));kwargs['antibody_input_dim']=int(config.get('antibody_input_dim',config['input_dim']));model=PartnerSpecificInteractionModel(**kwargs).to(device);model.load_state_dict(ck['model_state_dict'],strict=True);model.eval()
  base_store=EmbeddingStore(BASE,8.0);cache={}
  def full(key):
   if key not in cache:
    b,e=load_full_features(key,base_store,source,device);cache[key]=(b,e)
   return cache[key]
  for pos,iid in enumerate(ids):
   row=frame.loc[iid];contact=load_contact_map(CONTACTS/f'{iid}.npz',iid);labels,mask=contact.dense()
   if not bool(mask.all()) or not labels[mask].any():raise ValueError('candidate/contact contract mismatch')
   vals_by_cond={}
   for condition in [('aligned',0)]+list(CONDITIONS):
    mode,k=condition;vals=[]
    for col,indices in [('ag_sequence_sha256',contact.antigen_indices),('ab_sequence_sha256',contact.antibody_indices)]:
     key=str(row[col]);b,e=full(key);parts=seqs[key]
     if mode=='aligned':candidate=e
     else:
      order=synthetic_order(parts,mode,k,plan['corruption_seed'],key);ii=torch.as_tensor(order,device=device,dtype=torch.long);corrupt=e.index_select(0,ii);candidate=hungarian_rescue(matcher,b,corrupt,parts)
     joined=torch.cat([b,candidate,torch.ones((len(b),1),device=device)],dim=1);sel=torch.as_tensor(indices,device=device,dtype=torch.long);vals.append(joined.index_select(0,sel).unsqueeze(0))
    with torch.inference_mode():
     pred=model(vals[0],vals[1],torch.ones(vals[0].shape[:2],dtype=torch.bool,device=device),torch.ones(vals[1].shape[:2],dtype=torch.bool,device=device))['pair_probabilities'][0].cpu().numpy()[mask]
    vals_by_cond[f'{mode}:{k}']=float(average_precision_score(labels[mask].astype(np.uint8),pred))
   # compute corrupted/no-rescue separately for each non-aligned condition
   for mode,k in CONDITIONS:
    vals=[]
    for col,indices in [('ag_sequence_sha256',contact.antigen_indices),('ab_sequence_sha256',contact.antibody_indices)]:
     key=str(row[col]);b,e=full(key);order=synthetic_order(seqs[key],mode,k,plan['corruption_seed'],key);ii=torch.as_tensor(order,device=device,dtype=torch.long);corrupt=e.index_select(0,ii);joined=torch.cat([b,corrupt,torch.ones((len(b),1),device=device)],dim=1);sel=torch.as_tensor(indices,device=device,dtype=torch.long);vals.append(joined.index_select(0,sel).unsqueeze(0))
    with torch.inference_mode():pred=model(vals[0],vals[1],torch.ones(vals[0].shape[:2],dtype=torch.bool,device=device),torch.ones(vals[1].shape[:2],dtype=torch.bool,device=device))['pair_probabilities'][0].cpu().numpy()[mask]
    records.append({'instance_id':iid,'component':str(split['component_id_by_instance'][iid]),'rescue_seed':rseed,'downstream_seed':dseed,'condition':f'{mode}:{k}','aligned_ap':vals_by_cond['aligned:0'],'corrupted_ap':float(average_precision_score(labels[mask],pred)),'rescued_ap':vals_by_cond[f'{mode}:{k}']})
   if (pos+1)%25==0:print('eval',rseed,pos+1,flush=True)
 out=args.output;out.parent.mkdir(parents=True,exist_ok=True);exclusive(out,{'status':'COMPLETE_DEVELOPMENT_RESCUE_EVALUATION','records':records,'test_labels_read':False,'ood_labels_read':False,'contact_metrics_used_for_rescue_selection':False,'plan_sha256':args.plan_sha256})
 print('records',len(records))


def main():
 p=argparse.ArgumentParser();sub=p.add_subparsers(dest='cmd',required=True)
 f=sub.add_parser('freeze');f.add_argument('--source-commit',required=True);f.add_argument('--output',type=Path,required=True)
 for cmd in ('train','evaluate'):
  s=sub.add_parser(cmd);s.add_argument('--plan',type=Path,required=True);s.add_argument('--plan-sha256',required=True);s.add_argument('--output',type=Path,required=True)
  if cmd=='train':s.add_argument('--index',type=int,required=True)
  else:s.add_argument('--rescue-root',type=Path,required=True)
 a=p.parse_args();{'freeze':freeze,'train':train,'evaluate':evaluate}[a.cmd](a)
if __name__=='__main__':main()
