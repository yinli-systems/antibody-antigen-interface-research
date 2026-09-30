#!/usr/bin/env python3
"""Train fixed strict partner-retrieval models from fold-local caches only."""
from __future__ import annotations
import argparse,hashlib,json,math,os,random,time
from pathlib import Path
import numpy as np
from data import TensorStore,load_sequence_cache,pair_tuples,deterministic_pair_batches,positive_mask,file_sha
from models import GlobalDualEncoder,CDRLateInteraction,PairAwareReranker,multi_positive_loss,balanced_contact_bce,parameter_signature,parameter_count

def strict_load(path:Path):
 return json.loads(path.read_text())
def plan(path:Path,expected:str):
 if file_sha(path)!=expected:raise ValueError("run-plan SHA mismatch")
 p=strict_load(path)
 if p.get("status")!="AUTHORIZED_RUN_PLAN" or p.get("test_access") is not False or p.get("ood_access") is not False:raise ValueError("run plan authority")
 if not p.get("training_authorized"):raise PermissionError("Gate0 training not authorized")
 return p
def set_seed(seed):
 import torch
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
 os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8";torch.use_deterministic_algorithms(True,warn_only=False)
def save_new(torch,path,obj):
 path.parent.mkdir(parents=True,exist_ok=True)
 if path.exists():raise FileExistsError(path)
 torch.save(obj,path)
def model_dims(store):
 a=next(k[1] for k in store.entries if k[0]=="antibody");g=next(k[1] for k in store.entries if k[0]=="antigen")
 return int(store.load("antibody",a)["embedding"].shape[1]),int(store.load("antigen",g)["embedding"].shape[1])
def sym_loss(score,pos):
 return .5*(multi_positive_loss(score,pos)+multi_positive_loss(score.T,pos.T))
def cosine_lr(step,total,warm):
 if step<warm:return (step+1)/max(warm,1)
 x=(step-warm)/max(total-warm,1);return .5*(1+math.cos(math.pi*min(max(x,0.),1.)))
def optimizer(model,lr,wd):
 import torch
 return torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=wd,betas=(.9,.98),eps=1e-8)
def batch_score(model_name,model,store,batch,pairset,device):
 import torch
 aa=sorted({a for a,g,p in batch});gg=sorted({g for a,g,p in batch})
 ab,am,cm=store.pad("antibody",aa,with_cdr=True);ag,gm=store.pad("antigen",gg)
 pos=positive_mask(torch,aa,gg,pairset,device)
 if not bool(pos.any(1).all()) or not bool(pos.any(0).all()):raise ValueError("batch has zero-positive axis")
 if model_name=="D0":score=model(ab,ag,am,gm)
 elif model_name=="L1":score=model.score_matrix(ab,ag,am,gm,cm)
 else:raise ValueError(model_name)
 return score,pos,len(aa)+len(gg)
def base_train(args,p):
 import torch
 device=torch.device(args.device);set_seed(args.seed)
 cache=load_sequence_cache(args.train_sequences);store=TensorStore(args.tensor_root,"training_only",device)
 if cache["fold"]!=store.receipt["fold"] or not store.receipt.get("training_and_model_selection_authorized"):raise PermissionError("tensor cache not training-authorized")
 pairs=pair_tuples(cache);pairset={(a,g) for a,g,_ in pairs};ad,gd=model_dims(store)
 model=GlobalDualEncoder(ad,gd,128,.07) if args.model=="D0" else CDRLateInteraction(ad,gd,128,.1,.07)
 model.to(device);epochs=58;batch_size=int(p["models"][args.model]["batch_pairs"]);lr=float(p["models"][args.model]["lr"]);wd=float(p["models"][args.model]["weight_decay"])
 batches_per_epoch=math.ceil(len(pairs)/batch_size);total=epochs*batches_per_epoch;warm=max(1,round(.05*total));opt=optimizer(model,lr,wd)
 history=[];step=0
 for epoch in range(1,epochs+1):
  loss_sum=0.;anchors=0
  for batch in deterministic_pair_batches(pairs,batch_size,args.seed,epoch):
   opt.zero_grad(set_to_none=True);score,pos,n=batch_score(args.model,model,store,batch,pairset,device);loss=sym_loss(score/.07,pos)
   if not bool(torch.isfinite(loss)):raise FloatingPointError("nonfinite loss")
   loss.backward();gn=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
   if not bool(torch.isfinite(gn)):raise FloatingPointError("nonfinite grad")
   opt.step();step+=1;scale=cosine_lr(step,total,warm)
   for group in opt.param_groups:group["lr"]=lr*scale
   loss_sum+=float(loss.detach())*n;anchors+=n
  history.append({"epoch":epoch,"loss":loss_sum/max(anchors,1),"optimizer_steps":step,"lr":opt.param_groups[0]["lr"]})
  print(json.dumps({"model":args.model,**history[-1]}),flush=True)
 out=args.output;out.mkdir(parents=True,exist_ok=False)
 ck={"format_version":"strict_partner_retrieval_checkpoint_v1","model":args.model,"fold":cache["fold"],"seed":args.seed,"state_dict":model.state_dict(),
     "parameter_signature":parameter_signature(model),"parameter_count":parameter_count(model),"train_sequence_cache_sha256":file_sha(args.train_sequences),
     "tensor_cache_receipt_sha256":file_sha(args.tensor_root/"CACHE_RECEIPT.json"),"run_plan_sha256":args.plan_sha256,
     "epochs":epochs,"optimizer_steps":step,"history":history,"checkpoint_selection":"fixed_last_epoch_no_validation"}
 save_new(torch,out/"last.pt",ck)
 (out/"COMPLETE.json").write_text(json.dumps({"status":"COMPLETE_FIXED_TRAINING","model":args.model,"fold":cache["fold"],"seed":args.seed,
   "checkpoint_sha256":file_sha(out/"last.pt"),"optimizer_steps":step,"test_access":False,"ood_access":False,"validation_scores_opened":False},indent=2,sort_keys=True)+"\n")

def load_checkpoint(torch,path,device):
 c=torch.load(path,map_location=device,weights_only=False)
 if c.get("format_version")!="strict_partner_retrieval_checkpoint_v1":raise ValueError("checkpoint")
 return c

def coarse_train_scores(args,p):
 import torch
 device=torch.device(args.device);cache=load_sequence_cache(args.train_sequences);store=TensorStore(args.tensor_root,"training_only",device)
 ck=load_checkpoint(torch,args.d0_checkpoint,device);ad,gd=model_dims(store);m=GlobalDualEncoder(ad,gd,128,.07).to(device);m.load_state_dict(ck["state_dict"],strict=True);m.eval()
 aa=sorted(cache["antibodies"]);gg=sorted(cache["antigens"])
 def encode(side,ids,bs=64):
  z=[]
  with torch.inference_mode():
   for i in range(0,len(ids),bs):
    batch=ids[i:i+bs]
    if side=="antibody":
     x,mask=store.pad(side,batch);dummy=torch.empty((len(batch),1,gd),device=device);dm=torch.ones((len(batch),1),dtype=torch.bool,device=device)
     a,_=m.encode(x,dummy,mask,dm);z.append(a.cpu())
    else:
     x,mask=store.pad(side,batch);dummy=torch.empty((len(batch),1,ad),device=device);dm=torch.ones((len(batch),1),dtype=torch.bool,device=device)
     _,g=m.encode(dummy,x,dm,mask);z.append(g.cpu())
  return torch.cat(z)
 A=encode("antibody",aa);G=encode("antigen",gg);score=(A@G.T/m.log_temperature.exp().detach().cpu().clamp(1e-3,1.0)).numpy().astype(np.float32)
 args.output.parent.mkdir(parents=True,exist_ok=True)
 if args.output.exists():raise FileExistsError(args.output)
 np.savez_compressed(args.output,scores=score,antibody_ids=np.asarray(aa),antigen_ids=np.asarray(gg),d0_checkpoint_sha256=np.asarray(file_sha(args.d0_checkpoint)))
 print(json.dumps({"status":"COMPLETE_TRAIN_COARSE_SCORES","shape":score.shape,"output":str(args.output)}))

class ContactStore:
 def __init__(self,root:Path,project_root:Path):
  import sys
  sys.path.insert(0,str((project_root/"src").resolve()));from interaction_first.contacts import load_contact_map
  self.load_contact_map=load_contact_map;self.root=root;self.receipt=strict_load(root/"CONTACT_CACHE_RECEIPT.json");self.by_pair={}
  for x in self.receipt["entries"]:self.by_pair.setdefault(x["pair_id"],[]).append(x)
 def pair_loss(self,logits,pair_id):
  import torch
  vals=[]
  for e in self.by_pair.get(pair_id,[]):
   cm=self.load_contact_map(self.root/e["relative_path"],e["instance_id"])
   abi=torch.as_tensor(cm.antibody_indices[cm.pair_antibody_local],dtype=torch.long,device=logits.device)
   agi=torch.as_tensor(cm.antigen_indices[cm.pair_antigen_local],dtype=torch.long,device=logits.device)
   y=torch.as_tensor(cm.pair_labels,dtype=torch.bool,device=logits.device);z=logits[abi,agi]
   vals.append(balanced_contact_bce(z,y,torch.ones_like(y,dtype=torch.bool)))
  return torch.stack(vals).mean() if vals else logits.sum()*0.

def candidate_lists(cache,coarse):
 aa=list(map(str,coarse["antibody_ids"]));gg=list(map(str,coarse["antigen_ids"]));S=np.asarray(coarse["scores"],float)
 if set(aa)!=set(cache["antibodies"]) or set(gg)!=set(cache["antigens"]):raise ValueError("coarse inventory")
 pairset={(a,g) for a,g,p in pair_tuples(cache)};out=[]
 for i,a in enumerate(aa):
  top=[gg[j] for j in np.lexsort((np.asarray(gg),-S[i]))[:min(64,len(gg))]];pos=sorted(g for x,g in pairset if x==a);c=sorted(set(top)|set(pos))
  out.append(("ab",a,c,pos))
 for j,g in enumerate(gg):
  top=[aa[i] for i in np.lexsort((np.asarray(aa),-S[:,j]))[:min(64,len(aa))]];pos=sorted(a for a,x in pairset if x==g);c=sorted(set(top)|set(pos))
  out.append(("ag",g,c,pos))
 return aa,gg,S,out

def pair_batch_rerank(model,store,direction,query,candidates,coarse,aa_index,gg_index,device,micro=8):
 import torch
 values=[];contacts=[]
 for start in range(0,len(candidates),micro):
  cs=candidates[start:start+micro]
  if direction=="ab":
   abids=[query]*len(cs);gids=cs;cv=torch.tensor([coarse[aa_index[query],gg_index[g]] for g in cs],dtype=torch.float32,device=device)
  else:
   abids=cs;gids=[query]*len(cs);cv=torch.tensor([coarse[aa_index[a],gg_index[query]] for a in cs],dtype=torch.float32,device=device)
  ab,am=store.pad("antibody",abids);ag,gm=store.pad("antigen",gids);o=model(ab,ag,am,gm,cv);values.append(o["retrieval_score"]);contacts.extend([z for z in o["contact_logits"]])
 return torch.cat(values),contacts

def reranker_train(args,p):
 import torch
 device=torch.device(args.device);set_seed(args.seed);cache=load_sequence_cache(args.train_sequences);store=TensorStore(args.tensor_root,"training_only",device);contact=ContactStore(args.contact_root,args.project_root)
 coarse=np.load(args.coarse_scores,allow_pickle=False);aa,gg,S,queries=candidate_lists(cache,coarse);ai={x:i for i,x in enumerate(aa)};gi={x:i for i,x in enumerate(gg)}
 ad,gd=model_dims(store);model=PairAwareReranker(ad,gd,256,8,512,2,.1,64).to(device);weight=0.0 if args.model=="R1" else .10
 epochs=40;lr=1e-4;opt=optimizer(model,lr,.01);total=epochs*len(queries);warm=max(1,round(.05*total));step=0;history=[];pairid={(a,g):pid for a,g,pid in pair_tuples(cache)}
 for epoch in range(1,epochs+1):
  order=list(range(len(queries)));random.Random((args.seed<<32)^epoch).shuffle(order);rs=cs=0.
  for qi in order:
   direction,q,cands,pos=queries[qi];opt.zero_grad(set_to_none=True);scores,clogs=pair_batch_rerank(model,store,direction,q,cands,S,ai,gi,device)
   pm=torch.tensor([x in set(pos) for x in cands],dtype=torch.bool,device=device)
   retrieval=-(torch.logsumexp((scores/.07).masked_fill(~pm,torch.finfo(scores.dtype).min),0)-torch.logsumexp(scores/.07,0))
   cvals=[]
   for cand,logit,ispos in zip(cands,clogs,pm.tolist()):
    if ispos:
     pair=(q,cand) if direction=="ab" else (cand,q);cvals.append(contact.pair_loss(logit,pairid[pair]))
   con=torch.stack(cvals).mean() if cvals else retrieval*0.
   loss=retrieval+weight*con
   if not bool(torch.isfinite(loss)):raise FloatingPointError("nonfinite reranker loss")
   loss.backward();gn=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
   if not bool(torch.isfinite(gn)):raise FloatingPointError("nonfinite grad")
   opt.step();step+=1;scale=cosine_lr(step,total,warm)
   for group in opt.param_groups:group["lr"]=lr*scale
   rs+=float(retrieval.detach());cs+=float(con.detach())
  history.append({"epoch":epoch,"retrieval_loss":rs/len(queries),"contact_loss_computed":cs/len(queries),"contact_loss_weight":weight,"optimizer_steps":step})
  print(json.dumps({"model":args.model,**history[-1]}),flush=True)
 out=args.output;out.mkdir(parents=True,exist_ok=False)
 ck={"format_version":"strict_partner_retrieval_checkpoint_v1","model":args.model,"fold":cache["fold"],"seed":args.seed,"state_dict":model.state_dict(),
     "parameter_signature":parameter_signature(model),"parameter_count":parameter_count(model),"contact_loss_weight":weight,
     "d0_checkpoint_sha256":file_sha(args.d0_checkpoint),"coarse_scores_sha256":file_sha(args.coarse_scores),"contact_cache_receipt_sha256":file_sha(args.contact_root/"CONTACT_CACHE_RECEIPT.json"),
     "train_sequence_cache_sha256":file_sha(args.train_sequences),"tensor_cache_receipt_sha256":file_sha(args.tensor_root/"CACHE_RECEIPT.json"),
     "run_plan_sha256":args.plan_sha256,"epochs":epochs,"optimizer_steps":step,"history":history,"checkpoint_selection":"fixed_last_epoch_no_validation"}
 save_new(torch,out/"last.pt",ck)
 (out/"COMPLETE.json").write_text(json.dumps({"status":"COMPLETE_FIXED_TRAINING","model":args.model,"fold":cache["fold"],"seed":args.seed,"checkpoint_sha256":file_sha(out/"last.pt"),"optimizer_steps":step,"contact_loss_weight":weight,"test_access":False,"ood_access":False,"validation_scores_opened":False},indent=2,sort_keys=True)+"\n")

def main():
 ap=argparse.ArgumentParser();sp=ap.add_subparsers(dest="cmd",required=True)
 def common(x):
  x.add_argument("--plan",type=Path,required=True);x.add_argument("--plan-sha256",required=True);x.add_argument("--train-sequences",type=Path,required=True);x.add_argument("--tensor-root",type=Path,required=True);x.add_argument("--seed",type=int,choices=(17,42,73),required=True);x.add_argument("--device",default="cuda")
 b=sp.add_parser("train-base");common(b);b.add_argument("--model",choices=("D0","L1"),required=True);b.add_argument("--output",type=Path,required=True)
 c=sp.add_parser("coarse-train-scores");common(c);c.add_argument("--d0-checkpoint",type=Path,required=True);c.add_argument("--output",type=Path,required=True)
 r=sp.add_parser("train-reranker");common(r);r.add_argument("--model",choices=("R1","R2"),required=True);r.add_argument("--d0-checkpoint",type=Path,required=True);r.add_argument("--coarse-scores",type=Path,required=True);r.add_argument("--contact-root",type=Path,required=True);r.add_argument("--project-root",type=Path,required=True);r.add_argument("--output",type=Path,required=True)
 a=ap.parse_args();p=plan(a.plan,a.plan_sha256)
 if int(a.seed) not in p["seeds"]:raise ValueError("seed not in plan")
 if a.cmd=="train-base":base_train(a,p)
 elif a.cmd=="coarse-train-scores":coarse_train_scores(a,p)
 else:reranker_train(a,p)
if __name__=="__main__":main()
