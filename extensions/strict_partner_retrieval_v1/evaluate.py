#!/usr/bin/env python3
"""Evaluate frozen retrieval checkpoints on one pre-frozen library."""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
from data import TensorStore,load_sequence_cache,file_sha
from metrics import bidirectional_report,score_matrix_sha256
from models import GlobalDualEncoder,CDRLateInteraction,PairAwareReranker,parameter_signature

def load_ck(torch,p,device):
 x=torch.load(p,map_location=device,weights_only=False)
 if x.get("format_version")!="strict_partner_retrieval_checkpoint_v1":raise ValueError("checkpoint format")
 return x
def dims(store):
 a=next(k[1] for k in store.entries if k[0]=="antibody");g=next(k[1] for k in store.entries if k[0]=="antigen")
 return store.load("antibody",a)["embedding"].shape[1],store.load("antigen",g)["embedding"].shape[1]
def authorized(cache,role,outer_authority,expected_outer_sha):
 if role=="inner":
  if not cache.get("training_and_model_selection_authorized"):raise PermissionError("inner model-selection authority absent")
 else:
  if outer_authority is None or expected_outer_sha is None:raise PermissionError("outer evaluation requires frozen authority")
  if file_sha(outer_authority)!=expected_outer_sha:raise PermissionError("outer authority file hash")
  a=json.loads(outer_authority.read_text())
  if a.get("format_version")!="strict_partner_retrieval_outer_authority_v1" or a.get("status")!="AUTHORIZED_OUTER_EVALUATION" or a.get("outer_evaluation_authorized") is not True or a.get("test_access") is not False or a.get("ood_access") is not False:raise PermissionError("invalid outer authority")
def base_scores(name,ck,store,cache,device,chunk=32):
 import torch
 ad,gd=dims(store)
 if name=="D0":
  m=GlobalDualEncoder(ad,gd,128,.07).to(device);m.load_state_dict(ck["state_dict"],strict=True);m.eval()
  aa=sorted(cache["antibodies"]);gg=sorted(cache["antigens"]);A=[];G=[]
  with torch.inference_mode():
   for i in range(0,len(aa),chunk):
    ids=aa[i:i+chunk];x,mask=store.pad("antibody",ids);dummy=torch.zeros((len(ids),1,gd),device=device);dm=torch.ones((len(ids),1),dtype=torch.bool,device=device);a,_=m.encode(x,dummy,mask,dm);A.append(a.cpu())
   for i in range(0,len(gg),chunk):
    ids=gg[i:i+chunk];x,mask=store.pad("antigen",ids);dummy=torch.zeros((len(ids),1,ad),device=device);dm=torch.ones((len(ids),1),dtype=torch.bool,device=device);_,g=m.encode(dummy,x,dm,mask);G.append(g.cpu())
  A=torch.cat(A);G=torch.cat(G);S=(A@G.T/m.log_temperature.exp().detach().cpu().clamp(1e-3,1.)).numpy()
  return aa,gg,S.astype(np.float32)
 if name=="L1":
  m=CDRLateInteraction(ad,gd,128,.1,.07).to(device);m.load_state_dict(ck["state_dict"],strict=True);m.eval();aa=sorted(cache["antibodies"]);gg=sorted(cache["antigens"]);S=np.empty((len(aa),len(gg)),np.float32)
  with torch.inference_mode():
   for ai in range(0,len(aa),8):
    a_ids=aa[ai:ai+8];ab,am,cm=store.pad("antibody",a_ids,True)
    for gi in range(0,len(gg),16):
     g_ids=gg[gi:gi+16];ag,gm=store.pad("antigen",g_ids);S[ai:ai+len(a_ids),gi:gi+len(g_ids)]=m.score_matrix(ab,ag,am,gm,cm).cpu().numpy()
  return aa,gg,S
 raise ValueError(name)
def rerank_scores(ck,d0ck,store,cache,device):
 import torch
 aa,gg,S=base_scores("D0",d0ck,store,cache,device);ad,gd=dims(store);m=PairAwareReranker(ad,gd,256,8,512,2,.1,64).to(device);m.load_state_dict(ck["state_dict"],strict=True);m.eval()
 union=set()
 for i,a in enumerate(aa):
  for j in np.lexsort((np.asarray(gg),-S[i]))[:min(64,len(gg))]:union.add((i,int(j)))
 for j,g in enumerate(gg):
  for i in np.lexsort((np.asarray(aa),-S[:,j]))[:min(64,len(aa))]:union.add((int(i),j))
 pairs=sorted(union);out=S.copy()
 with torch.inference_mode():
  for st in range(0,len(pairs),8):
   q=pairs[st:st+8];aids=[aa[i] for i,j in q];gids=[gg[j] for i,j in q];ab,am=store.pad("antibody",aids);ag,gm=store.pad("antigen",gids);coarse=torch.tensor([S[i,j] for i,j in q],dtype=torch.float32,device=device);z=m(ab,ag,am,gm,coarse)["retrieval_score"].cpu().numpy()
   for (i,j),v in zip(q,z):out[i,j]=v
 return aa,gg,out,{"reranked_pair_count":len(pairs),"total_pair_count":len(aa)*len(gg),"coarse_checkpoint_sha256":file_sha(Path(d0ck["_path"]))}
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--model",choices=("D0","L1","R1","R2"),required=True);ap.add_argument("--checkpoint",type=Path,required=True);ap.add_argument("--d0-checkpoint",type=Path);ap.add_argument("--eval-sequences",type=Path,required=True);ap.add_argument("--tensor-root",type=Path,required=True);ap.add_argument("--role",choices=("inner","outer"),required=True);ap.add_argument("--outer-authority",type=Path);ap.add_argument("--expected-outer-authority-sha256");ap.add_argument("--device",default="cuda");ap.add_argument("--output",type=Path,required=True)
 a=ap.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 import torch
 cache=load_sequence_cache(a.eval_sequences,"evaluation_only");authorized(cache,a.role,a.outer_authority,a.expected_outer_authority_sha256);device=torch.device(a.device);store=TensorStore(a.tensor_root,"evaluation_only",device);ck=load_ck(torch,a.checkpoint,device)
 if ck["model"]!=a.model:raise ValueError("model name checkpoint mismatch")
 if a.model in ("D0","L1"):aa,gg,S=base_scores(a.model,ck,store,cache,device);extra={}
 else:
  if a.d0_checkpoint is None:raise ValueError("reranker requires D0 checkpoint")
  d0=load_ck(torch,a.d0_checkpoint,device);d0["_path"]=str(a.d0_checkpoint);aa,gg,S,extra=rerank_scores(ck,d0,store,cache,device)
 pp=[(x["antibody_id"],x["antigen_id"]) for x in cache["pairs"]];report=bidirectional_report(S,aa,gg,pp)
 a.output.mkdir(parents=True);np.savez_compressed(a.output/"scores.npz",scores=S,antibody_ids=np.asarray(aa),antigen_ids=np.asarray(gg))
 result={"format_version":"strict_partner_retrieval_evaluation_v1","status":"COMPLETE_INNER_EVALUATION" if a.role=="inner" else "COMPLETE_OUTER_EVALUATION","model":a.model,"fold":cache["fold"],"seed":ck["seed"],"role":a.role,"checkpoint_sha256":file_sha(a.checkpoint),"d0_checkpoint_sha256":file_sha(a.d0_checkpoint) if a.d0_checkpoint else None,"eval_sequence_cache_sha256":file_sha(a.eval_sequences),"tensor_cache_receipt_sha256":file_sha(a.tensor_root/"CACHE_RECEIPT.json"),"score_matrix_sha256":score_matrix_sha256(S,aa,gg),"metrics":report,"rerank":extra,"nonpositive_semantics":"unlabeled_ranking_candidates","contact_metrics_computed":False,"affinity_metrics_computed":False,"test_access":False,"ood_access":False}
 (a.output/"result.json").write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n");print(json.dumps({"status":result["status"],"model":a.model,"fold":cache["fold"],"seed":ck["seed"],"bidirectional":report["bidirectional"]},indent=2))
if __name__=="__main__":main()
