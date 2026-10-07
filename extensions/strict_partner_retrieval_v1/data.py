"""Fold-local tensor data helpers; model processes never receive global bank paths."""
from __future__ import annotations
import hashlib,json,random
from pathlib import Path

def file_sha(path:Path)->str:
 h=hashlib.sha256()
 with path.open("rb") as f:
  for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
 return h.hexdigest()

class TensorStore:
 def __init__(self,root:Path,required_role:str,device):
  import torch
  self.torch=torch;self.root=root.resolve();self.device=device
  r=json.loads((self.root/"CACHE_RECEIPT.json").read_text())
  if r.get("status")!="COMPLETE_FOLD_LOCAL_TENSOR_CACHE" or r.get("role")!=required_role:raise ValueError("tensor cache role/status")
  self.receipt=r;self.entries={(x["side"],x["entity_id"]):x for x in r["entries"]}
  if len(self.entries)!=len(r["entries"]):raise ValueError("duplicate tensor entry")
  self.cache={}
 def load(self,side:str,entity:str):
  key=(side,entity)
  if key in self.cache:return self.cache[key]
  e=self.entries.get(key)
  if e is None:raise KeyError(key)
  p=self.root/e["relative_path"]
  if file_sha(p)!=e["sha256"]:raise ValueError("tensor artifact hash drift")
  a=self.torch.load(p,map_location="cpu",weights_only=False)
  if a.get("format_version")!="strict_partner_retrieval_embedding_v1" or a.get("entity_id")!=entity or a.get("side")!=side or a.get("role")!=self.receipt["role"]:raise ValueError("tensor artifact identity")
  x=a["embedding"].float()
  if x.ndim!=2 or not bool(self.torch.isfinite(x).all()):raise ValueError("bad tensor")
  out={"embedding":x.to(self.device),"mask":a["mask"].bool().to(self.device)}
  if side=="antibody":
   out["cdr_masks"]=a["cdr_masks"].bool().to(self.device)
   out["vh_length"]=int(a["vh_length"]);out["vl_length"]=int(a["vl_length"])
  self.cache[key]=out;return out

 def pad(self,side,ids,with_cdr=False):
  torch=self.torch;items=[self.load(side,x) for x in ids];D={int(x["embedding"].shape[1]) for x in items}
  if len(D)!=1:raise ValueError("mixed embedding dims")
  L=max(int(x["embedding"].shape[0]) for x in items);d=D.pop()
  z=torch.zeros((len(items),L,d),dtype=torch.float32,device=self.device);m=torch.zeros((len(items),L),dtype=torch.bool,device=self.device)
  c=torch.zeros((len(items),6,L),dtype=torch.bool,device=self.device) if with_cdr else None
  for i,x in enumerate(items):
   n=len(x["embedding"]);z[i,:n]=x["embedding"];m[i,:n]=x["mask"]
   if with_cdr:c[i,:,:n]=x["cdr_masks"]
  return (z,m,c) if with_cdr else (z,m)

def load_sequence_cache(path:Path,expected_role="training_only"):
 d=json.loads(path.read_text())
 if d.get("role")!=expected_role:raise ValueError("sequence cache role")
 if d.get("test_access") is not False or d.get("ood_access") is not False:raise ValueError("sealed access")
 return d

def pair_tuples(cache):
 return [(str(x["antibody_id"]),str(x["antigen_id"]),str(x["pair_id"])) for x in cache["pairs"]]

def deterministic_pair_batches(pairs,batch_size,seed,epoch):
 x=list(pairs);r=random.Random((int(seed)<<32)^int(epoch));r.shuffle(x)
 return [x[i:i+batch_size] for i in range(0,len(x),batch_size)]

def positive_mask(torch,antibodies,antigens,pairset,device):
 return torch.tensor([[(a,g) in pairset for g in antigens] for a in antibodies],dtype=torch.bool,device=device)

def source_pair_map(cache):
 out={}
 for p in cache["pairs"]:
  for iid in p["source_instance_ids"]:
   if iid in out:raise ValueError("source row reused across exact pairs")
   out[str(iid)]=str(p["pair_id"])
 return out
