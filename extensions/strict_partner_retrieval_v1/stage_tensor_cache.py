#!/usr/bin/env python3
"""Stage only allowlisted residue embeddings into a fold-local model root."""
from __future__ import annotations
import argparse,hashlib,json,os,tempfile
from pathlib import Path
from cdr import annotate_antibody

OUT_FORMAT="strict_partner_retrieval_embedding_v1"
def sha(p:Path):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
 return h.hexdigest()
def seqkey(s):return hashlib.sha256(s.encode("ascii")).hexdigest()
def path(root,key):return root/key[:2]/f"{key}.pt"
def load_json(p):return json.loads(p.read_text())
def save_new(torch,p,obj):
 p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():raise FileExistsError(p)
 fd,tmp=tempfile.mkstemp(dir=p.parent,prefix=".tmp-",suffix=".pt");os.close(fd)
 try:
  torch.save(obj,tmp);os.link(tmp,p)
 finally:
  Path(tmp).unlink(missing_ok=True)
def merge_generic(torch,a,key,expected_len):
 if a.get("format_version")!="pretrained_representation_v1" or a.get("sequence_sha256")!=key:raise ValueError("generic artifact identity")
 L=int(a["sequence_length"]);D=int(a["representation_dim"])
 if L!=expected_len:raise ValueError("generic length")
 out=torch.zeros((L,D),dtype=torch.float32);count=torch.zeros((L,1),dtype=torch.float32)
 for s,e,x in zip(a["starts"],a["ends"],a["chunks"]):
  x=x.detach().cpu().float()
  if tuple(x.shape)!=(int(e)-int(s),D):raise ValueError("generic chunk")
  out[int(s):int(e)]+=x;count[int(s):int(e)]+=1
 if bool((count==0).any()) or not bool(torch.isfinite(out).all()):raise ValueError("generic coverage")
 return (out/count).contiguous(),{"source_format":"pretrained_representation_v1","model_name":a["model_name"],"model_revision":a.get("model_revision"),"dim":D}
def load_antibody(torch,p,entity,vh,vl,expected_weight):
 a=torch.load(p,map_location="cpu",weights_only=False)
 if a.get("format_version")!="sabdab2_antiberty_embedding_v1":raise ValueError("AntiBERTy format")
 key=seqkey(vh+vl)
 if a.get("sequence_key")!=key or a.get("source_sequence_sha256")!=key:raise ValueError("AntiBERTy key")
 if int(a.get("vh_length"))!=len(vh) or int(a.get("vl_length"))!=len(vl):raise ValueError("AntiBERTy chain lengths")
 if a.get("model_name")!="AntiBERTy_md_smooth" or a.get("model_weights_sha256")!=expected_weight:raise ValueError("AntiBERTy weights")
 x=a["embedding"].detach().cpu().float().contiguous()
 if tuple(x.shape)!=(len(vh)+len(vl),512) or not bool(torch.isfinite(x).all()):raise ValueError("AntiBERTy tensor")
 return x,{"source_format":a["format_version"],"model_name":a["model_name"],"model_weights_sha256":a["model_weights_sha256"],"dim":512}
def load_antigen(torch,p,key,seq,expected_name):
 a=torch.load(p,map_location="cpu",weights_only=False)
 x,meta=merge_generic(torch,a,key,len(seq))
 if expected_name and meta["model_name"]!=expected_name:raise ValueError(f"antigen model {meta['model_name']}")
 return x,meta
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--sequence-cache",type=Path,required=True);ap.add_argument("--antibody-root",type=Path,required=True);ap.add_argument("--antigen-root",type=Path,required=True);ap.add_argument("--expected-antiberty-weights-sha256",required=True);ap.add_argument("--expected-antigen-model-name",default="esm2_t33_650M_UR50D");ap.add_argument("--source-bank-receipt",type=Path,required=True);ap.add_argument("--source-bank-receipt-sha256",required=True);ap.add_argument("--output-root",type=Path,required=True)
 a=ap.parse_args()
 if a.output_root.exists():raise FileExistsError(a.output_root)
 if sha(a.source_bank_receipt)!=a.source_bank_receipt_sha256:raise ValueError("bank receipt hash")
 bank=load_json(a.source_bank_receipt)
 if bank.get("test_access") is not False:raise ValueError("source bank receipt lacks sealed-test assertion")
 cache=load_json(a.sequence_cache);role=cache.get("role")
 if role not in ("training_only","evaluation_only"):raise ValueError("sequence cache role")
 if cache.get("test_access") is not False or cache.get("ood_access") is not False:raise ValueError("forbidden cache")
 import torch
 a.output_root.mkdir(parents=True)
 entries=[]
 for entity,d in sorted(cache["antibodies"].items()):
  vh=d["vh"];vl=d.get("vl","");key=seqkey(vh+vl);src=path(a.antibody_root,key)
  if not src.is_file():raise FileNotFoundError(src)
  x,meta=load_antibody(torch,src,entity,vh,vl,a.expected_antiberty_weights_sha256);cdr=annotate_antibody(vh,vl)
  dst=a.output_root/"antibody"/entity[:2]/f"{entity}.pt"
  payload={"format_version":OUT_FORMAT,"entity_id":entity,"side":"antibody","role":role,"embedding":x,
           "mask":torch.ones(len(x),dtype=torch.bool),"vh_length":len(vh),"vl_length":len(vl),
           "cdr_masks":torch.tensor(cdr["cdr_masks"],dtype=torch.bool),"source_file_sha256":sha(src),"source_metadata":meta,
           "cdr_receipt":cdr}
  save_new(torch,dst,payload);entries.append({"entity_id":entity,"side":"antibody","relative_path":str(dst.relative_to(a.output_root)),"sha256":sha(dst),"length":len(x),"dim":x.shape[1]})
 for entity,seq in sorted(cache["antigens"].items()):
  key=seqkey(seq);src=path(a.antigen_root,key)
  if not src.is_file():raise FileNotFoundError(src)
  x,meta=load_antigen(torch,src,key,seq,a.expected_antigen_model_name);dst=a.output_root/"antigen"/entity[:2]/f"{entity}.pt"
  payload={"format_version":OUT_FORMAT,"entity_id":entity,"side":"antigen","role":role,"embedding":x,
           "mask":torch.ones(len(x),dtype=torch.bool),"source_file_sha256":sha(src),"source_metadata":meta}
  save_new(torch,dst,payload);entries.append({"entity_id":entity,"side":"antigen","relative_path":str(dst.relative_to(a.output_root)),"sha256":sha(dst),"length":len(x),"dim":x.shape[1]})
 receipt={"format_version":"strict_partner_retrieval_tensor_cache_receipt_v1","status":"COMPLETE_FOLD_LOCAL_TENSOR_CACHE",
          "role":role,"fold":cache["fold"],"sequence_cache_sha256":sha(a.sequence_cache),"source_bank_receipt_sha256":sha(a.source_bank_receipt),
          "entry_count":len(entries),"entries":entries,"antibody_count":len(cache["antibodies"]),"antigen_count":len(cache["antigens"]),
          "training_and_model_selection_authorized":cache.get("training_and_model_selection_authorized") is True,"outer_evaluation_authorized":cache.get("outer_evaluation_authorized") is True,"whole_pool_bank_visible_to_model_process":False,
          "test_access":False,"ood_access":False}
 (a.output_root/"CACHE_RECEIPT.json").write_text(json.dumps(receipt,indent=2,sort_keys=True,allow_nan=False)+"\n")
 print(json.dumps({"status":receipt["status"],"fold":receipt["fold"],"role":role,"entries":len(entries),"training_authorized":receipt["training_and_model_selection_authorized"],"outer_evaluation_authorized":receipt["outer_evaluation_authorized"]},indent=2))
if __name__=="__main__":main()
