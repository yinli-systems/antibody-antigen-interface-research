#!/usr/bin/env python3
"""Materialize disjoint fold-local sequence caches from a frozen nested split."""
from __future__ import annotations
import argparse,csv,hashlib,json
from pathlib import Path
from typing import Any
VALID=set("ACDEFGHIKLMNPQRSTVWY")

def sha(p:Path)->str:
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
 return h.hexdigest()
def frame(s:str)->bytes:return str(len(s.encode("ascii"))).encode()+b":"+s.encode("ascii")
def aid(vh,vl):return hashlib.sha256(b"antibody-v1\0"+frame(vh)+frame(vl)).hexdigest()
def gid(ag):return hashlib.sha256(b"antigen-v1\0"+frame(ag)).hexdigest()
def pid(a,g):return hashlib.sha256(b"pair-v1\0"+a.encode()+b"\0"+g.encode()).hexdigest()
def norm(x,empty=False):
 s="".join(str(x).split()).upper()
 if not s and not empty:raise ValueError("empty sequence")
 if any(c not in VALID for c in s):raise ValueError("noncanonical sequence")
 return s
def load(p):return json.loads(p.read_text())

def inventory(csv_path:Path):
 antibodies={};antigens={};pairs={};rows={}
 with csv_path.open(newline="",encoding="utf-8") as f:
  r=csv.DictReader(f)
  if r.fieldnames!=["INSTANCE","vh_seq","vl_seq","ag_seq"]:raise ValueError(f"exact 4-column manifest required, got {r.fieldnames}")
  for x in r:
   iid=str(x["INSTANCE"]);vh=norm(x["vh_seq"]);vl=norm(x["vl_seq"],True);ag=norm(x["ag_seq"])
   a=aid(vh,vl);g=gid(ag);p=pid(a,g)
   if iid in rows:raise ValueError("duplicate instance")
   if a in antibodies and antibodies[a]!={"vh":vh,"vl":vl}:raise ValueError("antibody ID collision")
   if g in antigens and antigens[g]!=ag:raise ValueError("antigen ID collision")
   antibodies[a]={"vh":vh,"vl":vl};antigens[g]=ag;pairs.setdefault(p,{"pair_id":p,"antibody_id":a,"antigen_id":g,"source_instance_ids":[]})["source_instance_ids"].append(iid)
   rows[iid]=p
 return antibodies,antigens,pairs,rows

def find_fold(assignment,label):
 outer=assignment["construction"]["outer_level"]["folds"]
 if label.startswith("outer_") and "_inner_" not in label:return outer[int(label.split("_")[1])]
 parts=label.split("_");oi=int(parts[1]);ii=int(parts[3]);return outer[oi]["inner_level"]["folds"][ii]

def ids(rows):return {x["pair_id"] for x in rows}

def write_new(p:Path,obj):
 if p.exists():raise FileExistsError(p)
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+"\n")

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--frozen-split",type=Path,required=True);ap.add_argument("--assignment",type=Path,required=True);ap.add_argument("--canonical-manifest",type=Path,required=True);ap.add_argument("--canonical-manifest-sha256",required=True);ap.add_argument("--fold",required=True);ap.add_argument("--output-root",type=Path,required=True)
 a=ap.parse_args()
 fs=load(a.frozen_split);ass=load(a.assignment)
 if sha(a.canonical_manifest)!=a.canonical_manifest_sha256:raise ValueError("canonical manifest hash")
 # Sequence caches may be prepared before full Gate0, but model training may not use them unless frozen split is authorized.
 antibodies,antigens,pairs,source_to_pair=inventory(a.canonical_manifest)
 fold=find_fold(ass,a.fold);tr=fold["training_pairs"];va=fold["validation_pairs"];qu=fold["quarantined_pairs"]
 tr_ids=ids(tr);va_ids=ids(va);qu_ids=ids(qu)
 if tr_ids&va_ids or tr_ids&qu_ids or va_ids&qu_ids:raise ValueError("role overlap")
 if tr_ids|va_ids|qu_ids != set(pairs):raise ValueError("physical manifest and fold pair universe differ")
 def subset(pairids):
  pp=[pairs[p] for p in sorted(pairids)];aa=sorted({x["antibody_id"] for x in pp});gg=sorted({x["antigen_id"] for x in pp})
  return pp,{x:antibodies[x] for x in aa},{x:antigens[x] for x in gg}
 trp,tra,trg=subset(tr_ids);vap,vaa,vag=subset(va_ids)
 base={"format_version":"strict_partner_retrieval_sequence_cache_v1","fold":a.fold,"frozen_split_sha256":sha(a.frozen_split),"assignment_sha256":sha(a.assignment),"canonical_manifest_sha256":sha(a.canonical_manifest),"training_and_model_selection_authorized":bool(fs.get("training_and_model_selection_authorized")),"outer_evaluation_authorized":bool(fs.get("outer_evaluation_authorized")),"test_access":False,"ood_access":False}
 train={**base,"role":"training_only","pairs":trp,"antibodies":tra,"antigens":trg,
        "source_instance_allowlist":sorted({i for x in trp for i in x["source_instance_ids"]})}
 evaluation={**base,"role":"evaluation_only","pairs":vap,"antibodies":vaa,"antigens":vag,
        "source_instance_allowlist":sorted({i for x in vap for i in x["source_instance_ids"]}),
        "nonpositive_semantics":"unlabeled_ranking_candidates"}
 h0={**base,"role":"h0_evaluation_only","training_pairs":trp,"training_antibodies":tra,"training_antigens":trg,
     "validation_pairs":vap,"validation_antibodies":vaa,"validation_antigens":vag,
     "format_version":"strict_partner_retrieval_h0_cache_v1","nonpositive_semantics":"unlabeled_ranking_candidates"}
 root=a.output_root/a.fold;write_new(root/"TRAIN_SEQUENCES.json",train);write_new(root/"EVAL_SEQUENCES.json",evaluation);write_new(root/"H0_CACHE.json",h0)
 receipt={**base,"status":"PREPARED_FOLD_CACHES","counts":{"train_pairs":len(trp),"validation_pairs":len(vap),"quarantined_pairs":len(qu_ids),"train_antibodies":len(tra),"train_antigens":len(trg),"validation_antibodies":len(vaa),"validation_antigens":len(vag)},"artifacts":{}}
 for name in ("TRAIN_SEQUENCES.json","EVAL_SEQUENCES.json","H0_CACHE.json"):receipt["artifacts"][name]={"sha256":sha(root/name),"bytes":(root/name).stat().st_size}
 write_new(root/"CACHE_RECEIPT.json",receipt)
 print(json.dumps({"fold":a.fold,"status":receipt["status"],"training_authorized":receipt["training_and_model_selection_authorized"],"outer_evaluation_authorized":receipt["outer_evaluation_authorized"],"counts":receipt["counts"]},indent=2))
if __name__=="__main__":main()
