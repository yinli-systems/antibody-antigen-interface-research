#!/usr/bin/env python3
"""Stage training-only sparse contact maps for R1/R2 auxiliary diagnostics."""
from __future__ import annotations
import argparse,hashlib,json,os,shutil
from pathlib import Path

def sha(p:Path):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
 return h.hexdigest()

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--train-sequences",type=Path,required=True);ap.add_argument("--contact-root",type=Path,required=True);ap.add_argument("--project-root",type=Path,required=True);ap.add_argument("--output-root",type=Path,required=True)
 a=ap.parse_args()
 if a.output_root.exists():raise FileExistsError(a.output_root)
 d=json.loads(a.train_sequences.read_text())
 if d.get("role")!="training_only" or d.get("test_access") is not False or d.get("ood_access") is not False:raise ValueError("training cache authority")
 # Import the strict no-pickle contact validator from reviewed project source.
 import sys
 sys.path.insert(0,str((a.project_root/"src").resolve()))
 from interaction_first.contacts import load_contact_map
 allow=set(map(str,d["source_instance_allowlist"]))
 pair_by_instance={}
 for p in d["pairs"]:
  for iid in p["source_instance_ids"]:
   if iid in pair_by_instance:raise ValueError("source instance maps to multiple exact pairs")
   pair_by_instance[iid]=p["pair_id"]
 if set(pair_by_instance)!=allow:raise ValueError("source allowlist differs from exact training-pair rows")
 a.output_root.mkdir(parents=True)
 entries=[]
 for iid in sorted(allow):
  src=a.contact_root/f"{iid}.npz"
  if not src.is_file():raise FileNotFoundError(src)
  cm=load_contact_map(src,iid)
  dst=a.output_root/"maps"/f"{iid}.npz";dst.parent.mkdir(parents=True,exist_ok=True)
  # Copy rather than symlink so the model process need not mount the whole source root.
  with src.open("rb") as fi,dst.open("xb") as fo:shutil.copyfileobj(fi,fo,8*1024*1024)
  if sha(src)!=sha(dst):raise RuntimeError("contact copy hash mismatch")
  entries.append({"instance_id":iid,"pair_id":pair_by_instance[iid],"relative_path":str(dst.relative_to(a.output_root)),
                  "sha256":sha(dst),"candidate_count":cm.candidate_count,"positive_count":cm.positive_count,
                  "antibody_length":cm.antibody_length,"antigen_length":cm.antigen_length})
 receipt={"format_version":"strict_partner_retrieval_contact_cache_v1","status":"COMPLETE_TRAINING_ONLY_CONTACT_CACHE",
          "fold":d["fold"],"train_sequence_cache_sha256":sha(a.train_sequences),"entry_count":len(entries),"entries":entries,
          "pair_count":len({x["pair_id"] for x in entries}),"source_instance_count":len(entries),
          "model_execution_authorized":d.get("model_execution_authorized") is True,
          "contact_distance_authority":"precomputed frozen sabdab2_sparse_contact_v1 at 4.5A source protocol",
          "validation_contact_labels_staged":False,"test_access":False,"ood_access":False}
 (a.output_root/"CONTACT_CACHE_RECEIPT.json").write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
 print(json.dumps({"status":receipt["status"],"fold":receipt["fold"],"instances":len(entries),"pairs":receipt["pair_count"],"authorized":receipt["model_execution_authorized"]},indent=2))
if __name__=="__main__":main()
