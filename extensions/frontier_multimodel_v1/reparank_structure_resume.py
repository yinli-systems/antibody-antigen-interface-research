#!/usr/bin/env python3
"""Resume RePaRank sequence+structure evaluation without repeating verified cases.

The wrapper is deliberately fail-closed. Existing raw pair-score NPZ files are
reused only after their candidate axes/labels and every structure input used by
that instance are proven compatible with the new complete structure authority.
"""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, sys, time
from pathlib import Path
from typing import Any
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import reparank_same_protocol as rp


def sha(path:Path)->str:
    return rp.sha256(Path(path))


def write_new(path:Path,payload:dict)->None:
    rp.write_exclusive_json(Path(path),payload)


def contact_loader(project_source:Path):
    src=project_source.resolve()/"src"
    if str(src) not in sys.path:sys.path.insert(0,str(src))
    from interaction_first.contacts import load_contact_map
    return load_contact_map


def expected_pair_arrays(contact):
    labels,mask=contact.dense()
    antigen_local,antibody_local=np.nonzero(mask)
    return {
        "antigen_residue_index":contact.antigen_indices[antigen_local].astype(np.int64),
        "antibody_residue_index":contact.antibody_indices[antibody_local].astype(np.int64),
        "label":labels[mask].astype(np.uint8),
        "mask":mask,
        "labels_matrix":labels,
    }


def inspect_npz(path:Path,instance:str,contact)->dict[str,Any]:
    expected=expected_pair_arrays(contact)
    with np.load(path,allow_pickle=False) as z:
        required={"instance_id","antigen_residue_index","antibody_residue_index","label","score","score_definition"}
        if set(z.files)!=required:raise ValueError(f"NPZ fields differ {path}: {z.files}")
        iid=str(z["instance_id"].item()); definition=str(z["score_definition"].item())
        if iid!=instance:raise ValueError(f"instance mismatch {path}: {iid} != {instance}")
        if definition!="negative_euclidean_distance":raise ValueError(f"score definition drift {path}")
        for key in ("antigen_residue_index","antibody_residue_index","label"):
            if not np.array_equal(z[key],expected[key]):raise ValueError(f"candidate/label drift {instance}:{key}")
        score=np.asarray(z["score"])
        if score.dtype!=np.float32:raise ValueError(f"score dtype drift {instance}: {score.dtype}")
        if score.ndim!=1 or len(score)!=len(expected["label"]):raise ValueError(f"score shape drift {instance}")
        if not np.isfinite(score).all():raise ValueError(f"non-finite score {instance}")
        return {"instance_id":instance,"sha256":sha(path),"pair_count":int(score.size),"positive_count":int(expected['label'].sum()),"score_min":float(score.min()),"score_max":float(score.max())}


def record_pdb_path(manifest:Path,record:dict)->Path:
    p=Path(record["pdb_path"])
    return p if p.is_absolute() else manifest.resolve().parent/p


def structure_equivalence_for_instance(instance:str,sequence:dict[str,str],old_manifest:Path,new_manifest:Path,old_by_key:dict,new_by_key:dict)->dict:
    chains=[]
    for name,seq in sequence.items():
        if name=="light" and not seq:continue
        key=rp.sequence_key(seq)
        if key not in old_by_key:raise ValueError(f"old structure authority missing reused chain {instance}:{name}:{key}")
        if key not in new_by_key:raise ValueError(f"new structure authority missing reused chain {instance}:{name}:{key}")
        old=old_by_key[key];new=new_by_key[key]
        op=record_pdb_path(old_manifest,old);np_=record_pdb_path(new_manifest,new)
        if sha(op)!=old["pdb_sha256"]:raise ValueError(f"old PDB hash drift {op}")
        if sha(np_)!=new["pdb_sha256"]:raise ValueError(f"new PDB hash drift {np_}")
        if old["pdb_sha256"]!=new["pdb_sha256"]:
            raise ValueError(f"reused case structure changed under new authority {instance}:{name}:{key}")
        chains.append({"role":name,"sequence_sha256":key,"pdb_sha256":old["pdb_sha256"],"old_path":str(op),"new_path":str(np_)})
    return {"instance_id":instance,"chains":chains}


def freeze(args):
    split=json.loads(args.split_manifest.read_text());validation=list(map(str,split["splits"]["validation"]))
    if len(validation)!=217 or len(set(validation))!=217:raise ValueError("expected exact 217 validation IDs")
    if set(validation)&(set(map(str,split["splits"]["train"]))|set(map(str,split["splits"]["test"]))):raise ValueError("split overlap")
    canonical=rp.canonical_rows(args.canonical_manifest)
    _,old_by=rp.structure_records(args.old_structure_manifest)
    _,new_by=rp.structure_records(args.new_structure_manifest)
    load_contact=contact_loader(args.project_source)
    partial=args.partial_output/"pair_predictions"
    files=sorted(partial.glob("*.npz")); ids=[p.stem for p in files]
    if len(ids)!=len(set(ids)):raise ValueError("duplicate partial IDs")
    foreign=sorted(set(ids)-set(validation))
    if foreign:raise ValueError(f"foreign partial IDs: {foreign}")
    reused=[]
    for p in files:
        iid=p.stem; contact=load_contact(args.contact_dir/f"{iid}.npz",iid)
        npz=inspect_npz(p,iid,contact)
        structures=structure_equivalence_for_instance(iid,canonical[iid],args.old_structure_manifest,args.new_structure_manifest,old_by,new_by)
        reused.append({**npz,"structure_equivalence":structures})
    missing=[x for x in validation if x not in set(ids)]
    if not reused or not missing:raise ValueError("resume requires both verified reused and missing cases")
    official_head=os.popen(f"git -C {args.reparank_source} rev-parse HEAD").read().strip()
    if official_head!=args.official_source_commit:raise ValueError(f"official source head drift {official_head}")
    pins={str(p.resolve()):sha(p) for p in [args.checkpoint,args.canonical_manifest,args.split_manifest,args.old_structure_manifest,args.new_structure_manifest,Path(__file__),HERE/'reparank_same_protocol.py']}
    plan={"schema":"reparank_structure_resume_v1","created_utc":time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),"checkpoint":str(args.checkpoint.resolve()),"checkpoint_sha256":sha(args.checkpoint),"official_source":str(args.reparank_source.resolve()),"official_source_commit":args.official_source_commit,"project_source":str(args.project_source.resolve()),"canonical_manifest":str(args.canonical_manifest.resolve()),"split_manifest":str(args.split_manifest.resolve()),"contact_dir":str(args.contact_dir.resolve()),"old_structure_manifest":str(args.old_structure_manifest.resolve()),"new_structure_manifest":str(args.new_structure_manifest.resolve()),"partial_output":str(args.partial_output.resolve()),"validation_ids":validation,"reused":reused,"reused_count":len(reused),"missing_ids":missing,"missing_count":len(missing),"input_sha256":pins,"contact_definition":"existing project contact labels/candidate grid; no threshold changed by this wrapper","score_definition":"negative Euclidean distance between official RePaRank complex embeddings","test_labels_read":False,"ood_labels_read":False,"checkpoint_training_overlap_audited":False,"scientific_change":"none; execution resumes only missing validation cases after exact compatibility audit"}
    write_new(args.output,plan);print(json.dumps({"plan":str(args.output),"sha256":sha(args.output),"reused":len(reused),"missing":len(missing)},indent=2))


def verify_plan(path:Path,expected:str)->dict:
    if sha(path)!=expected:raise ValueError("plan hash drift")
    p=json.loads(path.read_text())
    if p["schema"]!="reparank_structure_resume_v1" or p["test_labels_read"] or p["ood_labels_read"]:raise ValueError("invalid plan")
    for f,h in p["input_sha256"].items():
        if sha(Path(f))!=h:raise ValueError("input/source drift: "+f)
    if sha(Path(p["checkpoint"]))!=p["checkpoint_sha256"]:raise ValueError("checkpoint drift")
    return p


def run(args):
    import torch
    p=verify_plan(args.plan,args.plan_sha256)
    if not os.environ.get("SLURM_JOB_ID"):raise RuntimeError("missing-case inference requires scheduler")
    if int(os.environ.get("SLURM_RESTART_COUNT","0")):raise RuntimeError("automatic restart forbidden")
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False);pairs=out/"pair_predictions";pairs.mkdir()
    source_partial=Path(p["partial_output"])/"pair_predictions"
    reused_by={x["instance_id"]:x for x in p["reused"]}
    for iid,rec in reused_by.items():
        src=source_partial/f"{iid}.npz";dst=pairs/src.name
        if sha(src)!=rec["sha256"]:raise ValueError("partial NPZ drift "+iid)
        shutil.copyfile(src,dst)
        if sha(dst)!=rec["sha256"]:raise ValueError("copied NPZ drift "+iid)
    split=json.loads(Path(p["split_manifest"]).read_text());canonical=rp.canonical_rows(Path(p["canonical_manifest"]));_,structures=rp.structure_records(Path(p["new_structure_manifest"]));load_contact=contact_loader(Path(p["project_source"]))
    model,hparams,prefix,_=rp.load_official_model(Path(p["checkpoint"]),Path(p["official_source"]),args.device)
    if bool(hparams.get("skip_position_encoding")):raise ValueError("checkpoint unexpectedly sequence-only")
    if hparams.get("feature_dims"):raise ValueError("checkpoint requires unsupported extra features")
    state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};device=torch.device(args.device)
    from reparank.models.reparank import RePaRankInput
    start=time.monotonic();made=[]
    with torch.inference_mode():
      for n,iid in enumerate(p["missing_ids"],1):
        seq=canonical[iid];sequences={"heavy":(seq["heavy"],),"antigen":(seq["antigen"],)}
        if seq["light"]:sequences["light"]=(seq["light"],)
        positions={}
        for name,s in seq.items():
            if name=="light" and not s:continue
            key=rp.sequence_key(s)
            if key not in structures:raise ValueError(f"new authority missing {iid}:{name}:{key}")
            rec=structures[key];manifest=Path(p["new_structure_manifest"]);pdb=record_pdb_path(manifest,rec)
            if sha(pdb)!=rec["pdb_sha256"]:raise ValueError("PDB hash drift "+str(pdb))
            b=rp.parse_backbone_pdb(pdb,s);pos_num=int(hparams.get("pos_num") or 1)
            if pos_num==1:b=b[:,1:2,:]
            elif pos_num!=4:raise ValueError("unsupported pos_num")
            positions[name]=torch.as_tensor(b,device=device).unsqueeze(0)
        result=model(RePaRankInput(sequences=sequences,positions=positions));ab=[result.complex_embeddings["heavy"]]
        if "light" in result.complex_embeddings:ab.append(result.complex_embeddings["light"])
        antibody=torch.cat(ab,dim=1)[0];antigen=result.complex_embeddings["antigen"][0]
        contact=load_contact(Path(p["contact_dir"])/f"{iid}.npz",iid);labels,mask=contact.dense()
        ai=torch.as_tensor(contact.antigen_indices,dtype=torch.long,device=device);bi=torch.as_tensor(contact.antibody_indices,dtype=torch.long,device=device)
        if len(antibody)!=len(seq["heavy"])+len(seq["light"]) or len(antigen)!=len(seq["antigen"]):raise ValueError("embedding length mismatch "+iid)
        score=(-torch.cdist(antigen.index_select(0,ai),antibody.index_select(0,bi))).cpu().numpy()
        if score.shape!=labels.shape or mask.shape!=labels.shape:raise ValueError("candidate shape mismatch "+iid)
        al,bl=np.nonzero(mask);dest=pairs/f"{iid}.npz"
        np.savez_compressed(dest,instance_id=np.asarray(iid),antigen_residue_index=contact.antigen_indices[al].astype(np.int64),antibody_residue_index=contact.antibody_indices[bl].astype(np.int64),label=labels[mask].astype(np.uint8),score=score[mask].astype(np.float32),score_definition=np.asarray("negative_euclidean_distance"))
        inspect_npz(dest,iid,contact);made.append({"instance_id":iid,"sha256":sha(dest)})
        if n%25==0 or n==len(p["missing_ids"]):print(f"inferred missing {n}/{len(p['missing_ids'])}",flush=True)
    for k,v in model.state_dict().items():
        if not torch.equal(v.cpu(),state[k]):raise RuntimeError("model state mutated")
    verify_plan(args.plan,args.plan_sha256)
    write_new(out/"INFERENCE_COMPLETE.json",{"state":"COMPLETE_RESUMED_MISSING_INFERENCE","plan_sha256":args.plan_sha256,"reused_count":len(reused_by),"new_inference_count":len(made),"new_files":made,"wall_seconds":time.monotonic()-start,"test_labels_read":False,"ood_labels_read":False,"checkpoint_training_overlap_audited":False})


def aggregate(args):
    import pandas as pd
    from sklearn.metrics import average_precision_score,roc_auc_score
    p=verify_plan(args.plan,args.plan_sha256);root=args.input.resolve();pairs=root/"pair_predictions";load_contact=contact_loader(Path(p["project_source"]));split=json.loads(Path(p["split_manifest"]).read_text())
    ids=p["validation_ids"];components={i:str(split["component_id_by_instance"][i]) for i in ids};rows=[];primary={};pl=[];ps=[];hashes={}
    for iid in ids:
        fp=pairs/f"{iid}.npz";contact=load_contact(Path(p["contact_dir"])/f"{iid}.npz",iid);inspect_npz(fp,iid,contact);hashes[iid]=sha(fp)
        with np.load(fp,allow_pickle=False) as z:score=np.asarray(z["score"],dtype=np.float64);lab=np.asarray(z["label"],dtype=np.uint8)
        metrics=rp.per_complex_metrics(lab,score);primary[iid]=metrics["average_precision"];pl.append(lab);ps.append(score)
        labels,mask=contact.dense();matrix=np.full(labels.shape,-np.inf,dtype=np.float64);matrix[mask]=score
        epi,para=contact.node_labels();metrics["epitope_average_precision"]=float(average_precision_score(epi,matrix.max(axis=1)));metrics["paratope_average_precision"]=float(average_precision_score(para,matrix.max(axis=0)))
        rows.append({"instance_id":iid,"component_id":components[iid],**metrics})
    pooled_label=np.concatenate(pl);pooled_score=np.concatenate(ps)
    pd.DataFrame(rows).to_csv(root/"per_complex.csv",index=False)
    payload={"format_version":"reparank_structure_resume_evaluation_v1","status":"COMPLETE_EXPOSED_PRETRAINED_VALIDATION_DIAGNOSTIC","mode":"sequence_plus_structure","checkpoint":p["checkpoint"],"checkpoint_sha256":p["checkpoint_sha256"],"official_source_commit":p["official_source_commit"],"split_manifest_sha256":sha(Path(p["split_manifest"])),"canonical_manifest_sha256":sha(Path(p["canonical_manifest"])),"structure_manifest_sha256":sha(Path(p["new_structure_manifest"])),"instance_count":len(rows),"primary_macro_mean_per_complex_AP":float(np.mean(list(primary.values()))),"primary_equal_component_analysis":rp.equal_component_bootstrap(primary,components),"epitope_macro_AP":float(np.mean([x["epitope_average_precision"] for x in rows])),"paratope_macro_AP":float(np.mean([x["paratope_average_precision"] for x in rows])),"pooled_pair_metrics":{"average_precision":float(average_precision_score(pooled_label,pooled_score)),"auroc":float(roc_auc_score(pooled_label,pooled_score)),"pair_count":int(pooled_label.size),"positive_count":int(pooled_label.sum())},"raw_pair_prediction_directory":str(pairs),"raw_pair_prediction_file_count":len(hashes),"raw_pair_prediction_sha256":hashes,"per_complex":rows,"score_definition":p["score_definition"],"candidate_grid":"exact existing complete antibody-by-antigen residue grid","resume_provenance":{"reused_count":p["reused_count"],"new_inference_count":p["missing_count"],"partial_output":p["partial_output"]},"checkpoint_training_overlap_audited":False,"test_labels_read":False,"ood_labels_read":False,"blind_claim_authorized":False,"sota_claim_authorized":False}
    write_new(root/"evaluation.json",payload);print(json.dumps({"status":payload["status"],"instance_count":217,"macro_AP":payload["primary_macro_mean_per_complex_AP"],"reused":p["reused_count"],"new":p["missing_count"]},indent=2))


def main():
    pa=argparse.ArgumentParser();s=pa.add_subparsers(dest="cmd",required=True)
    f=s.add_parser("freeze")
    for x in ("checkpoint","reparank-source","project-source","canonical-manifest","split-manifest","contact-dir","old-structure-manifest","new-structure-manifest","partial-output","output"):f.add_argument("--"+x,type=Path,required=True)
    f.add_argument("--official-source-commit",required=True)
    for cmd in ("run","aggregate"):
        q=s.add_parser(cmd);q.add_argument("--plan",type=Path,required=True);q.add_argument("--plan-sha256",required=True);q.add_argument("--input",type=Path) if cmd=="aggregate" else q.add_argument("--output",type=Path,required=True)
        if cmd=="run":q.add_argument("--device",choices=("cuda",),default="cuda")
    a=pa.parse_args();{"freeze":freeze,"run":run,"aggregate":aggregate}[a.cmd](a)
if __name__=="__main__":main()
