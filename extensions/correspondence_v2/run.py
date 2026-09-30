#!/usr/bin/env python3
"""Freeze and run bounded development-only experiments on existing checkpoints."""
from __future__ import annotations
import argparse
import csv
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import sys
import time
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(Path(__file__).parent)]
from core import (digest_file, write_new_json, chain_permutation,
                  transform_extra_columns, ranking_summary, guard_validation_ids)

D = Path("/ssd/scxi253/model_deployment/data")
R = Path("/ssd/scxi253/repr_extract/out")
RUNS = Path("/ssd/scxi253/repr_ablation/out")
ENCODERS = {"esm_if1": ("esm_if1_only", 512),
            "antiberty": ("antiberty_only", 512),
            "proteinmpnn": ("proteinmpnn_aligned", 128)}
CONDITIONS = [("identity", 0), ("chain_shift", 1), ("chain_shift", -1),
              ("chain_shift", 2), ("chain_shift", 4), ("chain_shift", 8),
              ("chain_shift", 16), ("local_shuffle", 4), ("local_shuffle", 8),
              ("same_aa_shuffle", 0), ("global_shuffle", 0), ("oracle_restore", 8)]


def sequence_metadata(path):
    sequences, instance = {}, {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            h, l, a = [(row.get(k) or "").strip() for k in ("vh_seq", "vl_seq", "ag_seq")]
            if not h or not a:
                continue
            ab = tuple(x for x in (h, l) if x)
            for parts in (ab, (a,)):
                key = hashlib.sha256("".join(parts).encode("ascii")).hexdigest()
                if key in sequences and sequences[key] != parts:
                    raise ValueError("same sequence has ambiguous biological chain boundaries")
                sequences[key] = parts
            instance[row["INSTANCE"]] = (hashlib.sha256((h+l).encode()).hexdigest(),
                                         hashlib.sha256(a.encode()).hexdigest())
    return sequences, instance


def freeze(args):
    import pandas as pd
    from interaction_first.composite_embeddings import build_sources
    from interaction_first.pretrained_representations import representation_path
    split = D / "interaction_first/double_remote_split.json"
    manifest = D / "interaction_first/canonical_manifest.csv"
    prepared = D / "interaction_first_sabdab2/instances.parquet"
    split_data = json.loads(split.read_text())
    ids = sorted(map(str, split_data["splits"]["validation"]))
    guard_validation_ids(ids, split_data["splits"])
    frame = pd.read_parquet(prepared, columns=["instance_id", "ab_sequence_sha256", "ag_sequence_sha256"])
    frame = frame.set_index("instance_id", verify_integrity=True).loc[ids]
    sequences, _ = sequence_metadata(manifest)
    tasks = []
    for encoder, (arm, dim) in ENCODERS.items():
        for seed in (1, 2, 3):
            checkpoint = RUNS / arm / f"seed_{seed}/train/best.pt"
            tasks.append({"task_id":len(tasks), "encoder":encoder, "dim":dim,
                          "training_seed":seed, "checkpoint":str(checkpoint),
                          "checkpoint_sha256":digest_file(checkpoint)})
    pins = {str(x):digest_file(x) for x in (split, manifest, prepared)}
    paths = set()
    for iid, row in frame.iterrows():
        paths.add(D / "interaction_first/contacts" / f"{iid}.npz")
        for column in ("ab_sequence_sha256", "ag_sequence_sha256"):
            key = str(row[column])
            if key not in sequences:
                raise ValueError(f"missing sequence metadata for {iid}")
            # Runtime EmbeddingStore uses this sequence-key sharding convention.
            paths.add(D / "sppider_seq_sabdab2_embeddings" / key[:2] / f"{key}.pt")
    # Bind present extra features AND absent ones; no silent appearance of files.
    absent = set()
    for encoder in ENCODERS:
        source = build_sources(manifest, **{encoder+"_root": R / encoder})[0]
        for key in set(frame.ab_sequence_sha256) | set(frame.ag_sequence_sha256):
            if source.covers(key):
                for part in source.parts_by_key[key]:
                    path = representation_path(source.root, part)
                    (paths if path.exists() else absent).add(path)
    for path in sorted(paths):
        pins[str(path)] = digest_file(path)
    source_pins = {str(p.relative_to(ROOT)):digest_file(p)
                   for folder in ("src", "scripts", "extensions/correspondence_v2")
                   for p in (ROOT/folder).rglob("*.py")}
    plan = {"schema":"correspondence_extension_v2", "base_commit":"55cfd137aee2b3b749f244fcb76dd11d30dda2c5",
            "created_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
            "evidence_role":"post-freeze exploratory development only",
            "test_access":"forbidden", "promotion_authorized":False,
            "split":str(split), "manifest":str(manifest), "prepared":str(prepared),
            "contacts":str(D/"interaction_first/contacts"),
            "embeddings":str(D/"sppider_seq_sabdab2_embeddings"),
            "representations":str(R), "ids":ids, "tasks":tasks,
            "conditions":[list(x) for x in CONDITIONS], "perturbation_seed":20260930,
            "ranking_anchor_count":16, "ranking_distractor_count":15,
            "ranking_condition_names":["identity:0","chain_shift:8","global_shuffle:0"],
            "ranking_semantics":"known-cognate retrieval among unlabelled alternatives, NOT binding specificity",
            "primary_metric":"per-complex non-interpolated average precision on pinned candidate pairs",
            "candidate_policy":"existing frozen residue axes and candidate mask; no new contact labels or negatives",
            "stats":"pointwise crossed seed/cluster bootstrap; no significance-driven model selection",
            "input_sha256":pins, "source_sha256":source_pins,
            "absent_inputs":sorted(map(str,absent)),
            "budget":{"tasks":9,"max_parallel_gpus":3,"minutes_per_task":45,
                      "max_gpu_hours_formal":6.75,"max_output_gib":8}}
    write_new_json(args.plan,plan)
    print(json.dumps({"plan":str(args.plan),"sha256":digest_file(args.plan),
                      "cases":len(ids),"input_files":len(pins),"tasks":len(tasks)}),flush=True)


def run(args):
    import numpy as np
    import pandas as pd
    import torch
    from sklearn.metrics import average_precision_score
    from interaction_first.runtime import EmbeddingStore, safe_torch_load, set_reproducible_seed
    from interaction_first.model import PartnerSpecificInteractionModel, merge_chunk_embeddings
    from interaction_first.composite_embeddings import CompositeEmbeddingStore, build_sources
    from interaction_first.contacts import load_contact_map

    if not torch.cuda.is_available():
        raise RuntimeError("this execution requires a scheduled CUDA device")
    if digest_file(args.plan) != args.plan_sha256:
        raise RuntimeError("external plan digest mismatch")
    plan = json.loads(args.plan.read_text())
    if args.task < 0 or args.task >= len(plan["tasks"]):
        raise ValueError("unknown task")
    task = plan["tasks"][args.task]
    suffix = f"smoke_{args.task}" if args.smoke else f"task_{args.task}"
    out = args.output / suffix
    out.mkdir(parents=True, exist_ok=False)
    write_new_json(out/"START.json", {"task":task,"plan_sha256":args.plan_sha256,
                                    "job_id":os.environ.get("SLURM_JOB_ID"),
                                    "smoke_not_science":args.smoke})
    began = time.monotonic()
    for rel, sha in plan["source_sha256"].items():
        if digest_file(ROOT/rel) != sha:
            raise RuntimeError(f"source changed: {rel}")
    split = json.loads(Path(plan["split"]).read_text())
    guard_validation_ids(plan["ids"], split["splits"])
    # Verify input bytes before tensor loading. Hashing is not scoring.
    for path, sha in plan["input_sha256"].items():
        if digest_file(Path(path)) != sha:
            raise RuntimeError(f"input changed: {path}")
    if any(Path(x).exists() for x in plan["absent_inputs"]):
        raise RuntimeError("previously missing representation appeared after freeze")
    checkpoint_path = Path(task["checkpoint"])
    if digest_file(checkpoint_path) != task["checkpoint_sha256"]:
        raise RuntimeError("checkpoint hash mismatch")
    device = torch.device("cuda")
    set_reproducible_seed(20260930, deterministic=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checkpoint = safe_torch_load(checkpoint_path,map_location=device)
    config = checkpoint["config"]
    if config["split_manifest_sha256"] != plan["input_sha256"][plan["split"]]:
        raise RuntimeError("checkpoint split does not match validation authority")
    kwargs = {name:config[name] for name in inspect.signature(PartnerSpecificInteractionModel).parameters
              if name in config}
    kwargs["antigen_input_dim"] = int(config.get("antigen_input_dim",config["input_dim"]))
    kwargs["antibody_input_dim"] = int(config.get("antibody_input_dim",config["input_dim"]))
    model = PartnerSpecificInteractionModel(**kwargs).to(device)
    model.load_state_dict(checkpoint["model_state_dict"],strict=True)
    model.eval()
    parameter_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    encoder, dim = task["encoder"],task["dim"]
    source = build_sources(Path(plan["manifest"]),**{encoder+"_root":Path(plan["representations"])/encoder})
    store = CompositeEmbeddingStore(EmbeddingStore(Path(plan["embeddings"]),1.0),source)
    sequences, known_rows = sequence_metadata(Path(plan["manifest"]))
    frame = pd.read_parquet(plan["prepared"],columns=["instance_id","ab_sequence_sha256","ag_sequence_sha256"])
    frame = frame.set_index("instance_id",verify_integrity=True)
    ids = plan["ids"][:3] if args.smoke else plan["ids"]
    guard_validation_ids(ids,split["splits"])
    components = split.get("component_id_by_instance",{})
    if any(i not in components for i in ids):
        raise RuntimeError("missing cluster identity; no fallback to independent rows")
    cache = {}
    def full(key):
        if key not in cache:
            chunks,starts,length = store.load(key,device)
            cache[key] = merge_chunk_embeddings(chunks,starts,length,device)
        return cache[key]
    def features(iid, mode, k):
        row=frame.loc[iid]
        contact=load_contact_map(Path(plan["contacts"])/f"{iid}.npz",iid)
        values,diagnostics = [],[]
        for side,col,indices in (("ag","ag_sequence_sha256",contact.antigen_indices),
                                 ("ab","ab_sequence_sha256",contact.antibody_indices)):
            key=str(row[col]); tensor=full(key)
            if source[0].covers(key):
                order,diag=chain_permutation(sequences[key],mode,k,plan["perturbation_seed"],key)
                tensor=transform_extra_columns(tensor,int(config["input_dim"]),dim,order)
                diag.update({"side":side,"source_present":bool(tensor[0,-1].item())})
                diagnostics.append(diag)
            idx=torch.as_tensor(indices.astype(np.int64),device=device)
            values.append(tensor.index_select(0,idx).unsqueeze(0))
        return values,contact,diagnostics
    def forward(ag,ab):
        # Labels never enter model.forward. Candidate labels are read below.
        return model(ag,ab,torch.ones(ag.shape[:2],dtype=torch.bool,device=device),
                     torch.ones(ab.shape[:2],dtype=torch.bool,device=device))
    def ap(y,p):
        return float(average_precision_score(y,p)) if np.any(y) else None
    records, rank_records = [],[]
    max_repeat_error, max_oracle_error = 0.0,0.0
    with torch.inference_mode():
        for pos,iid in enumerate(ids):
            reference = None
            for mode,k in plan["conditions"]:
                values,contact,diagnostics=features(iid,mode,k)
                result=forward(*values)
                probabilities=result["pair_probabilities"][0]
                if mode == "identity":
                    reference=probabilities.clone()
                    repeat=forward(*values)["pair_probabilities"][0]
                    error=float((repeat-reference).abs().max())
                    max_repeat_error=max(max_repeat_error,error)
                    if error>1e-6: raise RuntimeError("repeatability gate failed")
                if mode == "oracle_restore":
                    error=float((probabilities-reference).abs().max())
                    max_oracle_error=max(max_oracle_error,error)
                    if error>1e-6: raise RuntimeError("inverse-permutation recovery gate failed")
                labels,mask=contact.dense()
                if not bool(mask.all()):
                    raise RuntimeError("sparse candidate support needs an explicit inference-mask contract")
                score=probabilities.cpu().numpy()[mask]
                truth=labels[mask].astype(np.uint8)
                epi=result["epitope_probability"][0].cpu().numpy()
                para=result["paratope_probability"][0].cpu().numpy()
                outer=(epi[:,None]*para[None,:])[mask]
                ey,py=contact.node_labels()
                records.append({"instance_id":iid,"cluster":str(components[iid]),
                    "encoder":encoder,"seed":task["training_seed"],"condition":f"{mode}:{k}",
                    "pair_ap":ap(truth,score),"outer_product_ap":ap(truth,outer),
                    "epitope_ap":ap(ey,epi),"paratope_ap":ap(py,para),
                    "pair_count":int(mask.sum()),"positive_count":int(truth.sum()),
                    "diagnostics":diagnostics})
            if (pos+1)%25==0 or pos+1==len(ids):
                print(f"contact {encoder} seed={task['training_seed']} {pos+1}/{len(ids)}",flush=True)
        # Bounded reciprocal retrieval; candidates are unknown, not negative labels.
        known=set(known_rows.values())
        anchors=sorted(ids,key=lambda i:hashlib.sha256(i.encode()).hexdigest())[:plan["ranking_anchor_count"]]
        for iid in anchors:
            for swap in ("antigen","antibody"):
                col="ag_sequence_sha256" if swap=="antigen" else "ab_sequence_sha256"
                other="ab_sequence_sha256" if swap=="antigen" else "ag_sequence_sha256"
                anchor_key=str(frame.loc[iid,col]); length=sum(map(len,sequences[anchor_key]))
                candidates,seen=[],{anchor_key}
                choices=sorted((j for j in ids if j!=iid),key=lambda j:(
                    abs(np.log(sum(map(len,sequences[str(frame.loc[j,col])]))/length)),j))
                for j in choices:
                    key=str(frame.loc[j,col]); fixed=str(frame.loc[iid,other])
                    pair=(fixed,key) if swap=="antigen" else (key,fixed)
                    if key in seen or pair in known or str(components[j])==str(components[iid]): continue
                    seen.add(key); candidates.append(j)
                    if len(candidates)==plan["ranking_distractor_count"]: break
                if not candidates: continue
                for condition in plan["ranking_condition_names"]:
                    mode,k=condition.split(":"); k=int(k)
                    base,_,_=features(iid,mode,k)
                    scores=[]
                    for j in [iid]+candidates:
                        donor,_,_=features(j,mode,k)
                        ag,ab=(donor[0],base[1]) if swap=="antigen" else (base[0],donor[1])
                        result=forward(ag,ab)
                        scores.append(float(result["complex_score"].item()))
                    rank_records.append({"anchor":iid,"swap":swap,"condition":condition,
                        "candidates":[iid]+candidates,"scores":scores,
                        **ranking_summary(scores,[True]+[False]*len(candidates))})
    for name,value in model.state_dict().items():
        if not torch.equal(value.cpu(),parameter_state[name]):
            raise RuntimeError("evaluation mutated model state")
    if digest_file(checkpoint_path)!=task["checkpoint_sha256"]:
        raise RuntimeError("checkpoint bytes changed while evaluating")
    # Detect concurrent edits to any selected inputs; results stay unpromoted on failure.
    for path,sha in plan["input_sha256"].items():
        if digest_file(Path(path))!=sha: raise RuntimeError(f"input changed during evaluation: {path}")
    result={"status":"PASS_ENGINEERING_SMOKE" if args.smoke else "COMPLETE_DEVELOPMENT_MEASUREMENT",
            "task":task,"plan_sha256":args.plan_sha256,"case_count":len(ids),
            "records":records,"ranking":rank_records,"source_sha256":plan["source_sha256"],
            "max_repeat_error":max_repeat_error,"max_oracle_error":max_oracle_error,
            "wall_seconds":time.monotonic()-began,"torch":torch.__version__,
            "cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(),
            "peak_allocated_bytes":torch.cuda.max_memory_allocated(),
            "python":platform.python_version(),"store_coverage":store.summary(),
            "test_labels_read":False,"training_performed":False,"promotion_authorized":False,
            "scope":plan["evidence_role"]}
    write_new_json(out/"RESULT.json",result)
    print(json.dumps({k:result[k] for k in ["status","case_count","wall_seconds","gpu",
                      "max_repeat_error","max_oracle_error","peak_allocated_bytes"]}),flush=True)


def main():
    p=argparse.ArgumentParser(__doc__)
    sub=p.add_subparsers(dest="command",required=True)
    f=sub.add_parser("freeze"); f.add_argument("--plan",type=Path,required=True)
    r=sub.add_parser("run"); r.add_argument("--plan",type=Path,required=True)
    r.add_argument("--plan-sha256",required=True); r.add_argument("--task",type=int,required=True)
    r.add_argument("--output",type=Path,required=True); r.add_argument("--smoke",action="store_true")
    a=p.parse_args()
    if a.command=="freeze": freeze(a)
    else:
        try: run(a)
        except Exception as error:
            target=a.output/(f"smoke_{a.task}" if a.smoke else f"task_{a.task}")/"FAILURE.json"
            if target.parent.exists() and not target.exists():
                write_new_json(target,{"status":"FAILED","error":repr(error),"promotion_authorized":False})
            raise

if __name__=="__main__": main()
