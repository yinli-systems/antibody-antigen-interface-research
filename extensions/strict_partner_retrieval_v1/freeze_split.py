#!/usr/bin/env python3
"""Freeze strict partner-retrieval libraries from an audited nested split.

This tool never upgrades a non-authorizing split receipt.  It can publish a
PREPARED_NOT_AUTHORIZED artifact from a passing precursor audit; model execution
requires a separately supplied Gate-0 authority receipt that explicitly binds the
same assignment and audit bytes.
"""
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path
from typing import Any

CONTRACT_FORMAT = "strict_partner_retrieval_contract_v1"
OUT_FORMAT = "strict_partner_retrieval_frozen_split_v1"

def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""): h.update(b)
    return h.hexdigest()

def canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode("ascii")).hexdigest()

def verify_scientific_hash(value:dict[str,Any],label:str)->str:
    claimed=value.get("scientific_payload_sha256")
    if not isinstance(claimed,str) or len(claimed)!=64:raise ValueError(f"{label} scientific hash missing")
    material=dict(value);material.pop("scientific_payload_sha256",None)
    actual=canonical_sha(material)
    if actual!=claimed:raise ValueError(f"{label} scientific hash mismatch")
    return actual

def membership_sha(values) -> str:
    return hashlib.sha256("".join(f"{x}\n" for x in sorted(set(values))).encode()).hexdigest()

def load(path: Path) -> dict[str,Any]:
    def pairs(items):
        out={}
        for k,v in items:
            if k in out: raise ValueError(f"duplicate JSON key {k!r} in {path}")
            out[k]=v
        return out
    value=json.loads(path.read_text(),object_pairs_hook=pairs,parse_constant=lambda x:(_ for _ in ()).throw(ValueError(f"nonfinite {x}")))
    if not isinstance(value,dict): raise ValueError(f"{path} must be object")
    return value

def chance(N:int,m:int,K:int)->float:
    if N<1 or m<1 or m>N: raise ValueError("invalid library")
    K=min(K,N)
    if K>=N-m+1: return 1.0
    return 1.0-math.comb(N-m,K)/math.comb(N,K)

def pair_payloads(fold:dict[str,Any], key:str)->list[dict[str,Any]]:
    xs=fold.get(key)
    if not isinstance(xs,list): raise ValueError(f"{key} missing")
    out=[]
    for x in xs:
        if not isinstance(x,dict): raise ValueError(f"bad {key} row")
        for k in ("pair_id","antibody_id","antigen_id"):
            if not isinstance(x.get(k),str) or not x[k]: raise ValueError(f"bad {key}.{k}")
        out.append(x)
    return out

def library(fold:dict[str,Any], label:str)->dict[str,Any]:
    val=pair_payloads(fold,"validation_pairs")
    train=pair_payloads(fold,"training_pairs")
    quar=pair_payloads(fold,"quarantined_pairs")
    va=sorted({x["antibody_id"] for x in val}); vg=sorted({x["antigen_id"] for x in val})
    positives={}
    for x in val: positives.setdefault(x["antibody_id"],set()).add(x["antigen_id"])
    positives_t={}
    for x in val: positives_t.setdefault(x["antigen_id"],set()).add(x["antibody_id"])
    if set(positives)!=set(va) or set(positives_t)!=set(vg): raise ValueError(f"{label}: zero-positive validation endpoint")
    ab_queries=[]
    for q in va:
        p=sorted(positives[q]); N=len(vg);m=len(p)
        ab_queries.append({"query_id":q,"candidate_count":N,"positive_ids":p,"positive_count":m,
                           "chance":{"r1":chance(N,m,1),"r5":chance(N,m,5),"r10":chance(N,m,10)}})
    ag_queries=[]
    for q in vg:
        p=sorted(positives_t[q]);N=len(va);m=len(p)
        ag_queries.append({"query_id":q,"candidate_count":N,"positive_ids":p,"positive_count":m,
                           "chance":{"r1":chance(N,m,1),"r5":chance(N,m,5),"r10":chance(N,m,10)}})
    c=fold.get("counts",{})
    U=int(c.get("universe_pair_count",len(train)+len(val)+len(quar)))
    if len(train)+len(val)+len(quar)!=U: raise ValueError(f"{label}: pair accounting")
    role_ids={
      "training":[x["pair_id"] for x in train],"validation":[x["pair_id"] for x in val],
      "quarantine":[x["pair_id"] for x in quar]
    }
    return {
      "label":label,
      "counts":{
        "universe_pairs":U,"training_pairs":len(train),"validation_pairs":len(val),"quarantined_pairs":len(quar),
        "pair_retention_fraction":len(train)/U if U else None,"quarantine_fraction":len(quar)/U if U else None,
        "validation_antibodies":len(va),"validation_antigens":len(vg),
        "training_antibodies":int(c.get("training_antibody_count",len({x["antibody_id"] for x in train}))),
        "training_antigens":int(c.get("training_antigen_count",len({x["antigen_id"] for x in train}))),
        "universe_antibodies":int(c.get("universe_antibody_count",0)),
        "universe_antigens":int(c.get("universe_antigen_count",0)),
        "training_source_rows":int(c.get("training_source_row_count",0)),
        "validation_source_rows":int(c.get("validation_source_row_count",0)),
        "quarantined_source_rows":int(c.get("quarantined_source_row_count",0)),
      },
      "memberships":{
        "training_pair_sha256":membership_sha(role_ids["training"]),
        "validation_pair_sha256":membership_sha(role_ids["validation"]),
        "quarantined_pair_sha256":membership_sha(role_ids["quarantine"]),
        "validation_antibody_sha256":membership_sha(va),
        "validation_antigen_sha256":membership_sha(vg),
      },
      "antibody_to_antigen":{"candidate_ids":vg,"queries":ab_queries,
        "mean_random_chance":{k:sum(q["chance"][k] for q in ab_queries)/len(ab_queries) for k in ("r1","r5","r10")}},
      "antigen_to_antibody":{"candidate_ids":va,"queries":ag_queries,
        "mean_random_chance":{k:sum(q["chance"][k] for q in ag_queries)/len(ag_queries) for k in ("r1","r5","r10")}},
      "semantics":{"nonpositive_cross_pairs":"unlabeled_ranking_candidates","measured_nonbinder_claim":False}
    }

def gate0_ok(g:dict[str,Any], assignment_sha:str,audit_sha:str)->bool:
    # This schema is deliberately strict and is not produced by the precursor auditor.
    return (
      g.get("format_version")=="strict_partner_retrieval_gate0_authority_v1"
      and g.get("status")=="AUTHORIZED_FOR_STRICT_PARTNER_RETRIEVAL_TRAINING"
      and g.get("scope")=="authority_train_development_only"
      and g.get("authorized_for_training") is True
      and g.get("authorized_for_nested_model_selection") is True
      and g.get("outer_evaluation_authorized") is False
      and g.get("full_v0_confirmatory_gate0_claimed") is False
      and g.get("sealed_authority_validation_access") is False
      and g.get("test_access") is False and g.get("ood_access") is False
      and g.get("assignment_sha256")==assignment_sha and g.get("independent_audit_sha256")==audit_sha
      and isinstance(g.get("authority_sha256"),str) and len(g["authority_sha256"])==64
    )

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--contract",type=Path,required=True);ap.add_argument("--assignment",type=Path,required=True)
    ap.add_argument("--audit",type=Path,required=True);ap.add_argument("--gate0-authority",type=Path)
    ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    contract=load(a.contract);assignment=load(a.assignment);audit=load(a.audit)
    if contract.get("format_version")!=CONTRACT_FORMAT: raise ValueError("contract format")
    if assignment.get("artifact_role")!="externally_pinned_core_feasibility_replay" or assignment.get("stage")!="completed_externally_pinned_core_feasibility_replay": raise ValueError("assignment is not a complete pinned replay")
    if assignment.get("core_feasibility_constraints_passed") is not True or assignment.get("construction_invariants_passed") is not True: raise ValueError("assignment feasibility/invariants failed")
    for key in ("authorized_for_training","authorized_for_model_selection","authorized_for_nested_model_selection","split_domain_gate0_requirement_satisfied"):
        if assignment.get(key) is not False: raise ValueError(f"assignment falsely enables {key}")
    assignment_scientific=verify_scientific_hash(assignment,"assignment")
    if audit.get("valid_audit_execution") is not True or audit.get("core_feasibility_precursor_passed") is not True: raise ValueError("precursor audit not passed")
    if audit.get("approval_status")!="core_feasibility_precursor_passed_non_authorizing": raise ValueError("unexpected audit approval status")
    for key in ("authorized_for_training","authorized_for_model_selection","authorized_for_nested_model_selection","split_domain_gate0_requirement_satisfied"):
        if audit.get(key) is not False: raise ValueError(f"audit falsely enables {key}")
    gates=audit.get("independent_gates")
    if not isinstance(gates,dict) or not gates or not all(value is True for value in gates.values()): raise ValueError("independent audit gates not all true")
    if int(audit.get("boundary_count",-1))!=12 or audit.get("exhaustive_accounting",{}).get("threshold_crossing_edge_count")!=0: raise ValueError("boundary audit failed")
    if audit.get("data_firewall",{}).get("validation_or_test_labels_predictions_or_scores_read")!=0: raise ValueError("split audit read forbidden data")
    audit_scientific=verify_scientific_hash(audit,"audit")
    asha=sha256_file(a.assignment); audsha=sha256_file(a.audit)
    audit_assignment=audit.get("inputs",{}).get("assignment",{})
    if audit_assignment.get("sha256")!=asha: raise ValueError("audit does not bind assignment file bytes")
    if audit.get("assignment_scientific_payload_sha256")!=assignment_scientific: raise ValueError("audit does not bind assignment scientific payload")
    aw=assignment.get("workflow_identity",{}); uw=audit.get("workflow_identity",{})
    if aw.get("joint_assignment_mode")!="externally_pinned_replay" or uw.get("joint_assignment_mode")!="externally_pinned_replay": raise ValueError("workflow is not pinned replay")
    if aw.get("pinned_replay_id")!=uw.get("pinned_replay_id") or aw.get("source_provisional_attempt_id")!=uw.get("source_provisional_attempt_id"): raise ValueError("audit/replay workflow identity mismatch")
    outer=assignment.get("construction",{}).get("outer_level",{})
    folds=outer.get("folds")
    if not isinstance(folds,list) or len(folds)!=3: raise ValueError("need 3 outer folds")
    libs=[]
    for oi,fold in enumerate(folds):
        libs.append(library(fold,f"outer_{oi}"))
        inner=fold.get("inner_level",{}).get("folds")
        if not isinstance(inner,list) or len(inner)!=3: raise ValueError(f"outer_{oi} missing 3 inner folds")
        for ii,x in enumerate(inner): libs.append(library(x,f"outer_{oi}_inner_{ii}"))
    if len(libs)!=12: raise AssertionError
    g=None; authorized=False
    if a.gate0_authority is not None:
        g=load(a.gate0_authority);authorized=gate0_ok(g,asha,audsha)
        if not authorized: raise ValueError("Gate-0 authority supplied but invalid")
    payload={
      "format_version":OUT_FORMAT,
      "status":"FROZEN_SPLIT_AUTHORIZED" if authorized else "PREPARED_SPLIT_NOT_AUTHORIZED",
      "model_execution_authorized":authorized,
      "training_and_model_selection_authorized":authorized,
      # H0 is a no-training, no-label baseline.  It is authorized only because
      # this freezer has already required the independent precursor audit to pass.
      "h0_execution_authorized":True,
      "fold_cache_materialization_authorized":True,
      # Gate 0 deliberately cannot grant outer evaluation.  A later, separately
      # frozen model-selection authority is required for that transition.
      "outer_evaluation_authorized":False,
      "contract":{"path":str(a.contract.resolve()),"sha256":sha256_file(a.contract)},
      "assignment":{"path":str(a.assignment.resolve()),"sha256":asha,"scientific_payload_sha256":assignment_scientific},
      "independent_audit":{"path":str(a.audit.resolve()),"sha256":audsha,"scientific_payload_sha256":audit_scientific},
      "gate0_authority":None if g is None else {"path":str(a.gate0_authority.resolve()),"sha256":sha256_file(a.gate0_authority),"authority_sha256":g["authority_sha256"]},
      "fold_count":12,"libraries":libs,
      "aggregate_disclosure":{
        "sum_outer_quarantined_pairs":sum(x["counts"]["quarantined_pairs"] for x in libs if x["label"].count("_")==1),
        "outer_pair_retention_fractions":[x["counts"]["pair_retention_fraction"] for x in libs if x["label"].count("_")==1],
        "outer_quarantine_fractions":[x["counts"]["quarantine_fraction"] for x in libs if x["label"].count("_")==1],
      },
      "claim_boundary":"authority-train development; foldwise direct-80/80 four-view dual-axis C3/S2-like; unobserved cross-pairs are unlabeled ranking candidates",
      "test_access":False,"ood_access":False
    }
    payload["scientific_payload_sha256"]=canonical_sha(payload)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open("x",encoding="utf-8") as handle:
        handle.write(json.dumps(payload,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps({"status":payload["status"],"training_authorized":payload["training_and_model_selection_authorized"],"outer_evaluation_authorized":payload["outer_evaluation_authorized"],"output":str(a.output),"scientific_payload_sha256":payload["scientific_payload_sha256"]},indent=2))
if __name__=="__main__": main()
