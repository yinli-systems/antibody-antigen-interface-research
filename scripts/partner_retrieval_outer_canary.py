#!/usr/bin/env python3
"""Solve the locked outer 3-fold retrieval split from a frozen graph authority.

This is a candidate-construction canary only.  It cannot authorize training.  A
successful candidate must be externally pinned/replayed and independently audited on
all four sequence views before any model process may see fold data.
"""
from __future__ import annotations

import argparse, hashlib, importlib.util, json, platform, time
from pathlib import Path
from typing import Any

from interaction_first import nested_joint_milp as njm
from interaction_first.decomposed_level_milp import build_level_feasibility_model, solve_level


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode("ascii")).hexdigest()


def load_builder(path: Path):
    spec=importlib.util.spec_from_file_location("nested_builder_for_outer_canary",path)
    if spec is None or spec.loader is None: raise ImportError(path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--graph-authority",type=Path,required=True)
    ap.add_argument("--graph-authority-sha256",required=True)
    ap.add_argument("--independent-audit",type=Path,required=True)
    ap.add_argument("--independent-audit-sha256",required=True)
    ap.add_argument("--builder",type=Path,required=True)
    ap.add_argument("--builder-sha256",required=True)
    ap.add_argument("--solver-source",type=Path,required=True)
    ap.add_argument("--solver-source-sha256",required=True)
    ap.add_argument("--time-limit-seconds",type=float,default=1800)
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    for p,h in [(args.graph_authority,args.graph_authority_sha256),(args.independent_audit,args.independent_audit_sha256),(args.builder,args.builder_sha256),(args.solver_source,args.solver_source_sha256)]:
        if sha(p)!=h: raise ValueError(f"hash mismatch: {p}")

    old=json.load(open(args.graph_authority)); audit=json.load(open(args.independent_audit))
    required_old={
        "independent_inventory_exactly_matches_assignment":True,
        "exhaustive_alignment_accounting_passed":True,
        "authority_train_only_firewall_passed":True,
        "all_four_views_completed_for_every_boundary":True,
    }
    for key,val in required_old.items():
        if audit["independent_gates"].get(key) is not val: raise ValueError(f"old authority gate unavailable: {key}")
    if audit["authority"]["authority_train_membership_sha256"]!=old["authority"]["authority_train_membership_sha256"]: raise ValueError("authority membership mismatch")
    if old["canonical_inventory"]["unique_exact_pair_count"]!=1830: raise ValueError("unexpected pair inventory")

    # Reconstruct each exact pair once from the complete role partition of old outer fold 0.
    fold0=old["construction"]["outer_level"]["folds"][0]
    pair_rows=[]
    for role in ("validation_pairs","training_pairs","quarantined_pairs"):
        pair_rows.extend(fold0[role])
    by_id={str(x["pair_id"]):x for x in pair_rows}
    if len(pair_rows)!=1830 or len(by_id)!=1830: raise ValueError("old role partition is not a complete exact-pair inventory")
    pairs=tuple(njm.CanonicalPair(pid,str(x["antibody_id"]),str(x["antigen_id"])) for pid,x in sorted(by_id.items()))
    antibody_ids={p.antibody_id for p in pairs}; antigen_ids={p.antigen_id for p in pairs}
    if len(antibody_ids)!=old["canonical_inventory"]["unique_antibody_entity_count"] or len(antigen_ids)!=old["canonical_inventory"]["unique_antigen_entity_count"]: raise ValueError("entity inventory mismatch")

    def graph(keys, universe):
        out={x:set() for x in universe}
        for key in keys:
            rec=old["exact_direct_graphs"][key]
            for edge in rec["direct_entity_edges"]:
                a=str(edge["left_entity_id"]);b=str(edge["right_entity_id"])
                if a not in out or b not in out: raise ValueError(f"foreign {key} edge")
                out[a].add(b);out[b].add(a)
        return {k:frozenset(v) for k,v in out.items()}
    ag=graph(("antigen",),antigen_ids)
    ab=graph(("vh","vl","concatenated_antibody"),antibody_ids)
    problem=njm.JointSplitProblem(pairs,ab,ag)

    builder=load_builder(args.builder)
    began=time.monotonic()
    model=build_level_feasibility_model(problem,policy=njm.LOCKED_OUTER_POLICY,level_label="outer")
    solved=solve_level(problem,policy=njm.LOCKED_OUTER_POLICY,level_label="outer",time_limit_seconds=args.time_limit_seconds)
    if solved.held_out_antibodies is None or solved.held_out_antigens is None:
        assembled=None
    else:
        level_policy=builder._level_policy_from_joint_policy(njm.LOCKED_OUTER_POLICY,len(pairs))
        selection={"algorithm":"decomposed_outer_exact_milp_canary_v1","score_free":True,"target_reached_in_every_fold":True,"matrix_sha256":model.matrix_sha256}
        # Source-row multiplicity is retained from old authority only for mechanical count reporting.
        groups=tuple(builder.PairGroup(pid,str(x["antibody_id"]),str(x["antigen_id"]),tuple(x["source_instance_ids"])) for pid,x in sorted(by_id.items()))
        assembled,_=builder._assemble_level_from_held_out_sets(groups,{k:set(v) for k,v in ab.items()},{k:set(v) for k,v in ag.items()},level_policy,held_out_antibodies=[set(x) for x in solved.held_out_antibodies],held_out_antigens=[set(x) for x in solved.held_out_antigens],selection=selection)
    selected={
        "antibody":[sorted(x) for x in (solved.held_out_antibodies or ())],
        "antigen":[sorted(x) for x in (solved.held_out_antigens or ())],
    }
    payload={
        "format_version":"partner_retrieval_outer_decomposed_canary_v1",
        "status":"CERTIFIED_OUTER_CANDIDATE_NOT_TRAINING_AUTHORITY" if solved.result.feasible_for_unscored_split_construction and assembled and assembled["all_level_hard_gates_passed"] else "NO_AUTHORIZED_OUTER_CANDIDATE",
        "claim_boundary":"score-free outer-fold feasibility canary only; no training/model selection authorization",
        "inputs":{"graph_authority":{"path":str(args.graph_authority),"sha256":args.graph_authority_sha256},"independent_old_audit":{"path":str(args.independent_audit),"sha256":args.independent_audit_sha256},"builder":{"path":str(args.builder),"sha256":args.builder_sha256},"decomposed_solver":{"path":str(args.solver_source),"sha256":args.solver_source_sha256}},
        "locked_policy_sha256":njm.LOCKED_JOINT_POLICY_SHA256,
        "outer_policy":njm.joint_policy_payload(njm.LOCKED_OUTER_POLICY,njm.LOCKED_INNER_POLICY)["outer"],
        "inventory":{"pairs":len(pairs),"antibodies":len(ab),"antigens":len(ag),"antibody_union_direct_edges":sum(len(v) for v in ab.values())//2,"antigen_direct_edges":sum(len(v) for v in ag.values())//2},
        "model":{"matrix_sha256":model.matrix_sha256,"estimate":model.estimate.__dict__},
        "solve":{"solver":solved.result.solver,"status":solved.result.solver_status,"message":solved.result.solver_message,"certified_optimal":solved.result.solver_certified_optimal,"mip_gap":solved.result.solver_reported_mip_gap,"exact_integer_row_audit_passed":solved.result.exact_integer_row_audit_passed,"feasible":solved.result.feasible_for_unscored_split_construction},
        "selected_heldout":selected,
        "selected_heldout_sha256":canonical_sha(selected),
        "assembled_outer_level":assembled,
        "elapsed_seconds":time.monotonic()-began,
        "runtime":{"python":platform.python_version(),"platform":platform.platform()},
        "old_assignment_authorized_for_training":False,
        "new_candidate_authorized_for_training":False,
        "required_next_gate":"solve all three inner levels under LOCKED_INNER_POLICY, externally pin complete nested membership, replay, then run independent v5 four-view audit",
        "test_labels_read":False,"validation_labels_read":False,"model_scores_read":False,
    }
    payload["scientific_payload_sha256"]=canonical_sha(payload)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"status":payload["status"],"matrix_sha256":model.matrix_sha256,"selected_heldout_sha256":payload["selected_heldout_sha256"],"elapsed_seconds":payload["elapsed_seconds"],"fold_counts":[x["counts"] for x in assembled["folds"]] if assembled else None},indent=2))

if __name__=="__main__": main()
