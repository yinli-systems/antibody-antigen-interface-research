#!/usr/bin/env python3
"""Freeze the exact nine-task validation-only score-dissociation campaign."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path

REPO=Path(__file__).resolve().parents[2]
import sys
sys.path[:0]=[str(REPO/'src'),str(REPO/'scripts'),str(REPO)]
from extensions.frontier_multimodel_v1.score_dissociation import sha256_file
from sabdab2_interaction_first import load_split_assignments


def exclusive(path:Path,payload:dict):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:
        json.dump(payload,f,indent=2,sort_keys=True,allow_nan=False)
        f.write('\n')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--commit',required=True)
    a=p.parse_args()
    import pandas as pd,torch
    D=Path('/ssd/scxi253/model_deployment/data')
    F=Path('/ssd/scxi253/frontier_multimodel_v1')
    R=Path('/ssd/scxi253/repr_ablation/out')
    split=D/'interaction_first/double_remote_split.json'
    prepare=D/'interaction_first_sabdab2/instances.parquet'
    canonical=D/'interaction_first/canonical_manifest.csv'
    structure=F/'data/structure_units_v2/STRUCTURE_UNITS_MANIFEST.json'
    embedding=D/'sppider_seq_sabdab2_embeddings'
    esmif1=Path('/ssd/scxi253/repr_extract/out/esm_if1')
    assignments=load_split_assignments(split)
    ids=sorted(i for i,r in assignments.items() if r=='validation')
    frame=pd.read_parquet(prepare,columns=['instance_id','ab_sequence_sha256','ag_sequence_sha256']).set_index('instance_id',verify_integrity=True).loc[ids]
    if len(ids)!=217 or frame.ab_sequence_sha256.nunique()!=122 or frame.ag_sequence_sha256.nunique()!=119:
        raise ValueError('candidate authority changed')
    source=REPO/'extensions/frontier_multimodel_v1/score_dissociation.py'
    protocol=REPO/'extensions/frontier_multimodel_v1/SCORE_DISSOCIATION_PROTOCOL.md'
    aggregate=REPO/'extensions/frontier_multimodel_v1/aggregate_score_dissociation.py'
    conditions=['identity:0','chain_shift:8','global_shuffle:0']
    tasks=[]
    for seed in [1,2,3]:
        checkpoint=R/f'esm_if1_only/seed_{seed}/train/best.pt'
        payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if payload['config']['seed']!=seed or payload['config']['split_manifest_sha256']!=sha256_file(split):
            raise ValueError('checkpoint authority mismatch')
        for condition in conditions:
            tasks.append({'task_id':len(tasks),'training_seed':seed,'condition':condition,'checkpoint':str(checkpoint),'checkpoint_sha256':sha256_file(checkpoint)})
    files=[split,prepare,canonical,structure,source,protocol,aggregate,REPO/'extensions/correspondence_v2/core.py']
    plan={'format_version':'score_dissociation_plan_v1','created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'commit':a.commit,
      'evidence_role':'development-validation score diagnosis; no model promotion','validation_instance_count':217,'unique_antibody_count':122,'unique_antigen_count':119,
      'unique_pair_count_per_task':14518,'tasks':tasks,'source_root':str(REPO),'prepare_dir':str(prepare.parent),'split_manifest':str(split),'canonical_manifest':str(canonical),
      'embedding_dir':str(embedding),'esm_if1_root':str(esmif1),'structure_manifest':str(structure),'protocol':str(protocol),'source':str(source),'aggregate':str(aggregate),
      'input_sha256':{str(path):sha256_file(path) for path in files},'contact_artifacts_authorized':False,'test_authorized':False,'training_authorized':False}
    exclusive(a.output,plan)
    print(json.dumps({'status':'PASS_PLAN_FROZEN','tasks':len(tasks),'sha256':sha256_file(a.output),'output':str(a.output)}))

if __name__=='__main__':main()
