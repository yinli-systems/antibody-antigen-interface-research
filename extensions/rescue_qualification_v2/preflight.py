#!/usr/bin/env python3
"""Read-only rescue input qualification. No features/labels/model are scored."""
from __future__ import annotations
import argparse, csv, hashlib, json, time
from pathlib import Path
from qualification_core import digest,new_json


def inspect(plan_path,expected):
    if digest(plan_path)!=expected: raise ValueError('plan SHA mismatch')
    plan=json.loads(Path(plan_path).read_text())
    if plan['schema']!='learned_correspondence_rescue_v1': raise ValueError('wrong plan schema')
    if any(plan.get(k) is not False for k in ('test_access','ood_access','contact_metric_access_during_training')):
        raise ValueError('invalid access authority')
    for p,h in plan['input_sha256'].items():
        if digest(p)!=h: raise ValueError('source/input hash drift: '+p)
    for row in plan['downstream_checkpoints']:
        if digest(row['path'])!=row['sha256']: raise ValueError('downstream checkpoint hash drift')
    pins=list(plan['input_sha256'])
    split_path=next(p for p in pins if Path(p).name=='double_remote_split.json')
    canonical_path=next(p for p in pins if Path(p).name=='canonical_manifest.csv')
    split=json.loads(Path(split_path).read_text())['splits']
    train=set(split['train']); val=set(split['validation']); test=set(split['test'])
    if len(val)!=217 or len(train)!=3229 or (val&train) or (val&test) or (train&test):
        raise ValueError('split identity/coverage violation')
    rows={}
    with Path(canonical_path).open(newline='') as handle:
        for row in csv.DictReader(handle):
            if row['INSTANCE'] in rows: raise ValueError('duplicate canonical ID')
            rows[row['INSTANCE']]=row
    hashed=lambda s:hashlib.sha256(s.encode('ascii')).hexdigest()
    def values(ids):
        full={};parts=set();assignments={}
        for iid in sorted(ids):
            row=rows[iid]; h,l,a=[(row.get(k) or '').strip() for k in ('vh_seq','vl_seq','ag_seq')]
            if not h or not a: raise ValueError('missing required sequence')
            assignments[iid]={}
            for side,pieces in [('antibody',tuple(x for x in (h,l) if x)),('antigen',(a,))]:
                key=hashed(''.join(pieces)); part_keys=[hashed(x) for x in pieces]
                if key in full and full[key]!=part_keys: raise ValueError('chain segmentation ambiguous')
                full[key]=part_keys;parts.update(part_keys);assignments[iid][side]=key
        return full,parts,assignments
    tf,tp,_=values(train); vf,vp,instances=values(val)
    if set(tf)&set(vf) or tp&vp: raise ValueError('exact sequence/chain-part leakage')
    feature_root=Path('/ssd/scxi253/repr_extract/out/esm_if1')
    covered={};absent={};available_bytes=0
    # Metadata only: no feature tensors are loaded and no contact files opened.
    for key,parts in {**tf,**vf}.items():
        missing=[str(feature_root/p[:2]/(p+'.pt')) for p in parts
                 if not (feature_root/p[:2]/(p+'.pt')).is_file()]
        if missing: absent[key]=missing
        else: covered[key]=parts
    for role,keys in [('train',tf),('validation',vf)]:
        if set(plan[role+'_keys'])!={k for k in keys if k in covered}:
            raise ValueError('frozen rescue availability disagrees with current stores')
    affected=[]
    for iid,sides in instances.items():
        bad={side:key for side,key in sides.items() if key in absent}
        if bad: affected.append({'instance_id':iid,'missing_side_keys':bad})
    return {'status':'PASS_METADATA_PREFLIGHT_WITH_EVALUATION_COVERAGE_WARNING',
            'at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            'frozen_rescue_plan_sha256':expected,'original_training_source_commit':plan['source_commit'],
            'train_complexes':len(train),'validation_complexes':len(val),
            'train_full_keys':len(tf),'validation_full_keys':len(vf),
            'train_chain_parts':len(tp),'validation_chain_parts':len(vp),
            'train_validation_full_key_overlap':0,'train_validation_chain_part_overlap':0,
            'covered_train_keys':len(plan['train_keys']),'covered_validation_keys':len(plan['validation_keys']),
            'absent_validation_keys':{k:absent[k] for k in vf if k in absent},
            'affected_validation_complexes':affected,
            'legacy_evaluator_risk':'load_full_features raises for unavailable source; all-one flags are not historical missingness',
            'supplemental_resolution':'retain every complex; exact zero-feature/zero-presence behavior of original CompositeEmbeddingStore',
            'homology_leakage_audited':False,'pretrained_encoder_contamination_audited':False,
            'feature_tensor_bytes_hashed':False,'input_source_and_downstream_checkpoint_hashes_verified':True,
            'contact_labels_read':False,'test_labels_read':False,'ood_labels_read':False,
            'model_inference_performed':False,'scheduler_changes':False}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--plan-sha256',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); result=inspect(a.plan,a.plan_sha256);new_json(a.output,result)
    print(json.dumps(result,indent=2))
