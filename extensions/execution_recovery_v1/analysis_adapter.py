#!/usr/bin/env python3
"""Repair only the frozen layer-sweep report schema, outside its source tree.

This adapter keeps the registered AP calculation, bootstrap and layer rule by
loading the exact original aggregator and replacing its metric-reader function.
It never changes training source, checkpoints, metadata or validation labels.
"""
from __future__ import annotations
import argparse
import importlib.util
import math
from pathlib import Path
import sys
import time
from af3 import digest, require_digest, read_json, new_json


def unit_interval(value):
    if type(value) not in (int,float) or not math.isfinite(value) or not 0<=value<=1:
        raise ValueError('expected finite unit-interval metric')
    return float(value)


def metric_snapshot(d):
    if d.get('stage')!='evaluate' or d.get('split')!='validation' or d.get('instance_count')!=217:
        raise ValueError('only the full registered 217-complex validation evaluation is eligible')
    retrieval=d['partner_retrieval']
    if retrieval.get('instance_count')!=217 or retrieval.get('candidate_count_per_instance')!=10:
        raise ValueError('partner-retrieval panel differs from the formal contract')
    if 'mrr' in retrieval:
        raise ValueError('ambiguous/unregistered alternate MRR field')
    return {'instance_count':217,
        'pair_macro_ap':unit_interval(d['pair']['macro']['auprc']),
        'pair_micro_ap':unit_interval(d['pair']['micro']['auprc']),
        'pair_macro_auroc':unit_interval(d['pair']['macro']['auroc']),
        'epitope_macro_ap':unit_interval(d['nodes']['epitope']['macro']['auprc']),
        'paratope_macro_ap':unit_interval(d['nodes']['paratope']['macro']['auprc']),
        'partner_mrr':unit_interval(retrieval['mean_reciprocal_rank']),
        'partner_recall_at_1':unit_interval(retrieval['recall_at_1'])}


def load_authority(path,sha):
    require_digest(path,sha)
    p=read_json(path)
    if p['test_authorized'] or p['ood_authorized'] or len(p['tasks'])!=36:
        raise ValueError('unexpected task/closed-data authority')
    if [t['task_id'] for t in p['tasks']] != list(range(36)):
        raise ValueError('task IDs are not the complete frozen grid')
    root=Path(p['source_root'])
    for relative,expected in p['source_sha256'].items(): require_digest(root/relative,expected)
    for file,expected in p['input_sha256'].items(): require_digest(file,expected)
    return p


def collect(plan,plan_sha,attempt,require_complete=False):
    rows=[]; missing=[]; inputs={}; params={}
    for task in plan['tasks']:
        run=Path(attempt)/'runs'/('task_%02d'%task['task_id'])
        if (run/'FAILURE.json').exists():
            raise ValueError('formal task has preserved failure: '+str(run))
        cp=run/'COMPLETE.json'
        if not cp.is_file(): missing.append(task['task_id']); continue
        d=read_json(cp)
        if d['status']!='COMPLETE_DEVELOPMENT_TRAINING' or d['task']!=task or d['plan_sha256']!=plan_sha:
            raise ValueError('terminal receipt/task/plan mismatch: '+str(cp))
        if d['test_labels_read'] or d['ood_labels_read'] or d['smoke']:
            raise ValueError('smoke or closed-data receipt in formal collection')
        inputs[str(cp)]=digest(cp)
        snapshots={}
        for side,weight in (('best','best.pt'),('last','last.pt')):
            path=run/('validation_'+side)/'evaluation.json'
            ev=read_json(path)
            expected=d[side+'_checkpoint_sha256']
            if ev['checkpoint_sha256']!=expected:
                raise ValueError('evaluation/checkpoint receipt mismatch')
            require_digest(run/'train'/weight,expected)
            if ev['split_manifest_sha256']!=plan['input_sha256'][plan['paths']['split_manifest']]:
                raise ValueError('evaluation uses a different split authority')
            snapshots[side]=metric_snapshot(ev)
            if snapshots[side]['pair_macro_ap']!=d[side+'_primary_pair_ap']:
                raise ValueError('primary result differs from terminal receipt')
            inputs[str(path)]=digest(path)
        if d['parameter_count']<=0:
            raise ValueError('invalid downstream parameter count')
        params.setdefault(task['family'],set()).add(d['parameter_count'])
        rows.append({'task':task,'snapshots':snapshots,'parameter_count':d['parameter_count']})
    if any(len(values)!=1 for values in params.values()):
        raise ValueError('real/control downstream parameter counts differ inside one family')
    if require_complete and missing:
        raise RuntimeError('layer sweep incomplete; missing tasks '+str(missing))
    return {'status':'READY_FOR_FULL_ANALYSIS' if not missing else 'INCOMPLETE_NO_LAYER_SELECTION',
            'terminal_receipt_count':len(rows),'missing_task_ids':missing,'rows':rows,
            'checked_input_sha256':inputs,'test_labels_read':False,'promotion_authorized':False}


def legacy_module(plan):
    path=Path(plan['source_root'])/'extensions/frontier_multimodel_v1/aggregate_structural_training.py'
    require_digest(path,plan['source_sha256']['extensions/frontier_multimodel_v1/aggregate_structural_training.py'])
    spec=importlib.util.spec_from_file_location('frozen_layer_aggregator',path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module,path


def run_analysis(plan_path,sha,attempt,out):
    plan=load_authority(plan_path,sha)
    data=collect(plan,sha,attempt,True)
    original,source=legacy_module(plan)
    # Explicit adapter: scientific algorithm is imported from hashed frozen source.
    # No file in that source tree is rewritten or modified.
    original.metric_snapshot=metric_snapshot
    out=Path(out); out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists(): raise FileExistsError(out)
    receipt=out.parent/'ADAPTER_START.json'
    new_json(receipt,{'status':'STARTED','adapter_sha256':digest(__file__),
             'helper_sha256':digest(Path(__file__).parent/'af3.py'),
             'original_aggregator_sha256':digest(source),'legacy_plan_sha256':sha,
             'checked_input_sha256':data['checked_input_sha256'],
             'change':'partner_retrieval.mean_reciprocal_rank replaces nonexistent mrr field; AP/bootstrap/selection algorithm unchanged'})
    saved=sys.argv[:]
    try:
        sys.argv=[str(source),'--plan',str(plan_path),'--plan-sha256',sha,
                  '--attempt-root',str(attempt),'--output',str(out)]
        original.main()
    finally:
        sys.argv=saved
    again=collect(plan,sha,attempt,True)
    if again['checked_input_sha256']!=data['checked_input_sha256']:
        raise ValueError('source receipts changed during aggregation')
    new_json(out.parent/'ADAPTER_COMPLETE.json',{'status':'COMPLETE_SCHEMA_CORRECTED_ANALYSIS',
             'output_sha256':digest(out),'adapter_sha256':digest(__file__),
             'legacy_plan_sha256':sha,'formal_tasks':36,'test_labels_read':False})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['inspect','run'])
    p.add_argument('--plan',required=True);p.add_argument('--plan-sha256',required=True)
    p.add_argument('--attempt-root',required=True);p.add_argument('--output',required=True)
    a=p.parse_args()
    if a.action=='run': run_analysis(a.plan,a.plan_sha256,a.attempt_root,a.output)
    else:
        plan=load_authority(a.plan,a.plan_sha256)
        d=collect(plan,a.plan_sha256,a.attempt_root)
        d['plan_sha256']=a.plan_sha256;d['at_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
        new_json(a.output,d)
        print(d['status'],d['terminal_receipt_count'],'/36')

if __name__=='__main__': main()
