#!/usr/bin/env python3
"""Strict complete-campaign aggregation, never silently dropping failed tasks."""
import argparse
import json
from pathlib import Path
import numpy as np
from core import digest_file, write_new_json, crossed_cluster_bootstrap

def main():
    p=argparse.ArgumentParser(); p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--results',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); plan=json.loads(a.plan.read_text()); digest=digest_file(a.plan)
    conditions=[f'{m}:{k}' for m,k in plan['conditions']]
    tasks={}; pins={}
    for task in plan['tasks']:
        path=a.results/f"task_{task['task_id']}"/'RESULT.json'
        if not path.exists(): raise RuntimeError(f"campaign incomplete: {path}")
        d=json.loads(path.read_text()); pins[str(path)]=digest_file(path)
        if d['status']!='COMPLETE_DEVELOPMENT_MEASUREMENT' or d['plan_sha256']!=digest or d['task']!=task:
            raise RuntimeError(f"mismatched task or source: {path}")
        expected={(i,c) for i in plan['ids'] for c in conditions}
        rows={(r['instance_id'],r['condition']):r for r in d['records']}
        if set(rows)!=expected or len(rows)!=len(d['records']):
            raise RuntimeError('missing/duplicate records')
        tasks[(task['encoder'],task['training_seed'])]=(rows,d)
    summaries=[]
    for encoder in sorted({t['encoder'] for t in plan['tasks']}):
        seeds=sorted(s for e,s in tasks if e==encoder)
        for condition in conditions:
            for metric in ['pair_ap','outer_product_ap','epitope_ap','paratope_ap']:
                eligible=[i for i in plan['ids'] if all(
                    tasks[(encoder,s)][0][(i,c)][metric] is not None
                    for s in seeds for c in [condition,'identity:0'])]
                if not eligible: continue
                clusters=[tasks[(encoder,seeds[0])][0][(i,'identity:0')]['cluster'] for i in eligible]
                x=np.array([[tasks[(encoder,s)][0][(i,condition)][metric]-
                             tasks[(encoder,s)][0][(i,'identity:0')][metric]
                             for i in eligible] for s in seeds])
                summaries.append({'encoder':encoder,'condition':condition,'metric':metric,
                    'excluded_zero_positive_or_undefined':len(plan['ids'])-len(eligible),
                    **crossed_cluster_bootstrap(x,clusters)})
    payload={'status':'COMPLETE_EXPLORATORY_DEVELOPMENT_ANALYSIS','plan_sha256':digest,
             'input_receipts_sha256':pins,'contrasts':summaries,
             'task_runtime_seconds':{f'{e}:{s}':d['wall_seconds'] for (e,s),(rows,d) in tasks.items()},
             'raw_condition_means':{
                 f'{e}:{s}:{c}':float(np.mean([r['pair_ap'] for (i,cc),r in rows.items() if cc==c and r['pair_ap'] is not None]))
                 for (e,s),(rows,d) in tasks.items() for c in conditions},
             'ranking_means':{
                 f'{e}:{s}:{swap}:{c}':{
                     m:float(np.mean([r[m] for r in d['ranking'] if r['swap']==swap and r['condition']==c]))
                     for m in ['mrr','recall_at_1','known_positive_minus_distractor_mean']}
                 for (e,s),(rows,d) in tasks.items() for swap in ['antigen','antibody']
                 for c in plan['ranking_condition_names']},
             'test_labels_read':False,'promotion_authorized':False,
             'limitations':['Only three fixed training seeds per encoder',
                 'Validation data previously used for development',
                 'Pointwise exploratory confidence intervals, no multiplicity-corrected claims',
                 'Unknown alternate partners are not experimentally verified non-binders',
                 'Inference-time robustness is not training-time efficacy',
                 'Oracle inverse is a sanity check, not learned realignment',
                 'Sequence-row interventions do not prove spatial causation']}
    write_new_json(a.output,payload)
    print(json.dumps({'status':payload['status'],'tasks':len(tasks),'contrasts':len(summaries),'output':str(a.output)}))
if __name__=='__main__': main()
