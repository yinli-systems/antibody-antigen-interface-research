#!/usr/bin/env python3
"""Independent, standard-library-only check against historical validation means."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

TOLERANCE = 1e-6

def audit(plan_path, references_path, results_path):
    plan=json.loads(plan_path.read_text())
    plan_sha=hashlib.sha256(plan_path.read_bytes()).hexdigest()
    references=json.loads(references_path.read_text())['references']
    if {r['task'] for r in references}!={t['task_id'] for t in plan['tasks']}:
        raise ValueError('reference task coverage mismatch')
    rows=[]
    for ref in references:
        path=results_path/f"task_{ref['task']}"/'RESULT.json'
        d=json.loads(path.read_text())
        if d['status']!='COMPLETE_DEVELOPMENT_MEASUREMENT' or d['plan_sha256']!=plan_sha:
            raise ValueError('result status or plan mismatch')
        identity=[r for r in d['records'] if r['condition']=='identity:0']
        if len(identity)!=len(plan['ids']) or {r['instance_id'] for r in identity}!=set(plan['ids']):
            raise ValueError('identity coverage mismatch')
        for metric in ['pair_ap','epitope_ap','paratope_ap']:
            valid=[r[metric] for r in identity if r[metric] is not None]
            observed=statistics.fmean(valid)
            diff=observed-ref[metric]
            rows.append({'task':ref['task'],'encoder':ref['encoder'],'seed':ref['seed'],
                         'metric':metric,'historical':ref[metric],'new_identity':observed,
                         'difference':diff,'eligible_cases':len(valid),'total_cases':len(identity),
                         'pass':abs(diff)<=TOLERANCE})
    return {'status':'PASS_IDENTITY_PARITY' if all(r['pass'] for r in rows) else 'HOLD_FOR_DIAGNOSIS',
            'tolerance':TOLERANCE,'max_absolute_difference':max(abs(r['difference']) for r in rows),
            'comparisons':rows,'plan_sha256':plan_sha,'test_results_read':False}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--references',type=Path,required=True)
    p.add_argument('--results',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); d=audit(a.plan,a.references,a.results)
    with a.output.open('x') as f: json.dump(d,f,indent=2,allow_nan=False); f.write('\n')
    print(json.dumps({'status':d['status'],'max_absolute_difference':d['max_absolute_difference'],
                      'comparisons':len(d['comparisons'])}))
    if d['status']!='PASS_IDENTITY_PARITY': raise SystemExit(2)

if __name__=='__main__': main()
