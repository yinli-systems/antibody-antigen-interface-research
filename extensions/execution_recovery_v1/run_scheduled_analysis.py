#!/usr/bin/env python3
"""Execute an externally SHA-bound analysis authority; never submit training."""
import argparse
import os
from pathlib import Path
from af3 import require_digest,read_json,new_json,digest
from analysis_adapter import run_analysis


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--authority',required=True);p.add_argument('--authority-sha256',required=True)
    args=p.parse_args()
    require_digest(args.authority,args.authority_sha256)
    a=read_json(args.authority)
    if a['schema']!='layer_schema_analysis_authority_v1' or a['training_submitted'] or a['test_access']:
        raise ValueError('unexpected analysis authority')
    if not os.environ.get('SLURM_JOB_ID') or int(os.environ.get('SLURM_RESTART_COUNT','0')):
        raise RuntimeError('scheduler allocation required; automatic restart rejected')
    for path,sha in a['executor_sha256'].items():require_digest(path,sha)
    require_digest(a['original_plan'],a['original_plan_sha256'])
    output=Path(a['output'])
    output.parent.mkdir(parents=True,exist_ok=False)
    new_json(output.parent/'SCHEDULER_START.json',{
        'state':'STARTED','job_id':os.environ['SLURM_JOB_ID'],
        'authority_sha256':args.authority_sha256,'executor_sha256':a['executor_sha256']})
    try:
        run_analysis(a['original_plan'],a['original_plan_sha256'],a['original_attempt'],output)
        for path,sha in a['executor_sha256'].items():require_digest(path,sha)
        new_json(output.parent/'SCHEDULER_COMPLETE.json',{
            'state':'COMPLETE_ANALYSIS_ONLY','job_id':os.environ['SLURM_JOB_ID'],
            'output_sha256':digest(output),'training_submitted':False,'test_labels_read':False})
    except BaseException as error:
        new_json(output.parent/'SCHEDULER_FAILURE.json',{'state':'FAILED','error':repr(error),
                   'job_id':os.environ['SLURM_JOB_ID'],'training_submitted':False})
        raise

if __name__=='__main__':main()
