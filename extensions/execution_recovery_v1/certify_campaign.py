#!/usr/bin/env python3
"""Certify 40 existing AF3 predictions and assemble; never execute a model.

Run only after the existing inference array completes. The caller supplies the
external plan digest, array ID and a new output root. Old sources stay frozen.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import af3
import assemble
import semantic_audit


def require_array_success(text: str, array_id: str, task_ids: list[int]):
    rows={}
    for line in text.splitlines():
        parts=line.split('|')
        if len(parts)!=3:raise ValueError('unexpected scheduler accounting columns')
        job,state,exitcode=parts
        if job in rows:raise ValueError('duplicate scheduler record')
        rows[job]=(state,exitcode)
    for task_id in task_ids:
        key=f'{array_id}_{task_id}'
        if rows.get(key)!=('COMPLETED','0:0'):
            raise RuntimeError('inference array is not completely successful: '+key)


def run(plan_path: Path, expected: str, array_id: str, output: Path):
    original,plan=semantic_audit.load_execution(plan_path,expected)
    submitted=af3.read_json(plan_path.parent/'remaining_SUBMITTED.json')
    if (submitted.get('job_id')!=array_id or submitted.get('plan_sha256')!=expected or
        submitted.get('tasks')!=list(range(1,40)) or len(plan['tasks'])!=40):
        raise ValueError('array identity/task grid does not match submission receipt')
    accounting=subprocess.check_output(['sacct','-X','-j',array_id,'--noheader',
        '--parsable2','--format=JobID,State,ExitCode'],text=True,timeout=60)
    require_array_success(accounting,array_id,list(range(1,40)))
    # Require all terminal task receipts before creating any closure outputs.
    for task in plan['tasks']:
        root=original.task_dir(plan,task['task_id'])
        if (root/'FAILURE.json').exists() or not (root/'COMPLETE.json').is_file():
            raise RuntimeError('missing or failed task: '+str(task['task_id']))
    output=output.resolve()
    if output.is_relative_to(Path(plan['run_root']).resolve()):
        raise ValueError('closure must not be written into prediction outputs')
    output.mkdir(parents=True,exist_ok=False)
    code_files=('af3.py','semantic_audit.py','assemble.py','certify_campaign.py')
    code={f:af3.digest(Path(__file__).parent/f) for f in code_files}
    af3.new_json(output/'START.json',{'state':'AUDITING_ALL_40','at_utc':af3.utc(),
        'execution_plan_sha256':expected,'inference_array_id':array_id,
        'auditor_source_sha256':code,'inference_performed':False,'accounting':accounting})
    try:
        records=[]
        for task in plan['tasks']:
            task_id=task['task_id']
            report=semantic_audit.audit_existing(plan_path,expected,task_id)
            dest=output/'audits'/f'task_{task_id:03d}.json'
            af3.new_json(dest,report)
            records.append((task,report,str(dest),af3.digest(dest)))
            print('strict AF3 audit',task_id+1,'/40',flush=True)
        if len(records)!=40:raise ValueError('incomplete strict audits')

        def checked_loader(path,digest):
            _original,p=semantic_audit.load_execution(Path(path),digest)
            return p

        def certified_reader(p,digest):
            if digest!=expected or p!=plan:raise ValueError('assembly plan changed')
            for task,report,path,sha in records:
                af3.require_digest(path,sha)
                root=original.task_dir(p,task['task_id'])
                actual={k:v for k,v in semantic_audit.file_manifest(root).items()
                        if k.startswith('results/')}
                if actual!=report['output_sha256']:
                    raise ValueError('prediction changed after strict audit')
            return records

        assemble.assemble(str(plan_path),expected,output/'structure_authority',
                          plan_loader=checked_loader,receipt_loader=certified_reader)
        final=af3.read_json(output/'structure_authority/COMPLETE.json')
        if final.get('state')!='COMPLETE_217_CASE_STRUCTURE_AUTHORITY':
            raise ValueError('structure assembly did not close')
        # Bind code and all runtime resources again. No live job/feature source is written.
        for name,sha in code.items():af3.require_digest(Path(__file__).parent/name,sha)
        checked_loader(plan_path,expected)
        for resource in plan['large_runtime_files']:
            af3.require_digest(resource['path'],resource['sha256'])
        af3.new_json(output/'COMPLETE.json',{'state':'COMPLETE_STRICT_SEMANTIC_CLOSURE',
            'at_utc':af3.utc(),'execution_plan_sha256':expected,
            'inference_array_id':array_id,'audited_tasks':40,'audited_samples':1000,
            'auditor_source_sha256':code,
            'strict_receipt_sha256':{str(t['task_id']):sha for t,_,_,sha in records},
            'structure_manifest_sha256':final['combined_manifest_sha256'],
            'coverage_after':final['complete_cases_after'],'inference_performed':False,
            'experimental_contact_labels_read':False,'test_labels_read':False,
            'ood_labels_read':False,'layer_winner_selected':False,
            'source_predictions_modified':False,'protected_training_jobs_modified':False})
    except BaseException as error:
        af3.new_json(output/'FAILURE.json',{'state':'FAILED_STRICT_CLOSURE',
            'at_utc':af3.utc(),'error':repr(error),'inference_performed':False})
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--plan-sha256',required=True)
    p.add_argument('--inference-array-id',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if not a.inference_array_id.isdigit():raise ValueError('invalid array id')
    run(a.plan,a.plan_sha256,a.inference_array_id,a.output)

if __name__=='__main__':main()
