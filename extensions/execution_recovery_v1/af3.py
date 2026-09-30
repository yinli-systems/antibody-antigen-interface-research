#!/usr/bin/env python3
"""Content-bound AF3 recovery with exclusive task/submission ownership.

The legacy scientific plan remains byte-identical. Only staging and runtime
selection change. No training/checkpoint selection, contact or OOD labels here.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def read_json(path):
    def unique(pairs):
        d = {}
        for k, v in pairs:
            if k in d:
                raise ValueError('duplicate JSON key: '+k)
            d[k] = v
        return d
    return json.loads(Path(path).read_text(), object_pairs_hook=unique,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def new_json(path, data):
    path = Path(path)
    encoded = json.dumps(data, indent=2, sort_keys=True, allow_nan=False)+'\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    # Never overwrite an earlier intent, result, or failure.
    with path.open('x') as f:
        f.write(encoded)
        f.flush()
        os.fsync(f.fileno())


def require_digest(path, expected):
    if not re.fullmatch('[0-9a-f]{64}', expected):
        raise ValueError('invalid SHA-256')
    if digest(path) != expected:
        raise ValueError('content changed: '+str(path))


def check_input(task, seeds):
    p = Path(task['input_path'])
    require_digest(p, task['input_sha256'])
    d = read_json(p)
    if not re.fullmatch('[a-z0-9_]+', task['name']):
        raise ValueError('unsafe or non-canonical task name')
    if d['name'] != task['name'] or d.get('dialect') != 'alphafold3' or d.get('version') != 1:
        raise ValueError('AF3 input dialect/name/version drift')
    if d.get('modelSeeds') != seeds or len(set(seeds)) != len(seeds):
        raise ValueError('seed grid drift')
    if len(d['sequences']) != 1 or set(d['sequences'][0]) != {'protein'}:
        raise ValueError('single-chain protein input required')
    protein = d['sequences'][0]['protein']
    if set(protein) != {'id','sequence','unpairedMsa','pairedMsa','templates'}:
        raise ValueError('unexpected protein fields')
    seq = protein['sequence']
    if (protein['id'] != ['A'] or protein['unpairedMsa'] != '' or
        protein['pairedMsa'] != '' or protein['templates'] != []):
        raise ValueError('chain/MSA/template contract changed')
    if seq != task['sequence'] or len(seq) != task['length']:
        raise ValueError('sequence/length mismatch')
    if hashlib.sha256(seq.encode('ascii')).hexdigest() != task['sequence_sha256']:
        raise ValueError('sequence identity mismatch')
    if not re.fullmatch('[ACDEFGHIKLMNPQRSTVWYX]+', seq):
        raise ValueError('invalid amino-acid sequence')
    return d


def check_static(plan):
    if Path(plan['runtime_environment']['AF3_SHARED_SANDBOX_PATH']).exists():
        raise ValueError('unexpected shared sandbox would change runtime selection')
    for path, expected in plan['content_sha256'].items():
        require_digest(path, expected)
    for relative, expected in plan['executor_sha256'].items():
        require_digest(HERE/relative, expected)
    for task in plan['tasks']:
        check_input(task, plan['model_seeds'])
    # Large runtime files are fully hashed at freeze. At every task we check
    # identity/size/mtime. Deep runtime rehash is an explicit final audit.
    for entry in plan['large_runtime_files']:
        st = Path(entry['path']).stat()
        if st.st_size != entry['bytes'] or st.st_mtime_ns != entry['mtime_ns']:
            raise ValueError('runtime metadata changed: '+entry['path'])


def load_plan(path, expected):
    require_digest(path, expected)
    p = read_json(path)
    if p['schema'] != 'af3_execution_recovery_v1' or p['test_access'] or p['contact_label_access']:
        raise ValueError('invalid execution authority')
    check_static(p)
    return p


def runtime_environment(plan):
    env = os.environ.copy()
    env.update(plan['runtime_environment'])
    return env


def task_dir(plan, task_id):
    if type(task_id) is not int or not 0 <= task_id < len(plan['tasks']):
        raise ValueError('unknown task id')
    return Path(plan['run_root'])/('task_%03d' % task_id)


def stage_task(plan, task_id):
    task = plan['tasks'][task_id]
    check_input(task, plan['model_seeds'])
    target = task_dir(plan, task_id)
    target.mkdir(parents=True, exist_ok=False)
    (target/'inputs').mkdir()
    source = Path(task['input_path'])
    staged = target/'inputs'/source.name
    shutil.copyfile(source, staged)  # independent bytes; never a hardlink
    require_digest(staged, task['input_sha256'])
    new_json(target/'START.json', {'state':'STARTED', 'task_id':task_id,
             'at_utc':utc(), 'slurm_job_id':os.environ.get('SLURM_JOB_ID'),
             'scientific_plan_sha256':plan['legacy_plan_sha256'],
             'input_sha256':task['input_sha256']})
    return target


def sample_grid(path, seeds, samples):
    with Path(path).open(newline='') as f:
        reader = csv.DictReader(f)
        if set(reader.fieldnames or []) != {'seed','sample','ranking_score'}:
            raise ValueError('ranking CSV fields differ')
        rows = [(int(r['seed']),int(r['sample']),float(r['ranking_score'])) for r in reader]
    expected = {(s,i) for s in seeds for i in range(samples)}
    actual = [(s,i) for s,i,_ in rows]
    if len(set(actual)) != len(actual) or set(actual) != expected:
        raise ValueError('incomplete/duplicate/foreign seed-sample grid')
    if not all(math.isfinite(r[2]) for r in rows):
        raise ValueError('non-finite ranking score')
    return rows


def protein_backbone(path, sequence):
    # Lazy import: orchestration/test suite does not require a GPU/model stack.
    from Bio.PDB import MMCIFParser
    from Bio.SeqUtils import seq1
    structure = MMCIFParser(QUIET=True).get_structure('prediction', str(path))
    models = list(structure)
    if len(models) != 1 or len(list(models[0])) != 1 or next(iter(models[0])).id != 'A':
        raise ValueError('expected exactly one model and chain')
    residues = list(next(iter(models[0])))
    observed = ''.join(seq1(r.resname, custom_map={'MSE':'M'}) for r in residues)
    if observed != sequence:
        raise ValueError('predicted CIF sequence mismatch')
    atoms = []
    for residue in residues:
        for name in ('N','CA','C','O'):
            if name not in residue:
                raise ValueError('missing backbone atom: '+name)
            atom = residue[name]
            if atom.is_disordered():
                raise ValueError('ambiguous backbone atom')
            coord = list(map(float,atom.coord))
            if not all(math.isfinite(x) for x in coord):
                raise ValueError('non-finite backbone coordinates')
            atoms.append([residue.resname, name, coord])
    return {'residues':len(residues), 'backbone_atoms':len(atoms),
            'backbone_sha256':hashlib.sha256(json.dumps(atoms,separators=(',',':')).encode()).hexdigest()}


def audit_task(plan, task_id):
    task = plan['tasks'][task_id]
    base = task_dir(plan,task_id)/'results'
    result = base/task['name']
    if [x.name for x in base.iterdir() if x.is_dir()] != [task['name']]:
        raise ValueError('unexpected/timestamped result directory')
    ranking = result/'ranking_scores.csv'
    rows = sample_grid(ranking, plan['model_seeds'], plan['samples_per_seed'])
    expected = {'seed-%d_sample-%d' % (s,i) for s,i,_ in rows}
    if {x.name for x in result.glob('seed-*_sample-*') if x.is_dir()} != expected:
        raise ValueError('seed/sample output directories differ')
    records = []
    for s,i,score in rows:
        sample = result/('seed-%d_sample-%d' % (s,i))
        model = sample/'model.cif'
        summary = read_json(sample/'summary_confidences.json')
        if not (sample/'confidences.json').is_file() or (sample/'confidences.json').stat().st_size == 0:
            raise ValueError('missing confidence payload')
        if not math.isclose(float(summary['ranking_score']),round(score,2),rel_tol=0,abs_tol=1e-7):
            raise ValueError('ranking CSV/summary mismatch')
        records.append({'seed':s,'sample':i,'score':score,'model_sha256':digest(model),
                        **protein_backbone(model,task['sequence'])})
    top = result/(task['name']+'_model.cif')
    top_sha = digest(top)
    top_structure = protein_backbone(top,task['sequence'])
    maximum = max(r['score'] for r in records)
    matching = [r for r in records if r['score']==maximum and
                r['backbone_sha256']==top_structure['backbone_sha256']]
    if not matching:
        raise ValueError('top-level model structure is not a maximum-score sample')
    # AF3 serializes the same InferenceResult twice; mmCIF bytes need not be identical.
    # Scientific identity is exact sequence + canonical N/CA/C/O coordinates, while
    # both raw byte hashes remain recorded for provenance.
    chosen = sorted(matching,key=lambda r:(r['seed'],r['sample']))[0]
    actual_input = read_json(result/(task['name']+'_data.json'))
    # Verify AF3 did not silently add MSA/templates; schema normalization allowed.
    protein = actual_input['sequences'][0]['protein']
    if (protein['sequence']!=task['sequence'] or actual_input['modelSeeds']!=plan['model_seeds'] or
        protein.get('templates')!=[] or protein.get('pairedMsa') not in ('',None) or
        protein.get('unpairedMsa') not in ('',None)):
        raise ValueError('AF3 processed inputs changed the scientific input contract')
    manifest = {str(p.relative_to(task_dir(plan,task_id))):digest(p)
                for p in sorted(base.rglob('*')) if p.is_file()}
    return {'state':'COMPLETE', 'task_id':task_id, 'at_utc':utc(),
            'sequence_sha256':task['sequence_sha256'], 'sequence_length':task['length'],
            'sample_count':len(records), 'selected':chosen, 'samples':records,
            'output_sha256':manifest, 'top_model':str(top), 'top_model_sha256':top_sha,
            'top_model_backbone_sha256':top_structure['backbone_sha256'],
            'top_model_byte_identical_to_selected_sample':top_sha==chosen['model_sha256'],
            'model_score_is_binding_evidence':False, 'contact_labels_read':False,
            'test_labels_read':False, 'ood_labels_read':False}


def run_task(plan_path, expected, task_id):
    plan = load_plan(plan_path, expected)
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('GPU prediction requires a scheduler allocation')
    if int(os.environ.get('SLURM_RESTART_COUNT','0')):
        raise RuntimeError('automatic task restart is forbidden; reconcile preserved state')
    target = stage_task(plan, task_id)
    try:
        # Relative paths are mandatory for the existing Singularity wrapper.
        # Load FUSE before the wrapper to avoid an unbounded SIF unpack on each node.
        command = ['bash','-lc','set -e; source /etc/profile.d/modules.sh; '
                   'module load squashfuse/0.6.1; exec bash "$1" inputs results','_',plan['runner']]
        new_json(target/'COMMAND.json', {'argv':command,'cwd':str(target),
                 'environment_overrides':plan['runtime_environment'], 'plan_sha256':expected})
        subprocess.run(command,cwd=target,env=runtime_environment(plan),check=True)
        result = audit_task(plan,task_id)
        check_static(plan)
        result['execution_plan_sha256'] = expected
        result['scientific_plan_sha256'] = plan['legacy_plan_sha256']
        new_json(target/'COMPLETE.json',result)
        print(json.dumps({k:result[k] for k in ('state','task_id','sample_count','sequence_length')},sort_keys=True),flush=True)
    except BaseException as e:
        new_json(target/'FAILURE.json', {'state':'FAILED','at_utc':utc(),
                 'task_id':task_id,'error_type':type(e).__name__,'error':str(e),
                 'execution_plan_sha256':expected,'slurm_job_id':os.environ.get('SLURM_JOB_ID')})
        raise


def freeze(args):
    require_digest(args.legacy_plan,args.legacy_sha256)
    old = read_json(args.legacy_plan)
    if old['task_count']!=40 or old['test_labels_read'] or old['contact_labels_read'] or old['ood_labels_read']:
        raise ValueError('legacy scientific authority differs')
    if [t['task_id'] for t in old['tasks']] != list(range(40)):
        raise ValueError('task IDs are not the exact 40-task grid')
    content = {str(Path(args.legacy_plan).resolve()):args.legacy_sha256}
    for field in ('canonical_manifest','split_manifest','base_structure_manifest','af3_runner'):
        require_digest(old[field],old[field+'_sha256'])
        content[old[field]] = old[field+'_sha256']
    for relative, expected in old['source_sha256'].items():
        path = Path(old['source_program_root'])/relative
        require_digest(path,expected)
        content[str(path)] = expected
    for task in old['tasks']:
        check_input(task,old['model_seeds'])
    resources = []
    for path in (Path(args.sif).resolve(),Path(args.model_dir).resolve()/'af3.bin'):
        st = path.stat()
        resources.append({'path':str(path),'bytes':st.st_size,'mtime_ns':st.st_mtime_ns,'sha256':digest(path)})
    if not Path(args.db_dir).is_dir():
        raise ValueError('missing database directory')
    dest = Path(args.output_root).resolve()
    dest.mkdir(parents=True,exist_ok=False)
    (dest/'logs').mkdir(); (dest/'runs').mkdir(); (dest/'submissions').mkdir()
    files = ('af3.py','PROTOCOL.md','RESEARCH.md')
    plan = {'schema':'af3_execution_recovery_v1','created_utc':utc(),
            'source_commit':args.source_commit,'source_root':str(HERE),
            'legacy_plan':str(Path(args.legacy_plan).resolve()),'legacy_plan_sha256':args.legacy_sha256,
            'content_sha256':content,'executor_sha256':{f:digest(HERE/f) for f in files},
            'large_runtime_files':resources,
            'runtime_environment':{'AF3_USE_NODE_SANDBOX':'0','AF3_SHARED_SANDBOX_PATH':str(dest/'no_shared_sandbox'),
                                   'AF3_SIF_PATH':str(Path(args.sif).resolve()),
                                   'MODEL_DIR':str(Path(args.model_dir).resolve()),'DB_DIR':str(Path(args.db_dir).resolve()),
                                   'PDB_DATABASE_PATH':str(Path(args.db_dir).resolve()/'pdb_2022_09_28_mmcif_files.tar')},
            'runner':old['af3_runner'],'tasks':old['tasks'],'model_seeds':old['model_seeds'],
            'samples_per_seed':5,'run_root':str(dest/'runs'),'test_access':False,'contact_label_access':False,
            'budget':{'tasks':40,'parallel_gpus':4,'gpu_per_task':1,'walltime':'03:00:00',
                      'max_allocated_gpu_hours':120},
            'scientific_change':'Input bytes and 25-sample design unchanged; resource weight bytes pinned. Software equivalence to all historical structures is not established.',
            'runtime_code_version':'3.0.1', 'parameter_resource_directory_label':'3.0.0',
            'runtime_validation':'full SHA-256 at freeze; per-task size/mtime plus source/input hashes; final deep hash audit'}
    new_json(dest/'PLAN.json',plan)
    print(json.dumps({'plan':str(dest/'PLAN.json'),'sha256':digest(dest/'PLAN.json'),'tasks':40},sort_keys=True))



def admit_existing(plan_path, expected, task_id, source_task_dir):
    """Admit successful prediction bytes from a failed audit into a fresh attempt.

    No model inference occurs here. The old directory is never modified; all
    bytes are copied into a new task-owned directory and re-audited under the
    current executor before COMPLETE can exist.
    """
    plan = load_plan(plan_path, expected)
    task = plan['tasks'][task_id]
    source = Path(source_task_dir).resolve()
    if not (source/'FAILURE.json').is_file():
        raise ValueError('source task must preserve the earlier failed audit receipt')
    if not (source/'results'/task['name']).is_dir() or not (source/'inputs').is_dir():
        raise ValueError('source task does not contain completed prediction outputs')
    target = task_dir(plan, task_id)
    target.mkdir(parents=True, exist_ok=False)
    new_json(target/'START.json', {'state':'ADMITTING_EXISTING_PREDICTION',
             'task_id':task_id,'at_utc':utc(),'execution_plan_sha256':expected,
             'source_task_dir':str(source),'source_failure_sha256':digest(source/'FAILURE.json'),
             'model_inference_performed':False})
    try:
        shutil.copytree(source/'inputs',target/'inputs')
        shutil.copytree(source/'results',target/'results')
        # Every copied prediction file must be byte-identical to its source.
        source_files={str(x.relative_to(source/'results')):digest(x)
                      for x in sorted((source/'results').rglob('*')) if x.is_file()}
        copied_files={str(x.relative_to(target/'results')):digest(x)
                      for x in sorted((target/'results').rglob('*')) if x.is_file()}
        if source_files != copied_files:
            raise ValueError('copied prediction tree differs from preserved source bytes')
        if digest(target/'inputs'/Path(task['input_path']).name) != task['input_sha256']:
            raise ValueError('copied input bytes differ')
        result = audit_task(plan,task_id)
        check_static(plan)
        result.update(execution_plan_sha256=expected,
                      scientific_plan_sha256=plan['legacy_plan_sha256'],
                      execution_mode='ADMITTED_EXISTING_SUCCESSFUL_PREDICTION_AFTER_AUDIT_FIX',
                      source_task_dir=str(source),source_prediction_file_count=len(source_files),
                      source_prediction_tree_sha256=hashlib.sha256(json.dumps(source_files,sort_keys=True).encode()).hexdigest(),
                      model_inference_performed=False)
        new_json(target/'COMPLETE.json',result)
        print(json.dumps({'state':result['state'],'task_id':task_id,
                          'sample_count':result['sample_count'],
                          'model_inference_performed':False,
                          'top_bytes_identical':result['top_model_byte_identical_to_selected_sample']},sort_keys=True))
    except BaseException as e:
        new_json(target/'FAILURE.json',{'state':'FAILED_ADMISSION','at_utc':utc(),
                 'task_id':task_id,'error_type':type(e).__name__,'error':str(e),
                 'model_inference_performed':False,'source_task_dir':str(source)})
        raise

def validate_task_set(task_ids, count):
    if not task_ids or len(set(task_ids))!=len(task_ids):
        raise ValueError('tasks must be nonempty and unique')
    if any(type(x) is not int or not 0<=x<count for x in task_ids):
        raise ValueError('unknown task IDs')
    return sorted(task_ids)


def submit(plan_path, expected, ids):
    plan = load_plan(plan_path,expected)
    ids = validate_task_set(ids,len(plan['tasks']))
    root = Path(plan_path).resolve().parent
    # One submission ledger per task: two controllers cannot submit the same ID.
    # An uncertain sbatch acknowledgement remains an intent requiring manual reconciliation.
    if ids != [0]:
        canary = read_json(task_dir(plan,0)/'COMPLETE.json')
        if canary['state']!='COMPLETE' or canary['execution_plan_sha256']!=expected:
            raise ValueError('task 0 output audit must pass first')
    for i in ids:
        if task_dir(plan,i).exists() or (root/'submissions'/('task_%03d.json'%i)).exists():
            raise FileExistsError('task already staged/submitted; inspect, do not repeat')
    name = 'ab-af3r-'+expected[:8]
    # Read actual scheduler before changing state. Tool failure is not an empty queue.
    query = subprocess.run(['squeue','--me','--array','--noheader','--format=%i|%j|%T'],
                           check=True,capture_output=True,text=True,timeout=30)
    for row in query.stdout.splitlines():
        fields = row.split('|')
        if len(fields)==3 and fields[1]==name:
            # Serial submit after full canary; no need to overlap with earlier allocation.
            raise RuntimeError('this recovery already has an active allocation; reconcile it first')
    for i in ids:
        new_json(root/'submissions'/('task_%03d.json'%i), {'state':'SUBMISSION_INTENT',
                 'task_id':i,'at_utc':utc(),'plan_sha256':expected})
    group = 'task0' if ids==[0] else 'remaining'
    script = root/(group+'.sbatch')
    command = [sys.executable,str(HERE/'af3.py'),'run','--plan',str(Path(plan_path).resolve()),
               '--plan-sha256',expected,'--task-id']
    with script.open('x') as f:
        f.write('#!/bin/bash\nset -euo pipefail\nexec '+shlex.join(command)+' "${SLURM_ARRAY_TASK_ID:?}"\n')
    cmd = ['sbatch','--parsable','--no-requeue','--array='+','.join(map(str,ids))+'%4',
           '--job-name='+name,'--partition=gpu_4090','--nodes=1','--ntasks=1',
           '--gres=gpu:1','--cpus-per-task=6','--time=03:00:00',
           '--chdir='+str(root),'--output='+str(root/'logs/%A_%a.out'),
           '--error='+str(root/'logs/%A_%a.err'),str(script)]
    record = {'argv':cmd,'script_sha256':digest(script),'tasks':ids,'plan_sha256':expected,
              'queue_before':query.stdout,'at_utc':utc()}
    new_json(root/(group+'_SUBMIT_COMMAND.json'),record)
    result = subprocess.run(cmd,capture_output=True,text=True,timeout=60)
    record.update(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
    if result.returncode or not re.fullmatch(r'\d+(;[^\s;]+)?\s*',result.stdout):
        new_json(root/(group+'_SUBMIT_FAILURE.json'),record)
        raise RuntimeError('submission failed or ambiguous; preserve intent, no automatic retry')
    record['job_id'] = result.stdout.strip().split(';')[0]
    record['state'] = 'SUBMITTED_NOT_COMPLETED'
    new_json(root/(group+'_SUBMITTED.json'),record)
    print(json.dumps({'job_id':record['job_id'],'tasks':ids,'state':record['state']}),flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action',required=True)
    f = sub.add_parser('freeze')
    for flag in ('legacy-plan','legacy-sha256','sif','model-dir','db-dir','source-commit','output-root'):
        f.add_argument('--'+flag,required=True)
    for action in ('run','submit','verify','admit-existing'):
        s = sub.add_parser(action); s.add_argument('--plan',required=True); s.add_argument('--plan-sha256',required=True)
        if action=='run': s.add_argument('--task-id',required=True,type=int)
        if action=='submit': s.add_argument('--tasks',required=True)
        if action=='admit-existing':
            s.add_argument('--task-id',required=True,type=int); s.add_argument('--source-task-dir',required=True)
        if action=='verify': s.add_argument('--deep-runtime',action='store_true')
    a = p.parse_args()
    if a.action=='freeze': freeze(a)
    elif a.action=='run': run_task(a.plan,a.plan_sha256,a.task_id)
    elif a.action=='submit': submit(a.plan,a.plan_sha256,[int(x) for x in a.tasks.split(',')])
    elif a.action=='admit-existing': admit_existing(a.plan,a.plan_sha256,a.task_id,a.source_task_dir)
    else:
        plan = load_plan(a.plan,a.plan_sha256)
        if a.deep_runtime:
            for x in plan['large_runtime_files']: require_digest(x['path'],x['sha256'])
        print('PASS_EXECUTION_PLAN_INTEGRITY; not a prediction/quality result')

if __name__=='__main__': main()
