#!/usr/bin/env python3
"""Strictly re-audit an existing AF3 attempt and assemble the 217-case authority.

This program never runs AF3 inference. It is intended for a completed attempt
whose original post-inference auditor was weaker than the current strict
semantic auditor. All 40 source task directories and terminal receipts are
checked before any output directory is created.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import af3
from assemble import validation_coverage, write_backbone


def source_execution_plan(path: Path, expected_sha256: str, *, deep_runtime: bool) -> dict[str, Any]:
    """Verify the original execution authority without substituting current code."""
    af3.require_digest(path, expected_sha256)
    plan = af3.read_json(path)
    if (plan.get('schema') != 'af3_execution_recovery_v1' or plan.get('test_access') or
        plan.get('contact_label_access')):
        raise ValueError('invalid source execution authority')
    source_root = Path(plan['source_root'])
    for raw_path, expected in plan['content_sha256'].items():
        af3.require_digest(raw_path, expected)
    for relative, expected in plan['executor_sha256'].items():
        af3.require_digest(source_root/relative, expected)
    for task in plan['tasks']:
        af3.check_input(task, plan['model_seeds'])
    for entry in plan['large_runtime_files']:
        path_obj = Path(entry['path'])
        stat = path_obj.stat()
        if stat.st_size != entry['bytes'] or stat.st_mtime_ns != entry['mtime_ns']:
            raise ValueError('source runtime metadata changed: '+entry['path'])
        if deep_runtime:
            af3.require_digest(path_obj, entry['sha256'])
    if len(plan['tasks']) != 40 or [x['task_id'] for x in plan['tasks']] != list(range(40)):
        raise ValueError('source task grid differs')
    return plan


def _selected_core(record: dict[str, Any]) -> dict[str, Any]:
    # `backbone_sha256` is intentionally excluded: source 79f6 and strict
    # auditor 93aa hash different canonical tuples for the same coordinates.
    # Selected model bytes and the full output-tree digest are exact, while the
    # strict auditor independently recomputes all-atom/backbone fingerprints.
    fields = ('seed','sample','score','model_sha256','residues','backbone_atoms')
    return {key:record.get(key) for key in fields}


def strict_task_receipt(plan: dict[str, Any], plan_sha256: str, task_id: int,
                        *, auditor_commit: str, auditor_sha256: str) -> dict[str, Any]:
    """Re-audit one completed source task without modifying its bytes."""
    task = plan['tasks'][task_id]
    root = Path(plan['run_root'])/('task_%03d' % task_id)
    if (root/'FAILURE.json').exists():
        raise ValueError('source task has a preserved failure: '+str(root))
    source_complete = root/'COMPLETE.json'
    if not source_complete.is_file():
        raise RuntimeError('source task is incomplete: '+str(root))
    old = af3.read_json(source_complete)
    expected = {
        'state':'COMPLETE', 'task_id':task_id,
        'execution_plan_sha256':plan_sha256,
        'scientific_plan_sha256':plan['legacy_plan_sha256'],
        'sequence_sha256':task['sequence_sha256'],
        'sequence_length':task['length'], 'sample_count':25,
        'contact_labels_read':False, 'test_labels_read':False,
        'ood_labels_read':False,
    }
    for key, value in expected.items():
        if old.get(key) != value:
            raise ValueError('source receipt disagrees on %s for task %d' % (key,task_id))
    actual = af3.audit_task(plan,task_id)
    if actual['output_sha256'] != old.get('output_sha256'):
        raise ValueError('source prediction bytes changed after completion')
    if _selected_core(actual['selected']) != _selected_core(old.get('selected',{})):
        raise ValueError('strict selected sample disagrees with source receipt')
    return {
        'state':'STRICT_AUDIT_COMPLETE', 'task_id':task_id,
        'source_execution_plan_sha256':plan_sha256,
        'scientific_plan_sha256':plan['legacy_plan_sha256'],
        'source_complete_path':str(source_complete),
        'source_complete_sha256':af3.digest(source_complete),
        'strict_auditor_commit':auditor_commit,
        'strict_af3_sha256':auditor_sha256,
        'source_selected':old['selected'],
        'digest_schema_transition':{
            'legacy_backbone_sha256':old['selected'].get('backbone_sha256'),
            'strict_backbone_sha256':actual['selected'].get('backbone_sha256'),
            'strict_all_atom_sha256':actual['selected'].get('all_atom_sha256'),
            'selection_identity_fields':list(_selected_core(actual['selected'])),
            'selection_identity_equal':True,
        },
        'strict_actual':actual,
        'model_inference_performed':False,
        'contact_labels_read':False, 'test_labels_read':False,
        'ood_labels_read':False,
    }


def preflight(plan: dict[str, Any], plan_sha256: str, *, auditor_commit: str,
              auditor_sha256: str) -> list[dict[str, Any]]:
    receipts = [strict_task_receipt(
        plan,plan_sha256,task['task_id'],auditor_commit=auditor_commit,
        auditor_sha256=auditor_sha256) for task in plan['tasks']]
    if len(receipts) != 40:
        raise ValueError('exactly 40 strict task audits required')
    return receipts


def finalize(source_plan_path: Path, source_plan_sha256: str, output: Path,
             *, auditor_commit: str, deep_runtime: bool) -> None:
    plan = source_execution_plan(source_plan_path,source_plan_sha256,
                                 deep_runtime=deep_runtime)
    auditor_sha256 = af3.digest(af3.__file__)
    receipts = preflight(plan,source_plan_sha256,
                         auditor_commit=auditor_commit,
                         auditor_sha256=auditor_sha256)
    legacy = af3.read_json(plan['legacy_plan'])
    base_path = Path(legacy['base_structure_manifest'])
    base = af3.read_json(base_path)
    af3.require_digest(base_path,legacy['base_structure_manifest_sha256'])
    records = copy.deepcopy(base['records'])
    keys = set()
    for record in records:
        key = record['sequence_sha256']
        if key in keys:
            raise ValueError('duplicate base sequence')
        keys.add(key)
        if hashlib.sha256(record['sequence'].encode('ascii')).hexdigest() != key:
            raise ValueError('base sequence digest mismatch')
        pdb = Path(record['pdb_path'])
        if not pdb.is_absolute():
            pdb = base_path.parent/pdb
        af3.require_digest(pdb,record['pdb_sha256'])
        record['pdb_path'] = str(pdb.resolve())
    before, missing = validation_coverage(legacy,keys)
    proposed = {task['sequence_sha256'] for task in plan['tasks']}
    if proposed != missing or len(proposed) != 40 or proposed & keys:
        raise ValueError('completion set is not exactly the frozen missing chains')

    output = output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    af3.new_json(output/'START.json',{
        'state':'STRICT_AUDIT_AND_ASSEMBLY_STARTED',
        'source_execution_plan_sha256':source_plan_sha256,
        'strict_auditor_commit':auditor_commit,
        'strict_af3_sha256':auditor_sha256,
        'model_inference_performed':False,
    })
    try:
        audit_root = output/'strict_task_audits'
        audit_root.mkdir()
        for receipt in receipts:
            af3.new_json(audit_root/('task_%03d.json'%receipt['task_id']),receipt)
        added = []
        for task, receipt in zip(plan['tasks'],receipts):
            actual = receipt['strict_actual']
            key = task['sequence_sha256']
            destination = output/'units'/key[:2]/(key+'.pdb')
            conversion = write_backbone(Path(actual['top_model']),task['sequence'],destination)
            added.append({
                'sequence_sha256':key, 'sequence':task['sequence'],
                'side':'antigen' if task['roles']==['antigen'] else 'antibody',
                'roles':task['roles'], 'instances':task['instances'],
                'source':'af3_strict_semantic_reaudit_v1',
                'pdb_path':str(destination), 'pdb_sha256':conversion['pdb_sha256'],
                'coordinate_rounding':conversion,
                'strict_task_audit':str(audit_root/('task_%03d.json'%task['task_id'])),
                'strict_task_audit_sha256':af3.digest(audit_root/('task_%03d.json'%task['task_id'])),
                'source_complete_path':receipt['source_complete_path'],
                'source_complete_sha256':receipt['source_complete_sha256'],
                'chosen':{
                    'model_path':actual['top_model'],
                    'top_model_sha256':actual['top_model_sha256'],
                    'selected_sample_model_sha256':actual['selected']['model_sha256'],
                    'score':actual['selected']['score'],
                    'seed':actual['selected']['seed'],
                    'sample':actual['selected']['sample'],
                    'semantic_comparison':actual['top_model_comparison'],
                },
                'runtime_code_version':plan['runtime_code_version'],
                'runtime_resources':plan['large_runtime_files'],
            })
        after, left = validation_coverage(legacy,keys|proposed)
        if left or not all(x['complete'] for x in after):
            raise ValueError('217-case coverage did not close')
        combined = copy.deepcopy(base)
        combined.update(
            records=sorted(records+added,key=lambda row:row['sequence_sha256']),
            authority_generation='base_v2_plus_strict_semantic_af3_reaudit',
            total_units=len(records)+40, base_unit_count=len(records),
            extension_unit_count=40, contact_labels_loaded=False,
            test_loaded=False, extension_source_execution_plan_sha256=source_plan_sha256,
            strict_auditor_commit=auditor_commit,
            strict_af3_sha256=auditor_sha256,
            base_manifest_sha256=af3.digest(base_path),
        )
        combined['counts_by_side'] = {
            side:sum(row['side']==side for row in combined['records'])
            for side in ('antibody','antigen')}
        af3.new_json(output/'EXTENSION_MANIFEST.json',{
            'format_version':'strict_af3_completion_extension_v1',
            'records':added, 'record_count':40,
            'source_execution_plan_sha256':source_plan_sha256,
            'strict_auditor_commit':auditor_commit,
            'strict_af3_sha256':auditor_sha256,
            'model_inference_performed':False,
            'contact_labels_read':False, 'test_labels_read':False,
            'ood_labels_read':False,
        })
        af3.new_json(output/'STRUCTURE_UNITS_MANIFEST.json',combined)
        # Recheck mutable scientific/runtime authority at closure.
        source_execution_plan(source_plan_path,source_plan_sha256,
                              deep_runtime=deep_runtime)
        af3.new_json(output/'COMPLETE.json',{
            'state':'COMPLETE_217_CASE_STRICT_STRUCTURE_AUTHORITY',
            'complete_cases_before':sum(x['complete'] for x in before),
            'complete_cases_after':217, 'new_chains':40,
            'total_units':len(records)+40,
            'strict_task_audit_count':40,
            'combined_manifest_sha256':af3.digest(output/'STRUCTURE_UNITS_MANIFEST.json'),
            'extension_manifest_sha256':af3.digest(output/'EXTENSION_MANIFEST.json'),
            'source_execution_plan_sha256':source_plan_sha256,
            'strict_auditor_commit':auditor_commit,
            'strict_af3_sha256':auditor_sha256,
            'model_inference_performed':False,
            'training_feature_banks_modified':False,
            'contact_labels_read':False, 'test_labels_read':False,
            'ood_labels_read':False, 'structure_accuracy_measured':False,
        })
    except BaseException as error:
        af3.new_json(output/'FAILURE.json',{
            'state':'FAILED_STRICT_AUDIT_OR_ASSEMBLY',
            'error_type':type(error).__name__, 'error':str(error),
            'model_inference_performed':False})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-plan',type=Path,required=True)
    parser.add_argument('--source-plan-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--auditor-commit',required=True)
    parser.add_argument('--deep-runtime',action='store_true')
    args = parser.parse_args()
    finalize(args.source_plan.resolve(),args.source_plan_sha256,args.output,
             auditor_commit=args.auditor_commit,deep_runtime=args.deep_runtime)


if __name__=='__main__':
    main()
