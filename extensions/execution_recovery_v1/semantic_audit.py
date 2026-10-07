#!/usr/bin/env python3
"""Read-only strict post-audit of predictions from a pinned execution snapshot.

Does not submit jobs, call a model, rewrite receipts or alter inference outputs.
It intentionally separates the old execution-plan SHA from this auditor's SHA.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import af3


def load_execution(plan_path: Path, expected: str):
    af3.require_digest(plan_path, expected)
    plan = af3.read_json(plan_path)
    path = Path(plan['source_root'])/'af3.py'
    af3.require_digest(path, plan['executor_sha256']['af3.py'])
    spec = importlib.util.spec_from_file_location('_pinned_execution_af3', path)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    original.load_plan(str(plan_path), expected)
    return original, plan


def file_manifest(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): af3.digest(p)
            for p in sorted(root.rglob('*')) if p.is_file()}


def audit_existing(plan_path: Path, expected: str, task_id: int,
                   allow_preserved_failure: bool = False, deep_runtime: bool = False):
    original, plan = load_execution(plan_path, expected)
    root = original.task_dir(plan, task_id)
    task = plan['tasks'][task_id]
    before = file_manifest(root)
    if not before:
        raise RuntimeError('no existing task outputs')
    failure = root/'FAILURE.json'
    complete = root/'COMPLETE.json'
    if failure.exists():
        record = af3.read_json(failure)
        if (not allow_preserved_failure or complete.exists() or
            record.get('task_id') != task_id or
            record.get('execution_plan_sha256') != expected or
            record.get('error') != 'top-level model is not a maximum-score sample byte for byte'):
            raise ValueError('only the explicit preserved timestamp audit failure may be re-audited')
        original_state = 'FAILED_BYTE_EQUALITY_AUDIT'
    else:
        record = af3.read_json(complete)
        if (record.get('state') != 'COMPLETE' or
            record.get('task_id') != task_id or
            record.get('execution_plan_sha256') != expected or
            record.get('scientific_plan_sha256') != plan['legacy_plan_sha256'] or
            record.get('sequence_sha256') != task['sequence_sha256'] or
            record.get('sample_count') != 25):
            raise ValueError('terminal execution receipt does not match authority')
        original_state = 'COMPLETE_UNDER_PREVIOUS_AUDITOR'
    staged = root/'inputs'/Path(task['input_path']).name
    af3.require_digest(staged, task['input_sha256'])
    # Strict CIF parsing, including duplicate-tag rejection, for every sample.
    result = root/'results'/task['name']
    for path in sorted(result.glob('seed-*_sample-*/model.cif')):
        af3._mmcif_dict(path, False)
    af3._mmcif_dict(result/(task['name']+'_model.cif'), False)
    report = af3.audit_task(plan, task_id)
    if original_state == 'COMPLETE_UNDER_PREVIOUS_AUDITOR':
        if record['output_sha256'] != report['output_sha256']:
            raise ValueError('original output manifest mismatch')
        for field in ('seed', 'sample', 'score', 'model_sha256'):
            if record['selected'][field] != report['selected'][field]:
                raise ValueError('selected model changed: '+field)
    origin = record.get('source_task_dir')
    origin_evidence = None
    if origin:
        source = Path(origin)
        source_before = file_manifest(source)
        if not (source/'FAILURE.json').is_file():
            raise ValueError('recovered source must preserve old failure receipt')
        original_prediction = {k: v for k,v in source_before.items() if k.startswith('results/')}
        copied_prediction = {k: v for k,v in before.items() if k.startswith('results/')}
        if original_prediction != copied_prediction:
            raise ValueError('recovered output tree differs from failed-audit prediction tree')
        origin_evidence = {'source_task_dir': str(source),
                           'source_failure_sha256': af3.digest(source/'FAILURE.json'),
                           'identical_prediction_files': len(original_prediction)}
    original.load_plan(str(plan_path), expected)
    if deep_runtime:
        for resource in plan['large_runtime_files']:
            af3.require_digest(resource['path'], resource['sha256'])
    if file_manifest(root) != before:
        raise ValueError('task input/output bytes changed during read-only audit')
    if origin and file_manifest(Path(origin)) != source_before:
        raise ValueError('original failed-audit source changed during read-only audit')
    import Bio
    report.update(state='PASS_STRICT_SEMANTIC_POST_AUDIT',
                  audit_schema='af3_semantic_postaudit_v1',
                  previous_state=original_state,
                  execution_plan_sha256=expected,
                  scientific_plan_sha256=plan['legacy_plan_sha256'],
                  execution_source_commit=plan['source_commit'],
                  original_execution_receipt_sha256=af3.digest(failure if failure.exists() else complete),
                  auditor_sha256={n: af3.digest(Path(__file__).parent/n)
                                  for n in ('af3.py','semantic_audit.py')},
                  checked_task_tree_sha256=before,
                  task_tree_unchanged=True,
                  recovered_source=origin_evidence,
                  all_atom_and_cif_semantics_checked=True,
                  allowed_difference='_ma_model_list.model_group_name: valid serialization timestamp only',
                  coordinate_tolerance=0,
                  full_confidence_and_summary_match_selected=True,
                  runtime_deep_hashes_verified=deep_runtime,
                  biopython_version=Bio.__version__,
                  inference_performed=False,
                  experimental_contact_labels_read=False,
                  predicted_confidence_arrays_read=True,
                  previous_receipts_modified=False,
                  job_submission_performed=False)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--task-id', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-preserved-audit-failure', action='store_true')
    parser.add_argument('--deep-runtime', action='store_true')
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    p=af3.read_json(args.plan)
    output=args.output.resolve()
    if output.is_relative_to(Path(p['run_root']).resolve()):
        raise ValueError('post-audit receipt must be outside immutable task outputs')
    report=audit_existing(args.plan, args.plan_sha256, args.task_id,
                          args.allow_preserved_audit_failure, args.deep_runtime)
    af3.new_json(output, report)
    print(json.dumps({k: report[k] for k in (
        'state','task_id','sample_count','sequence_length','selected',
        'top_model_comparison','recovered_source','task_tree_unchanged',
        'runtime_deep_hashes_verified','inference_performed')}, indent=2))


if __name__=='__main__':
    main()
