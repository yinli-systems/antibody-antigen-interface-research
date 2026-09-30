#!/usr/bin/env python3
"""Assemble a new RePaRank structure authority only after all 40 AF3 audits.

Never changes the base manifest or live SaProt/ProSST feature-bank authority.
Only frozen validation sequence metadata is read, not contact/test/OOD labels.
"""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
from pathlib import Path
import sys
from af3 import load_plan, read_json, digest, require_digest, new_json, task_dir, audit_task


def complete_receipts(plan, plan_sha):
    receipts=[]
    for task in plan['tasks']:
        root=task_dir(plan,task['task_id'])
        if (root/'FAILURE.json').exists():
            raise ValueError('preserved task failure: '+str(root))
        path=root/'COMPLETE.json'
        if not path.is_file():
            raise RuntimeError('incomplete structure campaign; missing task '+str(task['task_id']))
        receipt=read_json(path)
        if (receipt['state']!='COMPLETE' or receipt['task_id']!=task['task_id'] or
            receipt['execution_plan_sha256']!=plan_sha or
            receipt['scientific_plan_sha256']!=plan['legacy_plan_sha256'] or
            receipt['sequence_sha256']!=task['sequence_sha256'] or
            receipt['sample_count']!=25 or receipt['test_labels_read'] or
            receipt['contact_labels_read'] or receipt['ood_labels_read']):
            raise ValueError('task receipt disagrees with frozen authority')
        actual=audit_task(plan,task['task_id'])
        if actual['output_sha256']!=receipt['output_sha256'] or actual['selected']!=receipt['selected']:
            raise ValueError('prediction bytes or selected model changed after task completion')
        receipts.append((task,receipt,str(path),digest(path)))
    if len(receipts)!=40:
        raise ValueError('exactly 40 completed tasks required')
    return receipts


def validation_coverage(old, keys):
    split=read_json(old['split_manifest'])
    ids=split['splits']['validation']
    if len(ids)!=217 or len(set(ids))!=217:
        raise ValueError('validation identity count differs')
    if set(ids)&(set(split['splits']['train'])|set(split['splits']['test'])):
        raise ValueError('split overlap')
    with Path(old['canonical_manifest']).open(newline='') as f:
        rows={}
        for row in csv.DictReader(f):
            if row['INSTANCE'] in rows:raise ValueError('duplicate canonical ID')
            rows[row['INSTANCE']]=row
    result=[]; missing=set()
    for iid in ids:
        r=rows[iid]; unseen=[]
        for field in ('vh_seq','vl_seq','ag_seq'):
            sequence=(r[field] or '').strip()
            if not sequence:
                if field!='vl_seq':raise ValueError('missing required chain')
                continue
            key=hashlib.sha256(sequence.encode('ascii')).hexdigest()
            if key not in keys: unseen.append(key);missing.add(key)
        result.append({'instance_id':iid,'complete':not unseen,'missing_keys':unseen})
    return result,missing


def write_backbone(source,sequence,destination):
    from Bio.PDB import MMCIFParser,PDBParser,PDBIO,Select
    from Bio.SeqUtils import seq1
    from af3 import protein_backbone
    protein_backbone(source,sequence)
    structure=MMCIFParser(QUIET=True).get_structure('selected',str(source))
    class BackboneOnly(Select):
        def accept_atom(self,atom):return atom.name in ('N','CA','C','O')
    writer=PDBIO();writer.set_structure(structure)
    destination.parent.mkdir(parents=True,exist_ok=True)
    # Exclusive open avoids changing any earlier assembly.
    with destination.open('x') as handle:writer.save(handle,select=BackboneOnly())
    restored=PDBParser(QUIET=True).get_structure('roundtrip',str(destination))
    chains=list(next(iter(restored)))
    if len(chains)!=1 or chains[0].id!='A':raise ValueError('PDB roundtrip chain mismatch')
    residues=list(chains[0]); observed=''.join(seq1(x.resname,custom_map={'MSE':'M'}) for x in residues)
    if observed!=sequence or any({a.name for a in r}!={'N','CA','C','O'} for r in residues):
        raise ValueError('PDB roundtrip sequence or backbone mismatch')
    original=list(next(iter(next(iter(structure)))))
    maximum=max(abs(float(a)-float(b)) for r,s in zip(original,residues)
                for name in ('N','CA','C','O') for a,b in zip(r[name].coord,s[name].coord))
    if maximum>0.0006:raise ValueError('PDB precision exceeded the declared 0.0006-Angstrom rounding bound')
    return {'pdb_sha256':digest(destination),'max_coordinate_rounding_A':maximum}


def assemble(plan_path,sha,output, *, plan_loader=None, receipt_loader=None):
    loader=plan_loader or load_plan
    reader=receipt_loader or complete_receipts
    plan=loader(plan_path,sha)
    receipts=reader(plan,sha)  # no output until full campaign admitted
    old=read_json(plan['legacy_plan'])
    basepath=Path(old['base_structure_manifest']);base=read_json(basepath)
    require_digest(basepath,old['base_structure_manifest_sha256'])
    records=copy.deepcopy(base['records'])
    keys=set()
    for record in records:
        key=record['sequence_sha256']
        if key in keys:raise ValueError('duplicate base sequence')
        keys.add(key)
        if hashlib.sha256(record['sequence'].encode('ascii')).hexdigest()!=key:
            raise ValueError('base sequence digest mismatch')
        path=Path(record['pdb_path'])
        if not path.is_absolute():path=basepath.parent/path
        require_digest(path,record['pdb_sha256'])
        record['pdb_path']=str(path.resolve())
    before,missing=validation_coverage(old,keys)
    proposed={task['sequence_sha256'] for task,_,_,_ in receipts}
    if proposed!=missing or len(proposed)!=40 or proposed&keys:
        raise ValueError('completion set is not exactly the frozen missing chains')
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    new_json(output/'START.json',{'state':'ASSEMBLING','plan_sha256':sha,
             'assembler_sha256':digest(__file__),'base_manifest_sha256':digest(basepath)})
    try:
        added=[]
        for task,receipt,rpath,rsha in receipts:
            key=task['sequence_sha256'];pdb=output/'units'/key[:2]/(key+'.pdb')
            audit=write_backbone(Path(receipt['top_model']),task['sequence'],pdb)
            added.append({'sequence_sha256':key,'sequence':task['sequence'],
                'side':'antigen' if task['roles']==['antigen'] else 'antibody',
                'roles':task['roles'],'instances':task['instances'],'source':'af3_execution_recovery_v1',
                'pdb_path':str(pdb),'pdb_sha256':audit['pdb_sha256'],
                'coordinate_rounding':audit,'task_receipt':rpath,'task_receipt_sha256':rsha,
                'chosen':{'model_path':receipt['top_model'],
                          'model_sha256':digest(Path(receipt['top_model'])),
                          'selected_sample_model_sha256':receipt['selected']['model_sha256'],
                          'score':receipt['selected']['score'],'seed':receipt['selected']['seed'],
                          'sample':receipt['selected']['sample']},
                'runtime_code_version':plan['runtime_code_version'],
                'runtime_resources':plan['large_runtime_files']})
        after,left=validation_coverage(old,keys|proposed)
        if left or not all(x['complete'] for x in after):raise ValueError('217-case coverage did not close')
        combined=copy.deepcopy(base)
        combined.update(records=sorted(records+added,key=lambda r:r['sequence_sha256']),
                        authority_generation='base_v2_plus_independently_audited_completion',
                        total_units=len(records)+40,base_unit_count=len(records),extension_unit_count=40,
                        contact_labels_loaded=False,test_loaded=False,
                        extension_plan_sha256=sha,base_manifest_sha256=digest(basepath))
        combined['counts_by_side']={side:sum(r['side']==side for r in combined['records'])
                                    for side in ('antibody','antigen')}
        new_json(output/'EXTENSION_MANIFEST.json',{'records':added,'record_count':40,'plan_sha256':sha})
        new_json(output/'STRUCTURE_UNITS_MANIFEST.json',combined)
        # Frozen input/source and large runtime bytes verified again before closure.
        loader(plan_path,sha)
        for resource in plan['large_runtime_files']:require_digest(resource['path'],resource['sha256'])
        new_json(output/'COMPLETE.json',{'state':'COMPLETE_217_CASE_STRUCTURE_AUTHORITY',
                 'complete_cases_before':sum(x['complete'] for x in before),
                 'complete_cases_after':217,'new_chains':40,'total_units':len(records)+40,
                 'combined_manifest_sha256':digest(output/'STRUCTURE_UNITS_MANIFEST.json'),
                 'plan_sha256':sha,'assembler_sha256':digest(__file__),
                 'training_feature_banks_modified':False,'contact_labels_read':False,
                 'test_labels_read':False,'ood_labels_read':False,
                 'structure_accuracy_measured':False})
    except BaseException as error:
        new_json(output/'FAILURE.json',{'state':'FAILED_ASSEMBLY','error':repr(error)})
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',required=True);p.add_argument('--plan-sha256',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args();assemble(a.plan,a.plan_sha256,a.output)

if __name__=='__main__':main()
