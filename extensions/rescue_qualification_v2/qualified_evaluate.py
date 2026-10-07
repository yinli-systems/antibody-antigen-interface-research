#!/usr/bin/env python3
"""Supplement frozen rescue training without mutating or resubmitting it.

prepare: require all three terminal training histories; freeze selected models
and ALL validation feature/contact bytes. run: one seed pair under a scheduler
allocation. No entry point submits jobs or selects a checkpoint using contact AP.
"""
from __future__ import annotations
import argparse, importlib.util, inspect as signatures, json, os
from pathlib import Path
import numpy as np
from qualification_core import (digest,new_json,validate_rescue_checkpoint,
                                observed_transforms,canonical_assignment,summarize_records)
from preflight import inspect as input_preflight


def import_original(path):
    spec=importlib.util.spec_from_file_location('frozen_rescue_original',path)
    original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
    return original


def prepare(args):
    import torch
    metadata=input_preflight(args.training_plan,args.training_plan_sha256)
    old=json.loads(args.training_plan.read_text())
    source=next(Path(x) for x in old['input_sha256'] if Path(x).name=='learned_rescue.py')
    o=import_original(source)
    checkpoint_rows=[]
    for seed in old['rescue_seeds']:
        ck=args.training_root/f'seed_{seed}/best.pt';history=args.training_root/f'seed_{seed}/history.json'
        if not ck.is_file() or not history.is_file():
            raise RuntimeError('WAIT_FOR_EXISTING_TRAINING: incomplete seed '+str(seed))
        h=json.loads(history.read_text())
        if len(h['history'])!=old['epochs']:raise ValueError('terminal training history is incomplete')
        checkpoint=torch.load(ck,map_location='cpu',weights_only=False)
        validate_rescue_checkpoint(checkpoint,h,args.training_plan_sha256,seed)
        checkpoint_rows.append({'seed':seed,'path':str(ck.resolve()),'sha256':digest(ck),
                                'history':str(history.resolve()),'history_sha256':digest(history)})
    seqs,instances=o.sequence_metadata();split=json.loads(o.SPLIT.read_text())
    ids=sorted(map(str,split['splits']['validation']))
    keys={key for iid in ids for key in instances[iid]}
    pins={str(args.training_plan.resolve()):args.training_plan_sha256}
    for iid in ids:
        p=o.CONTACTS/f'{iid}.npz';pins[str(p)]=digest(p)
    absent=[]
    for key in sorted(keys):
        base=o.BASE/key[:2]/(key+'.pt');pins[str(base)]=digest(base)
        extra=o.ESMIF1/key[:2]/(key+'.pt')
        if extra.is_file():pins[str(extra)]=digest(extra)
        else:absent.append(str(extra))
    refs=[]
    for down in old['downstream_checkpoints']:
        ref=Path(down['path']).parent.parent/'validation/evaluation.json'
        pins[str(ref)]=digest(ref);refs.append({'seed':down['seed'],'path':str(ref)})
    here=Path(__file__).parent.resolve()
    source_hashes={str(here/f):digest(here/f) for f in ('qualified_evaluate.py','qualification_core.py','preflight.py','PROTOCOL.md')}
    plan={'schema':'supplemental_rescue_qualification_v2','training_plan':str(args.training_plan.resolve()),
          'training_plan_sha256':args.training_plan_sha256,'original_source':str(source),
          'input_sha256':pins,'source_sha256':source_hashes,'absent_features':absent,
          'rescue_checkpoints':checkpoint_rows,'historical_identity_receipts':refs,'ids':ids,
          'conditions':[f'{m}:{k}' for m,k in old['conditions']],
          'seed_pairs':list(map(list,zip(old['rescue_seeds'],old['downstream_seeds']))),
          'primary_solver':'original v1 assignment','secondary_solver':'content-canonical assignment',
          'test_access':False,'ood_access':False,'contact_selection':False,'metadata_preflight':metadata}
    new_json(args.plan,plan);print('QUALIFICATION_PLAN',digest(args.plan))


def run(args):
    import torch,pandas as pd
    from sklearn.metrics import average_precision_score
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('contact evaluation requires a scheduler allocation')
    if digest(args.plan)!=args.plan_sha256:raise ValueError('evaluation plan hash mismatch')
    p=json.loads(args.plan.read_text())
    def verify():
        for path,h in {**p['source_sha256'],**p['input_sha256']}.items():
            if digest(path)!=h:raise ValueError('frozen bytes changed: '+path)
        if any(Path(x).exists() for x in p['absent_features']):raise ValueError('missing feature appeared after freeze')
        for c in p['rescue_checkpoints']:
            if digest(c['path'])!=c['sha256'] or digest(c['history'])!=c['history_sha256']:raise ValueError('rescue model/history drift')
    verify()
    if p['test_access'] or p['ood_access'] or p['contact_selection']:raise ValueError('invalid access authority')
    if args.index not in range(len(p['seed_pairs'])):raise ValueError('unknown seed index')
    o=import_original(p['original_source']);old=o.verify_plan(Path(p['training_plan']),p['training_plan_sha256'])
    rseed,dseed=p['seed_pairs'][args.index]
    out=args.output/f'seed_{rseed}';out.mkdir(parents=True,exist_ok=False)
    new_json(out/'START.json',{'plan_sha256':args.plan_sha256,'rescue_seed':rseed,'downstream_seed':dseed,'job_id':os.environ['SLURM_JOB_ID']})
    try:
        o.set_reproducible_seed(rseed,True);device=torch.device('cuda')
        untrained=o.Matcher().to(device);untrained.eval()
        loaded=torch.load(p['rescue_checkpoints'][args.index]['path'],map_location='cpu',weights_only=False)
        validate_rescue_checkpoint(loaded,json.loads(Path(p['rescue_checkpoints'][args.index]['history']).read_text()),p['training_plan_sha256'],rseed)
        matcher=o.Matcher().to(device);matcher.load_state_dict(loaded['state']);matcher.eval()
        downstream=next(x for x in old['downstream_checkpoints'] if x['seed']==dseed)
        ck=o.safe_torch_load(Path(downstream['path']),map_location=device);cfg=ck['config']
        kwargs={n:cfg[n] for n in signatures.signature(o.PartnerSpecificInteractionModel).parameters if n in cfg}
        for side in ('antigen','antibody'):kwargs[side+'_input_dim']=int(cfg.get(side+'_input_dim',cfg['input_dim']))
        model=o.PartnerSpecificInteractionModel(**kwargs).to(device);model.load_state_dict(ck['model_state_dict'],strict=True);model.eval()
        before={k:v.cpu().clone() for k,v in model.state_dict().items()}
        representation=o.build_sources(o.MANIFEST,esm_if1_root=o.ESMIF1)
        store=o.CompositeEmbeddingStore(o.EmbeddingStore(o.BASE,.5),representation)
        seqs,_=o.sequence_metadata();split=json.loads(o.SPLIT.read_text())
        frame=pd.read_parquet(o.PREP,columns=['instance_id','ab_sequence_sha256','ag_sequence_sha256']).set_index('instance_id',verify_integrity=True)
        def full(key):
            chunks,starts,length=store.load(key,device)
            return o.merge_chunk_embeddings(chunks,starts,length,device)
        def primary(b,e,parts):return o.hungarian_rescue(matcher,b,e,parts)
        def canonical(b,e,parts):
            restored=e.clone()
            for sl in o.chain_slices(parts):
                mapping=canonical_assignment(matcher.logits(b[sl],e[sl]).cpu().numpy(),e[sl].cpu().numpy())
                restored[sl]=e[sl][torch.tensor(mapping,device=device)]
            return restored
        def score(ag,ab,y,mask):
            prediction=model(ag.unsqueeze(0),ab.unsqueeze(0),torch.ones((1,len(ag)),dtype=torch.bool,device=device),torch.ones((1,len(ab)),dtype=torch.bool,device=device))['pair_probabilities'][0].cpu().numpy()
            return float(average_precision_score(y[mask],prediction[mask]))
        records=[];identity={}
        with torch.inference_mode():
            for iid in p['ids']:
                c=o.load_contact_map(o.CONTACTS/f'{iid}.npz',iid);y,mask=c.dense()
                if not mask.all() or not y[mask].any():raise ValueError('candidate support/positive count differs')
                for condition in p['conditions']:
                    mode,k=condition.split(':');sides=[];canonical_sides=[];untrained_sides=[]
                    for col,indices in [('ag_sequence_sha256',c.antigen_indices),('ab_sequence_sha256',c.antibody_indices)]:
                        key=str(frame.loc[iid,col]);joined=full(key);parts=seqs[key]
                        order=o.synthetic_order(parts,mode,int(k),old['corruption_seed'],key)
                        variants,present=observed_transforms(joined,parts,order,primary)
                        alternative=variants['corrupted'].clone();random_control=alternative.clone()
                        if present:
                            alternative[:,1280:-1]=canonical(alternative[:,:1280],alternative[:,1280:-1],parts)
                            random_control[:,1280:-1]=o.hungarian_rescue(untrained,random_control[:,:1280],random_control[:,1280:-1],parts)
                        ix=torch.as_tensor(indices.astype(np.int64),device=device)
                        sides.append({name:value.index_select(0,ix) for name,value in variants.items()})
                        canonical_sides.append(alternative.index_select(0,ix));untrained_sides.append(random_control.index_select(0,ix))
                    values={name+'_ap':score(sides[0][name],sides[1][name],y,mask) for name in ('aligned','corrupted','rescued','clean_rescued')}
                    values['untrained_rescued_ap']=score(*untrained_sides,y,mask)
                    values['canonical_rescued_ap']=score(*canonical_sides,y,mask)
                    identity[iid]=values['aligned_ap']
                    records.append({'instance_id':iid,'component':str(split['component_id_by_instance'][iid]),'rescue_seed':rseed,'downstream_seed':dseed,'condition':condition,**values})
        reference=json.loads(Path(next(x['path'] for x in p['historical_identity_receipts'] if x['seed']==dseed)).read_text())
        if reference['checkpoint_sha256']!=downstream['sha256']:raise ValueError('historical reference belongs to another checkpoint')
        identity_error=abs(float(np.mean(list(identity.values())))-reference['pair']['macro']['auprc'])
        if identity_error>1e-6:raise ValueError('historical identity parity failed '+str(identity_error))
        summary=summarize_records(records,p['ids'],[(rseed,dseed)],p['conditions'])
        if any(not torch.equal(v.cpu(),before[k]) for k,v in model.state_dict().items()):raise ValueError('downstream state mutated')
        verify();o.verify_plan(Path(p['training_plan']),p['training_plan_sha256'])
        new_json(out/'COMPLETE.json',{'status':'COMPLETE_SUPPLEMENTAL_RESCUE_QUALIFICATION','plan_sha256':args.plan_sha256,'records':records,'summary':summary,'identity_parity_error':identity_error,'test_labels_read':False,'ood_labels_read':False,'contact_selected':False})
    except BaseException as e:
        new_json(out/'FAILURE.json',{'error':repr(e),'plan_sha256':args.plan_sha256});raise


def aggregate(args):
    if digest(args.plan)!=args.plan_sha256:raise ValueError('evaluation plan hash mismatch')
    p=json.loads(args.plan.read_text());records=[];pins={}
    for seed,_ in p['seed_pairs']:
        path=args.input/f'seed_{seed}/COMPLETE.json'
        d=json.loads(path.read_text())
        if d['status']!='COMPLETE_SUPPLEMENTAL_RESCUE_QUALIFICATION' or d['plan_sha256']!=args.plan_sha256:
            raise ValueError('incomplete/mismatched seed receipt')
        records.extend(d['records']);pins[str(path)]=digest(path)
    summary=summarize_records(records,p['ids'],p['seed_pairs'],p['conditions'])
    comparisons=[]
    for condition in p['conditions']:
        filtered=[r for r in records if r['condition']==condition]
        components=sorted({r['component'] for r in filtered})
        if len(components)<2:raise ValueError('too few independent components')
        for plus,minus in [('rescued_ap','corrupted_ap'),('clean_rescued_ap','aligned_ap'),('rescued_ap','untrained_rescued_ap'),('canonical_rescued_ap','rescued_ap')]:
            matrix=[]
            for seed,_ in p['seed_pairs']:
                by=[r for r in filtered if r['rescue_seed']==seed]
                row=[]
                for c in components:
                    values=[r[plus]-r[minus] for r in by if r['component']==c]
                    if not values or not np.isfinite(values).all():raise ValueError('incomplete/nonfinite paired component')
                    row.append(float(np.mean(values)))
                matrix.append(row)
            matrix=np.asarray(matrix);rng=np.random.default_rng(20260930);draws=[]
            for _ in range(5000):
                si=rng.integers(len(matrix),size=len(matrix));ci=rng.integers(len(components),size=len(components))
                draws.append(float(matrix[si][:,ci].mean()))
            comparisons.append({'condition':condition,'contrast':plus+'_minus_'+minus,
                                'equal_component_equal_paired_seed_effect':float(matrix.mean()),
                                'pointwise_crossed_bootstrap_ci95':list(map(float,np.quantile(draws,[.025,.975]))),
                                'components':len(components),'rescue_downstream_seed_pairs':len(p['seed_pairs'])})
    new_json(args.output,{'status':'COMPLETE_EXPLORATORY_RESCUE_QUALIFICATION_ANALYSIS','summary':summary,'contrasts':comparisons,
                         'input_receipt_sha256':pins,'records':records,'plan_sha256':args.plan_sha256,
                         'test_labels_read':False,'ood_labels_read':False,'contact_selected':False,
                         'limitations':['Previously exposed validation, not blind','Paired rescue/downstream seeds are not a fully crossed factorial',
                                        'Pure-permutation conditions are not independent replications','Intervals are pointwise, exploratory, not multiplicity-adjusted']})


if __name__=='__main__':
    p=argparse.ArgumentParser();subs=p.add_subparsers(dest='action',required=True)
    f=subs.add_parser('prepare');f.add_argument('--training-plan',type=Path,required=True);f.add_argument('--training-plan-sha256',required=True);f.add_argument('--training-root',type=Path,required=True);f.add_argument('--plan',type=Path,required=True)
    f=subs.add_parser('run');f.add_argument('--plan',type=Path,required=True);f.add_argument('--plan-sha256',required=True);f.add_argument('--index',type=int,required=True);f.add_argument('--output',type=Path,required=True)
    f=subs.add_parser('aggregate');f.add_argument('--plan',type=Path,required=True);f.add_argument('--plan-sha256',required=True);f.add_argument('--input',type=Path,required=True);f.add_argument('--output',type=Path,required=True)
    a=p.parse_args();{'prepare':prepare,'run':run,'aggregate':aggregate}[a.action](a)
