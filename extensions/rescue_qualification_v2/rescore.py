#!/usr/bin/env python3
"""Fixed, label-independent rescoring of stored RePaRank validation outputs."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
from qualification_core import digest,new_json

METHODS=('identity','row_center','column_center','double_center','local_scale_10')


def transform(score,method):
    s=np.asarray(score,dtype=np.float64)
    if s.ndim!=2 or min(s.shape)<1 or not np.isfinite(s).all(): raise ValueError('invalid score matrix')
    if method=='identity':return s.copy()
    row=s.mean(1,keepdims=True);col=s.mean(0,keepdims=True)
    if method=='row_center':return s-row
    if method=='column_center':return s-col
    if method=='double_center':return s-row-col+s.mean()
    if method=='local_scale_10':
        kr=min(10,s.shape[1]);kc=min(10,s.shape[0])
        r=np.partition(s,s.shape[1]-kr,axis=1)[:,-kr:].mean(1,keepdims=True)
        c=np.partition(s,s.shape[0]-kc,axis=0)[-kc:,:].mean(0,keepdims=True)
        return 2*s-r-c
    raise ValueError('unregistered score method')


def dense_grid(ag,ab,labels,scores):
    ag=np.asarray(ag);ab=np.asarray(ab);y=np.asarray(labels);s=np.asarray(scores)
    if not (ag.ndim==ab.ndim==y.ndim==s.ndim==1 and len(ag)==len(ab)==len(y)==len(s)):
        raise ValueError('candidate array lengths/shape mismatch')
    if not len(y) or not np.isin(y,[0,1]).all() or not np.isfinite(s).all():raise ValueError('invalid labels/scores')
    if not np.issubdtype(ag.dtype,np.integer) or not np.issubdtype(ab.dtype,np.integer):raise ValueError('residue indices must be integers')
    ua,ia=np.unique(ag,return_inverse=True);ub,ib=np.unique(ab,return_inverse=True)
    codes=ia*len(ub)+ib
    if len(y)!=len(ua)*len(ub) or len(np.unique(codes))!=len(y):raise ValueError('incomplete/duplicate candidate grid')
    shape=(len(ua),len(ub));sm=np.empty(shape,dtype=np.float64);ym=np.empty(shape,dtype=np.uint8)
    sm[ia,ib]=s;ym[ia,ib]=y
    return sm,ym


def precision_top_k(y,s,k=10):
    y=np.asarray(y).ravel();s=np.asarray(s).ravel();k=min(k,len(s))
    if k<1:raise ValueError('empty top-k')
    cut=np.partition(s,len(s)-k)[-k];before=s>cut;tied=s==cut
    return float((y[before].sum()+(k-before.sum())*y[tied].mean())/k)


def freeze(args):
    source=Path(__file__).parent
    split=json.loads(args.split.read_text());ids=sorted(map(str,split['splits']['validation']))
    if len(ids)!=217 or set(ids)&(set(split['splits']['train'])|set(split['splits']['test'])):
        raise ValueError('validation authority mismatch')
    files=sorted((args.input/'pair_predictions').glob('*.npz'))
    if {p.stem for p in files}!=set(ids):raise ValueError('raw output filenames must cover exactly 217 cases')
    inputs={str(p.resolve()):digest(p) for p in files}
    identity=args.input/'evaluation.json';inputs[str(identity.resolve())]=digest(identity)
    inputs[str(args.split.resolve())]=digest(args.split)
    result={'schema':'fixed_rescore_diagnostic_v1','created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            'input_root':str(args.input.resolve()),'split':str(args.split.resolve()),'ids':ids,'methods':list(METHODS),
            'primary_contrast':'local_scale_10_minus_identity','bootstrap_draws':5000,'bootstrap_seed':20260930,
            'source_sha256':{str(source/n):digest(source/n) for n in ('rescore.py','qualification_core.py','RESCORE_PROTOCOL.md')},
            'input_sha256':inputs,'identity_receipt':str(identity.resolve()),
            'test_access':False,'ood_access':False,'training_or_inference':False,'blind_claim':False}
    new_json(args.plan,result);print('FROZEN',digest(args.plan))


def run(args):
    from sklearn.metrics import average_precision_score
    if digest(args.plan)!=args.plan_sha256:raise ValueError('plan digest differs')
    p=json.loads(args.plan.read_text())
    if p['methods']!=list(METHODS) or p['test_access'] or p['ood_access']:raise ValueError('unregistered methods/data access')
    for path,h in {**p['source_sha256'],**p['input_sha256']}.items():
        if digest(path)!=h:raise ValueError('input/source drift '+path)
    split=json.loads(Path(p['split']).read_text());component={i:str(split['component_id_by_instance'][i]) for i in p['ids']}
    if len(component)!=217:raise ValueError('incomplete components')
    expected=json.loads(Path(p['identity_receipt']).read_text())
    rows=[];pair_count=positive_count=0
    start=time.monotonic()
    for iid in p['ids']:
        with np.load(Path(p['input_root'])/'pair_predictions'/(iid+'.npz'),allow_pickle=False) as z:
            if str(z['instance_id'].item())!=iid:raise ValueError('NPZ ID differs')
            s,y=dense_grid(z['antigen_residue_index'],z['antibody_residue_index'],z['label'],z['score'])
        if not y.any() or y.all():raise ValueError('undefined AP case cannot silently be dropped')
        pair_count+=y.size;positive_count+=int(y.sum())
        for method in METHODS:
            values=transform(s,method)
            rows.append({'instance_id':iid,'component':component[iid],'method':method,
                         'pair_ap':float(average_precision_score(y.ravel(),values.ravel())),
                         'precision_top10':precision_top_k(y,values),'pair_count':int(y.size),'positive_count':int(y.sum()),
                         'antigen_candidates':s.shape[0],'antibody_candidates':s.shape[1]})
    baseline={r['instance_id']:r for r in rows if r['method']=='identity'}
    identity_ap=float(np.mean([r['pair_ap'] for r in baseline.values()]))
    delta=identity_ap-expected['primary_macro_mean_per_complex_AP']
    if abs(delta)>1e-7:raise ValueError('identity parity failed '+str(delta))
    old=expected['pooled_pair_metrics']
    if pair_count!=old['pair_count'] or positive_count!=old['positive_count']:raise ValueError('candidate counts differ')
    group_ids=sorted(set(component.values()));arms=[];contrasts=[]
    for method in METHODS:
        selected=[r for r in rows if r['method']==method]
        arms.append({'method':method,'macro_pair_ap':float(np.mean([r['pair_ap'] for r in selected])),
                     'mean_top10_precision':float(np.mean([r['precision_top10'] for r in selected]))})
        if method=='identity':continue
        diffs={r['instance_id']:r['pair_ap']-baseline[r['instance_id']]['pair_ap'] for r in selected}
        groups=np.array([np.mean([diffs[i] for i in p['ids'] if component[i]==g]) for g in group_ids])
        rng=np.random.default_rng(p['bootstrap_seed']);samples=groups[rng.integers(len(groups),size=(p['bootstrap_draws'],len(groups)))].mean(1)
        contrasts.append({'method_minus_identity':method,'equal_complex_difference':float(np.mean(list(diffs.values()))),
                          'equal_component_difference':float(groups.mean()),'component_bootstrap_ci95':list(map(float,np.quantile(samples,[.025,.975]))),
                          'positive_component_fraction':float((groups>0).mean()),'negative_component_fraction':float((groups<0).mean())})
    for path,h in {**p['source_sha256'],**p['input_sha256']}.items():
        if digest(path)!=h:raise ValueError('input/source changed during read '+path)
    result={'status':'COMPLETE_EXPLORATORY_FIXED_RESCORE_DIAGNOSTIC','at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            'plan_sha256':args.plan_sha256,'source_sha256':p['source_sha256'],'input_sha256':p['input_sha256'],
            'complex_count':217,'component_count':len(group_ids),'pair_count':pair_count,'positive_count':positive_count,
            'arms':arms,'contrasts':contrasts,'records':rows,'identity_parity_abs_difference':abs(delta),
            'wall_seconds':time.monotonic()-start,'new_training':False,'new_model_inference':False,'new_gpu_jobs':0,
            'test_labels_read':False,'ood_labels_read':False,'baseline_training_overlap_unknown':True,
            'interpretation':'fixed exploratory rescoring on exposed development; no learned model or SOTA promotion'}
    new_json(args.output,result);print(json.dumps({k:v for k,v in result.items() if k not in ('records','input_sha256','source_sha256')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();subs=p.add_subparsers(dest='action',required=True)
    f=subs.add_parser('freeze');f.add_argument('--input',type=Path,required=True);f.add_argument('--split',type=Path,required=True);f.add_argument('--plan',type=Path,required=True)
    r=subs.add_parser('run');r.add_argument('--plan',type=Path,required=True);r.add_argument('--plan-sha256',required=True);r.add_argument('--output',type=Path,required=True)
    a=p.parse_args();{'freeze':freeze,'run':run}[a.action](a)
