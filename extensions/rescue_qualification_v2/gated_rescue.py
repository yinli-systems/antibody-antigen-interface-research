#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, os
from pathlib import Path
import numpy as np
from qualification_core import digest,new_json,validate_rescue_checkpoint,observed_transforms,summarize_records
from qualified_evaluate import import_original


def empirical_upper_quantile(values,q=.95):
    x=sorted(float(v) for v in values)
    if not x or not all(math.isfinite(v) for v in x):raise ValueError('invalid calibration values')
    # nearest-rank / higher empirical quantile; version-independent
    i=max(0,min(len(x)-1,math.ceil(q*len(x))-1))
    return x[i]


def assignment_gain(matcher,base,extra,parts):
    import torch
    from scipy.optimize import linear_sum_assignment
    total=0.0;n=0
    with torch.inference_mode():
        for sl in matcher_chain_slices(parts):
            logits=matcher.logits(base[sl],extra[sl]).detach().cpu().numpy()
            rows,cols=linear_sum_assignment(-logits)
            if not np.array_equal(rows,np.arange(len(rows))):raise AssertionError('row assignment changed')
            total+=float(logits[rows,cols].sum()-np.trace(logits));n+=len(rows)
    return total/max(n,1)


def matcher_chain_slices(parts):
    start=0
    for seq in parts:
        yield slice(start,start+len(seq));start+=len(seq)


def load_eval_plan(path,sha):
    if digest(path)!=sha:raise ValueError('evaluation plan hash mismatch')
    p=json.loads(Path(path).read_text())
    if p['test_access'] or p['ood_access'] or p['contact_selection']:raise ValueError('closed-data authority violation')
    return p


def calibrate(args):
    import torch
    p=load_eval_plan(args.evaluation_plan,args.evaluation_plan_sha256);o=import_original(p['original_source']);old=o.verify_plan(Path(p['training_plan']),p['training_plan_sha256'])
    rseed,_=p['seed_pairs'][args.index];ckrow=p['rescue_checkpoints'][args.index]
    o.set_reproducible_seed(rseed,True);device=torch.device('cuda')
    untrained=o.Matcher(hidden=old['hidden_dim']).to(device);untrained.eval()
    loaded=torch.load(ckrow['path'],map_location='cpu',weights_only=False);validate_rescue_checkpoint(loaded,json.loads(Path(ckrow['history']).read_text()),p['training_plan_sha256'],rseed)
    trained=o.Matcher(hidden=old['hidden_dim']).to(device);trained.load_state_dict(loaded['state']);trained.eval()
    seqs,_=o.sequence_metadata();source=o.build_sources(o.MANIFEST,esm_if1_root=o.ESMIF1)[0];store=o.EmbeddingStore(o.BASE,.5)
    clean_t=[];clean_u=[];corrupt_t={c:[] for c in p['conditions']};corrupt_u={c:[] for c in p['conditions']};per=[]
    for n,key in enumerate(old['train_keys'],1):
        base,extra=o.load_full_features(key,store,source,device);parts=seqs[key]
        gt=assignment_gain(trained,base,extra,parts);gu=assignment_gain(untrained,base,extra,parts);clean_t.append(gt);clean_u.append(gu)
        row={'sequence_key':key,'trained_clean_gain':gt,'untrained_clean_gain':gu,'corruptions':{}}
        for cond in p['conditions']:
            mode,k=cond.split(':');order=o.synthetic_order(parts,mode,int(k),old['corruption_seed'],key);ii=torch.as_tensor(order,device=device,dtype=torch.long);x=extra.index_select(0,ii)
            a=assignment_gain(trained,base,x,parts);b=assignment_gain(untrained,base,x,parts);corrupt_t[cond].append(a);corrupt_u[cond].append(b);row['corruptions'][cond]={'trained_gain':a,'untrained_gain':b}
        per.append(row)
        if n%250==0:print('calibration',rseed,n,len(old['train_keys']),flush=True)
    tt=empirical_upper_quantile(clean_t,.95);tu=empirical_upper_quantile(clean_u,.95)
    payload={'status':'COMPLETE_TRAIN_ONLY_GATE_CALIBRATION','rescue_seed':rseed,'train_key_count':len(old['train_keys']),'trained_threshold':tt,'untrained_threshold':tu,
             'trained_clean_activation':float(np.mean(np.asarray(clean_t)>tt)),'untrained_clean_activation':float(np.mean(np.asarray(clean_u)>tu)),
             'condition_activation':{c:{'trained':float(np.mean(np.asarray(corrupt_t[c])>tt)),'untrained':float(np.mean(np.asarray(corrupt_u[c])>tu))} for c in p['conditions']},
             'per_key':per,'threshold_selection':'higher empirical 95th percentile of clean training assignment gain only','contact_labels_read':False,'test_labels_read':False,'ood_labels_read':False,
             'evaluation_plan_sha256':args.evaluation_plan_sha256,'checkpoint_sha256':ckrow['sha256']}
    new_json(args.output,payload);print(json.dumps({k:payload[k] for k in ('rescue_seed','trained_threshold','untrained_threshold','trained_clean_activation','condition_activation')},indent=2))


def prepare(args):
    p=load_eval_plan(args.evaluation_plan,args.evaluation_plan_sha256);rows=[]
    for rseed,_ in p['seed_pairs']:
        f=args.calibration_root/f'seed_{rseed}.json';d=json.loads(f.read_text())
        if d['status']!='COMPLETE_TRAIN_ONLY_GATE_CALIBRATION' or d['rescue_seed']!=rseed or d['evaluation_plan_sha256']!=args.evaluation_plan_sha256:raise ValueError('calibration receipt mismatch')
        rows.append({'seed':rseed,'path':str(f.resolve()),'sha256':digest(f),'trained_threshold':d['trained_threshold'],'untrained_threshold':d['untrained_threshold']})
    here=Path(__file__).resolve().parent
    plan={'schema':'train_only_gated_rescue_v1','created_from_evaluation_plan':str(args.evaluation_plan.resolve()),'evaluation_plan_sha256':args.evaluation_plan_sha256,'calibrations':rows,
          'seed_pairs':p['seed_pairs'],'conditions':p['conditions'],'ids':p['ids'],'source_sha256':{str(here/'gated_rescue.py'):digest(here/'gated_rescue.py'),str(here/'GATED_RESCUE_PROTOCOL.md'):digest(here/'GATED_RESCUE_PROTOCOL.md')},
          'threshold_selected_with_contact_metrics':False,'test_access':False,'ood_access':False}
    new_json(args.output,plan);print('GATED_PLAN',digest(args.output))


def run(args):
    import torch,pandas as pd,inspect as signatures
    from sklearn.metrics import average_precision_score
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('validation contact evaluation requires scheduler')
    if digest(args.plan)!=args.plan_sha256:raise ValueError('gated plan hash mismatch')
    gp=json.loads(args.plan.read_text());
    for f,h in gp['source_sha256'].items():
        if digest(f)!=h:raise ValueError('source drift '+f)
    p=load_eval_plan(Path(gp['created_from_evaluation_plan']),gp['evaluation_plan_sha256']);o=import_original(p['original_source']);old=o.verify_plan(Path(p['training_plan']),p['training_plan_sha256'])
    rseed,dseed=gp['seed_pairs'][args.index];cal=next(x for x in gp['calibrations'] if x['seed']==rseed)
    if digest(cal['path'])!=cal['sha256']:raise ValueError('calibration drift')
    out=args.output/f'seed_{rseed}';out.mkdir(parents=True,exist_ok=False);new_json(out/'START.json',{'plan_sha256':args.plan_sha256,'job_id':os.environ['SLURM_JOB_ID'],'rescue_seed':rseed,'downstream_seed':dseed})
    o.set_reproducible_seed(rseed,True);device=torch.device('cuda');untrained=o.Matcher(hidden=old['hidden_dim']).to(device);untrained.eval()
    ckrow=p['rescue_checkpoints'][args.index];loaded=torch.load(ckrow['path'],map_location='cpu',weights_only=False);validate_rescue_checkpoint(loaded,json.loads(Path(ckrow['history']).read_text()),p['training_plan_sha256'],rseed)
    matcher=o.Matcher(hidden=old['hidden_dim']).to(device);matcher.load_state_dict(loaded['state']);matcher.eval()
    downstream=next(x for x in old['downstream_checkpoints'] if x['seed']==dseed);ck=o.safe_torch_load(Path(downstream['path']),map_location=device);cfg=ck['config'];kwargs={n:cfg[n] for n in signatures.signature(o.PartnerSpecificInteractionModel).parameters if n in cfg};
    for side in ('antigen','antibody'):kwargs[side+'_input_dim']=int(cfg.get(side+'_input_dim',cfg['input_dim']))
    model=o.PartnerSpecificInteractionModel(**kwargs).to(device);model.load_state_dict(ck['model_state_dict'],strict=True);model.eval();before={k:v.cpu().clone() for k,v in model.state_dict().items()}
    reps=o.build_sources(o.MANIFEST,esm_if1_root=o.ESMIF1);store=o.CompositeEmbeddingStore(o.EmbeddingStore(o.BASE,.5),reps);seqs,_=o.sequence_metadata();split=json.loads(o.SPLIT.read_text());frame=pd.read_parquet(o.PREP,columns=['instance_id','ab_sequence_sha256','ag_sequence_sha256']).set_index('instance_id',verify_integrity=True)
    def full(key):chunks,starts,length=store.load(key,device);return o.merge_chunk_embeddings(chunks,starts,length,device)
    def score(ag,ab,y,mask):
        with torch.inference_mode():pred=model(ag.unsqueeze(0),ab.unsqueeze(0),torch.ones((1,len(ag)),dtype=torch.bool,device=device),torch.ones((1,len(ab)),dtype=torch.bool,device=device))['pair_probabilities'][0].cpu().numpy()
        return float(average_precision_score(y[mask],pred[mask]))
    records=[]
    for pos,iid in enumerate(gp['ids'],1):
        c=o.load_contact_map(o.CONTACTS/f'{iid}.npz',iid);y,mask=c.dense();
        for cond in gp['conditions']:
            mode,k=cond.split(':');side_rows=[];activation={'trained_corrupt':0,'trained_clean':0,'untrained_corrupt':0};present_sides=0
            for col,indices in [('ag_sequence_sha256',c.antigen_indices),('ab_sequence_sha256',c.antibody_indices)]:
                key=str(frame.loc[iid,col]);joined=full(key);parts=seqs[key];order=o.synthetic_order(parts,mode,int(k),old['corruption_seed'],key);variants,present=observed_transforms(joined,parts,order,lambda b,e,p:o.hungarian_rescue(matcher,b,e,p));gcor=variants['corrupted'].clone();gclean=variants['aligned'].clone();ug=variants['corrupted'].clone()
                if present:
                    present_sides+=1;base=joined[:,:1280];extra=joined[:,1280:-1];corrupt=variants['corrupted'][:,1280:-1]
                    tc=assignment_gain(matcher,base,corrupt,parts)>cal['trained_threshold'];tl=assignment_gain(matcher,base,extra,parts)>cal['trained_threshold'];uc=assignment_gain(untrained,base,corrupt,parts)>cal['untrained_threshold']
                    activation['trained_corrupt']+=int(tc);activation['trained_clean']+=int(tl);activation['untrained_corrupt']+=int(uc)
                    if tc:gcor[:,1280:-1]=o.hungarian_rescue(matcher,base,corrupt,parts)
                    if tl:gclean[:,1280:-1]=o.hungarian_rescue(matcher,base,extra,parts)
                    if uc:ug[:,1280:-1]=o.hungarian_rescue(untrained,base,corrupt,parts)
                ix=torch.as_tensor(indices.astype(np.int64),device=device);side_rows.append({'aligned':variants['aligned'].index_select(0,ix),'corrupted':variants['corrupted'].index_select(0,ix),'always_rescued':variants['rescued'].index_select(0,ix),'gated_rescued':gcor.index_select(0,ix),'clean_gated':gclean.index_select(0,ix),'untrained_gated':ug.index_select(0,ix)})
            vals={name+'_ap':score(side_rows[0][name],side_rows[1][name],y,mask) for name in side_rows[0]}
            records.append({'instance_id':iid,'component':str(split['component_id_by_instance'][iid]),'rescue_seed':rseed,'downstream_seed':dseed,'condition':cond,'present_sides':present_sides,**activation,**vals})
        if pos%25==0:print('gated eval',rseed,pos,flush=True)
    if any(not torch.equal(v.cpu(),before[k]) for k,v in model.state_dict().items()):raise ValueError('downstream state mutated')
    new_json(out/'COMPLETE.json',{'status':'COMPLETE_GATED_RESCUE_DEVELOPMENT_EVALUATION','plan_sha256':args.plan_sha256,'records':records,'test_labels_read':False,'ood_labels_read':False,'threshold_selected_with_contact_metrics':False})


def aggregate(args):
    if digest(args.plan)!=args.plan_sha256:raise ValueError('plan drift')
    p=json.loads(args.plan.read_text());records=[];pins={}
    for seed,_ in p['seed_pairs']:
        f=args.input/f'seed_{seed}/COMPLETE.json';d=json.loads(f.read_text());
        if d['status']!='COMPLETE_GATED_RESCUE_DEVELOPMENT_EVALUATION' or d['plan_sha256']!=args.plan_sha256:raise ValueError('incomplete seed')
        records+=d['records'];pins[str(f)]=digest(f)
    comps=sorted(set(r['component'] for r in records));out=[]
    for cond in p['conditions']:
        rr=[r for r in records if r['condition']==cond]
        row={'condition':cond,'means':{k:float(np.mean([r[k] for r in rr])) for k in ['aligned_ap','corrupted_ap','always_rescued_ap','gated_rescued_ap','clean_gated_ap','untrained_gated_ap']},'activation_rates':{k:float(sum(r[k] for r in rr)/max(sum(r['present_sides'] for r in rr),1)) for k in ['trained_corrupt','trained_clean','untrained_corrupt']},'contrasts':[]}
        for plus,minus in [('gated_rescued_ap','corrupted_ap'),('clean_gated_ap','aligned_ap'),('gated_rescued_ap','untrained_gated_ap'),('gated_rescued_ap','always_rescued_ap')]:
            mat=[]
            for seed,_ in p['seed_pairs']:
                by=[r for r in rr if r['rescue_seed']==seed];mat.append([float(np.mean([r[plus]-r[minus] for r in by if r['component']==c])) for c in comps])
            mat=np.asarray(mat);rng=np.random.default_rng(20260930);draw=[]
            for _ in range(5000):
                si=rng.integers(3,size=3);ci=rng.integers(len(comps),size=len(comps));draw.append(float(mat[si][:,ci].mean()))
            row['contrasts'].append({'contrast':plus+'_minus_'+minus,'equal_component_equal_paired_seed_effect':float(mat.mean()),'pointwise_crossed_bootstrap_ci95':list(map(float,np.quantile(draw,[.025,.975])))})
        out.append(row)
    new_json(args.output,{'status':'COMPLETE_TRAIN_ONLY_GATED_RESCUE_ANALYSIS','conditions':out,'records':records,'input_sha256':pins,'components':len(comps),'seed_pairs':3,'test_labels_read':False,'ood_labels_read':False,'limitations':['development validation only','paired seeds not full factorial','three corruption families are not independent biological replications','no post-hoc clean noninferiority margin']})


def main():
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='cmd',required=True)
    q=s.add_parser('calibrate');q.add_argument('--evaluation-plan',type=Path,required=True);q.add_argument('--evaluation-plan-sha256',required=True);q.add_argument('--index',type=int,required=True);q.add_argument('--output',type=Path,required=True)
    q=s.add_parser('prepare');q.add_argument('--evaluation-plan',type=Path,required=True);q.add_argument('--evaluation-plan-sha256',required=True);q.add_argument('--calibration-root',type=Path,required=True);q.add_argument('--output',type=Path,required=True)
    q=s.add_parser('run');q.add_argument('--plan',type=Path,required=True);q.add_argument('--plan-sha256',required=True);q.add_argument('--index',type=int,required=True);q.add_argument('--output',type=Path,required=True)
    q=s.add_parser('aggregate');q.add_argument('--plan',type=Path,required=True);q.add_argument('--plan-sha256',required=True);q.add_argument('--input',type=Path,required=True);q.add_argument('--output',type=Path,required=True)
    a=p.parse_args();{'calibrate':calibrate,'prepare':prepare,'run':run,'aggregate':aggregate}[a.cmd](a)
if __name__=='__main__':main()
