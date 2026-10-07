#!/usr/bin/env python3
"""Real feature/matcher algebra check. No training update or contact scoring."""
import argparse, importlib.util, json, time
from pathlib import Path
import numpy as np
import torch
from qualification_core import canonical_assignment,new_json,digest,observed_transforms
from preflight import inspect


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--plan-sha256',required=True);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    metadata=inspect(a.plan,a.plan_sha256)
    plan=json.loads(a.plan.read_text())
    source=a.source.resolve()
    if str(source) not in plan['input_sha256'] or digest(source)!=plan['input_sha256'][str(source)]:
        raise ValueError('original matcher source is not exactly pinned')
    spec=importlib.util.spec_from_file_location('original_rescue_v1',source)
    original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
    torch.set_num_threads(2);torch.manual_seed(101)
    matcher=original.Matcher().to(torch.device('cpu'));matcher.eval()
    seqs,_=original.sequence_metadata();representation=original.build_sources(original.MANIFEST,esm_if1_root=original.ESMIF1)[0]
    base_store=original.EmbeddingStore(original.BASE,.25)
    samples=[]
    for role in ['train','validation']:
        eligible=[k for k in plan[role+'_keys'] if 8<=sum(map(len,seqs[k]))<=256]
        chosen=sorted(eligible)[:2]
        if len(chosen)!=2:raise ValueError('probe requires two eligible chains per role')
        for key in chosen:
            base,extra=original.load_full_features(key,base_store,representation,torch.device('cpu'))
            # Float64 isolates algebra from ordinary matrix-reduction FP32 roundoff.
            model=original.Matcher();model.model=model.model.double();model.temperature.data=model.temperature.data.double()
            model.load_state_dict(matcher.state_dict());model.eval()
            base=base.double();extra=extra.double();parts=seqs[key]
            losses=[];gradients=[];rescued=[];canonical=[]
            conditions=[('identity',0)]+list(original.CONDITIONS)
            for mode,k in conditions:
                loss=original.loss_for_key(model,base,extra,parts,mode,k,plan['corruption_seed'],key)
                grad=torch.autograd.grad(loss,model.parameters())
                losses.append(float(loss));gradients.append(torch.cat([x.flatten() for x in grad]))
                order=original.synthetic_order(parts,mode,k,plan['corruption_seed'],key)
                corrupt=extra[torch.tensor(order)]
                rescued.append(original.hungarian_rescue(model,base,corrupt,parts))
                output=corrupt.clone()
                for sl in original.chain_slices(parts):
                    with torch.no_grad():scores=model.logits(base[sl],corrupt[sl]).numpy()
                    mapping=canonical_assignment(scores,corrupt[sl].numpy());output[sl]=corrupt[sl][torch.tensor(mapping)]
                canonical.append(output)
            loss_error=max(abs(v-losses[0]) for v in losses)
            grad_error=max(float((v-gradients[0]).abs().max()) for v in gradients)
            record={'role':role,'sequence_key':key,'chain_lengths':list(map(len,parts)),
                    'loss_max_abs_delta':loss_error,'parameter_gradient_max_abs_delta':grad_error,
                    'original_hard_output_equal_across_all_pure_permutations':all(torch.equal(v,rescued[0]) for v in rescued),
                    'canonical_hard_output_equal_across_all_pure_permutations':all(torch.equal(v,canonical[0]) for v in canonical),
                    'conditions':[f'{m}:{k}' for m,k in conditions]}
            if loss_error>1e-10 or grad_error>1e-10 or not record['canonical_hard_output_equal_across_all_pure_permutations']:
                raise RuntimeError('algebra probe failed: '+json.dumps(record))
            samples.append(record)
    missing_checks=[]
    for key in metadata['absent_validation_keys']:
        try:
            original.load_full_features(key,base_store,representation,torch.device('cpu'))
        except ValueError as error: old_error=str(error)
        else: raise RuntimeError('expected old evaluator missing-feature failure not reproduced')
        composite=original.CompositeEmbeddingStore(base_store,[representation])
        chunks,starts,length=composite.load(key,torch.device('cpu'))
        joined=original.merge_chunk_embeddings(chunks,starts,length,torch.device('cpu'))
        def must_not_run(*args): raise RuntimeError('matcher called for absent source')
        variants,present=observed_transforms(joined,seqs[key],np.arange(length),must_not_run)
        if present or not all(torch.equal(x,joined) for x in variants.values()): raise RuntimeError('missingness preservation failed')
        missing_checks.append({'sequence_key':key,'length':length,'original_failure':old_error,
                               'replacement_presence_zero':True,'all_four_arms_unchanged':True,'matcher_called':False})
    result={'status':'PASS_REAL_FEATURE_EQUIVARIANCE_PROBE','at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            'samples':samples,'missing_feature_checks':missing_checks,'feature_pairs':len(samples),'dtype':'float64','device':'CPU',
            'matcher':'untrained seeded original v1 model; not a trained performance measurement',
            'metadata':metadata,'contact_labels_read':False,'test_labels_read':False,'ood_labels_read':False,
            'model_training_performed':False,'downstream_contact_inference_performed':False,
            'original_source_sha256':digest(source),'live_training_modified':False}
    new_json(a.output,result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
