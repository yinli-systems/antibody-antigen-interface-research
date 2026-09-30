#!/usr/bin/env python3
"""End-to-end SYNTHETIC fixture of the 36-task aggregation; no real scores.

Temporary checkpoints are labelled byte fixtures, not model weights. All fixture
files are removed when the isolated test directory closes. Only a test receipt
may be saved by --receipt. Never use this output as an experiment result.
"""
import argparse
import copy
import json
import tempfile
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from af3 import digest,new_json
from analysis_adapter import run_analysis


def main():
    import pandas as pd
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--real-plan',required=True)
    p.add_argument('--receipt',required=True)
    a=p.parse_args()
    original=json.loads(Path(a.real_plan).read_text())
    with tempfile.TemporaryDirectory(prefix='SYNTHETIC-analysis-') as tmp:
        root=Path(tmp); ids=['synthetic_%03d'%i for i in range(217)]
        split=root/'split.json'
        new_json(split,{'splits':{'validation':ids,'train':[],'test':[]},
                       'component_id_by_instance':{i:'cluster_%02d'%(k%68) for k,i in enumerate(ids)}})
        plan=copy.deepcopy(original)
        plan['paths']['split_manifest']=str(split)
        plan['input_sha256']={str(split):digest(split)}
        # Original aggregator bytes remain the exact production source; only
        # the dataset and weight files are isolated synthetic fixtures.
        plan['source_sha256']={k:v for k,v in original['source_sha256'].items()
                              if k.endswith('aggregate_structural_training.py')}
        new_json(root/'PLAN.json',plan); sha=digest(root/'PLAN.json')
        for task in plan['tasks']:
            r=root/'runs'/('task_%02d'%task['task_id']); (r/'train').mkdir(parents=True)
            real=task['kind']=='real'; ap=1.0 if real else 0.5
            for name in ('best.pt','last.pt'):
                (r/'train'/name).write_bytes(b'SYNTHETIC TEST BYTES, NOT MODEL WEIGHTS')
            wsha=digest(r/'train/best.pt')
            for side in ('best','last'):
                ev={'stage':'evaluate','split':'validation','instance_count':217,
                    'checkpoint_sha256':wsha,'split_manifest_sha256':digest(split),
                    'pair':{'macro':{'auprc':ap,'auroc':1.0 if real else 0.0},'micro':{'auprc':ap}},
                    'nodes':{'epitope':{'macro':{'auprc':0.5}},'paratope':{'macro':{'auprc':0.5}}},
                    'partner_retrieval':{'candidate_count_per_instance':10,'instance_count':217,
                                        'mean_reciprocal_rank':0.25,'recall_at_1':0.1}}
                new_json(r/('validation_'+side)/'evaluation.json',ev)
            new_json(r/'COMPLETE.json',{'status':'COMPLETE_DEVELOPMENT_TRAINING','task':task,
                'plan_sha256':sha,'test_labels_read':False,'ood_labels_read':False,'smoke':False,
                'best_checkpoint_sha256':wsha,'last_checkpoint_sha256':wsha,
                'best_primary_pair_ap':ap,'last_primary_pair_ap':ap,
                'parameter_count':1234,'wall_seconds':1})
            frame=pd.DataFrame({'instance_id':[i for i in ids for _ in range(2)],
                                'label':[1,0]*217,'score':([1.0,0.0] if real else [0.0,1.0])*217})
            frame.to_parquet(r/'validation_best/pair_predictions.parquet',index=False)
        output=root/'isolated_output/ANALYSIS.json'
        run_analysis(root/'PLAN.json',sha,root,output)
        d=json.loads(output.read_text())
        assert d['task_count']==36 and d['arm_count']==12
        assert d['validation_complex_count']==217 and d['component_count']==68
        assert len(d['real_vs_width_matched_control'])==10
        for c in d['real_vs_width_matched_control']:
            analysis=c['paired_complex_component_analysis']
            assert abs(analysis['mean_difference']-0.5)<1e-12
            assert all(abs(x-0.5)<1e-12 for x in analysis['crossed_seed_component_ci95'])
        assert d['development_layer_selection']['saprot']['selected_layer']==0
        assert d['development_layer_selection']['prosst']['selected_layer']==0
        assert (output.parent/'ADAPTER_COMPLETE.json').is_file()
    new_json(a.receipt,{'status':'PASS_SYNTHETIC_END_TO_END_AGGREGATION',
        'synthetic_tasks':36,'synthetic_complexes':217,'synthetic_components':68,
        'known_effects_checked':10,'shallower_layer_tie_rule_checked':True,
        'real_model_or_dataset_scored':False,'scientific_result':False,
        'adapter_sha256':digest(Path(__file__).parent/'analysis_adapter.py'),
        'legacy_aggregator_sha256':original['source_sha256']['extensions/frontier_multimodel_v1/aggregate_structural_training.py']})
    print('PASS: complete SYNTHETIC 36-task pipeline; not a scientific measurement')

if __name__=='__main__':main()
