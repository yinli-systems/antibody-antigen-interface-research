import copy
import math
from pathlib import Path
import tempfile
import unittest
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from analysis_adapter import metric_snapshot, unit_interval, collect


def evaluation():
    return {'stage':'evaluate','split':'validation','instance_count':217,
        'pair':{'macro':{'auprc':.02,'auroc':.8},'micro':{'auprc':.01}},
        'nodes':{'epitope':{'macro':{'auprc':.2}},'paratope':{'macro':{'auprc':.6}}},
        'partner_retrieval':{'instance_count':217,'candidate_count_per_instance':10,
                            'mean_reciprocal_rank':.3,'recall_at_1':.1}}

class SchemaTests(unittest.TestCase):
    def test_correct_mrr_source(self):
        self.assertEqual(metric_snapshot(evaluation())['partner_mrr'],.3)
    def test_primary_ap_unchanged(self):
        d=evaluation(); x=metric_snapshot(d)
        self.assertEqual(x['pair_macro_ap'],d['pair']['macro']['auprc'])
        self.assertEqual(x['pair_micro_ap'],d['pair']['micro']['auprc'])
    def test_old_key_failure_reproduced(self):
        with self.assertRaises(KeyError): evaluation()['partner_retrieval']['mrr']
    def test_ambiguous_key_rejected(self):
        d=evaluation();d['partner_retrieval']['mrr']=.9
        with self.assertRaises(ValueError):metric_snapshot(d)
    def test_test_split_rejected(self):
        d=evaluation();d['split']='test'
        with self.assertRaises(ValueError):metric_snapshot(d)
    def test_smoke_rejected(self):
        d=evaluation();d['instance_count']=24
        with self.assertRaises(ValueError):metric_snapshot(d)
    def test_distractor_change_rejected(self):
        d=evaluation();d['partner_retrieval']['candidate_count_per_instance']=16
        with self.assertRaises(ValueError):metric_snapshot(d)
    def test_invalid_metric(self):
        for v in (True,None,float('nan'),float('inf'),-1,1.01,'0.2'):
            with self.assertRaises(ValueError):unit_interval(v)
    def test_partial_collection_cannot_promote(self):
        with tempfile.TemporaryDirectory() as tmp:
            p={'tasks':[{'task_id':i} for i in range(36)]}
            d=collect(p,'a'*64,Path(tmp))
            self.assertEqual(d['status'],'INCOMPLETE_NO_LAYER_SELECTION')
            self.assertEqual(len(d['missing_task_ids']),36)
            with self.assertRaises(RuntimeError):collect(p,'a'*64,Path(tmp),True)

if __name__=='__main__':unittest.main(verbosity=2)
