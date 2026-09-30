from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
import af3
import strict_finalize_existing as strict


class StrictFinalizer(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.plan={'run_root':str(self.root/'runs'),'legacy_plan_sha256':'l'*64,
                   'tasks':[{'task_id':0,'sequence_sha256':'s'*64,'length':9}],
                   'model_seeds':[42], 'samples_per_seed':1}
        task=self.root/'runs/task_000';task.mkdir(parents=True)
        self.actual={'state':'COMPLETE','task_id':0,'sequence_sha256':'s'*64,
                     'sequence_length':9,'sample_count':25,
                     'selected':{'seed':42,'sample':0,'score':.5,'model_sha256':'m',
                                 'backbone_sha256':'strict-b','residues':9,'backbone_atoms':36,
                                 'all_atom_sha256':'a'},
                     'output_sha256':{'x':'y'},'contact_labels_read':False,
                     'test_labels_read':False,'ood_labels_read':False,
                     'top_model':'/model.cif','top_model_sha256':'t',
                     'top_model_comparison':{'equivalent':True}}
        self.old={'state':'COMPLETE','task_id':0,'execution_plan_sha256':'p'*64,
                  'scientific_plan_sha256':'l'*64,'sequence_sha256':'s'*64,
                  'sequence_length':9,'sample_count':25,
                  'selected':{k:self.actual['selected'][k] for k in
                              ('seed','sample','score','model_sha256',
                               'residues','backbone_atoms')},
                  'output_sha256':{'x':'y'},'contact_labels_read':False,
                  'test_labels_read':False,'ood_labels_read':False}
        self.old['selected']['backbone_sha256']='legacy-b'
        af3.new_json(task/'COMPLETE.json',self.old)

    def test_strict_receipt_accepts_extra_strict_fields(self):
        with patch.object(af3,'audit_task',return_value=self.actual):
            receipt=strict.strict_task_receipt(
                self.plan,'p'*64,0,auditor_commit='c',auditor_sha256='h')
        self.assertEqual(receipt['state'],'STRICT_AUDIT_COMPLETE')
        self.assertEqual(receipt['strict_actual']['selected']['all_atom_sha256'],'a')
        self.assertEqual(receipt['digest_schema_transition']['legacy_backbone_sha256'],'legacy-b')
        self.assertEqual(receipt['digest_schema_transition']['strict_backbone_sha256'],'strict-b')
        self.assertTrue(receipt['digest_schema_transition']['selection_identity_equal'])
        self.assertFalse(receipt['model_inference_performed'])

    def test_changed_prediction_tree_rejected(self):
        changed=dict(self.actual);changed['output_sha256']={'x':'changed'}
        with patch.object(af3,'audit_task',return_value=changed),self.assertRaises(ValueError):
            strict.strict_task_receipt(
                self.plan,'p'*64,0,auditor_commit='c',auditor_sha256='h')

    def test_changed_selected_model_bytes_rejected(self):
        changed=dict(self.actual);changed['selected']=dict(self.actual['selected'],model_sha256='other')
        with patch.object(af3,'audit_task',return_value=changed),self.assertRaises(ValueError):
            strict.strict_task_receipt(
                self.plan,'p'*64,0,auditor_commit='c',auditor_sha256='h')

    def test_changed_selected_sample_rejected(self):
        changed=dict(self.actual);changed['selected']=dict(self.actual['selected'],sample=1)
        with patch.object(af3,'audit_task',return_value=changed),self.assertRaises(ValueError):
            strict.strict_task_receipt(
                self.plan,'p'*64,0,auditor_commit='c',auditor_sha256='h')

    def test_source_failure_rejected(self):
        af3.new_json(self.root/'runs/task_000/FAILURE.json',{'state':'FAILED'})
        with self.assertRaises(ValueError):
            strict.strict_task_receipt(
                self.plan,'p'*64,0,auditor_commit='c',auditor_sha256='h')

    def test_preflight_requires_exact_40(self):
        with patch.object(strict,'strict_task_receipt',return_value={'state':'x'}), \
             self.assertRaises(ValueError):
            strict.preflight(self.plan,'p'*64,auditor_commit='c',auditor_sha256='h')

    def test_finalize_creates_no_output_when_preflight_fails(self):
        output=self.root/'out'
        with patch.object(strict,'source_execution_plan',return_value=self.plan), \
             patch.object(strict,'preflight',side_effect=RuntimeError('incomplete')), \
             self.assertRaises(RuntimeError):
            strict.finalize(self.root/'PLAN.json','p'*64,output,
                            auditor_commit='c',deep_runtime=False)
        self.assertFalse(output.exists())


if __name__=='__main__':unittest.main(verbosity=2)
