import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from af3 import new_json
from assemble import complete_receipts,assemble,validation_coverage

class Assembly(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.plan={'run_root':str(self.root/'runs'),'tasks':[{'task_id':i} for i in range(40)]}
    def test_incomplete_campaign_fails(self):
        with self.assertRaises(RuntimeError):complete_receipts(self.plan,'a'*64)
    def test_no_output_on_incomplete_campaign(self):
        out=self.root/'output'
        with patch('assemble.load_plan',return_value=self.plan),self.assertRaises(RuntimeError):
            assemble(self.root/'PLAN.json','a'*64,out)
        self.assertFalse(out.exists())
    def test_failed_task_not_a_completion(self):
        new_json(self.root/'runs/task_000/FAILURE.json',{'state':'FAILED'})
        with self.assertRaises(ValueError):complete_receipts(self.plan,'a'*64)
    def test_foreign_receipt_rejected(self):
        new_json(self.root/'runs/task_000/COMPLETE.json',{'state':'WRONG'})
        with self.assertRaises(ValueError):complete_receipts(self.plan,'a'*64)
    def test_coverage_is_sequence_exact_not_fuzzy(self):
        split=self.root/'split.json';csv=self.root/'canonical.csv'
        ids=['case_%03d'%i for i in range(217)]
        new_json(split,{'splits':{'validation':ids,'train':[],'test':[]}})
        csv.write_text('INSTANCE,vh_seq,vl_seq,ag_seq\n'+''.join(i+',ACD,,EFG\n' for i in ids))
        a=hashlib.sha256(b'ACD').hexdigest();b=hashlib.sha256(b'EFG').hexdigest()
        old={'split_manifest':str(split),'canonical_manifest':str(csv)}
        rows,missing=validation_coverage(old,{a})
        self.assertEqual(missing,{b});self.assertEqual(sum(x['complete'] for x in rows),0)
        rows,missing=validation_coverage(old,{a,b})
        self.assertFalse(missing);self.assertEqual(sum(x['complete'] for x in rows),217)
    def test_no_closed_split_overlap(self):
        split=self.root/'split.json';ids=['case_%03d'%i for i in range(217)]
        new_json(split,{'splits':{'validation':ids,'train':[ids[0]],'test':[]}})
        with self.assertRaises(ValueError):validation_coverage({'split_manifest':str(split)},set())

if __name__=='__main__':unittest.main(verbosity=2)
