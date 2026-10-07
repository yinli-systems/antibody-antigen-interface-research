import json,os,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from qualification_core import digest
import qualified_evaluate as q

class RunnerGates(unittest.TestCase):
    def test_no_unscheduled_contact_inference(self):
        with patch.dict(os.environ,{},clear=True),self.assertRaisesRegex(RuntimeError,'scheduler'):
            q.run(SimpleNamespace())
    def test_reject_wrong_plan_before_model_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan=Path(tmp)/'p.json';plan.write_text('{}')
            with patch.dict(os.environ,{'SLURM_JOB_ID':'fixture-not-real'}),self.assertRaisesRegex(ValueError,'hash'):
                q.run(SimpleNamespace(plan=plan,plan_sha256='not-a-digest'))
    def test_prepare_waits_for_existing_training_without_contact_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=root/'plan.json';p.write_text(json.dumps({'input_sha256':{str(root/'learned_rescue.py'):'x'},'rescue_seeds':[101,202,303]}))
            args=SimpleNamespace(training_plan=p,training_plan_sha256='fixture',training_root=root/'absent')
            with patch.object(q,'input_preflight',return_value={}),patch.object(q,'import_original',return_value=None),self.assertRaisesRegex(RuntimeError,'WAIT_FOR_EXISTING_TRAINING'):
                q.prepare(args)
    def test_aggregation_never_drops_missing_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=root/'plan.json';p.write_text(json.dumps({'seed_pairs':[[101,1],[202,2],[303,3]]}))
            with self.assertRaises(FileNotFoundError):q.aggregate(SimpleNamespace(plan=p,plan_sha256=digest(p),input=root))

if __name__=='__main__':unittest.main(verbosity=2)
