import unittest
from unittest.mock import patch
import tempfile
from pathlib import Path
import certify_campaign
import assemble

class ClosureGates(unittest.TestCase):
    def test_all_39_terminal_records_required(self):
        text='\n'.join(f'1641712_{i}|COMPLETED|0:0' for i in range(1,40))
        certify_campaign.require_array_success(text,'1641712',list(range(1,40)))
    def test_pending_failed_cancelled_rejected(self):
        for state in ('PENDING','RUNNING','FAILED','CANCELLED'):
            with self.subTest(state=state),self.assertRaises(RuntimeError):
                certify_campaign.require_array_success('99_1|'+state+'|0:0','99',[1])
    def test_missing_record_rejected(self):
        with self.assertRaises(RuntimeError):certify_campaign.require_array_success('99_1|COMPLETED|0:0','99',[1,2])
    def test_nonzero_exit_rejected(self):
        with self.assertRaises(RuntimeError):certify_campaign.require_array_success('99_1|COMPLETED|1:0','99',[1])
    def test_wrong_array_rejected(self):
        with self.assertRaises(RuntimeError):certify_campaign.require_array_success('98_1|COMPLETED|0:0','99',[1])
    def test_duplicate_record_rejected(self):
        with self.assertRaises(ValueError):certify_campaign.require_array_success('99_1|COMPLETED|0:0\n99_1|COMPLETED|0:0','99',[1])
    def test_assembly_calls_injected_receipt_gate_before_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'not_created'
            def loader(path,sha):return {'pinned':True}
            def reader(plan,sha):
                self.assertEqual(plan,{'pinned':True})
                raise RuntimeError('missing strict audit')
            with self.assertRaisesRegex(RuntimeError,'missing strict audit'):
                assemble.assemble('plan','sha',out,plan_loader=loader,receipt_loader=reader)
            self.assertFalse(out.exists())

if __name__=='__main__':unittest.main(verbosity=2)
