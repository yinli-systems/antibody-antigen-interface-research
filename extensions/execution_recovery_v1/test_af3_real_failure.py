"""Regression against the preserved 1641607_0 AF3 audit failure.

The test is portable: it skips unless AF3_REAL_FAILURE_ATTEMPT points at the
preserved attempt. On ParaCloud it exercises all 25 real samples without model
inference or any contact/test/OOD label access.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parent))
import af3


class PreservedTask000Regression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw=os.environ.get('AF3_REAL_FAILURE_ATTEMPT')
        if not raw:
            raise unittest.SkipTest('AF3_REAL_FAILURE_ATTEMPT is not set')
        cls.attempt=Path(raw).resolve()
        if not cls.attempt.is_dir():
            raise unittest.SkipTest('preserved attempt directory is unavailable')
        cls.plan=af3.read_json(cls.attempt/'PLAN.json')

    def test_timestamp_only_reserialization_is_admitted_without_inference(self):
        failure=af3.read_json(self.attempt/'runs/task_000/FAILURE.json')
        self.assertEqual(failure['error'],
                         'top-level model is not a maximum-score sample byte for byte')
        result=af3.audit_task(self.plan,0)
        self.assertEqual(result['state'],'COMPLETE')
        self.assertEqual(result['sample_count'],25)
        self.assertEqual((result['selected']['seed'],result['selected']['sample']),(42,2))
        self.assertAlmostEqual(result['selected']['score'],0.6402947117306571,places=15)
        self.assertFalse(result['top_model_byte_identical_to_selected_sample'])
        comparison=result['top_model_comparison']
        self.assertTrue(comparison['equivalent'])
        self.assertEqual(comparison['comparison_mode'],
                         'STRICT_MMCIF_SEMANTIC_EQUIVALENCE')
        self.assertEqual(comparison['raw_difference_keys'],
                         ['_ma_model_list.model_group_name'])
        self.assertEqual(comparison['normalized_difference_keys'],[])
        self.assertEqual(comparison['top_semantic_sha256'],
                         comparison['sample_semantic_sha256'])
        self.assertTrue(
            result['top_confidence_payload_semantically_identical_to_selected_sample'])
        self.assertTrue(all(sample['residues']==78 for sample in result['samples']))
        self.assertTrue(all(sample['backbone_atoms']==312 for sample in result['samples']))
        self.assertTrue(all(sample['atom_count']==597 for sample in result['samples']))
        self.assertFalse(result['contact_labels_read'])
        self.assertFalse(result['test_labels_read'])
        self.assertFalse(result['ood_labels_read'])


if __name__=='__main__':
    unittest.main(verbosity=2)
