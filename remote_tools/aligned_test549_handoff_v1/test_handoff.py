import tempfile
import unittest
from pathlib import Path

from common import audit_splits, local_path, require, sha, split_rows, verify_bundle, write_json, verify_development_receipt


class Guards(unittest.TestCase):
    def test_actual_development_receipt_v2(self):
        receipt = {'format_version': 'framecontact_same_protocol_adapter_development_v2',
                   'test_loaded': False, 'external_test_loaded': False,
                   'selected_checkpoint_sha256': 'a' * 64, 'result_sha256': 'b' * 64, 'seed': 17}
        verify_development_receipt(receipt, 'a' * 64, 'b' * 64, 17)
        with self.assertRaisesRegex(ValueError, 'digest'):
            verify_development_receipt(receipt, 'c' * 64, 'b' * 64, 17)
        with self.assertRaisesRegex(ValueError, 'seed'):
            verify_development_receipt(receipt, 'a' * 64, 'b' * 64, 43)

    def test_incorrect_receipt_field_not_silently_accepted(self):
        receipt = {'format_version': 'framecontact_same_protocol_adapter_development_v2',
                   'test_loaded': False, 'external_test_loaded': False,
                   'checkpoint_sha256': 'a' * 64, 'result_sha256': 'b' * 64, 'seed': 17}
        with self.assertRaisesRegex(ValueError, 'digest'):
            verify_development_receipt(receipt, 'a' * 64, 'b' * 64, 17)

    def test_overlap_rejected(self):
        with self.assertRaisesRegex(ValueError, 'overlaps'):
            audit_splits({'a': {'role': 'train'}}, {'a': {'role': 'test'}}, ['a'])

    def test_legacy_membership_not_subset(self):
        result = audit_splits({'a': {'role': 'train'}, 'b': {'role': 'validation'}},
                              {'c': {'role': 'test'}, 'd': {'role': 'test'}}, ['a', 'b', 'c'])
        self.assertEqual(result['old455_role_counts'], {'train': 1, 'validation': 1, 'test': 1})
        self.assertEqual(result['test_not_in_old455_count'], 1)
        self.assertFalse(result['fresh_blind_test'])

    def test_unresolved_and_duplicate_ids_rejected(self):
        for ids in (['missing'], ['a', 'a']):
            with self.assertRaises(ValueError):
                audit_splits({'a': {'role': 'train'}}, {}, ids)

    def test_non_test_row_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Non-test'):
            audit_splits({}, {'a': {'role': 'train'}}, ['a'])

    def test_path_escape_and_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            for rel in ('../escape', '/absolute'):
                with self.assertRaises(ValueError):
                    local_path(root, rel)
            (root / 'link').symlink_to('/tmp')
            with self.assertRaises(ValueError):
                local_path(root, 'link/artifact')

    def test_duplicate_split_rows_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'split.csv'
            p.write_text('instance_id,role\na,test\na,test\n')
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                split_rows(p)

    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'receipt.json'
            write_json(p, {'ok': True})
            before = sha(p)
            with self.assertRaises(FileExistsError):
                write_json(p, {'ok': False})
            self.assertEqual(sha(p), before)

    def test_substituted_manifest_or_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            checkpoint = root / 'checkpoint.pt'
            checkpoint.write_bytes(b'original')
            write_json(root / 'BUNDLE.json', {'format_version': 'aligned_test549_handoff_v1',
                                             'files': {'checkpoint.pt': sha(checkpoint)}})
            digest = sha(root / 'BUNDLE.json')
            with self.assertRaisesRegex(ValueError, 'authority SHA'):
                verify_bundle(root, '0' * 64)
            checkpoint.write_bytes(b'substituted')
            with self.assertRaisesRegex(ValueError, 'Artifact SHA'):
                verify_bundle(root, digest)


if __name__ == '__main__':
    unittest.main()
