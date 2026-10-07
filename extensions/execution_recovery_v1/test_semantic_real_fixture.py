"""Regression tests using the exact two CIFs from failed audit 1641607_0."""
import copy
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import af3

FIXTURE=Path(__file__).parent/'fixtures/real_af3_timestamp_fixture'


class RealTimestampRegression(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        manifest=json.loads((FIXTURE/'MANIFEST.json').read_text())
        self.seq=manifest['sequence']
        for row in manifest['files']:
            stored=(FIXTURE/row['file']).read_bytes()
            self.assertEqual(hashlib.sha256(stored).hexdigest(),row['stored_sha256'])
            raw=gzip.decompress(stored) if row['file'].endswith('.gz') else stored
            self.assertEqual(hashlib.sha256(raw).hexdigest(),row['original_sha256'])
            (self.root/row['file'].removesuffix('.gz')).write_bytes(raw)
        self.top=self.root/'top.cif';self.sample=self.root/'sample.cif'
        self.structure=af3.protein_backbone(self.sample,self.seq)

    def comparison(self):
        return af3.mmcif_semantic_comparison(self.top,self.sample)

    def test_exact_real_failure_is_timestamp_only(self):
        self.assertNotEqual(af3.digest(self.top),af3.digest(self.sample))
        result=self.comparison()
        self.assertTrue(result['equivalent'])
        self.assertEqual(result['raw_difference_keys'],['_ma_model_list.model_group_name'])
        self.assertEqual(result['normalized_difference_keys'],[])
        self.assertEqual(result['top_semantic_sha256'],result['sample_semantic_sha256'])
        top=af3.protein_backbone(self.top,self.seq)
        self.assertEqual(top,self.structure)
        self.assertEqual(top['residues'],78)
        self.assertEqual(top['atom_count'],597)

    def test_every_coordinate_column_is_exact(self):
        original=af3._mmcif_dict(self.top,False)
        for field in ('_atom_site.Cartn_x','_atom_site.Cartn_y','_atom_site.Cartn_z'):
            for idx in (0,4,596):
                with self.subTest(field=field,idx=idx):
                    altered=copy.deepcopy(original)
                    altered[field][idx]=str(float(altered[field][idx])+0.001)
                    with patch.object(af3,'_mmcif_dict',side_effect=[original,altered,original,altered]):
                        self.assertFalse(self.comparison()['equivalent'])

    def test_real_sidechain_coordinate_change_rejected(self):
        text=self.top.read_text(); lines=text.splitlines(True)
        for i,line in enumerate(lines):
            parts=line.split()
            if parts[:1]==['ATOM'] and len(parts)>14 and parts[3]=='CB':
                parts[10]=str(float(parts[10])+0.001)
                lines[i]=' '.join(parts)+'\n';break
        else:self.fail('no real sidechain atom in fixture')
        self.top.write_text(''.join(lines))
        self.assertFalse(self.comparison()['equivalent'])
        self.assertNotEqual(af3.protein_backbone(self.top,self.seq)['all_atom_sha256'],self.structure['all_atom_sha256'])
        self.assertEqual(af3.protein_backbone(self.top,self.seq)['backbone_sha256'],self.structure['backbone_sha256'])

    def test_real_backbone_mutation_rejected(self):
        self.top.write_text(self.top.read_text().replace('-5.489','-5.488',1))
        self.assertFalse(self.comparison()['equivalent'])

    def test_model_version_is_not_normalized(self):
        self.top.write_text(self.top.read_text().replace('3.0.1 @','3.0.2 @',1))
        self.assertFalse(self.comparison()['equivalent'])

    def test_other_metadata_is_not_dropped(self):
        self.top.write_text(self.top.read_text().replace('Top ranked model','Wrong selected model',1))
        self.assertFalse(self.comparison()['equivalent'])

    def test_invalid_timestamp_is_rejected(self):
        self.top.write_text(self.top.read_text().replace('16:40:11','99:40:11',1))
        with self.assertRaises(ValueError):self.comparison()

    def test_duplicate_cif_item_is_rejected(self):
        self.top.write_text(self.top.read_text()+'\n_entry.id duplicate\n')
        with self.assertRaises(ValueError):self.comparison()

    def test_actual_confidence_payload_validates(self):
        record=af3.confidence_payload(self.root,0.6402947117306571,self.structure)
        self.assertEqual(len(record['confidence']['atom_plddts']),597)
        self.assertEqual(len(record['confidence']['pae']),78)

    def test_actual_csv_selects_seed_42_sample_2(self):
        rows=af3.sample_grid(self.root/'ranking_scores.csv',[42,100,18,37,79],5)
        top=max(rows,key=lambda row:row[2])
        self.assertEqual(top[:2],(42,2))
        self.assertEqual(top[2],0.6402947117306571)

    def corrupt_confidence(self,change):
        p=self.root/'confidences.json';d=json.loads(p.read_text());change(d);p.write_text(json.dumps(d))
        with self.assertRaises(ValueError):af3.confidence_payload(self.root,0.6402947117306571,self.structure)

    def test_confidence_plddt_must_match_actual_atom_bfactors(self):
        self.corrupt_confidence(lambda d:d['atom_plddts'].__setitem__(0,28.78))

    def test_pae_dimension_corruption_rejected(self):
        self.corrupt_confidence(lambda d:d['pae'][0].pop())

    def test_token_residue_fraction_is_not_truncated(self):
        self.corrupt_confidence(lambda d:d['token_res_ids'].__setitem__(0,1.1))

    def test_token_chain_corruption_rejected(self):
        self.corrupt_confidence(lambda d:d['token_chain_ids'].__setitem__(0,'B'))

    def test_negative_contact_probability_rejected(self):
        self.corrupt_confidence(lambda d:d['contact_probs'][0].__setitem__(0,-0.01))

    def test_nonfinite_confidence_rejected(self):
        self.corrupt_confidence(lambda d:d['pae'][0].__setitem__(0,float('inf')))

    def test_missing_sidechain_atom_confidence_rejected(self):
        self.corrupt_confidence(lambda d:d['atom_plddts'].pop())

    def test_wrong_sequence_rejected(self):
        with self.assertRaises(ValueError):af3.protein_backbone(self.sample,'A'+self.seq[1:])


if __name__=='__main__':unittest.main(verbosity=2)
