import tempfile, unittest
from pathlib import Path
import numpy as np
from reparank_structure_resume import inspect_npz

class FakeContact:
    antigen_indices=np.array([4,8],dtype=np.int64)
    antibody_indices=np.array([2,7,9],dtype=np.int64)
    def dense(self):
        labels=np.array([[0,1,0],[1,0,0]],dtype=bool)
        mask=np.ones_like(labels,dtype=bool)
        return labels,mask

def save(path,**overrides):
    c=FakeContact(); labels,mask=c.dense(); ai,bi=np.nonzero(mask)
    d=dict(instance_id=np.asarray('x'),antigen_residue_index=c.antigen_indices[ai],antibody_residue_index=c.antibody_indices[bi],label=labels[mask].astype(np.uint8),score=np.arange(6,dtype=np.float32),score_definition=np.asarray('negative_euclidean_distance'))
    d.update(overrides);np.savez_compressed(path,**d)

class ResumeNPZ(unittest.TestCase):
    def setUp(self):
        self.t=tempfile.TemporaryDirectory();self.p=Path(self.t.name)/'x.npz'
    def tearDown(self):self.t.cleanup()
    def test_valid(self):
        save(self.p);r=inspect_npz(self.p,'x',FakeContact());self.assertEqual(r['pair_count'],6);self.assertEqual(r['positive_count'],2)
    def test_instance_fails(self):
        save(self.p,instance_id=np.asarray('y'))
        with self.assertRaises(ValueError):inspect_npz(self.p,'x',FakeContact())
    def test_axis_fails(self):
        save(self.p,antigen_residue_index=np.arange(6,dtype=np.int64))
        with self.assertRaises(ValueError):inspect_npz(self.p,'x',FakeContact())
    def test_label_fails(self):
        save(self.p,label=np.zeros(6,dtype=np.uint8))
        with self.assertRaises(ValueError):inspect_npz(self.p,'x',FakeContact())
    def test_nonfinite_fails(self):
        s=np.arange(6,dtype=np.float32);s[2]=np.nan;save(self.p,score=s)
        with self.assertRaises(ValueError):inspect_npz(self.p,'x',FakeContact())
    def test_score_dtype_fails(self):
        save(self.p,score=np.arange(6,dtype=np.float64))
        with self.assertRaises(ValueError):inspect_npz(self.p,'x',FakeContact())
    def test_definition_fails(self):
        save(self.p,score_definition=np.asarray('other'))
        with self.assertRaises(ValueError):inspect_npz(self.p,'x',FakeContact())
    def test_foreign_field_fails(self):
        save(self.p,extra=np.asarray(1))
        with self.assertRaises(ValueError):inspect_npz(self.p,'x',FakeContact())
if __name__=='__main__':unittest.main(verbosity=2)
