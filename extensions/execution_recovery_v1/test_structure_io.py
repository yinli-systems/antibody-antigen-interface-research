"""Synthetic structural I/O tests using real Bio.PDB readers and writers."""
import tempfile
from pathlib import Path
import unittest
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from af3 import protein_backbone
from assemble import write_backbone


def make_cif(path,omit=None,chain_name='A'):
    import numpy as np
    from Bio.PDB.Structure import Structure
    from Bio.PDB.Model import Model
    from Bio.PDB.Chain import Chain
    from Bio.PDB.Residue import Residue
    from Bio.PDB.Atom import Atom
    from Bio.PDB import MMCIFIO
    structure=Structure('synthetic'); model=Model(0);structure.add(model)
    chain=Chain(chain_name);model.add(chain)
    serial=1
    for i,resname in enumerate(['ALA','GLY','SER'],1):
        residue=Residue((' ',i,' '),resname,' ');chain.add(residue)
        for j,name in enumerate(['N','CA','C','O']):
            if (i,name)==omit:continue
            xyz=np.array([i*3.8+j*.2+0.1234,j*1.1+0.2345,i*.5+0.3456],dtype=np.float32)
            atom=Atom(name,xyz,20.0,1.0,' ',name.rjust(4),serial,element=name[0]);serial+=1
            residue.add(atom)
    writer=MMCIFIO();writer.set_structure(structure);writer.save(str(path))

class StructuralIO(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
    def test_cif_and_pdb_roundtrip(self):
        p=self.root/'input.cif';make_cif(p)
        d=protein_backbone(p,'AGS');self.assertEqual(d['residues'],3);self.assertEqual(d['backbone_atoms'],12)
        v=write_backbone(p,'AGS',self.root/'new.pdb')
        self.assertLessEqual(v['max_coordinate_rounding_A'],.0006)
        self.assertEqual(len(v['pdb_sha256']),64)
    def test_sequence_mismatch_rejected(self):
        p=self.root/'input.cif';make_cif(p)
        with self.assertRaises(ValueError):protein_backbone(p,'AAA')
    def test_missing_atom_rejected(self):
        p=self.root/'input.cif';make_cif(p,omit=(2,'O'))
        with self.assertRaises(ValueError):protein_backbone(p,'AGS')
    def test_foreign_chain_rejected(self):
        p=self.root/'input.cif';make_cif(p,chain_name='B')
        with self.assertRaises(ValueError):protein_backbone(p,'AGS')
    def test_pdb_output_never_overwritten(self):
        p=self.root/'input.cif';make_cif(p);out=self.root/'existing.pdb';out.write_text('original')
        with self.assertRaises(FileExistsError):write_backbone(p,'AGS',out)
        self.assertEqual(out.read_text(),'original')

if __name__=='__main__':unittest.main(verbosity=2)
