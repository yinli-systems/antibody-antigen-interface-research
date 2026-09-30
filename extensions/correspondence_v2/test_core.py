"""CPU contract tests. Run directly; do not hide historical pytest failures."""
import itertools
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from core import *

class PermutationTests(unittest.TestCase):
    def test_chain_boundaries_multiset_and_reproducibility(self):
        for parts in [("A",),("ACDEF","GGH"),("AAAAAA","AC")]:
            for mode in MODES:
                for k in (1,4,16):
                    with self.subTest(parts=parts,mode=mode,k=k):
                        p,d=chain_permutation(parts,mode,k,17,"key")
                        q,_=chain_permutation(parts,mode,k,17,"key")
                        self.assertTrue(np.array_equal(p,q))
                        self.assertEqual(sorted(p.tolist()),list(range(sum(map(len,parts)))))
                        self.assertEqual(d["chain_crossings"],0)
    def test_same_amino_acid(self):
        parts=("ACACACDDD","GGAAGG")
        p,_=chain_permutation(parts,"same_aa_shuffle",0,17,"x")
        seq=np.array(list("".join(parts)))
        self.assertTrue(np.array_equal(seq[p],seq))
    def test_shift_sign_and_wrap(self):
        p,d=chain_permutation(("ABCDE","XYZ"),"chain_shift",1,0,"k")
        self.assertEqual(p.tolist(),[1,2,3,4,0,6,7,5])
        self.assertEqual(d["wraparound_rows"],2)
        q,_=chain_permutation(("ABCDE","XYZ"),"chain_shift",-1,0,"k")
        self.assertTrue(np.array_equal(p[q],np.arange(8)))
    def test_local_shuffle_bound(self):
        p,_=chain_permutation(("ABCDEFGHIJKLMNOPQ",),"local_shuffle",4,10,"k")
        self.assertTrue(np.all(np.abs(p-np.arange(len(p)))<=3))
    def test_oracle_identity(self):
        p,d=chain_permutation(("ABCDE","XYZ"),"oracle_restore",8,0,"k")
        self.assertTrue(np.array_equal(p,np.arange(8)))
        self.assertTrue(d["oracle_composition_is_identity"])
    def test_invalid_mode(self):
        with self.assertRaises(ValueError): chain_permutation(("ABC",),"surprise",1,0,"k")
    def test_empty_chain(self):
        with self.assertRaises(ValueError): chain_permutation(("",),"identity",0,0,"k")
    def test_invalid_local(self):
        with self.assertRaises(ValueError): chain_permutation(("ABC",),"local_shuffle",0,0,"k")
    def test_integer_severity(self):
        with self.assertRaises(ValueError): chain_permutation(("ABC",),"chain_shift",1.2,0,"k")

class TensorTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch=torch
        self.x=torch.arange(35,dtype=torch.float32).reshape(5,7)
        self.x[:,-1]=1
    def test_preserve_base_and_presence(self):
        p=np.array([1,2,3,4,0]); y=transform_extra_columns(self.x,4,2,p)
        self.assertTrue(self.torch.equal(y[:,:4],self.x[:,:4]))
        self.assertTrue(self.torch.equal(y[:,-1],self.x[:,-1]))
        self.assertTrue(self.torch.equal(y[:,4:6],self.x[p,4:6]))
    def test_explicit_inverse(self):
        p=np.array([1,2,3,4,0]); y=transform_extra_columns(self.x,4,2,p)
        z=transform_extra_columns(y,4,2,np.argsort(p))
        self.assertTrue(self.torch.equal(z,self.x))
    def test_shape_guard(self):
        with self.assertRaises(ValueError): transform_extra_columns(self.x,3,2,np.arange(5))
    def test_permutation_guard(self):
        with self.assertRaises(ValueError): transform_extra_columns(self.x,4,2,np.zeros(5,dtype=int))
    def test_finite_guard(self):
        self.x[0,0]=float('nan')
        with self.assertRaises(ValueError): transform_extra_columns(self.x,4,2,np.arange(5))
    def test_missingness_guard(self):
        self.x[0,-1]=0
        with self.assertRaises(ValueError): transform_extra_columns(self.x,4,2,np.arange(5))
    def test_missing_source_unchanged(self):
        self.x[:,4:]=0
        y=transform_extra_columns(self.x,4,2,np.array([1,2,3,4,0]))
        self.assertTrue(self.torch.equal(y,self.x))

class RankingTests(unittest.TestCase):
    def test_unique_winner(self):
        d=ranking_summary([3,2,1],[True,False,False]); self.assertEqual(d['mrr'],1)
    def test_loser(self):
        self.assertEqual(ranking_summary([1,2,3],[True,False,False])['mrr'],1/3)
    def test_all_tied(self):
        d=ranking_summary([1,1,1,1],[True,False,False,False])
        self.assertAlmostEqual(d['mrr'],sum(1/k for k in range(1,5))/4)
        self.assertEqual(d['recall_at_1'],.25)
    def test_multiple_positive_ties_exact_enumeration(self):
        positions=list(itertools.combinations(range(1,5),2))
        expected=np.mean([1/min(p) for p in positions])
        self.assertAlmostEqual(ranking_summary([1]*4,[True,True,False,False])['mrr'],expected)
    def test_invalid_labels(self):
        for labels in ([False,False],[True,True]):
            with self.assertRaises(ValueError): ranking_summary([1,2],labels)
    def test_nonfinite(self):
        with self.assertRaises(ValueError): ranking_summary([1,np.nan],[True,False])

class StatisticsTests(unittest.TestCase):
    def test_constant_effect(self):
        d=crossed_cluster_bootstrap(np.ones((3,6))*.2,['a','a','a','b','c','c'],repeats=200)
        self.assertTrue(np.allclose(d['crossed_seed_cluster_ci95'],[.2,.2]))
    def test_cluster_weight_not_row_weight(self):
        d=crossed_cluster_bootstrap(np.array([[0,0,0,1],[0,0,0,1]]),['a','a','a','b'],repeats=200)
        self.assertEqual(d['mean_difference'],.5)
        self.assertEqual(d['complex_weighted_mean_difference'],.25)
    def test_pairing_sign_reversal(self):
        x=np.array([[1,2,3],[2,1,2],[0,2,1]])/100
        a=crossed_cluster_bootstrap(x,['a','b','c'],repeats=200)
        b=crossed_cluster_bootstrap(-x,['a','b','c'],repeats=200)
        self.assertTrue(np.allclose(a['crossed_seed_cluster_ci95'],-np.array(b['crossed_seed_cluster_ci95'])[::-1]))
    def test_complete_case_guard(self):
        with self.assertRaises(ValueError): crossed_cluster_bootstrap(np.array([[1,np.nan]]),['a','b'])
    def test_single_cluster_guard(self):
        with self.assertRaises(ValueError): crossed_cluster_bootstrap(np.ones((3,2)),['a','a'])

class FirewallTests(unittest.TestCase):
    def setUp(self): self.s={'train':['a'],'validation':['b','c'],'test':['d']}
    def test_accept_validation(self): guard_validation_ids(['b','c'],self.s)
    def test_reject_test(self):
        with self.assertRaises(ValueError): guard_validation_ids(['d'],self.s)
    def test_reject_train(self):
        with self.assertRaises(ValueError): guard_validation_ids(['a'],self.s)
    def test_reject_overlap(self):
        self.s['test'].append('b')
        with self.assertRaises(ValueError): guard_validation_ids(['b'],self.s)
    def test_reject_duplicate(self):
        with self.assertRaises(ValueError): guard_validation_ids(['b','b'],self.s)
    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'receipt.json'; write_new_json(p,{'ok':True})
            with self.assertRaises(FileExistsError): write_new_json(p,{'ok':False})
            self.assertTrue(json.loads(p.read_text())['ok'])
    def test_no_nan_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError): write_new_json(Path(tmp)/'x',{'value':np.nan})

if __name__=='__main__': unittest.main(verbosity=2)
