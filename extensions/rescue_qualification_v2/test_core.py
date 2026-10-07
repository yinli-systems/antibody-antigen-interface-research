import copy,itertools,unittest
import numpy as np
import torch
from qualification_core import canonical_assignment, signed_row_column_loss, observed_transforms, validate_rescue_checkpoint, summarize_records

class Assignment(unittest.TestCase):
    def test_unique_max_matches_original(self):
        from scipy.optimize import linear_sum_assignment
        rng=np.random.default_rng(17)
        for n in (1,2,7,23):
            with self.subTest(n=n):
                s=rng.normal(size=(n,n)); e=rng.normal(size=(n,3))
                _,expected=linear_sum_assignment(s,maximize=True)
                self.assertTrue(np.array_equal(canonical_assignment(s,e),expected))
    def test_column_permutation_invariance_unique_and_tied(self):
        rng=np.random.default_rng(42);e=rng.normal(size=(4,3))
        for s in (rng.normal(size=(4,4)),np.zeros((4,4)),np.ones((4,4))):
            base=e[canonical_assignment(s,e)]
            for p in itertools.permutations(range(4)):
                p=np.array(p);self.assertTrue(np.array_equal(e[p][canonical_assignment(s[:,p],e[p])],base))
    def test_duplicates_do_not_create_new_vectors(self):
        e=np.array([[1.,2.],[1.,2.],[2.,1.]])
        p=canonical_assignment(np.zeros((3,3)),e)
        self.assertEqual(sorted(p.tolist()),[0,1,2])
    def test_nan_rejected(self):
        with self.assertRaises(ValueError):canonical_assignment(np.array([[np.nan]]),np.ones((1,2)))
    def test_rectangular_contact_map_rejected(self):
        with self.assertRaises(ValueError):canonical_assignment(np.ones((3,2)),np.ones((2,4)))
    def test_empty_rejected(self):
        with self.assertRaises(ValueError):canonical_assignment(np.ones((0,0)),np.ones((0,2)))
    def test_infinite_feature_rejected(self):
        with self.assertRaises(ValueError):canonical_assignment(np.ones((1,1)),np.array([[np.inf]]))

class Equivariance(unittest.TestCase):
    def test_all_permutations_same_loss_and_gradient(self):
        torch.manual_seed(42)
        x=torch.randn(4,4,dtype=torch.float64,requires_grad=True)
        original=signed_row_column_loss(x)
        grad=torch.autograd.grad(original,x)[0]
        for p in itertools.permutations(range(4)):
            perm=np.array(p);loss=signed_row_column_loss(x[:,perm],perm)
            other=torch.autograd.grad(loss,x)[0]
            self.assertAlmostEqual(loss.item(),original.item(),places=12)
            self.assertTrue(torch.allclose(grad,other,atol=1e-12,rtol=0))
    def test_invalid_permutation_rejected(self):
        with self.assertRaises(ValueError):signed_row_column_loss(torch.ones((2,2)),[0,0])
    def test_float_indices_rejected(self):
        with self.assertRaises(ValueError):signed_row_column_loss(torch.ones((2,2)),[0.,1.])

class Features(unittest.TestCase):
    def setUp(self):
        self.x=torch.arange(35,dtype=torch.float32).reshape(5,7);self.x[:,-1]=1
        self.parts=['ABC','DE'];self.order=np.array([1,2,0,4,3])
        self.calls=[]
    def rescue(self,b,e,parts):
        self.calls.append(len(e));out=e.clone();start=0
        for part in parts:out[start:start+len(part)]=e[start:start+len(part)].flip(0);start+=len(part)
        return out
    def test_base_and_presence_preserved(self):
        variants,present=observed_transforms(self.x,self.parts,self.order,self.rescue,4,2)
        self.assertTrue(present);self.assertEqual(len(self.calls),2)
        for x in variants.values():
            self.assertTrue(torch.equal(x[:,:4],self.x[:,:4]));self.assertTrue(torch.equal(x[:,-1],self.x[:,-1]))
    def test_missing_source_keeps_full_case(self):
        self.x[:,4:]=0
        variants,present=observed_transforms(self.x,self.parts,self.order,self.rescue,4,2)
        self.assertFalse(present);self.assertFalse(self.calls)
        self.assertTrue(all(torch.equal(x,self.x) for x in variants.values()))
    def test_absent_nonzero_features_rejected(self):
        self.x[:,-1]=0
        with self.assertRaises(ValueError):observed_transforms(self.x,self.parts,self.order,self.rescue,4,2)
    def test_partial_missingness_rejected(self):
        self.x[0,-1]=0
        with self.assertRaises(ValueError):observed_transforms(self.x,self.parts,self.order,self.rescue,4,2)
    def test_nonbinary_presence_rejected(self):
        self.x[:,-1]=.5
        with self.assertRaises(ValueError):observed_transforms(self.x,self.parts,self.order,self.rescue,4,2)
    def test_cross_chain_shuffle_rejected(self):
        with self.assertRaises(ValueError):observed_transforms(self.x,self.parts,[4,1,2,3,0],self.rescue,4,2)
    def test_wrong_width_rejected(self):
        with self.assertRaises(ValueError):observed_transforms(self.x,self.parts,self.order,self.rescue,3,2)
    def test_synthesized_features_rejected(self):
        with self.assertRaises(ValueError):observed_transforms(self.x,self.parts,self.order,lambda b,e,p:e+1,4,2)
    def test_cross_chain_rescue_rejected(self):
        with self.assertRaises(ValueError):observed_transforms(self.x,self.parts,self.order,lambda b,e,p:e.flip(0),4,2)
    def test_nonfinite_rescue_rejected(self):
        with self.assertRaises(ValueError):observed_transforms(self.x,self.parts,self.order,lambda b,e,p:e*float('nan'),4,2)

class Receipt(unittest.TestCase):
    def setUp(self):
        self.ck={'plan_sha256':'abc','seed':101,'epoch':1,'validation_matching_accuracy':.8}
        self.h={'selection_uses_contact_metrics':False,'test_access':False,'selected_epoch':1,
                'best_validation_matching_accuracy':.8,'history':[{'epoch':1,'validation_matching_accuracy':.8},{'epoch':2,'validation_matching_accuracy':.8}]}
    def test_correct_earliest_tie(self):validate_rescue_checkpoint(self.ck,self.h,'abc',101)
    def test_wrong_seed(self):
        with self.assertRaises(ValueError):validate_rescue_checkpoint(self.ck,self.h,'abc',202)
    def test_wrong_plan(self):
        with self.assertRaises(ValueError):validate_rescue_checkpoint(self.ck,self.h,'def',101)
    def test_wrong_epoch(self):
        self.ck['epoch']=2
        with self.assertRaises(ValueError):validate_rescue_checkpoint(self.ck,self.h,'abc',101)
    def test_contact_selected(self):
        self.h['selection_uses_contact_metrics']=True
        with self.assertRaises(ValueError):validate_rescue_checkpoint(self.ck,self.h,'abc',101)
    def test_nonfinite_history(self):
        self.h['history'][0]['validation_matching_accuracy']=float('nan')
        with self.assertRaises(ValueError):validate_rescue_checkpoint(self.ck,self.h,'abc',101)

class Reporting(unittest.TestCase):
    def setUp(self):
        self.rows=[{'instance_id':str(i),'rescue_seed':s,'downstream_seed':d,'condition':c,
           'aligned_ap':.03,'corrupted_ap':.01,'rescued_ap':.02,'clean_rescued_ap':.025}
           for i in range(2) for s,d in [(101,1),(202,2)] for c in ['same','global']]
    def report(self,r):return summarize_records(r,['0','1'],[(101,1),(202,2)],['same','global'])
    def test_no_silent_clean_damage(self):
        r=self.report(self.rows)[0];self.assertAlmostEqual(r['clean_rescued_minus_aligned_ap'],-.005)
    def test_gap_fraction_unclipped(self):
        for row in self.rows:row['rescued_ap']=.05
        self.assertAlmostEqual(self.report(self.rows)[0]['gap_recovered_ratio'],2)
    def test_nonpositive_gap_null(self):
        for row in self.rows:row['aligned_ap']=.001
        self.assertIsNone(self.report(self.rows)[0]['gap_recovered_ratio'])
    def test_missing_case_fails(self):
        with self.assertRaises(ValueError):self.report(self.rows[1:])
    def test_duplicate_case_fails(self):
        with self.assertRaises(ValueError):self.report(self.rows+[self.rows[0]])
    def test_nonfinite_ap_fails(self):
        self.rows[0]['rescued_ap']=float('nan')
        with self.assertRaises(ValueError):self.report(self.rows)

if __name__=='__main__':unittest.main(verbosity=2)
