import unittest
import numpy as np
from rescore import METHODS,transform,dense_grid,precision_top_k

class Rescoring(unittest.TestCase):
    def test_double_center_removes_additive_nuisance(self):
        rng=np.random.default_rng(42);x=rng.normal(size=(8,5));a=rng.normal(size=(8,1));b=rng.normal(size=(1,5))
        self.assertTrue(np.allclose(transform(x,'double_center'),transform(x+a+b,'double_center')))
    def test_double_center_zero_marginal_means(self):
        s=transform(np.arange(24).reshape(6,4),'double_center')
        self.assertTrue(np.allclose(s.mean(0),0));self.assertTrue(np.allclose(s.mean(1),0))
    def test_identity_exact(self):
        s=np.arange(15).reshape(3,5);self.assertTrue(np.array_equal(s,transform(s,'identity')))
    def test_small_csls_matches_brute_force(self):
        rng=np.random.default_rng(7)
        for shape in [(1,1),(5,6),(24,31)]:
            s=rng.normal(size=shape);k=min(10,shape[1]);j=min(10,shape[0])
            expected=2*s-np.sort(s,axis=1)[:,-k:].mean(1,keepdims=True)-np.sort(s,axis=0)[-j:,:].mean(0,keepdims=True)
            self.assertTrue(np.allclose(transform(s,'local_scale_10'),expected))
    def test_symmetric_methods_transpose(self):
        s=np.arange(21).reshape(3,7)
        for m in ['identity','double_center','local_scale_10']:
            self.assertTrue(np.allclose(transform(s,m).T,transform(s.T,m)))
    def test_finite_guard(self):
        for m in METHODS:
            with self.assertRaises(ValueError):transform(np.array([[np.nan]]),m)
    def test_method_guard(self):
        with self.assertRaises(ValueError):transform(np.ones((3,3)),'posthoc_best')
    def test_duplicates_guard(self):
        with self.assertRaises(ValueError):dense_grid([0,0],[0,0],[0,1],[1.,2.])
    def test_incomplete_guard(self):
        with self.assertRaises(ValueError):dense_grid([0,0,1],[0,1,1],[0,1,0],[1.,2.,3.])
    def test_dense_grid_order_invariance(self):
        ag=np.array([1,0,1,0]);ab=np.array([2,4,4,2]);y=np.array([1,0,0,1]);s=np.array([.1,.2,.3,.4])
        a,b=dense_grid(ag,ab,y,s);c,d=dense_grid(ag[::-1],ab[::-1],y[::-1],s[::-1]);self.assertTrue(np.array_equal(a,c));self.assertTrue(np.array_equal(b,d))
    def test_tied_top10_is_expected_fraction_not_sort_bias(self):
        self.assertAlmostEqual(precision_top_k([1,0,0,0],np.ones(4),1),.25)
    def test_complete_assignment_unchanged_by_additive_correction(self):
        from scipy.optimize import linear_sum_assignment
        rng=np.random.default_rng(17);s=rng.normal(size=(12,12))
        r,c=linear_sum_assignment(s,maximize=True)
        for method in METHODS:
            rr,cc=linear_sum_assignment(transform(s,method),maximize=True)
            self.assertTrue(np.array_equal(c,cc))

if __name__=='__main__':unittest.main(verbosity=2)
