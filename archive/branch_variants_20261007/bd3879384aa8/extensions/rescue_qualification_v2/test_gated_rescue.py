import unittest
from gated_rescue import empirical_upper_quantile,matcher_chain_slices
class Gate(unittest.TestCase):
 def test_quantile_higher(self):self.assertEqual(empirical_upper_quantile(range(1,101),.95),95.0)
 def test_quantile_small(self):self.assertEqual(empirical_upper_quantile([2,1],.95),2.0)
 def test_nonfinite_fails(self):
  with self.assertRaises(ValueError):empirical_upper_quantile([1,float('nan')])
 def test_chain_slices(self):self.assertEqual([(x.start,x.stop) for x in matcher_chain_slices(['ABC','DE'])],[(0,3),(3,5)])
if __name__=='__main__':unittest.main(verbosity=2)
