import numpy as np
import torch
from gated_rescue import gate_score
class M:
 def __init__(self,L):self.L=torch.tensor(L,dtype=torch.float32)
 def logits(self,b,e):return self.L[:len(b),:len(e)]
def test_identity_gain_zero():
 b=torch.zeros((3,2));e=torch.zeros((3,2));assert gate_score(M([[3,0,0],[0,3,0],[0,0,3]]),b,e,['AAA'])==0
def test_nonidentity_gain_positive():
 b=torch.zeros((3,2));e=torch.zeros((3,2));assert gate_score(M([[0,3,0],[3,0,0],[0,0,3]]),b,e,['AAA'])==2.0
def test_chain_weighting():
 class C:
  def logits(self,b,e):
   return torch.tensor([[0.,2.],[2.,0.]]) if float(b[0,0])==0 else torch.tensor([[3.,0.],[0.,3.]])
 b=torch.tensor([[0.,0.],[0.,0.],[1.,0.],[1.,0.]]);e=torch.zeros((4,2));assert gate_score(C(),b,e,['AA','AA'])==1.0
