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
 b=torch.zeros((4,2));e=torch.zeros((4,2));m=M([[0,2,0,0],[2,0,0,0],[0,0,3,0],[0,0,0,3]]);assert gate_score(m,b,e,['AA','AA'])==1.0
