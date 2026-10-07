"""Shared model stack for strict partner retrieval.

No model in this file knows split labels, test data, affinity, or biological
negative labels. R1/R2 instantiate the same PairAwareReranker class; auxiliary
contact loss weight lives in the trainer configuration only.
"""
from __future__ import annotations
from dataclasses import dataclass
import math

def _torch():
    import torch
    return torch

class ResidueProjector:
    """Thin wrapper built lazily so importing metadata does not require torch."""
    @staticmethod
    def build(in_dim:int,out_dim:int):
        torch=_torch()
        return torch.nn.Sequential(torch.nn.Linear(in_dim,out_dim),torch.nn.LayerNorm(out_dim))

def masked_mean(x,mask):
    torch=_torch()
    if x.ndim!=3 or mask.ndim!=2 or x.shape[:2]!=mask.shape:raise ValueError("masked mean shape")
    w=mask.to(x.dtype).unsqueeze(-1);den=w.sum(1).clamp_min(1.0)
    return (x*w).sum(1)/den

class GlobalDualEncoder:
    def __new__(cls, antibody_dim:int, antigen_dim:int, width:int=128, temperature:float=.07):
        torch=_torch()
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.antibody_projection=ResidueProjector.build(antibody_dim,width)
                self.antigen_projection=ResidueProjector.build(antigen_dim,width)
                self.log_temperature=torch.nn.Parameter(torch.tensor(math.log(float(temperature))))
            def residue_embeddings(self,ab,ag):
                return self.antibody_projection(ab),self.antigen_projection(ag)
            def encode(self,ab,ag,ab_mask,ag_mask):
                a,g=self.residue_embeddings(ab,ag)
                a=torch.nn.functional.normalize(masked_mean(a,ab_mask),dim=-1)
                g=torch.nn.functional.normalize(masked_mean(g,ag_mask),dim=-1)
                return a,g
            def forward(self,ab,ag,ab_mask,ag_mask):
                a,g=self.encode(ab,ag,ab_mask,ag_mask)
                temp=self.log_temperature.exp().clamp(1e-3,1.0)
                return a@g.T/temp
        return M()

def _normalized_logmeanexp(values,mask,temperature:float=.1):
    torch=_torch()
    # values B x slots x residues; mask B x residues
    valid=mask[:,None,:]
    neg=torch.finfo(values.dtype).min
    z=(values/temperature).masked_fill(~valid,neg)
    count=valid.sum(-1).clamp_min(1)
    return temperature*(torch.logsumexp(z,dim=-1)-count.log())

class CDRLateInteraction:
    def __new__(cls, antibody_dim:int, antigen_dim:int, width:int=128, pool_temperature:float=.1,
                score_temperature:float=.07):
        torch=_torch()
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.antibody_projection=ResidueProjector.build(antibody_dim,width)
                self.antigen_projection=ResidueProjector.build(antigen_dim,width)
                self.log_temperature=torch.nn.Parameter(torch.tensor(math.log(float(score_temperature))))
            def encode_residues(self,ab,ag):
                a=torch.nn.functional.normalize(self.antibody_projection(ab),dim=-1)
                g=torch.nn.functional.normalize(self.antigen_projection(ag),dim=-1)
                return a,g
            def encode_antibodies(self,ab,ab_mask,cdr_masks):
                if cdr_masks.ndim!=3 or cdr_masks.shape[0]!=ab.shape[0] or cdr_masks.shape[2]!=ab.shape[1]:
                    raise ValueError("CDR mask shape")
                a=torch.nn.functional.normalize(self.antibody_projection(ab),dim=-1)
                cm=cdr_masks & ab_mask[:,None,:]
                denom=cm.sum(-1,keepdim=True).clamp_min(1)
                slots=(a[:,None,:,:]*cm[:,:,:,None]).sum(2)/denom
                slots=torch.nn.functional.normalize(slots,dim=-1)
                return slots,cm.any(-1)
            def encode_antigens(self,ag):
                return torch.nn.functional.normalize(self.antigen_projection(ag),dim=-1)
            def score_encoded(self,slots,present,g,ag_mask):
                # A x slots x G x residues. All antigen residues remain eligible.
                sim=torch.einsum("asd,gtd->asgt",slots,g)
                valid=ag_mask[None,None,:,:]
                neg=torch.finfo(sim.dtype).min
                z=(sim/pool_temperature).masked_fill(~valid,neg)
                count=valid.sum(-1).clamp_min(1)
                local=pool_temperature*(torch.logsumexp(z,dim=-1)-count.log())
                local=local.masked_fill(~present[:,:,None],0.)
                score=local.sum(1)/present.sum(1).clamp_min(1)[:,None]
                return score/self.log_temperature.exp().clamp(1e-3,1.0)
            def score_matrix(self,ab,ag,ab_mask,ag_mask,cdr_masks):
                slots,present=self.encode_antibodies(ab,ab_mask,cdr_masks)
                g=self.encode_antigens(ag)
                return self.score_encoded(slots,present,g,ag_mask)
            def forward(self,ab,ag,ab_mask,ag_mask,cdr_masks):
                return self.score_matrix(ab,ag,ab_mask,ag_mask,cdr_masks)
        return M()

class BiCrossBlock:
    def __new__(cls,width:int=256,heads:int=8,ffn:int=512,dropout:float=.1):
        torch=_torch()
        class B(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a_ln1=torch.nn.LayerNorm(width);self.g_ln1=torch.nn.LayerNorm(width)
                self.a_to_g=torch.nn.MultiheadAttention(width,heads,dropout=dropout,batch_first=True)
                self.g_to_a=torch.nn.MultiheadAttention(width,heads,dropout=dropout,batch_first=True)
                self.a_ln2=torch.nn.LayerNorm(width);self.g_ln2=torch.nn.LayerNorm(width)
                self.a_ff=torch.nn.Sequential(torch.nn.Linear(width,ffn),torch.nn.GELU(),torch.nn.Dropout(dropout),torch.nn.Linear(ffn,width),torch.nn.Dropout(dropout))
                self.g_ff=torch.nn.Sequential(torch.nn.Linear(width,ffn),torch.nn.GELU(),torch.nn.Dropout(dropout),torch.nn.Linear(ffn,width),torch.nn.Dropout(dropout))
            def forward(self,a,g,am,gm):
                qa=self.a_ln1(a);qg=self.g_ln1(g)
                da=self.a_to_g(qa,qg,qg,key_padding_mask=~gm,need_weights=False)[0]
                dg=self.g_to_a(qg,qa,qa,key_padding_mask=~am,need_weights=False)[0]
                a=a+da;g=g+dg
                a=a+self.a_ff(self.a_ln2(a));g=g+self.g_ff(self.g_ln2(g))
                return a,g
        return B()

class PairAwareReranker:
    def __new__(cls, antibody_dim:int, antigen_dim:int, width:int=256,heads:int=8,
                ffn:int=512,layers:int=2,dropout:float=.1,contact_dim:int=64):
        torch=_torch()
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.antibody_projection=ResidueProjector.build(antibody_dim,width)
                self.antigen_projection=ResidueProjector.build(antigen_dim,width)
                self.blocks=torch.nn.ModuleList([BiCrossBlock(width,heads,ffn,dropout) for _ in range(layers)])
                self.pair_head=torch.nn.Sequential(torch.nn.Linear(width*4,width),torch.nn.GELU(),torch.nn.Dropout(dropout),torch.nn.Linear(width,1))
                # present in BOTH R1 and R2; only trainer loss weight differs.
                self.contact_antibody=torch.nn.Linear(width,contact_dim)
                self.contact_antigen=torch.nn.Linear(width,contact_dim)
                self.contact_bias=torch.nn.Parameter(torch.zeros(()))
                self.contact_dim=contact_dim
            def encode_pair(self,ab,ag,ab_mask,ag_mask):
                a=self.antibody_projection(ab);g=self.antigen_projection(ag)
                for block in self.blocks:a,g=block(a,g,ab_mask,ag_mask)
                ma=masked_mean(a,ab_mask);mg=masked_mean(g,ag_mask)
                z=torch.cat([ma,mg,ma*mg,(ma-mg).abs()],-1)
                residual=self.pair_head(z).squeeze(-1)
                ca=self.contact_antibody(a);cg=self.contact_antigen(g)
                contact=torch.einsum("bid,bjd->bij",ca,cg)/math.sqrt(self.contact_dim)+self.contact_bias
                return residual,contact,a,g
            def forward(self,ab,ag,ab_mask,ag_mask,coarse_score):
                residual,contact,a,g=self.encode_pair(ab,ag,ab_mask,ag_mask)
                score=coarse_score+0.5*torch.tanh(residual)
                return {"retrieval_score":score,"contact_logits":contact,"antibody_residues":a,"antigen_residues":g}
        return M()

def parameter_signature(model)->list[tuple[str,tuple[int,...],str]]:
    return [(name,tuple(int(x) for x in p.shape),str(p.dtype)) for name,p in model.named_parameters()]

def parameter_count(model)->int:
    return sum(int(p.numel()) for p in model.parameters())

def assert_r1_r2_identical_schema(r1,r2)->dict:
    a=parameter_signature(r1);b=parameter_signature(r2)
    if a!=b:raise ValueError("R1/R2 parameter name/shape/dtype mismatch")
    ca=parameter_count(r1);cb=parameter_count(r2)
    if ca!=cb:raise ValueError("R1/R2 parameter count mismatch")
    return {"parameter_count":ca,"parameter_signature":a,"identical":True}

@dataclass(frozen=True)
class LossConfig:
    contact_weight:float
    retrieval_temperature:float=.07

def multi_positive_loss(score_matrix,positive_mask):
    torch=_torch()
    if score_matrix.shape!=positive_mask.shape:raise ValueError("loss shape")
    valid=positive_mask.any(1)
    if not bool(valid.all()):raise ValueError("zero-positive training query")
    denom=torch.logsumexp(score_matrix,dim=1)
    num=torch.logsumexp(score_matrix.masked_fill(~positive_mask,torch.finfo(score_matrix.dtype).min),dim=1)
    return -(num-denom).mean()

def balanced_contact_bce(logits,labels,mask):
    torch=_torch()
    if logits.shape!=labels.shape or logits.shape!=mask.shape:raise ValueError("contact shape")
    pos=mask & labels.bool();neg=mask & ~labels.bool()
    if not bool(pos.any()) or not bool(neg.any()):return logits.sum()*0.
    lp=torch.nn.functional.binary_cross_entropy_with_logits(logits[pos],torch.ones_like(logits[pos]))
    ln=torch.nn.functional.binary_cross_entropy_with_logits(logits[neg],torch.zeros_like(logits[neg]))
    return .5*(lp+ln)
