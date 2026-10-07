"""Protocol-locked strict partner-retrieval model additions.

This module does not modify historical frontier_model.py.  It adds:
- L1: sequence-derived CDR late interaction with no interface/contact labels.
- R1/R2: one parameter-identical reranker class; only auxiliary supervision differs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError:  # importable for static/audit environments
    torch = None
    nn = None
    F = None

from interaction_first.frontier_model import (
    ProtocolD0Retriever,
    ProtocolReranker,
    balanced_contact_bce,
)

_ModuleBase = nn.Module if nn is not None else object


def require_torch() -> None:
    if torch is None or nn is None or F is None:
        raise RuntimeError("PyTorch is required for strict partner retrieval")


def _finite_residue_batch(features: Any, mask: Any, dim: int, name: str) -> None:
    if not torch.is_tensor(features) or features.ndim != 3 or int(features.shape[-1]) != dim:
        raise ValueError(f"{name} must have shape [batch,residue,{dim}]")
    if not torch.is_floating_point(features) or not torch.isfinite(features[mask]).all():
        raise ValueError(f"{name} valid residues must be finite floating point")
    if not torch.is_tensor(mask) or mask.dtype != torch.bool or mask.shape != features.shape[:2]:
        raise ValueError(f"{name} mask mismatch")
    if bool((mask.sum(1) == 0).any()):
        raise ValueError(f"{name} cannot contain empty examples")


def symmetric_multi_positive_infonce(
    scores: Any,
    positive_mask: Any,
    temperature: float = 0.07,
) -> Any:
    """Symmetric multi-positive InfoNCE without converting unlabeled pairs to negatives.

    `positive_mask` identifies observed positives. All other *batch candidates* enter the
    denominator as retrieval candidates, not experimentally verified non-binders.
    """
    require_torch()
    if not torch.is_tensor(scores) or scores.ndim != 2 or not torch.isfinite(scores).all():
        raise ValueError("scores must be a finite 2D tensor")
    if not torch.is_tensor(positive_mask) or positive_mask.dtype != torch.bool:
        raise ValueError("positive_mask must be boolean")
    if positive_mask.shape != scores.shape:
        raise ValueError("positive mask shape mismatch")
    if not bool(positive_mask.any(1).all()) or not bool(positive_mask.any(0).all()):
        raise ValueError("each batch anchor must have an observed positive")
    if not 0.0 < float(temperature):
        raise ValueError("temperature must be positive")
    logits = scores / float(temperature)

    def direction(x: Any, y: Any) -> Any:
        numerator = torch.logsumexp(x.masked_fill(~y, float("-inf")), dim=1)
        denominator = torch.logsumexp(x, dim=1)
        return -(numerator - denominator).mean()

    return 0.5 * (direction(logits, positive_mask) + direction(logits.T, positive_mask.T))


class CDRLateInteractionRetriever(_ModuleBase):
    """L1: CDR-slot to antigen-residue late interaction.

    CDR region IDs are sequence-derived IMGT regions only.  Values 0..5 represent
    H1/H2/H3/L1/L2/L3, while -1 means framework/padding.  Structural contact,
    paratope or epitope labels are never accepted by this class.
    """

    def __init__(
        self,
        antibody_input_dim: int = 512,
        antigen_input_dim: int = 1280,
        d_model: int = 128,
        cdr_slot_count: int = 6,
        temperature: float = 0.07,
    ) -> None:
        require_torch()
        super().__init__()
        if d_model <= 0 or cdr_slot_count != 6 or temperature <= 0:
            raise ValueError("invalid L1 dimensions/temperature")
        self.antibody_input_dim = int(antibody_input_dim)
        self.antigen_input_dim = int(antigen_input_dim)
        self.d_model = int(d_model)
        self.cdr_slot_count = int(cdr_slot_count)
        self.temperature = float(temperature)
        self.antibody_projection = nn.Sequential(
            nn.LayerNorm(self.antibody_input_dim),
            nn.Linear(self.antibody_input_dim, self.d_model),
        )
        self.antigen_projection = nn.Sequential(
            nn.LayerNorm(self.antigen_input_dim),
            nn.Linear(self.antigen_input_dim, self.d_model),
        )

    def encode_antibody_slots(self, features: Any, mask: Any, cdr_region_ids: Any) -> tuple[Any, Any]:
        _finite_residue_batch(features, mask, self.antibody_input_dim, "antibody")
        if not torch.is_tensor(cdr_region_ids) or cdr_region_ids.dtype != torch.long:
            raise ValueError("cdr_region_ids must be int64")
        if cdr_region_ids.shape != mask.shape or cdr_region_ids.device != mask.device:
            raise ValueError("cdr_region_ids shape/device mismatch")
        valid_ids = cdr_region_ids[mask]
        if bool(((valid_ids < -1) | (valid_ids >= self.cdr_slot_count)).any()):
            raise ValueError("CDR IDs must be -1 or 0..5")
        if bool((cdr_region_ids[~mask] != -1).any()):
            raise ValueError("padded residues must use CDR ID -1")
        z = F.normalize(self.antibody_projection(features), dim=-1)
        slots=[];present=[]
        for region in range(self.cdr_slot_count):
            m = mask & (cdr_region_ids == region)
            count=m.sum(1)
            pooled=(z*m.unsqueeze(-1).to(z.dtype)).sum(1)/count.clamp_min(1).unsqueeze(-1).to(z.dtype)
            slots.append(F.normalize(pooled,dim=-1))
            present.append(count>0)
        slot_tensor=torch.stack(slots,dim=1)
        slot_mask=torch.stack(present,dim=1)
        if bool((slot_mask.sum(1)==0).any()):
            raise ValueError("every antibody requires at least one sequence-derived CDR slot")
        slot_tensor=slot_tensor.masked_fill(~slot_mask.unsqueeze(-1),0.0)
        return slot_tensor,slot_mask

    def encode_antigen(self, features: Any, mask: Any) -> tuple[Any, Any]:
        _finite_residue_batch(features, mask, self.antigen_input_dim, "antigen")
        z=F.normalize(self.antigen_projection(features),dim=-1)
        return z.masked_fill(~mask.unsqueeze(-1),0.0),mask

    @staticmethod
    def cross_library_scores(
        antibody_slots: Any,
        antibody_slot_mask: Any,
        antigen_residues: Any,
        antigen_mask: Any,
    ) -> Any:
        """Return [antibody, antigen] local late-interaction scores."""
        if antibody_slots.ndim!=3 or antigen_residues.ndim!=3:
            raise ValueError("encoded residues must be 3D")
        if antibody_slots.shape[-1]!=antigen_residues.shape[-1]:
            raise ValueError("embedding dimensions differ")
        # [A,G,S,L]
        sim=torch.einsum("asd,gld->agsl",antibody_slots,antigen_residues)
        sim=sim.masked_fill(~antigen_mask[None,:,None,:],float("-inf"))
        best=sim.max(dim=-1).values
        best=best.masked_fill(~antibody_slot_mask[:,None,:],0.0)
        denom=antibody_slot_mask.sum(-1).clamp_min(1).to(best.dtype)[:,None]
        scores=best.sum(-1)/denom
        if not torch.isfinite(scores).all():
            raise ValueError("nonfinite L1 score")
        return scores

    def score_batch(
        self,
        antibody_features: Any,
        antigen_features: Any,
        antibody_mask: Any,
        antigen_mask: Any,
        cdr_region_ids: Any,
    ) -> Any:
        slots,slot_mask=self.encode_antibody_slots(antibody_features,antibody_mask,cdr_region_ids)
        antigen,amask=self.encode_antigen(antigen_features,antigen_mask)
        return self.cross_library_scores(slots,slot_mask,antigen,amask)


@dataclass(frozen=True)
class AuxiliaryLossReceipt:
    mode: str
    contact_labels_read: bool
    contact_loss_weight: float
    valid_complex_count: int


class MatchedBudgetReranker(_ModuleBase):
    """Parameter-identical R1/R2 wrapper.

    Both modes instantiate the same ProtocolReranker including a 64-d contact head.
    R1 never accepts contact labels; it includes a label-free zero-valued head path so
    parameter count and forward graph are comparable. R2 alone consumes fold-local
    training contact targets and adds the registered 0.10 auxiliary loss.
    """

    def __init__(
        self,
        coarse_retriever: ProtocolD0Retriever,
        mode: str,
        contact_loss_weight: float = 0.1,
    ) -> None:
        require_torch()
        super().__init__()
        if mode not in {"R1","R2"}:
            raise ValueError("mode must be R1 or R2")
        if float(contact_loss_weight)!=0.1:
            raise ValueError("strict v1 locks R2 contact weight at 0.10")
        self.mode=mode
        self.contact_loss_weight=float(contact_loss_weight)
        self.model=ProtocolReranker(
            coarse_retriever=coarse_retriever,
            antibody_input_dim=512,
            antigen_input_dim=1280,
            d_model=256,
            num_heads=8,
            num_layers=2,
            ffn_dim=512,
            dropout=0.1,
            contact_pair_dim=64,
            contact_initial_bias=-4.0,
        )

    def forward(self,*args:Any,**kwargs:Any)->dict[str,Any]:
        out=self.model(*args,**kwargs)
        out["stage"]=self.mode
        return out

    def auxiliary_loss(
        self,
        outputs: dict[str,Any],
        contact_targets: Any|None=None,
        resolved_mask: Any|None=None,
    )->tuple[Any,AuxiliaryLossReceipt]:
        logits=outputs.get("contact_logits")
        if logits is None:
            raise ValueError("matched-budget reranker must expose contact logits")
        if self.mode=="R1":
            if contact_targets is not None or resolved_mask is not None:
                raise PermissionError("R1 must not read contact labels")
            zero=logits.sum()*0.0
            return zero,AuxiliaryLossReceipt("R1",False,0.0,0)
        if contact_targets is None or resolved_mask is None:
            raise ValueError("R2 requires fold-local contact targets and resolved mask")
        details=balanced_contact_bce(logits,contact_targets,resolved_mask,return_details=True)
        return self.contact_loss_weight*details["loss"],AuxiliaryLossReceipt(
            "R2",True,self.contact_loss_weight,int(details["valid_complex_count"])
        )

    def parameter_signature(self)->dict[str,Any]:
        rows=[(n,tuple(p.shape),int(p.numel())) for n,p in self.named_parameters()]
        return {"parameter_count":sum(x[2] for x in rows),"named_shapes":rows}


def assert_r1_r2_parameter_parity(r1:MatchedBudgetReranker,r2:MatchedBudgetReranker)->None:
    a=r1.parameter_signature();b=r2.parameter_signature()
    if a!=b:
        raise AssertionError("R1/R2 parameter signatures differ")
