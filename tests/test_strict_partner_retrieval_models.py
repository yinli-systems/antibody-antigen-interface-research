from __future__ import annotations
import copy
import torch

from interaction_first.frontier_model import ProtocolD0Retriever
from strict_partner_retrieval_v1.contract_models import (
    CDRLateInteractionRetriever,
    MatchedBudgetReranker,
    assert_r1_r2_parameter_parity,
    symmetric_multi_positive_infonce,
)


def _d0(seed: int = 7) -> ProtocolD0Retriever:
    torch.manual_seed(seed)
    return ProtocolD0Retriever()


def test_l1_uses_only_sequence_cdr_ids_and_scores_cross_library() -> None:
    torch.manual_seed(1)
    m=CDRLateInteractionRetriever()
    ab=torch.randn(2,8,512);ag=torch.randn(3,11,1280)
    abmask=torch.ones(2,8,dtype=torch.bool);agmask=torch.ones(3,11,dtype=torch.bool)
    cdr=torch.tensor([[0,0,-1,1,1,-1,2,2],[0,0,1,1,2,2,2,-1]],dtype=torch.long)
    slots,sm=m.encode_antibody_slots(ab,abmask,cdr);arz,am=m.encode_antigen(ag,agmask)
    s=m.cross_library_scores(slots,sm,arz,am)
    assert s.shape==(2,3) and torch.isfinite(s).all()


def test_l1_rejects_no_cdr() -> None:
    m=CDRLateInteractionRetriever();ab=torch.randn(1,4,512);mask=torch.ones(1,4,dtype=torch.bool)
    try:m.encode_antibody_slots(ab,mask,torch.full((1,4),-1,dtype=torch.long))
    except ValueError:pass
    else:raise AssertionError('missing CDR mask accepted')


def test_r1_r2_parameter_signature_identical_and_initial_retrieval_equal() -> None:
    torch.manual_seed(9);base=_d0(9);state=copy.deepcopy(base.state_dict())
    torch.manual_seed(21);r1=MatchedBudgetReranker(base,'R1')
    base2=_d0(9);base2.load_state_dict(state)
    torch.manual_seed(21);r2=MatchedBudgetReranker(base2,'R2')
    assert_r1_r2_parameter_parity(r1,r2)
    assert r1.state_dict().keys()==r2.state_dict().keys()
    for key in r1.state_dict():
        assert torch.equal(r1.state_dict()[key], r2.state_dict()[key]), key
    # Protocol training contains dropout; fairness is checked at identical bytes and
    # deterministic evaluation, not by expecting two stochastic train-mode forwards
    # to consume the same RNG stream.
    r1.eval(); r2.eval()
    ab=torch.randn(2,5,512);ag=torch.randn(2,7,1280);am=torch.ones(2,5,dtype=torch.bool);gm=torch.ones(2,7,dtype=torch.bool);chain=torch.zeros(2,5,dtype=torch.long)
    with torch.inference_mode():
        a=r1(ab,ag,am,gm,chain)['retrieval_score'];b=r2(ab,ag,am,gm,chain)['retrieval_score']
    assert torch.equal(a,b)


def test_r1_never_accepts_contact_labels() -> None:
    r1=MatchedBudgetReranker(_d0(),'R1');out={'contact_logits':torch.randn(1,2,2,requires_grad=True)}
    loss,receipt=r1.auxiliary_loss(out)
    assert float(loss.detach())==0.0 and receipt.contact_labels_read is False
    try:r1.auxiliary_loss(out,torch.zeros(1,2,2),torch.ones(1,2,2,dtype=torch.bool))
    except PermissionError:pass
    else:raise AssertionError('R1 accepted contact labels')


def test_r2_requires_contact_labels() -> None:
    r2=MatchedBudgetReranker(_d0(),'R2');out={'contact_logits':torch.randn(1,2,2,requires_grad=True)}
    try:r2.auxiliary_loss(out)
    except ValueError:pass
    else:raise AssertionError('R2 accepted missing labels')
    y=torch.tensor([[[1.,0.],[0.,1.]]]);mask=torch.ones(1,2,2,dtype=torch.bool)
    loss,receipt=r2.auxiliary_loss(out,y,mask)
    assert torch.isfinite(loss) and receipt.contact_labels_read and receipt.contact_loss_weight==0.1


def test_multi_positive_loss_finite() -> None:
    s=torch.tensor([[2.,1.],[1.,2.]],requires_grad=True);p=torch.eye(2,dtype=torch.bool)
    loss=symmetric_multi_positive_infonce(s,p);assert torch.isfinite(loss);loss.backward();assert s.grad is not None
