"""Sequence-only IMGT CDR annotation for the L1 retrieval baseline."""
from __future__ import annotations
import importlib.metadata
from typing import Any

REGION_TO_SLOT={"CDR1":0,"CDR2":1,"CDR3":2}

def runtime_receipt()->dict[str,Any]:
    out={}
    for name in ("abnumber","anarci"):
        try: out[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: out[name]=None
    return out

def annotate_chain(sequence:str, expected_kind:str)->tuple[list[list[bool]],dict[str,Any]]:
    """Return 3 x L CDR mask for one VH or VL sequence.

    AbNumber's IMGT numbering and IMGT CDR definition are used. The numbered
    amino-acid stream must reconstruct the exact supplied sequence; no alignment
    gap, tail truncation, or heuristic fallback is permitted.
    """
    try:
        from abnumber import Chain
    except ImportError as e:
        raise RuntimeError("AbNumber/ANARCI is required for frozen sequence-only CDR annotation") from e
    chain=Chain(sequence,scheme="imgt",cdr_definition="imgt")
    if expected_kind=="heavy" and not chain.is_heavy_chain():raise ValueError("VH did not number as heavy")
    if expected_kind=="light" and not chain.is_light_chain():raise ValueError("VL did not number as light")
    numbered=list(chain)
    reconstructed="".join(aa for _,aa in numbered)
    if reconstructed!=sequence:raise ValueError("IMGT numbering does not reconstruct exact sequence")
    masks=[[False]*len(sequence) for _ in range(3)]
    regions=[]
    for i,(pos,aa) in enumerate(numbered):
        region=pos.get_region()
        regions.append(region)
        if region in REGION_TO_SLOT:masks[REGION_TO_SLOT[region]][i]=True
    if not all(any(x) for x in masks):raise ValueError("numbered chain lacks one or more IMGT CDRs")
    return masks,{"chain_type":chain.chain_type,"scheme":chain.scheme,"cdr_definition":chain.cdr_definition,
                  "cdr_lengths":[sum(x) for x in masks],"runtime":runtime_receipt()}

def annotate_antibody(vh:str,vl:str)->dict[str,Any]:
    hm,hr=annotate_chain(vh,"heavy")
    total=len(vh)+len(vl);masks=[[False]*total for _ in range(6)]
    for s in range(3):
        masks[s][:len(vh)]=hm[s]
    light_receipt=None
    if vl:
        lm,light_receipt=annotate_chain(vl,"light")
        for s in range(3):
            masks[3+s][len(vh):]=lm[s]
    return {"cdr_masks":masks,"vh_length":len(vh),"vl_length":len(vl),
            "slot_order":["H1","H2","H3","L1","L2","L3"],
            "heavy":hr,"light":light_receipt,"sequence_only":True,
            "bound_structure_or_contact_used":False}
