"""No contact labels, checkpoints, scheduler writes or hidden true permutations.

The canonical solver orders candidate columns by the feature vectors themselves,
not their original residue index, before solving a square assignment. Tied costs
therefore cannot acquire a different arbitrary solution merely through column
reordering. This is only for a synthetic bijection within ONE biological chain;
never apply a one-to-one constraint to antibody-antigen contacts.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def new_json(path, value):
    encoded=json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n'
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as handle: handle.write(encoded)


def exact_permutation(order,n):
    value=np.asarray(order)
    if (value.shape!=(n,) or not np.issubdtype(value.dtype,np.integer)
        or not np.array_equal(np.sort(value),np.arange(n))):
        raise ValueError('not a complete integer permutation')
    return value.astype(np.int64,copy=False)


def canonical_assignment(logits, features):
    """Maximum-weight assignment with input-order-independent tie breaking.

    No truth indices/permutation are accepted. Finite feature byte values form
    the deterministic ordering; duplicate feature rows are interchangeable.
    Returns query-row -> original observed-column, not a biological truth map.
    """
    from scipy.optimize import linear_sum_assignment
    scores=np.asarray(logits,dtype=np.float64)
    extra=np.asarray(features)
    if (extra.ndim!=2 or not len(extra) or scores.shape!=(len(extra),len(extra))
        or extra.shape[1]<1 or not np.issubdtype(extra.dtype,np.floating)):
        raise ValueError('square scores and nonempty floating feature matrix required')
    if not np.isfinite(scores).all() or not np.isfinite(extra).all():
        raise ValueError('nonfinite assignment input')
    # Do not round/downcast vectors: preserve feature identity, including ties.
    canonical=np.ascontiguousarray(extra.astype(extra.dtype.newbyteorder('<'),copy=False))
    keys=[row.tobytes() for row in canonical]
    order=np.array(sorted(range(len(extra)),key=keys.__getitem__),dtype=np.int64)
    rows,cols=linear_sum_assignment(scores[:,order],maximize=True)
    if not np.array_equal(rows,np.arange(len(extra))): raise AssertionError('incomplete assignment')
    assignment=order[cols]
    exact_permutation(assignment,len(extra))
    return assignment


def signed_row_column_loss(logits, observed_to_original=None):
    """Symmetric CE; corruption labels only define the training target.

    This function is NOT used by the inference solver. With Lp=L[:,p], row
    targets are argsort(p), column targets are p, making CE unchanged by p.
    """
    import torch
    if logits.ndim!=2 or logits.shape[0]!=logits.shape[1] or len(logits)==0:
        raise ValueError('nonempty square logits required')
    n=len(logits)
    p=exact_permutation(np.arange(n) if observed_to_original is None else observed_to_original,n)
    idx=torch.tensor(p.copy(),dtype=torch.long,device=logits.device)
    truth=torch.argsort(idx)
    return .5*(torch.nn.functional.cross_entropy(logits,truth)
               +torch.nn.functional.cross_entropy(logits.T,idx))


def observed_transforms(joined,parts,order,rescue_fn,base_dim=1280,extra_dim=512):
    """Return aligned/corrupted/rescued/clean-rescued composites; no mask drift.

    Rescue is skipped for a wholly missing source. Partial-residue presence is
    rejected rather than changing availability semantics. The original base
    features and presence column remain bitwise identical in every arm.
    """
    import torch
    n=len(joined)
    if (joined.ndim!=2 or joined.shape[1]!=base_dim+extra_dim+1
        or n==0 or sum(len(x) for x in parts)!=n or any(not x for x in parts)):
        raise ValueError('feature shape or biological chain boundary mismatch')
    if not torch.isfinite(joined).all(): raise ValueError('nonfinite composite')
    p=exact_permutation(order,n)
    presence=joined[:,-1]
    if not bool(((presence==0)|(presence==1)).all()): raise ValueError('nonbinary presence')
    if not bool((presence==presence[0]).all()): raise ValueError('partial-residue missingness')
    start=0
    for part in parts:
        stop=start+len(part)
        if np.any((p[start:stop]<start)|(p[start:stop]>=stop)): raise ValueError('cross-chain permutation')
        start=stop
    variants={name:joined.clone() for name in ('aligned','corrupted','rescued','clean_rescued')}
    if not bool(presence[0]):
        if bool((joined[:,base_dim:-1]!=0).any()): raise ValueError('absent source has nonzero features')
        return variants,False
    base=joined[:,:base_dim]; extra=joined[:,base_dim:-1]
    idx=torch.tensor(p.copy(),dtype=torch.long,device=joined.device)
    corrupt=extra.index_select(0,idx)
    restored=rescue_fn(base.clone(),corrupt.clone(),parts)
    clean=rescue_fn(base.clone(),extra.clone(),parts)
    for candidate in (restored,clean):
        if candidate.shape!=extra.shape or not torch.isfinite(candidate).all():
            raise ValueError('rescue returned malformed features')
        start=0
        for part in parts:
            stop=start+len(part)
            original=extra[start:stop].detach().cpu().numpy()
            returned=candidate[start:stop].detach().cpu().numpy()
            if sorted(row.tobytes() for row in original)!=sorted(row.tobytes() for row in returned):
                raise ValueError('rescue altered feature multiset or crossed chains')
            start=stop
    variants['corrupted'][:,base_dim:-1]=corrupt
    variants['rescued'][:,base_dim:-1]=restored
    variants['clean_rescued'][:,base_dim:-1]=clean
    for candidate in variants.values():
        if not torch.equal(candidate[:,:base_dim],base) or not torch.equal(candidate[:,-1],presence):
            raise AssertionError('base/presence changed')
    return variants,True


def validate_rescue_checkpoint(checkpoint, history, plan_sha,seed):
    """Bind the loaded matcher to the exact frozen plan and selection receipt."""
    if checkpoint.get('plan_sha256')!=plan_sha or checkpoint.get('seed')!=seed:
        raise ValueError('rescue checkpoint plan/seed mismatch')
    if history.get('selection_uses_contact_metrics') is not False or history.get('test_access') is not False:
        raise ValueError('invalid selection authority')
    rows=history.get('history')
    if not isinstance(rows,list) or not rows: raise ValueError('missing selection history')
    values=np.asarray([row['validation_matching_accuracy'] for row in rows],dtype=float)
    if not np.isfinite(values).all(): raise ValueError('nonfinite selection metric')
    winner=rows[int(values.argmax())]  # historical strict > rule keeps first exact tie
    if (checkpoint.get('epoch')!=winner['epoch'] or history.get('selected_epoch')!=winner['epoch']
        or checkpoint.get('validation_matching_accuracy')!=winner['validation_matching_accuracy']
        or history.get('best_validation_matching_accuracy')!=winner['validation_matching_accuracy']):
        raise ValueError('selected checkpoint does not match declared criterion')


def summarize_records(records,expected_ids,seed_pairs,conditions):
    """Complete denominators only; gap recovery is nullable, never clipped."""
    expected={(str(i),int(s),int(d),str(c)) for i in expected_ids
              for s,d in seed_pairs for c in conditions}
    actual={}
    for r in records:
        key=(str(r['instance_id']),int(r['rescue_seed']),int(r['downstream_seed']),str(r['condition']))
        if key in actual: raise ValueError('duplicate result row')
        for field in ('aligned_ap','corrupted_ap','rescued_ap','clean_rescued_ap'):
            if not np.isfinite(r[field]) or not 0<=r[field]<=1: raise ValueError('invalid AP')
        actual[key]=r
    if set(actual)!=expected: raise ValueError('incomplete/foreign case-seed-condition grid')
    output=[]
    for condition in conditions:
        rows=[r for r in records if r['condition']==condition]
        means={f:float(np.mean([r[f] for r in rows])) for f in
               ('aligned_ap','corrupted_ap','rescued_ap','clean_rescued_ap')}
        gap=means['aligned_ap']-means['corrupted_ap']
        effect=means['rescued_ap']-means['corrupted_ap']
        output.append({'condition':condition,**means,
                       'rescued_minus_corrupted_ap':effect,
                       'clean_rescued_minus_aligned_ap':means['clean_rescued_ap']-means['aligned_ap'],
                       'gap_recovered_ratio':effect/gap if gap>1e-12 else None,
                       'gap_status':'positive' if gap>1e-12 else 'undefined_nonpositive_gap',
                       'complex_count':len(expected_ids),'seed_pair_count':len(seed_pairs),
                       'unique_complexes_not_multiplied_by_conditions':True})
    return output
