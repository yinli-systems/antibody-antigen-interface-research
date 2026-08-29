# Retrieval-aligned partner specificity candidate

**Status:** post-freeze exploratory candidate; implemented, fully regression
tested, and train-role GPU stress-tested, but not model-selected, trained, or
quality-benchmarked.

## Failure being addressed

The frozen development runs train each focal complex against one wrong antigen
and one wrong antibody per epoch.  For direction `d`, the current loss is

```text
softplus(m + s_wrong,d - s_cognate).
```

The wrong partner is a deterministic epoch-dependent draw from a frozen,
label-free panel.  Validation is harder: each cognate is ranked among 31
candidates separately for fixed-antibody/wrong-antigen and
fixed-antigen/wrong-antibody retrieval.  Model selection uses the poorer
directional cluster-weighted margin; R@1 and MRR are diagnostics.

This creates three objective/evaluation mismatches:

1. one sampled negative versus a 31-candidate ranking panel;
2. mean loss over directions versus a worst-direction selection gate;
3. pairwise separation versus R@1/MRR listwise ranking.

It plausibly explains why the live distogram snapshot can improve the registered
margin while directional R@1 does not improve.  This is a diagnosis, not proof
of causality.

## Implemented candidate

For each direction, draw `K` unique candidates from the already frozen
training-role panel using a seed/epoch/focal/direction/candidate hash.  No task
labels are read and wrong pairs are not assigned fabricated zero contact maps.
All candidates are scored with the same interaction-only top-fraction score used
by validation.

For cognate score `c` and wrong scores `w_1 ... w_K`, use

```text
L_d = logsumexp(0, (m + w_1 - c)/T, ..., (m + w_K - c)/T).
```

The two directional losses are either averaged or combined with a normalized
smooth maximum:

```text
L_specificity = tau * (logsumexp(L_d / tau) - log(D)).
```

The smooth maximum gives both directions gradients while emphasizing the worse
direction.  The old path remains the default:

```text
K=1, pairwise_softplus, mean direction aggregation.
```

Therefore existing frozen protocols and running arrays are unchanged.

## Completed engineering evidence

The source-authenticated exploratory tree passed `269` tests with `5` skips and
no failures.  The skips are confined to the optional Boltz adapter because its
pinned source is not installed in that test environment; no FrameContact,
specificity, split, frozen-test gate, or evidence-lock test was skipped.

On the longest antigen in the SAbDab2 **training role** (`2363` antigen and
`231` antibody residues), K=1, K=4, and K=8 all completed a finite BF16
forward/backward pass on an RTX 4090.  K=4 and K=8 initially exhausted host
memory because saved-tensor CPU offload retained every negative branch.
Checkpointing only the wrong-partner branches reduced Slurm batch `MaxRSS` to
approximately `1.95` and `2.10` GiB and allowed both to finish.  Their recorded
peak allocated CUDA memory was `10.46` and `10.69` GB, respectively.
The regression suite also verifies that checkpoint recomputation preserves the
training-mode dropout RNG path and propagates finite parameter gradients.

K=4 is the single engineering-frozen candidate for the later development-only
comparison.  K=8 remains a stress comparator, not an additional hyperparameter
trial.  This choice avoids validation fishing and uses less work than K=8; the
single-example timings are not a throughput benchmark.  Full receipts and the
source manifest are in
`receipts/listwise-specificity-v1-1346200/ENGINEERING_RECEIPT.json`.

The candidate uses `T=1.0`, not the `T=0.1` used by the first memory stress.
At `T=1` and K=1, the listwise expression is exactly the old pairwise
`softplus`, including its gradient.  This preserves the old single-negative
loss scale and isolates the K-negative change; `T=0.1` would amplify score
gradients by up to tenfold and confound the comparison.  The exact K=4/T=1
configuration also completed the same train-role GPU stress with finite
forward/backward values and no validation or test access (Slurm `1346250`).

The current pre-existing arrays remain in flight.  At their latest synchronized
snapshots, unary-only had reached epoch 14 and distogram epoch 11.  Distogram
showed higher descriptive contact pooled AUPRC (`0.01262` versus `0.01035`) and
a better worst-direction retrieval margin (`0.00607` versus `-0.01207`), but
lower fixed-antigen/wrong-antibody R@1 (`0.0499` versus `0.0517`).  These are
different epochs from unfinished runs, so they are diagnostic evidence of the
objective/ranking mismatch, not a model-selection result.

## Evidence from current literature

- CALM explicitly evaluates bidirectional cognate antibody-antigen retrieval;
  its reported 80%-identity split is weaker than the present antigen-fold audit,
  but the two-direction retrieval formulation is directly relevant.
- EpiCLIP casts approximate epitope/paratope matching as dense retrieval, which
  supports treating specificity as ranking rather than an after-the-fact scalar
  control.
- EpiFormer is a critical negative result: standard intra/inter-chain
  residue-level InfoNCE reduced its reported F1 from 0.496 to 0.467.  That loss
  is not the same as cognate-vs-wrong-partner listwise ranking, but it rules out
  assuming that any contrastive term will help sparse contact classification.
- AAMFM uses antigen geometry and epitope information through a small adapter,
  then Cal-DPO for design preferences.  It reinforces explicit partner
  conditioning but is a generation pipeline, not evidence for contact retrieval.

## Promotion protocol

Only after the current five-condition selection is frozen:

1. use the already engineering-frozen `K=4`, `T=1.0`, smooth-maximum candidate;
2. compare the frozen-choice pair trunk with old pairwise specificity versus
   listwise specificity under five seeds;
3. primary retrieval gate: improve the minimum of the two directional R@1 and
   MRR, plus the registered worst-direction cluster-weighted margin;
4. non-degradation gates: contact macro AUPRC, P@10, P@Lmin/5, epitope and
   paratope AUPRC/F1/MCC, and calibration;
5. inspect correct-partner versus wrong-partner score distributions by antigen
   fold and partner panel difficulty;
6. do not use frozen or external test to select `K`, temperature, epoch, or loss
   weight.

Random or metadata-matched wrong partners are ranking counterfactuals, not
experimentally proven non-binders.  The resulting loss can support a retrieval
claim, but not a biochemical non-binding claim.

## Novelty boundary

The combination of contact-derived compatibility, two-direction label-free
partner panels, listwise loss, and worst-direction aggregation may be useful for
FrameContact.  It is not yet established as novel.  A broader prior-art and
patent search plus positive five-seed evidence is required before making a
novelty claim.
