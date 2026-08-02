# Contact-adapted ESM x partner-aware pair-trunk candidate

Status: post-freeze candidate design, not a registered experiment. No training
or validation result is claimed here.

## Question

Does long-range intra-chain contact supervision improve the monomer
representation itself, and is any resulting interface gain independent of the
partner-aware pair trunk?

This question is motivated by the 2026-07 LC-SEPLM preprint, but the experiment
is deliberately narrower and leakage-controlled for the present SAbDab2 task.

## Proposed 2 x 2 factorial

| ESM representation | Interface module | What it isolates |
|---|---|---|
| authenticated frozen ESM2-650M | unary residue head | current monomer-propensity reference |
| authenticated frozen ESM2-650M | frozen-choice FrameContact pair trunk | partner-conditioning effect |
| contact-adapted ESM2-650M | identical unary residue head | representation-only effect |
| contact-adapted ESM2-650M | identical FrameContact pair trunk | combined effect and interaction |

“Frozen-choice pair trunk” means the architecture selected by the already
running development matrix before this factorial begins. This study may not
reopen that choice.

## Leakage-safe adapter route

- Fit LoRA and its long-range intra-chain contact probe using only an inner
  train/inner-validation partition of the current SAbDab2 train role.
- Construct the inner partition by existing antibody and antigen cluster
  authorities; the outer validation role is never used for adapter selection.
- Use only intra-chain labels derived from real N/CA/C-backed monomer geometry.
  Do not use inter-chain contacts, epitope/paratope labels or wrong-partner
  identities during adapter pretraining.
- After inner selection, freeze the adapter, materialize residue embeddings
  with sequence, adapter, source and structure-manifest hashes, and train the
  four outer models under identical budgets.
- External AlphaFold/Swiss-Prot pretraining is a separate experiment and is
  forbidden until a release-cutoff, sequence, Foldseek/TM-score and test-overlap
  audit exists.

LoRA rank, adapted layers, contact definition, negative sampling, inner split
and optimization budget must be fixed only after a train-role-only memory and
gradient smoke. They are intentionally not guessed in this candidate document.

## Required readout

- contact macro and pooled AUPRC, P@10, P@Lmin/5 and Recall@Lmin/5;
- epitope/paratope macro and pooled AUPRC plus fixed-threshold
  precision/recall/F1/MCC;
- worst-direction cognate margin, both directional R@1 and MRR;
- geometry shuffle and partner shuffle controls;
- parameter count, peak memory, embedding materialization time and epoch time;
- five seeds with paired seed deltas and cluster bootstrap after selection.

The central falsifier is simple: if contact adaptation improves unary
propensity but not contact localization or worst-direction partner retrieval,
it is a better monomer prior, not a solution to partner specificity. Conversely,
if the pair trunk improves retrieval under both ESM variants, the interaction
module is the primary causal contribution.

