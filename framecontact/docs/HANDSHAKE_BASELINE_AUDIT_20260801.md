# Handshake baseline audit — 2026-08-01

## Decision

Handshake is a high-value **partner-aware sequence-only control**, but its
public repository must not be copied into FrameContact or presented as a fair
benchmark result in its current form. The correct path is an independent,
minimal reimplementation evaluated on the exact same double-cold roles,
candidate panels, labels, seeds, and metrics as FrameContact.

This audit is based on the [Handshake preprint](https://www.biorxiv.org/content/10.64898/2026.06.04.730144v1)
and the official [`nurith/Handshake`](https://github.com/nurith/Handshake)
repository pinned at commit
`cfb10ffcf0d5745473fe1deb3a6e849b9f5986f4` (commit timestamp
2026-06-08T16:22:31-04:00).

## What is scientifically useful

- ProstT5 plus LoRA supplies a sequence representation with structure-aware
  pretraining, providing a stronger control than frozen ESM alone.
- Bidirectional cross-chain attention makes both residue predictions depend on
  the supplied partner.
- A residue-pair contact objective is trained jointly with residue-level
  binding-site labels.
- The paper's redundancy-threshold study is direct evidence that split
  construction can materially inflate partner-specific interface metrics.

These ideas support a same-data, same-budget baseline and a representation
factorial experiment. They do not validate FrameContact's geometry, its
double-cold split, or antibody-specific generalization by themselves.

## Public-code boundary

The audited revision contains no `LICENSE` or `COPYING` file. Consequently,
the repository is readable research evidence but there is no affirmative
permission to copy, modify, or redistribute its implementation. FrameContact
must use a clean-room implementation from the paper-level description unless
the authors add a compatible license.

No official release or model checkpoint is present in the audited repository.
The comparison therefore requires training; published numbers are
non-comparable literature context only.

## Reproducibility and evaluation findings

The following findings are reasons to reimplement the baseline under the
FrameContact evaluator rather than invoke the script directly:

1. The default training path performs a random 70/15/15 pair split unless
   explicit lists are supplied. This is not endpoint-disjoint and is not an
   antigen structural-fold split.
2. The provided non-redundancy script clusters the concatenated
   `seq_A + seq_B`. Selecting one representative per pair cluster does not
   prove that the same antibody or antigen endpoint is absent across roles.
3. Its contact head consumes only the two normalized directional attention
   weights. It predicts a submatrix selected by the current unary residue head
   at a fixed threshold, with a top-20% fallback. Contact recall is therefore
   structurally dependent on the unary prefilter.
4. Contact-matrix export switches the model into `train()` so the forward path
   returns contact tensors, then restores `eval()`. Dropout can therefore make
   exported contact matrices stochastic even inside `torch.no_grad()`.
5. The generic `--inference-only` path tunes a threshold on the same labeled
   inference set. It is suitable for exploratory diagnostics only; it cannot
   be used for frozen-test threshold metrics.
6. The normal train/evaluate path does select the residue threshold on the
   validation set before applying it to test, which is the correct boundary,
   but checkpoint selection uses residue MCC rather than reciprocal partner
   retrieval.

These observations do not invalidate the preprint. They define the exact
engineering differences that must be controlled before a head-to-head claim.

## Clean-room baseline specification

Call the implementation **ProstT5-cross-contact control**, not Handshake, until
the official code can be reused under a license.

### Inputs and architecture

1. Encode antibody and antigen sequences with the same authenticated ProstT5
   revision. Record model, tokenizer, revision, license, and weight hashes.
2. Compare frozen embeddings against LoRA at a pre-registered rank and equalize
   added trainable parameters where practical.
3. Use one deterministic bidirectional cross-attention block.
4. Predict the full antibody-by-antigen contact matrix from directional pair
   features; do not hide candidate pairs behind a hard unary threshold.
5. Derive epitope and paratope scores from the same contact marginalization and
   gated-unary alternatives already frozen for FrameContact.
6. At inference, keep the entire module in `eval()` and assert repeatable logits
   on a fixed device when deterministic kernels permit it.

### Fair-data contract

- train and validation: the authenticated SAbDab2 double-cold roles only;
- no test panel or test label read before architecture and checkpoint rules are
  frozen;
- identical real pair labels, 8 Å contact convention, residue scope, missing
  residue policy, and reciprocal 31-candidate panels;
- five seeds `17, 42, 73, 101, 137`;
- audit ProstT5 pretraining cutoff and possible SAbDab2/PDB overlap, and label
  the representation as externally pretrained rather than data-equivalent if
  overlap cannot be excluded.

### Required output

- pair contact pooled and macro AUPRC, P@10, P@Lmin/10, P@Lmin/5, P@Lmin;
- epitope and paratope pooled and macro AUPRC;
- fixed validation-selected precision, recall, F1, MCC and positive rate;
- Ab-to-Ag and Ag-to-Ab R@1/R@5/MRR and worst-direction margin;
- partner shuffle, endpoint-preserving hard-negative retrieval and calibration;
- paired seed deltas, exact sign-flip diagnostics, cluster bootstrap intervals,
  and per-complex failure strata.

## Promotion rule

This baseline is valuable even if it wins. If the ProstT5 control matches
FrameContact, the result says structural pretraining plus partner interaction
may be sufficient and the explicit monomer geometry contribution is small. If
FrameContact improves the worst-direction retrieval and contact metrics under
the same double-cold protocol, the geometry claim becomes much more credible.

No promotion decision is allowed from the current in-flight v3.1 development
arrays, because those arrays use the earlier split and do not contain this
control.
