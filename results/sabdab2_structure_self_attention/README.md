# SAbDab2 structure-aware self-attention ablation

This directory records the lightweight, development-only result of the real
SAbDab2 structure adapter. Raw structures, residue embeddings, labels, full
record manifests, logs, and model weights are intentionally excluded.

## Data boundary

- The authenticated Needleman--Wunsch development authority contains 3,393
  train and 52 validation instances.
- Intersecting it with the newer SAbDab2 release's native training role leaves
  2,986 train and 43 validation candidates. The 416 native-release test cases
  are excluded before reading their mmCIF members.
- Exact structure-to-canonical-axis validation accepts 2,985 train and all 43
  validation structures. One train complex is rejected because its new-release
  VH sequence differs from the canonical embedding/label axis.
- The preregistered `antigen_length <= 384` runtime cap leaves 2,676 train and
  36 validation complexes for this run.
- All 3,028 accepted artifacts have complete valid `N/CA/C` frames. The explicit
  missing-frame path is tested, but the accepted real cohort has zero positive
  missing-frame examples.

## Fixed-final-epoch result

All modes use frozen ESM2 residue inputs, separate antibody and antigen
self-attention parameters, the same seed and initialization, 271,670 parameters,
three epochs, and the same same-chain `k=32`, 16-A spatial graph. `none` is a
spatial-topology-only control, not a dense sequence-only Transformer.

| Edge features | Paratope AUPRC | Epitope AUPRC | Macro AUPRC | Macro AUROC |
|---|---:|---:|---:|---:|
| topology only | 0.627464 | 0.253331 | 0.440398 | 0.829702 |
| distance RBF | 0.630673 | 0.256434 | 0.443554 | 0.831124 |
| distance + local direction | 0.655229 | 0.255474 | 0.455351 | 0.829258 |
| full relative pose | 0.652331 | 0.257624 | 0.454978 | 0.830546 |

Metrics pool 5,149 antibody residues and 9,572 antigen residues from the 36
validation complexes. Epoch 3 is reported for every mode without validation-best
checkpoint selection.

The numerical pattern is a development signal only. This is one seed, a short
budget, a small validation intersection, and has no confidence interval or
held-out confirmation. It does not establish a benchmark improvement.

## Files

- `ablation.json`: exact configuration, cohort digests, runtime, and all epoch
  metrics. SHA-256:
  `4c5114e96dc4514c20695332a877afd9299c70a188d859aa5c713b7be84933cc`.
- `run_receipt.json`: result digest and final cohort counts.
- `structure_cache_receipt.json`: authenticated cache manifest digest and
  pre-length-cap record counts.

## Claim boundary

No frozen test was read. This is not a real PairFormer checkpoint result, an
inter-chain contact/cross-attention result, a fused-kernel benchmark, or evidence
for biological, clinical, external-generalization, state-of-the-art, or
benchmark-improvement claims.
