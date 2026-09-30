# Learned residue-correspondence rescue v1

Frozen before any rescued contact metric is read. The experiment asks whether a
trainable matcher can recover deliberately displaced ESM-IF1 residue features
using only the frozen ESM-2 representation and the displaced ESM-IF1 features.

## Model and supervision

A two-tower 128-d projection maps ESM-2 and ESM-IF1 residue vectors to a shared
space. Cosine logits are trained with symmetric row/column cross-entropy against
the **known synthetic permutation**. No contact, epitope, paratope, partner,
test, OOD or affinity label enters rescue training or checkpoint selection.
Training corruptions are chain-preserving shifts, local shuffles, same-amino-acid
shuffles and global within-chain permutations. Validation checkpoint selection is
fixed synthetic matching accuracy only.

The hard inference mapping is the maximum-weight one-to-one assignment (Hungarian)
of the learned similarity matrix, separately within each biological chain. This
is deliberately interpretable and related to recent residue-alignment work using
differentiable/optimal transport; the method here is a small project-specific
matcher, not a claim to reproduce PLASMA or OTalign.

## Frozen evaluation

Three rescue seeds `{101,202,303}` are paired in order with the three existing
ESM-IF1 downstream seeds `{1,2,3}`. The 217 existing development-validation
complexes are evaluated once after rescue selection. Primary corruption is
same-amino-acid within-chain shuffling; secondary conditions are shift-8 and
global within-chain permutation. Report aligned AP, corrupted AP, rescued AP,
absolute rescue effect and fraction of the aligned-minus-corrupted gap recovered.
Mean-per-complex pair AP is primary. Equal authority-component bootstrap is
reported. No closed test or OOD data is opened.

A positive result means synthetic correspondence is recoverable enough to improve
the frozen contact model. It does **not** show that naturally misregistered
biological representations exist, or establish universal residue correspondence.
No model selection may use contact AP. Failed seeds and negative rescue results
remain in the report.
