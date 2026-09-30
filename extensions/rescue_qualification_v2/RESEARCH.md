# Primary-source review and an exact architectural deduction

Reviewed 2026-09-30 before the supplemental qualification implementation.

1. PLASMA, Fast and Interpretable Protein Substructure Alignment via Optimal
Transport (ICLR 2026): https://arxiv.org/html/2510.11752v1 ; official implementation
https://github.com/ZW471/PLASMA-Protein-Local-Alignment . It uses a transport
planner and assessor for protein substructure alignment. Our small synthetic
row-recovery matcher is NOT a reproduction or an algorithmic novelty claim.
2. OTalign, Bioinformatics 42(7), btag476 (published 2026-06-30):
https://academic.oup.com/bioinformatics/article/42/7/btag476/8722293 ; official
https://github.com/DeepFoldProtein/OTalign . Its unbalanced transport plus dynamic
programming addresses gaps in homolog alignment. Balanced matching is appropriate
for the present known-size permutation, not for general biological gaps or
antibody-antigen contact assignments. Their ablations attribute much of the
benefit to learned/contextual gap costs; simply adding Sinkhorn is not sufficient.
3. Mena et al., Learning Latent Permutations with Gumbel-Sinkhorn Networks,
https://arxiv.org/html/1802.08665v1 . Permutation-equivariant matching is established
prior art. Entropy-regularized soft assignment is not identical to a hard optimum.
4. SciPy official assignment API:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html .
The implementation is modified Jonker-Volgenant, not literally a Hungarian
implementation despite the parent helper's name. It returns a maximum-weight
one-to-one assignment when maximize=True; optimal assignments can have ties.

## Deduction for the exact current matcher, not an empirical biology claim

Let X be correctly ordered base features, E extra features, and p a permutation
such that corrupted[j] = E[p[j]]. Rowwise q and k imply Lp = q(X)k(E[p])^T
= L[:,p]. The correct row target is argsort(p), and the column target is p.
Both cross-entropies are reindexings of the uncorrupted losses, hence

  CE_rows(L[:,p], argsort(p)) = CE_rows(L, identity)
  CE_cols(L[:,p], p) = CE_cols(L, identity).

Their gradients with respect to matcher parameters are equal in exact arithmetic.
This holds for shift, local, same-AA and global permutations, within each chain,
provided the matcher has no position-dependent feature corruption/attention and
any stochastic state is held fixed. Floating-point reduction order can differ.

With a unique optimal assignment, the recovered feature tensor is likewise
independent of which pure permutation was applied. Thus the three corruption
families do not constitute three independent generalization tests of this matcher.
They DO remain different inputs to the fixed downstream contact model without
rescue. Do not conflate those two statements.

With tied optima, arbitrary solver tie-breaking can depend on column order. The
secondary solver canonicalizes by candidate feature content (no truth indices)
before assignment. Identical feature rows are interchangeable in the reconstructed
tensor. No epsilon is added to the cost and no near-tie is silently declared exact.

Consequences: evaluate clean inputs through the same matcher; retain untrained
controls; and never promote a claimed 3x biological replication from these three
pure-permutation families. The 3-to-1 reduction is an algebraic opportunity for
redundant validation computation, NOT a measured walltime/GPU-throughput claim.
