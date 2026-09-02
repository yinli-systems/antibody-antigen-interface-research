# PINDER antibody S40 four-view diagnostic v1

The train-sequence-only relation job `1533008` completed `0:0` with empty
stderr. It materialized MMseqs2 relations at identity `>=0.40` and reciprocal
coverage `>=0.80` in VH, non-empty VL, concatenated-antibody, and antigen views.
No structures, task labels, predictions, validation rows, or test rows were
read. The result SHA-256 is
`aceaf0b53b60f4c124c3c3b26e57082477678ccebf3b0c8cb939b5692b1f6b5d`.

The component diagnostic job `1533060` also completed `0:0` with empty stderr.
The strict no-discard binary split is infeasible: `2,716 / 2,718` unique exact
pairs fall in the largest antibody-side component, while the antigen side
retains 605 components and independently has feasible 15--25% subset sums.

This demonstrates that a uniform 40% threshold over full VH/VL sequences is
not a usable antibody benchmark rule for this pool. It does not authorize
threshold relaxation, adaptation, model selection, or external-test access.
The scientifically meaningful next protocol must either:

1. report the ultra-cold four-view cohort as infeasible/empty;
2. preregister a discard policy and disclose its retained coverage; or
3. use an explicit asymmetric benchmark, such as antigen-40 plus a separately
   justified antibody/CDR coldness rule, without calling it four-view-40.
