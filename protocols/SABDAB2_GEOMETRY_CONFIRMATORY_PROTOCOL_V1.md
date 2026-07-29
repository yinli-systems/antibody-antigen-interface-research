# SAbDab2 geometry-aware self-attention confirmatory protocol v1

## Status and claim boundary

This protocol is frozen before generating per-complex predictions from the
existing AB03/A100 checkpoints. It upgrades the original development ablation
from pooled descriptive metrics to paired, complex-level inference. It does not
unseal a native or repository test set, authorize model selection on validation,
or establish state of the art.

The original fixed-final-epoch checkpoints remain immutable. No checkpoint,
threshold, seed, feature mode, or validation member may be changed after this
protocol is frozen. A failure is a result and must be retained.

## Scientific question

Does adding intra-chain geometry to equal-parameter, independently initialized
antibody and antigen self-attention improve residue-level paratope and epitope
ranking over the same spatial topology with zero pair features?

The primary comparison is:

```text
distance_direction - topology_only
```

The primary estimand is the paired mean difference in per-complex joint
macro-AUPRC:

```text
0.5 * (paratope AUPRC + epitope AUPRC)
```

Secondary comparisons are:

1. `distance - topology_only`;
2. `full_pose - topology_only`;
3. `full_pose - distance_direction`.

The final comparison is a complexity diagnostic, not a superiority gate.

## Authority and immutability

The analysis must authenticate:

1. the canonical SAbDab2 structure-cache manifest;
2. the original ablation result and run receipt;
3. every fixed-final-epoch checkpoint;
4. the ordered validation membership;
5. the exact analysis protocol and implementation source.

The validation universe is the intersection that was fixed before training:

```text
locked development validation membership
INTERSECT
SAbDab2 native ab_ag_split=train
```

The native SAbDab2 test role and the repository frozen test remain unread.
SAbDab2 provides standardized, versioned machine-learning splits; antigen-aware
claims require the antibody-antigen split rather than an antibody-only split:

- https://www.biorxiv.org/content/10.64898/2026.06.16.732554v1
- https://www.blopig.com/blog/2026/06/sabdab2-the-structural-antibody-database-in-the-age-of-machine-learning/

## Metrics

### Per-complex metrics

For each complex and each side, report:

- non-interpolated, tie-aware AUPRC;
- tie-aware AUROC;
- prevalence;
- Brier score and negative log-likelihood;
- F1 and MCC at the locked threshold `0.5`.

The primary unit is a complex, not a residue. Pooled residue metrics are retained
only to reproduce the historical receipt and are explicitly descriptive because
long complexes otherwise receive more weight.

### Paired uncertainty

Endpoint-connected components are reconstructed directly from the validation
manifest. Two complexes share a component when they share an exact antibody or
antigen sequence digest, including transitive connections. The complete
component is the bootstrap and permutation unit.

For every comparison:

- use `10,000` paired component bootstrap replicates;
- use PCG64DXSM seed `20260729`;
- report a two-sided `95%` percentile interval;
- use `10,000` component sign-flip permutations with PCG64DXSM seed
  `20260730`;
- report one-sided and two-sided p-values with the plus-one correction;
- apply Holm correction across the three superiority comparisons.

If fewer than ten independent components remain, intervals are descriptive and
cannot authorize a positive claim.

The primary superiority gate passes only if all conditions hold:

1. the primary mean difference is positive;
2. its paired component-bootstrap lower bound is greater than zero;
3. its Holm-adjusted one-sided permutation p-value is below `0.05`;
4. paratope and epitope effects have the same non-negative direction;
5. the result is reproduced across the preregistered multi-seed extension.

The existing single-seed replay alone can never satisfy condition 5.

## Partner-specificity boundary

This experiment contains no inter-chain fusion. It can show that intra-chain
geometry improves marginal interface-site prediction, but it cannot prove
partner-specific recognition. SPPIDER-seq demonstrates partner dependence with
matched partner shuffling, and EpiFormer obtains its principal gains from
interleaved antibody-antigen cross-attention:

- https://academic.oup.com/bioinformatics/article/42/7/btag388/8707840
- https://arxiv.org/abs/2606.04154

Partner specificity therefore requires a separate frozen experiment containing:

1. cognate versus matched partner shuffle;
2. antibody-only and antigen-only controls;
3. a native antibody-residue by antigen-residue contact matrix;
4. bidirectional partner retrieval;
5. a hard antibody-antigen leakage-controlled outer test.

No marginal self-attention result may be relabeled as evidence for these tasks.

## Multi-seed extension

The exact A100 replay is an engineering reproduction, not a seed replicate.
Scientific stability requires fixed-final-epoch runs for seeds `17`, `42`,
`73`, `101`, and `137`, with identical samples, epoch count, batch order
contract, initialization procedure, optimizer, and compute precision for every
mode. Seed-level predictions are averaged per complex before the component
bootstrap. Seed variability is reported separately.

The validation set cannot select epochs or modes. The primary mode was frozen as
`distance_direction` from the original ablation; `full_pose` remains a negative
complexity control.

## Confirmatory evaluation sequence

1. Freeze this protocol and implementation hashes.
2. Replay inference from immutable final checkpoints on AB03/A100.
3. Verify pooled historical metrics within the preregistered numerical
   tolerance.
4. Create per-complex predictions and statistics receipts atomically.
5. Run an independent read-only verifier from the canonical artifacts.
6. Complete the five-seed fixed-budget extension.
7. Freeze the selected architecture without reading test labels.
8. Run exactly one leakage-controlled outer or external evaluation.
9. Run the separate partner-specific pair-matrix protocol.

## Interpretation rules

- A positive development result is reported as a development result.
- A confidence interval crossing zero is not evidence of superiority.
- `full_pose <= distance_direction` is retained as evidence against unnecessary
  geometric complexity.
- Results from different datasets, splits, contact definitions, or aggregation
  units are never compared as if they were an identical benchmark.
- SOTA or world-leading language is prohibited without identical-benchmark,
  independently reproducible evidence.

