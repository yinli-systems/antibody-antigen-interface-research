# AB03 CHIMERA frozen epitope case study

## Evidence boundary

This is a post-hoc descriptive analysis of real, frozen, three-seed external
predictions from the existing AB03 CHIMERA run. It compares
`spatial_topology` with `sequence_dense` for residue-level epitope prediction.
It is not residue-residue contact prediction, is not the canonical CHIMERA CDR
co-design task, and does not supersede the preregistered external result. The
original adapter covers 192/292 antigen-fold test complexes and 196/293 temporal
test complexes, so the analysis is informative but incomplete.

Prediction manifest SHA-256:
`d06606ab7210cda31891846d2cac621a9dd0ac1a6050c3832ad5587fad5661cb`.
Adapter-cache manifest SHA-256:
`ad45a5a901045784aa7fdf038bf5e3d0d1018a26eb4c107c9b48fdcf11bc62af`.

## Result

| Split/cohort | n | Dense mean complex AUPRC | Spatial mean complex AUPRC | Delta | 95% antigen-sequence group bootstrap CI |
|---|---:|---:|---:|---:|---:|
| antigen-fold, all available | 192 | 0.5724 | 0.5768 | +0.0045 | [-0.0085, 0.0178] |
| antigen-fold, antigen length >= 50 | 104 | 0.2915 | 0.3134 | +0.0219 | [0.0061, 0.0380] |
| antigen-fold, epitope prevalence <= 0.5 | 109 | 0.2994 | 0.3294 | +0.0300 | [0.0122, 0.0491] |
| temporal, all available | 196 | 0.3078 | 0.3247 | +0.0169 | [0.0039, 0.0309] |
| temporal, antigen length >= 50 | 184 | 0.2746 | 0.2942 | +0.0195 | [0.0054, 0.0339] |
| temporal, epitope prevalence <= 0.5 | 185 | 0.2759 | 0.2953 | +0.0194 | [0.0055, 0.0338] |

The sensitivity cutoffs are descriptive robustness checks, not tuned selection
rules. Short peptide-like antigens and nearly all-positive residue axes make
per-complex average precision close to degenerate and can conceal the behavior
on ordinary protein antigens.

## Difficulty associations

On the available antigen-fold subset, longer antigens are much harder in
absolute terms: the longest quartile has dense/spatial mean AUPRC
0.1205/0.1467. However, antigen length correlates positively, not negatively,
with the spatial-minus-dense delta (Spearman rho=0.2424, 95% group-bootstrap CI
[0.1007, 0.3678], nominal p=0.000704). Epitope radius of gyration has rho=0.1555
([0.0152, 0.2905], p=0.0313); 8-A component count has rho=0.1360 but its interval
crosses zero ([-0.0132, 0.2786], p=0.0600). Epitope count is weak and uncertain
(rho=0.1020, [-0.0444, 0.2463], p=0.1592).

On the temporal subset, epitope count (rho=0.1516, [0.0156, 0.2783], p=0.0340)
and sequence-run count (rho=0.1755, [0.0361, 0.3046], p=0.0139) correlate
positively with delta. Length, radius of gyration, 8-A component count, and
nearest-positive distance have intervals crossing zero.

These are unadjusted post-hoc associations. They do not establish a causal
mechanism and should be used to propose training-only stratification tests, not
to choose a model on the external set.

## Concrete antigen-fold cases

The largest spatial-topology failures include:

| Complex | Antigen length | Epitope count | 8-A components | Rg (A) | Dense AP | Spatial AP | Delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| `7vaz_D_C_G` | 10 | 8 | 1 | 6.16 | 0.9750 | 0.7261 | -0.2489 |
| `8gho_A_B_C` | 17 | 11 | 1 | 6.71 | 0.8881 | 0.6650 | -0.2231 |
| `6dc9_H_L_P` | 5 | 4 | 1 | 3.49 | 1.0000 | 0.8042 | -0.1958 |
| `8us8_H_L_R` | 17 | 10 | 1 | 8.44 | 0.7080 | 0.5149 | -0.1931 |
| `4qww_D_C_B` | 529 | 20 | 2 | 10.25 | 0.3329 | 0.1500 | -0.1830 |

The first four failures are short, high-prevalence antigen axes rather than
long, spatially fragmented proteins. `4qww_D_C_B` is the clearest case matching
the proposed long/discontinuous-failure hypothesis and should be inspected in
3D together with its attention map.

## Interpretation for antigen-fold generalization

The official CHIMERA description says the antigen-fold split groups by antigen
identity so the test targets are unseen. That is evidence of held-out antigen
identity performance, but it is not by itself proof that the network learned a
transferable local structural motif basis. A mechanistic test still needs:

1. a training-only local-patch reference bank built from invariant descriptors;
2. nearest-training-patch novelty for every test residue/epitope;
3. performance-versus-novelty curves and a matched sequence-only control;
4. an intervention such as geometry shuffling or frame corruption that leaves
   sequence and labels unchanged;
5. evaluation on the untouched external set without model selection.

## AB03 paths

- Antigen-fold report:
  `/home/liyin/geometry-self-attention-frozen-eval-20260730/outputs/chimera-case-study-antigen_fold-v2/CASE_STUDY.md`
- Antigen-fold per-complex table:
  `/home/liyin/geometry-self-attention-frozen-eval-20260730/outputs/chimera-case-study-antigen_fold-v2/PER_COMPLEX_CASES.csv`
- Temporal report:
  `/home/liyin/geometry-self-attention-frozen-eval-20260730/outputs/chimera-case-study-temporal-v2/CASE_STUDY.md`
- Temporal per-complex table:
  `/home/liyin/geometry-self-attention-frozen-eval-20260730/outputs/chimera-case-study-temporal-v2/PER_COMPLEX_CASES.csv`

The corresponding Markdown SHA-256 values are
`7f3569fb28c8eb49e4085fd9e42b3dea3a30994783c770ddd0fd7f26aee530d3`
and `bac2dae67f10a0b36c55b86e4e7a051f2bf54ff2b462de186e34aaba448cd829`.
