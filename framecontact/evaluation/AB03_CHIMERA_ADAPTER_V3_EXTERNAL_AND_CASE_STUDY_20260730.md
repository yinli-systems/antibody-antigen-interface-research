# AB03 CHIMERA adapter-v3 external result and case study

## Claim boundary

This is real A100 inference from six frozen checkpoints (two candidates times
seeds 17/42/73), followed by manifest-bound scoring and 5,000-replicate
antigen-sequence group bootstrap case analysis. The model predicts per-residue
epitope and paratope labels; it does not predict residue-residue contacts.

Adapter-v3 was developed after inspection of the first external adapter's
coverage failures. It raises usable coverage to 283/292 (96.92%) on
`antigen_fold.test` and 288/293 (98.29%) on `temporal.test`, but is post-access
secondary evidence. It cannot be promoted to a confirmatory claim, regardless
of significance. It is also not the canonical CHIMERA CDR sequence-structure
co-design task.

## Frozen external result

| Split | Candidate | Epitope mean complex AUPRC | Epitope pooled AUPRC | Paratope mean complex AUPRC | Paratope pooled AUPRC |
|---|---|---:|---:|---:|---:|
| antigen-fold | sequence dense | 0.587241 | 0.422923 | 0.652136 | 0.601424 |
| antigen-fold | spatial topology | 0.581367 | 0.239074 | 0.671796 | 0.624964 |
| temporal | sequence dense | 0.294910 | 0.259393 | 0.699297 | 0.664476 |
| temporal | spatial topology | 0.308357 | 0.241579 | 0.723849 | 0.697914 |

The frozen primary comparison is spatial topology minus sequence dense in mean
per-complex epitope AUPRC:

- antigen-fold: -0.005874, 95% CI [-0.017912, 0.005603], Holm p=0.310369;
- temporal: +0.013446, 95% CI [0.003404, 0.023935], Holm p=0.017598.

The temporal result is statistically positive within this secondary analysis,
but `confirmatory_support=false` by construction. The pooled epitope AUPRC is
lower for spatial topology on both splits, while paratope improves on both.
This combination points to cross-complex score calibration/global ranking as a
specific unresolved failure rather than a uniform lack of within-complex
benefit.

## High-coverage case-study sensitivity

| Split/cohort | n | Dense mean complex AUPRC | Spatial mean complex AUPRC | Delta | 95% group-bootstrap CI |
|---|---:|---:|---:|---:|---:|
| antigen-fold, all | 283 | 0.5872 | 0.5814 | -0.0059 | [-0.0176, 0.0058] |
| antigen-fold, antigen length >= 50 | 145 | 0.2833 | 0.2967 | +0.0134 | [-0.0008, 0.0271] |
| antigen-fold, epitope prevalence <= 0.5 | 151 | 0.2937 | 0.3108 | +0.0171 | [0.0013, 0.0332] |
| temporal, all | 288 | 0.2949 | 0.3084 | +0.0134 | [0.0036, 0.0238] |
| temporal, antigen length >= 50 | 272 | 0.2640 | 0.2794 | +0.0154 | [0.0046, 0.0262] |
| temporal, epitope prevalence <= 0.5 | 273 | 0.2649 | 0.2802 | +0.0153 | [0.0053, 0.0260] |

These sensitivity cutoffs only expose short-peptide and near-all-positive
average-precision degeneracy. They are post-hoc descriptive checks and were not
used for model selection.

## Does spatial topology fail more on long or discontinuous epitopes?

Not as a general trend in this run. On antigen-fold, the AUPRC delta correlates
positively with antigen length (Spearman rho=0.228, 95% group-bootstrap CI
[0.115, 0.334]), epitope count (0.152, [0.033, 0.261]), 8-A epitope component
count (0.140, [0.010, 0.263]), radius of gyration (0.182, [0.069, 0.295]), and
sequence-run count (0.177, [0.067, 0.285]). It correlates negatively with
epitope prevalence (-0.220, [-0.328, -0.110]). Temporal has the same positive
direction for length (0.146, [0.029, 0.260]) and sequence-run count (0.158,
[0.040, 0.271]); most other dispersion intervals cross zero.

This does not mean long antigens are easy. Their absolute AUPRC remains much
lower. It means spatial topology usually loses less, or helps more relative to
dense attention, as the problem becomes longer and sparser. The teacher's
hypothesis is still supported at the level of individual failures, not the
population trend.

Representative failures are:

| Split | Complex | Length | Epitope n | 8-A components | Rg (A) | Dense AP | Spatial AP | Delta |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| antigen-fold | `8dyy_A_B_I` | 72 | 8 | 1 | 6.11 | 0.506 | 0.069 | -0.437 |
| antigen-fold | `8fym_E_D_F` | 19 | 10 | 2 | 6.86 | 0.881 | 0.529 | -0.352 |
| antigen-fold | `6plh_B_A_C` | 8 | 4 | 2 | 5.90 | 0.950 | 0.604 | -0.346 |
| temporal | `9mib_O_Q_D` | 125 | 7 | 1 | 4.93 | 0.350 | 0.049 | -0.300 |
| temporal | `9hco_H_L_A` | 541 | 20 | 1 | 9.10 | 0.372 | 0.132 | -0.240 |
| temporal | `9og2_B_C_A` | 347 | 16 | 3 | 11.79 | 0.685 | 0.453 | -0.233 |

The next structural inspection set should include `8dyy_A_B_I` as the largest
antigen-fold regression, `9hco_H_L_A` as a long-antigen regression, and
`9og2_B_C_A` as a long, three-component epitope regression. Attention maps
should be compared with spatial-neighbor enrichment, label components, chain
boundaries, and frame-valid masks.

## What antigen-fold currently proves

The official CHIMERA description groups the antigen-fold split by antigen
identity so test targets are unseen. Therefore this result is held-out antigen
identity evidence. It is not sufficient to claim that a learned local
structural vocabulary transfers to structurally novel folds under a specified
TM-score/Foldseek threshold.

A stronger mechanistic test should fit an invariant local-patch reference bank
on training structures only, measure each test patch's nearest-training novelty,
and report performance as a function of novelty. Geometry-shuffle,
frame-corruption, and neighbor-graph rewiring controls should leave sequences
and labels unchanged. None of these interventions may be selected on the
external test result.

## Engineering implications

1. Preserve dense global heads alongside spatial heads instead of replacing all
   long-range edges. Use a learned validation-only gate per head/residue.
2. Add local-frame relative pose as an attention bias, then compare distance,
   direction, pose, and topology with the same parameter budget.
3. Add a validation-fitted calibration layer and report pooled AUPRC/ECE/Brier;
   the current pooled epitope regression is too large to ignore.
4. Use multi-scale neighborhoods (for example 8/12/16 A) and component tokens so
   spatially disconnected epitopes can communicate without a fully dense graph.
5. Run case-level geometry interventions and training-patch novelty analysis
   before proposing SE(3)-RoPE as the cause of any gain.

## AB03 paths and SHA-256

- Frozen result:
  `/home/liyin/geometry-self-attention-frozen-eval-20260730/outputs/chimera-secondary-result-adapter-v3/CHIMERA_STRICT_EXTERNAL_RESULT.json`
  (`fc6e2e0a9a308cf696a91d92716c3aea6b24cdb96228a8d78f5dd1bdec9b71f9`)
- Prediction manifest:
  `/home/liyin/geometry-self-attention-frozen-eval-20260730/outputs/chimera-secondary-predictions-adapter-v3/PREDICTION_MANIFEST.json`
  (`f39330fdc0b840d5f62444b02fb26222f968210931e4da0bd7de8f912b4eb6e2`)
- ESM embedding manifest:
  `/home/liyin/geometry-self-attention-frozen-eval-20260730/outputs/chimera-esm2-embeddings-adapter-v3/CHIMERA_ESM_EMBEDDING_MANIFEST.json`
  (`2abfa37c0be3959b4dd2431f16fecd79457ff7a37a4063a8907381acaa57baeb`)
- Antigen-fold case report:
  `/home/liyin/geometry-self-attention-frozen-eval-20260730/outputs/chimera-case-study-antigen_fold-adapter-v3/CASE_STUDY.md`
  (`803bd5a52cc09ab4cb89c0787d63d979c7913c51117aba3e324d886fa4e2d1dc`)
- Temporal case report:
  `/home/liyin/geometry-self-attention-frozen-eval-20260730/outputs/chimera-case-study-temporal-adapter-v3/CASE_STUDY.md`
  (`ea1024d4ad1cc314528c6c1710335c6bd3db9b95e764c3987d467937baeac70a`)
