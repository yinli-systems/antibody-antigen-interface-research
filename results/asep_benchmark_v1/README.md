# AsEP benchmark, development stage

Status: development-only. Trained on AsEP's own train split and evaluated on its
own validation split. **The benchmark test split was not read**, so nothing here
is a comparison against published numbers.

Date: 2026-09-10

Machine-readable: [`development.json`](development.json).

## Why this exists

No number in this project could be placed beside published work. AsEP is the
benchmark WALLE and EpiFormer are quoted on, and its `epitope_group` split is
the harder of the two it publishes.

Three things had to be true before a number could be produced at all, and each
turned out to need work:

1. **The split had to be AsEP's, exactly.** The official file stores integer
   indices into a canonical identifier list, not identifiers. Reproduced to
   train 1383 / validation 169 / test 169.
2. **The cohorts had to be checked for overlap.** 23.2% of AsEP's test
   structures are in sabdab2's training split, so a model trained on sabdab2
   cannot be reported on AsEP. Training on AsEP's own split, which is what this
   does, is unaffected. See `results/asep_cohort_overlap_v1/`.
3. **The structures had to be complete.** The stored backbone file covered 1,101
   of 1,721 instances, unevenly across the split (65.5% / 59.8% / 55.0%). Now
   1,721 of 1,721, at 100% on every split.

## Result: ESM-2 arm, ten seeds, validation

| macro | mean | sd | 95% CI |
|---|---:|---:|---|
| **auprc** | **0.051293** | 0.010341 | [0.044842, 0.056805] |
| **mcc** | **0.081802** | 0.005341 | [0.078231, 0.084348] |
| auroc | 0.852065 | 0.018741 | [0.839972, 0.861384] |
| f1 | 0.029544 | 0.000942 | [0.028951, 0.030049] |
| precision at true contact count | 0.067455 | 0.014981 | [0.058222, 0.075621] |
| prevalence | 0.004632 | 0 | — |

169 validation complexes. AUPRC is 11x the prevalence floor.

## What this is not

**WALLE reports MCC 0.077 on the `epitope_group` split.** This record shows
0.0818 on validation. That is not a comparison, for three reasons, and all three
have to be cleared before one can be claimed:

1. **Different split.** WALLE's figure is on test; this is on validation.
2. **The threshold is not known to match.** MCC here is computed at a fixed 0.5
   threshold. On the sabdab2 cohort a fixed 0.5 threshold understated F1 by
   3.4x, so a fixed-threshold MCC is a measurement convention as much as a model
   property, and WALLE's convention has not been checked against this one.
3. **Different task formulation.** This model predicts pairs with node marginals
   alongside; WALLE is a graph network over epitope nodes.

The honest summary is that the magnitudes are comparable and a comparison
requires a single measurement on the test split, made once, with the
configuration already fixed.

## What comes next

The configuration that won on sabdab2 — ESM-2 plus ESM-IF1, which beat ESM-2
alone by 30.5% over twenty paired seeds — is being extracted for AsEP now that
its structures are complete. When that arm has been measured on validation, one
configuration gets one measurement on the test split.

## Provenance

| | |
|---|---|
| split | AsEP `epitope_group`, from `split_dict_corrected.pt` |
| split manifest sha256 | `678a20cdfaa5567dcca20007795d03364934a680f1e54065afc312ea4a1f0226` |
| launcher | `slurm/asep_benchmark.sbatch` |
| converter | `scripts/convert_asep_official_split.py` |
| backbones | `scripts/build_asep_backbones.py`, 1721/1721 |
