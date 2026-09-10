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

**WALLE's MCC is an epitope-node MCC.** This model emits two sets of
predictions, pairs and nodes, and both carry an MCC. Only the node one is the
same quantity — they are computed over different objects at different
prevalences, 11.4% against 0.46%, and quoting the pair figure beside WALLE's
compares nothing.

### Epitope nodes — the comparable quantity

| macro | mean | sd | 95% CI |
|---|---:|---:|---|
| **mcc** | **0.166274** | 0.024634 | [0.150798, 0.179505] |
| auprc | 0.340828 | 0.044894 | [0.312599, 0.364310] |
| auroc | 0.741848 | 0.032824 | [0.721061, 0.758998] |
| f1 | 0.264781 | 0.014003 | [0.256086, 0.272575] |
| precision | 0.162268 | 0.010624 | [0.155840, 0.168338] |
| recall | 0.880573 | 0.033272 | [0.860749, 0.899267] |
| prevalence | 0.114060 | 0 | — |

AUPRC is 3x the prevalence floor.

### Pairs — this project's own endpoint, not comparable to WALLE

| macro | mean | sd | 95% CI |
|---|---:|---:|---|
| auprc | 0.051293 | 0.010341 | [0.044842, 0.056805] |
| mcc | 0.081802 | 0.005341 | [0.078231, 0.084348] |
| auroc | 0.852065 | 0.018741 | [0.839972, 0.861384] |
| prevalence | 0.004632 | 0 | — |

169 validation complexes.

## What this is not

WALLE reports **MCC 0.077** on `epitope_group`. This record shows **0.166** on
epitope nodes, roughly 2.2x. That is still not a comparison, and one caveat
matters more than it did before the quantities were lined up:

1. **Different split.** WALLE's figure is on test; this is on validation.
2. **The operating point is now bounded on this side, and it does not account
   for the gap.** A sweep over the stored predictions
   ([`threshold_epitope.json`](threshold_epitope.json), ten seeds, micro over
   all residues) puts the epitope MCC at three operating points that can be
   named without knowing anyone else's convention:

   | operating point | threshold | mcc | precision | recall |
   |---|---:|---:|---:|---:|
   | fixed 0.5 | 0.500 | 0.206 | 0.133 | 0.880 |
   | best attainable | 0.853 | **0.274** | 0.261 | 0.491 |
   | prevalence matched | 0.907 | 0.261 | 0.322 | 0.319 |

   The whole span is **1.33x**. The distance to the published 0.077 is about
   2.2x, so the operating point cannot explain it — for this model.

   **The bound is one-sided and that matters.** WALLE's figure may be just as
   threshold-sensitive, and nothing here measures it. If it moved by the same
   1.33x it would reach roughly 0.10, still below, but that is an assumption and
   not a measurement.

   The sweep also shows the model is badly calibrated: its optimum sits at 0.853
   and its prevalence-matched point at 0.907, so **0.5 is far from where this
   model works**. The same shift appeared on the sabdab2 cohort, where F1 peaked
   at 0.85. Every fixed-0.5 figure in this project understates it.

   Note that 0.206 and the 0.166 above are different quantities: the sweep pools
   residues across complexes, the table averages per complex.
3. **Different task formulation.** This model predicts pairs and derives node
   marginals; WALLE is a graph network over epitope nodes directly.

Until (1) and (2) are cleared, the statement this record supports is that the
epitope-node MCC is in the same regime as the published one and may be higher,
not that it is higher.

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
