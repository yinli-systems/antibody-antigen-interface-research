# What this line of work has established, and what it has not

Compiled 2026-09-11. One page per question is elsewhere; this is the ledger.

Everything below is development-cohort work unless a row says otherwise. Both
AsEP test splits are now spent: `epitope_group` was read twice, is permanently
frozen, and was deliberately not re-evaluated after a later hypothesis was
found; `epitope_ratio` has had both of its two registered readings.

## Established

| finding | evidence | strength |
|---|---|---|
| The pair head beats a predicted-marginal outer product | +30.5% macro pair AUPRC, 20 seeds, 20/20 positive, sign bound 0.861 | replicated; the original single-seed +32.7% was about 2× the truth |
| ESM-IF1's contribution is in the residue correspondence, not the vectors | aligned +22.8% over ESM-2 (10/10); width-matched random −2.6% (undecided); shuffled −13.8% (decided worse); aligned vs shuffled +42.4% (10/10) | the full chain, one cohort, one encoder |
| Displacing a frozen per-residue representation is harmful | reproduced on a second cohort with a second representation — a **sequence** model, AntiBERTy, not a second structural encoder. Quote the **pair AUPRC** endpoint: +30.2%, 6/6 seeds, sign bound 0.607 | two representations, two cohorts, confounded by coverage and provenance |
| Correspondence damage saturates by ~8 **representation rows** | k=8, k=16 and global permutation within a hair of each other on both ladders | 10 seeds ESM-IF1, 6 seeds AntiBERTy; **the displacement is in CDR-selected row index, not verified as peptide distance** |
| A one-residue register error already costs a lot | 52% of total damage (ESM-IF1), 44% (AntiBERTy) | ESM-IF1 lands between the registered cliff and slope boundaries; reported as intermediate |
| Down-weighting the relational term improves **both** tasks | interior optimum near λ=0.25; node +13.3% and pair +10.8% together | refuted the competition framing this project started from |
| The relational branch does not feed the node heads | node-only 0.1590 vs pair-supervised 0.1572 | makes the external comparison fair |
| sabdab2 and AsEP overlap by PDB entry | measured before any cross-cohort claim | constrains what either cohort may say about the other |

## Refuted, withdrawn, or failed

These are listed because they cost real compute and because the paper is
stronger for saying them out loud.

| claim | what happened |
|---|---|
| "Pair supervision and node prediction compete" | the dose curve found an interior optimum. The project's own registered readability condition fired against its framing |
| "The alignment benefit generalises across structural encoders" | **the second encoder was never structural.** It is AntiBERTy from AsEP's cache, verified by tensor identity. The test was not run, and the claim is withdrawn along with the framing that a second structural encoder had been tried |
| "We are above SOTA on AsEP `epitope_group`" | micro figures had been compared to AsEP's per-complex table. Recomputed per complex: MCC 0.1893→0.1572, AUROC 0.7515→0.6970. Withdrawn |
| "Precision and recall both higher than published" | held only against one older WALLE row, not against PEPNet+LE or the later WALLE tabulation. Withdrawn |
| "PCGrad is a negative control here" | it projects on ~1.6 of 2 ordered pairs per step. The premise rested on a cosine measured with a term missing. Withdrawn |
| "`partner_rank` selects badly because its trunk gradient is small" | does not follow: selection quality depends on ordering, not gradient norm. Withdrawn; replaced with a measurement that has not read out |
| "The optimisation pathology is the ranking term" | refuted by the three-way geometry. The ranking gradient is 23-40× smaller than the node gradient and near-orthogonal |
| "Checkpoint selection on total loss is worth +0.0278" | **the comparison was never run.** Both arms used `--selection-metric total`; they differed in `d_model`, 256 against 128. The number is a width effect. Withdrawn |
| "The checkpoint-selection effect is split-specific" | stage three changed **both** the selection metric and the width relative to the frozen configuration, so it confirmed a bundle. What is recorded: a development-set configuration did not confirm on the ratio test. No mechanism is attributed |

## Open, with measurements in flight

| question | what would settle it |
|---|---|
| ~~Why does cos(node, edge) differ between a converged checkpoint and early training?~~ | **Answered.** Both were AsEP, not two cohorts — the earlier attribution to sabdab2 was an error. The cosine is negative for ~100 steps and positive after; +0.70 is the endpoint of a trajectory that starts with the opposite sign |
| Is contact supervision or ranking the relational component that matters? | four arms × 10 seeds: node only, node+edge, node+rank, node+edge+rank |
| Is `partner_rank` a poorly aligned selection signal? | Spearman against the per-epoch node metric, and paired selection regret against `total`. Both are required before the claim may be written |
| Do gradient-balancing methods recover what hand-tuning λ recovers? | on hold. Whether the comparison is even scientific depends on the first question |

## Method defects that change what may be quoted

| defect | consequence |
|---|---|
| `igfold_generality_v1` marks a comparison "decided" from a bootstrap interval alone, at 3/6 seeds and a sign bound of 0.153 | that row may not be called consistent across seeds. The project's endpoint, pair AUPRC, is 6/6 on the same comparison and is the figure to quote |
| Stage two's arms differ in width, not in selection metric | the +0.0278 is a width effect; the selection-metric question is untested at ten seeds |
| Stage three changes width **and** selection metric at once | it confirms a configuration bundle and can attribute nothing to either knob |

## What may not be written yet

- **"Gradient conflict is ruled out."** A converged cosine cannot speak for the
  trajectory that produced it, and the in-situ measurement disagrees in sign.
- **"Loss-scale imbalance is the mechanism."** Its three supporting
  measurements still agree with each other, but the load-bearing one is now
  known to be stage-specific.
- **"Residue-level alignment is novel."** There is precedent (RLA, and
  DiscoTope-3.0 for inverse-folding latents in epitope prediction). The
  contribution is the controlled attribution and the severity curve, not the
  observation that correspondence or structure matters.
- **"Two structural encoders."** One. The other is a sequence model.
- **"k = 8 residues" as a biological length scale.** The displacement moves rows
  of a CDR-selected array. Whether eight rows is eight residues along the chain
  is unverified — the selection can jump segments and chain boundaries. The
  numerical statement stands; the biological reading does not.
- **"~8 residues is a universal critical length scale."** Two encoders, one
  domain. The safe form: degradation saturates rapidly with displacement,
  reaching near-global-shuffle levels by an eight-residue offset in both
  encoders.

The optimisation claim is frozen at: *relational supervision is necessary for
pair prediction, but its effect on node learning depends on training dynamics
and weighting.*

## Benchmark position, stated plainly

| split | this work | best published | standing |
|---|---|---|---|
| AsEP `epitope_group` | MCC 0.157 | 0.171 | same tier |
| AsEP `epitope_ratio` | MCC 0.258 | 0.410 | clearly behind |

Both are per-complex figures with the threshold selected on validation, which is
the convention the published tables use. Neither is a headline this work can
lead with, and the case for the work is the controls and the mechanism, not the
leaderboard row.
