# Abstract, draft 1

> **Status.** Every number below is bound to an artifact in
> `results/CLAIMS_LEDGER.md`. All eight cold-reproduction steps verify with zero
> numeric mismatches. **Revised after the held-out read** — the correspondence
> claim is now scoped to what confirmed on a split that took no part in
> development, and the paragraph that carried it has been rewritten downward.

---

## Title candidates

1. **When Frozen Representations Help or Harm: Residue Correspondence and
   Auxiliary Supervision in Protein Interaction Prediction**
2. Residue Correspondence and Auxiliary Supervision in Antibody–Antigen
   Interaction Prediction

The first is broader and names the finding; the second is the conservative
fallback. Neither claims a theorem.

---

## Abstract

Frozen pretrained representations are routinely appended to task models as
feature banks, on the assumption that adding them is at worst neutral. We ask
what makes such a representation transfer, using antibody–antigen interface
prediction as a controlled setting, and report two results that qualify the
assumption.

First, the value of a frozen per-residue representation depends on its
correspondence to the residues it describes, and disrupting that correspondence
does not merely remove the benefit but reverses it. Displacing the rows of a
frozen inverse-folding representation against the residues they describe makes
the model **worse than using no structural features at all** (−13.8% macro pair
AUPRC against a sequence-only baseline, 10 paired seeds), while the aligned
representation improves on that baseline by +22.8%. The effect is not explained
by input width or parameter count: a width-matched random projection of the
model's own input is indistinguishable from the baseline. A displacement ladder
shows the damage is not confined to total scrambling — a displacement of eight
rows already costs 94% of what a global permutation costs, and on this cohort
those eight rows are eight contiguous residues for 90% of positions, verified
geometrically rather than assumed.

**On a held-out split that took no part in development, one of two encoders
confirms this.** A single preregistered read of 217 unseen complexes, using the
same checkpoints without retraining, finds the aligned-minus-shuffled contrast
positive for both encoders: decided for ProteinMPNN (+0.0036, 9 of 10 paired
seeds) and undecided for ESM-IF1 (+0.0023, 6 of 10, interval [−7e−6, +0.0049]),
whose development advantage shrank roughly threefold. The encoder with the larger
development effect is the one that did not confirm.

Second, correspondence is necessary but not sufficient, and the asymmetry
replicates where the benefit does not. Across a second inverse-folding encoder
of unrelated architecture and a sequence-only antibody language model, displacing
rows is consistently harmful — −12.6% and −12.6% against the same baseline — while
a large aligned benefit appears for only one of the three. Preserving
correspondence does not guarantee positive transfer.

Third, two relational auxiliary objectives that are usually treated as one are
not interchangeable. In a four-arm factorial ablation, contact supervision raises
pairwise AUPRC by an order of magnitude with no detected cost to node prediction,
whereas deleting a complex-level ranking objective improves **both** endpoints on
all ten seeds. The advantage appears before either arm's learning-rate schedule
diverges, so it is not an artefact of checkpoint selection; we report it as an
intervention on the training procedure rather than as an isolated gradient
mechanism, because first-order gradient geometry on the shared trunk does not
account for its size.

Finally, we find that gradient diagnostics taken at convergence can invert the
picture they are used to explain. The cosine between node and contact gradients
is strongly positive at convergence (+0.70) and **negative for roughly the first
hundred optimisation steps**, crossing at step 77–110 of the first epoch on one
cohort and 38–50 on another, reproduced step-for-step on an unrelated toolchain.

We make no state-of-the-art claim: on the benchmark this model is mid-table on
the harder split and clearly behind on the easier one. The contribution is the
controlled attribution and its boundaries, including a pre-registered
confirmation that failed and is reported in full.

---

## Word count

~360 words. ICLR abstracts have no hard limit but are used for reviewer bidding,
so this stays close to what a bidder will actually read.

## Numbers used, and where each is bound

| claim in the abstract | ledger row | artifact |
|---|---|---|
| shuffled −13.8% vs ESM-2 | 5 | `representation_attribution_v1/report.json` |
| aligned +22.8% vs ESM-2 | 2 | same |
| random control undecided | 4 | same |
| eight rows = 94% of global | 7 | `curve_esm_if1.json` |
| eight rows ≈ eight residues, 90% | — | `audit_row_to_residue_mapping.py` |
| AntiBERTy −12.6%, ProteinMPNN −12.6% | 8, — | `curve_antiberty_sabdab2.json`, `second_structural_encoder_v1` |
| contact +0.0833 pair, node undecided | 15, 16 | `objective_component_localisation_v1/PART_TWO.md` |
| ranking deletion +0.0214 / +0.0369 | 14 | same |
| advantage predates LR divergence | — | `RUN_CONTRACT.md` |
| cosine +0.70 at convergence | 9 | `gradient_conflict_v1` |
| crossings 77–110 / 38–50 | 9 | `step_trace_summary.json` |
| benchmark position | 10, 11 | `asep_benchmark_v1` |
| held-out: ProteinMPNN +0.0036, 9/10 | — | `holdout_confirmation_v1/confirmation.json` |
| held-out: ESM-IF1 +0.0023, undecided | — | same |

## Deliberately absent

- Any phrase of the form "alignment is necessary and sufficient".
- "Across model families" — two inverse-folding encoders are two encoders.
- The checkpoint-selection line, which did not confirm and belongs in the
  appendix.
- "Contact supervision is free" — the node comparison is undecided, and
  undecided is not equivalence.
- Any causal reading of the group-versus-ratio split contrast.
- "Confirmed on held-out data" without saying one of two. ESM-IF1's interval
  misses zero by 7e−6 and that is reported as undecided, not as nearly decided.
- The 185-complex subset used in place of the registered 217. It is a sensitivity
  check on a 14.7% exposure to an unrelated baseline's test set, and it agrees in
  direction; it does not replace the registered answer.
